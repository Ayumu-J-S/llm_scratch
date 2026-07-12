from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator
from urllib.request import urlopen


class CacheSpaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    multiplier: float = 2.0

    def run(self, operation: Callable[[], None]) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        delay = self.initial_delay_seconds
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                operation()
                return
            except Exception as error:  # noqa: BLE001 - retain final download cause.
                last_error = error
                if attempt == self.max_attempts - 1:
                    break
                time.sleep(min(delay, self.max_delay_seconds))
                delay *= self.multiplier
        assert last_error is not None
        raise last_error


class BoundedShardCache:
    """Checksum-addressed shard cache with process-safe leases and disk reserve."""

    def __init__(
        self,
        cache_dir: str | Path,
        max_size_bytes: int,
        retry_policy: RetryPolicy | None = None,
        *,
        min_free_bytes: int = 0,
        wait_timeout_seconds: float = 30.0,
    ) -> None:
        if max_size_bytes < 1:
            raise ValueError("max_size_bytes must be positive")
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes must be non-negative")
        if wait_timeout_seconds <= 0:
            raise ValueError("wait_timeout_seconds must be positive")
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.wait_timeout_seconds = float(wait_timeout_seconds)
        self.retry_policy = retry_policy or RetryPolicy()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.cache_dir / ".locks"
        self.lease_dir = self.cache_dir / ".leases"
        self.reservation_dir = self.cache_dir / ".reservations"
        for directory in (self.lock_dir, self.lease_dir, self.reservation_dir):
            directory.mkdir(exist_ok=True)
        self._global_lock_path = self.lock_dir / "cache.lock"
        self._condition = threading.Condition()
        self._active_paths: dict[Path, int] = {}
        self._lease_files: dict[Path, list[BinaryIO]] = {}
        self._downloading_keys: set[str] = set()
        self._telemetry = {
            "hits": 0,
            "misses": 0,
            "downloads": 0,
            "downloaded_bytes": 0,
            "retries": 0,
            "evictions": 0,
            "corruptions": 0,
            "wait_timeouts": 0,
        }

    @contextmanager
    def acquire(
        self,
        key: str,
        downloader: Callable[[Path], None],
        *,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> Iterator[Path]:
        path = self._acquire_path(key, downloader, expected_sha256, expected_size_bytes)
        try:
            yield path
        finally:
            self.release(path)

    def fetch_url(self, url: str, timeout_seconds: float = 30.0) -> Path:
        def downloader(tmp_path: Path) -> None:
            download_url_to_path(url, tmp_path, timeout_seconds)

        path = self._acquire_path(url, downloader)
        self.release(path)
        return path

    def release(self, path: str | Path) -> None:
        resolved = Path(path)
        with self._condition:
            leases = self._lease_files.get(resolved, [])
            if leases:
                lease = leases.pop()
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
            if not leases:
                self._lease_files.pop(resolved, None)
            count = self._active_paths.get(resolved, 0)
            if count <= 1:
                self._active_paths.pop(resolved, None)
            else:
                self._active_paths[resolved] = count - 1
            self._touch_if_exists(resolved)
            self._condition.notify_all()

    @property
    def size_bytes(self) -> int:
        with self._condition:
            return self._current_size()

    @property
    def telemetry(self) -> dict[str, int]:
        with self._condition:
            return {
                **self._telemetry,
                "size_bytes": self._current_size(),
                "free_bytes": shutil.disk_usage(self.cache_dir).free,
                "active_leases": sum(self._active_paths.values()),
            }

    def _acquire_path(
        self,
        key: str,
        downloader: Callable[[Path], None],
        expected_sha256: str | None,
        expected_size_bytes: int | None,
    ) -> Path:
        if expected_size_bytes is not None and expected_size_bytes < 0:
            raise ValueError("expected_size_bytes must be non-negative")
        identity = key if expected_sha256 is None else f"{key}\0sha256={expected_sha256}"
        path = self._path_for_key(identity)
        deadline = time.monotonic() + self.wait_timeout_seconds
        with self._condition:
            while identity in self._downloading_keys:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timeout("timed out waiting for an in-process cache download")
                self._condition.wait(timeout=min(remaining, 0.1))
            self._downloading_keys.add(identity)

        reservation = self.reservation_dir / f"{uuid.uuid4().hex}.json"
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with self._timed_file_lock(self._key_lock_path(identity), deadline):
                existing = self._acquire_existing(
                    path, expected_sha256, expected_size_bytes, deadline
                )
                if existing:
                    with self._condition:
                        self._telemetry["hits"] += 1
                    return path
                with self._condition:
                    self._telemetry["misses"] += 1
                if expected_size_bytes is not None:
                    self._reserve(reservation, expected_size_bytes, deadline)
                attempts = 0

                def tracked_download() -> None:
                    nonlocal attempts
                    if attempts:
                        with self._condition:
                            self._telemetry["retries"] += 1
                    attempts += 1
                    self._download_once(downloader, tmp_path)

                self.retry_policy.run(tracked_download)
                downloaded_size = tmp_path.stat().st_size
                if not _matches_identity(tmp_path, expected_sha256, expected_size_bytes):
                    with self._condition:
                        self._telemetry["corruptions"] += 1
                    raise ValueError("downloaded cache entry failed its immutable identity")
                if downloaded_size > self.max_size_bytes:
                    raise CacheSpaceError(
                        f"shard exceeds cache capacity: {downloaded_size} > {self.max_size_bytes}"
                    )
                if expected_size_bytes is None:
                    self._reserve(reservation, downloaded_size, deadline, temp_already_present=True)
                with self._timed_file_lock(self._global_lock_path, deadline):
                    if shutil.disk_usage(self.cache_dir).free < self.min_free_bytes:
                        raise CacheSpaceError("download would violate cache free-space floor")
                    os.replace(tmp_path, path)
                    reservation.unlink(missing_ok=True)
                    self._mark_active(path)
                with self._condition:
                    self._telemetry["downloads"] += 1
                    self._telemetry["downloaded_bytes"] += downloaded_size
                return path
        finally:
            tmp_path.unlink(missing_ok=True)
            reservation.unlink(missing_ok=True)
            with self._condition:
                self._downloading_keys.discard(identity)
                self._condition.notify_all()

    def _acquire_existing(
        self,
        path: Path,
        expected_sha256: str | None,
        expected_size_bytes: int | None,
        deadline: float,
    ) -> bool:
        while True:
            with self._timed_file_lock(self._global_lock_path, deadline):
                if not path.exists():
                    return False
                if _matches_identity(path, expected_sha256, expected_size_bytes):
                    self._mark_active(path)
                    return True
                if not self._path_has_external_lease(path):
                    path.unlink(missing_ok=True)
                    with self._condition:
                        self._telemetry["corruptions"] += 1
                    return False
            if time.monotonic() >= deadline:
                self._timeout("timed out waiting to replace a corrupt leased cache entry")
            time.sleep(0.05)

    def _reserve(
        self,
        reservation: Path,
        size: int,
        deadline: float,
        *,
        temp_already_present: bool = False,
    ) -> None:
        while True:
            with self._timed_file_lock(self._global_lock_path, deadline):
                reservation.write_text(
                    json.dumps({"size_bytes": size, "pid": os.getpid()}),
                    encoding="utf-8",
                )
                if self._make_space(size, reservation):
                    free = shutil.disk_usage(self.cache_dir).free
                    required = self.min_free_bytes + (0 if temp_already_present else size)
                    if free >= required:
                        return
                reservation.unlink(missing_ok=True)
            if time.monotonic() >= deadline:
                self._timeout("timed out reserving bounded cache and disk headroom")
            time.sleep(0.05)

    def _make_space(self, incoming_size: int, own_reservation: Path) -> bool:
        reserved = 0
        for path in self.reservation_dir.glob("*.json"):
            if path == own_reservation:
                continue
            try:
                reservation = json.loads(path.read_text(encoding="utf-8"))
                pid = int(reservation["pid"])
                if not _process_exists(pid):
                    path.unlink(missing_ok=True)
                    continue
                reserved += int(reservation["size_bytes"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
        while self._current_size() + reserved + incoming_size > self.max_size_bytes:
            path = self._oldest_evictable_path()
            if path is None:
                return False
            path.unlink()
            with self._condition:
                self._telemetry["evictions"] += 1
        return True

    def _oldest_evictable_path(self) -> Path | None:
        candidates = []
        for path in self.cache_dir.glob("*.shard"):
            if path in self._active_paths or self._path_has_external_lease(path):
                continue
            candidates.append(path)
        return min(candidates, key=lambda item: item.stat().st_mtime_ns) if candidates else None

    def _path_has_external_lease(self, path: Path) -> bool:
        lease_path = self._lease_path(path)
        with lease_path.open("a+b") as lease:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        return False

    def _mark_active(self, path: Path) -> None:
        lease = self._lease_path(path).open("a+b")
        fcntl.flock(lease.fileno(), fcntl.LOCK_SH)
        with self._condition:
            self._lease_files.setdefault(path, []).append(lease)
            self._active_paths[path] = self._active_paths.get(path, 0) + 1
        self._touch_if_exists(path)

    def _download_once(self, downloader: Callable[[Path], None], tmp_path: Path) -> None:
        tmp_path.unlink(missing_ok=True)
        downloader(tmp_path)
        if not tmp_path.exists():
            raise FileNotFoundError(f"downloader did not create {tmp_path}")

    @contextmanager
    def _timed_file_lock(self, path: Path, deadline: float) -> Iterator[None]:
        with path.open("a+b") as lock:
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self._timeout(f"timed out waiting for cache lock {path.name}")
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _timeout(self, message: str) -> None:
        with self._condition:
            self._telemetry["wait_timeouts"] += 1
        raise CacheSpaceError(message)

    def _current_size(self) -> int:
        return sum(path.stat().st_size for path in self.cache_dir.glob("*.shard"))

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.shard"

    def _key_lock_path(self, key: str) -> Path:
        return self.lock_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.lock"

    def _lease_path(self, path: Path) -> Path:
        return self.lease_dir / f"{path.stem}.lease"

    @staticmethod
    def _touch_if_exists(path: Path) -> None:
        if path.exists():
            path.touch()


def download_url_to_path(url: str, path: str | Path, timeout_seconds: float = 30.0) -> None:
    with urlopen(url, timeout=timeout_seconds) as response:
        with Path(path).open("wb") as file:
            shutil.copyfileobj(response, file)


def _matches_identity(
    path: Path,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
) -> bool:
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        return False
    if expected_sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def _process_exists(pid: int) -> bool:
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
