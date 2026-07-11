from __future__ import annotations

import hashlib
import fcntl
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
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
            except Exception as error:  # noqa: BLE001 - caller needs the final cause.
                last_error = error
                if attempt == self.max_attempts - 1:
                    break
                time.sleep(min(delay, self.max_delay_seconds))
                delay *= self.multiplier

        assert last_error is not None
        raise last_error


class BoundedShardCache:
    def __init__(
        self,
        cache_dir: str | Path,
        max_size_bytes: int,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if max_size_bytes < 1:
            raise ValueError("max_size_bytes must be positive")

        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_bytes)
        self.retry_policy = retry_policy or RetryPolicy()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.cache_dir / ".locks"
        self.lock_dir.mkdir(exist_ok=True)

        self._condition = threading.Condition()
        self._active_paths: dict[Path, int] = {}
        self._downloading_keys: set[str] = set()

    @contextmanager
    def acquire(
        self,
        key: str,
        downloader: Callable[[Path], None],
        *,
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> Iterator[Path]:
        path = self._acquire_path(
            key=key,
            downloader=downloader,
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        try:
            yield path
        finally:
            self.release(path)

    def fetch_url(self, url: str, timeout_seconds: float = 30.0) -> Path:
        def downloader(tmp_path: Path) -> None:
            download_url_to_path(url, tmp_path, timeout_seconds=timeout_seconds)

        path = self._acquire_path(key=url, downloader=downloader)
        self.release(path)
        return path

    def release(self, path: str | Path) -> None:
        resolved_path = Path(path)
        with self._condition:
            active_count = self._active_paths.get(resolved_path, 0)
            if active_count <= 1:
                self._active_paths.pop(resolved_path, None)
            else:
                self._active_paths[resolved_path] = active_count - 1
            self._touch_if_exists(resolved_path)
            self._condition.notify_all()

    @property
    def size_bytes(self) -> int:
        with self._condition:
            return self._current_size_locked()

    def _acquire_path(
        self,
        key: str,
        downloader: Callable[[Path], None],
        expected_sha256: str | None = None,
        expected_size_bytes: int | None = None,
    ) -> Path:
        identity_key = key if expected_sha256 is None else f"{key}\0sha256={expected_sha256}"
        path = self._path_for_key(identity_key)
        while True:
            with self._condition:
                if path.exists():
                    if _matches_identity(path, expected_sha256, expected_size_bytes):
                        self._mark_active_locked(path)
                        return path

                if identity_key not in self._downloading_keys:
                    self._downloading_keys.add(identity_key)
                    break

                self._condition.wait()

        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with self._key_file_lock(identity_key):
                if path.exists():
                    if _matches_identity(path, expected_sha256, expected_size_bytes):
                        with self._condition:
                            self._mark_active_locked(path)
                        return path
                    path.unlink()

                self.retry_policy.run(lambda: self._download_once(downloader, tmp_path))
                downloaded_size = tmp_path.stat().st_size
                if not _matches_identity(tmp_path, expected_sha256, expected_size_bytes):
                    raise ValueError("downloaded cache entry failed its immutable identity")
                if downloaded_size > self.max_size_bytes:
                    raise CacheSpaceError(
                        f"Shard is larger than cache capacity: "
                        f"{downloaded_size} > {self.max_size_bytes}"
                    )

                with self._condition:
                    while not self._make_space_locked(downloaded_size):
                        self._condition.wait(timeout=0.25)

                    os.replace(tmp_path, path)
                    self._mark_active_locked(path)
                    return path
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            with self._condition:
                self._downloading_keys.discard(identity_key)
                self._condition.notify_all()

    def _download_once(
        self,
        downloader: Callable[[Path], None],
        tmp_path: Path,
    ) -> None:
        if tmp_path.exists():
            tmp_path.unlink()
        downloader(tmp_path)
        if not tmp_path.exists():
            raise FileNotFoundError(f"Downloader did not create {tmp_path}")

    def _make_space_locked(self, incoming_size: int) -> bool:
        while self._current_size_locked() + incoming_size > self.max_size_bytes:
            evictable_path = self._oldest_evictable_path_locked()
            if evictable_path is None:
                return False
            evictable_path.unlink()
        return True

    def _oldest_evictable_path_locked(self) -> Path | None:
        candidates = [
            path
            for path in self.cache_dir.iterdir()
            if path.is_file() and not path.name.startswith(".") and path not in self._active_paths
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda path: path.stat().st_mtime_ns)

    def _current_size_locked(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.cache_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.shard"

    def _mark_active_locked(self, path: Path) -> None:
        self._active_paths[path] = self._active_paths.get(path, 0) + 1
        self._touch_if_exists(path)

    def _touch_if_exists(self, path: Path) -> None:
        if path.exists():
            path.touch()

    @contextmanager
    def _key_file_lock(self, key: str) -> Iterator[None]:
        lock_path = self.lock_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def download_url_to_path(
    url: str,
    path: str | Path,
    timeout_seconds: float = 30.0,
) -> None:
    destination = Path(path)
    with urlopen(url, timeout=timeout_seconds) as response:
        with destination.open("wb") as file:
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
