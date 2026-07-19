from __future__ import annotations

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from operations.artifacts import (
    AttemptError,
    atomic_write_json,
    create_attempt,
    process_identity,
)
from operations.runner import status_payload


def _attempt(tmp_path, attempt_id="attempt-0001", *, retry_from=None):
    return create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id=attempt_id,
        action="smoke",
        executor="host",
        device="cpu",
        retry_from=retry_from,
    )


def _fail(attempt) -> None:
    attempt.transition("preflighting")
    atomic_write_json(attempt.path / "result.json", {"outcome": "failed"})
    attempt.transition("failed")


def test_invalid_retry_does_not_reserve_an_empty_attempt(tmp_path):
    parent = _attempt(tmp_path)

    with pytest.raises(AttemptError, match="failed or stopped"):
        _attempt(tmp_path, "attempt-0002", retry_from=parent.attempt_id)

    assert not (parent.run_dir / "attempts" / "attempt-0002").exists()


def test_retry_binds_terminal_sibling_state_and_result_hashes(tmp_path):
    parent = _attempt(tmp_path)
    _fail(parent)
    child = _attempt(tmp_path, "attempt-0002", retry_from=parent.attempt_id)

    assert status_payload(child, root_dir=tmp_path)["integrity"]["retry_binding_valid"]

    atomic_write_json(parent.path / "result.json", {"outcome": "tampered"})
    assert not status_payload(child, root_dir=tmp_path)["integrity"]["retry_binding_valid"]


def test_succeeded_attempt_cannot_be_used_as_failure_retry(tmp_path):
    parent = _attempt(tmp_path)
    parent.transition("preflighting")
    parent.transition("ready")
    atomic_write_json(parent.path / "result.json", {"outcome": "succeeded"})
    parent.transition("succeeded")

    with pytest.raises(AttemptError, match="failed or stopped"):
        _attempt(tmp_path, "attempt-0002", retry_from=parent.attempt_id)


def test_atomic_events_are_unique_and_monotonic_under_concurrent_writers(tmp_path):
    attempt = _attempt(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda index: attempt.event("fixture", index=index), range(64)))

    assert len(set(paths)) == 64
    sequences = sorted(int(path.stem) for path in paths)
    # attempt_created is sequence 1.
    assert sequences == list(range(2, 66))


def test_concurrent_state_transition_has_one_authoritative_winner(tmp_path):
    attempt = _attempt(tmp_path)

    def transition():
        try:
            return attempt.transition("preflighting")["status"]
        except AttemptError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: transition(), range(2)))

    assert sorted(outcomes) == ["preflighting", "rejected"]
    assert attempt.state()["status"] == "preflighting"


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="Linux /proc required")
def test_proc_identity_parses_comm_with_spaces_and_parentheses():
    libc = ctypes.CDLL(None)
    old_name = open("/proc/self/comm", encoding="utf-8").read().strip().encode()
    new_name = b"ops a)b(c"
    try:
        assert libc.prctl(15, ctypes.c_char_p(new_name), 0, 0, 0) == 0
        identity = process_identity(os.getpid())
        assert identity["pid"] == os.getpid()
        assert identity["proc_start_ticks"] > 0
    finally:
        libc.prctl(15, ctypes.c_char_p(old_name[:15]), 0, 0, 0)


def test_status_reports_stale_pid_without_signaling(tmp_path):
    attempt = _attempt(tmp_path)
    attempt.transition("preflighting")
    attempt.transition("ready")
    attempt.transition("running")
    atomic_write_json(
        attempt.path / "pid.json",
        {
            "schema_version": 1,
            "pid": 2_000_000_000,
            "process_group_id": 2_000_000_000,
            "proc_start_ticks": 1,
            "boot_id": "not-this-boot",
            "cmdline_sha256": "0" * 64,
        },
    )

    assert status_payload(attempt, root_dir=tmp_path)["ownership"]["status"] == "stale"
