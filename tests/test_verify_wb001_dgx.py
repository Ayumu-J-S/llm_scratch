import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "evidence"
    / "verify_wb001_dgx.py"
)
SPEC = importlib.util.spec_from_file_location("verify_wb001_dgx", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_nearest_rank_percentile_does_not_interpolate():
    assert VERIFY.nearest_rank_percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.95) == 5.0
    assert VERIFY.nearest_rank_percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="at least one"):
        VERIFY.nearest_rank_percentile([], 0.95)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0B", 0),
        ("1.5kB", 1500),
        ("1.5KiB", 1536),
        ("2GiB", 2 * 1024**3),
    ],
)
def test_parse_size_bytes_distinguishes_si_and_iec(value, expected):
    assert VERIFY.parse_size_bytes(value) == expected


def test_parse_vmstat_discards_since_boot_row_and_retains_swap_fields():
    text = """procs -----------------------memory---------------------- ---swap-- -----io---- -system-- -------cpu------- -----timestamp-----
 r  b         swpd         free         buff        cache   si   so    bi    bo   in   cs  us  sy  id  wa  st  gu                 UTC
 1  0            0      1000000       100000      2000000    9    8     1     2    3    4   5   6  89   0   0   0 2026-07-13 00:00:00
 0  0            0       900000       100000      2000000    0    0     1     2    3    4   5   6  89   0   0   0 2026-07-13 00:00:01
 0  0            4       800000       100000      2000000    1    0     1     2    3    4   5   6  89   0   0   0 2026-07-13 00:00:02
"""
    rows = VERIFY.parse_vmstat(text)
    assert len(rows) == 2
    assert rows[0]["free"] == 900000
    assert rows[0]["si"] == 0
    assert rows[1]["swpd"] == 4
    assert rows[1]["si"] == 1


def test_normalized_config_removes_only_predeclared_arm_selectors():
    config = {
        "training": {"max_steps": 100},
        "wandb": {
            "mode": "offline",
            "name": "r1-p2-offline-off",
            "project": "llm-scratch",
            "watch": {"enabled": False, "log": "gradients", "log_freq": 100},
            "artifact": {"policy": "none"},
        },
    }
    normalized = VERIFY.normalized_config(config)
    assert normalized == {
        "training": {"max_steps": 100},
        "wandb": {
            "project": "llm-scratch",
            "watch": {"log": "gradients", "log_freq": 100},
            "artifact": {"policy": "none"},
        },
    }
    assert config["wandb"]["mode"] == "offline"


def test_summarize_steps_uses_aggregate_targets_and_post_warmup_wall_time():
    rows = []
    for step in range(1, 13):
        rows.append(
            {
                "event": "optimizer_step",
                "optimizer_step": step,
                "warmup": step <= 10,
                "target_tokens_step": 64,
                "step_wall_seconds": 2.0 if step == 11 else 1.0,
                "host_seconds": {"data_wait": 0.1, "forward": 0.2},
                "cuda_milliseconds": {"forward": 100.0},
                "pytorch_allocated_bytes": 1000 + step,
                "pytorch_reserved_bytes": 2000 + step,
            }
        )
    summary = VERIFY.summarize_steps(rows)
    assert summary["steps"] == 2
    assert summary["target_tokens"] == 128
    assert summary["wall_seconds"] == 3.0
    assert summary["target_tokens_per_second"] == pytest.approx(128 / 3)
    assert summary["step_median_seconds"] == 1.5
    assert summary["step_p95_seconds"] == 2.0
    assert summary["data_wait_fraction"] == pytest.approx(0.2 / 3)


def test_paired_regression_percent_has_positive_overhead_sign():
    assert VERIFY.paired_regression_percent(100.0, 95.0) == pytest.approx(5.0)
    assert VERIFY.paired_regression_percent(100.0, 105.0) == pytest.approx(-5.0)
    with pytest.raises(ValueError, match="positive"):
        VERIFY.paired_regression_percent(0.0, 1.0)


def test_canonical_digest_handles_scalar_and_bfloat16_tensors():
    first = {"step": torch.tensor(1.0), "weights": torch.ones(2, dtype=torch.bfloat16)}
    second = {"step": torch.tensor(1.0), "weights": torch.ones(2, dtype=torch.bfloat16)}
    changed = {"step": torch.tensor(2.0), "weights": torch.ones(2, dtype=torch.bfloat16)}
    assert VERIFY._canonical_digest(first) == VERIFY._canonical_digest(second)
    assert VERIFY._canonical_digest(first) != VERIFY._canonical_digest(changed)
