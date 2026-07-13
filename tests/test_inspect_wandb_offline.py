import importlib.util
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "experiments"
    / "evidence"
    / "inspect_wandb_offline.py"
)
SPEC = importlib.util.spec_from_file_location("inspect_wandb_offline", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
INSPECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECT)


def test_inspect_root_counts_watch_histogram_records(tmp_path):
    path = tmp_path / "run-test.wandb"
    store = DataStore()
    store.open_for_write(str(path))

    scalar = wandb_internal_pb2.Record()
    scalar_item = scalar.history.item.add()
    scalar_item.nested_key.extend(["train/loss"])
    scalar_item.value_json = "1.0"
    store.write(scalar)

    histogram = wandb_internal_pb2.Record()
    type_item = histogram.history.item.add()
    type_item.nested_key.extend(["gradients/layer.weight", "_type"])
    type_item.value_json = '"histogram"'
    values_item = histogram.history.item.add()
    values_item.nested_key.extend(["gradients/layer.weight", "values"])
    values_item.value_json = "[1, 2]"
    bins_item = histogram.history.item.add()
    bins_item.nested_key.extend(["gradients/layer.weight", "bins"])
    bins_item.value_json = "[-1.0, 0.0, 1.0]"
    store.write(histogram)
    store.close()

    summary = INSPECT.inspect_root(tmp_path)
    assert summary["file_count"] == 1
    assert summary["watch_histogram_records"] == 1
    assert summary["watch_histograms"] == 1
    assert summary["watch_histogram_series"] == ["gradients/layer.weight"]
    assert summary["files"][0]["record_types"]["history"] == 2


def test_inspect_root_with_no_wandb_file_is_empty(tmp_path):
    assert INSPECT.inspect_root(tmp_path)["file_count"] == 0
