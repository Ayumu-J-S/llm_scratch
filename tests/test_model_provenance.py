import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_model_provenance.py"


def run_capture(*args: str) -> dict:
    result = subprocess.run([sys.executable, str(SCRIPT), *args], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_actual_and_requested_are_separate_and_unavailable_is_explicit():
    payload = run_capture(
        "--requested-model", "gpt-5.6-sol",
        "--requested-reasoning-mode", "xhigh",
        "--actual-product", "Codex",
        "--actual-model-family", "GPT-5",
    )
    assert payload["actual"]["product"]["value"] == "Codex"
    assert payload["actual"]["displayed_model_family"]["value"] == "GPT-5"
    assert payload["actual"]["exact_model_identifier"]["value"] == "not exposed by runtime"
    assert payload["actual"]["exact_model_identifier"]["status"] == "unavailable"
    assert payload["actual"]["exact_model_identifier"]["unavailable_reason"]
    assert payload["actual"]["reasoning_mode"]["value"] == "not exposed by runtime"
    assert payload["requested"]["model"]["value"] == "gpt-5.6-sol"
    assert payload["requested"]["reasoning_mode"]["value"] == "xhigh"


def test_explicit_actual_values_are_not_replaced_by_config():
    payload = run_capture(
        "--requested-model", "gpt-5.6-sol",
        "--requested-reasoning-mode", "xhigh",
        "--actual-product", "Codex",
        "--actual-model-family", "GPT-5",
        "--actual-exact-model", "deployment-visible-id",
        "--actual-reasoning-mode", "Extra Thinking",
    )
    assert payload["actual"]["exact_model_identifier"]["value"] == "deployment-visible-id"
    assert payload["actual"]["reasoning_mode"]["value"] == "Extra Thinking"
    assert payload["actual"]["exact_model_identifier"]["source"] == "active runtime display"


def test_default_capture_is_privacy_safe_and_utc():
    payload = run_capture()
    assert payload["environment"]["thread_id"] == "not recorded (privacy)"
    assert payload["privacy"] == {
        "raw_thread_id_recorded": False,
        "prompts_recorded": False,
        "hidden_chain_of_thought_recorded": False,
        "token_counts_recorded": False,
        "secrets_recorded": False,
    }
    assert payload["captured_at"].endswith("Z")
    assert payload["schema_version"] == "1.0"


def test_output_file_is_valid_json(tmp_path):
    output = tmp_path / "provenance.json"
    subprocess.run([sys.executable, str(SCRIPT), "--output", str(output)], check=True)
    assert json.loads(output.read_text(encoding="utf-8"))["phase"] == "implementation"
