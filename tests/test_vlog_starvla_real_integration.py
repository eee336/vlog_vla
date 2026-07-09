from pathlib import Path

from tools.inspect_vlog_starvla_paths import inspect_paths


def test_qwen_oft_vlog_registry_bridge_exists():
    report = inspect_paths()
    assert report["qwen_oft_vlog_class_path_exists"]
    assert report["qwen_oft_vlog_registry_bridge_exists"]
    assert report["qwen_oft_vlog_registered"]


def test_server_policy_supports_vlog_logging():
    text = Path("deployment/model_server/server_policy.py").read_text()
    assert "--enable_vlog_logging" in text
    assert "set_vlog_logging" in text


def test_vlog_train_and_eval_scripts_exist():
    required = [
        "examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml",
        "examples/LIBERO/train_files/run_vlog_libero_train.sh",
        "examples/LIBERO/eval_files/run_vlog_policy_server.sh",
        "examples/LIBERO/eval_files/eval_vlog_libero.sh",
    ]
    for path in required:
        assert Path(path).exists(), path
