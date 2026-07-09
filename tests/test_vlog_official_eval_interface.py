from pathlib import Path

import numpy as np


def test_server_policy_accepts_vlog_arguments():
    text = Path("deployment/model_server/server_policy.py").read_text()
    assert "--framework_name" in text
    assert "--enable_vlog_logging" in text
    assert "set_vlog_logging" in text


def test_policy_wrapper_preserves_vlog_info():
    text = Path("deployment/model_server/policy_wrapper.py").read_text()
    assert '"vlog_info"' in text
    assert 'result["vlog_info"]' in text


def test_vlog_eval_scripts_exist():
    assert Path("examples/LIBERO/eval_files/run_vlog_policy_server.sh").exists()
    assert Path("examples/LIBERO/eval_files/eval_vlog_libero.sh").exists()
