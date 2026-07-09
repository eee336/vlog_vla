#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    report = inspect_paths()
    out = Path("outputs/vlog_real_train_eval/inspect_starvla_paths.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def inspect_paths() -> dict:
    paths = {
        "qwen_oft_class_path": "starVLA/model/framework/VLM4A/QwenOFT.py",
        "qwen_oft_vlog_class_path": "starVLA/model/vlog_vla/qwen_oft_vlog.py",
        "qwen_oft_vlog_registry_bridge": "starVLA/model/framework/VLM4A/QwenOFTVLOG.py",
        "framework_registry_path": "starVLA/model/tools.py",
        "build_framework_path": "starVLA/model/framework/base_framework.py",
        "train_entry": "starVLA/training/train_starvla.py",
        "libero_train_script": "examples/LIBERO/train_files/run_libero_train.sh",
        "vlog_train_script": "examples/LIBERO/train_files/run_vlog_libero_train.sh",
        "libero_eval_server_script": "examples/LIBERO/eval_files/run_policy_server.sh",
        "vlog_eval_server_script": "examples/LIBERO/eval_files/run_vlog_policy_server.sh",
        "libero_eval_client_script": "examples/LIBERO/eval_files/eval_libero.sh",
        "vlog_eval_client_script": "examples/LIBERO/eval_files/eval_vlog_libero.sh",
        "mixtures_path": "starVLA/dataloader/gr00t_lerobot/mixtures.py",
        "data_config_path": "starVLA/dataloader/gr00t_lerobot/data_config.py",
        "modality_template": "examples/LIBERO/train_files/modality.json",
        "lerobot_data_root": "playground/Datasets/LEROBOT_LIBERO_DATA",
    }
    exists = {key + "_exists": Path(value).exists() for key, value in paths.items()}
    report = {**paths, **exists}
    report["qwen_oft_vlog_registered"] = _contains(paths["qwen_oft_vlog_class_path"], '@FRAMEWORK_REGISTRY.register("QwenOFTVLOG")')
    report["server_supports_vlog_logging"] = _contains("deployment/model_server/server_policy.py", "enable_vlog_logging")
    return report


def _contains(path: str, needle: str) -> bool:
    p = Path(path)
    return p.exists() and needle in p.read_text(errors="ignore")


if __name__ == "__main__":
    main()
