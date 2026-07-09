from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    out = Path("outputs/vlog_real_train_eval/inspect_starvla_paths.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "qwen_oft_class_path": "starVLA/model/framework/VLM4A/QwenOFT.py: Qwenvl_OFT",
        "qwen_oft_vlog_class_path": "starVLA/model/vlog_vla/qwen_oft_vlog.py: QwenOFTVLOG",
        "qwen_oft_vlog_registry_bridge": "starVLA/model/framework/VLM4A/QwenOFTVLOG.py",
        "framework_registry_path": "starVLA/model/tools.py: FRAMEWORK_REGISTRY",
        "train_entry": "starVLA/training/train_starvla.py",
        "libero_train_script": "examples/LIBERO/train_files/run_libero_train.sh",
        "libero_eval_server_script": "examples/LIBERO/eval_files/run_policy_server.sh",
        "libero_eval_client_script": "examples/LIBERO/eval_files/eval_libero.sh",
        "mixtures_path": "starVLA/dataloader/gr00t_lerobot/mixtures.py",
        "libero_data_registry_path": "examples/LIBERO/train_files/data_registry/data_config.py",
        "data_config_path": "starVLA/dataloader/gr00t_lerobot/data_config.py",
        "dataset_root": "playground/Datasets/LEROBOT_LIBERO_DATA",
        "checkpoint_loading": "train_starvla.py uses trainer.pretrained_checkpoint; TrainerUtils.load_pretrained_backbones(... strict=False) supports missing VLOG keys.",
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
