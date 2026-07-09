from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base_model_class": "Qwenvl_OFT",
        "base_model_file": "starVLA/model/framework/VLM4A/QwenOFT.py",
        "backbone_forward_function": "Qwenvl_OFT.predict_action -> self.qwen_vl_interface(..., output_hidden_states=True)",
        "hidden_token_source": "qwenvl_outputs.hidden_states[-1], gathered at action special token positions",
        "hidden_token_candidate_names": [
            "qwenvl_outputs.hidden_states[-1]",
            "action_queries = _gather_action_token_embeddings(last_hidden, input_ids, action_token_id)",
        ],
        "action_head_function": "self.action_model.predict_action(action_queries)",
        "predict_action_function": "Qwenvl_OFT.predict_action",
        "checkpoint_load_function": "baseframework.from_pretrained -> Qwenvl_OFT",
        "server_policy_entry": "deployment/model_server/server_policy.py with PolicyServerWrapper",
        "vlog_framework_entry": "starVLA/model/vlog_vla/qwen_oft_vlog.py::QwenOFTVLOG",
        "can_extract_hidden_tokens": True,
        "uses_forward_hook": False,
        "notes": "QwenOFT already exposes hidden_states and action token gathering; direct extraction is more stable than a hook.",
    }
    (out_dir / "inspect_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / "hidden_token_probe_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
