from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlog_vla.stage7_common import parse_config, write_failure
from starVLA.model.vlog_vla.starvla_hidden_adapter import StarVLAHiddenAdapter
from starVLA.model.framework.base_framework import baseframework


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_config(args.config)
    attempts = [
        "load baseframework.from_pretrained(base_checkpoint)",
        "build StarVLAHiddenAdapter(allow_surrogate_hidden=false)",
        "call encode_hidden on a real-shaped dummy observation",
        "decode action from action-token hidden via original action_model.predict_action",
    ]
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = baseframework.from_pretrained(args.base_checkpoint).to(device).eval()
        adapter = StarVLAHiddenAdapter(model, {"allow_surrogate_hidden": False})
        image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        obs = {
            "image": image,
            "lang": "pick up the black bowl next to the cookie box and place it on the plate",
            "state": np.zeros((1, 8), dtype=np.float32),
        }
        with torch.no_grad():
            hidden, aux = adapter.encode_hidden(obs)
            action = adapter.decode_action_from_hidden(hidden, obs, aux=aux)
        report = {
            "success": True,
            "hidden_tokens_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype),
            "hidden_device": str(hidden.device),
            "action_shape": list(action.shape),
            "base_action_available": True,
            "adapter_action_available": True,
            "uses_real_starvla_hidden": True,
            "uses_surrogate_hidden": False,
            "base_checkpoint": args.base_checkpoint,
            "last_hidden_shape": aux.get("last_hidden_shape"),
        }
        (out_dir / "hidden_shape_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        (out_dir / "starvla_hidden_adapter_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as exc:
        write_failure(out_dir, "probe_starvla_hidden_tokens", exc, attempts)
        report = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "uses_real_starvla_hidden": False,
            "uses_surrogate_hidden": False,
            "attempts": attempts,
        }
        (out_dir / "hidden_shape_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
