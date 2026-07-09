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

from scripts.vlog_vla.stage7_common import load_base_policy, make_vlog_from_cfg, parse_config, save_checkpoint, write_failure
from starVLA.model.vlog_vla.starvla_hidden_adapter import StarVLAHiddenAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_config(args.config)
    attempts = ["load base StarVLA", "extract real action-token hidden", "decode base/adapted hidden through action_model"]
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = load_base_policy(cfg["base"]["checkpoint"], device=str(device))
        adapter = StarVLAHiddenAdapter(base, {"allow_surrogate_hidden": False})
        hidden_dim = int(base.config.framework.action_model.action_hidden_dim)
        vlog = make_vlog_from_cfg(cfg, hidden_dim, device)
        rows = _rows(cfg)
        diffs = []
        log = out_dir / "train_log.jsonl"
        log.write_text("")
        steps = int(cfg["training"].get("max_steps_stage1", 3))
        for step in range(steps):
            examples = [_example(rows[step % len(rows)], cfg)]
            with torch.no_grad():
                hidden, aux = adapter.encode_hidden(examples)
                base_action = adapter.decode_action_from_hidden(hidden, examples, aux=aux)
            robot_state = torch.tensor(np.asarray(examples[0]["state"], dtype=np.float32).reshape(1, -1), device=hidden.device, dtype=hidden.dtype)
            out = vlog.forward_train({"hidden_tokens": hidden.detach(), "robot_state": robot_state, "future_actions": _future(rows[step % len(rows)], cfg, hidden.device, hidden.dtype)}, stage=1)
            with torch.no_grad():
                vlog_action = adapter.decode_action_from_hidden(out["adapted_hidden_tokens"], examples, aux=aux)
            diff = torch.abs(vlog_action.float() - base_action.float())
            row = {"step": step, "mean_l1_action_diff": float(diff.mean()), "max_l1_action_diff": float(diff.max())}
            diffs.append(diff.reshape(-1).detach().cpu())
            with log.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        all_diff = torch.cat(diffs)
        report = {
            "uses_real_starvla_hidden": True,
            "uses_surrogate_hidden": False,
            "alpha_value": float(vlog.option_adapter.alpha.detach().cpu().item()),
            "mean_l1_action_diff": float(all_diff.mean()),
            "max_l1_action_diff": float(all_diff.max()),
            "num_batches": steps,
            "base_checkpoint": cfg["base"]["checkpoint"],
        }
        (out_dir / "real_hidden_action_diff_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        save_checkpoint(out_dir / "checkpoint.pt", "stage1_real_hidden", cfg, vlog, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as exc:
        write_failure(out_dir, "stage1_real_hidden", exc, attempts)
        raise


def _rows(cfg: dict) -> list[dict]:
    return [json.loads(line) for line in Path(cfg["dataset"]["jsonl_path"]).read_text().splitlines() if line.strip()]


def _example(row: dict, cfg: dict) -> dict:
    root = Path(cfg["dataset"]["jsonl_path"]).parent
    return {
        "image": Image.open(root / row["image"]).convert("RGB"),
        "lang": row["lang"],
        "state": np.asarray(row["state"], dtype=np.float32)[None, :],
        "action": np.asarray(row["action_chunk"], dtype=np.float32),
    }


def _future(row: dict, cfg: dict, device, dtype) -> torch.Tensor:
    window = int(cfg["dataset"]["window_size"])
    action_dim = int(cfg["dataset"]["action_dim"])
    action = np.asarray(row["action_chunk"], dtype=np.float32)
    if action.shape[0] < window:
        pad = np.repeat(action[-1:], window - action.shape[0], axis=0)
        action = np.concatenate([action, pad], axis=0)
    return torch.tensor(action[:window, :action_dim][None], device=device, dtype=dtype)


if __name__ == "__main__":
    main()
