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

from scripts.vlog_vla.stage7_common import action_window_pseudo_options, load_base_policy, make_vlog_from_cfg, option_usage, parse_config, save_checkpoint, write_failure
from scripts.vlog_vla.train_common import duration_histogram
from starVLA.model.vlog_vla.losses import option_balance_loss, router_distill_loss
from starVLA.model.vlog_vla.starvla_hidden_adapter import StarVLAHiddenAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_config(args.config)
    attempts = ["load base StarVLA", "extract real hidden", "train VLOG stage2 on real future action windows"]
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = load_base_policy(cfg["base"]["checkpoint"], device=str(device))
        adapter = StarVLAHiddenAdapter(base, {"allow_surrogate_hidden": False})
        vlog = make_vlog_from_cfg(cfg, int(base.config.framework.action_model.action_hidden_dim), device)
        if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
            state = torch.load(args.stage1_checkpoint, map_location=device)
            vlog.load_state_dict(state["vlog_state_dict"], strict=False)
        opt = torch.optim.AdamW(vlog.parameters(), lr=float(cfg["training"].get("option_lr", 1e-4)))
        rows = _rows(cfg)
        option_idx_all = []
        losses = []
        log = out_dir / "train_log.jsonl"
        log.write_text("")
        steps = int(cfg["training"].get("max_steps_stage2", 8))
        for step in range(steps):
            row = rows[(step * 37) % len(rows)]
            examples = [_example(row, cfg)]
            with torch.no_grad():
                hidden, _ = adapter.encode_hidden(examples)
            future = _future(row, cfg, hidden.device, hidden.dtype)
            robot_state = torch.tensor(np.asarray(examples[0]["state"], dtype=np.float32).reshape(1, -1), device=hidden.device, dtype=hidden.dtype)
            out = vlog.forward_train({"hidden_tokens": hidden.detach(), "robot_state": robot_state, "future_actions": future}, stage=2)
            pseudo_idx = action_window_pseudo_options(future.float(), int(cfg["vlog"]["num_options"]))
            loss_distill = router_distill_loss(out["router_logits"], pseudo_idx)
            loss_balance = option_balance_loss(out["option_probs"])
            loss = loss_distill + float(cfg["loss"].get("lambda_balance", 0.05)) * loss_balance + out.get("vq_loss", torch.zeros_like(loss_distill))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            option_idx_all.extend(pseudo_idx.detach().cpu().tolist())
            rec = {"step": step, "loss": float(loss.detach().cpu()), "router_distill_loss": float(loss_distill.detach().cpu()), "vq_loss": float(out.get("vq_loss", torch.zeros_like(loss)).detach().cpu())}
            losses.append(rec)
            with log.open("a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        usage = option_usage(option_idx_all, int(cfg["vlog"]["num_options"]))
        (out_dir / "option_usage_real_hidden.json").write_text(json.dumps(usage, indent=2, sort_keys=True))
        (out_dir / "router_distill_report_real_hidden.json").write_text(json.dumps({"uses_real_starvla_hidden": True, "uses_surrogate_hidden": False, "last": losses[-1], "num_samples": len(option_idx_all)}, indent=2, sort_keys=True))
        (out_dir / "option_duration_histogram_real_hidden.json").write_text(json.dumps(duration_histogram(option_idx_all), indent=2, sort_keys=True))
        save_checkpoint(out_dir / "checkpoint.pt", "stage2_real_hidden", cfg, vlog, {"option_idx": option_idx_all, "usage": usage})
        print(json.dumps({"usage": usage, "last": losses[-1]}, indent=2, sort_keys=True))
    except Exception as exc:
        write_failure(out_dir, "stage2_real_hidden", exc, attempts)
        raise


def _rows(cfg: dict) -> list[dict]:
    return [json.loads(line) for line in Path(cfg["dataset"]["jsonl_path"]).read_text().splitlines() if line.strip()]


def _example(row: dict, cfg: dict) -> dict:
    root = Path(cfg["dataset"]["jsonl_path"]).parent
    return {"image": Image.open(root / row["image"]).convert("RGB"), "lang": row["lang"], "state": np.asarray(row["state"], dtype=np.float32)[None, :], "action": np.asarray(row["action_chunk"], dtype=np.float32)}


def _future(row: dict, cfg: dict, device, dtype) -> torch.Tensor:
    window = int(cfg["dataset"]["window_size"])
    action = np.asarray(row["action_chunk"], dtype=np.float32)
    if action.shape[0] < window:
        action = np.concatenate([action, np.repeat(action[-1:], window - action.shape[0], axis=0)], axis=0)
    return torch.tensor(action[:window, : int(cfg["dataset"]["action_dim"])][None], device=device, dtype=dtype)


if __name__ == "__main__":
    main()
