from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlog_vla.train_common import (
    action_window_pseudo_options,
    duration_histogram,
    edge_sparse_loss,
    graph_transition_loss,
    load_jsonl_rows,
    make_real_jsonl_batch,
    option_balance_loss,
    router_distill_loss,
    transition_matrix,
    write_json,
    write_yaml,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.vlog_vla.losses import conservative_option_critic_loss
from starVLA.model.vlog_vla.vlog_policy_wrapper import PersistenceConfig, VLOGPolicyWrapper


STAGE7_ROOT = Path("outputs/vlog_stage7_real_starvla")


def parse_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_base_policy(checkpoint: str, device: str = "cuda"):
    model = baseframework.from_pretrained(checkpoint)
    return model.to(device).eval()


def make_vlog_from_cfg(cfg: dict, hidden_dim: int, device: torch.device) -> VLOGPolicyWrapper:
    vlog_cfg = cfg["vlog"]
    dataset = cfg["dataset"]
    return VLOGPolicyWrapper(
        hidden_dim=hidden_dim,
        state_dim=int(dataset.get("state_dim", 8)),
        action_dim=int(dataset.get("action_dim", 7)),
        num_options=int(vlog_cfg.get("num_options", 16)),
        option_dim=int(vlog_cfg.get("option_dim", 256)),
        window_size=int(dataset.get("window_size", 10)),
        persistence=PersistenceConfig(
            d_min=int(vlog_cfg.get("d_min", 2)),
            d_max=int(vlog_cfg.get("d_max", 8)),
            beta_threshold=float(vlog_cfg.get("beta_threshold", 0.7)),
            q_switch_margin=float(vlog_cfg.get("q_switch_margin", 0.05)),
        ),
        commitment_cost=float(vlog_cfg.get("commitment_cost", 0.25)),
        alpha_init=0.0,
    ).to(device)


def make_stage7_batch(cfg: dict, device: torch.device, step: int, batch_size: int | None = None) -> dict:
    data_cfg = cfg["dataset"]
    compat = {
        "vlog": {
            "hidden_dim": int(cfg["vlog"].get("hidden_dim", 256)),
            "num_tokens": int(cfg["vlog"].get("num_tokens", 8)),
            "state_dim": int(data_cfg.get("state_dim", 8)),
            "action_dim": int(data_cfg.get("action_dim", 7)),
            "window_size": int(data_cfg.get("window_size", 10)),
            "num_options": int(cfg["vlog"].get("num_options", 16)),
        },
        "data": {"use_real_jsonl": True, "jsonl_path": data_cfg["jsonl_path"], "episode_mod": 128},
    }
    return make_real_jsonl_batch(compat, batch_size=batch_size or int(data_cfg.get("batch_size", 2)), device=device, step=step)


def write_failure(output_dir: Path, title: str, exc: BaseException, attempts: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "HIDDEN_EXTRACTION_FAILURE_REPORT.md").write_text(
        "# Hidden Extraction Failure\n\n"
        f"Stage: {title}\n\n"
        f"Error: `{type(exc).__name__}: {exc}`\n\n"
        "Attempts:\n\n"
        + "\n".join(f"- {item}" for item in attempts)
        + "\n\nRequired human confirmation: inspect `QwenOFT.forward`, `QwenOFT.predict_action`, and `QwenOFTVLOG.predict_action`.\n"
    )


def save_checkpoint(path: Path, stage: str, cfg: dict, vlog: VLOGPolicyWrapper, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": stage, "config": cfg, "vlog_state_dict": vlog.state_dict(), "extra": extra or {}}, path)


def option_usage(idx: list[int], num_options: int) -> dict:
    counts = torch.bincount(torch.tensor(idx, dtype=torch.long), minlength=num_options)
    probs = counts.float() / counts.sum().clamp_min(1)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())
    return {
        "num_options": num_options,
        "usage_per_option": counts.tolist(),
        "entropy": entropy,
        "dead_options": torch.nonzero(counts == 0).reshape(-1).tolist(),
        "num_used_options": int((counts > 0).sum().item()),
        "uses_real_starvla_hidden": True,
        "uses_surrogate_hidden": False,
    }


def stage_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--stage2_checkpoint", default=None)
    parser.add_argument("--stage3_checkpoint", default=None)
    return parser
