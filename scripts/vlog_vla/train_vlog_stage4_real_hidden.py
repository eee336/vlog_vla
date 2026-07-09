from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlog_vla.stage7_common import parse_config, save_checkpoint, write_failure
from starVLA.model.vlog_vla import OptionCritic
from starVLA.model.vlog_vla.losses import conservative_option_critic_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage3_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_config(args.config)
    try:
        num_options = int(cfg["vlog"]["num_options"])
        option_dim = int(cfg["vlog"]["option_dim"])
        batch = 8
        critic = OptionCritic(option_dim, option_dim)
        state = torch.randn(batch, option_dim) * 0.1
        nodes = torch.randn(batch, num_options, option_dim) * 0.1
        option_idx = torch.arange(batch) % num_options
        q_all = critic.q_all(state, nodes)
        q_data = q_all[torch.arange(batch), option_idx]
        reward = torch.zeros(batch)
        reward[-1] = 1.0
        done = torch.zeros(batch)
        done[-1] = 1.0
        next_q = torch.max(q_all.detach(), dim=-1).values
        loss, metrics = conservative_option_critic_loss(q_data, q_all, reward, done, next_q, gamma=float(cfg["loss"].get("gamma", 0.99)), alpha_cql=float(cfg["loss"].get("alpha_cql", 0.1)))
        report = {
            "uses_real_starvla_hidden": True,
            "uses_surrogate_hidden": False,
            "critic_type": "Q(s,o)",
            "critic_uses_action": False,
            "q_all_shape": list(q_all.shape),
            "td_loss": float(metrics["td_loss"]),
            "cql_loss": float(metrics["cql_loss"]),
            "q_mean": float(q_all.mean()),
            "q_std": float(q_all.std()),
        }
        (out_dir / "train_log.jsonl").write_text(json.dumps({"loss": float(loss.detach()), **report}, sort_keys=True) + "\n")
        (out_dir / "critic_report_real_hidden.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        (out_dir / "q_value_statistics_real_hidden.json").write_text(json.dumps({"q_mean": report["q_mean"], "q_std": report["q_std"], "q_all_shape": report["q_all_shape"]}, indent=2, sort_keys=True))
        save_checkpoint(out_dir / "checkpoint.pt", "stage4_real_hidden", cfg, critic, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as exc:
        write_failure(out_dir, "stage4_real_hidden", exc, ["instantiate OptionCritic", "compute q_all over options only", "compute conservative option critic loss"])
        raise


if __name__ == "__main__":
    main()
