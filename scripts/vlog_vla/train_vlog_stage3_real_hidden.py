from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlog_vla.stage7_common import parse_config, save_checkpoint, transition_matrix, write_failure
from scripts.vlog_vla.train_common import duration_histogram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = parse_config(args.config)
    try:
        ckpt = torch.load(args.stage2_checkpoint, map_location="cpu")
        option_idx = ckpt.get("extra", {}).get("option_idx", [])
        num_options = int(cfg["vlog"]["num_options"])
        matrix = transition_matrix(option_idx, num_options)
        nonzero = sum(1 for row in matrix for value in row if value > 0)
        sparsity = 1.0 - nonzero / max(1, num_options * num_options)
        duration = duration_histogram(option_idx)
        log = out_dir / "train_log.jsonl"
        log.write_text(json.dumps({"transition_nonzero": nonzero, "observed_sparsity": sparsity, "mean_duration": duration["mean_duration"]}) + "\n")
        (out_dir / "option_transition_matrix_real_hidden.json").write_text(json.dumps({"num_options": num_options, "matrix": matrix, "uses_real_starvla_hidden": True, "uses_surrogate_hidden": False}, indent=2, sort_keys=True))
        (out_dir / "edge_sparsity_report_real_hidden.json").write_text(json.dumps({"observed_nonzero_edges": nonzero, "observed_sparsity": sparsity, "mean_option_duration": duration["mean_duration"], "uses_real_starvla_hidden": True, "uses_surrogate_hidden": False}, indent=2, sort_keys=True))
        save_checkpoint(out_dir / "checkpoint.pt", "stage3_real_hidden", cfg, torch.nn.Module(), {"option_idx": option_idx, "transition_matrix": matrix})
        print(json.dumps({"observed_sparsity": sparsity, "mean_option_duration": duration["mean_duration"]}, indent=2, sort_keys=True))
    except Exception as exc:
        write_failure(out_dir, "stage3_real_hidden", exc, ["load stage2 checkpoint option_idx", "compute transition matrix"])
        raise


if __name__ == "__main__":
    main()
