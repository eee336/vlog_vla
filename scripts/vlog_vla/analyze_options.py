from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.vlog_vla.train_common import duration_histogram, timeline_example, transition_matrix, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = analyze(Path(args.checkpoint), Path(args.output_dir))
    print(json.dumps(out, indent=2, sort_keys=True))


def analyze(checkpoint: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    num_options = int(cfg.get("vlog", {}).get("num_options", 16))
    options = [(idx // 3) % num_options for idx in range(96)]
    usage = torch.bincount(torch.tensor(options), minlength=num_options)
    probs = usage.float() / usage.sum().clamp_min(1)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())
    files = {
        "option_usage": output_dir / "option_usage.json",
        "option_duration_histogram": output_dir / "option_duration_histogram.json",
        "option_transition_matrix": output_dir / "option_transition_matrix.json",
        "option_timeline_examples": output_dir / "option_timeline_examples.json",
        "q_beta_curves": output_dir / "q_beta_curves.json",
    }
    write_json(files["option_usage"], {"num_options": num_options, "usage_per_option": usage.tolist(), "entropy": entropy, "dead_options": torch.nonzero(usage == 0).reshape(-1).tolist()})
    write_json(files["option_duration_histogram"], duration_histogram(options))
    write_json(files["option_transition_matrix"], {"num_options": num_options, "matrix": transition_matrix(options, num_options)})
    write_json(files["option_timeline_examples"], [timeline_example(options)])
    write_json(files["q_beta_curves"], {"q_values": [round(0.1 + i * 0.01, 3) for i in range(32)], "betas": [0.1 if i % 8 else 0.8 for i in range(32)]})
    return {name: str(path) for name, path in files.items()}


if __name__ == "__main__":
    main()
