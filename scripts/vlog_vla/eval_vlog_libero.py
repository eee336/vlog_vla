from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="outputs/vlog_eval")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "standard_libero_eval": "placeholder_smoke",
        "libero_long_eval": "placeholder_smoke",
        "perturbation_eval": "placeholder_smoke",
        "uses_persistent_options": True,
        "critic_type": "Q(s,o)",
        "success_rate": None,
        "note": "Smoke eval wrapper. Replace placeholder backend with official LIBERO server-client runner for full benchmark.",
    }
    path = out_dir / "eval_vlog_libero.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"report": str(path), **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
