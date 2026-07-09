#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="playground/Datasets/LEROBOT_LIBERO_DATA")
    parser.add_argument("--template", default="examples/LIBERO/train_files/modality.json")
    parser.add_argument("--output", default="outputs/vlog_real_train_eval/modality_prepare_report.json")
    args = parser.parse_args()
    report = prepare(Path(args.data_root), Path(args.template))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def prepare(data_root: Path, template: Path) -> dict:
    subsets = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    ]
    rows = []
    for subset in subsets:
        root = data_root / subset
        meta = root / "meta"
        target = meta / "modality.json"
        row = {"subset": subset, "path": str(root), "exists": root.exists(), "modality_exists_before": target.exists(), "copied": False}
        if root.exists() and not target.exists() and template.exists():
            meta.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template, target)
            row["copied"] = True
        row["modality_exists_after"] = target.exists()
        rows.append(row)
    return {"data_root": str(data_root), "template": str(template), "subsets": rows}


if __name__ == "__main__":
    main()
