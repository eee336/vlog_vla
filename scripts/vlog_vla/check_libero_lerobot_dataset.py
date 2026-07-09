from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SUBSETS = [
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modality_template", default="examples/LIBERO/train_files/modality.json")
    args = parser.parse_args()
    report = check_dataset(Path(args.data_root), Path(args.modality_template))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def check_dataset(data_root: Path, modality_template: Path) -> dict:
    root = data_root.resolve()
    subsets = {}
    all_ok = root.exists()
    for name in SUBSETS:
        path = root / name
        meta = path / "meta"
        modality = meta / "modality.json"
        copied_modality = False
        if path.exists() and meta.exists() and not modality.exists() and modality_template.exists():
            shutil.copy2(modality_template, modality)
            copied_modality = True
        files = {
            "exists": path.exists(),
            "meta_exists": meta.exists(),
            "data_exists": (path / "data").exists(),
            "videos_exists": (path / "videos").exists(),
            "modality_json_exists": modality.exists(),
            "stats_json_exists": (meta / "stats.json").exists(),
            "info_json_exists": (meta / "info.json").exists(),
            "episodes_jsonl_exists": (meta / "episodes.jsonl").exists(),
            "tasks_jsonl_exists": (meta / "tasks.jsonl").exists(),
            "copied_modality_json": copied_modality,
        }
        files["ok_for_loader_smoke"] = files["exists"] and files["meta_exists"] and files["data_exists"] and files["videos_exists"] and files["modality_json_exists"]
        all_ok = all_ok and files["ok_for_loader_smoke"]
        subsets[name] = files
    return {
        "data_root": str(root),
        "all_required_subsets_loader_ready": all_ok,
        "action_dim_expected": 7,
        "state_dim_config": 8,
        "language_field_expected": "annotation.human.action.task_description",
        "subsets": subsets,
    }


if __name__ == "__main__":
    main()
