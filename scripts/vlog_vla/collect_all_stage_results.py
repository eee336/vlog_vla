#!/usr/bin/env python3
"""Collect and compare LIBERO eval results across training runs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TOTAL_SR_RE = re.compile(r"Total success rate:\s*([0-9]*\.?[0-9]+)")
TOTAL_EP_RE = re.compile(r"Total episodes:\s*(\d+)")


def parse_eval_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    srs = [float(m.group(1)) for m in TOTAL_SR_RE.finditer(text)]
    eps = [int(m.group(1)) for m in TOTAL_EP_RE.finditer(text)]
    if not srs:
        return None
    return {
        "success_rate": srs[-1],
        "success_rate_percent": round(srs[-1] * 100, 2),
        "total_episodes": eps[-1] if eps else None,
    }


def load_run(root: Path, run_id: str, label: str) -> dict:
    suites = {}
    for suite in SUITE_ORDER:
        parsed = parse_eval_log(root / run_id / suite / "eval.log")
        if parsed:
            suites[suite] = parsed
    vals = [suites[s]["success_rate_percent"] for s in SUITE_ORDER if s in suites]
    return {
        "label": label,
        "run_id": run_id,
        "root": str(root),
        "suites": suites,
        "average_percent": round(sum(vals) / len(vals), 2) if vals else None,
        "num_suites_done": len(suites),
    }


def write_markdown(rows: list[dict], out: Path) -> None:
    lines = [
        "# LIBERO Stage Comparison",
        "",
        "| Model | Spatial | Object | Goal | LIBERO-10 | Avg | Done |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        suites = row["suites"]
        cells = [row["label"]]
        for s in SUITE_ORDER:
            cells.append("N/A" if s not in suites else f"{suites[s]['success_rate_percent']:.1f}%")
        avg = "N/A" if row["average_percent"] is None else f"{row['average_percent']:.2f}%"
        lines.append(f"| {' | '.join(cells)} | {avg} | {row['num_suites_done']}/4 |")
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="comparison")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("StarVLA_Pretrained", "StarVLA Pretrained", repo / "outputs/libero_stage_eval"),
        ("VLOG_VLA_LIBERO_STAGE1_REAL", "V1 Stage 1", repo / "outputs/libero_stage_eval"),
        ("VLOG_VLA_LIBERO_STAGE2_REAL", "V1 Stage 2", repo / "outputs/libero_stage_eval"),
        ("VLOG_VLA_LIBERO_STAGE3_REAL", "V1 Stage 3", repo / "outputs/libero_stage_eval"),
        ("VLOG_VLA_LIBERO_STAGE4_REAL", "V1 Stage 4", repo / "outputs/libero_stage_eval"),
        ("VLOG_VLA_LIBERO_STAGE5_REAL", "V1 Stage 5", repo / "outputs/libero_stage_eval"),
        ("VLOG_VLA_LIBERO_FULL_REAL", "V1 Stage 6", repo / "outputs/libero_stage6_eval"),
        ("VLOG_VLA_LIBERO_STAGE1_V2", "V2 Stage 1", repo / "outputs/libero_stage_eval_v2"),
        ("VLOG_VLA_LIBERO_STAGE2_V2", "V2 Stage 2", repo / "outputs/libero_stage_eval_v2"),
        ("VLOG_VLA_LIBERO_STAGE3_V2", "V2 Stage 3", repo / "outputs/libero_stage_eval_v2"),
        ("VLOG_VLA_LIBERO_STAGE4_V2", "V2 Stage 4", repo / "outputs/libero_stage_eval_v2"),
        ("VLOG_VLA_LIBERO_STAGE5_V2", "V2 Stage 5", repo / "outputs/libero_stage_eval_v2"),
        ("VLOG_VLA_LIBERO_FULL_V2", "V2 Stage 6", repo / "outputs/libero_stage_eval_v2"),
    ("VLOG_VLA_LIBERO_FULL_RETRY1", "Retry1 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY2", "Retry2 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY3", "Retry3 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY4", "Retry4 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY5", "Retry5 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY6", "Retry6 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY7", "Retry7 Stage6", repo / "outputs/libero_stage56_eval"),
    ("VLOG_VLA_LIBERO_FULL_RETRY8", "Retry8 Stage6", repo / "outputs/libero_stage56_eval"),
    ]

    rows = []
    for run_id, label, root in runs:
        row = load_run(root, run_id, label)
        if row["num_suites_done"] > 0:
            rows.append(row)

    json_path = args.output_dir / f"{args.tag}.json"
    md_path = args.output_dir / f"{args.tag}.md"
    csv_path = args.output_dir / f"{args.tag}.csv"

    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    write_markdown(rows, md_path)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "run_id", *SUITE_ORDER, "average_percent", "num_suites_done"])
        for row in rows:
            writer.writerow([
                row["label"], row["run_id"],
                *[row["suites"].get(s, {}).get("success_rate_percent") for s in SUITE_ORDER],
                row["average_percent"], row["num_suites_done"],
            ])

    print(json.dumps({"rows": len(rows), "json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
