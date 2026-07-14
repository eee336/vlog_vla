#!/usr/bin/env python3
"""Plot LIBERO eval success rates: bar + line charts vs training stage."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SUITE_ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}

RUN_ORDER = [
    "StarVLA_Pretrained",
    "VLOG_VLA_LIBERO_STAGE1_REAL",
    "VLOG_VLA_LIBERO_STAGE2_REAL",
    "VLOG_VLA_LIBERO_STAGE3_REAL",
    "VLOG_VLA_LIBERO_STAGE4_REAL",
    "VLOG_VLA_LIBERO_STAGE5_REAL",
]

RUN_LABELS = {
    "StarVLA_Pretrained": "StarVLA\n(pretrained)",
    "VLOG_VLA_LIBERO_STAGE1_REAL": "Stage 1",
    "VLOG_VLA_LIBERO_STAGE2_REAL": "Stage 2",
    "VLOG_VLA_LIBERO_STAGE3_REAL": "Stage 3",
    "VLOG_VLA_LIBERO_STAGE4_REAL": "Stage 4",
    "VLOG_VLA_LIBERO_STAGE5_REAL": "Stage 5",
}

STAGE_IDX = {
    "StarVLA_Pretrained": 0,
    "VLOG_VLA_LIBERO_STAGE1_REAL": 1,
    "VLOG_VLA_LIBERO_STAGE2_REAL": 2,
    "VLOG_VLA_LIBERO_STAGE3_REAL": 3,
    "VLOG_VLA_LIBERO_STAGE4_REAL": 4,
    "VLOG_VLA_LIBERO_STAGE5_REAL": 5,
}

COLORS = {
    "libero_spatial": "#4C72B0",
    "libero_object": "#55A868",
    "libero_goal": "#C44E52",
    "libero_10": "#8172B3",
}


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_matrix(rows: list[dict]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {run: {} for run in RUN_ORDER}
    for row in rows:
        run_id = row["run_id"]
        suite = row["suite"]
        if run_id in matrix and row.get("success_rate_percent") is not None:
            matrix[run_id][suite] = float(row["success_rate_percent"])
    return matrix


def plot_grouped_bar(matrix: dict[str, dict[str, float]], out: Path) -> None:
    n_runs = len(RUN_ORDER)
    n_suites = len(SUITE_ORDER)
    x = np.arange(n_runs)
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, suite in enumerate(SUITE_ORDER):
        vals = [matrix[run].get(suite, np.nan) for run in RUN_ORDER]
        offset = (i - (n_suites - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=SUITE_LABELS[suite], color=COLORS[suite], alpha=0.9)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[r] for r in RUN_ORDER], fontsize=10)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(90, 101)
    ax.set_title("LIBERO Success Rate by Training Stage (50 trials × 10 tasks per suite)")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_line_by_suite(matrix: dict[str, dict[str, float]], out: Path) -> None:
    stages = [STAGE_IDX[r] for r in RUN_ORDER]
    stage_labels = ["Pretrained", "S1", "S2", "S3", "S4", "S5"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for suite in SUITE_ORDER:
        vals = [matrix[run].get(suite, np.nan) for run in RUN_ORDER]
        ax.plot(stages, vals, marker="o", linewidth=2, markersize=7,
                label=SUITE_LABELS[suite], color=COLORS[suite])
        for s, v in zip(stages, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}", (s, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=COLORS[suite])

    ax.set_xticks(stages)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(93, 100.5)
    ax.set_title("LIBERO Success Rate vs VLOG Training Stage")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_baseline_comparison(matrix: dict[str, dict[str, float]], out: Path) -> None:
    """Bar chart: StarVLA pretrained vs best VLOG stage per suite."""
    pretrained = matrix["StarVLA_Pretrained"]
    best_vlog: dict[str, tuple[str, float]] = {}
    for suite in SUITE_ORDER:
        best_run, best_val = "", -1.0
        for run in RUN_ORDER[1:]:
            val = matrix[run].get(suite)
            if val is not None and val > best_val:
                best_val, best_run = val, run
        best_vlog[suite] = (best_run, best_val)

    x = np.arange(len(SUITE_ORDER))
    width = 0.35
    pre_vals = [pretrained.get(s, np.nan) for s in SUITE_ORDER]
    vlog_vals = [best_vlog[s][1] for s in SUITE_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, pre_vals, width, label="StarVLA Pretrained", color="#888888")
    b2 = ax.bar(x + width / 2, vlog_vals, width, label="Best VLOG Stage", color="#4C72B0")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}%",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([SUITE_LABELS[s] for s in SUITE_ORDER])
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(90, 101)
    ax.set_title("StarVLA Pretrained vs Best VLOG Stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_average_line(matrix: dict[str, dict[str, float]], out: Path) -> None:
    stages = [STAGE_IDX[r] for r in RUN_ORDER]
    stage_labels = ["Pretrained", "S1", "S2", "S3", "S4", "S5"]
    avg_vals = []
    for run in RUN_ORDER:
        vals = [matrix[run].get(s) for s in SUITE_ORDER if matrix[run].get(s) is not None]
        avg_vals.append(sum(vals) / len(vals) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stages, avg_vals, marker="s", linewidth=2.5, markersize=9, color="#333333")
    for s, v in zip(stages, avg_vals):
        ax.annotate(f"{v:.2f}%", (s, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(stages)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Average Success Rate (%)")
    ax.set_ylim(96, 99.5)
    ax.set_title("Average LIBERO Success Rate (4 suites)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary_md(matrix: dict[str, dict[str, float]], out: Path) -> None:
    lines = [
        "# LIBERO Eval Summary",
        "",
        "## StarVLA Pretrained (baseline, 4 suites × 500 episodes)",
        "",
        "| Suite | Success Rate |",
        "| --- | ---: |",
    ]
    for suite in SUITE_ORDER:
        v = matrix["StarVLA_Pretrained"].get(suite, float("nan"))
        lines.append(f"| {SUITE_LABELS[suite]} | {v:.2f}% |")

    pre_avg = np.mean([matrix["StarVLA_Pretrained"][s] for s in SUITE_ORDER])
    lines += ["", f"**Average: {pre_avg:.2f}%**", "", "## All checkpoints", ""]
    lines.append("| Stage | Spatial | Object | Goal | LIBERO-10 | Avg |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for run in RUN_ORDER:
        vals = [matrix[run].get(s, float("nan")) for s in SUITE_ORDER]
        avg = np.nanmean(vals)
        label = RUN_LABELS[run].replace("\n", " ")
        lines.append(f"| {label} | {vals[0]:.1f}% | {vals[1]:.1f}% | {vals[2]:.1f}% | {vals[3]:.1f}% | {avg:.2f}% |")

    lines += [
        "",
        "## Figures",
        "",
        "- `libero_success_grouped_bar.png` — grouped bar by stage",
        "- `libero_success_line_by_suite.png` — line chart per suite",
        "- `libero_success_baseline_vs_best.png` — pretrained vs best VLOG",
        "- `libero_success_average_line.png` — 4-suite average vs stage",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("outputs/libero_stage_eval/libero_success_table.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/libero_stage_eval/figures"))
    args = parser.parse_args()

    rows = load_rows(args.input)
    matrix = build_matrix(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_bar(matrix, args.output_dir / "libero_success_grouped_bar.png")
    plot_line_by_suite(matrix, args.output_dir / "libero_success_line_by_suite.png")
    plot_baseline_comparison(matrix, args.output_dir / "libero_success_baseline_vs_best.png")
    plot_average_line(matrix, args.output_dir / "libero_success_average_line.png")
    write_summary_md(matrix, args.output_dir / "libero_eval_summary.md")

    print(json.dumps({"figures_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
