from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TOTAL_SR_RE = re.compile(r"Total success rate:\s*([0-9]*\.?[0-9]+)")
TOTAL_EP_RE = re.compile(r"Total episodes:\s*(\d+)")


def parse_eval_log(path: Path) -> dict:
    text = path.read_text(errors="replace") if path.exists() else ""
    success_rates = [float(match.group(1)) for match in TOTAL_SR_RE.finditer(text)]
    episodes = [int(match.group(1)) for match in TOTAL_EP_RE.finditer(text)]
    success_rate = success_rates[-1] if success_rates else None
    total_episodes = episodes[-1] if episodes else None
    return {
        "log_path": str(path),
        "success_rate": success_rate,
        "success_rate_percent": None if success_rate is None else round(success_rate * 100.0, 2),
        "total_episodes": total_episodes,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["suite", "data_mix", "run_id", "checkpoint", "success_rate", "success_rate_percent", "total_episodes", "log_path"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| LIBERO task suite | data_mix | run_id | checkpoint | success rate | episodes |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        sr = "N/A" if row["success_rate_percent"] is None else f'{row["success_rate_percent"]:.2f}%'
        episodes = "N/A" if row["total_episodes"] is None else str(row["total_episodes"])
        lines.append(
            f'| {row["suite"]} | {row["data_mix"]} | {row["run_id"]} | {row["checkpoint"]} | {sr} | {episodes} |'
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSONL manifest written by run_vlog_libero_all_tasks.sh")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.update(parse_eval_log(Path(row["eval_log"])))
        rows.append(row)

    (output_dir / "libero_success_table.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_csv(output_dir / "libero_success_table.csv", rows)
    write_markdown(output_dir / "libero_success_table.md", rows)
    print(json.dumps({"rows": rows, "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
