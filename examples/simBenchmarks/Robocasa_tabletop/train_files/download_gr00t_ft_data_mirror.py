#!/usr/bin/env python3
"""Download RoboCasa GR1 finetune data via hf-mirror (curl-based)."""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
LOCAL_DIR = Path("./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim")

FOLDERS = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
]


def curl_request(url: str, max_retries: int = 5) -> tuple[list | dict, dict[str, str]]:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        proc = subprocess.run(
            [
                "curl", "-sfL", "--http1.1",
                "--connect-timeout", "30", "--max-time", "300",
                "-D", "-", url,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            body = proc.stdout
            header_blob, _, payload = body.partition("\r\n\r\n")
            if not payload and "\n\n" in body:
                header_blob, _, payload = body.partition("\n\n")
            headers = {}
            for line in header_blob.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            return json.loads(payload), headers
        last_err = RuntimeError(f"curl exit {proc.returncode}: {proc.stderr[-200:]}")
        time.sleep(min(5 * attempt, 30))
    raise last_err  # type: ignore[misc]


def next_url(headers: dict[str, str]) -> str | None:
    link = headers.get("link", "")
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    if not m:
        return None
    return m.group(1).replace("https://huggingface.co", HF_MIRROR)


def list_folder_files(folder: str) -> list[str]:
    files: list[str] = []
    url = (
        f"{HF_MIRROR}/api/datasets/{REPO_ID}/tree/main/{urllib.parse.quote(folder, safe='.')}"
        f"?recursive=true&limit=1000"
    )
    while url:
        batch, headers = curl_request(url)
        for item in batch:
            if item.get("type") == "file":
                files.append(item["path"])
        url = next_url(headers)
    return files


def download_file(relpath: str, max_retries: int = 5) -> bool:
    dest = LOCAL_DIR / relpath
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_MIRROR}/datasets/{REPO_ID}/resolve/main/{urllib.parse.quote(relpath, safe='/')}"
    for attempt in range(1, max_retries + 1):
        proc = subprocess.run(
            ["curl", "-sfL", "--http1.1", "--connect-timeout", "30", "--max-time", "3600", "-o", str(dest), url],
            capture_output=True,
        )
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
        dest.unlink(missing_ok=True)
        time.sleep(random.uniform(1.0, 3.0) * attempt)
    print(f"Giving up: {relpath}")
    return False


def main() -> None:
    print(f"Using mirror: {HF_MIRROR}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    target_files: list[str] = []
    for folder in FOLDERS:
        print(f"Listing {folder} ...")
        folder_files = list_folder_files(folder)
        print(f"  -> {len(folder_files)} files")
        target_files.extend(folder_files)

    print(f"Total {len(target_files)} files to download.")
    failed: list[str] = []
    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_file, path): path for path in target_files}
        for idx, fut in enumerate(as_completed(futures), start=1):
            path = futures[fut]
            ok = fut.result()
            if ok:
                if idx % 100 == 0 or idx == len(target_files):
                    print(f"[{idx}/{len(target_files)}] latest: {path}")
            else:
                failed.append(path)

    if failed:
        print(f"Failed {len(failed)} files")
        for path in failed[:20]:
            print(f"  - {path}")
        raise SystemExit(1)
    print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
