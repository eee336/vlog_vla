#!/usr/bin/env bash
# Download RoboCasa GR1 tabletop finetune data via hf-mirror.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/robocasa_data_download"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_ENDPOINT}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
echo "[data] HF endpoint=${HF_ENDPOINT}"

cd "${REPO_ROOT}"
PYTHONUNBUFFERED=1 python examples/simBenchmarks/Robocasa_tabletop/train_files/download_gr00t_ft_data_mirror.py \
  2>&1 | tee "${LOG_DIR}/download.log"

echo "[data] done -> playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
