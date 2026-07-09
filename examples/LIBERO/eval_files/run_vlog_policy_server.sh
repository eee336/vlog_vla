#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${STARVLA_DIR}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt}"
VLOG_CHECKPOINT="${VLOG_CHECKPOINT:-}"
CKPT="${CKPT:-${VLOG_CHECKPOINT:-${BASE_CHECKPOINT}}}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
USE_BF16="${USE_BF16:-1}"
VLOG_LOG_DIR="${VLOG_LOG_DIR:-${STARVLA_DIR}/results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_FULL/eval_option_logs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_checkpoint) BASE_CHECKPOINT="$2"; CKPT="${CKPT:-$2}"; shift 2 ;;
    --vlog_checkpoint) VLOG_CHECKPOINT="$2"; CKPT="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --output_dir) VLOG_LOG_DIR="$2"; shift 2 ;;
    --vlog_log_dir) VLOG_LOG_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --framework_name QwenOFTVLOG
  --vlog_checkpoint "${VLOG_CHECKPOINT}"
  --enable_vlog_logging
  --vlog_log_dir "${VLOG_LOG_DIR}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
