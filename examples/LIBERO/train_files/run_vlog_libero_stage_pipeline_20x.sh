#!/usr/bin/env bash
# Run LIBERO VLOG-VLA Stage 1-6 sequentially.
# Each stage uses 20x default steps; checkpoint saved only at stage end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SH="${SCRIPT_DIR}/run_vlog_libero_train.sh"
LOG_ROOT="${REPO_ROOT}/outputs/vlog_libero_stage_pipeline_20x"
mkdir -p "${LOG_ROOT}"

COMMON=(
  USE_DEEPSPEED=0
  VIDEO_BACKEND=pyav
  EVAL_INTERVAL=100000000
  CONFIG_YAML=examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
  BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  DATA_ROOT="${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA"
  RUN_ROOT_DIR=results/Checkpoints
  PER_DEVICE_BATCH_SIZE=1
  NUM_PROCESSES=1
  DATA_MIX=libero_all
)

CURRENT_CKPT=""

run_stage() {
  local stage_name="$1"
  local run_id="$2"
  local train_stage="$3"
  local max_steps="$4"
  local base_ckpt="$5"
  local log_file="${LOG_ROOT}/${run_id}.log"
  local ckpt="${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${max_steps}_pytorch_model.pt"

  echo "================================================================"
  echo "[$(date -Is)] START ${stage_name}: run_id=${run_id}, steps=${max_steps}"
  echo "  base_ckpt=${base_ckpt}"
  echo "  log=${log_file}"
  echo "================================================================"

  cd "${REPO_ROOT}"
  env "${COMMON[@]}" \
    RUN_ID="${run_id}" \
    TRAIN_STAGE="${train_stage}" \
    MAX_TRAIN_STEPS="${max_steps}" \
    SAVE_INTERVAL="${max_steps}" \
    BASE_CKPT="${base_ckpt}" \
    bash "${TRAIN_SH}" 2>&1 | tee "${log_file}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: expected checkpoint not found: ${ckpt}" >&2
    exit 1
  fi

  CURRENT_CKPT="${ckpt}"
  echo "[$(date -Is)] DONE ${stage_name}: ${CURRENT_CKPT}"
}

BASE_STARVLA="${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt"

run_stage "Stage 1" "VLOG_VLA_LIBERO_STAGE1_REAL" "stage1_preserve" 20000 "${BASE_STARVLA}"
run_stage "Stage 2" "VLOG_VLA_LIBERO_STAGE2_REAL" "stage2_option_discovery" 60000 "${CURRENT_CKPT}"
run_stage "Stage 3" "VLOG_VLA_LIBERO_STAGE3_REAL" "stage3_graph" 20000 "${CURRENT_CKPT}"
run_stage "Stage 4" "VLOG_VLA_LIBERO_STAGE4_REAL" "stage4_critic" 20000 "${CURRENT_CKPT}"
run_stage "Stage 5" "VLOG_VLA_LIBERO_STAGE5_REAL" "stage5_router" 20000 "${CURRENT_CKPT}"
run_stage "Stage 6" "VLOG_VLA_LIBERO_FULL_REAL" "stage6_full" 100000 "${CURRENT_CKPT}"

SUMMARY="${LOG_ROOT}/pipeline_summary.txt"
{
  echo "VLOG-VLA Stage 1-6 pipeline (20x steps) completed at $(date -Is)"
  echo "Final checkpoint: ${CURRENT_CKPT}"
  find "${REPO_ROOT}/results/Checkpoints" -path "*VLOG_VLA_LIBERO*/*/steps_*_pytorch_model.pt" -print | sort
} | tee "${SUMMARY}"

echo "Pipeline complete. Summary: ${SUMMARY}"
