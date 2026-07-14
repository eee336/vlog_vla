#!/usr/bin/env bash
# RoboCasa VLOG-VLA Stage 1-6 pipeline (mirrors LIBERO V2 tuned schedule).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAIN_SH="${SCRIPT_DIR}/run_vlog_robocasa_train.sh"
LOG_ROOT="${REPO_ROOT}/outputs/vlog_robocasa_stage_pipeline"
mkdir -p "${LOG_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

COMMON=(
  USE_DEEPSPEED=0
  VIDEO_BACKEND=pyav
  EVAL_INTERVAL=100000000
  CONFIG_YAML=examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_oft_vlog_robocasa.yaml
  BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  DATA_ROOT="${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
  RUN_ROOT_DIR=results/Checkpoints
  PER_DEVICE_BATCH_SIZE=4
  NUM_PROCESSES=1
  DATA_MIX=fourier_gr1_unified_1000
  TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python
)

CURRENT_CKPT=""

run_stage() {
  local stage_name="$1"
  local run_id="$2"
  local train_stage="$3"
  local max_steps="$4"
  local base_ckpt="$5"
  local extra_args="${6:-}"
  local log_file="${LOG_ROOT}/${run_id}.log"
  local ckpt="${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${max_steps}_pytorch_model.pt"

  echo "================================================================"
  echo "[$(date -Is)] START ${stage_name}: run_id=${run_id}, steps=${max_steps}"
  echo "  base_ckpt=${base_ckpt}"
  echo "================================================================"

  cd "${REPO_ROOT}"
  env "${COMMON[@]}" \
    RUN_ID="${run_id}" \
    TRAIN_STAGE="${train_stage}" \
    MAX_TRAIN_STEPS="${max_steps}" \
    SAVE_INTERVAL="${max_steps}" \
    BASE_CKPT="${base_ckpt}" \
    EXTRA_TRAIN_ARGS="${extra_args}" \
    bash "${TRAIN_SH}" >"${log_file}" 2>&1

  ckpt="$(ls -t "${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "ERROR: no checkpoint for ${run_id}" >&2
    tail -50 "${log_file}" >&2
    exit 1
  fi
  CURRENT_CKPT="${ckpt}"
  echo "[$(date -Is)] DONE ${stage_name}: ${CURRENT_CKPT}"
}

BASE_STARVLA="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt"
EXTRA_DATA="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true"

run_stage "Stage 1" "VLOG_VLA_ROBOCASA_STAGE1_REAL" "stage1_preserve" 8000 "${BASE_STARVLA}" "${EXTRA_DATA}"
run_stage "Stage 2" "VLOG_VLA_ROBOCASA_STAGE2_REAL" "stage2_option_discovery" 25000 "${CURRENT_CKPT}" "${EXTRA_DATA}"
run_stage "Stage 3" "VLOG_VLA_ROBOCASA_STAGE3_REAL" "stage3_graph" 12000 "${CURRENT_CKPT}" "${EXTRA_DATA}"
run_stage "Stage 4" "VLOG_VLA_ROBOCASA_STAGE4_REAL" "stage4_critic" 15000 "${CURRENT_CKPT}" "${EXTRA_DATA}"
run_stage "Stage 5" "VLOG_VLA_ROBOCASA_STAGE5_REAL" "stage5_router" 20000 "${CURRENT_CKPT}" "${EXTRA_DATA}"

STAGE6_EXTRA="${EXTRA_DATA} --trainer.learning_rate.vlog 1.0e-04 --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 8000 --trainer.early_stop_metric vlog/action_loss"
run_stage "Stage 6" "VLOG_VLA_ROBOCASA_FULL_REAL" "stage6_full" 30000 "${CURRENT_CKPT}" "${STAGE6_EXTRA}"

{
  echo "RoboCasa pipeline completed at $(date -Is)"
  echo "Final checkpoint: ${CURRENT_CKPT}"
  find "${REPO_ROOT}/results/Checkpoints" -path "*ROBOCASA*/checkpoints/steps_*_pytorch_model.pt" -print | sort
} | tee "${LOG_ROOT}/pipeline_summary.txt"
