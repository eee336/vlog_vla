#!/usr/bin/env bash
# V2 stage pipeline: tuned steps/params based on V1 eval analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SH="${SCRIPT_DIR}/run_vlog_libero_train.sh"
LOG_ROOT="${REPO_ROOT}/outputs/vlog_libero_stage_pipeline_v2"
mkdir -p "${LOG_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

COMMON=(
  USE_DEEPSPEED=0
  VIDEO_BACKEND=pyav
  EVAL_INTERVAL=100000000
  CONFIG_YAML=examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
  BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  DATA_ROOT="${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA"
  RUN_ROOT_DIR=results/Checkpoints
  PER_DEVICE_BATCH_SIZE=12
  NUM_PROCESSES=1
  DATA_MIX=libero_all
  TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python
  NUM_WORKERS=8
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
    bash "${TRAIN_SH}" 2>&1 | tee "${log_file}"

  # early-stop may save at fewer steps
  ckpt="$(ls -t "${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "ERROR: no checkpoint for ${run_id}" >&2
    exit 1
  fi
  CURRENT_CKPT="${ckpt}"
  echo "[$(date -Is)] DONE ${stage_name}: ${CURRENT_CKPT}"
}

BASE_STARVLA="${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt"
EXTRA_DATA="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true"

# V2 rationale (from V1 eval):
# S1: preserve converges fast -> fewer steps
# S2: 60k was too long, 25k enough for option discovery
# S3: graph modest impact -> 12k
# S4: critic helped goal -> keep 15k
# S5: best overall -> keep 20k
# S6: early-stop on action_loss, higher vlog lr, max 30k

run_stage "Stage 1" "VLOG_VLA_LIBERO_STAGE1_V2" "stage1_preserve" 8000 "${BASE_STARVLA}" "${EXTRA_DATA}"
run_stage "Stage 2" "VLOG_VLA_LIBERO_STAGE2_V2" "stage2_option_discovery" 25000 "${CURRENT_CKPT}" "${EXTRA_DATA}"
run_stage "Stage 3" "VLOG_VLA_LIBERO_STAGE3_V2" "stage3_graph" 12000 "${CURRENT_CKPT}" "${EXTRA_DATA}"
run_stage "Stage 4" "VLOG_VLA_LIBERO_STAGE4_V2" "stage4_critic" 15000 "${CURRENT_CKPT}" "${EXTRA_DATA}"
run_stage "Stage 5" "VLOG_VLA_LIBERO_STAGE5_V2" "stage5_router" 20000 "${CURRENT_CKPT}" "${EXTRA_DATA}"

STAGE6_EXTRA="${EXTRA_DATA} --trainer.learning_rate.vlog 1.0e-04 --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 8000 --trainer.early_stop_metric vlog/action_loss"
run_stage "Stage 6" "VLOG_VLA_LIBERO_FULL_V2" "stage6_full" 30000 "${CURRENT_CKPT}" "${STAGE6_EXTRA}"

{
  echo "V2 pipeline completed at $(date -Is)"
  echo "Final checkpoint: ${CURRENT_CKPT}"
  find "${REPO_ROOT}/results/Checkpoints" -path "*_V2/checkpoints/steps_*_pytorch_model.pt" -print | sort
} | tee "${LOG_ROOT}/pipeline_summary.txt"
