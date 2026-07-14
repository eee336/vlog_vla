#!/usr/bin/env bash
# Stage 6: train from Stage 5 with early stopping + 4-way parallel LIBERO eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/vlog_libero_stage6_pipeline"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

STAGE5_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE5_REAL/checkpoints/steps_20000_pytorch_model.pt"
RUN_ID="VLOG_VLA_LIBERO_FULL_REAL"
CKPT_DIR="${REPO_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints"
EVAL_OUTPUT="${REPO_ROOT}/outputs/libero_stage6_eval"
EVAL_SCRIPT="${REPO_ROOT}/examples/LIBERO/eval_files/run_libero_eval_stage6.sh"

: "${PER_DEVICE_BATCH_SIZE:=12}"
: "${NUM_WORKERS:=8}"
: "${MAX_TRAIN_STEPS:=100000}"
: "${EARLY_STOP_PATIENCE:=1500}"
: "${EARLY_STOP_WINDOW:=100}"
: "${EARLY_STOP_MIN_DELTA:=0.01}"
: "${EARLY_STOP_MIN_STEPS:=5000}"

find_stage6_ckpt() {
  local ckpt=""
  ckpt="$(ls -t "${CKPT_DIR}"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true)"
  if [[ -n "${ckpt}" ]]; then
    echo "${ckpt}"
    return 0
  fi
  if [[ -f "${REPO_ROOT}/results/Checkpoints/${RUN_ID}/final_model/pytorch_model.pt" ]]; then
    echo "${REPO_ROOT}/results/Checkpoints/${RUN_ID}/final_model/pytorch_model.pt"
    return 0
  fi
  return 1
}

echo "[$(date -Is)] Stage 6 restart (batch=${PER_DEVICE_BATCH_SIZE}, early_stop patience=${EARLY_STOP_PATIENCE})" | tee "${LOG_DIR}/pipeline.log"

if [[ -f "${REPO_ROOT}/results/Checkpoints/${RUN_ID}/logs/train_log.jsonl" ]]; then
  cp "${REPO_ROOT}/results/Checkpoints/${RUN_ID}/logs/train_log.jsonl" \
    "${LOG_DIR}/train_log_before_restart_$(date +%Y%m%d_%H%M%S).jsonl"
  : > "${REPO_ROOT}/results/Checkpoints/${RUN_ID}/logs/train_log.jsonl"
fi

cd "${REPO_ROOT}"
CONFIG_YAML=examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml \
BASE_CKPT="${STAGE5_CKPT}" \
BASE_VLM=playground/Pretrained_models/Qwen3-VL-4B-Instruct \
DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_DATA \
RUN_ROOT_DIR=results/Checkpoints \
RUN_ID="${RUN_ID}" \
TRAIN_STAGE=stage6_full \
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
SAVE_INTERVAL="${MAX_TRAIN_STEPS}" \
EVAL_INTERVAL=100000000 \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
NUM_PROCESSES=1 \
DATA_MIX=libero_all \
USE_DEEPSPEED=0 \
VIDEO_BACKEND=pyav \
TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python \
EXTRA_TRAIN_ARGS="--datasets.vla_data.num_workers ${NUM_WORKERS} --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true --trainer.learning_rate.vlog 5.0e-05 --trainer.early_stop_patience ${EARLY_STOP_PATIENCE} --trainer.early_stop_window ${EARLY_STOP_WINDOW} --trainer.early_stop_min_delta ${EARLY_STOP_MIN_DELTA} --trainer.early_stop_min_steps ${EARLY_STOP_MIN_STEPS}" \
  bash examples/LIBERO/train_files/run_vlog_libero_train.sh \
  2>&1 | tee "${LOG_DIR}/train.log"

STAGE6_CKPT="$(find_stage6_ckpt || true)"
if [[ -z "${STAGE6_CKPT}" ]]; then
  echo "[$(date -Is)] ERROR: no Stage 6 checkpoint found under ${CKPT_DIR}" | tee -a "${LOG_DIR}/pipeline.log"
  exit 1
fi

echo "[$(date -Is)] Stage 6 train done -> ${STAGE6_CKPT}" | tee -a "${LOG_DIR}/pipeline.log"
echo "[$(date -Is)] Stage 6 eval start (4 suites parallel, 50 trials x 10 tasks)" | tee -a "${LOG_DIR}/pipeline.log"

OUTPUT_ROOT="${EVAL_OUTPUT}" \
MANIFEST="${EVAL_OUTPUT}/manifest.jsonl" \
PARALLEL_JOBS=4 \
BASE_PORT=5800 \
GPU_ID=0 \
CKPT="${STAGE6_CKPT}" \
NUM_TRIALS_PER_TASK=50 \
MAX_TASKS=-1 \
  bash "${EVAL_SCRIPT}" \
  2>&1 | tee -a "${LOG_DIR}/eval.log"

echo "[$(date -Is)] Pipeline complete." | tee -a "${LOG_DIR}/pipeline.log"
echo "  checkpoint: ${STAGE6_CKPT}"
echo "  eval:       ${EVAL_OUTPUT}"
