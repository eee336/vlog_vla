#!/usr/bin/env bash
# Evaluate Stage 6 checkpoint on all 4 LIBERO suites (default 4-way parallel).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_stage6_eval}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifest.jsonl}"
BASE_PORT="${BASE_PORT:-5800}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
WORKER="${SCRIPT_DIR}/run_one_libero_eval_job.sh"
CKPT="${CKPT:-${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_FULL_REAL/checkpoints/steps_100000_pytorch_model.pt}"
RUN_ID="${RUN_ID:-VLOG_VLA_LIBERO_FULL_REAL}"
SUITES="libero_spatial libero_object libero_goal libero_10"

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }

port="${BASE_PORT}"
for suite in ${SUITES}; do
  eval_dir="${OUTPUT_ROOT}/${RUN_ID}/${suite}"
  printf '%s|%s|%s|%s|%s\n' "${RUN_ID}" "${suite}" "${CKPT}" "${port}" "${eval_dir}"
  port=$((port + 1))
done | xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
  IFS="|" read -r run_id suite ckpt port eval_dir <<< "{}"
  OUTPUT_ROOT="'"${OUTPUT_ROOT}"'" MANIFEST="'"${MANIFEST}"'" GPU_ID="'"${GPU_ID}"'" \
    exec bash "'"${WORKER}"'" "$run_id" "$suite" "$ckpt" "$port" "$eval_dir"
'

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla
python3 "${REPO_ROOT}/scripts/vlog_vla/collect_libero_success_table.py" \
  --manifest "${MANIFEST}" --output-dir "${OUTPUT_ROOT}"
python3 "${REPO_ROOT}/scripts/vlog_vla/plot_libero_success.py" \
  --input "${OUTPUT_ROOT}/libero_success_table.json" \
  --output-dir "${OUTPUT_ROOT}/figures"

echo "Stage 6 eval done -> ${OUTPUT_ROOT}"
