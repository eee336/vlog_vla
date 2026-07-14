#!/usr/bin/env bash
# Run StarVLA original pretrained checkpoint on all 4 LIBERO suites (4-way parallel).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_starvla_pretrained_baseline}"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
BASE_PORT="${BASE_PORT:-5694}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
WORKER="${SCRIPT_DIR}/run_one_libero_eval_job.sh"
CKPT="${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt"
RUN_ID="StarVLA_Pretrained"
SUITES="libero_spatial libero_object libero_goal libero_10"

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

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

echo "Done -> ${OUTPUT_ROOT}"
