#!/usr/bin/env bash
# Parallel LIBERO eval: each (checkpoint, suite) runs as an independent job.
# Default PARALLEL_JOBS=6 fits ~10GB/policy-server on a 98GB GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_stage_eval}"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
BASE_PORT="${BASE_PORT:-5694}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
JOB_FILE="${OUTPUT_ROOT}/job_queue.txt"
WORKER="${SCRIPT_DIR}/run_one_libero_eval_job.sh"

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

declare -a CHECKPOINTS=(
  "StarVLA_Pretrained|${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE1_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE1_REAL/checkpoints/steps_20000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE2_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE2_REAL/checkpoints/steps_60000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE3_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE3_REAL/checkpoints/steps_20000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE4_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE4_REAL/checkpoints/steps_20000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE5_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE5_REAL/checkpoints/steps_20000_pytorch_model.pt"
)

job_is_done() {
  grep -q "Total success rate:" "$1" 2>/dev/null
}

build_job_queue() {
  : > "${JOB_FILE}"
  local port="${BASE_PORT}"
  for entry in "${CHECKPOINTS[@]}"; do
    local run_id="${entry%%|*}"
    local ckpt="${entry#*|}"
    [[ -f "${ckpt}" ]] || { echo "[skip] missing ckpt: ${ckpt}" >&2; continue; }
    for suite in ${SUITES}; do
      local eval_dir="${OUTPUT_ROOT}/${run_id}/${suite}"
      local eval_log="${eval_dir}/eval.log"
      if job_is_done "${eval_log}"; then
        echo "[skip] already done: ${run_id}/${suite}" >&2
        continue
      fi
      printf '%s|%s|%s|%s|%s\n' "${run_id}" "${suite}" "${ckpt}" "${port}" "${eval_dir}" >> "${JOB_FILE}"
      port=$((port + 1))
    done
  done
}

export REPO_ROOT OUTPUT_ROOT MANIFEST GPU_ID HOST NUM_TRIALS_PER_TASK MAX_TASKS UNNORM_KEY

build_job_queue
total_jobs=$(wc -l < "${JOB_FILE}" | tr -d ' ')
echo "[$(date -Is)] Parallel LIBERO eval: ${total_jobs} jobs, PARALLEL_JOBS=${PARALLEL_JOBS}" | tee "${OUTPUT_ROOT}/parallel.log"

if [[ "${total_jobs}" == "0" ]]; then
  echo "All jobs already complete."
else
  xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
    IFS="|" read -r run_id suite ckpt port eval_dir <<< "{}"
    exec bash "'"${WORKER}"'" "$run_id" "$suite" "$ckpt" "$port" "$eval_dir"
  ' < "${JOB_FILE}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla
python3 "${REPO_ROOT}/scripts/vlog_vla/collect_libero_success_table.py" \
  --manifest "${MANIFEST}" --output-dir "${OUTPUT_ROOT}"

echo "[$(date -Is)] All parallel eval complete -> ${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/parallel.log"
