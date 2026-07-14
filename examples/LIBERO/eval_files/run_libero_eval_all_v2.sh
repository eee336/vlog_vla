#!/usr/bin/env bash
# Eval all V2 checkpoints + pretrained baseline (4-way parallel per batch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_stage_eval_v2}"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
BASE_PORT="${BASE_PORT:-5900}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
WORKER="${SCRIPT_DIR}/run_one_libero_eval_job.sh"
JOB_FILE="${OUTPUT_ROOT}/job_queue.txt"

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

declare -a CHECKPOINTS=(
  "StarVLA_Pretrained|${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE1_V2|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE1_V2/checkpoints/steps_8000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE2_V2|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE2_V2/checkpoints/steps_25000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE3_V2|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE3_V2/checkpoints/steps_12000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE4_V2|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE4_V2/checkpoints/steps_15000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE5_V2|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE5_V2/checkpoints/steps_20000_pytorch_model.pt"
)

# Stage 6 may early-stop at fewer steps
STAGE6_CKPT="$(ls -t "${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_FULL_V2/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true)"
if [[ -n "${STAGE6_CKPT}" ]]; then
  CHECKPOINTS+=("VLOG_VLA_LIBERO_FULL_V2|${STAGE6_CKPT}")
fi

job_is_done() { grep -q "Total success rate:" "$1" 2>/dev/null; }

: > "${JOB_FILE}"
port="${BASE_PORT}"
for entry in "${CHECKPOINTS[@]}"; do
  run_id="${entry%%|*}"
  ckpt="${entry#*|}"
  [[ -f "${ckpt}" ]] || { echo "[skip] missing ${ckpt}" >&2; continue; }
  for suite in libero_spatial libero_object libero_goal libero_10; do
    eval_dir="${OUTPUT_ROOT}/${run_id}/${suite}"
    eval_log="${eval_dir}/eval.log"
    if job_is_done "${eval_log}"; then
      echo "[skip] done ${run_id}/${suite}" >&2
      continue
    fi
    printf '%s|%s|%s|%s|%s\n' "${run_id}" "${suite}" "${ckpt}" "${port}" "${eval_dir}" >> "${JOB_FILE}"
    port=$((port + 1))
  done
done

total_jobs=$(wc -l < "${JOB_FILE}" | tr -d ' ')
echo "[$(date -Is)] V2 eval jobs=${total_jobs}, PARALLEL_JOBS=${PARALLEL_JOBS}" | tee "${OUTPUT_ROOT}/parallel.log"

if [[ "${total_jobs}" != "0" ]]; then
  xargs -P "${PARALLEL_JOBS}" -I {} bash -c '
    IFS="|" read -r run_id suite ckpt port eval_dir <<< "{}"
    OUTPUT_ROOT="'"${OUTPUT_ROOT}"'" MANIFEST="'"${MANIFEST}"'" GPU_ID="'"${GPU_ID}"'" \
      exec bash "'"${WORKER}"'" "$run_id" "$suite" "$ckpt" "$port" "$eval_dir"
  ' < "${JOB_FILE}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla
python3 "${REPO_ROOT}/scripts/vlog_vla/collect_libero_success_table.py" \
  --manifest "${MANIFEST}" --output-dir "${OUTPUT_ROOT}"
python3 "${REPO_ROOT}/scripts/vlog_vla/collect_all_stage_results.py" \
  --output-dir "${REPO_ROOT}/outputs/libero_all_results" --tag all_stages_comparison
python3 "${REPO_ROOT}/scripts/vlog_vla/plot_libero_success.py" \
  --input "${OUTPUT_ROOT}/libero_success_table.json" \
  --output-dir "${OUTPUT_ROOT}/figures"

echo "V2 eval done -> ${OUTPUT_ROOT}"
