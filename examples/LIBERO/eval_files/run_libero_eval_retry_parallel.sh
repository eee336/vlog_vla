#!/usr/bin/env bash
# Parallel eval for all 4 RETRY Stage-6 checkpoints (16 jobs = 4 models x 4 suites).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_stage56_eval}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifest.jsonl}"
BASE_PORT="${BASE_PORT:-6400}"
GPU_ID="${GPU_ID:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
JOB_FILE="${OUTPUT_ROOT}/retry_job_queue.txt"
WORKER="${SCRIPT_DIR}/run_one_libero_eval_job.sh"
BEST_SCORE_FILE="${REPO_ROOT}/outputs/vlog_stage56_retry_loop/best_score.txt"
BEST_META="${REPO_ROOT}/outputs/vlog_stage56_retry_loop/best_model.json"

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

declare -a CHECKPOINTS=(
  "VLOG_VLA_LIBERO_FULL_RETRY1|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_FULL_RETRY1/checkpoints/steps_7199_pytorch_model.pt"
  "VLOG_VLA_LIBERO_FULL_RETRY3|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_FULL_RETRY3/checkpoints/steps_7199_pytorch_model.pt"
  "VLOG_VLA_LIBERO_FULL_RETRY4|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_FULL_RETRY4/checkpoints/steps_7199_pytorch_model.pt"
  "VLOG_VLA_LIBERO_FULL_RETRY7|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_FULL_RETRY7/checkpoints/steps_7199_pytorch_model.pt"
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

export REPO_ROOT OUTPUT_ROOT MANIFEST GPU_ID

build_job_queue
total_jobs=$(wc -l < "${JOB_FILE}" | tr -d ' ')
echo "[$(date -Is)] RETRY parallel eval: ${total_jobs} jobs, PARALLEL_JOBS=${PARALLEL_JOBS}" | tee "${OUTPUT_ROOT}/parallel.log"

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

# Pick best model by 4-suite average
python3 - <<PY
import re, glob, json, os
root="${OUTPUT_ROOT}"
runs = [
  "VLOG_VLA_LIBERO_FULL_RETRY1",
  "VLOG_VLA_LIBERO_FULL_RETRY3",
  "VLOG_VLA_LIBERO_FULL_RETRY4",
  "VLOG_VLA_LIBERO_FULL_RETRY7",
]
best_avg, best_meta = 0.0, None
rows = []
for run_id in runs:
    scores = {}
    for s in ["libero_spatial","libero_object","libero_goal","libero_10"]:
        for ep in glob.glob(f"{root}/{run_id}/{s}/eval.log"):
            m = re.search(r"Total success rate:\s*([0-9.]+)", open(ep).read())
            if m:
                scores[s] = float(m.group(1)) * 100
    if len(scores) == 4:
        avg = sum(scores.values()) / 4
        rows.append({"run_id": run_id, "scores": scores, "avg": avg})
        if avg > best_avg:
            best_avg = avg
            ckpt = f"${REPO_ROOT}/results/Checkpoints/{run_id}/checkpoints/steps_7199_pytorch_model.pt"
            best_meta = {
                "run_id": run_id,
                "avg_percent": avg,
                "scores": scores,
                "stage6_ckpt": ckpt,
                "eval_dir": f"{root}/{run_id}",
            }

open(f"{root}/retry_summary.json", "w").write(json.dumps(rows, indent=2))
if best_meta:
    open("${BEST_SCORE_FILE}", "w").write(str(best_avg))
    json.dump(best_meta, open("${BEST_META}", "w"), indent=2)
    print(f"BEST: {best_meta['run_id']} avg={best_avg:.2f}%")
else:
    print("No complete eval results yet")
PY

python3 "${REPO_ROOT}/scripts/vlog_vla/collect_all_stage_results.py" \
  --output-dir "${REPO_ROOT}/outputs/libero_all_results" --tag stage56_retry_final 2>/dev/null || true

echo "[$(date -Is)] RETRY eval done -> ${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/parallel.log"
