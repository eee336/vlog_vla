#!/usr/bin/env bash
# Master loop: wait for batch1, run batch2+, copy best model, until target met.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/vlog_stage56_retry_loop"
BEST_SCORE_FILE="${LOG_DIR}/best_score.txt"
BEST_META="${LOG_DIR}/best_model.json"
TARGET_SCORE="${TARGET_SCORE:-98.45}"
CANONICAL_DIR="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_BEST_FINAL"

mkdir -p "${LOG_DIR}"

wait_for_batch1() {
  echo "[$(date -Is)] Waiting for batch1 (run_stage56_retry_loop.sh) to finish..."
  while pgrep -f "run_stage56_retry_loop.sh" >/dev/null 2>&1; do
    sleep 120
    if [[ -f "${LOG_DIR}/loop.log" ]]; then
      tail -1 "${LOG_DIR}/loop.log"
    fi
  done
  echo "[$(date -Is)] Batch1 finished."
}

copy_best_model() {
  if [[ ! -f "${BEST_META}" ]]; then
    echo "No best model found."
    return 1
  fi
  local ckpt
  ckpt=$(python3 -c "import json; print(json.load(open('${BEST_META}'))['stage6_ckpt'])")
  mkdir -p "${CANONICAL_DIR}/checkpoints"
  cp "${ckpt}" "${CANONICAL_DIR}/checkpoints/best_pytorch_model.pt"
  cp "${BEST_META}" "${CANONICAL_DIR}/best_model.json"
  echo "[$(date -Is)] Copied best ckpt to ${CANONICAL_DIR}/checkpoints/best_pytorch_model.pt"
}

finalize() {
  python3 "${REPO_ROOT}/scripts/vlog_vla/collect_all_stage_results.py" \
    --output-dir "${REPO_ROOT}/outputs/libero_all_results" --tag stage56_retry_final 2>/dev/null || true
  copy_best_model || true
  {
    echo "# Stage 5-6 Retry Loop Final"
    echo ""
    echo "Target: ${TARGET_SCORE}%"
    echo "Best score: $(cat "${BEST_SCORE_FILE}" 2>/dev/null || echo 0)%"
    echo ""
    cat "${BEST_META}" 2>/dev/null || echo "No successful model."
  } > "${REPO_ROOT}/outputs/libero_all_results/stage56_retry_final.md"
}

# If batch1 not running and not done, start it
if ! pgrep -f "run_stage56_retry_loop.sh" >/dev/null 2>&1; then
  if ! grep -q "Loop done" "${LOG_DIR}/loop.log" 2>/dev/null; then
    echo "[$(date -Is)] Starting batch1..."
    nohup bash "${REPO_ROOT}/examples/LIBERO/train_files/run_stage56_retry_loop.sh" \
      >> "${LOG_DIR}/nohup.log" 2>&1 &
  fi
fi

wait_for_batch1

BEST=$(cat "${BEST_SCORE_FILE}" 2>/dev/null || echo 0)
python3 -c "import sys; sys.exit(0 if float('${BEST}') >= float('${TARGET_SCORE}') else 1)" && {
  echo "[$(date -Is)] Target ${TARGET_SCORE}% met (best=${BEST}%). Done."
  finalize
  exit 0
}

echo "[$(date -Is)] Best ${BEST}% < target ${TARGET_SCORE}%. Running batch2..."
bash "${REPO_ROOT}/examples/LIBERO/train_files/run_stage56_retry_loop_batch2.sh"

BEST=$(cat "${BEST_SCORE_FILE}" 2>/dev/null || echo 0)
echo "[$(date -Is)] After batch2: best=${BEST}%"
finalize
