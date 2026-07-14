#!/usr/bin/env bash
# Full V2 workflow: record V1 results -> train V2 pipeline -> eval -> record all.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/vlog_v2_full_workflow"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

echo "[$(date -Is)] === Record V1 results ===" | tee "${LOG_DIR}/workflow.log"
python3 "${REPO_ROOT}/scripts/vlog_vla/collect_all_stage_results.py" \
  --output-dir "${REPO_ROOT}/outputs/libero_all_results" \
  --tag v1_baseline

echo "[$(date -Is)] === Train V2 pipeline ===" | tee -a "${LOG_DIR}/workflow.log"
bash "${REPO_ROOT}/examples/LIBERO/train_files/run_vlog_libero_stage_pipeline_v2.sh" \
  2>&1 | tee "${LOG_DIR}/train_v2.log"

echo "[$(date -Is)] === Eval V2 checkpoints ===" | tee -a "${LOG_DIR}/workflow.log"
bash "${REPO_ROOT}/examples/LIBERO/eval_files/run_libero_eval_all_v2.sh" \
  2>&1 | tee "${LOG_DIR}/eval_v2.log"

echo "[$(date -Is)] === Final comparison ===" | tee -a "${LOG_DIR}/workflow.log"
python3 "${REPO_ROOT}/scripts/vlog_vla/collect_all_stage_results.py" \
  --output-dir "${REPO_ROOT}/outputs/libero_all_results" \
  --tag all_stages_final

# Write adjustment rationale
cat > "${REPO_ROOT}/outputs/libero_all_results/v2_training_rationale.md" <<'EOF'
# V2 Training Adjustments (based on V1 eval)

| Stage | V1 Steps | V2 Steps | Change Rationale |
|-------|----------|----------|------------------|
| 1 preserve | 20000 | **8000** | preserve_loss→0 quickly; 20k overkill |
| 2 option | 60000 | **25000** | VQ/distill plateau; 60k slow with marginal gain |
| 3 graph | 20000 | **12000** | modest eval impact on spatial/goal |
| 4 critic | 20000 | **15000** | critic improved goal to 99%; keep substantial training |
| 5 router | 20000 | **20000** | best V1 overall (LIBERO-10 97.0%); unchanged |
| 6 full | 6599 (early stop) | **≤30000** | early-stop on `vlog/action_loss`; vlog lr 1e-4 |

## Other V2 changes
- `PER_DEVICE_BATCH_SIZE=12`, `num_workers=8` (faster training)
- Stage 6 early-stop metric: `vlog/action_loss` (not total loss dominated by router_adv)
- Numerical stability: option_adapter tanh + skip NaN steps (from V1 fixes)
EOF

echo "[$(date -Is)] Workflow complete." | tee -a "${LOG_DIR}/workflow.log"
echo "Results: ${REPO_ROOT}/outputs/libero_all_results/"
