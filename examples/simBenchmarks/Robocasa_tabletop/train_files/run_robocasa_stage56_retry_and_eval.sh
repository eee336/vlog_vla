#!/usr/bin/env bash
# RoboCasa Stage5->6 NaN-safe retry from Stage4, then evaluate the best complete model.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_train.sh"
EVAL_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh"
LOG_DIR="${REPO_ROOT}/outputs/vlog_robocasa_stage56_retry"
EVAL_ROOT="${REPO_ROOT}/outputs/robocasa_stage56_eval"
mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

STAGE4_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE4_REAL/checkpoints/steps_15000_pytorch_model.pt"
[[ -f "${STAGE4_CKPT}" ]] || { echo "Missing Stage4 ckpt: ${STAGE4_CKPT}" >&2; exit 1; }

BEST_SCORE_FILE="${LOG_DIR}/best_score.txt"
BEST_META="${LOG_DIR}/best_model.json"
echo "-1" > "${BEST_SCORE_FILE}"

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

# attempt_id|s5_run|s5_steps|s6_run|s6_steps|lr_vlog|lr_action|lambda_router|lambda_critic|grad_clip|s5_extra
ATTEMPTS=(
  "1|VLOG_VLA_ROBOCASA_STAGE5_RETRY1|15000|VLOG_VLA_ROBOCASA_FULL_RETRY1|25000|1e-5|1e-5|0.15|0.15|0.5|--trainer.max_consecutive_nan_steps 30"
  "2|VLOG_VLA_ROBOCASA_STAGE5_RETRY2|12000|VLOG_VLA_ROBOCASA_FULL_RETRY2|20000|2e-5|2e-5|0.2|0.1|1.0|--trainer.max_consecutive_nan_steps 30 --framework.vlog.losses.alpha_cql 0.05"
  "3|VLOG_VLA_ROBOCASA_STAGE5_RETRY3|10000|VLOG_VLA_ROBOCASA_FULL_RETRY3|20000|5e-6|1e-5|0.1|0.1|0.5|--trainer.max_consecutive_nan_steps 20 --framework.vlog.losses.alpha_cql 0.02"
  "4|VLOG_VLA_ROBOCASA_STAGE5_RETRY4|15000|VLOG_VLA_ROBOCASA_FULL_RETRY4|25000|2e-5|5e-5|0.2|0.2|1.0|--trainer.max_consecutive_nan_steps 50"
)

pick_ckpt() {
  local run_id="$1"
  ls -t "${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true
}

check_train_ok() {
  local log="$1"
  local min_finite="${2:-500}"
  # diagnostics go to stderr so callers can safely capture exit code only
  python3 - "$log" "$min_finite" <<'PY' >&2
import json, math, sys
path, min_finite = sys.argv[1], int(sys.argv[2])
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except FileNotFoundError:
    print("missing log"); sys.exit(1)
if not rows:
    print("empty log"); sys.exit(1)

def ok(r):
    al = r.get("vlog/action_loss")
    if not isinstance(al, float) or not math.isfinite(al) or al <= 0:
        return False
    # reject runs that already went into zero/nan regime
    for k in ("vlog/vq_loss", "vlog/mean_beta", "vlog/adapter_alpha", "vlog/critic_td_loss"):
        v = r.get(k)
        if isinstance(v, float) and not math.isfinite(v):
            return False
    return True

finite = [r for r in rows if ok(r)]
ratio = len(finite) / max(1, len(rows))
last100 = rows[-100:]
last_ok = sum(1 for r in last100 if ok(r))
print(f"finite_ratio={ratio:.3f} finite={len(finite)}/{len(rows)} last100_ok={last_ok}")
# Require overall health and that the end of training is still healthy.
sys.exit(0 if len(finite) >= min_finite and ratio >= 0.90 and last_ok >= 80 else 1)
PY
}

train_stage() {
  local run_id="$1" stage="$2" max_steps="$3" base_ckpt="$4" save_int="$5" extra="$6"
  # IMPORTANT: only the checkpoint path may go to stdout (callers capture it).
  echo "[$(date -Is)] TRAIN ${run_id} stage=${stage} steps=${max_steps}" >> "${LOG_DIR}/loop.log"
  cd "${REPO_ROOT}"
  : > "${LOG_DIR}/${run_id}.log"
  local extra_args="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true ${extra}"
  env "${COMMON[@]}" \
    RUN_ID="${run_id}" \
    TRAIN_STAGE="${stage}" \
    MAX_TRAIN_STEPS="${max_steps}" \
    SAVE_INTERVAL="${save_int}" \
    BASE_CKPT="${base_ckpt}" \
    EXTRA_TRAIN_ARGS="${extra_args}" \
    bash "${TRAIN_SH}" > "${LOG_DIR}/${run_id}.log" 2>&1 || true
  if [[ -f "${REPO_ROOT}/results/Checkpoints/${run_id}/logs/train_log.jsonl" ]]; then
    cp "${REPO_ROOT}/results/Checkpoints/${run_id}/logs/train_log.jsonl" "${LOG_DIR}/${run_id}_train_log.jsonl"
  fi
  local ckpt
  ckpt="$(pick_ckpt "${run_id}")"
  echo "[$(date -Is)] TRAIN done ${run_id} ckpt=${ckpt:-NONE}" >> "${LOG_DIR}/loop.log"
  # stdout: ckpt path only
  printf '%s' "${ckpt}"
}

SUCCESS_CKPT=""
SUCCESS_RUN=""

for entry in "${ATTEMPTS[@]}"; do
  IFS='|' read -r aid s5_run s5_steps s6_run s6_steps lr_vlog lr_action lambda_router lambda_critic grad_clip s5_extra <<< "${entry}"
  echo "======== ATTEMPT ${aid} ========" | tee -a "${LOG_DIR}/loop.log"

  S5_EXTRA="--trainer.learning_rate.vlog ${lr_vlog} --trainer.learning_rate.action_model ${lr_action} --framework.vlog.losses.lambda_router ${lambda_router} --framework.vlog.losses.lambda_critic ${lambda_critic} --trainer.gradient_clipping ${grad_clip} ${s5_extra}"
  S5_LOG="${REPO_ROOT}/results/Checkpoints/${s5_run}/logs/train_log.jsonl"
  S5_CKPT="$(pick_ckpt "${s5_run}")"
  if [[ -n "${S5_CKPT}" && -f "${S5_CKPT}" ]] && check_train_ok "${S5_LOG}" 800; then
    echo "[$(date -Is)] ATTEMPT ${aid} reusing healthy Stage5: ${S5_CKPT}" | tee -a "${LOG_DIR}/loop.log"
  else
    S5_CKPT="$(train_stage "${s5_run}" "stage5_router" "${s5_steps}" "${STAGE4_CKPT}" "${s5_steps}" "${S5_EXTRA}")"
    S5_LOG="${REPO_ROOT}/results/Checkpoints/${s5_run}/logs/train_log.jsonl"
  fi

  if [[ -z "${S5_CKPT}" || ! -f "${S5_CKPT}" ]] || ! check_train_ok "${S5_LOG}" 800; then
    echo "[$(date -Is)] ATTEMPT ${aid} Stage5 FAILED (NaN or weak finite ratio)" | tee -a "${LOG_DIR}/loop.log"
    continue
  fi
  echo "[$(date -Is)] Stage5 OK: ${S5_CKPT}" | tee -a "${LOG_DIR}/loop.log"

  S6_EXTRA="--trainer.learning_rate.vlog ${lr_vlog} --trainer.learning_rate.action_model ${lr_action} --framework.vlog.losses.lambda_router ${lambda_router} --framework.vlog.losses.lambda_critic ${lambda_critic} --trainer.gradient_clipping ${grad_clip} --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 5000 --trainer.early_stop_metric vlog/action_loss --trainer.max_consecutive_nan_steps 30 ${s5_extra}"
  S6_CKPT="$(train_stage "${s6_run}" "stage6_full" "${s6_steps}" "${S5_CKPT}" 2000 "${S6_EXTRA}")"
  S6_LOG="${REPO_ROOT}/results/Checkpoints/${s6_run}/logs/train_log.jsonl"

  if [[ -z "${S6_CKPT}" || ! -f "${S6_CKPT}" ]] || ! check_train_ok "${S6_LOG}" 400; then
    echo "[$(date -Is)] ATTEMPT ${aid} Stage6 FAILED" | tee -a "${LOG_DIR}/loop.log"
    continue
  fi

  echo "[$(date -Is)] ATTEMPT ${aid} SUCCESS: ${S6_CKPT}" | tee -a "${LOG_DIR}/loop.log"
  SUCCESS_CKPT="${S6_CKPT}"
  SUCCESS_RUN="${s6_run}"
  python3 - <<PY
import json
json.dump({
  "attempt": int("${aid}"),
  "stage5_ckpt": "${S5_CKPT}",
  "stage6_ckpt": "${S6_CKPT}",
  "run_id": "${s6_run}",
}, open("${BEST_META}", "w"), indent=2)
PY
  break
done

if [[ -z "${SUCCESS_CKPT}" ]]; then
  echo "[$(date -Is)] All Stage5/6 attempts failed. Falling back to Stage4 for eval." | tee -a "${LOG_DIR}/loop.log"
  SUCCESS_CKPT="${STAGE4_CKPT}"
  SUCCESS_RUN="VLOG_VLA_ROBOCASA_STAGE4_REAL"
  python3 - <<PY
import json
json.dump({
  "attempt": "fallback_stage4",
  "stage6_ckpt": "${SUCCESS_CKPT}",
  "run_id": "${SUCCESS_RUN}",
  "note": "Stage5/6 retries all failed; evaluating Stage4",
}, open("${BEST_META}", "w"), indent=2)
PY
fi

echo "[$(date -Is)] Starting RoboCasa eval for ${SUCCESS_RUN}" | tee -a "${LOG_DIR}/loop.log"
CKPT="${SUCCESS_CKPT}" \
RUN_ID="${SUCCESS_RUN}" \
OUTPUT_ROOT="${EVAL_ROOT}" \
N_EPISODES="${N_EPISODES:-20}" \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/${SUCCESS_RUN}_eval.log"

# Also eval pretrained baseline for comparison
BASELINE_CKPT="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt"
echo "[$(date -Is)] Starting baseline eval" | tee -a "${LOG_DIR}/loop.log"
CKPT="${BASELINE_CKPT}" \
RUN_ID="StarVLA_OFT_Robocasa_Pretrained" \
OUTPUT_ROOT="${EVAL_ROOT}" \
N_EPISODES="${N_EPISODES:-20}" \
EXTRA_EVAL_ARGS="--args.no_send_state" \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/baseline_eval.log"

python3 - <<PY
import json, re
from pathlib import Path
root = Path("${EVAL_ROOT}")
summary = {}
for run_dir in sorted(root.iterdir()):
    if not run_dir.is_dir():
        continue
    scores = {}
    for log in run_dir.glob("*/eval.log"):
        text = log.read_text(errors="ignore")
        m = re.search(r"Success rate:\s*([0-9.]+)", text)
        if m:
            scores[log.parent.name] = float(m.group(1)) * 100
    if scores:
        avg = sum(scores.values()) / len(scores)
        summary[run_dir.name] = {"avg": avg, "n": len(scores), "scores": scores}
        print(f"{run_dir.name}: avg={avg:.2f}% over {len(scores)} tasks")
Path("${EVAL_ROOT}/summary.json").write_text(json.dumps(summary, indent=2))
Path("${LOG_DIR}/eval_summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote", "${EVAL_ROOT}/summary.json")
PY

echo "[$(date -Is)] Done. Best model meta:" | tee -a "${LOG_DIR}/loop.log"
cat "${BEST_META}" | tee -a "${LOG_DIR}/loop.log"
