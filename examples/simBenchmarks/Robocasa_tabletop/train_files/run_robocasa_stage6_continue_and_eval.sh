#!/usr/bin/env bash
# Continue from an already-healthy Stage5 checkpoint -> Stage6 -> eval (+ baseline).
# Does NOT retrain Stage5. Avoids stdout pollution bugs when capturing ckpt paths.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_train.sh"
EVAL_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh"
LOG_DIR="${REPO_ROOT}/outputs/vlog_robocasa_stage56_retry"
EVAL_ROOT="${REPO_ROOT}/outputs/robocasa_stage56_eval"
mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

# Prefer RETRY2 (lower final action loss); fall back to RETRY1.
S5_CKPT="${S5_CKPT:-}"
S5_RUN="${S5_RUN:-}"
if [[ -z "${S5_CKPT}" ]]; then
  if [[ -f "${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE5_RETRY2/checkpoints/steps_12000_pytorch_model.pt" ]]; then
    S5_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE5_RETRY2/checkpoints/steps_12000_pytorch_model.pt"
    S5_RUN="VLOG_VLA_ROBOCASA_STAGE5_RETRY2"
    S6_RUN="VLOG_VLA_ROBOCASA_FULL_RETRY2"
    LR_VLOG="2e-5"; LR_ACTION="2e-5"; LAMBDA_ROUTER="0.2"; LAMBDA_CRITIC="0.1"; GRAD_CLIP="1.0"
    S6_STEPS=20000
    EXTRA_CQL="--framework.vlog.losses.alpha_cql 0.05"
  elif [[ -f "${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE5_RETRY1/checkpoints/steps_15000_pytorch_model.pt" ]]; then
    S5_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE5_RETRY1/checkpoints/steps_15000_pytorch_model.pt"
    S5_RUN="VLOG_VLA_ROBOCASA_STAGE5_RETRY1"
    S6_RUN="VLOG_VLA_ROBOCASA_FULL_RETRY1"
    LR_VLOG="1e-5"; LR_ACTION="1e-5"; LAMBDA_ROUTER="0.15"; LAMBDA_CRITIC="0.15"; GRAD_CLIP="0.5"
    S6_STEPS=25000
    EXTRA_CQL=""
  else
    echo "No healthy Stage5 checkpoint found" >&2
    exit 1
  fi
fi

S6_RUN="${S6_RUN:-VLOG_VLA_ROBOCASA_FULL_CONTINUE}"
S6_STEPS="${S6_STEPS:-20000}"
LR_VLOG="${LR_VLOG:-2e-5}"
LR_ACTION="${LR_ACTION:-2e-5}"
LAMBDA_ROUTER="${LAMBDA_ROUTER:-0.2}"
LAMBDA_CRITIC="${LAMBDA_CRITIC:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EXTRA_CQL="${EXTRA_CQL:-}"

check_train_ok() {
  local log="$1" min_finite="${2:-400}"
  python3 - "$log" "$min_finite" <<'PY' >&2
import json, math, sys
path, min_finite = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(path) if l.strip()]
def ok(r):
    al = r.get("vlog/action_loss")
    if not isinstance(al, float) or not math.isfinite(al) or al <= 0:
        return False
    for k in ("vlog/vq_loss", "vlog/mean_beta", "vlog/adapter_alpha", "vlog/critic_td_loss"):
        v = r.get(k)
        if isinstance(v, float) and not math.isfinite(v):
            return False
    return True
finite = [r for r in rows if ok(r)]
ratio = len(finite) / max(1, len(rows))
last_ok = sum(1 for r in rows[-100:] if ok(r))
print(f"finite_ratio={ratio:.3f} finite={len(finite)}/{len(rows)} last100_ok={last_ok}")
sys.exit(0 if len(finite) >= min_finite and ratio >= 0.90 and last_ok >= 80 else 1)
PY
}

{
  echo "======== CONTINUE Stage6 ========"
  echo "[$(date -Is)] S5_RUN=${S5_RUN}"
  echo "[$(date -Is)] S5_CKPT=${S5_CKPT}"
  echo "[$(date -Is)] S6_RUN=${S6_RUN} steps=${S6_STEPS}"
} | tee -a "${LOG_DIR}/loop.log"

# Validate Stage5 health first
S5_LOG="${REPO_ROOT}/results/Checkpoints/${S5_RUN}/logs/train_log.jsonl"
check_train_ok "${S5_LOG}" 800

cd "${REPO_ROOT}"
: > "${LOG_DIR}/${S6_RUN}.log"

echo "[$(date -Is)] TRAIN ${S6_RUN} stage=stage6_full steps=${S6_STEPS}" | tee -a "${LOG_DIR}/loop.log"

USE_DEEPSPEED=0 \
VIDEO_BACKEND=pyav \
EVAL_INTERVAL=100000000 \
CONFIG_YAML=examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_oft_vlog_robocasa.yaml \
BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct" \
DATA_ROOT="${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim" \
RUN_ROOT_DIR=results/Checkpoints \
PER_DEVICE_BATCH_SIZE=4 \
NUM_PROCESSES=1 \
DATA_MIX=fourier_gr1_unified_1000 \
TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python \
RUN_ID="${S6_RUN}" \
TRAIN_STAGE=stage6_full \
MAX_TRAIN_STEPS="${S6_STEPS}" \
SAVE_INTERVAL=2000 \
BASE_CKPT="${S5_CKPT}" \
EXTRA_TRAIN_ARGS="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true --trainer.learning_rate.vlog ${LR_VLOG} --trainer.learning_rate.action_model ${LR_ACTION} --framework.vlog.losses.lambda_router ${LAMBDA_ROUTER} --framework.vlog.losses.lambda_critic ${LAMBDA_CRITIC} --trainer.gradient_clipping ${GRAD_CLIP} --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 5000 --trainer.early_stop_metric vlog/action_loss --trainer.max_consecutive_nan_steps 30 ${EXTRA_CQL}" \
bash "${TRAIN_SH}" > "${LOG_DIR}/${S6_RUN}.log" 2>&1

S6_CKPT="$(ls -t "${REPO_ROOT}/results/Checkpoints/${S6_RUN}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true)"
if [[ -z "${S6_CKPT}" || ! -f "${S6_CKPT}" ]]; then
  echo "[$(date -Is)] ERROR: no Stage6 checkpoint" | tee -a "${LOG_DIR}/loop.log"
  tail -80 "${LOG_DIR}/${S6_RUN}.log" | tee -a "${LOG_DIR}/loop.log"
  exit 1
fi

S6_LOG="${REPO_ROOT}/results/Checkpoints/${S6_RUN}/logs/train_log.jsonl"
cp "${S6_LOG}" "${LOG_DIR}/${S6_RUN}_train_log.jsonl" 2>/dev/null || true
if ! check_train_ok "${S6_LOG}" 400; then
  echo "[$(date -Is)] ERROR: Stage6 unhealthy (NaN)" | tee -a "${LOG_DIR}/loop.log"
  exit 1
fi

echo "[$(date -Is)] Stage6 OK: ${S6_CKPT}" | tee -a "${LOG_DIR}/loop.log"
python3 - <<PY
import json
json.dump({
  "source_stage5": "${S5_RUN}",
  "stage5_ckpt": "${S5_CKPT}",
  "stage6_ckpt": "${S6_CKPT}",
  "run_id": "${S6_RUN}",
}, open("${LOG_DIR}/best_model.json", "w"), indent=2)
PY

echo "[$(date -Is)] Eval our model ${S6_RUN}" | tee -a "${LOG_DIR}/loop.log"
CKPT="${S6_CKPT}" \
RUN_ID="${S6_RUN}" \
OUTPUT_ROOT="${EVAL_ROOT}" \
N_EPISODES="${N_EPISODES:-20}" \
N_PARALLEL="${N_PARALLEL:-6}" \
N_ENVS="${N_ENVS:-1}" \
SKIP_DONE="${SKIP_DONE:-1}" \
BASE_PORT="${BASE_PORT:-5800}" \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/${S6_RUN}_eval.log"

echo "[$(date -Is)] Eval baseline OFT" | tee -a "${LOG_DIR}/loop.log"
CKPT="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt" \
RUN_ID="StarVLA_OFT_Robocasa_Pretrained" \
OUTPUT_ROOT="${EVAL_ROOT}" \
N_EPISODES="${N_EPISODES:-20}" \
N_PARALLEL="${N_PARALLEL:-6}" \
N_ENVS="${N_ENVS:-1}" \
SKIP_DONE="${SKIP_DONE:-1}" \
EXTRA_EVAL_ARGS="--args.no_send_state" \
BASE_PORT=5900 \
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
        m = re.search(r"Success rate:\s*([0-9.]+)", log.read_text(errors="ignore"))
        if m:
            scores[log.parent.name] = float(m.group(1)) * 100
    if scores:
        avg = sum(scores.values()) / len(scores)
        summary[run_dir.name] = {"avg": avg, "n": len(scores), "scores": scores}
        print(f"{run_dir.name}: avg={avg:.2f}% over {len(scores)} tasks")
(root / "summary.json").write_text(json.dumps(summary, indent=2))
Path("${LOG_DIR}/eval_summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote", root / "summary.json")
PY

echo "[$(date -Is)] All done." | tee -a "${LOG_DIR}/loop.log"
cat "${LOG_DIR}/best_model.json" | tee -a "${LOG_DIR}/loop.log"
