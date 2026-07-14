#!/usr/bin/env bash
# RoboCasa VLOG retrain V2:
#   - resume healthy Stage2 (skip destroyed Stage3-6)
#   - Stage3 with codebook-detach + no balance (graph only)
#   - larger batch to saturate 98GB GPU
#   - include_state=false (aligned with OFT base)
#   - Stage5/6 healthy LRs; Stage6 saves mid-ckpts, no action_loss early-stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAIN_SH="${SCRIPT_DIR}/run_vlog_robocasa_train.sh"
EVAL_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh"
LOG_ROOT="${REPO_ROOT}/outputs/vlog_robocasa_retrain_v2"
EVAL_ROOT="${REPO_ROOT}/outputs/robocasa_retrain_v2_eval"
mkdir -p "${LOG_ROOT}" "${EVAL_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

STAGE2_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE2_REAL/checkpoints/steps_25000_pytorch_model.pt"
[[ -f "${STAGE2_CKPT}" ]] || { echo "Missing Stage2: ${STAGE2_CKPT}" >&2; exit 1; }

# Max-throughput defaults for single RTX PRO 6000 98GB.
BATCH="${PER_DEVICE_BATCH_SIZE:-16}"
COMMON=(
  USE_DEEPSPEED=0
  VIDEO_BACKEND=pyav
  EVAL_INTERVAL=100000000
  CONFIG_YAML=examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_oft_vlog_robocasa.yaml
  BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  DATA_ROOT="${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
  RUN_ROOT_DIR=results/Checkpoints
  PER_DEVICE_BATCH_SIZE="${BATCH}"
  NUM_PROCESSES=1
  DATA_MIX=fourier_gr1_unified_1000
  TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python
)

EXTRA_BASE="--datasets.vla_data.include_state false --datasets.vla_data.num_workers 12 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true --trainer.gradient_checkpointing false"

pick_ckpt() {
  local run_id="$1"
  ls -t "${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true
}

check_ok() {
  local log="$1" min_finite="${2:-300}"
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
    alpha = r.get("vlog/adapter_alpha")
    if isinstance(alpha, float) and alpha < -1e-6:
        return False
    return True
finite = [r for r in rows if ok(r)]
ratio = len(finite) / max(1, len(rows))
last_ok = sum(1 for r in rows[-min(100,len(rows)):] if ok(r))
print(f"finite_ratio={ratio:.3f} finite={len(finite)}/{len(rows)} last_ok={last_ok}")
# Also require Stage2-style structure not destroyed: entropy should not stay flat-max for Stage3+ if vq stays exploded
sys.exit(0 if len(finite) >= min_finite and ratio >= 0.90 else 1)
PY
}

run_stage() {
  local stage_name="$1" run_id="$2" train_stage="$3" max_steps="$4" base_ckpt="$5" save_int="$6" extra="$7"
  local log_file="${LOG_ROOT}/${run_id}.log"
  # IMPORTANT: status goes to stderr / loop.log; stdout is ckpt path only.
  {
    echo "================================================================"
    echo "[$(date -Is)] START ${stage_name}: ${run_id} steps=${max_steps} batch=${BATCH}"
    echo "  base=${base_ckpt}"
  } | tee -a "${LOG_ROOT}/loop.log" >&2
  cd "${REPO_ROOT}"
  : > "${log_file}"
  env "${COMMON[@]}" \
    RUN_ID="${run_id}" \
    TRAIN_STAGE="${train_stage}" \
    MAX_TRAIN_STEPS="${max_steps}" \
    SAVE_INTERVAL="${save_int}" \
    BASE_CKPT="${base_ckpt}" \
    EXTRA_TRAIN_ARGS="${EXTRA_BASE} ${extra}" \
    bash "${TRAIN_SH}" >"${log_file}" 2>&1 || true

  local ckpt
  ckpt="$(pick_ckpt "${run_id}")"
  local train_log="${REPO_ROOT}/results/Checkpoints/${run_id}/logs/train_log.jsonl"
  if [[ -f "${train_log}" ]]; then
    cp "${train_log}" "${LOG_ROOT}/${run_id}_train_log.jsonl"
  fi
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]] || ! check_ok "${train_log}" 200; then
    echo "[$(date -Is)] FAIL ${stage_name}" | tee -a "${LOG_ROOT}/loop.log" >&2
    tail -80 "${log_file}" | tee -a "${LOG_ROOT}/loop.log" >&2
    exit 1
  fi
  # Extra structure checks for Stage3+
  python3 - "${train_log}" "${stage_name}" <<'PY' >&2
import json, sys, math
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
r=rows[-1]
ent=r.get("vlog/option_entropy")
vq=r.get("vlog/vq_loss")
alpha=r.get("vlog/adapter_alpha")
print(f"{sys.argv[2]} last: entropy={ent} vq={vq} alpha={alpha} action={r.get('vlog/action_loss')}")
# Stage3 must not explode vq again
if "Stage3" in sys.argv[2] or "stage3" in sys.argv[2].lower():
    if isinstance(vq, float) and vq > 50:
        print("Stage3 vq still exploded"); sys.exit(1)
    if isinstance(ent, float) and ent > 2.7 and isinstance(vq, float) and vq > 20:
        print("Stage3 option space collapsed to uniform+bad vq"); sys.exit(1)
sys.exit(0)
PY
  echo "[$(date -Is)] OK ${stage_name}: ${ckpt}" | tee -a "${LOG_ROOT}/loop.log" >&2
  printf '%s' "${ckpt}"
}

{
  echo "======== RoboCasa VLOG Retrain V2 ========"
  echo "[$(date -Is)] STAGE2_CKPT=${STAGE2_CKPT}"
  echo "[$(date -Is)] BATCH=${BATCH}"
} | tee "${LOG_ROOT}/loop.log"

# Probe batch fit: if OOM later user can lower. Optional 1-step smoke skipped for time.

S3_CKPT="$(run_stage "Stage3_graph_safe" "VLOG_VLA_ROBOCASA_V2_STAGE3" "stage3_graph" 6000 "${STAGE2_CKPT}" 2000 \
  "--framework.vlog.losses.lambda_balance 0.0 --framework.vlog.losses.lambda_sparse 0.005 --trainer.learning_rate.vlog 5.0e-05 --trainer.learning_rate.action_model 1.0e-05")"

# NOTE: do NOT pass dotted nested LR keys like learning_rate.vlog.critic —
# OmegaConf AccessTrackedConfig breaks LambdaLR (TypeError: * float).
S4_CKPT="$(run_stage "Stage4_critic" "VLOG_VLA_ROBOCASA_V2_STAGE4" "stage4_critic" 12000 "${S3_CKPT}" 4000 \
  "--trainer.learning_rate.vlog 1.0e-04 --trainer.learning_rate.action_model 1.0e-05")"

S5_CKPT="$(run_stage "Stage5_router" "VLOG_VLA_ROBOCASA_V2_STAGE5" "stage5_router" 15000 "${S4_CKPT}" 3000 \
  "--trainer.learning_rate.vlog 5.0e-05 --trainer.learning_rate.action_model 2.0e-05 --framework.vlog.losses.lambda_router 0.3 --framework.vlog.losses.lambda_critic 0.3 --trainer.gradient_clipping 1.0 --trainer.max_consecutive_nan_steps 30")"

S6_CKPT="$(run_stage "Stage6_full" "VLOG_VLA_ROBOCASA_V2_FULL" "stage6_full" 25000 "${S5_CKPT}" 2500 \
  "--trainer.learning_rate.vlog 5.0e-05 --trainer.learning_rate.action_model 2.0e-05 --framework.vlog.losses.lambda_router 0.3 --framework.vlog.losses.lambda_term 0.1 --trainer.gradient_clipping 1.0 --trainer.max_consecutive_nan_steps 30")"

python3 - <<PY
import json
json.dump({
  "stage2": "${STAGE2_CKPT}",
  "stage3": "${S3_CKPT}",
  "stage4": "${S4_CKPT}",
  "stage5": "${S5_CKPT}",
  "stage6": "${S6_CKPT}",
  "batch": ${BATCH},
}, open("${LOG_ROOT}/best_model.json","w"), indent=2)
print(open("${LOG_ROOT}/best_model.json").read())
PY

echo "[$(date -Is)] Eval V2 FULL (no_send_state)" | tee -a "${LOG_ROOT}/loop.log"
CKPT="${S6_CKPT}" \
RUN_ID="VLOG_VLA_ROBOCASA_V2_FULL" \
OUTPUT_ROOT="${EVAL_ROOT}" \
N_EPISODES="${N_EPISODES:-20}" \
N_PARALLEL=6 N_ENVS=1 FILL_WORKERS=1 \
EXTRA_EVAL_ARGS="--args.no_send_state" \
BASE_PORT=6000 \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_ROOT}/eval_v2.log"

echo "[$(date -Is)] Eval baseline OFT" | tee -a "${LOG_ROOT}/loop.log"
CKPT="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt" \
RUN_ID="StarVLA_OFT_Robocasa_Pretrained" \
OUTPUT_ROOT="${EVAL_ROOT}" \
N_EPISODES="${N_EPISODES:-20}" \
N_PARALLEL=6 N_ENVS=1 FILL_WORKERS=1 \
EXTRA_EVAL_ARGS="--args.no_send_state" \
BASE_PORT=6100 \
SKIP_DONE=1 \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_ROOT}/eval_baseline.log"

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
        print(f"{run_dir.name}: avg={avg:.2f}%")
(root / "summary.json").write_text(json.dumps(summary, indent=2))
Path("${LOG_ROOT}/eval_summary.json").write_text(json.dumps(summary, indent=2))
PY

echo "[$(date -Is)] Retrain V2 ALL DONE" | tee -a "${LOG_ROOT}/loop.log"
