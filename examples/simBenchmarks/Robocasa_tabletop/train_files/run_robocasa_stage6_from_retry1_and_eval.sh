#!/usr/bin/env bash
# Rescue path: Stage5 RETRY1 was healthy but mis-detected; continue Stage6 then eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/train_files/run_vlog_robocasa_train.sh"
EVAL_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh"
LOG_DIR="${REPO_ROOT}/outputs/vlog_robocasa_stage56_retry"
EVAL_ROOT="${REPO_ROOT}/outputs/robocasa_stage56_eval"
mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

S5_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_STAGE5_RETRY1/checkpoints/steps_15000_pytorch_model.pt"
S6_RUN="VLOG_VLA_ROBOCASA_FULL_RETRY1"
[[ -f "${S5_CKPT}" ]] || { echo "Missing healthy Stage5: ${S5_CKPT}" >&2; exit 1; }

cd "${REPO_ROOT}"
echo "[$(date -Is)] Rescue Stage6 from RETRY1 Stage5: ${S5_CKPT}" | tee -a "${LOG_DIR}/loop.log"

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
MAX_TRAIN_STEPS=25000 \
SAVE_INTERVAL=2000 \
BASE_CKPT="${S5_CKPT}" \
EXTRA_TRAIN_ARGS="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true --trainer.learning_rate.vlog 1e-5 --trainer.learning_rate.action_model 1e-5 --framework.vlog.losses.lambda_router 0.15 --framework.vlog.losses.lambda_critic 0.15 --trainer.gradient_clipping 0.5 --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 5000 --trainer.early_stop_metric vlog/action_loss --trainer.max_consecutive_nan_steps 30" \
bash "${TRAIN_SH}" 2>&1 | tee "${LOG_DIR}/${S6_RUN}.log"

S6_CKPT="$(ls -t "${REPO_ROOT}/results/Checkpoints/${S6_RUN}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1 || true)"
[[ -n "${S6_CKPT}" ]] || { echo "No Stage6 ckpt" >&2; exit 1; }

python3 - "${REPO_ROOT}/results/Checkpoints/${S6_RUN}/logs/train_log.jsonl" <<'PY'
import json, math, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok=0
for r in rows:
    al=r.get('vlog/action_loss')
    if isinstance(al,float) and math.isfinite(al) and al>0:
        if all(not (isinstance(r.get(k),float) and not math.isfinite(r[k])) for k in ('vlog/vq_loss','vlog/mean_beta','vlog/adapter_alpha')):
            ok+=1
ratio=ok/max(1,len(rows))
print(f'Stage6 health finite_ratio={ratio:.3f} ok={ok}/{len(rows)}')
if ratio < 0.90 or ok < 400:
    raise SystemExit('Stage6 unhealthy')
PY

python3 - <<PY
import json
json.dump({
  "attempt": "rescue_retry1",
  "stage5_ckpt": "${S5_CKPT}",
  "stage6_ckpt": "${S6_CKPT}",
  "run_id": "${S6_RUN}",
}, open("${LOG_DIR}/best_model.json","w"), indent=2)
PY

echo "[$(date -Is)] Stage6 OK -> ${S6_CKPT}; starting eval" | tee -a "${LOG_DIR}/loop.log"

CKPT="${S6_CKPT}" RUN_ID="${S6_RUN}" OUTPUT_ROOT="${EVAL_ROOT}" N_EPISODES="${N_EPISODES:-20}" \
  bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/${S6_RUN}_eval.log"

CKPT="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt" \
RUN_ID="StarVLA_OFT_Robocasa_Pretrained" OUTPUT_ROOT="${EVAL_ROOT}" N_EPISODES="${N_EPISODES:-20}" \
EXTRA_EVAL_ARGS="--args.no_send_state" \
  bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/baseline_eval.log"

python3 - <<PY
import json, re
from pathlib import Path
root=Path("${EVAL_ROOT}")
summary={}
for run_dir in sorted(root.iterdir()):
    if not run_dir.is_dir(): continue
    scores={}
    for log in run_dir.glob("*/eval.log"):
        m=re.search(r"Success rate:\s*([0-9.]+)", log.read_text(errors="ignore"))
        if m: scores[log.parent.name]=float(m.group(1))*100
    if scores:
        avg=sum(scores.values())/len(scores)
        summary[run_dir.name]={"avg":avg,"n":len(scores),"scores":scores}
        print(f"{run_dir.name}: avg={avg:.2f}% over {len(scores)} tasks")
(root/"summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote", root/"summary.json")
PY

echo "[$(date -Is)] Rescue+eval done" | tee -a "${LOG_DIR}/loop.log"
