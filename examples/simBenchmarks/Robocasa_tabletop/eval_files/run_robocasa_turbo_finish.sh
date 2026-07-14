#!/usr/bin/env bash
# Kill serial eval if running, finish VLOG remaining tasks + baseline with max parallel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
EVAL_SH="${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_eval_suite.sh"
LOG_DIR="${REPO_ROOT}/outputs/vlog_robocasa_stage56_retry"
EVAL_ROOT="${REPO_ROOT}/outputs/robocasa_stage56_eval"
mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

echo "[$(date -Is)] TURBO: stopping serial eval processes" | tee -a "${LOG_DIR}/loop.log"
# Stop the continue-pipeline eval only (not unrelated jobs).
pkill -f 'run_robocasa_stage6_continue_and_eval.sh' 2>/dev/null || true
pkill -f 'run_robocasa_eval_suite.sh' 2>/dev/null || true
pkill -f 'server_policy.py --ckpt_path .*ROBOCASA_FULL_RETRY2' 2>/dev/null || true
pkill -f 'simulation_env.py .*robocasa_stage56_eval' 2>/dev/null || true
sleep 3

# Clear incomplete potato/wine dirs that lack Success rate
python3 - <<'PY'
from pathlib import Path
import re, shutil
root = Path('/home/nlk/project/vlog_vla/outputs/robocasa_stage56_eval/VLOG_VLA_ROBOCASA_FULL_RETRY2')
for d in root.glob('*_Env'):
    log = d / 'eval.log'
    text = log.read_text(errors='ignore') if log.exists() else ''
    if not re.search(r'Success rate:\s*[0-9.]+', text):
        print('reset incomplete', d.name)
        if (d / 'videos').exists():
            shutil.rmtree(d / 'videos')
        if log.exists():
            log.unlink()
PY

export N_PARALLEL="${N_PARALLEL:-6}"
export N_ENVS="${N_ENVS:-1}"
export FILL_WORKERS="${FILL_WORKERS:-1}"
export N_EPISODES="${N_EPISODES:-20}"
export SKIP_DONE=1
export BASE_PORT="${BASE_PORT:-5800}"
export OUTPUT_ROOT="${EVAL_ROOT}"

S6_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_FULL_RETRY2/checkpoints/steps_7841_pytorch_model.pt"
[[ -f "${S6_CKPT}" ]] || S6_CKPT="$(ls -t "${REPO_ROOT}/results/Checkpoints/VLOG_VLA_ROBOCASA_FULL_RETRY2/checkpoints"/steps_*_pytorch_model.pt | head -1)"

echo "[$(date -Is)] TURBO VLOG N_PARALLEL=${N_PARALLEL} N_ENVS=${N_ENVS} FILL_WORKERS=${FILL_WORKERS}" | tee -a "${LOG_DIR}/loop.log"
CKPT="${S6_CKPT}" \
RUN_ID="VLOG_VLA_ROBOCASA_FULL_RETRY2" \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/VLOG_VLA_ROBOCASA_FULL_RETRY2_eval_turbo.log"

echo "[$(date -Is)] TURBO baseline OFT" | tee -a "${LOG_DIR}/loop.log"
CKPT="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt" \
RUN_ID="StarVLA_OFT_Robocasa_Pretrained" \
EXTRA_EVAL_ARGS="--args.no_send_state" \
BASE_PORT=5900 \
bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/baseline_eval_turbo.log"

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

echo "[$(date -Is)] TURBO All done." | tee -a "${LOG_DIR}/loop.log"
