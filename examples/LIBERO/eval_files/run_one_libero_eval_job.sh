#!/usr/bin/env bash
# Worker: one (checkpoint, suite) eval with its own policy server port.
set -euo pipefail

run_id="${1:?run_id}"
suite="${2:?suite}"
ckpt="${3:?ckpt}"
port="${4:?port}"
eval_dir="${5:?eval_dir}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_stage_eval}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/manifest.jsonl}"
GPU_ID="${GPU_ID:-0}"
HOST="${HOST:-127.0.0.1}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/playground/Code/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/nlk/.conda/envs/vlog_vla/bin/python}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/nlk/.conda/envs/vlog_vla/bin/python}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${eval_dir}/videos" "${OUTPUT_ROOT}/${run_id}"
server_log="${OUTPUT_ROOT}/${run_id}/policy_server_${suite}.log"
parallel_log="${OUTPUT_ROOT}/parallel.log"

wait_for_server() {
  local log="$1"
  for _ in $(seq 1 120); do
    if grep -q "server listening" "${log}" 2>/dev/null; then
      sleep 3
      return 0
    fi
    sleep 2
  done
  return 1
}

echo "[$(date -Is)] START ${run_id}/${suite} port=${port}" >> "${parallel_log}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${ckpt}" --port "${port}" --use_bf16 \
  >"${server_log}" 2>&1 &
spid=$!

cleanup() {
  kill "${spid}" 2>/dev/null || true
  wait "${spid}" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_server "${server_log}"; then
  echo "[$(date -Is)] FAIL server ${run_id}/${suite}" >> "${parallel_log}"
  exit 1
fi

STARVLA_DIR="${REPO_ROOT}" LIBERO_HOME="${LIBERO_HOME}" LIBERO_PYTHON="${LIBERO_PYTHON}" \
CKPT="${ckpt}" HOST="${HOST}" PORT="${port}" TASK_SUITE_NAME="${suite}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" MAX_TASKS="${MAX_TASKS}" \
UNNORM_KEY="${UNNORM_KEY}" OUTPUT_DIR="${eval_dir}" \
  bash "${SCRIPT_DIR}/eval_vlog_libero.sh" \
    --suite "${suite}" --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --max_tasks "${MAX_TASKS}" --ckpt "${ckpt}" --port "${port}" \
    --output_dir "${eval_dir}" \
    >"${eval_dir}/eval.log" 2>&1

python3 - <<PY >> "${MANIFEST}"
import json
print(json.dumps({
  "run_id": "${run_id}",
  "suite": "${suite}",
  "data_mix": "${suite}",
  "checkpoint": "${ckpt}",
  "eval_log": "${eval_dir}/eval.log",
  "episode_logs": "${eval_dir}/episode_logs.jsonl",
  "videos_dir": "${eval_dir}/videos",
}))
PY

echo "[$(date -Is)] DONE ${run_id}/${suite}" >> "${parallel_log}"
