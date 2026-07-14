#!/usr/bin/env bash
# LIBERO online eval for StarVLA base + VLOG Stage 1-5 checkpoints.
# Follows StarVLA two-terminal workflow (server + simulator) in one orchestrator.
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/libero_stage_eval}"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
PORT="${PORT:-5694}"
GPU_ID="${GPU_ID:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/playground/Code/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/home/nlk/.conda/envs/vlog_vla/bin/python}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/nlk/.conda/envs/vlog_vla/bin/python}"
HOST="${HOST:-127.0.0.1}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

mkdir -p "${OUTPUT_ROOT}"
: > "${MANIFEST}"

export PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

wait_for_server() {
  local port="$1"
  local tries="${2:-120}"
  for _ in $(seq 1 "${tries}"); do
    if grep -q "server listening" "${3:-/dev/null}" 2>/dev/null || \
       "${STARVLA_PYTHON}" - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", ${port}))
s.close()
PY
    then
      sleep 5
      return 0
    fi
    sleep 2
  done
  echo "Policy server did not become ready on port ${port}" >&2
  return 1
}

stop_server() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

run_eval_for_checkpoint() {
  local run_id="$1"
  local ckpt="$2"
  local server_log="${OUTPUT_ROOT}/${run_id}/policy_server.log"
  mkdir -p "${OUTPUT_ROOT}/${run_id}"

  echo "================================================================"
  echo "[$(date -Is)] Evaluating ${run_id}"
  echo "  checkpoint: ${ckpt}"
  echo "================================================================"

  cd "${REPO_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt}" \
    --port "${PORT}" \
    --use_bf16 \
    >"${server_log}" 2>&1 &
  local server_pid=$!

  if ! wait_for_server "${PORT}" 90 "${server_log}"; then
    echo "Server failed to start. Log:" >&2
    tail -50 "${server_log}" >&2 || true
    stop_server "${server_pid}"
    exit 1
  fi

  for suite in ${SUITES}; do
    local eval_dir="${OUTPUT_ROOT}/${run_id}/${suite}"
    mkdir -p "${eval_dir}/videos"
    echo "[$(date -Is)]  suite=${suite} trials=${NUM_TRIALS_PER_TASK}"

    STARVLA_DIR="${REPO_ROOT}" \
    LIBERO_HOME="${LIBERO_HOME}" \
    LIBERO_PYTHON="${LIBERO_PYTHON}" \
    CKPT="${ckpt}" \
    HOST="${HOST}" \
    PORT="${PORT}" \
    TASK_SUITE_NAME="${suite}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
    MAX_TASKS="${MAX_TASKS}" \
    UNNORM_KEY="${UNNORM_KEY}" \
    OUTPUT_DIR="${eval_dir}" \
    MUJOCO_GL_VALUE="${MUJOCO_GL}" \
    PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM}" \
      bash "${SCRIPT_DIR}/eval_vlog_libero.sh" \
        --suite "${suite}" \
        --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
        --ckpt "${ckpt}" \
        --port "${PORT}" \
        --max_tasks "${MAX_TASKS}" \
        --output_dir "${eval_dir}" \
        2>&1 | tee "${eval_dir}/eval.log"

    printf '%s\n' \
      "$(python3 - <<PY
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
)" >> "${MANIFEST}"
  done

  stop_server "${server_pid}"
  echo "[$(date -Is)] Finished ${run_id}"
}

declare -a CHECKPOINTS=(
  "StarVLA_Pretrained|${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE1_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE1_REAL/checkpoints/steps_20000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE2_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE2_REAL/checkpoints/steps_60000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE3_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE3_REAL/checkpoints/steps_20000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE4_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE4_REAL/checkpoints/steps_20000_pytorch_model.pt"
  "VLOG_VLA_LIBERO_STAGE5_REAL|${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE5_REAL/checkpoints/steps_20000_pytorch_model.pt"
)

for entry in "${CHECKPOINTS[@]}"; do
  run_id="${entry%%|*}"
  ckpt="${entry#*|}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping missing checkpoint: ${ckpt}" >&2
    continue
  fi
  run_eval_for_checkpoint "${run_id}" "${ckpt}"
done

python3 "${REPO_ROOT}/scripts/vlog_vla/collect_libero_success_table.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}"

echo "All eval complete. Results in ${OUTPUT_ROOT}"
