#!/usr/bin/env bash
# Smoke test: one RoboCasa task with StarVLA-OFT pretrained checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CKPT="${CKPT:-${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt}"
PORT="${PORT:-5678}"
GPU_ID="${GPU_ID:-0}"
ENV_NAME="${ENV_NAME:-gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env}"
N_EPISODES="${N_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-12}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/robocasa_smoke_eval}"
VIDEO_OUT="${OUTPUT_ROOT}/videos/smoke_$(basename "${CKPT}" .pt)_${ENV_NAME##*/}"
LOG_DIR="${OUTPUT_ROOT}/logs"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/nlk/.conda/envs/vlog_vla/bin/python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/home/nlk/.conda/envs/robocasa/bin/python}"

mkdir -p "${LOG_DIR}" "${VIDEO_OUT}"

source "$(conda info --base)/etc/profile.d/conda.sh"

[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
[[ -x "${ROBOCASA_PYTHON}" ]] || { echo "Missing robocasa env: ${ROBOCASA_PYTHON}" >&2; exit 1; }

echo "[smoke] ckpt=${CKPT}"
echo "[smoke] env=${ENV_NAME} episodes=${N_EPISODES} port=${PORT}"

# Step 1: policy server (vlog_vla env)
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${STARVLA_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
  --ckpt_path "${CKPT}" --port "${PORT}" --use_bf16 \
  >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 120); do
  if grep -q "server listening" "${LOG_DIR}/server.log" 2>/dev/null; then
    sleep 2
    break
  fi
  sleep 2
done
grep -q "server listening" "${LOG_DIR}/server.log" || {
  echo "Policy server failed to start. See ${LOG_DIR}/server.log"
  tail -30 "${LOG_DIR}/server.log"
  exit 1
}

# Step 2: simulation (robocasa env)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${ROBOCASA_PYTHON}" "${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py" \
  --args.env_name "${ENV_NAME}" \
  --args.port "${PORT}" \
  --args.n_episodes "${N_EPISODES}" \
  --args.n_envs 1 \
  --args.max_episode_steps "${MAX_STEPS}" \
  --args.n_action_steps "${N_ACTION_STEPS}" \
  --args.no_send_state \
  --args.video_out_path "${VIDEO_OUT}" \
  --args.pretrained_path "${CKPT}" \
  2>&1 | tee "${LOG_DIR}/simulation.log"

echo "[smoke] done. logs=${LOG_DIR} videos=${VIDEO_OUT}"
