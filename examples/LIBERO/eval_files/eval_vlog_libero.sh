#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
ROBOSUITE_PATH="${ROBOSUITE_PATH:-}"
CKPT="${CKPT:-${STARVLA_DIR}/results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_FULL/checkpoints/steps_5000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-${SUITE:-libero_spatial}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
MAX_TASKS="${MAX_TASKS:--1}"
SEED="${SEED:-7}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
OUTPUT_DIR="${OUTPUT_DIR:-${STARVLA_DIR}/outputs/vlog_stage7_real_starvla/official_eval_smoke}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"
ENV_OVERRIDE_NAMES=(
  STARVLA_DIR LIBERO_HOME LIBERO_PYTHON ROBOSUITE_PATH CKPT HOST PORT TASK_SUITE_NAME SUITE
  NUM_TRIALS_PER_TASK MAX_TASKS SEED UNNORM_KEY OUTPUT_DIR
  MUJOCO_GL_VALUE PYOPENGL_PLATFORM_VALUE
)
ENV_OVERRIDES=()
for name in "${ENV_OVERRIDE_NAMES[@]}"; do
  if [[ -v "${name}" ]]; then
    ENV_OVERRIDES+=("$(declare -p "${name}")")
  fi
done

if [[ -f "${STARVLA_DIR}/.vlog_vla.env" ]]; then
  set -a
  source "${STARVLA_DIR}/.vlog_vla.env"
  set +a
fi
for declaration in "${ENV_OVERRIDES[@]}"; do
  eval "${declaration}"
done

: "${LIBERO_HOME:?LIBERO_HOME is required. Set it in .vlog_vla.env or export it before running.}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) TASK_SUITE_NAME="$2"; shift 2 ;;
    --num_trials_per_task) NUM_TRIALS_PER_TASK="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --max_tasks) MAX_TASKS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
if [[ -z "${ROBOSUITE_PATH}" && -d "${STARVLA_DIR}/playground/Code/robosuite" ]]; then
  ROBOSUITE_PATH="${STARVLA_DIR}/playground/Code/robosuite"
fi
if [[ -n "${ROBOSUITE_PATH}" && ! -f "${ROBOSUITE_PATH}/robosuite/environments/manipulation/single_arm_env.py" ]]; then
  echo "Ignoring incompatible ROBOSUITE_PATH=${ROBOSUITE_PATH}; using installed robosuite package instead." >&2
  ROBOSUITE_PATH=""
fi
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${ROBOSUITE_PATH}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

OUT_ROOT="${OUTPUT_DIR}"
mkdir -p "${OUT_ROOT}/videos"

"${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.seed "${SEED}" \
  --args.video-out-path "${OUT_ROOT}/videos" \
  --args.unnorm-key "${UNNORM_KEY}" | tee "${OUT_ROOT}/episode_logs.jsonl"
