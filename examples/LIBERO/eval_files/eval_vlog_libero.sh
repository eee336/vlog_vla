#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_FULL/checkpoints/steps_5000_pytorch_model.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5694}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-${SUITE:-libero_spatial}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-2}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
OUTPUT_DIR="${OUTPUT_DIR:-${STARVLA_DIR}/outputs/vlog_stage7_real_starvla/official_eval_smoke}"
MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE:-egl}"
PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE:-egl}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) TASK_SUITE_NAME="$2"; shift 2 ;;
    --num_trials_per_task) NUM_TRIALS_PER_TASK="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
done

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

OUT_ROOT="${OUTPUT_DIR}"
mkdir -p "${OUT_ROOT}/videos"

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${OUT_ROOT}/videos" \
  --args.unnorm-key "${UNNORM_KEY}" | tee "${OUT_ROOT}/episode_logs.jsonl"
