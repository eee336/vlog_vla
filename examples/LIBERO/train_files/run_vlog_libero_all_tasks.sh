#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_OVERRIDE_NAMES=(
  TRAIN_PYTHON LIBERO_PYTHON BASE_CKPT BASE_VLM DATA_ROOT RUN_ROOT_DIR RUN_PREFIX
  TRAIN_STAGE MAX_TRAIN_STEPS SAVE_INTERVAL EVAL_INTERVAL PER_DEVICE_BATCH_SIZE
  NUM_PROCESSES VIDEO_BACKEND LIBERO_SUITES NUM_TRIALS_PER_TASK MAX_TASKS
  UNNORM_KEY HOST PORT_BASE GPU_ID CHAIN_SUITES RUN_EVAL OUTPUT_ROOT
  MUJOCO_GL_VALUE PYOPENGL_PLATFORM_VALUE WANDB_MODE CUDA_HOME USE_DEEPSPEED
)
ENV_OVERRIDES=()
for name in "${ENV_OVERRIDE_NAMES[@]}"; do
  if [[ -v "${name}" ]]; then
    ENV_OVERRIDES+=("$(declare -p "${name}")")
  fi
done

if [[ -f "${REPO_ROOT}/.vlog_vla.env" ]]; then
  set -a
  source "${REPO_ROOT}/.vlog_vla.env"
  set +a
fi
for declaration in "${ENV_OVERRIDES[@]}"; do
  eval "${declaration}"
done

: "${TRAIN_PYTHON:=/home/nlk/.conda/envs/vlog_vla/bin/python}"
: "${LIBERO_PYTHON:=${TRAIN_PYTHON}}"
: "${BASE_CKPT:=${REPO_ROOT}/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt}"
: "${BASE_VLM:=${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
: "${DATA_ROOT:=${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA}"
: "${RUN_ROOT_DIR:=results/Checkpoints}"
: "${RUN_PREFIX:=VLOG_VLA_QwenOFT}"
: "${TRAIN_STAGE:=stage6_full}"
: "${MAX_TRAIN_STEPS:=20000}"
: "${SAVE_INTERVAL:=20000}"
: "${EVAL_INTERVAL:=100000000}"
: "${PER_DEVICE_BATCH_SIZE:=1}"
: "${NUM_PROCESSES:=1}"
: "${USE_DEEPSPEED:=0}"
: "${VIDEO_BACKEND:=pyav}"
: "${LIBERO_SUITES:=libero_spatial libero_object libero_goal libero_10}"
: "${NUM_TRIALS_PER_TASK:=50}"
: "${MAX_TASKS:=-1}"
: "${UNNORM_KEY:=franka}"
: "${HOST:=127.0.0.1}"
: "${PORT_BASE:=5694}"
: "${GPU_ID:=0}"
: "${CHAIN_SUITES:=0}"
: "${RUN_EVAL:=1}"
: "${OUTPUT_ROOT:=${REPO_ROOT}/outputs/vlog_libero_all_tasks}"
: "${MUJOCO_GL_VALUE:=egl}"
: "${PYOPENGL_PLATFORM_VALUE:=egl}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "${OUTPUT_ROOT}"
MANIFEST="${OUTPUT_ROOT}/manifest.jsonl"
: > "${MANIFEST}"

ckpt_for_next="${BASE_CKPT}"
suite_index=0

for suite in ${LIBERO_SUITES}; do
  data_mix="${suite}"
  run_id="${RUN_PREFIX}_${suite}_steps${MAX_TRAIN_STEPS}"
  eval_dir="${OUTPUT_ROOT}/${suite}/eval"
  train_log="${OUTPUT_ROOT}/${suite}/train.log"
  server_log="${OUTPUT_ROOT}/${suite}/policy_server.log"
  port="$((PORT_BASE + suite_index))"
  mkdir -p "${OUTPUT_ROOT}/${suite}"

  echo "==> Training ${suite} with data_mix=${data_mix}, run_id=${run_id}"
  BASE_CKPT="${ckpt_for_next}" \
  BASE_VLM="${BASE_VLM}" \
  DATA_ROOT="${DATA_ROOT}" \
  DATA_MIX="${data_mix}" \
  RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
  RUN_ID="${run_id}" \
  TRAIN_STAGE="${TRAIN_STAGE}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  USE_DEEPSPEED="${USE_DEEPSPEED}" \
  VIDEO_BACKEND="${VIDEO_BACKEND}" \
  TRAIN_PYTHON="${TRAIN_PYTHON}" \
  "${SCRIPT_DIR}/run_vlog_libero_train.sh" 2>&1 | tee "${train_log}"

  final_ckpt="${REPO_ROOT}/${RUN_ROOT_DIR}/${run_id}/final_model/pytorch_model.pt"
  if [[ ! -f "${final_ckpt}" ]]; then
    final_ckpt="${REPO_ROOT}/${RUN_ROOT_DIR}/${run_id}/checkpoints/steps_${MAX_TRAIN_STEPS}_pytorch_model.pt"
  fi
  if [[ ! -f "${final_ckpt}" ]]; then
    echo "Missing trained checkpoint for ${suite}: ${final_ckpt}" >&2
    exit 1
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "==> Evaluating ${suite} from ${final_ckpt}"
    mkdir -p "${eval_dir}"
    GPU_ID="${GPU_ID}" \
    PORT="${port}" \
    CKPT="${final_ckpt}" \
    VLOG_CHECKPOINT="${final_ckpt}" \
    VLOG_LOG_DIR="${eval_dir}/vlog_option_logs" \
    STARVLA_PYTHON="${TRAIN_PYTHON}" \
    "${REPO_ROOT}/examples/LIBERO/eval_files/run_vlog_policy_server.sh" > "${server_log}" 2>&1 &
    server_pid=$!

    "${TRAIN_PYTHON}" - <<PY
import socket, sys, time
host = "${HOST}"
port = int("${port}")
deadline = time.time() + 600
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError:
        time.sleep(2)
sys.exit("policy server did not open port %s:%d in time" % (host, port))
PY

    set +e
    LIBERO_PYTHON="${LIBERO_PYTHON}" \
    HOST="${HOST}" \
    PORT="${port}" \
    CKPT="${final_ckpt}" \
    TASK_SUITE_NAME="${suite}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
    MAX_TASKS="${MAX_TASKS}" \
    UNNORM_KEY="${UNNORM_KEY}" \
    OUTPUT_DIR="${eval_dir}" \
    MUJOCO_GL_VALUE="${MUJOCO_GL_VALUE}" \
    PYOPENGL_PLATFORM_VALUE="${PYOPENGL_PLATFORM_VALUE}" \
    "${REPO_ROOT}/examples/LIBERO/eval_files/eval_vlog_libero.sh"
    eval_status=$?
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" >/dev/null 2>&1 || true
    set -e
    if [[ "${eval_status}" -ne 0 ]]; then
      echo "Evaluation failed for ${suite}; see ${eval_dir}/episode_logs.jsonl and ${server_log}" >&2
      exit "${eval_status}"
    fi
  fi

  printf '{"suite":"%s","data_mix":"%s","run_id":"%s","checkpoint":"%s","train_log":"%s","eval_log":"%s"}\n' \
    "${suite}" "${data_mix}" "${run_id}" "${final_ckpt}" "${train_log}" "${eval_dir}/episode_logs.jsonl" >> "${MANIFEST}"

  if [[ "${CHAIN_SUITES}" == "1" ]]; then
    ckpt_for_next="${final_ckpt}"
  fi
  suite_index=$((suite_index + 1))
done

"${TRAIN_PYTHON}" "${REPO_ROOT}/scripts/vlog_vla/collect_libero_success_table.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}"

echo "LIBERO success table: ${OUTPUT_ROOT}/libero_success_table.md"
