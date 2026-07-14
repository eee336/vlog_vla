#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_OVERRIDE_NAMES=(
  WANDB_MODE CONFIG_YAML BASE_CKPT BASE_VLM DATA_ROOT RUN_ROOT_DIR RUN_ID
  TRAIN_STAGE MAX_TRAIN_STEPS SAVE_INTERVAL EVAL_INTERVAL PER_DEVICE_BATCH_SIZE
  NUM_PROCESSES DATA_MIX VIDEO_BACKEND TRAIN_PYTHON ACCELERATE_CONFIG USE_DEEPSPEED EXTRA_TRAIN_ARGS
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

: "${WANDB_MODE:=disabled}"
: "${CONFIG_YAML:=examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_oft_vlog_robocasa.yaml}"
: "${BASE_CKPT:=playground/Pretrained_models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt}"
: "${BASE_VLM:=playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
: "${DATA_ROOT:=playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
: "${RUN_ROOT_DIR:=results/Checkpoints}"
: "${RUN_ID:=VLOG_VLA_QwenOFT_ROBOCASA_REAL}"
: "${TRAIN_STAGE:=stage1_preserve}"
: "${MAX_TRAIN_STEPS:=1000}"
: "${SAVE_INTERVAL:=${MAX_TRAIN_STEPS}}"
: "${EVAL_INTERVAL:=${MAX_TRAIN_STEPS}}"
: "${PER_DEVICE_BATCH_SIZE:=4}"
: "${NUM_PROCESSES:=1}"
: "${DATA_MIX:=fourier_gr1_unified_1000}"
: "${VIDEO_BACKEND:=pyav}"
: "${TRAIN_PYTHON:=python}"
: "${ACCELERATE_CONFIG:=${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
: "${USE_DEEPSPEED:=0}"
: "${EXTRA_TRAIN_ARGS:=}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE
export STARVLA_USE_DEEPSPEED="${USE_DEEPSPEED}"

cd "${REPO_ROOT}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

LAUNCH_ARGS=(--num_processes "${NUM_PROCESSES}")
if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  LAUNCH_ARGS=(--config_file "${ACCELERATE_CONFIG}" "${LAUNCH_ARGS[@]}")
fi

TRAIN_CMD=(
  "${TRAIN_PYTHON}" -m accelerate.commands.launch
  "${LAUNCH_ARGS[@]}"
  "${REPO_ROOT}/starVLA/training/train_starvla.py"
  --config_yaml "${CONFIG_YAML}"
  --framework.name QwenOFTVLOG
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --framework.vlog.train_stage "${TRAIN_STAGE}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.pretrained_checkpoint "${BASE_CKPT}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
)
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_TRAIN_ARGS} )
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi
"${TRAIN_CMD[@]}"
