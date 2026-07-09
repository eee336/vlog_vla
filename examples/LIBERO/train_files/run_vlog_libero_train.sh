#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml}"
BASE_CKPT="${BASE_CKPT:-playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
RUN_ID="${RUN_ID:-VLOG_VLA_QwenOFT_LIBERO_REAL}"
TRAIN_STAGE="${TRAIN_STAGE:-stage1_preserve}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${MAX_TRAIN_STEPS}}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
DATA_MIX="${DATA_MIX:-libero_all}"
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name QwenOFTVLOG \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.vlog.train_stage "${TRAIN_STAGE}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.pretrained_checkpoint "${BASE_CKPT}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
