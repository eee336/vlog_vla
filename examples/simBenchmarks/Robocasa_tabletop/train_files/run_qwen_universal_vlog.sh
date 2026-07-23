#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

: "${BASE_CKPT:?Set BASE_CKPT to the official GR00T or a compatible UniversalVLOG checkpoint}"

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_universal_vlog_robocasa.yaml}"
TRAIN_STAGE="${TRAIN_STAGE:-u1_oracle}"
MAX_STEPS="${MAX_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RUN_ID="${RUN_ID:-qwen_universal_vlog_robocasa_${TRAIN_STAGE}}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

case "${TRAIN_STAGE}" in
  u0_base)
    echo "u0_base is an evaluation-only stage; run scripts/vlog_vla/diagnose_universal_base.py and Core-6 evaluation." >&2
    exit 2
    ;;
  u0_libero)
    VLOG_ENABLED=false
    FUSION_ENABLED=false
    ;;
  u1_oracle|u2_router|u3_joint)
    VLOG_ENABLED=true
    FUSION_ENABLED=true
    ;;
  *)
    echo "Unsupported TRAIN_STAGE=${TRAIN_STAGE}" >&2
    exit 2
    ;;
esac

STARVLA_GRAD_ACCUM="${GRAD_ACCUM}" accelerate launch starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  --framework.name QwenUniversalVLOG \
  --framework.vlog.train_stage "${TRAIN_STAGE}" \
  --framework.vlog.enabled "${VLOG_ENABLED}" \
  --framework.vlog.fusion_enabled "${FUSION_ENABLED}" \
  --trainer.pretrained_checkpoint "${BASE_CKPT}" \
  --trainer.is_resume false \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.pin_memory true \
  --datasets.vla_data.persistent_workers true \
  --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR}"
