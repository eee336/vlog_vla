#!/usr/bin/env bash
set -euo pipefail
BASE_CKPT="${BASE_CKPT:-results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_STAGE4/checkpoints/steps_1000_pytorch_model.pt}" RUN_ID="${RUN_ID:-VLOG_VLA_QwenOFT_LIBERO_STAGE5}" TRAIN_STAGE=stage5_router MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}" bash "$(dirname "$0")/run_vlog_libero_train.sh"
