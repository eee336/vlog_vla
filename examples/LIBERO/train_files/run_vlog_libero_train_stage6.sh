#!/usr/bin/env bash
set -euo pipefail
BASE_CKPT="${BASE_CKPT:-results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_STAGE5/checkpoints/steps_1000_pytorch_model.pt}" RUN_ID="${RUN_ID:-VLOG_VLA_QwenOFT_LIBERO_FULL}" TRAIN_STAGE=stage6_full MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}" bash "$(dirname "$0")/run_vlog_libero_train.sh"
