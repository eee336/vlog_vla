#!/usr/bin/env bash
set -euo pipefail
BASE_CKPT="${BASE_CKPT:-results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_STAGE1/checkpoints/steps_1000_pytorch_model.pt}" RUN_ID="${RUN_ID:-VLOG_VLA_QwenOFT_LIBERO_STAGE2}" TRAIN_STAGE=stage2_option_discovery MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}" bash "$(dirname "$0")/run_vlog_libero_train.sh"
