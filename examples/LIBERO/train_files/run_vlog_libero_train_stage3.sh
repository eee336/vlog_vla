#!/usr/bin/env bash
set -euo pipefail
BASE_CKPT="${BASE_CKPT:-results/Checkpoints/VLOG_VLA_QwenOFT_LIBERO_STAGE2/checkpoints/steps_3000_pytorch_model.pt}" RUN_ID="${RUN_ID:-VLOG_VLA_QwenOFT_LIBERO_STAGE3}" TRAIN_STAGE=stage3_graph MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}" bash "$(dirname "$0")/run_vlog_libero_train.sh"
