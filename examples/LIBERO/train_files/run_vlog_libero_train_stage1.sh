#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-VLOG_VLA_QwenOFT_LIBERO_STAGE1}" TRAIN_STAGE=stage1_preserve MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}" bash "$(dirname "$0")/run_vlog_libero_train.sh"
