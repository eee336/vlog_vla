#!/usr/bin/env bash
# Create robocasa conda env for RoboCasa GR1 tabletop simulation (separate from vlog_vla).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
ROBOSUITE_DIR="${REPO_ROOT}/playground/Code/robosuite"
ROBOCASA_DIR="${REPO_ROOT}/playground/Code/robocasa-gr1-tabletop-tasks"
ENV_NAME="${ROBOCASA_ENV_NAME:-robocasa}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env '${ENV_NAME}' already exists"
else
  echo "[setup] creating conda env '${ENV_NAME}' (python=3.10)"
  conda create -y -c conda-forge -n "${ENV_NAME}" python=3.10 pip
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing robosuite from ${ROBOSUITE_DIR}"
pip install -e "${ROBOSUITE_DIR}"
pip install mink==0.0.5 robosuite_models

echo "[setup] installing robocasa-gr1-tabletop-tasks from ${ROBOCASA_DIR}"
pip install -e "${ROBOCASA_DIR}"

echo "[setup] installing eval client deps"
pip install tyro websocket-client websockets msgpack opencv-python matplotlib mediapy av pydantic typing_extensions gymnasium

echo "[setup] done. Activate with: conda activate ${ENV_NAME}"
