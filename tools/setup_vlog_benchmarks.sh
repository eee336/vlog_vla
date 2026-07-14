#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.vlog_vla.env" ]]; then
  set -a
  source "${REPO_ROOT}/.vlog_vla.env"
  set +a
elif [[ -f "${REPO_ROOT}/.vlog_vla.env.example" ]]; then
  set -a
  source "${REPO_ROOT}/.vlog_vla.env.example"
  set +a
fi

: "${PROJECT_ROOT:=${REPO_ROOT}}"
: "${PLAYGROUND_ROOT:=${PROJECT_ROOT}/playground}"
: "${PRETRAINED_ROOT:=${PLAYGROUND_ROOT}/Pretrained_models}"
: "${DATASETS_ROOT:=${PLAYGROUND_ROOT}/Datasets}"
: "${CODE_ROOT:=${PLAYGROUND_ROOT}/Code}"
: "${CHECKPOINT_ROOT:=${PLAYGROUND_ROOT}/Checkpoints}"
: "${BASE_VLM:=${PRETRAINED_ROOT}/Qwen3-VL-4B-Instruct}"
: "${LIBERO_DATA_ROOT:=${DATASETS_ROOT}/LEROBOT_LIBERO_DATA}"
: "${ROBOTWIN_DATA_ROOT:=${DATASETS_ROOT}/RoboTwin}"
: "${ROBOCASA365_DATA_ROOT:=${DATASETS_ROOT}/robocasa365}"
: "${ROBOCASA_TABLETOP_DATA_ROOT:=${DATASETS_ROOT}/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
: "${LIBERO_HOME:=${CODE_ROOT}/LIBERO}"
: "${ROBOTWIN_PATH:=${CODE_ROOT}/RoboTwin}"
: "${ROBOCASA365_PATH:=${CODE_ROOT}/robocasa365}"
: "${ROBOSUITE_PATH:=${CODE_ROOT}/robosuite}"
: "${ROBOCASA_TABLETOP_PATH:=${CODE_ROOT}/robocasa-gr1-tabletop-tasks}"

mkdir -p "${PRETRAINED_ROOT}" "${DATASETS_ROOT}" "${CODE_ROOT}" "${CHECKPOINT_ROOT}"

hf_download() {
  local repo="$1"
  local dest="$2"
  local repo_type="${3:-model}"
  mkdir -p "${dest}"
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo}" --repo-type "${repo_type}" --local-dir "${dest}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo}" --repo-type "${repo_type}" --local-dir "${dest}"
  else
    python -m pip install -U "huggingface-hub>=0.35.3"
    huggingface-cli download "${repo}" --repo-type "${repo_type}" --local-dir "${dest}"
  fi
}

clone_or_update() {
  local url="$1"
  local dest="$2"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" pull --ff-only
  else
    git clone "${url}" "${dest}"
  fi
}

download_libero_data() {
  local libero_raw="${DATASETS_ROOT}/libero_hf"
  mkdir -p "${libero_raw}"
  for repo in \
    IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
    IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
  do
    hf_download "${repo}" "${libero_raw}/${repo##*/}" dataset
  done
  mkdir -p "${DATASETS_ROOT}"
  ln -sfn "${libero_raw}" "${LIBERO_DATA_ROOT}"
  for suite in libero_10_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot libero_object_no_noops_1.0.0_lerobot libero_spatial_no_noops_1.0.0_lerobot; do
    mkdir -p "${LIBERO_DATA_ROOT}/${suite}/meta"
    cp "${REPO_ROOT}/examples/simBenchmarks/LIBERO/train_files/modality.json" "${LIBERO_DATA_ROOT}/${suite}/meta/modality.json"
  done
}

download_vlm_cotrain_data() {
  hf_download "StarVLA/LLaVA-OneVision-COCO" "${DATASETS_ROOT}/LLaVA-OneVision-COCO" dataset
  if [[ -f "${DATASETS_ROOT}/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" ]]; then
    unzip -n "${DATASETS_ROOT}/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "${DATASETS_ROOT}/LLaVA-OneVision-COCO"
  fi
}

download_checkpoints() {
  hf_download "StarVLA/Qwen3-VL-OFT-LIBERO-4in1" "${PRETRAINED_ROOT}/StarVLA_Qwen3_VL_OFT_LIBERO_4in1"
  hf_download "StarVLA/Qwen3-VL-OFT-RoboTwin2-All" "${PRETRAINED_ROOT}/Qwen3-VL-OFT-RoboTwin2-All"
  hf_download "StarVLA/Qwen3-VL-OFT-Robocasa" "${PRETRAINED_ROOT}/Qwen3-VL-OFT-Robocasa"
  hf_download "StarVLA/Qwen3-VL-GR00T-Robocasa-gr1" "${PRETRAINED_ROOT}/Qwen3-VL-GR00T-Robocasa-gr1"
}

setup_robotwin_code() {
  clone_or_update "https://github.com/RoboTwin-Platform/RoboTwin.git" "${ROBOTWIN_PATH}"
  echo "RoboTwin checkout: ${ROBOTWIN_PATH}"
  echo "Apply the policy_ckpt_path patch documented in examples/simBenchmarks/Robotwin/README.md before evaluation."
}

setup_robocasa365_code() {
  clone_or_update "https://github.com/ARISE-Initiative/robosuite.git" "${ROBOSUITE_PATH}"
  clone_or_update "https://github.com/robocasa/robocasa.git" "${ROBOCASA365_PATH}"
  echo "RoboCasa365 checkout: ${ROBOCASA365_PATH}"
  echo "In the robocasa365 env, run: pip install -e ${ROBOSUITE_PATH} -e ${ROBOCASA365_PATH} lerobot mujoco"
  echo "Then run setup_macros.py and set DATASET_BASE_PATH=${ROBOCASA365_DATA_ROOT}"
}

setup_robocasa_tabletop_code() {
  clone_or_update "https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git" "${ROBOCASA_TABLETOP_PATH}"
  echo "RoboCasa tabletop checkout: ${ROBOCASA_TABLETOP_PATH}"
}

download_robocasa_tabletop_data() {
  hf_download "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim" "${ROBOCASA_TABLETOP_DATA_ROOT}" dataset
}

print_check() {
  local missing=0
  for path in \
    "${BASE_VLM}" \
    "${LIBERO_DATA_ROOT}" \
    "${ROBOTWIN_PATH}" \
    "${ROBOCASA365_PATH}" \
    "${ROBOCASA_TABLETOP_PATH}" \
    "${PRETRAINED_ROOT}/StarVLA_Qwen3_VL_OFT_LIBERO_4in1" \
    "${PRETRAINED_ROOT}/Qwen3-VL-OFT-RoboTwin2-All" \
    "${PRETRAINED_ROOT}/Qwen3-VL-OFT-Robocasa" \
    "${PRETRAINED_ROOT}/Qwen3-VL-GR00T-Robocasa-gr1"
  do
    if [[ -e "${path}" ]]; then
      printf "[ok]      %s\n" "${path}"
    else
      printf "[missing] %s\n" "${path}"
      missing=1
    fi
  done
  return "${missing}"
}

usage() {
  cat <<'EOF'
Usage: bash tools/setup_vlog_benchmarks.sh <command>

Commands:
  check                    Print local path status.
  download-libero          Download LIBERO LeRobot datasets and copy modality.json.
  download-vlm-data        Download LLaVA-OneVision-COCO co-training data.
  download-checkpoints     Download released LIBERO/RoboTwin/RoboCasa checkpoints.
  robotwin-code            Clone or update RoboTwin.
  robocasa365-code         Clone or update official RoboCasa + robosuite.
  robocasa-tabletop-code   Clone or update RoboCasa GR1 tabletop tasks.
  robocasa-tabletop-data   Download NVIDIA GR00T RoboCasa tabletop finetune data.
  all-light                Run code setup plus path check; does not download large datasets/checkpoints.
EOF
}

cmd="${1:-check}"
case "${cmd}" in
  check) print_check ;;
  download-libero) download_libero_data ;;
  download-vlm-data) download_vlm_cotrain_data ;;
  download-checkpoints) download_checkpoints ;;
  robotwin-code) setup_robotwin_code ;;
  robocasa365-code) setup_robocasa365_code ;;
  robocasa-tabletop-code) setup_robocasa_tabletop_code ;;
  robocasa-tabletop-data) download_robocasa_tabletop_data ;;
  all-light)
    setup_robotwin_code
    setup_robocasa365_code
    setup_robocasa_tabletop_code
    print_check || true
    ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
