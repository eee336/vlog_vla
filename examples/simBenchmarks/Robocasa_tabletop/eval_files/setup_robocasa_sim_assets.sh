#!/usr/bin/env bash
# Download RoboCasa GR1 tabletop simulation assets (fixtures + textures).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
ASSETS_DIR="${REPO_ROOT}/playground/Code/robocasa-gr1-tabletop-tasks/robocasa/models/assets"
LOG_DIR="${REPO_ROOT}/outputs/robocasa_asset_setup"
mkdir -p "${LOG_DIR}" "${ASSETS_DIR}"

download_and_unzip() {
  local name="$1"
  local url="$2"
  local dest_dir="$3"
  local zip_path="${ASSETS_DIR}/${name}.zip"

  echo "[assets] downloading ${name} ..."
  for attempt in 1 2 3; do
    if curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 --max-time 7200 \
      -o "${zip_path}" "${url}"; then
      break
    fi
    echo "[assets] ${name} attempt ${attempt} failed, retrying..."
    rm -f "${zip_path}"
    sleep 15
  done

  [[ -s "${zip_path}" ]] || { echo "ERROR: empty download ${zip_path}" >&2; exit 1; }
  echo "[assets] extracting ${name} -> ${dest_dir}"
  mkdir -p "${dest_dir}"
  unzip -q -o "${zip_path}" -d "${ASSETS_DIR}"
  rm -f "${zip_path}"
}

download_and_unzip "textures" \
  "https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip" \
  "${ASSETS_DIR}/textures"

download_and_unzip "fixtures" \
  "https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip" \
  "${ASSETS_DIR}/fixtures"

echo "[assets] verify sample mesh"
test -f "${ASSETS_DIR}/fixtures/accessories/paper_towel_holders/holder_4/visuals/model_0.obj"
echo "[assets] done"
