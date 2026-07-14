#!/usr/bin/env bash
# Resume Stage 6 + eval from already-completed Stage 5 retry checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/examples/LIBERO/train_files/run_vlog_libero_train.sh"
EVAL_SH="${REPO_ROOT}/examples/LIBERO/eval_files/run_libero_eval_stage6.sh"
LOG_DIR="${REPO_ROOT}/outputs/vlog_stage56_retry_loop"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

BEST_SCORE_FILE="${LOG_DIR}/best_score.txt"
BEST_META="${LOG_DIR}/best_model.json"
[[ -f "${BEST_SCORE_FILE}" ]] || echo "0" > "${BEST_SCORE_FILE}"

COMMON=(
  USE_DEEPSPEED=0 VIDEO_BACKEND=pyav EVAL_INTERVAL=100000000
  CONFIG_YAML=examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
  BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  DATA_ROOT="${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA"
  RUN_ROOT_DIR=results/Checkpoints PER_DEVICE_BATCH_SIZE=12 NUM_PROCESSES=1
  DATA_MIX=libero_all TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python
)

# s5_run|s6_run|s6_max|lr|grad|s6_extra
RESUME=(
  "VLOG_VLA_LIBERO_STAGE5_RETRY1|VLOG_VLA_LIBERO_FULL_RETRY1|30000|5e-5|0.5|"
  "VLOG_VLA_LIBERO_STAGE5_RETRY3|VLOG_VLA_LIBERO_FULL_RETRY3|25000|1e-5|0.2|--trainer.max_consecutive_nan_steps 50"
  "VLOG_VLA_LIBERO_STAGE5_RETRY4|VLOG_VLA_LIBERO_FULL_RETRY4|20000|5e-5|0.3|--framework.vlog.losses.lambda_critic 0.2"
  "VLOG_VLA_LIBERO_STAGE5_RETRY7|VLOG_VLA_LIBERO_FULL_RETRY7|30000|2e-5|0.3|--trainer.max_consecutive_nan_steps 50"
)

pick_ckpt() {
  ls -t "${REPO_ROOT}/results/Checkpoints/$1/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1
}

train_s6() {
  local s6_run="$1" s5_ckpt="$2" s6_max="$3" lr="$4" grad="$5" extra="$6"
  echo "[$(date -Is)] S6 ${s6_run} from ${s5_ckpt}" | tee -a "${LOG_DIR}/resume.log"
  local s6_extra="--trainer.learning_rate.vlog ${lr} --trainer.learning_rate.action_model ${lr} --framework.vlog.losses.lambda_router 0.2 --trainer.gradient_clipping ${grad} --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 5000 --trainer.early_stop_metric vlog/action_loss ${extra}"
  cd "${REPO_ROOT}"
  env "${COMMON[@]}" RUN_ID="${s6_run}" TRAIN_STAGE="stage6_full" \
    MAX_TRAIN_STEPS="${s6_max}" SAVE_INTERVAL=2000 BASE_CKPT="${s5_ckpt}" \
    EXTRA_TRAIN_ARGS="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true ${s6_extra}" \
    bash "${TRAIN_SH}" >> "${LOG_DIR}/${s6_run}.log" 2>&1 || true
  pick_ckpt "${s6_run}"
}

eval_and_score() {
  local run_id="$1" ckpt="$2" out="$3"
  mkdir -p "${out}"
  RUN_ID="${run_id}" OUTPUT_ROOT="${out}" MANIFEST="${out}/manifest.jsonl" CKPT="${ckpt}" \
    PARALLEL_JOBS=4 BASE_PORT=6300 \
    bash "${EVAL_SH}" >> "${LOG_DIR}/${run_id}_eval.log" 2>&1
  python3 - <<PY
import re, glob
root="${out}"; run_id="${run_id}"
vals=[]
for s in ['libero_spatial','libero_object','libero_goal','libero_10']:
    for ep in glob.glob(f"{root}/{run_id}/{s}/eval.log"):
        m=re.search(r'Total success rate:\s*([0-9.]+)', open(ep).read())
        if m: vals.append(float(m.group(1))*100)
avg=sum(vals)/len(vals) if vals else 0
print(f"avg={avg:.2f} suites={len(vals)}")
open(f"{root}/avg_score.txt","w").write(str(avg))
PY
}

for entry in "${RESUME[@]}"; do
  IFS='|' read -r s5_run s6_run s6_max lr grad s6_extra <<< "${entry}"
  S5_CKPT=$(pick_ckpt "${s5_run}")
  if [[ -z "${S5_CKPT}" || ! -f "${S5_CKPT}" ]]; then
    echo "[$(date -Is)] Skip ${s5_run}: no ckpt" | tee -a "${LOG_DIR}/resume.log"
    continue
  fi
  echo "======== RESUME ${s5_run} -> ${s6_run} ========" | tee -a "${LOG_DIR}/resume.log"
  S6_CKPT=$(train_s6 "${s6_run}" "${S5_CKPT}" "${s6_max}" "${lr}" "${grad}" "${s6_extra}")
  if [[ -z "${S6_CKPT}" || ! -f "${S6_CKPT}" ]]; then
    echo "[$(date -Is)] ${s6_run} S6 FAILED" | tee -a "${LOG_DIR}/resume.log"
    continue
  fi
  EVAL_OUT="${REPO_ROOT}/outputs/libero_stage56_eval/${s6_run}"
  eval_and_score "${s6_run}" "${S6_CKPT}" "${EVAL_OUT}"
  AVG=$(cat "${EVAL_OUT}/avg_score.txt")
  BEST=$(cat "${BEST_SCORE_FILE}")
  python3 - <<PY
import json
best=float("${BEST}"); avg=float("${AVG}")
if avg>0 and avg>=best:
    open("${BEST_SCORE_FILE}","w").write(str(avg))
    json.dump({"s5_run":"${s5_run}","s6_run":"${s6_run}","avg_percent":avg,"stage5_ckpt":"${S5_CKPT}","stage6_ckpt":"${S6_CKPT}","eval_dir":"${EVAL_OUT}"}, open("${BEST_META}","w"), indent=2)
    print(f"NEW BEST {avg:.2f}%")
else:
    print(f"avg {avg:.2f}% not better than {best:.2f}%")
PY
done

echo "[$(date -Is)] Resume done." | tee -a "${LOG_DIR}/resume.log"
cat "${BEST_META}" 2>/dev/null | tee -a "${LOG_DIR}/resume.log" || echo "none"
