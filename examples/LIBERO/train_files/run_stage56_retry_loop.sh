#!/usr/bin/env bash
# Robust Stage 5->6 retry loop with param search on NaN failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/examples/LIBERO/train_files/run_vlog_libero_train.sh"
EVAL_SH="${REPO_ROOT}/examples/LIBERO/eval_files/run_libero_eval_stage6.sh"
LOG_DIR="${REPO_ROOT}/outputs/vlog_stage56_retry_loop"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlog_vla

STAGE4_CKPT="${REPO_ROOT}/results/Checkpoints/VLOG_VLA_LIBERO_STAGE4_V2/checkpoints/steps_15000_pytorch_model.pt"
BEST_SCORE_FILE="${LOG_DIR}/best_score.txt"
BEST_META="${LOG_DIR}/best_model.json"
echo "0" > "${BEST_SCORE_FILE}"

COMMON=(
  USE_DEEPSPEED=0 VIDEO_BACKEND=pyav EVAL_INTERVAL=100000000
  CONFIG_YAML=examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
  BASE_VLM="${REPO_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
  DATA_ROOT="${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA"
  RUN_ROOT_DIR=results/Checkpoints PER_DEVICE_BATCH_SIZE=12 NUM_PROCESSES=1
  DATA_MIX=libero_all TRAIN_PYTHON=/home/nlk/.conda/envs/vlog_vla/bin/python
)

ATTEMPTS=(
  "1|VLOG_VLA_LIBERO_STAGE5_RETRY1|20000|VLOG_VLA_LIBERO_FULL_RETRY1|30000|5e-5|5e-5|0.3|0.5|2000|"
  "2|VLOG_VLA_LIBERO_STAGE5_RETRY2|20000|VLOG_VLA_LIBERO_FULL_RETRY2|30000|2e-5|2e-5|0.2|0.3|1000|"
  "3|VLOG_VLA_LIBERO_STAGE5_RETRY3|15000|VLOG_VLA_LIBERO_FULL_RETRY3|25000|1e-5|1e-5|0.1|0.2|1000|--trainer.max_consecutive_nan_steps 50"
  "4|VLOG_VLA_LIBERO_STAGE5_RETRY4|12000|VLOG_VLA_LIBERO_FULL_RETRY4|20000|5e-5|5e-5|0.15|0.3|1000|--framework.vlog.losses.lambda_critic 0.2"
)

check_train_ok() {
  local log="$1" min_finite="$2"
  python3 - "$log" "$min_finite" <<'PY'
import json, math, sys
path=sys.argv[1]; min_finite=int(sys.argv[2])
if not __import__('os').path.exists(path):
    print('missing log'); sys.exit(1)
rows=[json.loads(l) for l in open(path) if l.strip()]
finite=[r for r in rows if isinstance(r.get('action_dit_loss'),float) and math.isfinite(r['action_dit_loss'])]
ratio=len(finite)/max(1,len(rows))
print(f'finite_ratio={ratio:.3f} finite={len(finite)} total={len(rows)}')
sys.exit(0 if len(finite)>=min_finite and ratio>=0.85 else 1)
PY
}

pick_ckpt() {
  local run_id="$1"
  ls -t "${REPO_ROOT}/results/Checkpoints/${run_id}/checkpoints"/steps_*_pytorch_model.pt 2>/dev/null | head -1
}

train_stage() {
  local run_id="$1" stage="$2" max_steps="$3" base_ckpt="$4" save_int="$5" extra="$6"
  echo "[$(date -Is)] TRAIN ${run_id} stage=${stage} steps=${max_steps}" | tee -a "${LOG_DIR}/loop.log"
  cd "${REPO_ROOT}"
  local extra_args="--datasets.vla_data.num_workers 8 --datasets.vla_data.prefetch_factor 4 --datasets.vla_data.pin_memory true ${extra}"
  env "${COMMON[@]}" RUN_ID="${run_id}" TRAIN_STAGE="${stage}" \
    MAX_TRAIN_STEPS="${max_steps}" SAVE_INTERVAL="${save_int}" BASE_CKPT="${base_ckpt}" \
    EXTRA_TRAIN_ARGS="${extra_args}" bash "${TRAIN_SH}" >> "${LOG_DIR}/${run_id}.log" 2>&1 || true
  pick_ckpt "${run_id}"
}

eval_and_score() {
  local run_id="$1" ckpt="$2" out="$3"
  mkdir -p "${out}"
  RUN_ID="${run_id}" OUTPUT_ROOT="${out}" MANIFEST="${out}/manifest.jsonl" CKPT="${ckpt}" \
    PARALLEL_JOBS=4 BASE_PORT=6100 \
    bash "${EVAL_SH}" 2>&1 | tee "${LOG_DIR}/${run_id}_eval.log"
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

for entry in "${ATTEMPTS[@]}"; do
  IFS='|' read -r aid s5_run s5_steps s6_run s6_max lr_vlog lr_action lr_router grad_clip s5_save s6_extra <<< "${entry}"
  echo "======== ATTEMPT ${aid} ========" | tee -a "${LOG_DIR}/loop.log"

  S5_CKPT=$(train_stage "${s5_run}" "stage5_router" "${s5_steps}" "${STAGE4_CKPT}" "${s5_save}" \
    "--trainer.learning_rate.vlog ${lr_vlog} --trainer.learning_rate.action_model ${lr_action} --framework.vlog.losses.lambda_router ${lr_router} --trainer.gradient_clipping ${grad_clip}")

  S5_LOG="${REPO_ROOT}/results/Checkpoints/${s5_run}/logs/train_log.jsonl"
  if [[ -z "${S5_CKPT}" || ! -f "${S5_CKPT}" ]] || ! check_train_ok "${S5_LOG}" 1000; then
    echo "[$(date -Is)] ATTEMPT ${aid} Stage5 FAILED" | tee -a "${LOG_DIR}/loop.log"
    continue
  fi

  S6_EXTRA="--trainer.learning_rate.vlog ${lr_vlog} --trainer.learning_rate.action_model ${lr_action} --framework.vlog.losses.lambda_router 0.2 --trainer.gradient_clipping ${grad_clip} --trainer.early_stop_patience 2000 --trainer.early_stop_window 200 --trainer.early_stop_min_delta 0.001 --trainer.early_stop_min_steps 5000 --trainer.early_stop_metric vlog/action_loss ${s6_extra}"

  S6_CKPT=$(train_stage "${s6_run}" "stage6_full" "${s6_max}" "${S5_CKPT}" 2000 "${S6_EXTRA}")
  S6_LOG="${REPO_ROOT}/results/Checkpoints/${s6_run}/logs/train_log.jsonl"
  if [[ -z "${S6_CKPT}" || ! -f "${S6_CKPT}" ]] || ! check_train_ok "${S6_LOG}" 500; then
    echo "[$(date -Is)] ATTEMPT ${aid} Stage6 FAILED" | tee -a "${LOG_DIR}/loop.log"
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
    json.dump({
      "attempt": int("${aid}"),
      "avg_percent": avg,
      "stage5_ckpt": "${S5_CKPT}",
      "stage6_ckpt": "${S6_CKPT}",
      "eval_dir": "${EVAL_OUT}",
    }, open("${BEST_META}","w"), indent=2)
    print(f"NEW BEST {avg:.2f}%")
else:
    print(f"avg {avg:.2f}% not better than {best:.2f}%")
PY
done

python3 "${REPO_ROOT}/scripts/vlog_vla/collect_all_stage_results.py" \
  --output-dir "${REPO_ROOT}/outputs/libero_all_results" --tag stage56_retry_final 2>/dev/null || true

echo "[$(date -Is)] Loop done. Best model:" | tee -a "${LOG_DIR}/loop.log"
cat "${BEST_META}" 2>/dev/null | tee -a "${LOG_DIR}/loop.log" || echo "none" | tee -a "${LOG_DIR}/loop.log"
