#!/usr/bin/env bash
# Evaluate one RoboCasa checkpoint on the 6 core GR1 tabletop tasks.
# Single-GPU max throughput:
#   - one policy server process per concurrent worker (sync infer ≠ parallel per server)
#   - VLOG requires batch size 1 → N_ENVS=1
#   - if fewer tasks than N_PARALLEL, shard episodes across workers and merge success rates
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CKPT="${CKPT:?CKPT required}"
RUN_ID="${RUN_ID:-$(basename "$(dirname "$(dirname "${CKPT}")")")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/robocasa_eval}"
BASE_PORT="${BASE_PORT:-${PORT:-5780}}"
GPU_ID="${GPU_ID:-0}"
N_EPISODES="${N_EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-12}"
N_PARALLEL="${N_PARALLEL:-6}"
N_ENVS="${N_ENVS:-1}"
SKIP_DONE="${SKIP_DONE:-1}"
# Auto-shard episodes across workers to fill N_PARALLEL slots.
FILL_WORKERS="${FILL_WORKERS:-1}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/nlk/.conda/envs/vlog_vla/bin/python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-/home/nlk/.conda/envs/robocasa/bin/python}"

ENV_NAMES=(
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
)

OUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
[[ -x "${ROBOCASA_PYTHON}" ]] || { echo "Missing robocasa python: ${ROBOCASA_PYTHON}" >&2; exit 1; }

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

SERVER_PIDS=()
WORKER_PIDS=()
SLOT_PORTS=()

cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${WORKER_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

task_done() {
  local log="$1"
  [[ -f "${log}" ]] || return 1
  grep -qE "Success rate:\s*[0-9.]+" "${log}"
}

PENDING=()
for ENV_NAME in "${ENV_NAMES[@]}"; do
  TASK_TAG="${ENV_NAME##*/}"
  TASK_LOG="${OUT_DIR}/${TASK_TAG}/eval.log"
  if [[ "${SKIP_DONE}" == "1" ]] && task_done "${TASK_LOG}"; then
    echo "[eval] skip done: ${TASK_TAG}"
    continue
  fi
  PENDING+=("${ENV_NAME}")
done

echo "[eval] run_id=${RUN_ID}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] episodes=${N_EPISODES} pending=${#PENDING[@]}/${#ENV_NAMES[@]} target_parallel=${N_PARALLEL} n_envs=${N_ENVS} fill=${FILL_WORKERS}"

if [[ ${#PENDING[@]} -eq 0 ]]; then
  echo "[eval] nothing pending"
else
  # Build job list: env_name|episodes|task_tag|shard_id|num_shards
  JOBS=()
  n_pending=${#PENDING[@]}
  if [[ "${FILL_WORKERS}" == "1" && ${N_PARALLEL} -gt ${n_pending} ]]; then
    # Distribute shard counts across tasks to fill workers.
    base=$((N_PARALLEL / n_pending))
    rem=$((N_PARALLEL % n_pending))
    for ((ti = 0; ti < n_pending; ti++)); do
      shards=${base}
      if (( ti < rem )); then shards=$((shards + 1)); fi
      if (( shards < 1 )); then shards=1; fi
      env_name="${PENDING[$ti]}"
      task_tag="${env_name##*/}"
      # split episodes as evenly as possible
      q=$((N_EPISODES / shards))
      r=$((N_EPISODES % shards))
      for ((s = 0; s < shards; s++)); do
        ep=${q}
        if (( s < r )); then ep=$((ep + 1)); fi
        if (( ep < 1 )); then continue; fi
        JOBS+=("${env_name}|${ep}|${task_tag}|${s}|${shards}")
      done
    done
  else
    for env_name in "${PENDING[@]}"; do
      task_tag="${env_name##*/}"
      JOBS+=("${env_name}|${N_EPISODES}|${task_tag}|0|1")
    done
  fi

  N_WORKERS=${#JOBS[@]}
  if (( N_WORKERS > N_PARALLEL )); then
    N_WORKERS=${N_PARALLEL}
  fi
  echo "[eval] jobs=${#JOBS[@]} concurrent_servers=${N_WORKERS}"

  for ((slot = 0; slot < N_WORKERS; slot++)); do
    port=$((BASE_PORT + slot))
    SLOT_PORTS+=("${port}")
    slog="${LOG_DIR}/server_slot${slot}_port${port}.log"
    echo "[eval] starting server slot=${slot} port=${port}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${STARVLA_PYTHON}" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
      --ckpt_path "${CKPT}" --port "${port}" --use_bf16 \
      >"${slog}" 2>&1 &
    SERVER_PIDS+=($!)
  done

  for ((slot = 0; slot < N_WORKERS; slot++)); do
    port=${SLOT_PORTS[$slot]}
    slog="${LOG_DIR}/server_slot${slot}_port${port}.log"
    ready=0
    for _ in $(seq 1 240); do
      if grep -q "server listening" "${slog}" 2>/dev/null; then
        ready=1
        break
      fi
      if ! kill -0 "${SERVER_PIDS[$slot]}" 2>/dev/null; then
        break
      fi
      sleep 2
    done
    if [[ "${ready}" != "1" ]]; then
      echo "Policy server slot ${slot} failed. See ${slog}" >&2
      tail -40 "${slog}" >&2
      exit 1
    fi
  done
  echo "[eval] ${N_WORKERS} servers ready"

  run_one_job() {
    local ENV_NAME="$1" EPISODES="$2" TASK_TAG="$3" SHARD="$4" NSHARDS="$5" PORT="$6"
    local TASK_DIR="${OUT_DIR}/${TASK_TAG}"
    local VIDEO_OUT SHARD_LOG
    mkdir -p "${TASK_DIR}"
    if (( NSHARDS > 1 )); then
      VIDEO_OUT="${TASK_DIR}/videos_shard${SHARD}"
      SHARD_LOG="${TASK_DIR}/eval_shard${SHARD}.log"
      rm -rf "${VIDEO_OUT}"
      mkdir -p "${VIDEO_OUT}"
    else
      VIDEO_OUT="${TASK_DIR}/videos"
      SHARD_LOG="${TASK_DIR}/eval.log"
      if [[ -d "${VIDEO_OUT}" ]] && ! task_done "${TASK_DIR}/eval.log"; then
        rm -rf "${VIDEO_OUT}"
        mkdir -p "${VIDEO_OUT}"
      else
        mkdir -p "${VIDEO_OUT}"
      fi
    fi
    echo "[eval] START ${TASK_TAG} shard=${SHARD}/${NSHARDS} eps=${EPISODES} port=${PORT}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      "${ROBOCASA_PYTHON}" "${REPO_ROOT}/examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py" \
      --args.env_name "${ENV_NAME}" \
      --args.port "${PORT}" \
      --args.n_episodes "${EPISODES}" \
      --args.n_envs "${N_ENVS}" \
      --args.max_episode_steps "${MAX_STEPS}" \
      --args.n_action_steps "${N_ACTION_STEPS}" \
      --args.video_out_path "${VIDEO_OUT}" \
      --args.pretrained_path "${CKPT}" \
      ${EXTRA_EVAL_ARGS} \
      >"${SHARD_LOG}" 2>&1
    local rc=$?
    if grep -qE "Success rate:\s*[0-9.]+" "${SHARD_LOG}"; then
      echo "[eval] DONE  ${TASK_TAG} shard=${SHARD} rc=${rc}"
    else
      echo "[eval] FAIL  ${TASK_TAG} shard=${SHARD} rc=${rc} (see ${SHARD_LOG})" >&2
      return 1
    fi
  }

  fail_count=0
  idx=0
  while (( idx < ${#JOBS[@]} )); do
    wave_pids=()
    for ((slot = 0; slot < N_WORKERS && idx < ${#JOBS[@]}; slot++)); do
      IFS='|' read -r env_name episodes task_tag shard nshards <<< "${JOBS[$idx]}"
      idx=$((idx + 1))
      run_one_job "${env_name}" "${episodes}" "${task_tag}" "${shard}" "${nshards}" "${SLOT_PORTS[$slot]}" &
      wpid=$!
      wave_pids+=("${wpid}")
      WORKER_PIDS+=("${wpid}")
    done
    for wpid in "${wave_pids[@]}"; do
      if ! wait "${wpid}"; then
        fail_count=$((fail_count + 1))
      fi
    done
  done

  if (( fail_count > 0 )); then
    echo "[eval] ${fail_count} job(s) failed" >&2
    exit 1
  fi

  # Merge sharded task logs into a single eval.log with overall success rate.
  python3 - <<PY
import re
from pathlib import Path
out = Path("${OUT_DIR}")
for task_dir in sorted(out.glob("*_Env")):
    shards = sorted(task_dir.glob("eval_shard*.log"))
    if not shards:
        continue
    rates = []
    eps = []
    # Prefer counting by Success rate * n? We only have rate. Parse "Collecting N episodes"
    # and Success rate. Weighted average by episodes.
    total_s, total_n = 0.0, 0
    for log in shards:
        text = log.read_text(errors="ignore")
        m = re.search(r"Success rate:\s*([0-9.]+)", text)
        n_m = re.search(r"Collecting\s+(\d+)\s+episodes", text) or re.search(r"Running\s+(\d+)\s+episodes", text)
        if not m:
            raise SystemExit(f"missing success in {log}")
        rate = float(m.group(1))
        n = int(n_m.group(1)) if n_m else 1
        total_s += rate * n
        total_n += n
        # merge videos
        shard_id = log.name.replace("eval_shard", "").replace(".log", "")
        vdir = task_dir / f"videos_shard{shard_id}"
        dest = task_dir / "videos"
        dest.mkdir(exist_ok=True)
        if vdir.exists():
            for p in vdir.glob("*.mp4"):
                p.rename(dest / p.name)
    avg = total_s / max(1, total_n)
    merged = task_dir / "eval.log"
    merged.write_text(
        f"# merged from {len(shards)} shards, episodes={total_n}\nSuccess rate: {avg:.2f}\n"
    )
    print(f"merged {task_dir.name}: {avg*100:.1f}% over {total_n} eps")
PY
fi

python3 - <<PY
import re
from pathlib import Path
root = Path("${OUT_DIR}")
scores = {}
for log in sorted(root.glob("*/eval.log")):
    m = re.search(r"Success rate:\s*([0-9.]+)", log.read_text(errors="ignore"))
    if m:
        scores[log.parent.name] = float(m.group(1)) * 100
        print(f"{log.parent.name}: {scores[log.parent.name]:.1f}%")
if scores:
    avg = sum(scores.values()) / len(scores)
    print(f"AVG ({len(scores)} tasks): {avg:.2f}%")
    (root / "avg_score.txt").write_text(f"{avg:.4f}\n")
else:
    print("No success rates parsed")
    (root / "avg_score.txt").write_text("0\n")
PY

echo "[eval] done -> ${OUT_DIR}"
