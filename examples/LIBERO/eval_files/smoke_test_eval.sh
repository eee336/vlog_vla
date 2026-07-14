#!/usr/bin/env bash
set -euo pipefail
PY=/home/nlk/.conda/envs/vlog_vla/bin/python
REPO=/home/nlk/project/vlog_vla
LIBERO_HOME=$REPO/playground/Code/LIBERO
cd "$REPO"
export PYTHONPATH=$REPO:$LIBERO_HOME
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

"$PY" -c "from libero.libero import benchmark; b=benchmark.get_benchmark_dict()['libero_spatial'](); s=b.get_task_init_states(0); print('init states ok', len(s))"

CKPT=$REPO/playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt
OUT=$REPO/outputs/libero_stage_eval_smoke/starvla_pretrained/libero_spatial2
PORT=5695
mkdir -p "$OUT/videos"

pkill -f "server_policy.py.*${PORT}" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=0 "$PY" deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" --port "$PORT" --use_bf16 > /tmp/policy_smoke2.log 2>&1 &
SPID=$!

for i in $(seq 1 120); do
  if grep -q "server listening" /tmp/policy_smoke2.log 2>/dev/null; then break; fi
  sleep 2
done
if ! grep -q "server listening" /tmp/policy_smoke2.log 2>/dev/null; then
  echo "policy server failed to start" >&2
  tail -30 /tmp/policy_smoke2.log >&2 || true
  kill "$SPID" 2>/dev/null || true
  exit 1
fi
sleep 5

STARVLA_DIR="$REPO" LIBERO_HOME="$LIBERO_HOME" LIBERO_PYTHON="$PY" \
CKPT="$CKPT" PORT="$PORT" TASK_SUITE_NAME=libero_spatial NUM_TRIALS_PER_TASK=1 MAX_TASKS=1 \
UNNORM_KEY=franka OUTPUT_DIR="$OUT" \
bash examples/LIBERO/eval_files/eval_vlog_libero.sh \
  --suite libero_spatial --num_trials_per_task 1 --max_tasks 1 --output_dir "$OUT" \
  2>&1 | tee "$OUT/eval.log"

kill "$SPID" 2>/dev/null || true
wait "$SPID" 2>/dev/null || true
grep -E "Total success|Total episodes|Traceback" "$OUT/eval.log" | tail -5
ls "$OUT/videos/" | head -3
