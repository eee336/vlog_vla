# Event-balanced latent-option refactor

This change addresses duration bias: uniformly sampled timesteps overrepresent
long arm-motion phases and can make both the VQ codebook and the U2 router
collapse onto a few frequent codes.

## Executable model change

The paper-level latent variable remains

\[
q_\phi(z_t\mid h_t,A_{t:t+H-1}),\qquad
p_\psi(z_t\mid h_t),\qquad
\pi_\theta(A_t\mid h_t,z_t).
\]

The future-action posterior now summarizes encoded action tokens with
`mean`, `std`, endpoint difference and maximum adjacent change. The original
mean projection is checkpoint-compatible; the new dynamics projection is
zero-initialized.

For U1 only, native action chunks are assigned to coarse event strata using
temporal change in configured action-dimension groups. These strata are not
Option labels. They produce mixed inverse-frequency sample weights

\[
w_i=(1-\eta)+\eta\,
\frac{(n_{g_i}+1)^{-1/2}}
{\mathbb E_j[(n_{g_j}+1)^{-1/2}]},
\]

used by VQ, commitment, motion reconstruction and differentiable usage
statistics. Code usage is constrained by an entropy floor instead of KL to a
uniform distribution:

\[
\mathcal L_{\mathrm{usage}}=
\left[\rho_H\log K-H(\bar q)\right]_+^2.
\]

U2 keeps U1 frozen and applies the same inverse-square-root idea to router CE
using posterior Option frequencies accumulated across training.

## Checkpoint compatibility

- Old Qwen/GR00T and UniversalVLOG parameter names are preserved.
- `vlog_core.posterior.dynamics_proj.weight` and balancing-count buffers are
  new keys. Loading an old checkpoint with the existing `strict=False`
  warm-start path initializes them safely.
- Old U2 checkpoints are not evidence for the repaired Gate R because their
  router was trained without balanced CE.
- Event count and router count buffers are checkpointed, so resumed runs retain
  the weighting history.

## Recommended repair sequence

Use the passed U1-repair step 6000 checkpoint as the warm start for a short
event-balanced U1 continuation. Do not reinitialize the conditioner:

```bash
cd /home/nlk/project/vlog_vla
git pull --ff-only origin codex/qwen-universal-vlog-v2

BASE_CKPT=results/Checkpoints/qwen_universal_vlog_robocasa_u1_option_repair/checkpoints/steps_6000_pytorch_model.pt \
RUN_ID=qwen_universal_vlog_robocasa_u1_event_balanced \
TRAIN_STAGE=u1_oracle \
REINITIALIZE_MODULES= \
MAX_STEPS=6000 BATCH_SIZE=16 GRAD_ACCUM=1 \
SAVE_INTERVAL=1000 EVAL_INTERVAL=1000000 NUM_WARMUP_STEPS=300 \
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_qwen_universal_vlog.sh
```

Run Gate O on every checkpoint. An Option is now effective only at at least 1%
usage, and the gate also requires effective Option number and event-alignment
NMI above a shuffled baseline:

```bash
mkdir -p outputs/universal_gates/u1_event_balanced
for STEP in 1000 2000 3000 4000 5000 6000; do
  python scripts/vlog_vla/evaluate_universal_gates.py \
    --checkpoint results/Checkpoints/qwen_universal_vlog_robocasa_u1_event_balanced/checkpoints/steps_${STEP}_pytorch_model.pt \
    --output outputs/universal_gates/u1_event_balanced/steps_${STEP}.json \
    --batch-size 16 --num-batches 32 --fm-repeats 2 \
    --min-option-share 0.01 --min-effective-options 3 \
    --min-event-option-nmi-gap 0.02 --device cuda
done
```

Select the earliest stable Gate O PASS, then train U2 with no reinitialization:

```bash
BASE_CKPT=/absolute/path/to/selected_u1_event_balanced_checkpoint.pt \
RUN_ID=qwen_universal_vlog_robocasa_u2_router_balanced \
TRAIN_STAGE=u2_router \
REINITIALIZE_MODULES= \
MAX_STEPS=6000 BATCH_SIZE=16 GRAD_ACCUM=1 \
SAVE_INTERVAL=1000 EVAL_INTERVAL=1000000 NUM_WARMUP_STEPS=300 \
bash examples/simBenchmarks/Robocasa_tabletop/train_files/run_qwen_universal_vlog.sh
```

Do not enter joint tuning or RL until Gate R passes with:

- at least four Options at 1% usage;
- effective Option number at least 3;
- dominant share at most 0.80;
- router FM degradation at most 2%;
- router-to-posterior FM gap at most 10%.

## Required server verification

```bash
STARVLA_USE_DEEPSPEED=0 pytest -q \
  tests/test_qwen_universal_vlog_modules.py

python -m py_compile \
  starVLA/model/vlog_vla/temporal_option_discovery.py \
  starVLA/model/vlog_vla/qwen_universal_vlog.py \
  scripts/vlog_vla/evaluate_universal_gates.py
```

The local source change does not itself establish rollout improvement. Gate O/R
remain offline E2 evidence; simulator rollout is still required after Gate R.
