# VLOG-VLA Git Upload Details

## Scope

This upload contains the key VLOG-VLA implementation and integration code only.
It intentionally excludes old Act-to-see, SkillQ, PlainBC, and OptionGraph experiment leftovers that are still present in the working tree.

## Branch

Recommended branch:

```text
feature/vlog-vla-stage7-real-starvla
```

## Core Implementation

### VLOG Modules

Directory:

```text
starVLA/model/vlog_vla/
```

Key files:

- `state_aggregator.py`: pools StarVLA/Qwen hidden tokens into compact state features.
- `latent_option_codebook.py`: straight-through VQ latent option codebook.
- `posterior_option_encoder.py`: training-time future action window encoder.
- `option_graph_layer.py`: state-conditioned latent option graph.
- `persistent_option_router.py`: differentiable router over latent options.
- `option_critic.py`: option-level critic `Q(s,o)`, not `Q(s,a)`.
- `termination_head.py`: predicts `beta(s,o)`.
- `option_adapter.py`: zero-initialized hidden-token adapter.
- `vlog_policy_wrapper.py`: persistence, value-drop switching, and stage forward logic.
- `dataset.py`: real LIBERO JSONL state/action window dataset.
- `starvla_hidden_adapter.py`: extracts real StarVLA/Qwen hidden tokens.
- `real_starvla_vlog_wrapper.py`: wraps a real StarVLA policy with VLOG.
- `qwen_oft_vlog.py`: QwenOFT subclass with VLOG inserted before action decoding.

### Framework Bridge

```text
starVLA/model/framework/VLM4A/QwenOFTVLOG.py
```

Registers `QwenOFTVLOG` into the existing StarVLA framework registry.

### Training and Eval Scripts

Directory:

```text
scripts/vlog_vla/
```

Important scripts:

- `train_common.py`: stage 1-6 common smoke/real-window trainer.
- `stage7_common.py`: real hidden-token Stage 7 training utilities.
- `train_vlog_stage1.py` ... `train_vlog_stage6.py`: stage smoke entrypoints.
- `train_vlog_stage1_real_hidden.py` ... `train_vlog_stage4_real_hidden.py`: real StarVLA hidden-token training entrypoints.
- `probe_starvla_hidden_tokens.py`: hidden-token extraction probe.
- `analyze_options.py`: usage, duration, transition, timeline, and Q/beta analysis.
- `eval_vlog_libero.py`: VLOG eval wrapper surface.

### Configs

Directory:

```text
configs/vlog_vla/
```

Includes:

- Stage 1-6 smoke configs.
- Real LIBERO JSONL configs.
- Real StarVLA hidden-token Stage 7 config.
- Ablation configs A0-A6.

### Official Server Compatibility

Modified:

- `deployment/model_server/policy_wrapper.py`
- `deployment/model_server/server_policy.py`

Purpose:

- Pass `vlog_info` through official server responses.
- Support optional VLOG logging without breaking the base policy server path.

### StarVLA Trainer Compatibility

Modified:

- `starVLA/training/train_starvla.py`
- `starVLA/training/trainer_utils/trainer_tools.py`

Purpose:

- Log scalar VLOG losses to `train_log.jsonl`.
- Avoid duplicate optimizer parameter groups when adding VLOG module-specific learning rates.

### LIBERO Train/Eval Examples

Added:

- `examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml`
- `examples/LIBERO/train_files/run_vlog_libero_train*.sh`
- `examples/LIBERO/eval_files/run_vlog_policy_server.sh`
- `examples/LIBERO/eval_files/eval_vlog_libero.sh`

## Reports

Included Markdown reports:

- `VLOG_VLA_IMPLEMENTATION_REPORT.md`
- `VLOG_REAL_LIBERO_WINDOW_REPORT.md`
- `VLOG_VLA_GIT_UPLOAD_DETAILS.md`

## Validation

Executed before upload:

```bash
conda run -n starVLA pytest \
  tests/test_vlog_vla_modules.py \
  tests/test_vlog_jsonl_dataset.py \
  tests/test_vlog_hidden_hook.py \
  tests/test_vlog_no_surrogate_hidden_stage7.py \
  tests/test_vlog_real_starvla_wrapper.py \
  tests/test_vlog_starvla_real_integration.py \
  tests/test_vlog_official_eval_interface.py \
  -q
```

Result:

```text
16 passed
```

## Current Capability

The committed code supports:

1. VLOG stage 1-6 module smoke workflow.
2. Real LIBERO JSONL trajectory-window training for Stage 1/2.
3. Real StarVLA/Qwen hidden-token extraction hooks for Stage 7.
4. `QwenOFTVLOG` framework registration.
5. Persistent option inference with `d_min`, `d_max`, beta threshold, and value-drop switching.
6. Option-level conservative critic `Q(s,o)`.
7. Official policy-server-compatible `vlog_info` passthrough.

## Current Limitations

- Full official LIBERO simulator success-rate evaluation for VLOG has not yet been run end to end.
- Real hidden-token Stage 7 is wired and tested at interface level, but still needs longer training and official rollout.
- Old non-VLOG experiment files remain in the working tree but are not part of this upload.

## Suggested Next Commands

```bash
# Real hidden-token stage smoke
python scripts/vlog_vla/probe_starvla_hidden_tokens.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml

python scripts/vlog_vla/train_vlog_stage1_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml

# Official eval surface
bash examples/LIBERO/eval_files/run_vlog_policy_server.sh
bash examples/LIBERO/eval_files/eval_vlog_libero.sh
```
