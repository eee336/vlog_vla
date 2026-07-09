# VLOG-VLA

Persistent Value-Guided Latent Option Graphs for Vision-Language-Action Models.

This repository contains the VLOG-VLA prototype integrated with StarVLA/QwenOFT. VLOG-VLA inserts a persistent latent option graph between pretrained VLA hidden tokens and the action head. Options are automatically discovered latent codes from demonstration trajectories, not manually named skills, language subgoals, or text tokens.

## What Is Included

- Core VLOG modules in `starVLA/model/vlog_vla/`
- QwenOFT integration via `QwenOFTVLOG`
- Real LIBERO JSONL trajectory-window loaders
- Real StarVLA hidden-token adapter and Stage 7 integration surface
- Stage 1-6 smoke training scripts
- Stage 7 real-hidden training/probing scripts
- Official policy-server compatibility hooks for `vlog_info`
- Tests for module shapes, option-level critic, hidden extraction, no-surrogate Stage 7, and official eval interface
- Detailed implementation reports

## Key Design

```text
image + instruction + robot state
        |
        v
Pretrained StarVLA / QwenOFT backbone
        |
        v
hidden tokens H_t
        |
        v
VLOG latent option graph
        |
        v
zero-initialized option adapter
        |
        v
adapted hidden tokens H'_t
        |
        v
original StarVLA action head
        |
        v
action chunk
```

VLOG components:

- `StateAggregator`: pools VLA hidden tokens into compact state features.
- `LatentOptionCodebook`: VQ latent option codes.
- `PosteriorOptionEncoder`: uses future action windows during training only.
- `OptionGraphLayer`: state-conditioned option transition graph.
- `PersistentOptionRouter`: routes over latent options.
- `OptionCritic`: learns `Q(s,o)` only, not `Q(s,a)`.
- `TerminationHead`: predicts option termination probability `beta(s,o)`.
- `OptionAdapter`: zero-initialized adapter so initial insertion preserves base behavior.

## Current Status

Implemented and tested:

- Stage 1-6 VLOG smoke workflow
- Real LIBERO-Spatial JSONL state/action window training for Stage 1/2
- Stage 7 real StarVLA hidden-token extraction interface
- `QwenOFTVLOG` registry bridge
- Official policy-server `vlog_info` passthrough

Validation before upload:

```text
16 passed
```

Test command:

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

## Important Files

Core modules:

```text
starVLA/model/vlog_vla/
starVLA/model/framework/VLM4A/QwenOFTVLOG.py
```

Configs:

```text
configs/vlog_vla/
```

Scripts:

```text
scripts/vlog_vla/
```

LIBERO examples:

```text
examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
examples/LIBERO/train_files/run_vlog_libero_train*.sh
examples/LIBERO/eval_files/run_vlog_policy_server.sh
examples/LIBERO/eval_files/eval_vlog_libero.sh
```

Reports:

```text
VLOG_VLA_IMPLEMENTATION_REPORT.md
VLOG_REAL_LIBERO_WINDOW_REPORT.md
VLOG_VLA_GIT_UPLOAD_DETAILS.md
```

## Quick Start

Run module tests:

```bash
conda run -n starVLA pytest tests/test_vlog_vla_modules.py -q
```

Run all uploaded VLOG tests:

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

Run Stage 1-6 smoke training:

```bash
python scripts/vlog_vla/train_vlog_stage1.py --config configs/vlog_vla/libero_vlog_vla_stage1.yaml
python scripts/vlog_vla/train_vlog_stage2.py --config configs/vlog_vla/libero_vlog_vla_stage2.yaml
python scripts/vlog_vla/train_vlog_stage3.py --config configs/vlog_vla/libero_vlog_vla_stage3.yaml
python scripts/vlog_vla/train_vlog_stage4.py --config configs/vlog_vla/libero_vlog_vla_stage4.yaml
python scripts/vlog_vla/train_vlog_stage5.py --config configs/vlog_vla/libero_vlog_vla_stage5.yaml
python scripts/vlog_vla/train_vlog_stage6.py --config configs/vlog_vla/libero_vlog_vla_stage6.yaml
```

Run real LIBERO JSONL Stage 1/2:

```bash
python scripts/vlog_vla/train_vlog_stage1.py --config configs/vlog_vla/libero_vlog_vla_stage1_realdata.yaml
python scripts/vlog_vla/train_vlog_stage2.py --config configs/vlog_vla/libero_vlog_vla_stage2_realdata.yaml
```

Probe real StarVLA hidden-token integration:

```bash
python scripts/vlog_vla/probe_starvla_hidden_tokens.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

Run real-hidden Stage 7 scripts:

```bash
python scripts/vlog_vla/train_vlog_stage1_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml

python scripts/vlog_vla/train_vlog_stage2_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

## Official Eval Surface

Server:

```bash
bash examples/LIBERO/eval_files/run_vlog_policy_server.sh
```

Client:

```bash
bash examples/LIBERO/eval_files/eval_vlog_libero.sh
```

The policy server is backward-compatible with normal StarVLA responses and can pass through `vlog_info` for option timeline logging.

## Notes

- The current baseline visual-language encoder is Qwen3-VL through StarVLA `QwenOFT`, not DINOv2.
- DINOv2 exists in another StarVLA branch (`QwenDual`) but is not the selected VLOG baseline here.
- The critic is option-level only: `Q(s,o)`.
- Options are latent codebook entries discovered from trajectories, not manual skill labels.
- Full official LIBERO simulator success-rate evaluation for VLOG still needs a longer run.

## Limitations

- Stage 1-6 smoke configs are lightweight by design.
- Real LIBERO JSONL Stage 1/2 uses real state/action windows, but full visual hidden-token training is represented by Stage 7 scripts.
- Official rollout has an interface wrapper but still needs full benchmark execution and result reporting.

