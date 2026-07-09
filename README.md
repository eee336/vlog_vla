# VLOG-VLA

Persistent Value-Guided Latent Option Graphs for StarVLA.

This repository is a clean VLOG-VLA overlay for an already working StarVLA setup. It is intended for the workflow where a new machine already has the StarVLA environment, Qwen/StarVLA checkpoint, and LIBERO dependencies installed. After cloning this repo, you can install it with `pip`, run the VLOG unit tests, run synthetic smoke experiments, and then point the config to the real StarVLA checkpoint/LIBERO JSONL data for the real hidden-token experiment.

## Repository Layout

```text
starVLA/model/vlog_vla/                 Core VLOG modules
starVLA/model/framework/VLM4A/          QwenOFTVLOG integration shim
scripts/vlog_vla/                       Training, probing, and analysis scripts
configs/vlog_vla/                       Stage 1-7 experiment configs
deployment/model_server/                Policy-server compatibility wrapper
examples/LIBERO/                        Example train/eval shell entrypoints
tests/                                  VLOG tests
tools/                                  Dataset/path helper tools
```

## Model Structure

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

The VLOG option layer uses latent option codes discovered from trajectories. Options are not manual skill labels, text subgoals, or language tokens.

## New Machine Setup

The commands below assume:

- Your conda environment is named `vlog_vla`.
- StarVLA itself has already been installed and tested on the machine.
- `torch` is already installed in the StarVLA-compatible CUDA version for this machine.
- You are cloning this repo as the VLOG experiment codebase.

Clone:

```bash
git clone -b feature/vlog-vla-stage7-real-starvla https://github.com/eee336/vlog_vla.git
cd vlog_vla
```

Install:

```bash
conda activate vlog_vla
pip install -r requirements.txt
pip install -e .
```

This repo intentionally does not install `torch`. Keep using the torch/CUDA build that already works for StarVLA. If you only need editable install after dependencies are present:

```bash
pip install -e ".[dev]"
```

## Basic Verification

Run the VLOG tests:

```bash
pytest \
  tests/test_vlog_vla_modules.py \
  tests/test_vlog_jsonl_dataset.py \
  tests/test_vlog_hidden_hook.py \
  tests/test_vlog_no_surrogate_hidden_stage7.py \
  tests/test_vlog_real_starvla_wrapper.py \
  tests/test_vlog_starvla_real_integration.py \
  tests/test_vlog_official_eval_interface.py \
  -q
```

Run a minimal smoke experiment:

```bash
python scripts/vlog_vla/train_vlog_stage1.py \
  --config configs/vlog_vla/libero_vlog_vla_stage1.yaml
```

Check outputs:

```bash
find outputs -maxdepth 3 -type f | sort
```

## Real StarVLA/LIBERO Paths

For the real hidden-token experiment, edit:

```bash
nano configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

Set these fields to paths that exist on the new machine:

```yaml
base:
  checkpoint: /ABS/PATH/TO/steps_50000_pytorch_model.pt
  model_dir: /ABS/PATH/TO/StarVLA_Qwen3_VL_OFT_LIBERO_4in1

dataset:
  jsonl_path: /ABS/PATH/TO/starvla_lerobot_standard_libero_spatial/train.jsonl
```

Common expected paths from the original machine were:

```text
playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt
playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/
data/starvla_lerobot_standard_libero_spatial/train.jsonl
```

Verify your paths:

```bash
ls /ABS/PATH/TO/steps_50000_pytorch_model.pt
ls /ABS/PATH/TO/StarVLA_Qwen3_VL_OFT_LIBERO_4in1
ls /ABS/PATH/TO/starvla_lerobot_standard_libero_spatial/train.jsonl
```

## Run Experiments

Stage 1 synthetic smoke:

```bash
python scripts/vlog_vla/train_vlog_stage1.py \
  --config configs/vlog_vla/libero_vlog_vla_stage1.yaml
```

Stage 2 synthetic smoke:

```bash
python scripts/vlog_vla/train_vlog_stage2.py \
  --config configs/vlog_vla/libero_vlog_vla_stage2.yaml
```

Stage 1/2 with real LIBERO JSONL state/action windows:

```bash
python scripts/vlog_vla/train_vlog_stage1.py \
  --config configs/vlog_vla/libero_vlog_vla_stage1_realdata.yaml

python scripts/vlog_vla/train_vlog_stage2.py \
  --config configs/vlog_vla/libero_vlog_vla_stage2_realdata.yaml
```

Probe real StarVLA hidden tokens:

```bash
python scripts/vlog_vla/probe_starvla_hidden_tokens.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

Run real-hidden VLOG Stage 1:

```bash
python scripts/vlog_vla/train_vlog_stage1_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

Run real-hidden VLOG Stage 2:

```bash
python scripts/vlog_vla/train_vlog_stage2_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

## Output Files

Default outputs are written under:

```text
outputs/
```

Useful checks:

```bash
find outputs -maxdepth 4 -type f | sort | tail -80
cat outputs/vlog_stage1/train_log.jsonl
```

## Optional Overlay Into an Existing StarVLA Tree

If you prefer to keep using an existing full StarVLA checkout, clone this repo next to it and overlay the VLOG files:

```bash
cd /PATH/TO/project
cp -a starvla starvla_backup_before_vlog
rsync -av --exclude='.git' vlog_vla/ starvla/
cd starvla
pip install -e /PATH/TO/project/vlog_vla
```

Then run the same commands from the full StarVLA directory.

## Official Eval Surface

Policy server:

```bash
bash examples/LIBERO/eval_files/run_vlog_policy_server.sh
```

LIBERO eval client:

```bash
bash examples/LIBERO/eval_files/eval_vlog_libero.sh
```

The policy server remains backward-compatible with normal StarVLA responses and can pass through `vlog_info` for option timeline logging.

## Troubleshooting

If `ModuleNotFoundError: starVLA...` appears, confirm installation:

```bash
python -c "import starVLA.model.vlog_vla as v; print(v.__file__)"
```

If real hidden-token scripts cannot import full StarVLA framework modules, run from the full StarVLA checkout and install this repo as an overlay:

```bash
pip install -e /PATH/TO/vlog_vla
export PYTHONPATH=/PATH/TO/full/starvla:$PYTHONPATH
```

If GitHub clone is unstable, use a shallow clone:

```bash
git clone --depth 1 -b feature/vlog-vla-stage7-real-starvla https://github.com/eee336/vlog_vla.git
```

## Notes

- Current visual-language backbone is StarVLA/QwenOFT, not DINOv2.
- The option critic is option-level: `Q(s, o)`, not `Q(s, a)`.
- Smoke configs are intentionally small; increase `training.train_steps` for real runs.
- Checkpoints, datasets, and generated outputs are intentionally ignored by Git.
