# New Machine Setup Guide for VLOG-VLA

This guide assumes the new machine already has the base StarVLA project and environment installed. The goal is to apply the VLOG-VLA code from Git and verify that the local checkpoint, dataset, smoke tests, real-data pipeline, and official eval entrypoints are configured correctly.

Repository:

```text
https://github.com/eee336/vlog_vla.git
branch: feature/vlog-vla-stage7-real-starvla
```

## 0. Expected Existing Setup

You should already have:

```text
1. A working StarVLA repo.
2. A conda env, usually named starVLA.
3. CUDA/PyTorch/transformers dependencies required by StarVLA.
4. LIBERO dependencies if you want official simulator eval.
5. Local StarVLA pretrained checkpoint and Qwen base model.
```

The VLOG repo is a clean upload containing the VLOG-related files. It is not a full StarVLA mirror. You should apply/copy it onto your StarVLA repo.

Recommended base StarVLA path:

```bash
cd /path/to/starVLA
```

## 1. Fetch VLOG-VLA Code from Git

Use a temporary directory:

```bash
cd /tmp
git clone -b feature/vlog-vla-stage7-real-starvla https://github.com/eee336/vlog_vla.git
```

Copy the VLOG files into your StarVLA repo:

```bash
cd /path/to/starVLA
rsync -av /tmp/vlog_vla/ ./
```

Alternatively, if you want to inspect first:

```bash
cd /tmp/vlog_vla
find . -maxdepth 3 -type f | sort
```

## 2. Confirm Required Paths

From your StarVLA repo:

```bash
cd /path/to/starVLA
```

Check VLOG files:

```bash
test -d starVLA/model/vlog_vla
test -f starVLA/model/framework/VLM4A/QwenOFTVLOG.py
test -f configs/vlog_vla/real_starvla_hidden_stage7.yaml
test -f scripts/vlog_vla/probe_starvla_hidden_tokens.py
```

Check StarVLA checkpoint. The default configs expect:

```text
playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints/steps_50000_pytorch_model.pt
playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/config.yaml
playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

Commands:

```bash
ls playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1
ls playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1/checkpoints
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

If your paths are different, edit:

```text
configs/vlog_vla/real_starvla_hidden_stage7.yaml
examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
examples/LIBERO/eval_files/run_vlog_policy_server.sh
```

## 3. Confirm Dataset

For real-data Stage 1/2, the default path is:

```text
data/starvla_lerobot_standard_libero_spatial/train.jsonl
```

Check:

```bash
ls data/starvla_lerobot_standard_libero_spatial
head -n 1 data/starvla_lerobot_standard_libero_spatial/train.jsonl
```

Expected JSONL fields include:

```text
episode_id
step_id
lang
image
wrist_image
state
action_chunk
```

If missing, either copy the converted dataset from the old machine or rebuild it using your existing StarVLA/LIBERO conversion script.

Then update these configs if needed:

```text
configs/vlog_vla/libero_vlog_vla_stage1_realdata.yaml
configs/vlog_vla/libero_vlog_vla_stage2_realdata.yaml
configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

## 4. Environment Check

Activate env:

```bash
conda activate starVLA
```

Check Python import:

```bash
python - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```

Check VLOG imports:

```bash
python - <<'PY'
from starVLA.model.vlog_vla import VLOGPolicyWrapper, OptionCritic
from starVLA.model.framework.VLM4A.QwenOFTVLOG import QwenOFTVLOG
print("VLOG imports OK")
PY
```

## 5. Run Unit Tests

Run the complete uploaded VLOG test set:

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

Expected:

```text
16 passed
```

If tests fail because the base StarVLA repo layout differs, first check whether:

```text
starVLA/model/framework/VLM4A/QwenOFT.py
deployment/model_server/policy_wrapper.py
deployment/model_server/server_policy.py
```

exist and match the expected StarVLA structure.

## 6. Run Stage 1-6 Smoke Tests

These are lightweight shape/training-path checks and do not require simulator rollout.

```bash
python scripts/vlog_vla/train_vlog_stage1.py --config configs/vlog_vla/libero_vlog_vla_stage1.yaml
python scripts/vlog_vla/train_vlog_stage2.py --config configs/vlog_vla/libero_vlog_vla_stage2.yaml
python scripts/vlog_vla/train_vlog_stage3.py --config configs/vlog_vla/libero_vlog_vla_stage3.yaml
python scripts/vlog_vla/train_vlog_stage4.py --config configs/vlog_vla/libero_vlog_vla_stage4.yaml
python scripts/vlog_vla/train_vlog_stage5.py --config configs/vlog_vla/libero_vlog_vla_stage5.yaml
python scripts/vlog_vla/train_vlog_stage6.py --config configs/vlog_vla/libero_vlog_vla_stage6.yaml
```

Expected outputs:

```text
outputs/vlog_stage1/action_diff_report.json
outputs/vlog_stage2/option_usage.json
outputs/vlog_stage3/option_transition_matrix.json
outputs/vlog_stage4/critic_report.json
outputs/vlog_stage5/router_advantage_report.json
outputs/vlog_stage6/termination_report.json
```

Check:

```bash
cat outputs/vlog_stage1/action_diff_report.json
```

Expected safe insertion:

```json
{
  "alpha_value": 0.0,
  "max_l1_action_diff": 0.0,
  "mean_l1_action_diff": 0.0
}
```

## 7. Run Real LIBERO JSONL Stage 1/2

These use real state/action windows from `train.jsonl`.

```bash
python scripts/vlog_vla/train_vlog_stage1.py \
  --config configs/vlog_vla/libero_vlog_vla_stage1_realdata.yaml

python scripts/vlog_vla/train_vlog_stage2.py \
  --config configs/vlog_vla/libero_vlog_vla_stage2_realdata.yaml
```

Expected outputs:

```text
outputs/vlog_stage1_realdata/action_diff_report.json
outputs/vlog_stage2_realdata/option_usage.json
outputs/vlog_stage2_realdata/router_distill_accuracy.json
```

Check:

```bash
cat outputs/vlog_stage1_realdata/action_diff_report.json
cat outputs/vlog_stage2_realdata/option_usage.json
```

Notes:

```text
Stage 1 should preserve base outputs because OptionAdapter alpha is zero.
Stage 2 uses trajectory-derived action-window bootstrap, not manual skill labels.
```

## 8. Probe Real StarVLA Hidden Tokens

This checks whether the real QwenOFT forward path can expose hidden tokens for VLOG.

```bash
python scripts/vlog_vla/probe_starvla_hidden_tokens.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

Expected output directory:

```text
outputs/vlog_stage7_real_starvla/inspect_starvla_forward/
```

Key files:

```text
inspect_report.json
hidden_token_probe_report.json
```

If this fails, check:

```text
1. Qwen3-VL model path.
2. StarVLA checkpoint path.
3. CUDA memory.
4. Whether QwenOFT.py still exposes hidden_states with output_hidden_states=True.
```

## 9. Run Real Hidden Stage 1/2

After hidden-token probe passes:

```bash
python scripts/vlog_vla/train_vlog_stage1_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml

python scripts/vlog_vla/train_vlog_stage2_real_hidden.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

Expected outputs:

```text
outputs/vlog_stage7_real_starvla/stage1_real_hidden/
outputs/vlog_stage7_real_starvla/stage2_real_hidden/
```

Stage 1 key report:

```text
real_hidden_action_diff_report.json
```

Stage 2 key reports:

```text
option_usage_real_hidden.json
router_distill_report_real_hidden.json
option_duration_histogram_real_hidden.json
```

## 10. Official LIBERO Eval Surface

Server:

```bash
bash examples/LIBERO/eval_files/run_vlog_policy_server.sh
```

Client:

```bash
bash examples/LIBERO/eval_files/eval_vlog_libero.sh
```

If your StarVLA repo uses different official eval scripts, keep the official server-client protocol and only point the server to `QwenOFTVLOG`.

The server wrapper supports `vlog_info` passthrough so option timelines can be logged without breaking normal action responses.

## 11. Important Config Values

In `configs/vlog_vla/real_starvla_hidden_stage7.yaml`, check:

```yaml
base_checkpoint: playground/Pretrained_models/StarVLA_Qwen3_VL_OFT_LIBERO_4in1
benchmark: libero_spatial
vlog:
  num_options: 16
  option_dim: 256
  d_min: 2
  d_max: 8
  beta_threshold: 0.7
  q_switch_margin: 0.05
```

In `examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml`, check:

```yaml
framework:
  name: QwenOFTVLOG
```

## 12. What Not To Change

Do not change:

```text
1. Critic into Q(s,a). It must remain Q(s,o).
2. Options into manual labels such as grasp/place/move.
3. Options into language/text tokens.
4. Official eval protocol into custom rollout.
5. Base StarVLA checkpoint loading logic unless absolutely necessary.
```

## 13. Common Problems

### Problem: Git repo is not full StarVLA

This upload is a VLOG code overlay. Apply it on top of your installed StarVLA repo.

### Problem: Qwen path missing

Edit:

```text
configs/vlog_vla/real_starvla_hidden_stage7.yaml
examples/LIBERO/train_files/starvla_qwen_oft_vlog_libero.yaml
```

### Problem: Dataset missing

Copy or rebuild:

```text
data/starvla_lerobot_standard_libero_spatial/
```

### Problem: CUDA OOM during hidden-token probe

Try:

```text
1. batch size 1
2. bf16
3. shorter action window
4. smaller max samples
```

### Problem: official eval does not show option logs

Check server flags and log directory in:

```text
examples/LIBERO/eval_files/run_vlog_policy_server.sh
```

## 14. Minimal Success Checklist

Before running long experiments, verify:

```bash
conda run -n starVLA pytest tests/test_vlog_vla_modules.py -q

python scripts/vlog_vla/train_vlog_stage1.py \
  --config configs/vlog_vla/libero_vlog_vla_stage1.yaml

python scripts/vlog_vla/train_vlog_stage1.py \
  --config configs/vlog_vla/libero_vlog_vla_stage1_realdata.yaml

python scripts/vlog_vla/probe_starvla_hidden_tokens.py \
  --config configs/vlog_vla/real_starvla_hidden_stage7.yaml
```

If all pass, continue to real hidden Stage 1/2 and then official eval.

