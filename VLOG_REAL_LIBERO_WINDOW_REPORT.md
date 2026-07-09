# VLOG-VLA Real LIBERO Window Progress Report

## Status

The VLOG-VLA pipeline has advanced beyond synthetic smoke batches to real LIBERO-Spatial demonstration trajectory windows.

Data source:

- `data/starvla_lerobot_standard_libero_spatial/train.jsonl`
- 2304 total rows in the dataset sanity report
- state dimension: 8
- action dimension: 7
- wrist image available
- task text includes the LIBERO-Spatial bowl placement instruction

Current real-window runs:

- `outputs/vlog_real_stage2/`
- `outputs/vlog_real_stage3/`
- `outputs/vlog_real_stage4/`

## Real-Window Implementation

Added:

- `starVLA/model/vlog_vla/dataset.py`
- `VLOGJsonlWindowDataset`

The dataset reads StarVLA/LIBERO JSONL rows and returns:

- real robot state
- real future action windows
- scalar reward/done proxy
- deterministic hidden-token features derived from real state/action context

The current hidden tokens are not Qwen hidden states yet. They are deterministic trajectory-conditioned features used to validate VLOG training logic on real demonstration windows.

## Latent Option Bootstrap

The first real Stage 2 run collapsed to one option. To avoid reporting an invalid latent discovery result, the real-demo path now supports automatic trajectory-derived bootstrap option targets:

- no manual grasp/place/move labels
- no language option labels
- bins are derived from dominant future action motion axis, sign, and gripper trend

Config flag:

```yaml
training:
  use_action_window_pseudo_options: true
```

## Results

Stage 2 real option usage:

```json
{
  "num_options": 16,
  "usage_per_option": [30, 9, 19, 117, 6, 12, 91, 23, 131, 105, 65, 32, 0, 0, 0, 0],
  "dead_options": [12, 13, 14, 15],
  "entropy": 2.1370010375976562
}
```

Effective options used: 12 / 16.

Stage 3 duration:

```json
{
  "mean_duration": 11.851851851851851
}
```

Stage 3 graph sparsity:

```json
{
  "num_edges": 256,
  "observed_nonzero_edges": 43,
  "observed_sparsity": 0.83203125
}
```

Stage 4 option critic:

```json
{
  "td_loss": 0.0001363605697406456,
  "cql_loss": 2.75528883934021,
  "q_data": 1.7689435482025146,
  "q_all": 1.751527190208435,
  "target_q_mean": 1.7614307403564453
}
```

## Verification

Executed:

```bash
conda run -n starVLA pytest tests/test_vlog_vla_modules.py tests/test_vlog_jsonl_dataset.py -q
```

Result:

```text
6 passed
```

## Remaining Work

The next required step is full StarVLA integration:

1. Extract real Qwen/StarVLA hidden tokens from the pretrained LIBERO checkpoint.
2. Replace deterministic trajectory-conditioned hidden features with actual VLM hidden tokens.
3. Run Stage 1 safe insertion using real base StarVLA action outputs.
4. Bind `eval_vlog_libero.py` to the official StarVLA server-client LIBERO evaluation path.
5. Run online LIBERO eval for Base, Adapter-only, Codebook, Graph, Critic, Persistence, and Full VLOG ablations.
