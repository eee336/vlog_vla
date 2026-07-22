# QwenUniversalVLOG source audit

This audit records the implementation boundaries used by the replacement
UniversalVLOG framework.  It describes executable paths in this checkout, not
claims inferred from names.

## Existing base path

`QwenGR00T.forward` builds Qwen inputs from images/instructions, takes the last
Qwen hidden state, repeats the batch according to
`framework.action_model.repeated_diffusion_steps`, and passes continuous state
and normalized actions to `FlowmatchingActionHead`.

The action head boundary is:

```text
native state -> state_encoder --------------------------┐
noise/action/t -> action_encoder -> action tokens ------+-> shared DiT
learned future queries ---------------------------------┘       |
Qwen last hidden -------------------------------- cross-attn    |
                                                              decoder
                                                                |
                                                    native velocity target
```

For the released RoboCasa shape this is state 58D, action 29D, horizon 16.
The base sequence is `[state, future_query, noisy_action]`.  The replacement
framework retains these module names and this exact sequence when fusion is
disabled.

## Checkpoint behavior

The generic trainer calls `model.load_state_dict(..., strict=False)`.  The new
framework overrides that call for a non-Universal checkpoint: every
`qwen_vl_interface` key and every original GR00T state/action/DiT key must be
present with the exact shape.  Missing or shape-mismatched `embed_tokens` is a
hard error.  Only new adapter/VLOG/conditioner keys may be absent.

Therefore the framework must be instantiated with the official checkpoint's
own tokenizer, Qwen config and GR00T action config.  QwenOFT and QwenPI
checkpoints are intentionally rejected by shape/key validation.

## Replacement shared boundary

For non-GR1 embodiments the data flow is now:

```text
native state(e)  -> StateAdapter[e]  -> D_dit tokens
native action(e) -> ActionEncoder[e] -> D_dit tokens
                                      -> one shared GR00T DiT
D_dit output     -> ActionDecoder[e] -> native velocity(e)
```

There is no LIBERO 7D -> GR1 29D projection.  Mixed-native-dimension training
batches are partitioned by `embodiment_id`, processed through the same Qwen and
shared DiT parameters, then reduced to one weighted scalar loss.

## Option path

Training-only posterior input is the current aggregated state plus future
actions *after* the native action encoder.  The router never receives future
actions.  An option enters the action expert as:

1. one explicit option token;
2. residual FiLM on noisy action tokens.

The sequence with fusion is
`[state, embodiment, option, future_query, noisy_action]`.  Correct/base/wrong
FM calls share observations, targets, masks, sampled `t` and sampled noise.

## Stages

- `u0_base`: diagnostic/evaluation only; everything frozen, fusion off.
- `u0_libero`: only non-base native adapters and embodiment embedding train.
- `u1_oracle`: posterior/codebook/aggregator/motion/conditioner train; router,
  Qwen, DiT and native adapters are frozen.
- `u2_router`: only router parameters train.  The conditioner is frozen.
- `u3_joint`: Qwen remains frozen; shared DiT, adapters, conditioner and VLOG
  train.

Graph, critic, learned termination and RL losses are not part of this version.

## Trainer and deployment observations

- The action head reads `repeated_diffusion_steps` from the framework config.
- The trainer now passes its optimizer-step counter through the optional
  `set_optimizer_step` hook so counterfactual frequency is not based on raw
  forward calls.
- Normalization remains an external dataset/deployment concern.  Adapters
  consume and return normalized *native* coordinates and never binarize a
  hard-coded action index.
- `SemiMarkovController` stores independent state by `env_id`, resets when
  `episode_id` changes, observes `d_min`, and performs hysteretic reselection.

## Evidence still required on the training machine

The committed unit tests cover module shapes, causal option sensitivity,
controller isolation/reset, fusion-off action-head parity, paired noise/time,
and action-mask reduction.  The following are deliberately not claimed here:

- official 4B checkpoint strict-load result;
- GPU forward/backward/predict memory and timing;
- normalization round-trip against the actual RoboCasa/LIBERO statistics;
- simulator success or Gate O/R outcomes.

Those require the checkpoint, dataset statistics, CUDA environment and
benchmark runtime used for training.

Run the strict-load/action-parity portion with:

```bash
python scripts/vlog_vla/diagnose_universal_base.py \
  --config examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_qwen_universal_vlog_robocasa.yaml \
  --checkpoint /absolute/path/to/steps_90000_pytorch_model.pt \
  --device cuda
```
