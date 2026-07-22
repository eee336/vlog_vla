#!/usr/bin/env python3
"""Strict-load and fusion-off action-head parity audit for QwenUniversalVLOG.

This script intentionally does not run a simulator or claim rollout quality.
It validates the checkpoint/model tensor contract and compares the replacement
action path with an independently instantiated original GR00T action head under
identical cached conditioning, state, action, t and noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(path, map_location="cpu")


def reference_velocity(
    head: FlowmatchingActionHead,
    vl: torch.Tensor,
    actions: torch.Tensor,
    state: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    t_discrete = (t * head.num_timestep_buckets).long()
    noisy = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
    action_tokens = head.action_encoder(noisy, t_discrete)
    if head.config.add_pos_embed:
        positions = head.position_embedding(
            torch.arange(actions.shape[1], device=actions.device)
        )[None]
        action_tokens = action_tokens + positions
    sequence = torch.cat(
        [
            head.state_encoder(state),
            head.future_tokens.weight[None].expand(actions.shape[0], -1, -1),
            action_tokens,
        ],
        dim=1,
    )
    output = head.model(
        hidden_states=sequence,
        encoder_hidden_states=vl,
        encoder_attention_mask=attention_mask,
        timestep=t_discrete,
    )
    return head.action_decoder(output)[:, -actions.shape[1] :], sequence.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--vl-tokens", type=int, default=8)
    args = parser.parse_args()

    cfg = apply_config_compat(OmegaConf.load(args.config))
    cfg.framework.name = "QwenUniversalVLOG"
    cfg.framework.vlog.enabled = False
    cfg.framework.vlog.fusion_enabled = False
    cfg.framework.vlog.train_stage = "u0_base"
    cfg.trainer.pretrained_checkpoint = None

    checkpoint = load_checkpoint(Path(args.checkpoint))
    universal = build_framework(cfg)
    incompatible = universal.load_state_dict(checkpoint, strict=False)

    reference = FlowmatchingActionHead(full_config=universal.config)
    action_state = {
        key.removeprefix("action_model."): value
        for key, value in checkpoint.items()
        if key.startswith("action_model.")
    }
    reference.load_state_dict(action_state, strict=True)

    device = torch.device(args.device)
    universal.action_model.to(device).eval()
    reference.to(device).eval()
    dtype = next(reference.parameters()).dtype
    torch.manual_seed(args.seed)
    batch = 1
    horizon = universal.action_horizon
    base_spec = universal.embodiment_specs[universal.base_embodiment]
    cross_dim = int(
        universal.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
    )
    vl = torch.randn(batch, args.vl_tokens, cross_dim, device=device, dtype=dtype)
    state = torch.randn(batch, 1, base_spec.state_dim, device=device, dtype=dtype)
    actions = torch.randn(
        batch, horizon, base_spec.action_dim, device=device, dtype=dtype
    )
    noise = torch.randn_like(actions)
    t = torch.full((batch,), 0.37, device=device, dtype=dtype)
    attention_mask = torch.ones(batch, args.vl_tokens, device=device, dtype=torch.bool)

    with torch.inference_mode():
        expected, reference_sequence_length = reference_velocity(
            reference, vl, actions, state, t, noise, attention_mask
        )
        actual = universal.action_model.flow_forward(
            vl,
            actions,
            state,
            universal.base_embodiment,
            fusion_enabled=False,
            t=t,
            noise=noise,
            encoder_attention_mask=attention_mask,
        )
    max_abs = float((actual["pred_velocity"] - expected).abs().max().float().cpu())
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "strict_base_contract": "PASS",
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "fusion_off_max_abs_velocity_error": max_abs,
        "reference_sequence_length": int(reference_sequence_length),
        "universal_sequence_length": int(actual["dit_sequence_length"].item()),
        "parity": "PASS" if max_abs == 0.0 else "FAIL",
        "rollout": "NOT_RUN",
        "gpu_backward": "NOT_RUN",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["parity"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
