#!/usr/bin/env python3
"""Evidence-producing offline Gate O/R audit for QwenUniversalVLOG.

The audit reuses the same observation, action target, mask, FM time and noise
for base/correct/wrong/shuffled/router interventions.  It reports paired
per-observation effects; it does not fabricate option timelines or claim
simulator success.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


def _per_example_fm(output: dict) -> torch.Tensor:
    per_step = (
        output["pred_velocity"].float() - output["target_velocity"].float()
    ).square().mean(dim=-1)
    mask = output["action_mask"].to(per_step.dtype)
    return (per_step * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _per_example_velocity_delta(
    first: dict, second: dict
) -> torch.Tensor:
    per_step = (
        first["pred_velocity"].float() - second["pred_velocity"].float()
    ).square().mean(dim=-1)
    mask = first["action_mask"].to(per_step.dtype)
    return torch.sqrt(
        (per_step * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    )


def _summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "p50": None,
            "p95": None,
        }
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _append(target: dict[str, list[float]], key: str, tensor: torch.Tensor) -> None:
    target[key].extend(tensor.detach().float().cpu().reshape(-1).tolist())


def _mean_over_repeats(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(values, dim=0).mean(dim=0)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    checkpoint = Path(args.checkpoint).resolve()
    model = baseframework.from_pretrained(str(checkpoint))
    if type(model).__name__ != "QwenUniversalVLOG":
        raise TypeError(
            f"Expected QwenUniversalVLOG checkpoint, got {type(model).__name__}"
        )
    if not bool(model.config.framework.vlog.enabled):
        raise ValueError("Gate audit requires framework.vlog.enabled=true")
    if not bool(model.config.framework.vlog.fusion_enabled):
        raise ValueError("Gate audit requires framework.vlog.fusion_enabled=true")

    model = model.to(args.device).eval()
    data_cfg = model.config.datasets.vla_data
    if args.data_root is not None:
        data_cfg.data_root_dir = str(Path(args.data_root).resolve())
    data_cfg.per_device_batch_size = int(args.batch_size)
    dataset = get_vla_dataset(data_cfg=data_cfg, seed=args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)

    measurements: dict[str, list[float]] = defaultdict(list)
    oracle_counts = torch.zeros(model.vlog_core.num_options, dtype=torch.long)
    router_counts = torch.zeros_like(oracle_counts)
    processed_batches = 0
    processed_observations = 0

    # The checkpoint can contain bf16 GR00T weights and fp32 option modules.
    # Mirror the framework's training autocast so those paths are audited
    # under the same mixed-dtype contract.
    autocast_context = model._autocast(torch.bfloat16)
    autocast_context.__enter__()
    for batch_index, examples in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        vl_hidden, attention_mask = model._encode_batch(examples)
        for embodiment, indices in model._group_indices(examples).items():
            index_tensor = torch.tensor(
                indices, device=vl_hidden.device, dtype=torch.long
            )
            vl = vl_hidden.index_select(0, index_tensor)
            mask_vl = (
                attention_mask.index_select(0, index_tensor)
                if attention_mask is not None
                else None
            )
            state = model._state_group(
                examples, indices, embodiment, vl.device, vl.dtype
            )
            actions, action_mask = model._action_group(
                examples, indices, embodiment, vl.device, vl.dtype
            )
            state_tokens = model.action_model.encode_state(state, embodiment)
            embodiment_embedding = model.action_model._embodiment_embedding(
                embodiment, vl.shape[0], vl.device
            )
            state_feature = model.vlog_core.aggregate(
                vl, state_tokens, embodiment_embedding
            )
            zero_t = torch.zeros(
                actions.shape[0], device=actions.device, dtype=torch.long
            )
            future_tokens = model.action_model.encode_action(
                actions, zero_t, embodiment
            )
            oracle = model.vlog_core.discover(
                state_feature, future_tokens, action_mask
            )
            routed = model.vlog_core.route(state_feature)
            hard_router_option = model.vlog_core.codebook.codebook(
                routed["option_idx"]
            )
            wrong_idx = (oracle["option_idx"] + 1) % model.vlog_core.num_options
            wrong_option = model.vlog_core.codebook.codebook(wrong_idx)
            permutation = torch.roll(
                torch.arange(actions.shape[0], device=actions.device), shifts=1
            )
            shuffled_idx = oracle["option_idx"].index_select(0, permutation)
            shuffled_option = oracle["option"].index_select(0, permutation)
            shuffle_changed = shuffled_idx != oracle["option_idx"]

            oracle_counts += torch.bincount(
                oracle["option_idx"].detach().cpu(),
                minlength=model.vlog_core.num_options,
            )
            router_counts += torch.bincount(
                routed["option_idx"].detach().cpu(),
                minlength=model.vlog_core.num_options,
            )
            _append(
                measurements,
                "router_correct",
                (routed["option_idx"] == oracle["option_idx"]).to(torch.float32),
            )
            _append(
                measurements,
                "router_cross_entropy",
                F.cross_entropy(
                    routed["logits"], oracle["option_idx"], reduction="none"
                ),
            )
            _append(
                measurements,
                "router_confidence",
                routed["probs"].max(dim=-1).values,
            )
            _append(
                measurements,
                "router_entropy",
                -(routed["probs"] * (routed["probs"] + 1e-8).log()).sum(-1),
            )

            repeat_values: dict[str, list[torch.Tensor]] = defaultdict(list)
            for repeat in range(args.fm_repeats):
                torch.manual_seed(
                    args.seed + batch_index * args.fm_repeats + repeat
                )
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        args.seed + batch_index * args.fm_repeats + repeat
                    )
                t = model.action_model.sample_time(
                    actions.shape[0], actions.device, actions.dtype
                )
                noise = torch.randn_like(actions)
                shared = {
                    "action_mask": action_mask,
                    "t": t,
                    "noise": noise,
                    "encoder_attention_mask": mask_vl,
                }
                base = model.action_model.flow_forward(
                    vl,
                    actions,
                    state,
                    embodiment,
                    fusion_enabled=False,
                    **shared,
                )
                correct = model.action_model.flow_forward(
                    vl,
                    actions,
                    state,
                    embodiment,
                    option_embedding=oracle["option"],
                    fusion_enabled=True,
                    **shared,
                )
                wrong = model.action_model.flow_forward(
                    vl,
                    actions,
                    state,
                    embodiment,
                    option_embedding=wrong_option,
                    fusion_enabled=True,
                    **shared,
                )
                shuffled = model.action_model.flow_forward(
                    vl,
                    actions,
                    state,
                    embodiment,
                    option_embedding=shuffled_option,
                    fusion_enabled=True,
                    **shared,
                )
                router = model.action_model.flow_forward(
                    vl,
                    actions,
                    state,
                    embodiment,
                    option_embedding=hard_router_option,
                    fusion_enabled=True,
                    **shared,
                )
                repeat_values["fm_base"].append(_per_example_fm(base))
                repeat_values["fm_correct"].append(_per_example_fm(correct))
                repeat_values["fm_wrong"].append(_per_example_fm(wrong))
                repeat_values["fm_shuffle"].append(_per_example_fm(shuffled))
                repeat_values["fm_router"].append(_per_example_fm(router))
                repeat_values["velocity_delta_wrong"].append(
                    _per_example_velocity_delta(correct, wrong)
                )
                repeat_values["velocity_delta_shuffle"].append(
                    _per_example_velocity_delta(correct, shuffled)
                )
                residual = correct["fusion_residual"].float().norm(dim=-1)
                residual = (
                    residual * action_mask.to(residual.dtype)
                ).sum(dim=-1) / action_mask.sum(dim=-1).clamp_min(1)
                repeat_values["fusion_residual"].append(residual)
                residual_ratio = correct["fusion_residual_ratio"].float()
                residual_ratio = (
                    residual_ratio * action_mask.to(residual_ratio.dtype)
                ).sum(dim=-1) / action_mask.sum(dim=-1).clamp_min(1)
                repeat_values["fusion_residual_ratio"].append(residual_ratio)
                repeat_values["option_token_norm"].append(
                    correct["option_token_norm"].float()
                )

            averaged = {
                key: _mean_over_repeats(value)
                for key, value in repeat_values.items()
            }
            for key, value in averaged.items():
                _append(measurements, key, value)
            _append(
                measurements,
                "gap_base_minus_correct",
                averaged["fm_base"] - averaged["fm_correct"],
            )
            _append(
                measurements,
                "gap_wrong_minus_correct",
                averaged["fm_wrong"] - averaged["fm_correct"],
            )
            if shuffle_changed.any():
                _append(
                    measurements,
                    "gap_shuffle_minus_correct_changed",
                    (averaged["fm_shuffle"] - averaged["fm_correct"])[
                        shuffle_changed
                    ],
                )
                _append(
                    measurements,
                    "velocity_delta_shuffle_changed",
                    averaged["velocity_delta_shuffle"][shuffle_changed],
                )
            _append(
                measurements,
                "gap_router_minus_correct",
                averaged["fm_router"] - averaged["fm_correct"],
            )
            _append(
                measurements,
                "gap_base_minus_router",
                averaged["fm_base"] - averaged["fm_router"],
            )
            _append(
                measurements,
                "shuffle_changed",
                shuffle_changed.to(torch.float32),
            )
            processed_observations += len(indices)
        processed_batches += 1

    if processed_observations == 0:
        raise RuntimeError("No audit observations were produced by the dataloader")

    summaries = {key: _summary(value) for key, value in measurements.items()}
    oracle_probs = oracle_counts.float() / oracle_counts.sum().clamp_min(1)
    router_probs = router_counts.float() / router_counts.sum().clamp_min(1)
    normalized_codes = F.normalize(
        model.vlog_core.codebook.codebook.weight.detach().float(), dim=-1
    )
    code_cosine = normalized_codes @ normalized_codes.transpose(0, 1)
    off_diagonal = ~torch.eye(
        model.vlog_core.num_options, dtype=torch.bool, device=code_cosine.device
    )
    pairwise_cosine = code_cosine[off_diagonal].cpu().numpy()
    option_usage = {
        "oracle_counts": oracle_counts.tolist(),
        "router_counts": router_counts.tolist(),
        "oracle_active": int((oracle_counts > 0).sum()),
        "router_active": int((router_counts > 0).sum()),
        "oracle_dominant_share": float(oracle_probs.max()),
        "router_dominant_share": float(router_probs.max()),
        "codebook_pairwise_cosine_mean": float(pairwise_cosine.mean()),
        "codebook_pairwise_cosine_max": float(pairwise_cosine.max()),
    }

    correct_mean = summaries["fm_correct"]["mean"]
    wrong_gap = summaries["gap_wrong_minus_correct"]
    shuffle_gap = summaries["gap_shuffle_minus_correct_changed"]
    relative_wrong = (
        wrong_gap["mean"] / max(correct_mean, 1e-12)
        if wrong_gap["mean"] is not None
        else None
    )
    relative_shuffle = (
        shuffle_gap["mean"] / max(correct_mean, 1e-12)
        if shuffle_gap["mean"] is not None
        else None
    )
    gate_o_checks = {
        "enough_observations": processed_observations >= args.min_observations,
        "wrong_gap_ci_positive": wrong_gap["ci95_low"] is not None
        and wrong_gap["ci95_low"] > 0.0,
        "shuffle_gap_ci_positive": shuffle_gap["ci95_low"] is not None
        and shuffle_gap["ci95_low"] > 0.0,
        "wrong_relative_gap": relative_wrong is not None
        and relative_wrong >= args.min_relative_wrong_gap,
        "shuffle_relative_gap": relative_shuffle is not None
        and relative_shuffle >= args.min_relative_shuffle_gap,
        "correct_not_worse_than_base": summaries["gap_base_minus_correct"][
            "mean"
        ]
        is not None
        and summaries["gap_base_minus_correct"]["mean"] >= 0.0,
        "oracle_options_active": option_usage["oracle_active"]
        >= args.min_active_options,
        "oracle_not_dominated": option_usage["oracle_dominant_share"]
        <= args.max_dominant_share,
        "fusion_residual_nonzero": summaries["fusion_residual"]["p50"]
        is not None
        and summaries["fusion_residual"]["p50"]
        >= args.min_residual_p50,
        "fusion_residual_ratio_bounded": summaries[
            "fusion_residual_ratio"
        ]["p95"]
        is not None
        and summaries["fusion_residual_ratio"]["p95"]
        <= args.max_residual_ratio_p95,
    }
    gate_r_checks = {
        "enough_observations": processed_observations >= args.min_observations,
        "router_accuracy": summaries["router_correct"]["mean"] is not None
        and summaries["router_correct"]["mean"] >= args.min_router_accuracy,
        "router_options_active": option_usage["router_active"]
        >= args.min_active_options,
        "router_not_dominated": option_usage["router_dominant_share"]
        <= args.max_dominant_share,
        "router_fm_not_worse_than_base": summaries["gap_base_minus_router"][
            "mean"
        ]
        is not None
        and summaries["gap_base_minus_router"]["mean"]
        >= -args.max_router_fm_degradation * summaries["fm_base"]["mean"],
        "router_close_to_oracle": summaries["gap_router_minus_correct"][
            "mean"
        ]
        is not None
        and summaries["gap_router_minus_correct"]["mean"]
        <= args.max_router_oracle_gap * correct_mean,
    }

    report = {
        "evidence_level": "E2_observed_offline",
        "scope": "paired offline FM interventions; no rollout claim",
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "git_revision": _git_revision(),
        "saved_train_stage": str(model.vlog_train_stage),
        "conditioning": {
            "mode": str(model.action_model.option_conditioner.conditioning_mode),
            "rho": float(model.action_model.option_conditioner.rho),
            "bound_outputs": bool(
                model.action_model.option_conditioner.bound_outputs
            ),
            "inject_base_embodiment_token": bool(
                model.action_model.inject_base_embodiment_token
            ),
        },
        "dataset": {
            "data_mix": str(data_cfg.data_mix),
            "data_root_dir": str(data_cfg.data_root_dir),
            "batch_size": args.batch_size,
            "num_batches": processed_batches,
            "observations": processed_observations,
            "fm_repeats_per_observation": args.fm_repeats,
            "seed": args.seed,
        },
        "option_usage": option_usage,
        "metrics": summaries,
        "derived": {
            "relative_wrong_gap": relative_wrong,
            "relative_shuffle_gap": relative_shuffle,
        },
        "thresholds": {
            "min_observations": args.min_observations,
            "min_relative_wrong_gap": args.min_relative_wrong_gap,
            "min_relative_shuffle_gap": args.min_relative_shuffle_gap,
            "min_active_options": args.min_active_options,
            "max_dominant_share": args.max_dominant_share,
            "min_residual_p50": args.min_residual_p50,
            "max_residual_ratio_p95": args.max_residual_ratio_p95,
            "min_router_accuracy": args.min_router_accuracy,
            "max_router_fm_degradation": args.max_router_fm_degradation,
            "max_router_oracle_gap": args.max_router_oracle_gap,
        },
        "gate_o": {
            "checks": gate_o_checks,
            "verdict": "PASS" if all(gate_o_checks.values()) else "FAIL",
        },
        "gate_r": {
            "checks": gate_r_checks,
            "verdict": "PASS" if all(gate_r_checks.values()) else "FAIL",
        },
    }
    autocast_context.__exit__(None, None, None)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--fm-repeats", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-observations", type=int, default=256)
    parser.add_argument("--min-relative-wrong-gap", type=float, default=0.05)
    parser.add_argument("--min-relative-shuffle-gap", type=float, default=0.02)
    parser.add_argument("--min-active-options", type=int, default=4)
    parser.add_argument("--max-dominant-share", type=float, default=0.80)
    parser.add_argument("--min-residual-p50", type=float, default=1.0e-4)
    parser.add_argument("--max-residual-ratio-p95", type=float, default=0.25)
    parser.add_argument("--min-router-accuracy", type=float, default=0.35)
    parser.add_argument("--max-router-fm-degradation", type=float, default=0.02)
    parser.add_argument("--max-router-oracle-gap", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be >=2 for shuffled-option intervention")
    if args.fm_repeats < 1 or args.num_batches < 1:
        raise ValueError("fm-repeats and num-batches must be positive")
    report = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
