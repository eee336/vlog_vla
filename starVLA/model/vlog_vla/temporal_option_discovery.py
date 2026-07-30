"""Temporal summaries and imbalance controls for latent-option discovery.

Event classes in this module are training strata, not semantic option labels.
They only reweight examples so long transport phases cannot dominate the VQ
objective.  The posterior remains free to discover its own discrete codes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn


def _validate_temporal_inputs(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError(f"values must be [B,T,D], got {tuple(values.shape)}")
    if values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("values must have a non-empty time and feature dimension")
    if mask.shape != values.shape[:2]:
        raise ValueError(
            f"mask must be {tuple(values.shape[:2])}, got {tuple(mask.shape)}"
        )
    return values, mask.to(device=values.device, dtype=torch.bool)


def masked_temporal_statistics(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean, std, endpoint delta and maximum adjacent change."""

    values, mask = _validate_temporal_inputs(values, mask)
    weights = mask.to(values.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / denominator
    variance = ((values - mean[:, None]) ** 2 * weights).sum(dim=1) / denominator
    std = variance.clamp_min(0.0).sqrt()

    batch, horizon, feature_dim = values.shape
    positions = torch.arange(horizon, device=values.device)[None].expand(batch, -1)
    first_index = torch.where(mask, positions, horizon).min(dim=1).values
    last_index = torch.where(mask, positions, -1).max(dim=1).values
    has_valid = last_index >= 0
    safe_first = first_index.clamp(max=max(horizon - 1, 0))
    safe_last = last_index.clamp(min=0)
    gather_shape = (-1, 1, feature_dim)
    first = values.gather(
        1, safe_first[:, None, None].expand(*gather_shape)
    ).squeeze(1)
    last = values.gather(
        1, safe_last[:, None, None].expand(*gather_shape)
    ).squeeze(1)
    endpoint_delta = torch.where(
        has_valid[:, None], last - first, torch.zeros_like(last)
    )

    if horizon < 2:
        max_change = torch.zeros_like(mean)
    else:
        pair_mask = mask[:, 1:] & mask[:, :-1]
        adjacent_change = (values[:, 1:] - values[:, :-1]).abs()
        adjacent_change = adjacent_change.masked_fill(
            ~pair_mask[..., None], float("-inf")
        )
        max_change = adjacent_change.max(dim=1).values
        max_change = torch.where(
            torch.isfinite(max_change), max_change, torch.zeros_like(max_change)
        )
    return mean, std, endpoint_delta, max_change


class FutureActionPosterior(nn.Module):
    """Training-only q(z | h, A_future) with short-event-sensitive statistics.

    ``action_proj`` retains the original mean path and parameter names.  The
    zero-initialized dynamics path makes old checkpoints behavior-compatible
    while allowing a repaired U1 run to learn temporal variation.
    """

    def __init__(self, state_dim: int, action_token_dim: int, option_dim: int) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, option_dim)
        self.action_proj = nn.Linear(action_token_dim, option_dim)
        self.dynamics_proj = nn.Linear(action_token_dim * 3, option_dim, bias=False)
        nn.init.zeros_(self.dynamics_proj.weight)
        self.fuse = nn.Sequential(
            nn.Linear(option_dim * 2, option_dim),
            nn.GELU(),
            nn.Linear(option_dim, option_dim),
        )

    def forward(
        self,
        state_feature: torch.Tensor,
        future_action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        mean, std, endpoint_delta, max_change = masked_temporal_statistics(
            future_action_tokens, action_mask
        )
        action_feature = self.action_proj(mean) + self.dynamics_proj(
            torch.cat([std, endpoint_delta, max_change], dim=-1)
        )
        return self.fuse(
            torch.cat([self.state_proj(state_feature), action_feature], dim=-1)
        )


def classify_action_events(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    groups: Sequence[Sequence[int]],
    min_motion: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify chunks into low-motion or their strongest configured group.

    Class 0 is low motion.  Classes 1..G correspond to configured dimension
    groups.  The energy combines mean and peak temporal change so a brief
    gripper transition is not averaged away by a long action chunk.
    """

    actions, action_mask = _validate_temporal_inputs(actions, action_mask)
    if not groups:
        raise ValueError("At least one action event group is required")
    if actions.shape[1] < 2:
        energy = actions.new_zeros((actions.shape[0], len(groups)))
        return torch.zeros(actions.shape[0], device=actions.device, dtype=torch.long), energy

    pair_mask = action_mask[:, 1:] & action_mask[:, :-1]
    delta = (actions[:, 1:] - actions[:, :-1]).abs()
    energies = []
    for raw_indices in groups:
        indices = torch.as_tensor(raw_indices, device=actions.device, dtype=torch.long)
        if indices.numel() == 0:
            raise ValueError("Action event groups cannot be empty")
        if int(indices.min()) < 0 or int(indices.max()) >= actions.shape[-1]:
            raise ValueError(
                f"Action event group {list(raw_indices)} is outside action_dim={actions.shape[-1]}"
            )
        group_delta = delta.index_select(-1, indices)
        # Reduce dimensions before time so a one-DOF gripper and a many-DOF arm
        # group remain comparable; group width must not dilute a short event.
        per_step_change = group_delta.max(dim=-1).values
        valid_count = pair_mask.sum(dim=1).clamp_min(1)
        mean_change = (per_step_change * pair_mask).sum(dim=1) / valid_count
        peak_change = per_step_change.masked_fill(
            ~pair_mask, float("-inf")
        ).max(dim=1).values
        peak_change = torch.where(
            torch.isfinite(peak_change), peak_change, torch.zeros_like(peak_change)
        )
        energies.append(mean_change + 0.5 * peak_change)
    energy = torch.stack(energies, dim=-1)
    strongest_energy, strongest_group = energy.max(dim=-1)
    event = strongest_group + 1
    event = torch.where(
        strongest_energy >= float(min_motion), event, torch.zeros_like(event)
    )
    return event, energy


class RunningClassBalancer(nn.Module):
    """Inverse-frequency sample weights with checkpointed global counts."""

    def __init__(
        self,
        num_classes: int,
        *,
        power: float = 0.5,
        mix: float = 1.0,
        max_weight: float = 5.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")
        if power < 0.0 or max_weight <= 0.0:
            raise ValueError("power must be non-negative and max_weight must be positive")
        self.num_classes = int(num_classes)
        self.power = float(power)
        self.mix = float(mix)
        self.max_weight = float(max_weight)
        self.register_buffer("counts", torch.zeros(self.num_classes, dtype=torch.float64))

    @torch.no_grad()
    def update(self, labels: torch.Tensor) -> None:
        labels = labels.detach().reshape(-1).to(torch.long)
        if labels.numel() == 0:
            return
        if int(labels.min()) < 0 or int(labels.max()) >= self.num_classes:
            raise ValueError(
                f"labels must be in [0,{self.num_classes - 1}], got "
                f"[{int(labels.min())},{int(labels.max())}]"
            )
        batch_counts = torch.bincount(
            labels, minlength=self.num_classes
        ).to(device=self.counts.device, dtype=self.counts.dtype)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
        self.counts.add_(batch_counts)

    def weights(self, labels: torch.Tensor, *, update: bool) -> torch.Tensor:
        labels = labels.reshape(-1).to(torch.long)
        if update:
            self.update(labels)
        counts = self.counts.to(device=labels.device, dtype=torch.float32)
        inverse = (counts + 1.0).pow(-self.power)
        balanced = inverse.index_select(0, labels)
        balanced = balanced / balanced.mean().clamp_min(1e-8)
        weights = (1.0 - self.mix) + self.mix * balanced
        weights = weights.clamp(max=self.max_weight)
        return weights / weights.mean().clamp_min(1e-8)


def weighted_mean(values: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    """Mean over the leading batch dimension with normalized sample weights."""

    if values.ndim == 0:
        return values
    per_sample = values.reshape(values.shape[0], -1).mean(dim=-1)
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.to(device=values.device, dtype=values.dtype).reshape(-1)
    if weights.shape[0] != values.shape[0]:
        raise ValueError(
            f"sample_weights has {weights.shape[0]} entries for batch {values.shape[0]}"
        )
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
