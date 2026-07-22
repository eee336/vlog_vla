"""Causal option injection at the shared action-expert boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class OptionConditioningOutput:
    action_tokens: torch.Tensor
    option_token: torch.Tensor
    residual: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor


class OptionConditioner(nn.Module):
    """Create an option token and residual FiLM modulation for action tokens.

    The final FiLM and token projections are zero-initialized.  There is no
    learned scalar gate that can globally turn every option off; ``rho`` is an
    explicit experiment/config value.  The caller must bypass this module (and
    omit the added token) when fusion is disabled or rho is zero so base GR00T
    parity remains exact.
    """

    def __init__(
        self,
        option_dim: int,
        state_dim: int,
        embodiment_dim: int,
        dit_dim: int,
        rho: float = 0.05,
    ) -> None:
        super().__init__()
        if rho < 0:
            raise ValueError("rho must be non-negative")
        self.rho = float(rho)
        self.context = nn.Sequential(
            nn.Linear(option_dim + state_dim + embodiment_dim, dit_dim),
            nn.GELU(),
            nn.Linear(dit_dim, dit_dim),
            nn.LayerNorm(dit_dim),
        )
        self.action_norm = nn.LayerNorm(dit_dim)
        self.film = nn.Linear(dit_dim, dit_dim * 2)
        self.to_option_token = nn.Linear(dit_dim, dit_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.to_option_token.weight)
        nn.init.zeros_(self.to_option_token.bias)

    def forward(
        self,
        action_tokens: torch.Tensor,
        option_embedding: torch.Tensor,
        state_summary: torch.Tensor,
        embodiment_embedding: torch.Tensor,
    ) -> OptionConditioningOutput:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"action_tokens must be [B,T,D], got {tuple(action_tokens.shape)}"
            )
        context = self.context(
            torch.cat([option_embedding, state_summary, embodiment_embedding], dim=-1)
        )
        gamma, beta = self.film(context).chunk(2, dim=-1)
        # Bounded scale prevents a newly trained conditioner from immediately
        # destabilising a warm-started flow expert.
        gamma = torch.tanh(gamma)
        residual = (
            gamma[:, None, :] * self.action_norm(action_tokens) + beta[:, None, :]
        )
        adapted = action_tokens + self.rho * residual
        option_token = self.rho * self.to_option_token(context)[:, None, :]
        return OptionConditioningOutput(adapted, option_token, residual, gamma, beta)
