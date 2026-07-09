from __future__ import annotations

import copy

import torch
import torch.nn as nn


class OptionCritic(nn.Module):
    """Option-level critic Q(s,o). It does not accept continuous actions."""

    def __init__(self, state_dim: int, option_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + option_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_feature: torch.Tensor, option_embedding: torch.Tensor) -> torch.Tensor:
        return self.q_net(torch.cat([state_feature, option_embedding], dim=-1)).squeeze(-1)

    def q_all(self, state_feature: torch.Tensor, option_nodes: torch.Tensor) -> torch.Tensor:
        batch, num_options, _ = option_nodes.shape
        state = state_feature[:, None, :].expand(batch, num_options, -1)
        return self.q_net(torch.cat([state, option_nodes], dim=-1)).squeeze(-1)

    def make_target(self) -> "OptionCritic":
        target = copy.deepcopy(self)
        for param in target.parameters():
            param.requires_grad_(False)
        return target
