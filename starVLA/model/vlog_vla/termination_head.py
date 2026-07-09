from __future__ import annotations

import torch
import torch.nn as nn


class TerminationHead(nn.Module):
    """Predict beta(s,o), the probability that an option should terminate."""

    def __init__(self, state_dim: int, option_dim: int) -> None:
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(state_dim + option_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, state_feature: torch.Tensor, option_embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.beta_net(torch.cat([state_feature, option_embedding], dim=-1))).squeeze(-1)
