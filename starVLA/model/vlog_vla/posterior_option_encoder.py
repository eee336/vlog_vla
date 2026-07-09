from __future__ import annotations

import torch
import torch.nn as nn


class PosteriorOptionEncoder(nn.Module):
    """Training-only encoder that discovers latent options from future actions."""

    def __init__(self, state_dim: int, action_dim: int, option_dim: int, window_size: int) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.action_dim = int(action_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.window_size * self.action_dim, 512),
            nn.GELU(),
            nn.Linear(512, option_dim),
        )
        self.state_encoder = nn.Linear(state_dim, option_dim)
        self.fuse = nn.Sequential(
            nn.Linear(option_dim * 2, option_dim),
            nn.GELU(),
            nn.Linear(option_dim, option_dim),
        )

    def forward(self, state_feature: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        batch = future_actions.shape[0]
        if future_actions.shape[1] != self.window_size or future_actions.shape[2] != self.action_dim:
            raise ValueError(
                f"future_actions must be [B,{self.window_size},{self.action_dim}], got {tuple(future_actions.shape)}"
            )
        action_feat = self.action_encoder(future_actions.reshape(batch, -1))
        state_feat = self.state_encoder(state_feature)
        return self.fuse(torch.cat([state_feat, action_feat], dim=-1))
