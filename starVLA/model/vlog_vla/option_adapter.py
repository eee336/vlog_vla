from __future__ import annotations

import torch
import torch.nn as nn


class OptionAdapter(nn.Module):
    """Zero-initialized FiLM adapter for injecting latent options."""

    def __init__(self, hidden_dim: int, option_dim: int, alpha_init: float = 0.0) -> None:
        super().__init__()
        self.film = nn.Linear(option_dim, hidden_dim * 2)
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, hidden_tokens: torch.Tensor, option_embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(option_embedding).chunk(2, dim=-1)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
        adapted = hidden_tokens * (1.0 + gamma) + beta
        return hidden_tokens + self.alpha * adapted
