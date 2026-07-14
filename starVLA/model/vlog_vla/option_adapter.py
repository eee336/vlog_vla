from __future__ import annotations

import torch
import torch.nn as nn


class OptionAdapter(nn.Module):
    """Zero-initialized residual FiLM adapter for injecting latent options.

    Uses h + α·(γ⊙h + β) with α clamped ≥0 so the adapter cannot invert the
    frozen base representation. Compatible with checkpoints that stored α as a
    direct scalar (no softplus reparameterization).
    """

    def __init__(self, hidden_dim: int, option_dim: int, alpha_init: float = 0.0) -> None:
        super().__init__()
        self.film = nn.Linear(option_dim, hidden_dim * 2)
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def effective_alpha(self) -> torch.Tensor:
        return self.alpha.clamp(min=0.0)

    def forward(self, hidden_tokens: torch.Tensor, option_embedding: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(option_embedding).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)[:, None, :]
        beta = beta[:, None, :]
        delta = hidden_tokens * gamma + beta
        return hidden_tokens + self.effective_alpha() * delta
