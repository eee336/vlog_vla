from __future__ import annotations

import torch
import torch.nn as nn


class StateAggregator(nn.Module):
    """Pool VLA hidden tokens into a compact state feature."""

    def __init__(self, hidden_dim: int, state_dim: int | None = None, out_dim: int | None = None, num_heads: int = 8) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim) if state_dim is not None else None
        self.out_proj = nn.Linear(hidden_dim, out_dim or hidden_dim)

    def forward(self, hidden_tokens: torch.Tensor, robot_state: torch.Tensor | None = None) -> torch.Tensor:
        if hidden_tokens.ndim != 3:
            raise ValueError(f"hidden_tokens must be [B,N,D], got {tuple(hidden_tokens.shape)}")
        batch = hidden_tokens.shape[0]
        query = self.query.expand(batch, -1, -1)
        pooled, _ = self.cross_attn(query, hidden_tokens, hidden_tokens, need_weights=False)
        pooled = pooled.squeeze(1)
        if self.state_proj is not None and robot_state is not None:
            pooled = pooled + self.state_proj(robot_state)
        return self.out_proj(self.norm(pooled))
