from __future__ import annotations

import torch
import torch.nn as nn


class OptionGraphLayer(nn.Module):
    """State-conditioned latent option transition graph."""

    def __init__(self, state_dim: int, option_dim: int, num_options: int, num_heads: int = 8) -> None:
        super().__init__()
        if option_dim % num_heads != 0:
            num_heads = 1
        self.num_options = int(num_options)
        self.option_dim = int(option_dim)
        self.state_proj = nn.Linear(state_dim, option_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(option_dim * 3, option_dim),
            nn.GELU(),
            nn.Linear(option_dim, 1),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=option_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=option_dim * 4,
            activation="gelu",
        )
        self.graph_transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, state_feature: torch.Tensor, option_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state_feature.shape[0]
        num_options = option_codes.shape[0]
        state = self.state_proj(state_feature)
        # Detach codebook weights so Stage3 graph/transition losses cannot
        # rewrite Stage2 option prototypes (was the RoboCasa collapse root cause).
        nodes = option_codes.detach().unsqueeze(0).expand(batch, num_options, -1)
        ci = nodes.unsqueeze(2).expand(batch, num_options, num_options, -1)
        cj = nodes.unsqueeze(1).expand(batch, num_options, num_options, -1)
        st = state[:, None, None, :].expand(batch, num_options, num_options, -1)
        edge_logits = self.edge_mlp(torch.cat([ci, cj, st], dim=-1)).squeeze(-1)
        updated_nodes = self.graph_transformer(nodes + state[:, None, :])
        return updated_nodes, edge_logits
