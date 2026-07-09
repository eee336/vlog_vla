from __future__ import annotations

import torch
import torch.nn as nn


class PersistentOptionRouter(nn.Module):
    """Router p(o | s) over latent option nodes.

    Persistence decisions are handled by VLOGPolicyWrapper so the router can
    remain differentiable during training.
    """

    def __init__(self, state_dim: int, option_dim: int, num_options: int) -> None:
        super().__init__()
        self.num_options = int(num_options)
        self.state_proj = nn.Linear(state_dim, option_dim)
        self.score = nn.Sequential(
            nn.Linear(option_dim * 2, option_dim),
            nn.GELU(),
            nn.Linear(option_dim, 1),
        )

    def forward(self, state_feature: torch.Tensor, option_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state_proj(state_feature)[:, None, :].expand_as(option_nodes)
        logits = self.score(torch.cat([state, option_nodes], dim=-1)).squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        soft_option = torch.sum(probs[..., None] * option_nodes, dim=1)
        option_idx = torch.argmax(probs, dim=-1)
        return soft_option, option_idx, probs, logits
