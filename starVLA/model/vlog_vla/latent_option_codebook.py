from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentOptionCodebook(nn.Module):
    """Straight-through vector-quantized latent option codebook."""

    def __init__(self, num_options: int = 16, option_dim: int = 256, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_options = int(num_options)
        self.option_dim = int(option_dim)
        self.commitment_cost = float(commitment_cost)
        self.codebook = nn.Embedding(self.num_options, self.option_dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2:
            raise ValueError(f"embedding must be [B,D], got {tuple(embedding.shape)}")
        distances = torch.cdist(embedding.unsqueeze(1), self.codebook.weight.unsqueeze(0)).squeeze(1)
        option_idx = torch.argmin(distances, dim=-1)
        quantized = self.codebook(option_idx)
        quantized_st = embedding + (quantized - embedding).detach()
        codebook_loss = F.mse_loss(quantized, embedding.detach())
        commitment_loss = F.mse_loss(embedding, quantized.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        return quantized_st, option_idx, vq_loss, commitment_loss

    @torch.no_grad()
    def usage_statistics(self, option_idx: torch.Tensor) -> dict:
        flat = option_idx.reshape(-1).to(torch.long)
        counts = torch.bincount(flat, minlength=self.num_options).cpu()
        probs = counts.float() / counts.sum().clamp_min(1)
        entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum().item()
        return {
            "num_options": self.num_options,
            "usage_per_option": counts.tolist(),
            "entropy": entropy,
            "dead_options": torch.nonzero(counts == 0, as_tuple=False).reshape(-1).tolist(),
        }
