from __future__ import annotations

import torch
import torch.nn as nn


class LatentOptionCodebook(nn.Module):
    """Straight-through vector-quantized latent option codebook."""

    def __init__(self, num_options: int = 16, option_dim: int = 256, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.num_options = int(num_options)
        self.option_dim = int(option_dim)
        self.commitment_cost = float(commitment_cost)
        self.codebook = nn.Embedding(self.num_options, self.option_dim)
        nn.init.normal_(self.codebook.weight, std=0.02)

    def forward(
        self,
        embedding: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2:
            raise ValueError(f"embedding must be [B,D], got {tuple(embedding.shape)}")
        distances = torch.cdist(embedding.unsqueeze(1), self.codebook.weight.unsqueeze(0)).squeeze(1)
        option_idx = torch.argmin(distances, dim=-1)
        quantized = self.codebook(option_idx)
        quantized_st = embedding + (quantized - embedding).detach()
        codebook_per_sample = (quantized - embedding.detach()).square().mean(dim=-1)
        commitment_per_sample = (embedding - quantized.detach()).square().mean(dim=-1)
        if sample_weights is None:
            codebook_loss = codebook_per_sample.mean()
            commitment_loss = commitment_per_sample.mean()
        else:
            weights = sample_weights.to(
                device=embedding.device, dtype=embedding.dtype
            ).reshape(-1)
            if weights.shape[0] != embedding.shape[0]:
                raise ValueError(
                    f"sample_weights has {weights.shape[0]} entries for batch {embedding.shape[0]}"
                )
            denominator = weights.sum().clamp_min(1e-8)
            codebook_loss = (codebook_per_sample * weights).sum() / denominator
            commitment_loss = (commitment_per_sample * weights).sum() / denominator
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
