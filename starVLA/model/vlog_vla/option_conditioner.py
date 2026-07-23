"""Causal option injection at the shared action-expert boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OptionConditioningOutput:
    action_tokens: torch.Tensor
    option_token: torch.Tensor
    residual: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    applied_residual: torch.Tensor


class OptionConditioner(nn.Module):
    """Create an option token and residual FiLM modulation for action tokens.

    The final FiLM and token projections are zero-initialized.  There is no
    learned scalar gate that can globally turn every option off; ``rho`` is an
    explicit experiment/config value.  The caller must bypass this module (and
    omit the added token) when fusion is disabled or rho is zero so base GR00T
    parity remains exact.
    """

    def __init__(
        self,
        option_dim: int,
        state_dim: int,
        embodiment_dim: int,
        dit_dim: int,
        rho: float = 0.05,
        conditioning_mode: str = "legacy_context",
        bound_outputs: bool = False,
    ) -> None:
        super().__init__()
        if rho < 0:
            raise ValueError("rho must be non-negative")
        if conditioning_mode not in {"legacy_context", "option_only"}:
            raise ValueError(
                "conditioning_mode must be 'legacy_context' or 'option_only'"
            )
        self.rho = float(rho)
        self.conditioning_mode = str(conditioning_mode)
        self.bound_outputs = bool(bound_outputs)
        self.context = nn.Sequential(
            nn.Linear(option_dim + state_dim + embodiment_dim, dit_dim),
            nn.GELU(),
            nn.Linear(dit_dim, dit_dim),
            nn.LayerNorm(dit_dim),
        )
        if self.conditioning_mode == "option_only":
            # No biases or affine LayerNorm: zero/constant state and embodiment
            # cannot create an option-independent residual through this path.
            self.option_projection = nn.Sequential(
                nn.LayerNorm(option_dim, elementwise_affine=False),
                nn.Linear(option_dim, dit_dim, bias=False),
                nn.GELU(),
                nn.Linear(dit_dim, dit_dim, bias=False),
                nn.LayerNorm(dit_dim, elementwise_affine=False),
            )
        self.action_norm = nn.LayerNorm(dit_dim)
        self.film = nn.Linear(dit_dim, dit_dim * 2)
        self.to_option_token = nn.Linear(dit_dim, dit_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Deterministically reinitialise a repair-stage conditioner."""

        for container in (self.context, getattr(self, "option_projection", None)):
            if container is None:
                continue
            for module in container.modules():
                if module is container:
                    continue
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()
        self.action_norm.reset_parameters()
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.to_option_token.weight)
        nn.init.zeros_(self.to_option_token.bias)

    def configure_trainable_parameters(self, enabled: bool = True) -> None:
        """Enable only parameters that participate in the selected mode."""

        for name, parameter in self.named_parameters():
            used = True
            if self.conditioning_mode == "option_only":
                used = not (
                    name.startswith("context.")
                    or name in {"film.bias", "to_option_token.bias"}
                )
            parameter.requires_grad_(bool(enabled) and used)

    def forward(
        self,
        action_tokens: torch.Tensor,
        option_embedding: torch.Tensor,
        state_summary: torch.Tensor,
        embodiment_embedding: torch.Tensor,
    ) -> OptionConditioningOutput:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"action_tokens must be [B,T,D], got {tuple(action_tokens.shape)}"
            )
        if self.conditioning_mode == "option_only":
            context = self.option_projection(option_embedding)
            # Ignore learned biases in repair mode: otherwise a shared bias is
            # itself a code-independent adapter that can pass base-vs-correct
            # while failing wrong/shuffle interventions.
            film_output = F.linear(context, self.film.weight, bias=None)
            token_output = F.linear(
                context, self.to_option_token.weight, bias=None
            )
        else:
            context = self.context(
                torch.cat(
                    [option_embedding, state_summary, embodiment_embedding], dim=-1
                )
            )
            film_output = self.film(context)
            token_output = self.to_option_token(context)
        gamma, beta = film_output.chunk(2, dim=-1)
        # Bounded scale prevents a newly trained conditioner from immediately
        # destabilising a warm-started flow expert.
        gamma = torch.tanh(gamma)
        if self.bound_outputs:
            beta = torch.tanh(beta)
            token_output = torch.tanh(token_output)
        residual = (
            gamma[:, None, :] * self.action_norm(action_tokens) + beta[:, None, :]
        )
        applied_residual = self.rho * residual
        adapted = action_tokens + applied_residual
        option_token = self.rho * token_output[:, None, :]
        return OptionConditioningOutput(
            action_tokens=adapted,
            option_token=option_token,
            residual=residual,
            gamma=gamma,
            beta=beta,
            applied_residual=applied_residual,
        )
