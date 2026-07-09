from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .latent_option_codebook import LatentOptionCodebook
from .option_adapter import OptionAdapter
from .option_critic import OptionCritic
from .option_graph_layer import OptionGraphLayer
from .persistent_option_router import PersistentOptionRouter
from .posterior_option_encoder import PosteriorOptionEncoder
from .state_aggregator import StateAggregator
from .termination_head import TerminationHead


@dataclass
class PersistenceConfig:
    d_min: int = 2
    d_max: int = 8
    beta_threshold: float = 0.7
    q_switch_margin: float = 0.05


class VLOGPolicyWrapper(nn.Module):
    """Persistent latent option wrapper for a pretrained VLA policy.

    This wrapper exposes two levels:
    - representation-level functions for training and smoke tests;
    - optional base_policy delegation for projects that expose hidden tokens
      and action-head APIs.

    It never implements an action-level Q(s,a) critic. The critic only sees
    compact state features and latent option embeddings.
    """

    def __init__(
        self,
        base_policy: nn.Module | None = None,
        hidden_dim: int = 256,
        state_dim: int = 8,
        action_dim: int = 7,
        num_options: int = 16,
        option_dim: int = 256,
        window_size: int = 16,
        persistence: PersistenceConfig | None = None,
        commitment_cost: float = 0.25,
        alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_options = int(num_options)
        self.option_dim = int(option_dim)
        self.window_size = int(window_size)
        self.persistence = persistence or PersistenceConfig()

        self.state_aggregator = StateAggregator(self.hidden_dim, state_dim=self.state_dim, out_dim=self.option_dim)
        self.posterior_encoder = PosteriorOptionEncoder(self.option_dim, self.action_dim, self.option_dim, self.window_size)
        self.codebook = LatentOptionCodebook(self.num_options, self.option_dim, commitment_cost=commitment_cost)
        self.option_graph = OptionGraphLayer(self.option_dim, self.option_dim, self.num_options)
        self.router = PersistentOptionRouter(self.option_dim, self.option_dim, self.num_options)
        self.critic = OptionCritic(self.option_dim, self.option_dim)
        self.target_critic = self.critic.make_target()
        self.termination_head = TerminationHead(self.option_dim, self.option_dim)
        self.option_adapter = OptionAdapter(self.hidden_dim, self.option_dim, alpha_init=alpha_init)

        self.current_option_idx: torch.Tensor | None = None
        self.option_age = 0

    def reset_option_state(self) -> None:
        self.current_option_idx = None
        self.option_age = 0

    def encode_options(
        self,
        hidden_tokens: torch.Tensor,
        robot_state: torch.Tensor | None = None,
        future_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_dtype = self.state_aggregator.query.dtype
        hidden_tokens = hidden_tokens.to(dtype=target_dtype)
        if robot_state is not None:
            robot_state = robot_state.to(dtype=target_dtype)
        if future_actions is not None:
            future_actions = future_actions.to(dtype=target_dtype)
        state_feature = self.state_aggregator(hidden_tokens, robot_state)
        option_codes = self.codebook.codebook.weight
        option_nodes, edge_logits = self.option_graph(state_feature, option_codes)
        z_router, router_idx, option_probs, router_logits = self.router(state_feature, option_nodes)
        out: dict[str, torch.Tensor] = {
            "state_feature": state_feature,
            "option_nodes": option_nodes,
            "edge_logits": edge_logits,
            "z_router": z_router,
            "router_idx": router_idx,
            "option_probs": option_probs,
            "router_logits": router_logits,
        }
        if future_actions is not None:
            e_post = self.posterior_encoder(state_feature, future_actions)
            z_post, posterior_idx, vq_loss, commit_loss = self.codebook(e_post)
            out.update(
                {
                    "e_post": e_post,
                    "z_post": z_post,
                    "posterior_idx": posterior_idx,
                    "vq_loss": vq_loss,
                    "commitment_loss": commit_loss,
                }
            )
        return out

    def adapt_hidden(self, hidden_tokens: torch.Tensor, option_embedding: torch.Tensor) -> torch.Tensor:
        return self.option_adapter(hidden_tokens, option_embedding)

    def forward_train(self, batch: dict[str, torch.Tensor], stage: int = 1) -> dict[str, torch.Tensor]:
        hidden_tokens = batch["hidden_tokens"]
        robot_state = batch.get("robot_state")
        future_actions = batch.get("future_actions")
        out = self.encode_options(hidden_tokens, robot_state, future_actions)
        option_embedding = out.get("z_post", out["z_router"]) if stage in (2, 3) else out["z_router"]
        adapted_hidden = self.adapt_hidden(hidden_tokens, option_embedding)
        q_all = self.critic.q_all(out["state_feature"], out["option_nodes"])
        beta = self.termination_head(out["state_feature"], option_embedding)
        out.update(
            {
                "adapted_hidden_tokens": adapted_hidden,
                "q_all": q_all,
                "beta": beta,
            }
        )
        return out

    @torch.no_grad()
    def select_option(
        self,
        hidden_tokens: torch.Tensor,
        robot_state: torch.Tensor | None = None,
        force_switch: bool = False,
    ) -> dict[str, Any]:
        out = self.encode_options(hidden_tokens, robot_state)
        q_all = self.critic.q_all(out["state_feature"], out["option_nodes"])
        proposed_idx = out["router_idx"]
        batch = proposed_idx.shape[0]
        if batch != 1:
            raise ValueError("persistent inference currently expects batch size 1")

        if self.current_option_idx is None:
            selected_idx = proposed_idx
            switched = True
        else:
            current_idx = self.current_option_idx.to(proposed_idx.device)
            current_option = out["option_nodes"][0, current_idx.item()].unsqueeze(0)
            beta = self.termination_head(out["state_feature"], current_option)
            q_current = q_all[0, current_idx.item()]
            q_best, q_best_idx = torch.max(q_all[0], dim=0)
            should_switch = self._should_switch(
                beta=float(beta.item()),
                q_current=float(q_current.item()),
                q_best=float(q_best.item()),
                force_switch=force_switch,
            )
            selected_idx = q_best_idx.reshape(1) if should_switch else current_idx.reshape(1)
            switched = should_switch

        self.current_option_idx = selected_idx.detach().cpu()
        self.option_age = 0 if switched else self.option_age + 1
        z = out["option_nodes"][0, selected_idx.item()].unsqueeze(0)
        beta = self.termination_head(out["state_feature"], z)
        adapted_hidden = self.adapt_hidden(hidden_tokens, z)
        return {
            "selected_option_idx": int(selected_idx.item()),
            "selected_option_embedding": z,
            "adapted_hidden_tokens": adapted_hidden,
            "option_age": self.option_age,
            "switched": switched,
            "q_all": q_all,
            "beta": beta,
            "router_probs": out["option_probs"],
        }

    def _should_switch(self, beta: float, q_current: float, q_best: float, force_switch: bool = False) -> bool:
        if self.current_option_idx is None:
            return True
        if self.option_age < self.persistence.d_min:
            return False
        if force_switch:
            return True
        if beta > self.persistence.beta_threshold:
            return True
        if q_best > q_current + self.persistence.q_switch_margin:
            return True
        if self.option_age > self.persistence.d_max:
            return True
        return False
