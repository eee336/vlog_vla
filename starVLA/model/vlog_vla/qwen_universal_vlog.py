"""Unified Qwen/GR00T flow expert with causal semi-Markov latent options.

The implementation deliberately preserves the official QwenGR00T parameter
paths for the RoboCasa/GR1 base embodiment.  Additional embodiments only add
native-to-DiT state/action boundaries; the transformer blocks remain shared.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

from .latent_option_codebook import LatentOptionCodebook
from .option_conditioner import OptionConditioner
from .persistent_option_router import PersistentOptionRouter
from .semimarkov_controller import SemiMarkovController
from .state_aggregator import StateAggregator
from .temporal_option_discovery import (
    FutureActionPosterior,
    RunningClassBalancer,
    classify_action_events,
    weighted_mean,
)
from .universal_adapters import (
    EmbodimentAdapterBank,
    EmbodimentSpec,
    normalize_embodiment_specs,
)

logger = initialize_overwatch(__name__)


@dataclass
class QwenUniversalVLOGDefaultConfig(QwenGR00TDefaultConfig):
    """Defaults target the official RoboCasa GR1 GR00T tensor contract."""

    name: str = "QwenUniversalVLOG"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2560,
        }
    )
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 29,
            "state_dim": 58,
            "action_horizon": 16,
            "repeated_diffusion_steps": 1,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2560,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )
    base_embodiment: str = "robocasa_gr1"
    embodiments: dict = field(
        default_factory=lambda: {
            "robocasa_gr1": {"state_dim": 58, "action_dim": 29, "state_tokens": 1},
            "libero": {"state_dim": 7, "action_dim": 7, "state_tokens": 1},
        }
    )
    vlog: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "train_stage": "u0_base",
            "num_options": 8,
            "option_dim": 256,
            "fusion_enabled": False,
            "fusion_residual_scale": 0.05,
            "conditioning_mode": "legacy_context",
            "bound_conditioning_outputs": False,
            "inject_base_embodiment_token": True,
            "d_min": 2,
            "d_max": 8,
            "router_hysteresis": 0.15,
            "router_fm_diagnostics_every": 100,
            "counterfactual_every": 4,
            "counterfactual_margin": 0.05,
            "counterfactual_margin_mode": "absolute",
            "counterfactual_wrong_sampling": "cyclic",
            "improve_margin": 0.0,
            "commitment_cost": 0.25,
            "codebook_cosine_margin": 0.2,
            # Penalize collapse only below this fraction of log(K).  Unlike
            # KL-to-uniform, this does not force long motion phases to split
            # artificially across synonymous codes.
            "usage_entropy_floor_ratio": 0.75,
            # Event strata reweight U1 examples but never supervise option IDs.
            "event_balance": {
                "enabled": False,
                "mix": 0.5,
                "power": 0.5,
                "max_weight": 5.0,
                "min_motion": 0.02,
                "groups": {
                    "robocasa_gr1": [
                        list(range(0, 14)),
                        list(range(14, 26)),
                        list(range(26, 29)),
                    ],
                    "libero": [[0, 1, 2], [3, 4, 5], [6]],
                },
            },
            "router_balance": {
                "enabled": False,
                "power": 0.5,
                "mix": 1.0,
                "max_weight": 5.0,
            },
            "losses": {
                "lambda_fm": 1.0,
                "lambda_vq": 1.0,
                "lambda_commitment": 0.25,
                "lambda_usage": 0.05,
                "lambda_codebook_separation": 0.0,
                "lambda_motion": 0.1,
                "lambda_router": 1.0,
                "lambda_counterfactual": 0.2,
                "lambda_improve": 0.2,
            },
        }
    )


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class UniversalVLOGCore(nn.Module):
    """Shared option discovery/router modules; no critic, graph or learned termination."""

    def __init__(
        self,
        vl_dim: int,
        dit_dim: int,
        option_dim: int,
        num_options: int,
        commitment_cost: float,
        codebook_cosine_margin: float = 0.2,
        usage_entropy_floor_ratio: float = 0.75,
        router_balance: Mapping | None = None,
    ) -> None:
        super().__init__()
        self.num_options = int(num_options)
        self.option_dim = int(option_dim)
        self.codebook_cosine_margin = float(codebook_cosine_margin)
        if not 0.0 <= float(usage_entropy_floor_ratio) <= 1.0:
            raise ValueError("usage_entropy_floor_ratio must be in [0, 1]")
        self.usage_entropy_floor_ratio = float(usage_entropy_floor_ratio)
        self.vl_projection = nn.Sequential(
            nn.LayerNorm(int(vl_dim)), nn.Linear(int(vl_dim), int(option_dim))
        )
        self.aggregator = StateAggregator(
            hidden_dim=int(option_dim),
            state_dim=int(dit_dim),
            out_dim=int(option_dim),
        )
        self.embodiment_proj = nn.Linear(dit_dim, option_dim)
        self.posterior = FutureActionPosterior(option_dim, dit_dim, option_dim)
        self.codebook = LatentOptionCodebook(num_options, option_dim, commitment_cost)
        self.router = PersistentOptionRouter(option_dim, option_dim, num_options)
        router_balance = router_balance or {}
        self.router_balance_enabled = bool(router_balance.get("enabled", False))
        self.router_balancer = RunningClassBalancer(
            num_options,
            power=float(router_balance.get("power", 0.5)),
            mix=float(router_balance.get("mix", 1.0)),
            max_weight=float(router_balance.get("max_weight", 5.0)),
        )
        self.motion_decoder = nn.Sequential(
            nn.Linear(option_dim * 2, dit_dim),
            nn.GELU(),
            nn.Linear(dit_dim, dit_dim),
        )

    def codebook_separation_loss(self) -> torch.Tensor:
        """Penalise pairs of option codes whose cosine similarity is too high."""

        normalized = F.normalize(self.codebook.codebook.weight.float(), dim=-1)
        cosine = normalized @ normalized.transpose(0, 1)
        off_diagonal = ~torch.eye(
            self.num_options, device=cosine.device, dtype=torch.bool
        )
        pairwise = cosine[off_diagonal]
        return F.relu(pairwise - self.codebook_cosine_margin).square().mean()

    def aggregate(
        self,
        vl_hidden: torch.Tensor,
        state_tokens: torch.Tensor,
        embodiment_embedding: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.aggregator(
            self.vl_projection(vl_hidden), state_tokens.mean(dim=1)
        )
        return pooled + self.embodiment_proj(embodiment_embedding)

    def discover(
        self,
        state_feature: torch.Tensor,
        future_action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> dict:
        posterior_embedding = self.posterior(
            state_feature, future_action_tokens, action_mask
        )
        option, option_idx, vq_loss, commitment_loss = self.codebook(
            posterior_embedding, sample_weights=sample_weights
        )
        target_motion = _masked_mean(future_action_tokens, action_mask)
        pred_motion = self.motion_decoder(torch.cat([state_feature, option], dim=-1))
        motion_loss = weighted_mean(
            (pred_motion - target_motion.detach()).square(), sample_weights
        )
        one_hot = F.one_hot(option_idx, num_classes=self.num_options).to(option.dtype)
        hard_usage = one_hot.mean(dim=0)
        # The hard nearest-code histogram is useful for logging but carries no
        # gradient.  Balance a differentiable soft assignment instead.
        distances = torch.cdist(
            posterior_embedding[:, None, :], self.codebook.codebook.weight[None]
        ).squeeze(1)
        soft_assignment = torch.softmax(-distances, dim=-1)
        if sample_weights is None:
            soft_usage = soft_assignment.mean(dim=0)
        else:
            weights = sample_weights.to(
                device=soft_assignment.device, dtype=soft_assignment.dtype
            ).reshape(-1, 1)
            soft_usage = (soft_assignment * weights).sum(dim=0) / weights.sum().clamp_min(
                1e-8
            )
        usage_entropy = -(
            soft_usage * (soft_usage + 1e-8).log()
        ).sum()
        entropy_floor = self.usage_entropy_floor_ratio * np.log(self.num_options)
        usage_loss = F.relu(
            torch.as_tensor(
                entropy_floor,
                device=usage_entropy.device,
                dtype=usage_entropy.dtype,
            )
            - usage_entropy
        ).square()
        separation_loss = self.codebook_separation_loss()
        return {
            "state_feature": state_feature,
            "option": option,
            "option_idx": option_idx,
            "posterior_embedding": posterior_embedding,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "usage_loss": usage_loss,
            "codebook_separation_loss": separation_loss,
            "motion_loss": motion_loss,
            "usage": hard_usage,
            "soft_usage": soft_usage,
            "soft_usage_entropy": usage_entropy,
        }

    def route(self, state_feature: torch.Tensor) -> dict:
        nodes = self.codebook.codebook.weight[None].expand(
            state_feature.shape[0], -1, -1
        )
        soft_option, option_idx, probs, logits = self.router(state_feature, nodes)
        return {
            "option": soft_option,
            "option_idx": option_idx,
            "probs": probs,
            "logits": logits,
        }

    def router_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        update_counts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CE and per-sample weights for the deployable router."""

        target = target.detach().to(torch.long)
        if self.router_balance_enabled:
            weights = self.router_balancer.weights(
                target, update=bool(update_counts)
            ).to(logits.dtype)
        else:
            weights = torch.ones(
                target.shape[0], device=logits.device, dtype=logits.dtype
            )
        per_sample = F.cross_entropy(logits, target, reduction="none")
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
        return loss, weights


class UniversalFlowmatchingActionHead(FlowmatchingActionHead):
    """One shared GR00T DiT with native state/action boundaries."""

    def __init__(
        self,
        full_config,
        specs: Mapping[str, EmbodimentSpec],
        base_embodiment: str,
        option_dim: int,
        fusion_residual_scale: float,
        conditioning_mode: str = "legacy_context",
        bound_conditioning_outputs: bool = False,
        inject_base_embodiment_token: bool = True,
    ) -> None:
        super().__init__(full_config=full_config)
        self.base_embodiment = str(base_embodiment)
        self.embodiment_names = list(specs)
        self.embodiment_to_index = {
            name: idx for idx, name in enumerate(self.embodiment_names)
        }
        self.inject_base_embodiment_token = bool(inject_base_embodiment_token)
        self.adapter_bank = EmbodimentAdapterBank(
            specs=specs,
            base_embodiment=self.base_embodiment,
            dit_dim=self.input_embedding_dim,
            dit_output_dim=int(self.model.config.output_dim),
            hidden_dim=self.hidden_size,
        )
        self.embodiment_embedding = nn.Embedding(len(specs), self.input_embedding_dim)
        nn.init.normal_(self.embodiment_embedding.weight, mean=0.0, std=0.02)
        self.option_conditioner = OptionConditioner(
            option_dim=option_dim,
            state_dim=self.input_embedding_dim,
            embodiment_dim=self.input_embedding_dim,
            dit_dim=self.input_embedding_dim,
            rho=float(fusion_residual_scale),
            conditioning_mode=str(conditioning_mode),
            bound_outputs=bool(bound_conditioning_outputs),
        )

    def embodiment_index(
        self, embodiment_id: str, batch_size: int, device
    ) -> torch.Tensor:
        try:
            idx = self.embodiment_to_index[str(embodiment_id)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown embodiment_id={embodiment_id!r}; registered={self.embodiment_names}"
            ) from exc
        return torch.full((batch_size,), idx, dtype=torch.long, device=device)

    def encode_state(self, state: torch.Tensor, embodiment_id: str) -> torch.Tensor:
        spec = self.adapter_bank.spec(embodiment_id)
        if state.ndim == 2:
            state = state[:, None, :]
        if state.ndim != 3 or state.shape[-1] != spec.state_dim:
            raise ValueError(
                f"{embodiment_id} state must be [B,N,{spec.state_dim}], got {tuple(state.shape)}"
            )
        if self.adapter_bank.is_base(embodiment_id):
            return self.state_encoder(state)
        return self.adapter_bank.state_adapters[embodiment_id](state)

    def encode_action(
        self, actions: torch.Tensor, timesteps: torch.Tensor, embodiment_id: str
    ) -> torch.Tensor:
        spec = self.adapter_bank.spec(embodiment_id)
        if actions.ndim != 3 or actions.shape[-1] != spec.action_dim:
            raise ValueError(
                f"{embodiment_id} action must be [B,T,{spec.action_dim}], got {tuple(actions.shape)}"
            )
        if self.adapter_bank.is_base(embodiment_id):
            return self.action_encoder(actions, timesteps)
        return self.adapter_bank.action_encoders[embodiment_id](actions, timesteps)

    def decode_velocity(
        self, model_output: torch.Tensor, embodiment_id: str
    ) -> torch.Tensor:
        if self.adapter_bank.is_base(embodiment_id):
            return self.action_decoder(model_output)
        return self.adapter_bank.action_decoders[embodiment_id](model_output)

    def _embodiment_embedding(
        self, embodiment_id: str, batch_size: int, device
    ) -> torch.Tensor:
        return self.embodiment_embedding(
            self.embodiment_index(embodiment_id, batch_size, device)
        )

    def _predict_velocity(
        self,
        vl_embs: torch.Tensor,
        noisy_actions: torch.Tensor,
        t_discretized: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: str,
        option_embedding: torch.Tensor | None,
        fusion_enabled: bool,
        encoder_attention_mask=None,
    ) -> tuple[torch.Tensor, dict]:
        action_features = self.encode_action(
            noisy_actions, t_discretized, embodiment_id
        )
        if self.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device,
            )
            action_features = action_features + self.position_embedding(pos_ids)[None]
        unconditioned_action_features = action_features
        state_features = self.encode_state(state, embodiment_id)
        future_tokens = self.future_tokens.weight[None].expand(vl_embs.shape[0], -1, -1)

        use_option = (
            bool(fusion_enabled)
            and option_embedding is not None
            and self.option_conditioner.rho > 0.0
        )
        # Preserve the exact historical GR00T sequence for base/fusion-off.
        add_embodiment_token = (
            not self.adapter_bank.is_base(embodiment_id)
            or (use_option and self.inject_base_embodiment_token)
        )
        sequence = [state_features]
        raw_residual = torch.zeros_like(action_features)
        applied_residual = torch.zeros_like(action_features)
        option_token = torch.zeros(
            action_features.shape[0],
            1,
            action_features.shape[-1],
            device=action_features.device,
            dtype=action_features.dtype,
        )
        if add_embodiment_token:
            embodiment = self._embodiment_embedding(
                embodiment_id, vl_embs.shape[0], vl_embs.device
            )
            sequence.append(embodiment[:, None, :])
        else:
            embodiment = self._embodiment_embedding(
                embodiment_id, vl_embs.shape[0], vl_embs.device
            )
        if use_option:
            conditioned = self.option_conditioner(
                action_features,
                option_embedding,
                state_features.mean(dim=1),
                embodiment,
            )
            action_features = conditioned.action_tokens
            raw_residual = conditioned.residual
            applied_residual = conditioned.applied_residual
            option_token = conditioned.option_token
            sequence.append(conditioned.option_token)
        sequence.extend([future_tokens, action_features])
        model_output = self.model(
            hidden_states=torch.cat(sequence, dim=1),
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.decode_velocity(model_output, embodiment_id)
        base_token_norm = unconditioned_action_features.float().norm(dim=-1)
        applied_norm = applied_residual.float().norm(dim=-1)
        return pred[:, -noisy_actions.shape[1] :], {
            # ``fusion_residual`` is the perturbation actually added after rho,
            # not the pre-scale value.  Keep the raw value separately.
            "fusion_residual": applied_residual,
            "fusion_residual_raw": raw_residual,
            "fusion_residual_ratio": applied_norm
            / base_token_norm.clamp_min(1.0e-6),
            "action_token_norm": base_token_norm,
            "option_token_norm": option_token.float().norm(dim=-1).squeeze(1),
            "dit_sequence_length": torch.tensor(
                model_output.shape[1], device=model_output.device, dtype=torch.float32
            ),
        }

    def flow_forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: str,
        action_mask: torch.Tensor | None = None,
        option_embedding: torch.Tensor | None = None,
        fusion_enabled: bool = False,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        encoder_attention_mask=None,
    ) -> dict:
        batch, horizon, _ = actions.shape
        if noise is None:
            noise = torch.randn_like(actions)
        if noise.shape != actions.shape:
            raise ValueError("noise must have exactly the same shape as actions")
        if t is None:
            t = self.sample_time(batch, actions.device, actions.dtype)
        t = t.reshape(batch).to(device=actions.device, dtype=actions.dtype)
        t_broadcast = t[:, None, None]
        noisy = (1.0 - t_broadcast) * noise + t_broadcast * actions
        velocity_target = actions - noise
        t_discretized = (t * self.num_timestep_buckets).long()
        pred_velocity, diagnostics = self._predict_velocity(
            vl_embs,
            noisy,
            t_discretized,
            state,
            embodiment_id,
            option_embedding,
            fusion_enabled,
            encoder_attention_mask,
        )
        if action_mask is None:
            action_mask = torch.ones(
                batch, horizon, device=actions.device, dtype=torch.bool
            )
        if action_mask.shape != (batch, horizon):
            raise ValueError(
                f"action_mask must be {(batch, horizon)}, got {tuple(action_mask.shape)}"
            )
        # Accumulate FM MSE in fp32 even when the action expert runs under
        # bf16 autocast.
        per_step = (
            (pred_velocity.float() - velocity_target.float()).square().mean(dim=-1)
        )
        loss = (
            per_step * action_mask.to(per_step.dtype)
        ).sum() / action_mask.sum().clamp_min(1)
        return {
            "loss": loss,
            "pred_velocity": pred_velocity,
            "target_velocity": velocity_target,
            "t": t,
            "noise": noise,
            "action_mask": action_mask,
            **diagnostics,
        }

    @torch.no_grad()
    def predict_native_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: str,
        option_embedding: torch.Tensor | None = None,
        fusion_enabled: bool = False,
        encoder_attention_mask=None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spec = self.adapter_bank.spec(embodiment_id)
        batch_size = vl_embs.shape[0]
        if initial_noise is None:
            actions = torch.randn(
                batch_size,
                self.action_horizon,
                spec.action_dim,
                dtype=vl_embs.dtype,
                device=vl_embs.device,
            )
        else:
            expected = (batch_size, self.action_horizon, spec.action_dim)
            if tuple(initial_noise.shape) != expected:
                raise ValueError(
                    f"initial_noise must be {expected}, got {tuple(initial_noise.shape)}"
                )
            actions = initial_noise.to(
                device=vl_embs.device, dtype=vl_embs.dtype
            ).clone()
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_discrete = int(
                step / float(self.num_inference_timesteps) * self.num_timestep_buckets
            )
            timesteps = torch.full(
                (batch_size,), t_discrete, device=vl_embs.device, dtype=torch.long
            )
            velocity, _ = self._predict_velocity(
                vl_embs,
                actions,
                timesteps,
                state,
                embodiment_id,
                option_embedding,
                fusion_enabled,
                encoder_attention_mask,
            )
            actions = actions + dt * velocity
        return actions


@FRAMEWORK_REGISTRY.register("QwenUniversalVLOG")
class QwenUniversalVLOG(baseframework):
    """A single Qwen/VLOG/shared-DiT framework for multiple embodiments."""

    _UNIVERSAL_MARKERS = (
        "vlog_core.",
        "action_model.adapter_bank.",
        "action_model.embodiment_embedding.",
        "action_model.option_conditioner.",
    )
    _BACKWARD_COMPATIBLE_ADDITIONS = (
        "vlog_core.posterior.dynamics_proj.",
        "vlog_core.router_balancer.counts",
        "event_balancers.",
    )

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenUniversalVLOGDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vl_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = vl_dim
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            vl_dim
        )

        self.base_embodiment = str(self.config.framework.base_embodiment)
        self.embodiment_specs = normalize_embodiment_specs(
            self.config.framework.embodiments, self.base_embodiment
        )
        base_spec = self.embodiment_specs[self.base_embodiment]
        action_cfg = self.config.framework.action_model
        if (
            int(action_cfg.state_dim) != base_spec.state_dim
            or int(action_cfg.action_dim) != base_spec.action_dim
        ):
            raise ValueError(
                "Base embodiment dimensions must match action_model.state_dim/action_dim so the "
                "official GR00T checkpoint remains shape-compatible"
            )
        vlog_cfg = self.config.framework.vlog
        self.action_model = UniversalFlowmatchingActionHead(
            full_config=self.config,
            specs=self.embodiment_specs,
            base_embodiment=self.base_embodiment,
            option_dim=int(vlog_cfg.option_dim),
            fusion_residual_scale=float(vlog_cfg.fusion_residual_scale),
            conditioning_mode=str(vlog_cfg.get("conditioning_mode", "legacy_context")),
            bound_conditioning_outputs=bool(
                vlog_cfg.get("bound_conditioning_outputs", False)
            ),
            inject_base_embodiment_token=bool(
                vlog_cfg.get("inject_base_embodiment_token", True)
            ),
        )
        self.action_horizon = int(action_cfg.action_horizon)
        self.vlog_core = UniversalVLOGCore(
            vl_dim=vl_dim,
            dit_dim=self.action_model.input_embedding_dim,
            option_dim=int(vlog_cfg.option_dim),
            num_options=int(vlog_cfg.num_options),
            commitment_cost=float(vlog_cfg.commitment_cost),
            codebook_cosine_margin=float(
                vlog_cfg.get("codebook_cosine_margin", 0.2)
            ),
            usage_entropy_floor_ratio=float(
                vlog_cfg.get("usage_entropy_floor_ratio", 0.75)
            ),
            router_balance=vlog_cfg.get("router_balance", {}),
        )
        event_cfg = vlog_cfg.get("event_balance", {})
        self.event_balance_enabled = bool(event_cfg.get("enabled", False))
        self.event_balance_min_motion = float(event_cfg.get("min_motion", 0.02))
        configured_groups = event_cfg.get("groups", {})
        self.event_groups: dict[str, list[list[int]]] = {}
        self.event_balancers = nn.ModuleDict()
        if self.event_balance_enabled:
            for embodiment, spec in self.embodiment_specs.items():
                raw_groups = configured_groups.get(embodiment, None)
                if raw_groups is None:
                    raise ValueError(
                        "event_balance.enabled=True requires action groups for "
                        f"embodiment {embodiment!r}"
                    )
                groups = [
                    [int(index) for index in group] for group in raw_groups
                ]
                # Validate at construction time rather than failing midway
                # through an expensive U1 run.
                flat = [index for group in groups for index in group]
                if (
                    not flat
                    or len(flat) != len(set(flat))
                    or min(flat) < 0
                    or max(flat) >= spec.action_dim
                ):
                    raise ValueError(
                        f"Event groups must be non-empty, disjoint and in range "
                        f"for {embodiment}: {groups}; "
                        f"action_dim={spec.action_dim}"
                    )
                self.event_groups[embodiment] = groups
                self.event_balancers[embodiment] = RunningClassBalancer(
                    len(groups) + 1,
                    power=float(event_cfg.get("power", 0.5)),
                    mix=float(event_cfg.get("mix", 0.5)),
                    max_weight=float(event_cfg.get("max_weight", 5.0)),
                )
        self.controller = SemiMarkovController(
            d_min=int(vlog_cfg.d_min),
            d_max=int(vlog_cfg.d_max),
            hysteresis=float(vlog_cfg.router_hysteresis),
        )
        self._optimizer_step = 0
        self.configure_train_stage(str(vlog_cfg.train_stage))

    def set_optimizer_step(self, step: int) -> None:
        """Trainer hook used to schedule expensive paired counterfactual forwards."""

        self._optimizer_step = int(step)

    def configure_train_stage(self, stage: str) -> None:
        """Apply an explicit freeze policy; U2 is strictly router-only."""

        stage = stage.lower()
        aliases = {
            "u0": "u0_base",
            "u0-l": "u0_libero",
            "o1": "u1_oracle",
            "u1": "u1_oracle",
            "u2": "u2_router",
            "u3": "u3_joint",
        }
        stage = aliases.get(stage, stage)
        valid = {"u0_base", "u0_libero", "u1_oracle", "u2_router", "u3_joint"}
        if stage not in valid:
            raise ValueError(
                f"Unknown UniversalVLOG train stage {stage!r}; expected {sorted(valid)}"
            )
        self.vlog_train_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == "u0_libero":
            for parameter in self.action_model.adapter_bank.parameters():
                parameter.requires_grad_(True)
            for parameter in self.action_model.embodiment_embedding.parameters():
                parameter.requires_grad_(True)
        elif stage == "u1_oracle":
            for name, parameter in self.vlog_core.named_parameters():
                parameter.requires_grad_(not name.startswith("router."))
            self.action_model.option_conditioner.configure_trainable_parameters(
                True
            )
        elif stage == "u2_router":
            for parameter in self.vlog_core.router.parameters():
                parameter.requires_grad_(True)
        elif stage == "u3_joint":
            for parameter in self.action_model.parameters():
                parameter.requires_grad_(True)
            for parameter in self.vlog_core.parameters():
                parameter.requires_grad_(True)
            for parameter in self.qwen_vl_interface.parameters():
                parameter.requires_grad_(False)
            self.action_model.option_conditioner.configure_trainable_parameters(
                True
            )

    def _autocast(self, dtype: torch.dtype):
        return (
            torch.autocast("cuda", dtype=dtype)
            if torch.cuda.is_available()
            else nullcontext()
        )

    def _encode_batch(
        self, examples: List[dict]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        images = []
        for example in examples:
            raw_images = example["image"]
            views = (
                raw_images if isinstance(raw_images, (list, tuple)) else [raw_images]
            )
            images.append([to_pil_preserve(img) for img in views])
        instructions = [str(example["lang"]) for example in examples]
        train_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_size:
            images = resize_images(images, target_size=train_size)
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions
        )
        mask = inputs.get("attention_mask", None)
        if mask is not None:
            mask = mask.to(dtype=torch.bool)
        with self._autocast(torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1], mask

    def _group_indices(self, examples: List[dict]) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            embodiment = str(example.get("embodiment_id", self.base_embodiment))
            self.action_model.adapter_bank.spec(embodiment)
            grouped[embodiment].append(index)
        return dict(grouped)

    def _state_group(
        self, examples: List[dict], indices: list[int], embodiment: str, device, dtype
    ):
        spec = self.embodiment_specs[embodiment]
        states = []
        for index in indices:
            if "state" not in examples[index]:
                raise ValueError(
                    f"Continuous state is required for QwenUniversalVLOG ({embodiment}, sample {index})"
                )
            state = np.asarray(examples[index]["state"], dtype=np.float32).reshape(
                -1, spec.state_dim
            )
            if state.shape[0] == 0:
                raise ValueError(f"State for sample {index} is empty")
            states.append(state[-1:])
        return torch.as_tensor(np.stack(states), device=device, dtype=dtype)

    def _action_group(
        self, examples: List[dict], indices: list[int], embodiment: str, device, dtype
    ):
        spec = self.embodiment_specs[embodiment]
        actions = torch.zeros(
            len(indices),
            self.action_horizon,
            spec.action_dim,
            device=device,
            dtype=dtype,
        )
        masks = torch.zeros(
            len(indices), self.action_horizon, device=device, dtype=torch.bool
        )
        for row, index in enumerate(indices):
            native = np.asarray(examples[index]["action"], dtype=np.float32).reshape(
                -1, spec.action_dim
            )
            native = native[-self.action_horizon :]
            length = native.shape[0]
            if length == 0:
                raise ValueError(f"Action chunk for sample {index} is empty")
            actions[row, :length] = torch.as_tensor(native, device=device, dtype=dtype)
            supplied_mask = examples[index].get("action_mask", None)
            if supplied_mask is None:
                masks[row, :length] = True
            else:
                supplied = np.asarray(supplied_mask, dtype=bool).reshape(-1)[-length:]
                if supplied.shape[0] != length:
                    raise ValueError(
                        f"action_mask for sample {index} has {supplied.shape[0]} entries, "
                        f"but its action chunk has {length}"
                    )
                masks[row, :length] = torch.as_tensor(supplied, device=device)
        return actions, masks

    def _repeat(self, tensor: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
        if tensor is None or repeats == 1:
            return tensor
        return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))

    def _vlog_enabled(self) -> bool:
        return bool(self.config.framework.vlog.enabled)

    def _fusion_enabled(self) -> bool:
        return self._vlog_enabled() and bool(self.config.framework.vlog.fusion_enabled)

    def _event_balance(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        embodiment: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self._vlog_enabled() or not self.event_balance_enabled:
            return None, None, None
        event_idx, event_energy = classify_action_events(
            actions,
            action_mask,
            self.event_groups[embodiment],
            self.event_balance_min_motion,
        )
        weights = self.event_balancers[embodiment].weights(
            event_idx,
            update=self.training and self.vlog_train_stage in {"u1_oracle", "u3_joint"},
        )
        return weights, event_idx, event_energy

    def _group_forward(
        self,
        vl_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        examples: List[dict],
        indices: list[int],
        embodiment: str,
    ) -> dict:
        index_tensor = torch.tensor(indices, device=vl_hidden.device, dtype=torch.long)
        vl = vl_hidden.index_select(0, index_tensor)
        mask_vl = (
            attention_mask.index_select(0, index_tensor)
            if attention_mask is not None
            else None
        )
        state = self._state_group(examples, indices, embodiment, vl.device, vl.dtype)
        actions, action_mask = self._action_group(
            examples, indices, embodiment, vl.device, vl.dtype
        )
        event_weights, event_idx, event_energy = self._event_balance(
            actions, action_mask, embodiment
        )
        is_router_only = (
            self._vlog_enabled() and self.vlog_train_stage == "u2_router"
        )
        configured_repeats = int(
            self.config.framework.action_model.get("repeated_diffusion_steps", 1)
        )
        # Repeating identical observations only supplies extra FM noise draws.
        # U2 has no FM objective, so replicas would duplicate router labels and
        # waste memory/compute without increasing the observation batch.
        repeats = 1 if is_router_only else configured_repeats
        vl = self._repeat(vl, repeats)
        mask_vl = self._repeat(mask_vl, repeats)
        state = self._repeat(state, repeats)
        actions = self._repeat(actions, repeats)
        action_mask = self._repeat(action_mask, repeats)
        event_weights = self._repeat(event_weights, repeats)
        event_idx = self._repeat(event_idx, repeats)
        event_energy = self._repeat(event_energy, repeats)

        base = None
        t = None
        noise = None
        if not is_router_only:
            t = self.action_model.sample_time(
                actions.shape[0], actions.device, actions.dtype
            )
            noise = torch.randn_like(actions)
            base = self.action_model.flow_forward(
                vl,
                actions,
                state,
                embodiment,
                action_mask=action_mask,
                fusion_enabled=False,
                t=t,
                noise=noise,
                encoder_attention_mask=mask_vl,
            )
        if not self._vlog_enabled() or self.vlog_train_stage in {
            "u0_base",
            "u0_libero",
        }:
            return {
                "total": base["loss"],
                "fm_base": base["loss"].detach(),
                "fm_correct": base["loss"].detach(),
                "fusion_residual_p50": torch.zeros((), device=vl.device),
                "counterfactual_triggered": torch.zeros((), device=vl.device),
            }

        feature_context = torch.no_grad() if is_router_only else nullcontext()
        with feature_context:
            state_tokens = self.action_model.encode_state(state, embodiment)
            embodiment_embedding = self.action_model._embodiment_embedding(
                embodiment, vl.shape[0], vl.device
            )
            state_feature = self.vlog_core.aggregate(
                vl, state_tokens, embodiment_embedding
            )
            zero_t = torch.zeros(
                actions.shape[0], device=actions.device, dtype=torch.long
            )
            future_tokens = self.action_model.encode_action(
                actions, zero_t, embodiment
            )
            oracle = self.vlog_core.discover(
                state_feature,
                future_tokens,
                action_mask,
                sample_weights=event_weights,
            )
        if self.vlog_train_stage == "u1_oracle":
            # Router is explicitly off in U1.  Compute only detached diagnostics
            # so a random frozen router cannot shape discovery representations.
            with torch.no_grad():
                routed = self.vlog_core.route(state_feature.detach())
        else:
            routed = self.vlog_core.route(state_feature)
        router_loss, router_weights = self.vlog_core.router_loss(
            routed["logits"],
            oracle["option_idx"],
            update_counts=self.training
            and self.vlog_train_stage in {"u2_router", "u3_joint"},
        )

        if self.vlog_train_stage == "u2_router":
            router_correct = (
                routed["option_idx"] == oracle["option_idx"].detach()
            ).to(torch.float32)
            result = {
                "total": router_loss,
                "router_loss": router_loss.detach(),
                "router_accuracy": router_correct.mean().detach(),
                "router_confidence": routed["probs"].max(dim=-1).values.mean().detach(),
                "router_weight_max": router_weights.max().detach(),
                "vq_loss": oracle["vq_loss"].detach(),
                "option_entropy": (
                    -(routed["probs"] * (routed["probs"] + 1e-8).log()).sum(-1).mean()
                ).detach(),
                "counterfactual_triggered": torch.zeros((), device=vl.device),
            }
            vlog_cfg = self.config.framework.vlog
            diagnostics_every = max(
                1, int(vlog_cfg.get("router_fm_diagnostics_every", 100))
            )
            trigger_diagnostics = self._optimizer_step % diagnostics_every == 0
            result["router_fm_diagnostics_triggered"] = torch.tensor(
                float(trigger_diagnostics), device=vl.device, dtype=torch.float32
            )
            if trigger_diagnostics:
                diagnostic_t = self.action_model.sample_time(
                    actions.shape[0], actions.device, actions.dtype
                )
                diagnostic_noise = torch.randn_like(actions)
                # Deployment uses the hard, persistent option code rather than
                # the router's soft codebook mixture.  Diagnose that exact path.
                hard_router_option = self.vlog_core.codebook.codebook(
                    routed["option_idx"]
                )
                with torch.no_grad():
                    base_diagnostic = self.action_model.flow_forward(
                        vl,
                        actions,
                        state,
                        embodiment,
                        action_mask=action_mask,
                        fusion_enabled=False,
                        t=diagnostic_t,
                        noise=diagnostic_noise,
                        encoder_attention_mask=mask_vl,
                    )
                    router_fm = self.action_model.flow_forward(
                        vl,
                        actions,
                        state,
                        embodiment,
                        action_mask=action_mask,
                        option_embedding=hard_router_option,
                        fusion_enabled=self._fusion_enabled(),
                        t=diagnostic_t,
                        noise=diagnostic_noise,
                        encoder_attention_mask=mask_vl,
                    )
                residual_norm = router_fm["fusion_residual"].float().norm(dim=-1)
                residual_ratio = router_fm["fusion_residual_ratio"].float()
                result.update(
                    {
                        "fm_base": base_diagnostic["loss"].detach(),
                        "fm_router": router_fm["loss"].detach(),
                        "fusion_residual_p50": residual_norm.quantile(0.5).detach(),
                        "fusion_residual_p95": residual_norm.quantile(0.95).detach(),
                        "fusion_residual_ratio_p50": residual_ratio.quantile(
                            0.5
                        ).detach(),
                        "fusion_residual_ratio_p95": residual_ratio.quantile(
                            0.95
                        ).detach(),
                    }
                )
            return result

        correct = self.action_model.flow_forward(
            vl,
            actions,
            state,
            embodiment,
            action_mask=action_mask,
            # L_correct is always defined by the posterior-discovered oracle
            # option.  The router is trained by distillation and evaluated as
            # a separate deployment path, rather than blurring the causal
            # counterfactual with a soft router mixture.
            option_embedding=oracle["option"],
            fusion_enabled=self._fusion_enabled(),
            t=t,
            noise=noise,
            encoder_attention_mask=mask_vl,
        )
        vlog_cfg = self.config.framework.vlog
        losses_cfg = vlog_cfg.losses
        improve = F.relu(
            float(vlog_cfg.improve_margin) + correct["loss"] - base["loss"]
        )
        counterfactual = torch.zeros((), device=vl.device, dtype=correct["loss"].dtype)
        wrong_loss = torch.full_like(counterfactual, float("nan"))
        cf_every = max(1, int(vlog_cfg.counterfactual_every))
        trigger_cf = self._optimizer_step % cf_every == 0
        if trigger_cf:
            wrong_sampling = str(
                vlog_cfg.get("counterfactual_wrong_sampling", "cyclic")
            )
            if wrong_sampling == "random":
                offsets = torch.randint(
                    1,
                    self.vlog_core.num_options,
                    oracle["option_idx"].shape,
                    device=oracle["option_idx"].device,
                )
                wrong_idx = (
                    oracle["option_idx"] + offsets
                ) % self.vlog_core.num_options
            elif wrong_sampling == "cyclic":
                wrong_idx = (
                    oracle["option_idx"] + 1
                ) % self.vlog_core.num_options
            else:
                raise ValueError(
                    "counterfactual_wrong_sampling must be 'cyclic' or 'random'"
                )
            wrong_option = self.vlog_core.codebook.codebook(wrong_idx)
            wrong = self.action_model.flow_forward(
                vl,
                actions,
                state,
                embodiment,
                action_mask=action_mask,
                option_embedding=wrong_option,
                fusion_enabled=self._fusion_enabled(),
                t=t,
                noise=noise,
                encoder_attention_mask=mask_vl,
            )
            wrong_loss = wrong["loss"]
            margin_mode = str(
                vlog_cfg.get("counterfactual_margin_mode", "absolute")
            )
            if margin_mode == "relative_base":
                counterfactual_margin = (
                    float(vlog_cfg.counterfactual_margin) * base["loss"].detach()
                )
            elif margin_mode == "absolute":
                counterfactual_margin = torch.as_tensor(
                    float(vlog_cfg.counterfactual_margin),
                    device=correct["loss"].device,
                    dtype=correct["loss"].dtype,
                )
            else:
                raise ValueError(
                    "counterfactual_margin_mode must be 'absolute' or 'relative_base'"
                )
            counterfactual = F.relu(
                counterfactual_margin + correct["loss"] - wrong["loss"]
            )
        total = (
            float(losses_cfg.lambda_fm) * correct["loss"]
            + float(losses_cfg.lambda_vq) * oracle["vq_loss"]
            + float(losses_cfg.lambda_commitment) * oracle["commitment_loss"]
            + float(losses_cfg.lambda_usage) * oracle["usage_loss"]
            + float(losses_cfg.get("lambda_codebook_separation", 0.0))
            * oracle["codebook_separation_loss"]
            + float(losses_cfg.lambda_motion) * oracle["motion_loss"]
            + (
                float(losses_cfg.lambda_router) * router_loss
                if self.vlog_train_stage == "u3_joint"
                else 0.0
            )
            + float(losses_cfg.lambda_counterfactual) * counterfactual
            + float(losses_cfg.lambda_improve) * improve
        )
        residual_norm = correct["fusion_residual"].float().norm(dim=-1)
        residual_ratio = correct["fusion_residual_ratio"].float()
        return {
            "total": total,
            "fm_base": base["loss"].detach(),
            "fm_correct": correct["loss"].detach(),
            "fm_oracle": correct["loss"].detach(),
            "fm_wrong": wrong_loss.detach(),
            "router_loss": router_loss.detach(),
            "vq_loss": oracle["vq_loss"].detach(),
            "commitment_loss": oracle["commitment_loss"].detach(),
            "usage_loss": oracle["usage_loss"].detach(),
            "soft_usage_entropy": oracle["soft_usage_entropy"].detach(),
            "codebook_separation_loss": oracle[
                "codebook_separation_loss"
            ].detach(),
            "motion_loss": oracle["motion_loss"].detach(),
            "counterfactual_loss": counterfactual.detach(),
            "improve_loss": improve.detach(),
            "fusion_residual_p50": residual_norm.quantile(0.5).detach(),
            "fusion_residual_p95": residual_norm.quantile(0.95).detach(),
            "fusion_residual_ratio_p50": residual_ratio.quantile(0.5).detach(),
            "fusion_residual_ratio_p95": residual_ratio.quantile(0.95).detach(),
            "option_token_norm_p50": correct["option_token_norm"]
            .float()
            .quantile(0.5)
            .detach(),
            "option_entropy": (
                -(oracle["usage"] * (oracle["usage"] + 1e-8).log()).sum()
            ).detach(),
            "dead_options": (oracle["usage"] < 1e-4).sum().to(torch.float32).detach(),
            "event_weight_max": (
                event_weights.max().detach()
                if event_weights is not None
                else torch.ones((), device=vl.device)
            ),
            "event_active": (
                torch.as_tensor(
                    torch.unique(event_idx).numel(),
                    device=vl.device,
                    dtype=torch.float32,
                )
                if event_idx is not None
                else torch.zeros((), device=vl.device)
            ),
            "counterfactual_triggered": torch.tensor(
                float(trigger_cf), device=vl.device, dtype=torch.float32
            ),
        }

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if not examples:
            raise ValueError("examples must be a non-empty list")
        vl_hidden, attention_mask = self._encode_batch(examples)
        group_outputs = []
        group_weights = []
        # New adapters/VLOG modules start in fp32 while the loaded checkpoint
        # may be bf16.  bf16 autocast keeps mixed module/input dtypes valid;
        # the scalar FM reduction itself is promoted to fp32 in flow_forward.
        with self._autocast(torch.bfloat16):
            for embodiment, indices in self._group_indices(examples).items():
                group_outputs.append(
                    self._group_forward(
                        vl_hidden, attention_mask, examples, indices, embodiment
                    )
                )
                group_weights.append(len(indices))
        total_weight = float(sum(group_weights))
        action_loss = sum(
            output["total"] * (weight / total_weight)
            for output, weight in zip(group_outputs, group_weights)
        )
        result = {"action_loss": action_loss}
        metric_keys = set().union(*(output.keys() for output in group_outputs)) - {
            "total"
        }
        for key in metric_keys:
            values = [
                (output[key], weight)
                for output, weight in zip(group_outputs, group_weights)
                if key in output and torch.isfinite(output[key]).all()
            ]
            if values:
                result[f"vlog/{key}"] = sum(
                    value * (weight / sum(w for _, w in values))
                    for value, weight in values
                )
        return result

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        vl_hidden, attention_mask = self._encode_batch(examples)
        predictions: list[np.ndarray | None] = [None] * len(examples)
        option_info: list[dict | None] = [None] * len(examples)
        fusion = self._fusion_enabled()
        with self._autocast(torch.bfloat16):
            for embodiment, indices in self._group_indices(examples).items():
                idx = torch.tensor(indices, device=vl_hidden.device, dtype=torch.long)
                vl = vl_hidden.index_select(0, idx)
                mask = (
                    attention_mask.index_select(0, idx)
                    if attention_mask is not None
                    else None
                )
                state = self._state_group(
                    examples, indices, embodiment, vl.device, vl.dtype
                )
                option = None
                metadata = [{} for _ in indices]
                if self._vlog_enabled():
                    state_tokens = self.action_model.encode_state(state, embodiment)
                    emb = self.action_model._embodiment_embedding(
                        embodiment, len(indices), vl.device
                    )
                    state_feature = self.vlog_core.aggregate(vl, state_tokens, emb)
                    routed = self.vlog_core.route(state_feature)
                    stream_ids = [examples[i].get("env_id", i) for i in indices]
                    episode_ids = [examples[i].get("episode_id", None) for i in indices]
                    active_idx, metadata = self.controller.update(
                        routed["logits"], stream_ids, episode_ids
                    )
                    option = self.vlog_core.codebook.codebook(active_idx)

                seed = kwargs.get("fm_noise_seed", None)
                initial_noise = None
                if seed is not None:
                    spec = self.embodiment_specs[embodiment]
                    chunks = []
                    for global_index in indices:
                        generator = torch.Generator(device=vl.device)
                        generator.manual_seed(int(seed) + int(global_index))
                        chunks.append(
                            torch.randn(
                                self.action_horizon,
                                spec.action_dim,
                                generator=generator,
                                device=vl.device,
                                dtype=vl.dtype,
                            )
                        )
                    initial_noise = torch.stack(chunks)
                native = self.action_model.predict_native_action(
                    vl,
                    state,
                    embodiment,
                    option_embedding=option,
                    fusion_enabled=fusion,
                    encoder_attention_mask=mask,
                    initial_noise=initial_noise,
                )
                for row, global_index in enumerate(indices):
                    predictions[global_index] = (
                        native[row].detach().float().cpu().numpy()
                    )
                    option_info[global_index] = metadata[row]
                if self._vlog_enabled():
                    self.controller.record_action_chunks(
                        [examples[i].get("env_id", i) for i in indices], native
                    )
        homogeneous = len({prediction.shape[-1] for prediction in predictions}) == 1
        normalized = np.stack(predictions) if homogeneous else predictions
        return {
            "normalized_actions": normalized,
            "normalized_actions_by_example": predictions,
            "vlog_info": option_info,
        }

    def reset_option_state(self, env_id=None) -> None:
        self.controller.reset(env_id)

    def _base_core_keys(self) -> set[str]:
        keys = set()
        action_base_prefixes = (
            "action_model.model.",
            "action_model.state_encoder.",
            "action_model.action_encoder.",
            "action_model.action_decoder.",
            "action_model.future_tokens.",
            "action_model.position_embedding.",
        )
        for key in self.state_dict():
            if key.startswith("qwen_vl_interface.") or key.startswith(
                action_base_prefixes
            ):
                keys.add(key)
        return keys

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Reject partial/mismatched official base loads even when trainer asks for strict=False."""

        keys = set(state_dict)
        is_universal_checkpoint = any(
            any(key.startswith(marker) for marker in self._UNIVERSAL_MARKERS)
            for key in keys
        )
        if not is_universal_checkpoint:
            if any(key.startswith("vlog.") or "QwenOFT" in key for key in keys):
                raise RuntimeError(
                    "QwenOFT/VLOG-OFT checkpoints are incompatible with QwenUniversalVLOG"
                )
            current = self.state_dict()
            core = self._base_core_keys()
            missing = sorted(core - keys)
            mismatched = sorted(
                key
                for key in core & keys
                if tuple(current[key].shape) != tuple(state_dict[key].shape)
            )
            if missing or mismatched:
                details = []
                if missing:
                    details.append(f"missing core keys ({len(missing)}): {missing[:8]}")
                if mismatched:
                    shape_text = [
                        f"{key}: model={tuple(current[key].shape)} checkpoint={tuple(state_dict[key].shape)}"
                        for key in mismatched[:8]
                    ]
                    details.append(
                        "shape-mismatched core keys: " + "; ".join(shape_text)
                    )
                raise RuntimeError(
                    "Refusing lossy GR00T warm-start; instantiate with the checkpoint's exact "
                    "Qwen/tokenizer/action config. " + " | ".join(details)
                )
            strict = False  # only UniversalVLOG additions may be absent
        elif strict:
            # `from_pretrained` is strict, whereas trainer warm-starts are not.
            # Permit only the explicitly zero/safely initialized keys added by
            # the duration-bias refactor; every older core key remains audited.
            missing = set(self.state_dict()) - keys
            if missing and all(
                key.startswith(self._BACKWARD_COMPATIBLE_ADDITIONS)
                for key in missing
            ):
                strict = False
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:  # torch versions before the assign= argument
            return super().load_state_dict(state_dict, strict=strict)


__all__ = [
    "QwenUniversalVLOG",
    "QwenUniversalVLOGDefaultConfig",
    "UniversalFlowmatchingActionHead",
    "UniversalVLOGCore",
]
