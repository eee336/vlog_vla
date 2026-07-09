from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .losses import behavior_preservation_loss, option_balance_loss, router_distill_loss
from .starvla_hidden_adapter import StarVLAHiddenAdapter
from .vlog_policy_wrapper import PersistenceConfig, VLOGPolicyWrapper


class RealStarVLAVLOGWrapper(nn.Module):
    """Real StarVLA hidden-token VLOG wrapper."""

    def __init__(self, base_policy, cfg: dict) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.cfg = cfg
        vlog_cfg = cfg.get("vlog", {})
        if bool(vlog_cfg.get("allow_surrogate_hidden", False)):
            raise ValueError("Stage 7 requires allow_surrogate_hidden=false.")
        self.hidden_adapter = StarVLAHiddenAdapter(base_policy, {"allow_surrogate_hidden": False})
        hidden_dim = int(vlog_cfg.get("hidden_dim", _infer_hidden_dim(base_policy)))
        action_dim = int(vlog_cfg.get("action_dim", _infer_action_dim(base_policy)))
        state_dim = int(vlog_cfg.get("state_dim", cfg.get("dataset", {}).get("state_dim", 8)))
        option_dim = int(vlog_cfg.get("option_dim", 256))
        num_options = int(vlog_cfg.get("num_options", 16))
        window_size = int(vlog_cfg.get("window_size", cfg.get("dataset", {}).get("window_size", 16)))
        self.vlog = VLOGPolicyWrapper(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            num_options=num_options,
            option_dim=option_dim,
            window_size=window_size,
            persistence=PersistenceConfig(
                d_min=int(vlog_cfg.get("d_min", 2)),
                d_max=int(vlog_cfg.get("d_max", 8)),
                beta_threshold=float(vlog_cfg.get("beta_threshold", 0.7)),
                q_switch_margin=float(vlog_cfg.get("q_switch_margin", 0.05)),
            ),
            commitment_cost=float(vlog_cfg.get("commitment_cost", 0.25)),
            alpha_init=0.0,
        )

    def forward_train(self, batch, stage: str = "stage1_preserve") -> dict[str, Any]:
        hidden, aux = self.hidden_adapter.encode_hidden(batch["examples"] if isinstance(batch, dict) and "examples" in batch else batch)
        robot_state = _state_tensor(batch, hidden.device, hidden.dtype, self.vlog.state_dim)
        future_actions = _future_actions(batch, hidden.device, hidden.dtype, self.vlog.window_size, self.vlog.action_dim)
        out = self.vlog.forward_train({"hidden_tokens": hidden, "robot_state": robot_state, "future_actions": future_actions}, stage=_stage_number(stage))
        base_action = self.hidden_adapter.decode_action_from_hidden(hidden, batch, aux=aux)
        vlog_action = self.hidden_adapter.decode_action_from_hidden(out["adapted_hidden_tokens"], batch, aux=aux)
        preserve = behavior_preservation_loss(vlog_action, base_action)
        result = {
            **out,
            "base_action": base_action,
            "vlog_action": vlog_action,
            "preservation_loss": preserve,
            "uses_real_starvla_hidden": True,
            "uses_surrogate_hidden": False,
            "hidden_aux": aux,
        }
        if stage == "stage2_option_discovery":
            result["router_distill_loss"] = router_distill_loss(out["router_logits"], out.get("posterior_idx", out["router_idx"]))
            result["option_balance_loss"] = option_balance_loss(out["option_probs"])
        return result

    @torch.no_grad()
    def predict_action(self, obs):
        hidden, aux = self.hidden_adapter.encode_hidden(obs)
        robot_state = _state_tensor(obs, hidden.device, hidden.dtype, self.vlog.state_dim)
        info = self.vlog.select_option(hidden, robot_state=robot_state)
        actions = self.hidden_adapter.decode_action_from_hidden(info["adapted_hidden_tokens"], obs, aux=aux)
        return {
            "normalized_actions": actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions),
            "vlog_info": {
                "option_idx": int(info["selected_option_idx"]),
                "option_age": int(info["option_age"]),
                "beta": float(info["beta"].detach().float().cpu().reshape(-1)[0]),
                "adapter_alpha": float(self.vlog.option_adapter.alpha.detach().cpu().item()),
            },
        }

    def reset_option_state(self) -> None:
        self.vlog.reset_option_state()


def _infer_hidden_dim(base_policy) -> int:
    return int(getattr(base_policy.config.framework.action_model, "action_hidden_dim", 256))


def _infer_action_dim(base_policy) -> int:
    return int(getattr(base_policy.config.framework.action_model, "action_dim", 7))


def _state_tensor(batch, device, dtype, state_dim: int) -> torch.Tensor:
    source = batch
    if isinstance(batch, dict) and "examples" in batch:
        source = batch["examples"]
    examples = source if isinstance(source, list) else [source]
    if examples and isinstance(examples[0], dict) and "state" in examples[0]:
        arr = np.asarray([ex["state"] for ex in examples], dtype=np.float32).reshape(len(examples), -1)
    elif isinstance(batch, dict) and "robot_state" in batch:
        arr = np.asarray(batch["robot_state"], dtype=np.float32).reshape(-1, state_dim)
    else:
        arr = np.zeros((len(examples), state_dim), dtype=np.float32)
    if arr.shape[1] < state_dim:
        arr = np.pad(arr, ((0, 0), (0, state_dim - arr.shape[1])))
    return torch.tensor(arr[:, :state_dim], device=device, dtype=dtype)


def _future_actions(batch, device, dtype, window_size: int, action_dim: int) -> torch.Tensor:
    if isinstance(batch, dict) and "future_actions" in batch:
        tensor = torch.as_tensor(batch["future_actions"], device=device, dtype=dtype)
    elif isinstance(batch, dict) and "action" in batch:
        tensor = torch.as_tensor(batch["action"], device=device, dtype=dtype)
    elif isinstance(batch, dict) and "examples" in batch and "action" in batch["examples"][0]:
        tensor = torch.as_tensor(np.asarray([ex["action"] for ex in batch["examples"]]), device=device, dtype=dtype)
    else:
        tensor = torch.zeros(1, window_size, action_dim, device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] < window_size:
        tensor = torch.cat([tensor, tensor[:, -1:, :].expand(-1, window_size - tensor.shape[1], -1)], dim=1)
    return tensor[:, :window_size, :action_dim]


def _stage_number(stage: str) -> int:
    return {
        "stage1_preserve": 1,
        "stage2_option_discovery": 2,
        "stage3_graph": 3,
        "stage4_critic": 4,
        "stage5_router": 5,
        "stage6_full": 6,
    }.get(stage, 1)
