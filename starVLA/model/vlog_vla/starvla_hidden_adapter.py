from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import resize_images


class StarVLAHiddenAdapter(nn.Module):
    """Extract and decode real StarVLA/QwenOFT action-token hidden states.

    The adapter is intentionally narrow: it supports StarVLA QwenOFT-style
    policies exposing `qwen_vl_interface`, `_gather_action_token_embeddings`,
    and `action_model.predict_action`. It does not fall back to surrogate
    hidden tokens when `allow_surrogate_hidden=False`.
    """

    def __init__(self, base_policy, cfg: dict | None = None) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.cfg = cfg or {}
        self.allow_surrogate_hidden = bool(self.cfg.get("allow_surrogate_hidden", False))

    def encode_hidden(self, batch_or_obs) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self._supports_real_qwenoft_path():
            if self.allow_surrogate_hidden:
                raise RuntimeError("Surrogate hidden fallback is intentionally not implemented for Stage 7.")
            raise RuntimeError(
                "Real StarVLA hidden extraction requires qwen_vl_interface, action_model, "
                "action_token_id, and _gather_action_token_embeddings."
            )
        examples = self._as_examples(batch_or_obs)
        batch_images = [self._normalize_images(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [self._normalize_state(example["state"]) for example in examples] if "state" in examples[0] else None
        if state is not None and hasattr(self.base_policy, "add_discretized_state_to_instruction"):
            instructions = self.base_policy.add_discretized_state_to_instruction(instructions, state)
        train_obs_image_size = getattr(self.base_policy.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = self._append_action_prompt(instructions)
        qwen_inputs = self.base_policy.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        device = next(self.base_policy.parameters()).device
        qwen_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in qwen_inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = self.base_policy.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = outputs.hidden_states[-1]
        input_ids = qwen_inputs.get("input_ids")
        action_queries = self.base_policy._gather_action_token_embeddings(
            last_hidden,
            input_ids,
            action_token_id=self.base_policy.action_token_id,
        )
        aux = {
            "qwen_inputs": qwen_inputs,
            "last_hidden_shape": list(last_hidden.shape),
            "input_ids_shape": list(input_ids.shape) if input_ids is not None else None,
            "uses_real_starvla_hidden": True,
            "uses_surrogate_hidden": False,
        }
        return action_queries, aux

    def decode_action_from_hidden(self, hidden_tokens: torch.Tensor, batch_or_obs, aux: dict | None = None):
        try:
            param = next(self.base_policy.action_model.parameters())
            hidden_tokens = hidden_tokens.to(device=param.device, dtype=param.dtype)
        except StopIteration:
            pass
        return self.base_policy.action_model.predict_action(hidden_tokens)

    def forward_base_action(self, batch_or_obs):
        return self.base_policy.predict_action(examples=self._as_examples(batch_or_obs))

    def _supports_real_qwenoft_path(self) -> bool:
        if not hasattr(self.base_policy, "action_model"):
            return False
        return all(
            [
                hasattr(self.base_policy, "qwen_vl_interface"),
                hasattr(self.base_policy, "action_model"),
                hasattr(self.base_policy.action_model, "predict_action"),
                hasattr(self.base_policy, "_gather_action_token_embeddings"),
                hasattr(self.base_policy, "action_token_id"),
            ]
        )

    def _append_action_prompt(self, instructions: list[str]) -> list[str]:
        chunk_len = int(getattr(self.base_policy, "chunk_len", getattr(self.base_policy, "action_horizon", 8)))
        action_token = getattr(self.base_policy, "action_token", "🔍")
        action_tokens = action_token * chunk_len
        suffix = f" Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."
        return [instruction + suffix for instruction in instructions]

    def _as_examples(self, batch_or_obs) -> list[dict]:
        if isinstance(batch_or_obs, list):
            return batch_or_obs
        return [batch_or_obs]

    def _normalize_state(self, state):
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def _normalize_images(self, image):
        if isinstance(image, (list, tuple)):
            return [to_pil_preserve(img) for img in image]
        return [to_pil_preserve(image)]


def action_diff_report(base_action, vlog_action) -> dict:
    base = torch.as_tensor(base_action if not isinstance(base_action, dict) else base_action["normalized_actions"])
    vlog = torch.as_tensor(vlog_action if not isinstance(vlog_action, dict) else vlog_action["normalized_actions"])
    diff = torch.abs(vlog.float() - base.float())
    return {
        "mean_l1_action_diff": float(diff.mean().item()),
        "max_l1_action_diff": float(diff.max().item()),
    }
