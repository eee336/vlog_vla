from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

from .losses import (
    conservative_option_critic_loss,
    edge_sparse_loss,
    graph_transition_loss,
    option_balance_loss,
    router_distill_loss,
    switch_penalty,
    temporal_consistency_loss,
    termination_loss,
)
from .vlog_policy_wrapper import PersistenceConfig, VLOGPolicyWrapper


@FRAMEWORK_REGISTRY.register("QwenOFTVLOG")
class QwenOFTVLOG(Qwenvl_OFT):
    """QwenOFT with persistent VLOG modules inserted before action decoding."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        vlog_cfg = self.config.framework.get("vlog", {})
        hidden_dim = int(self.config.framework.action_model.action_hidden_dim)
        action_dim = int(self.config.framework.action_model.action_dim)
        state_dim = int(self.config.framework.action_model.get("state_dim", 8))
        option_dim = int(vlog_cfg.get("option_dim", 256))
        num_options = int(vlog_cfg.get("num_options", 16))
        window_size = int(vlog_cfg.get("window_size", self.action_horizon))
        self.vlog_train_stage = str(vlog_cfg.get("train_stage", "stage1_preserve"))
        self.vlog_log_dir = None
        self.vlog_infer_step = 0
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
            alpha_init=0.0 if bool(vlog_cfg.get("zero_init_adapter", True)) else float(vlog_cfg.get("alpha_init", 0.0)),
        )
        if bool(vlog_cfg.get("freeze_base", False)):
            for param in self.qwen_vl_interface.parameters():
                param.requires_grad_(False)
            for param in self.action_model.parameters():
                param.requires_grad_(False)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        batch_images, instructions, actions, state = self._prepare_batch_fields(examples)
        instructions = self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        instructions = self._append_action_prompt(instructions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)
            robot_state = self._state_tensor(state, action_queries.device, action_queries.dtype)
            target_actions = torch.tensor(np.array(actions), device=action_queries.device, dtype=action_queries.dtype)
            target_actions = target_actions[:, -self.action_horizon :, :]
            future_actions = self._future_action_window(target_actions, self.vlog.window_size)

            vlog_out = self.vlog.forward_train(
                {"hidden_tokens": action_queries, "robot_state": robot_state, "future_actions": future_actions},
                stage=self._stage_number(),
            )
            pred_base = self.action_model.predict_action(action_queries)
            pred_vlog = self.action_model.predict_action(vlog_out["adapted_hidden_tokens"])
            action_loss = self.l1_loss(pred_vlog, target_actions)
            loss_dict = self._vlog_losses(vlog_out, pred_vlog, pred_base, target_actions)
            loss_dict["action_loss"] = loss_dict.pop("total_loss")
        return loss_dict

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs: str) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        instructions = self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = self._append_action_prompt(instructions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)
            robot_state = self._state_tensor(state, action_queries.device, action_queries.dtype)
            vlog_info = self.vlog.select_option(action_queries, robot_state=robot_state)
            pred_actions = self.action_model.predict_action(vlog_info["adapted_hidden_tokens"])
        normalized_actions = pred_actions.detach().cpu().numpy()
        info = self._format_vlog_info(vlog_info)
        self._write_vlog_log(info)
        return {"normalized_actions": normalized_actions, "vlog_info": info}

    def reset_option_state(self) -> None:
        self.vlog.reset_option_state()

    def set_vlog_logging(self, log_dir: str | None) -> None:
        self.vlog_log_dir = Path(log_dir) if log_dir else None
        if self.vlog_log_dir:
            self.vlog_log_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_batch_fields(self, examples: List[dict]):
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        return batch_images, instructions, actions, state

    def _append_action_prompt(self, instructions: List[str]) -> List[str]:
        action_tokens = self.action_token * self.chunk_len
        suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        return [instruction + suffix for instruction in instructions]

    def _state_tensor(self, state, device, dtype) -> torch.Tensor:
        state_dim = int(self.config.framework.action_model.get("state_dim", 8))
        if state is None:
            return torch.zeros(1, state_dim, device=device, dtype=dtype)
        arr = np.array(state)
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] < state_dim:
            arr = np.pad(arr, ((0, 0), (0, state_dim - arr.shape[1])))
        return torch.tensor(arr[:, :state_dim], device=device, dtype=dtype)

    def _future_action_window(self, target_actions: torch.Tensor, window_size: int) -> torch.Tensor:
        if target_actions.shape[1] >= window_size:
            return target_actions[:, :window_size, :]
        pad = target_actions[:, -1:, :].expand(-1, window_size - target_actions.shape[1], -1)
        return torch.cat([target_actions, pad], dim=1)

    def _stage_number(self) -> int:
        mapping = {
            "stage1_preserve": 1,
            "stage2_option_discovery": 2,
            "stage3_graph": 3,
            "stage4_critic": 4,
            "stage5_router": 5,
            "stage6_full": 6,
        }
        return mapping.get(self.vlog_train_stage, 6)

    def _vlog_losses(self, out: dict, pred_vlog: torch.Tensor, pred_base: torch.Tensor, target: torch.Tensor) -> dict:
        cfg = self.config.framework.vlog.get("losses", {})
        device = pred_vlog.device
        action_loss = self.l1_loss(pred_vlog, target)
        preserve_loss = self.l1_loss(pred_vlog, pred_base.detach())
        posterior_idx = out.get("posterior_idx", out["router_idx"])
        next_idx = torch.roll(posterior_idx, shifts=-1, dims=0)
        balance = option_balance_loss(out["option_probs"])
        distill = router_distill_loss(out["router_logits"], posterior_idx)
        vq = out.get("vq_loss", torch.zeros((), device=device))
        transition = graph_transition_loss(out["edge_logits"], posterior_idx, next_idx)
        sparse = edge_sparse_loss(out["edge_logits"])
        q_data = out["q_all"][torch.arange(out["q_all"].shape[0], device=device), posterior_idx]
        reward = torch.zeros_like(q_data)
        reward[-1] = 1.0
        done = torch.zeros_like(q_data)
        done[-1] = 1.0
        next_q = torch.max(out["q_all"].detach(), dim=-1).values
        critic_loss, critic_metrics = conservative_option_critic_loss(
            q_data,
            out["q_all"],
            reward,
            done,
            next_q,
            gamma=float(cfg.get("gamma", 0.99)),
            alpha_cql=float(cfg.get("alpha_cql", 0.1)),
        )
        q_mean = out["q_all"].mean(dim=-1)
        adv = torch.clamp(q_data - q_mean, -5.0, 5.0).detach()
        logp = torch.log_softmax(out["router_logits"], dim=-1)[torch.arange(posterior_idx.shape[0], device=device), posterior_idx]
        router_adv = -(logp * torch.exp(adv)).mean()
        boundary = (posterior_idx != next_idx).float()
        value_label = (torch.max(out["q_all"], dim=-1).values > q_data + float(self.config.framework.vlog.get("q_switch_margin", 0.05))).float()
        term = termination_loss(out["beta"], boundary)
        value_term = termination_loss(out["beta"], value_label)
        consistency = torch.zeros((), device=device)
        switch = torch.zeros((), device=device)
        stage = self._stage_number()
        if stage == 1:
            total = preserve_loss
        elif stage == 2:
            total = (
                action_loss
                + float(cfg.get("lambda_vq", 1.0)) * vq
                + float(cfg.get("lambda_distill", 1.0)) * distill
                + float(cfg.get("lambda_balance", 0.05)) * balance
            )
        elif stage == 3:
            total = transition + float(cfg.get("lambda_sparse", 0.01)) * sparse + float(cfg.get("lambda_balance", 0.05)) * balance
        elif stage == 4:
            total = critic_loss
        elif stage == 5:
            total = action_loss + float(cfg.get("lambda_router", 0.5)) * router_adv + float(cfg.get("lambda_critic", 0.5)) * critic_loss
        else:
            total = action_loss + float(cfg.get("lambda_router", 0.3)) * router_adv + float(cfg.get("lambda_term", 0.1)) * term + float(cfg.get("lambda_value_term", 0.05)) * value_term
        metrics = {
            "total_loss": total,
            "vlog/action_loss": action_loss.detach(),
            "vlog/preserve_loss": preserve_loss.detach(),
            "vlog/vq_loss": vq.detach(),
            "vlog/router_distill_loss": distill.detach(),
            "vlog/option_balance_loss": balance.detach(),
            "vlog/graph_transition_loss": transition.detach(),
            "vlog/edge_sparse_loss": sparse.detach(),
            "vlog/critic_td_loss": critic_metrics["td_loss"].detach(),
            "vlog/critic_cql_loss": critic_metrics["cql_loss"].detach(),
            "vlog/termination_loss": term.detach(),
            "vlog/value_termination_loss": value_term.detach(),
            "vlog/router_adv_loss": router_adv.detach(),
            "vlog/mean_q_current": q_data.detach().mean(),
            "vlog/mean_beta": out["beta"].detach().mean(),
            "vlog/adapter_alpha": self.vlog.option_adapter.alpha.detach().mean(),
            "vlog/option_entropy": self._entropy(out["option_probs"]).detach(),
            "vlog/dead_options": (out["option_probs"].mean(dim=0) < 1e-4).sum().float().detach(),
            "vlog/switch_frequency": boundary.mean().detach(),
        }
        return metrics

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        avg = probs.mean(dim=0)
        return -(avg * (avg + 1e-8).log()).sum()

    def _format_vlog_info(self, info: dict) -> dict:
        q_all = info["q_all"].detach().float().cpu().reshape(-1).tolist()
        selected = int(info["selected_option_idx"])
        return {
            "timestep": int(self.vlog_infer_step),
            "selected_option_idx": selected,
            "option_age": int(info["option_age"]),
            "option_probs": info["router_probs"].detach().float().cpu().reshape(-1).tolist(),
            "q_current": float(q_all[selected]),
            "q_all": q_all,
            "beta": float(info["beta"].detach().float().cpu().reshape(-1)[0]),
            "switch_reason": "init_or_policy" if info["switched"] else "keep",
            "adapter_alpha": float(self.vlog.option_adapter.alpha.detach().cpu().item()),
        }

    def _write_vlog_log(self, info: dict) -> None:
        self.vlog_infer_step += 1
        if self.vlog_log_dir is None:
            return
        with (self.vlog_log_dir / "vlog_option_timeline.jsonl").open("a") as f:
            f.write(json.dumps(info, sort_keys=True) + "\n")
