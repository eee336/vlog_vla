from __future__ import annotations

import torch
import torch.nn.functional as F


def behavior_preservation_loss(action_vlog: torch.Tensor, action_base: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(action_vlog, action_base.detach())


def action_bc_loss(action_pred: torch.Tensor, action_target: torch.Tensor, loss_type: str = "l1") -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(action_pred, action_target)
    return F.l1_loss(action_pred, action_target)


def future_delta_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_delta, target_delta)


def router_distill_loss(router_logits: torch.Tensor, posterior_idx_or_probs: torch.Tensor) -> torch.Tensor:
    if posterior_idx_or_probs.ndim == 1:
        return F.cross_entropy(router_logits, posterior_idx_or_probs.detach())
    target = posterior_idx_or_probs.detach()
    log_probs = F.log_softmax(router_logits, dim=-1)
    return F.kl_div(log_probs, target, reduction="batchmean")


def option_balance_loss(option_probs: torch.Tensor) -> torch.Tensor:
    avg_probs = option_probs.mean(dim=0)
    target = torch.ones_like(avg_probs) / avg_probs.numel()
    return F.kl_div((avg_probs + 1e-8).log(), target, reduction="batchmean")


def temporal_consistency_loss(probs_t: torch.Tensor, probs_next: torch.Tensor) -> torch.Tensor:
    return F.kl_div((probs_t + 1e-8).log(), probs_next.detach(), reduction="batchmean")


def switch_penalty(probs_t: torch.Tensor, probs_prev: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.sum(probs_t * probs_prev, dim=-1).mean()


def graph_transition_loss(edge_logits: torch.Tensor, option_idx_t: torch.Tensor, option_idx_next: torch.Tensor) -> torch.Tensor:
    batch = edge_logits.shape[0]
    rows = edge_logits[torch.arange(batch, device=edge_logits.device), option_idx_t]
    return F.cross_entropy(rows, option_idx_next)


def edge_sparse_loss(edge_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(edge_logits).mean()


def conservative_option_critic_loss(
    q_data: torch.Tensor,
    q_all: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    next_q_value: torch.Tensor,
    gamma: float = 0.99,
    alpha_cql: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    q_data_safe = torch.nan_to_num(q_data, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    q_all_safe = torch.nan_to_num(q_all, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    next_q_safe = torch.nan_to_num(next_q_value, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
    reward_safe = torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=0.0)
    done_safe = torch.nan_to_num(done.float(), nan=0.0, posinf=1.0, neginf=0.0)
    target = reward_safe + gamma * (1.0 - done_safe) * next_q_safe.detach()
    td_loss = F.mse_loss(q_data_safe, target)
    # Clamp input to logsumexp to avoid overflow -> NaN corruption.
    cql_loss = torch.logsumexp(q_all_safe, dim=-1).mean() - q_data_safe.mean()
    loss = td_loss + alpha_cql * cql_loss
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
    return loss, {
        "td_loss": td_loss.detach(),
        "cql_loss": cql_loss.detach(),
        "q_data": q_data_safe.detach().mean(),
        "q_all": q_all_safe.detach().mean(),
        "target_q_mean": target.detach().mean(),
    }


def termination_loss(beta: torch.Tensor, boundary_label: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=beta.device.type, enabled=False):
        beta_fp32 = torch.nan_to_num(beta.float(), nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6).clamp(1e-6, 1.0 - 1e-6)
        label_fp32 = torch.nan_to_num(boundary_label.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return F.binary_cross_entropy(beta_fp32, label_fp32)
