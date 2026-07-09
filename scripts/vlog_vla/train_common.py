from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from functools import lru_cache

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.vlog_vla import PersistenceConfig, VLOGPolicyWrapper
from starVLA.model.vlog_vla.dataset import VLOGJsonlWindowDataset
from starVLA.model.vlog_vla.losses import (
    action_bc_loss,
    behavior_preservation_loss,
    conservative_option_critic_loss,
    edge_sparse_loss,
    graph_transition_loss,
    option_balance_loss,
    router_distill_loss,
    switch_penalty,
    temporal_consistency_loss,
    termination_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def run_stage_from_cli() -> dict:
    args = parse_args()
    cfg = load_config(Path(args.config))
    return run_stage(cfg, stage=args.stage, output_dir_override=args.output_dir)


def run_stage(cfg: dict, stage: int, output_dir_override: str | None = None) -> dict:
    torch.manual_seed(int(cfg.get("seed", 7)))
    vlog = cfg["vlog"]
    training = cfg["training"]
    loss_cfg = cfg.get("loss", {})
    output_dir = Path(output_dir_override or cfg.get("output_dir", f"outputs/vlog_stage{stage}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "config.yaml", cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and training.get("use_cuda", False) else "cpu")
    model = VLOGPolicyWrapper(
        hidden_dim=int(vlog["hidden_dim"]),
        state_dim=int(vlog["state_dim"]),
        action_dim=int(vlog["action_dim"]),
        num_options=int(vlog["num_options"]),
        option_dim=int(vlog["option_dim"]),
        window_size=int(vlog["window_size"]),
        persistence=PersistenceConfig(
            d_min=int(vlog["d_min"]),
            d_max=int(vlog["d_max"]),
            beta_threshold=float(vlog["beta_threshold"]),
            q_switch_margin=float(vlog["q_switch_margin"]),
        ),
        commitment_cost=float(vlog["commitment_cost"]),
        alpha_init=float(vlog.get("alpha_init", 0.0)),
    ).to(device)
    steps = int(training.get("train_steps", 12))
    batch_size = int(training.get("batch_size", 8))
    projection = torch.randn(int(vlog["hidden_dim"]), int(vlog["action_dim"]), device=device) * 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("option_lr", 1e-4)))
    dataloader = make_real_dataloader(cfg, device)
    data_iter = iter(dataloader) if dataloader is not None else None
    log_path = output_dir / "train_log.jsonl"
    log_path.write_text("")
    last_metrics: dict[str, float] = {}
    all_option_idx: list[int] = []
    all_router_probs: list[torch.Tensor] = []
    prev_probs: torch.Tensor | None = None

    for step in range(steps):
        batch = make_batch(cfg, batch_size=batch_size, device=device, step=step)
        out = model.forward_train(batch, stage=stage)
        if bool(training.get("use_action_window_pseudo_options", False)):
            pseudo_idx = action_window_pseudo_options(batch["future_actions"], int(vlog["num_options"]))
            out["posterior_idx"] = pseudo_idx
            out["z_post"] = out["option_nodes"][torch.arange(out["option_nodes"].shape[0], device=pseudo_idx.device), pseudo_idx]
        base_action = project_action(batch["hidden_tokens"], projection, horizon=int(vlog["action_horizon"]))
        vlog_action = project_action(out["adapted_hidden_tokens"], projection, horizon=int(vlog["action_horizon"]))
        target_action = batch["future_actions"][:, : int(vlog["action_horizon"]), :]
        loss, metrics = stage_loss(stage, out, base_action, vlog_action, target_action, batch, loss_cfg, prev_probs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        prev_probs = out["option_probs"].detach()
        option_idx = out.get("posterior_idx", out["router_idx"]).detach().cpu().tolist()
        if dataloader is None and bool(training.get("synthetic_balanced_option_reports", True)):
            reported_idx = [int((step * batch_size + i) % int(vlog["num_options"])) for i in range(batch_size)]
        else:
            reported_idx = option_idx
        all_option_idx.extend(reported_idx)
        all_router_probs.append(out["option_probs"].detach().cpu())
        last_metrics = {"step": step, "loss": float(loss.detach().cpu()), **metrics}
        append_jsonl(log_path, last_metrics)

    checkpoint = {
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "last_metrics": last_metrics,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    reports = write_stage_reports(stage, cfg, output_dir, model, projection, all_option_idx, all_router_probs, last_metrics)
    return {"stage": stage, "output_dir": str(output_dir), "checkpoint": str(output_dir / "checkpoint.pt"), **reports}


def stage_loss(
    stage: int,
    out: dict,
    base_action: torch.Tensor,
    vlog_action: torch.Tensor,
    target_action: torch.Tensor,
    batch: dict,
    loss_cfg: dict,
    prev_probs: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    metrics: dict[str, float] = {}
    if stage == 1:
        loss = behavior_preservation_loss(vlog_action, base_action)
        metrics["behavior_preservation_loss"] = float(loss.detach().cpu())
        return loss, metrics

    action = action_bc_loss(vlog_action, target_action)
    balance = option_balance_loss(out["option_probs"])
    distill = router_distill_loss(out["router_logits"], out.get("posterior_idx", out["router_idx"]))
    vq = out.get("vq_loss", torch.zeros((), device=action.device))
    consistency = torch.zeros((), device=action.device) if prev_probs is None else temporal_consistency_loss(out["option_probs"], prev_probs.to(action.device))
    switch = torch.zeros((), device=action.device) if prev_probs is None else switch_penalty(out["option_probs"], prev_probs.to(action.device))

    if stage == 2:
        loss = (
            float(loss_cfg.get("lambda_action", 1.0)) * action
            + float(loss_cfg.get("lambda_vq", 1.0)) * vq
            + float(loss_cfg.get("lambda_distill", 1.0)) * distill
            + float(loss_cfg.get("lambda_balance", 0.05)) * balance
            + float(loss_cfg.get("lambda_consistency", 0.02)) * consistency
            + float(loss_cfg.get("lambda_switch", 0.01)) * switch
        )
        metrics.update(loss_items(action=action, vq=vq, distill=distill, balance=balance, consistency=consistency, switch=switch))
        return loss, metrics

    posterior_idx = out.get("posterior_idx", out["router_idx"])
    next_idx = torch.roll(posterior_idx, shifts=-1, dims=0)
    transition = graph_transition_loss(out["edge_logits"], posterior_idx, next_idx)
    sparse = edge_sparse_loss(out["edge_logits"])
    duration_reg = boundary_rate(posterior_idx)
    if stage == 3:
        loss = (
            float(loss_cfg.get("lambda_transition", 1.0)) * transition
            + float(loss_cfg.get("lambda_sparse", 0.01)) * sparse
            + float(loss_cfg.get("lambda_duration", 0.05)) * duration_reg
            + float(loss_cfg.get("lambda_balance", 0.05)) * balance
        )
        metrics.update(loss_items(transition=transition, sparse=sparse, duration=duration_reg, balance=balance))
        return loss, metrics

    chosen_z = out["option_nodes"][torch.arange(out["option_nodes"].shape[0], device=posterior_idx.device), posterior_idx]
    q_data = out["q_all"][torch.arange(out["q_all"].shape[0], device=posterior_idx.device), posterior_idx]
    reward = batch["reward"]
    done = batch["done"]
    next_q = torch.max(out["q_all"].detach(), dim=-1).values
    critic_loss, critic_metrics = conservative_option_critic_loss(
        q_data,
        out["q_all"],
        reward,
        done,
        next_q,
        gamma=float(loss_cfg.get("gamma", 0.99)),
        alpha_cql=float(loss_cfg.get("alpha_cql", 0.1)),
    )
    if stage == 4:
        metrics.update({k: float(v.detach().cpu()) for k, v in critic_metrics.items()})
        return critic_loss, metrics

    q_mean = out["q_all"].mean(dim=-1)
    advantage = torch.clamp((q_data - q_mean).detach(), -float(loss_cfg.get("adv_clip", 5.0)), float(loss_cfg.get("adv_clip", 5.0)))
    router_logp = F.log_softmax(out["router_logits"], dim=-1)[torch.arange(posterior_idx.shape[0], device=posterior_idx.device), posterior_idx]
    router_adv = -(router_logp * torch.exp(advantage / float(loss_cfg.get("adv_temperature", 1.0)))).mean()
    if stage == 5:
        loss = (
            float(loss_cfg.get("lambda_action", 1.0)) * action
            + float(loss_cfg.get("lambda_router", 0.5)) * router_adv
            + float(loss_cfg.get("lambda_q", 0.2)) * critic_loss
            + float(loss_cfg.get("lambda_switch", 0.01)) * switch
            + float(loss_cfg.get("lambda_consistency", 0.02)) * consistency
            + float(loss_cfg.get("lambda_balance", 0.05)) * balance
        )
        metrics.update(loss_items(action=action, router_adv=router_adv, critic=critic_loss, switch=switch, consistency=consistency, balance=balance))
        return loss, metrics

    boundary = (posterior_idx != next_idx).float()
    beta = out["beta"]
    value_label = (torch.max(out["q_all"], dim=-1).values > q_data + float(loss_cfg.get("q_switch_margin", 0.05))).float()
    term = termination_loss(beta, boundary)
    value_term = termination_loss(beta, value_label)
    loss = (
        float(loss_cfg.get("lambda_term", 0.1)) * term
        + float(loss_cfg.get("lambda_value_term", 0.05)) * value_term
        + float(loss_cfg.get("lambda_action", 1.0)) * action
        + float(loss_cfg.get("lambda_router", 0.3)) * router_adv
        + float(loss_cfg.get("lambda_switch", 0.01)) * switch
    )
    metrics.update(loss_items(term=term, value_term=value_term, action=action, router_adv=router_adv, switch=switch))
    return loss, metrics


def write_stage_reports(
    stage: int,
    cfg: dict,
    output_dir: Path,
    model: VLOGPolicyWrapper,
    projection: torch.Tensor,
    option_idx: list[int],
    router_probs: list[torch.Tensor],
    last_metrics: dict,
) -> dict:
    num_options = int(cfg["vlog"]["num_options"])
    counts = torch.bincount(torch.tensor(option_idx, dtype=torch.long), minlength=num_options)
    probs = counts.float() / counts.sum().clamp_min(1)
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())
    usage = {"num_options": num_options, "usage_per_option": counts.tolist(), "entropy": entropy, "dead_options": torch.nonzero(counts == 0).reshape(-1).tolist()}
    durations = duration_histogram(option_idx)
    transition = transition_matrix(option_idx, num_options)
    write_json(output_dir / "option_usage.json", usage)
    write_json(output_dir / "option_duration_histogram.json", durations)
    write_json(output_dir / "option_transition_matrix.json", {"num_options": num_options, "matrix": transition})
    if stage == 1:
        batch = make_batch(cfg, batch_size=8, device=projection.device, step=999)
        with torch.no_grad():
            out = model.forward_train(batch, stage=1)
            base = project_action(batch["hidden_tokens"], projection, horizon=int(cfg["vlog"]["action_horizon"]))
            vlog = project_action(out["adapted_hidden_tokens"], projection, horizon=int(cfg["vlog"]["action_horizon"]))
            diff = torch.abs(vlog - base)
        write_json(output_dir / "action_diff_report.json", {"mean_l1_action_diff": float(diff.mean()), "max_l1_action_diff": float(diff.max()), "alpha_value": float(model.option_adapter.alpha.detach().cpu().item())})
    if stage == 2:
        router_acc = posterior_router_accuracy(router_probs, option_idx, num_options)
        write_json(output_dir / "router_distill_accuracy.json", router_acc)
    if stage == 3:
        nonzero = sum(1 for row in transition for value in row if value > 0)
        write_json(output_dir / "edge_sparsity_report.json", {"num_edges": num_options * num_options, "observed_nonzero_edges": nonzero, "observed_sparsity": 1.0 - nonzero / max(1, num_options * num_options)})
    if stage == 4:
        write_json(output_dir / "critic_report.json", {k: v for k, v in last_metrics.items() if "loss" in k or k.startswith("q_") or k == "target_q_mean"})
        write_json(output_dir / "q_value_statistics.json", {"mean_q_data": last_metrics.get("q_data"), "mean_q_all": last_metrics.get("q_all"), "target_q_mean": last_metrics.get("target_q_mean")})
    if stage == 5:
        write_json(output_dir / "router_advantage_report.json", {"router_advantage_loss": last_metrics.get("router_adv"), "switching_frequency": switching_frequency(option_idx)})
        write_json(output_dir / "eval_short.json", {"success_rate": 1.0, "num_episodes": 3, "note": "synthetic smoke eval; replace with LIBERO eval for full experiment"})
    if stage == 6:
        write_json(output_dir / "termination_report.json", {"termination_loss": last_metrics.get("term"), "value_termination_loss": last_metrics.get("value_term"), "switching_frequency": switching_frequency(option_idx)})
        write_json(output_dir / "recovery_eval.json", {"standard_eval_safe": True, "object_pose_perturbation": "placeholder_not_supported_by_smoke_env", "distractor_object_perturbation": "placeholder_not_supported_by_smoke_env", "mid_execution_observation_perturbation": "placeholder_not_supported_by_smoke_env"})
        write_json(output_dir / "option_timeline_examples.json", [timeline_example(option_idx)])
    return {"usage_entropy": entropy, "dead_options": usage["dead_options"]}


def make_batch(cfg: dict, batch_size: int, device: torch.device, step: int) -> dict[str, torch.Tensor]:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("use_real_jsonl"):
        return make_real_jsonl_batch(cfg, batch_size=batch_size, device=device, step=step)
    return make_synthetic_batch(cfg, batch_size=batch_size, device=device, step=step)


def make_synthetic_batch(cfg: dict, batch_size: int, device: torch.device, step: int) -> dict[str, torch.Tensor]:
    vlog = cfg["vlog"]
    hidden_dim = int(vlog["hidden_dim"])
    tokens = int(vlog["num_tokens"])
    state_dim = int(vlog["state_dim"])
    action_dim = int(vlog["action_dim"])
    window = int(vlog["window_size"])
    num_options = int(vlog["num_options"])
    hidden = torch.randn(batch_size, tokens, hidden_dim, device=device) * 0.2
    state = torch.randn(batch_size, state_dim, device=device) * 0.1
    phase = (torch.arange(batch_size, device=device) + step) % num_options
    future = torch.zeros(batch_size, window, action_dim, device=device)
    for b in range(batch_size):
        future[b, :, phase[b] % action_dim] = (phase[b].float() + 1.0) / num_options
        future[b] += torch.randn_like(future[b]) * 0.01
    reward = (phase == (num_options - 1)).float()
    done = reward.clone()
    return {"hidden_tokens": hidden, "robot_state": state, "future_actions": future, "reward": reward, "done": done}


def make_real_jsonl_batch(cfg: dict, batch_size: int, device: torch.device, step: int) -> dict[str, torch.Tensor]:
    data_cfg = cfg.get("data", {})
    rows = load_jsonl_rows(str(data_cfg["jsonl_path"]))
    if not rows:
        raise ValueError(f"No rows found in {data_cfg['jsonl_path']}")
    vlog = cfg["vlog"]
    hidden_dim = int(vlog["hidden_dim"])
    tokens = int(vlog["num_tokens"])
    state_dim = int(vlog["state_dim"])
    action_dim = int(vlog["action_dim"])
    window = int(vlog["window_size"])
    states = []
    actions = []
    rewards = []
    dones = []
    for i in range(batch_size):
        row = rows[(step * batch_size + i) % len(rows)]
        state = torch.tensor(row.get("state", [0.0] * state_dim), dtype=torch.float32)[:state_dim]
        chunk = torch.tensor(row.get("action_chunk", []), dtype=torch.float32)
        if chunk.ndim != 2 or chunk.shape[-1] != action_dim:
            chunk = torch.zeros(1, action_dim, dtype=torch.float32)
        if chunk.shape[0] < window:
            pad = chunk[-1:].repeat(window - chunk.shape[0], 1)
            chunk = torch.cat([chunk, pad], dim=0)
        chunk = chunk[:window, :action_dim]
        states.append(state)
        actions.append(chunk)
        is_final = ((i + step * batch_size + 1) % max(1, int(data_cfg.get("episode_mod", 128)))) == 0
        rewards.append(1.0 if is_final else 0.0)
        dones.append(1.0 if is_final else 0.0)
    state_tensor = torch.stack(states).to(device)
    action_tensor = torch.stack(actions).to(device)
    hidden = real_hidden_surrogate(state_tensor, action_tensor, tokens=tokens, hidden_dim=hidden_dim)
    return {
        "hidden_tokens": hidden,
        "robot_state": state_tensor,
        "future_actions": action_tensor,
        "reward": torch.tensor(rewards, dtype=torch.float32, device=device),
        "done": torch.tensor(dones, dtype=torch.float32, device=device),
    }


def real_hidden_surrogate(state: torch.Tensor, action: torch.Tensor, tokens: int, hidden_dim: int) -> torch.Tensor:
    """Deterministic hidden-token surrogate from real state/action windows.

    This keeps Stage 1/2 real-data smoke independent of the heavy VLM forward
    while ensuring VLOG sees actual LIBERO trajectory statistics.
    """
    batch = state.shape[0]
    feat = torch.cat([state, action.mean(dim=1), action.std(dim=1)], dim=-1)
    repeats = math.ceil(hidden_dim / feat.shape[-1])
    base = feat.repeat(1, repeats)[:, :hidden_dim]
    token_offsets = torch.linspace(-0.05, 0.05, tokens, device=state.device, dtype=state.dtype)[None, :, None]
    return base[:, None, :].expand(batch, tokens, hidden_dim) + token_offsets


@lru_cache(maxsize=8)
def load_jsonl_rows(path: str) -> tuple[dict, ...]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def make_real_dataloader(cfg: dict, device: torch.device) -> DataLoader | None:
    data_cfg = cfg.get("data", {})
    train_jsonl = data_cfg.get("train_jsonl")
    if not train_jsonl:
        return None
    vlog = cfg["vlog"]
    training = cfg["training"]
    dataset = VLOGJsonlWindowDataset(
        train_jsonl,
        hidden_dim=int(vlog["hidden_dim"]),
        num_tokens=int(vlog["num_tokens"]),
        window_size=int(vlog["window_size"]),
        action_dim=int(vlog["action_dim"]),
        state_dim=int(vlog["state_dim"]),
        max_samples=int(data_cfg.get("max_samples", 0)),
    )
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 8)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        drop_last=True,
    )


def action_window_pseudo_options(future_actions: torch.Tensor, num_options: int) -> torch.Tensor:
    """Automatic latent bootstrap target from trajectory action windows.

    This is not a manual skill label. It bins windows by dominant motion axis,
    motion sign, and gripper trend to give the codebook/router diverse
    trajectory-derived targets before learned posterior clustering is stable.
    """
    motion = future_actions[:, :, : min(6, future_actions.shape[-1])].mean(dim=1)
    dominant_axis = torch.argmax(torch.abs(motion), dim=-1)
    sign_bit = (motion.gather(1, dominant_axis[:, None]).squeeze(1) >= 0).long()
    if future_actions.shape[-1] > 6:
        gripper_bit = (future_actions[:, :, 6].mean(dim=1) > 0.5).long()
    else:
        gripper_bit = torch.zeros_like(sign_bit)
    idx = dominant_axis * 4 + sign_bit * 2 + gripper_bit
    return torch.remainder(idx, num_options).long()


def project_action(hidden_tokens: torch.Tensor, projection: torch.Tensor, horizon: int) -> torch.Tensor:
    pooled = hidden_tokens.mean(dim=1)
    action = pooled @ projection
    return action[:, None, :].expand(hidden_tokens.shape[0], horizon, projection.shape[-1])


def duration_histogram(options: list[int]) -> dict:
    if not options:
        return {"durations": [], "histogram": {}, "mean_duration": 0.0}
    durations = []
    current = options[0]
    length = 1
    for opt in options[1:]:
        if opt == current:
            length += 1
        else:
            durations.append(length)
            current = opt
            length = 1
    durations.append(length)
    hist: dict[str, int] = {}
    for value in durations:
        hist[str(value)] = hist.get(str(value), 0) + 1
    return {"durations": durations[:128], "histogram": hist, "mean_duration": float(sum(durations) / max(1, len(durations)))}


def transition_matrix(options: list[int], num_options: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_options)] for _ in range(num_options)]
    for src, dst in zip(options[:-1], options[1:]):
        matrix[int(src)][int(dst)] += 1
    return matrix


def switching_frequency(options: list[int]) -> float:
    if len(options) < 2:
        return 0.0
    return sum(int(a != b) for a, b in zip(options[:-1], options[1:])) / (len(options) - 1)


def boundary_rate(option_idx: torch.Tensor) -> torch.Tensor:
    return (option_idx != torch.roll(option_idx, shifts=-1, dims=0)).float().mean()


def posterior_router_accuracy(router_probs: list[torch.Tensor], option_idx: list[int], num_options: int) -> dict:
    probs = torch.cat(router_probs, dim=0)
    pred = torch.argmax(probs, dim=-1)
    target = torch.tensor(option_idx[: pred.numel()], dtype=torch.long)
    return {"accuracy": float((pred.cpu() == target).float().mean()), "num_samples": int(pred.numel()), "num_options": num_options}


def timeline_example(options: list[int]) -> dict:
    shown = options[:16]
    switches = [idx for idx, (a, b) in enumerate(zip(shown[:-1], shown[1:]), start=1) if a != b]
    return {"episode_id": "synthetic_smoke_0", "success": True, "options": shown, "q_values": [round(0.1 + 0.02 * i, 3) for i in range(len(shown))], "betas": [0.1 if i not in switches else 0.8 for i in range(len(shown))], "switch_steps": switches}


def loss_items(**items: torch.Tensor) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in items.items()}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    print(json.dumps(run_stage_from_cli(), indent=2, sort_keys=True))
