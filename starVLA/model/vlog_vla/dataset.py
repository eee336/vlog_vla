from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class VLOGJsonlWindowDataset(Dataset):
    """Trajectory-window dataset from StarVLA/LIBERO JSONL rows.

    The dataset returns real robot state and future action windows. Hidden
    tokens are deterministic feature tokens derived from state/action context;
    full Qwen hidden-token extraction is intentionally left to the official
    VLA integration path.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        hidden_dim: int,
        num_tokens: int,
        window_size: int,
        action_dim: int,
        state_dim: int,
        max_samples: int = 0,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.window_size = int(window_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        rows = _read_jsonl(self.jsonl_path)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[int(row.get("episode_id", 0))].append(row)
        for episode_rows in grouped.values():
            episode_rows.sort(key=lambda item: int(item.get("step_id", 0)))
        samples = []
        for episode_id, episode_rows in sorted(grouped.items()):
            for local_idx, row in enumerate(episode_rows):
                samples.append((episode_id, local_idx, row, episode_rows))
        self.samples = samples[:max_samples] if max_samples > 0 else samples
        self._token_positions = torch.linspace(-1.0, 1.0, self.num_tokens).view(self.num_tokens, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _, local_idx, row, episode_rows = self.samples[idx]
        state = torch.tensor(row["state"][: self.state_dim], dtype=torch.float32)
        future_actions = self._future_actions(row, episode_rows, local_idx)
        hidden = self._hidden_tokens(state, future_actions)
        reward = torch.tensor(float(row.get("success") or 0.0), dtype=torch.float32)
        if local_idx >= len(episode_rows) - 1:
            reward = torch.tensor(1.0, dtype=torch.float32)
        done = torch.tensor(float(local_idx >= len(episode_rows) - 1), dtype=torch.float32)
        return {
            "hidden_tokens": hidden,
            "robot_state": state,
            "future_actions": future_actions,
            "reward": reward,
            "done": done,
        }

    def _future_actions(self, row: dict, episode_rows: list[dict], local_idx: int) -> torch.Tensor:
        if "action_chunk" in row and row["action_chunk"]:
            actions = row["action_chunk"][: self.window_size]
        else:
            actions = []
        while len(actions) < self.window_size:
            src = episode_rows[min(local_idx + len(actions), len(episode_rows) - 1)]
            action = src.get("action_chunk", [[0.0] * self.action_dim])[0]
            actions.append(action)
        tensor = torch.tensor(actions, dtype=torch.float32)
        return tensor[:, : self.action_dim]

    def _hidden_tokens(self, state: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        context = torch.cat([state, future_actions.mean(dim=0)], dim=0)
        repeats = (self.hidden_dim + context.numel() - 1) // context.numel()
        base = context.repeat(repeats)[: self.hidden_dim]
        token_scale = 1.0 + 0.05 * self._token_positions
        token_bias = torch.sin(torch.arange(self.hidden_dim, dtype=torch.float32) / 17.0) * 0.01
        return token_scale * base.view(1, -1) + token_bias.view(1, -1)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
