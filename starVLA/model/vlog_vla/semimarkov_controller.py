"""Per-environment persistent option selection for UniversalVLOG inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import torch


@dataclass
class ControllerState:
    active_option: int | None = None
    option_age: int = 0
    episode_id: Hashable | None = None
    previous_action_chunk: torch.Tensor | None = None


class SemiMarkovController:
    """Keep independent option state for every rollout stream."""

    def __init__(
        self, d_min: int = 2, d_max: int = 8, hysteresis: float = 0.15
    ) -> None:
        if d_min < 0 or d_max < d_min:
            raise ValueError(f"Require 0 <= d_min <= d_max, got {d_min}, {d_max}")
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.hysteresis = float(hysteresis)
        self._states: dict[Hashable, ControllerState] = {}

    def reset(self, stream_id: Hashable | None = None) -> None:
        if stream_id is None:
            self._states.clear()
        else:
            self._states.pop(stream_id, None)

    def state_for(self, stream_id: Hashable) -> ControllerState:
        return self._states.setdefault(stream_id, ControllerState())

    def update(
        self,
        router_logits: torch.Tensor,
        stream_ids: Iterable[Hashable],
        episode_ids: Iterable[Hashable | None],
    ) -> tuple[torch.Tensor, list[dict]]:
        if router_logits.ndim != 2:
            raise ValueError(
                f"router_logits must be [B,K], got {tuple(router_logits.shape)}"
            )
        stream_ids = list(stream_ids)
        episode_ids = list(episode_ids)
        if (
            len(stream_ids) != router_logits.shape[0]
            or len(episode_ids) != router_logits.shape[0]
        ):
            raise ValueError("stream_ids/episode_ids must match router batch size")

        selected: list[int] = []
        metadata: list[dict] = []
        detached = router_logits.detach().float().cpu()
        for row, stream_id, episode_id in zip(detached, stream_ids, episode_ids):
            state = self.state_for(stream_id)
            if state.episode_id != episode_id:
                state.active_option = None
                state.option_age = 0
                state.previous_action_chunk = None
                state.episode_id = episode_id

            candidate = int(torch.argmax(row).item())
            previous = state.active_option
            reason = "keep"
            if previous is None:
                state.active_option = candidate
                state.option_age = 0
                reason = "episode_init"
            elif state.option_age < self.d_min:
                reason = "d_min"
            elif state.option_age >= self.d_max:
                # Reselect at d_max; the best option may still be the current
                # one, so this is not falsely reported as a forced switch.
                state.active_option = candidate
                reason = "d_max_reselect"
            elif (
                candidate != previous
                and float(row[candidate] - row[previous]) > self.hysteresis
            ):
                state.active_option = candidate
                reason = "hysteresis"

            switched = previous is not None and state.active_option != previous
            state.option_age = 0 if switched else state.option_age + 1
            selected.append(int(state.active_option))
            metadata.append(
                {
                    "active_option": int(state.active_option),
                    "option_age_decisions": int(state.option_age),
                    "switched": bool(switched),
                    "switch_reason": reason,
                }
            )
        return torch.tensor(
            selected, device=router_logits.device, dtype=torch.long
        ), metadata

    def record_action_chunks(
        self, stream_ids: Iterable[Hashable], chunks: Iterable[torch.Tensor]
    ) -> None:
        for stream_id, chunk in zip(stream_ids, chunks):
            self.state_for(stream_id).previous_action_chunk = chunk.detach().cpu()
