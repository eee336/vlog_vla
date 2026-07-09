from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RolloutBuffer:
    """Small typed container for VLOG option-level transitions."""

    transitions: list[dict] = field(default_factory=list)

    def add(self, **transition) -> None:
        self.transitions.append(dict(transition))

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)
