"""Native embodiment boundaries for the shared UniversalVLOG action expert.

The adapters in this module map *directly* between a robot's normalized native
coordinates and the shared DiT token space.  In particular, a 7-D LIBERO
action is never projected into the 29-D GR1 action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import ActionEncoder, MLP


@dataclass(frozen=True)
class EmbodimentSpec:
    """Tensor contract for one robot embodiment."""

    state_dim: int
    action_dim: int
    state_tokens: int = 1

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0 or self.state_tokens <= 0:
            raise ValueError(f"Invalid embodiment dimensions: {self}")


def normalize_embodiment_specs(
    raw_specs: Mapping, base_embodiment: str
) -> dict[str, EmbodimentSpec]:
    """Convert OmegaConf/dict-like specifications to validated dataclasses."""

    specs: dict[str, EmbodimentSpec] = {}
    for name, raw in raw_specs.items():
        getter = (
            raw.get
            if hasattr(raw, "get")
            else lambda key, default=None: getattr(raw, key, default)
        )
        specs[str(name)] = EmbodimentSpec(
            state_dim=int(getter("state_dim")),
            action_dim=int(getter("action_dim")),
            state_tokens=int(getter("state_tokens", 1)),
        )
    if base_embodiment not in specs:
        raise ValueError(
            f"base_embodiment={base_embodiment!r} is not registered in {sorted(specs)}"
        )
    return specs


class NativeStateAdapter(nn.Module):
    """Map native proprioception to one or more shared DiT tokens."""

    def __init__(
        self, native_dim: int, dit_dim: int, hidden_dim: int, num_tokens: int = 1
    ) -> None:
        super().__init__()
        self.native_dim = int(native_dim)
        self.dit_dim = int(dit_dim)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.Linear(self.native_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_tokens * self.dit_dim),
        )
        self.norm = nn.LayerNorm(self.dit_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 2:
            state = state[:, None, :]
        if state.ndim != 3 or state.shape[-1] != self.native_dim:
            raise ValueError(
                f"Expected state [B,N,{self.native_dim}], got {tuple(state.shape)}"
            )
        # Native observations normally contain one current-state vector.  If a
        # caller provides history, pool it explicitly rather than treating time
        # steps as independent DiT state tokens.
        pooled = state.mean(dim=1)
        tokens = self.net(pooled).reshape(state.shape[0], self.num_tokens, self.dit_dim)
        return self.norm(tokens)


class NativeActionEncoder(ActionEncoder):
    """Native normalized action -> shared DiT action tokens."""

    def __init__(self, native_dim: int, dit_dim: int) -> None:
        super().__init__(action_dim=int(native_dim), hidden_size=int(dit_dim))
        self.native_dim = int(native_dim)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.native_dim:
            raise ValueError(
                f"Expected action [B,T,{self.native_dim}], got {tuple(actions.shape)}"
            )
        return super().forward(actions, timesteps)


class NativeActionDecoder(MLP):
    """Shared DiT output tokens -> native normalized velocity."""

    def __init__(self, dit_output_dim: int, hidden_dim: int, native_dim: int) -> None:
        super().__init__(
            input_dim=int(dit_output_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(native_dim),
        )
        self.native_dim = int(native_dim)


class EmbodimentAdapterBank(nn.Module):
    """Small per-embodiment modules surrounding one shared DiT core.

    The base embodiment is intentionally omitted.  Its modules remain at the
    historical GR00T state-dict paths (``state_encoder``, ``action_encoder`` and
    ``action_decoder``), which permits an official checkpoint to warm-start the
    framework without key remapping or lossy copying.
    """

    def __init__(
        self,
        specs: Mapping[str, EmbodimentSpec],
        base_embodiment: str,
        dit_dim: int,
        dit_output_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.specs = dict(specs)
        self.base_embodiment = str(base_embodiment)
        nonbase = {
            name: spec for name, spec in specs.items() if name != self.base_embodiment
        }
        self.state_adapters = nn.ModuleDict(
            {
                name: NativeStateAdapter(
                    spec.state_dim, dit_dim, hidden_dim, spec.state_tokens
                )
                for name, spec in nonbase.items()
            }
        )
        self.action_encoders = nn.ModuleDict(
            {
                name: NativeActionEncoder(spec.action_dim, dit_dim)
                for name, spec in nonbase.items()
            }
        )
        self.action_decoders = nn.ModuleDict(
            {
                name: NativeActionDecoder(dit_output_dim, hidden_dim, spec.action_dim)
                for name, spec in nonbase.items()
            }
        )

    def spec(self, embodiment_id: str) -> EmbodimentSpec:
        try:
            return self.specs[str(embodiment_id)]
        except KeyError as exc:
            raise KeyError(
                f"Unknown embodiment_id={embodiment_id!r}; registered={sorted(self.specs)}"
            ) from exc

    def is_base(self, embodiment_id: str) -> bool:
        self.spec(embodiment_id)
        return str(embodiment_id) == self.base_embodiment
