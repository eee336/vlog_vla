"""VLOG-VLA modules.

VLOG-VLA inserts persistent value-guided latent options between a pretrained
VLA representation and its action head. Options are latent codebook entries,
not manually named skills or text labels.
"""

from .latent_option_codebook import LatentOptionCodebook
from .hidden_hook_utils import HiddenTokenHook
from .option_adapter import OptionAdapter
from .option_conditioner import OptionConditioner, OptionConditioningOutput
from .option_critic import OptionCritic
from .option_graph_layer import OptionGraphLayer
from .persistent_option_router import PersistentOptionRouter
from .posterior_option_encoder import PosteriorOptionEncoder
from .real_starvla_vlog_wrapper import RealStarVLAVLOGWrapper
from .starvla_hidden_adapter import StarVLAHiddenAdapter
from .state_aggregator import StateAggregator
from .semimarkov_controller import ControllerState, SemiMarkovController
from .universal_adapters import (
    EmbodimentAdapterBank,
    EmbodimentSpec,
    NativeActionDecoder,
    NativeActionEncoder,
    NativeStateAdapter,
)
from .termination_head import TerminationHead
from .vlog_policy_wrapper import PersistenceConfig, VLOGPolicyWrapper

__all__ = [
    "LatentOptionCodebook",
    "HiddenTokenHook",
    "OptionAdapter",
    "OptionConditioner",
    "OptionConditioningOutput",
    "OptionCritic",
    "OptionGraphLayer",
    "PersistentOptionRouter",
    "PosteriorOptionEncoder",
    "RealStarVLAVLOGWrapper",
    "StarVLAHiddenAdapter",
    "StateAggregator",
    "ControllerState",
    "SemiMarkovController",
    "EmbodimentAdapterBank",
    "EmbodimentSpec",
    "NativeActionDecoder",
    "NativeActionEncoder",
    "NativeStateAdapter",
    "TerminationHead",
    "PersistenceConfig",
    "VLOGPolicyWrapper",
]
