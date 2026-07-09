import pytest
import torch.nn as nn

from starVLA.model.vlog_vla.real_starvla_vlog_wrapper import RealStarVLAVLOGWrapper
from starVLA.model.vlog_vla.starvla_hidden_adapter import StarVLAHiddenAdapter


def test_stage7_rejects_surrogate_hidden_config():
    with pytest.raises(ValueError):
        RealStarVLAVLOGWrapper(nn.Module(), {"vlog": {"allow_surrogate_hidden": True}, "dataset": {}})


def test_hidden_adapter_raises_when_real_path_missing_and_no_surrogate():
    adapter = StarVLAHiddenAdapter(nn.Module(), {"allow_surrogate_hidden": False})
    with pytest.raises(RuntimeError, match="Real StarVLA hidden extraction requires"):
        adapter.encode_hidden({"image": None, "lang": "", "state": []})
