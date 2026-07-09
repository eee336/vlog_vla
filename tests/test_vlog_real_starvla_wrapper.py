from types import SimpleNamespace

import torch
import torch.nn as nn

from starVLA.model.vlog_vla.real_starvla_vlog_wrapper import RealStarVLAVLOGWrapper


class MockActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 7)

    def predict_action(self, hidden):
        return self.proj(hidden)


class MockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = MockActionModel()
        self.config = SimpleNamespace(framework=SimpleNamespace(action_model=SimpleNamespace(action_hidden_dim=16, action_dim=7)))


def test_real_starvla_wrapper_constructs_and_stage1_with_mock_adapter(monkeypatch):
    base = MockBase()
    cfg = {"vlog": {"allow_surrogate_hidden": False, "option_dim": 16, "num_options": 4}, "dataset": {"state_dim": 8, "action_dim": 7, "window_size": 4}}
    wrapper = RealStarVLAVLOGWrapper(base, cfg)

    def fake_encode(batch):
        return torch.randn(1, 4, 16), {"uses_real_starvla_hidden": True, "uses_surrogate_hidden": False}

    monkeypatch.setattr(wrapper.hidden_adapter, "encode_hidden", fake_encode)
    monkeypatch.setattr(wrapper.hidden_adapter, "decode_action_from_hidden", lambda hidden, batch, aux=None: base.action_model.predict_action(hidden))
    out = wrapper.forward_train({"robot_state": [[0.0] * 8], "future_actions": torch.zeros(1, 4, 7)}, stage="stage1_preserve")
    assert out["uses_real_starvla_hidden"] is True
    assert out["uses_surrogate_hidden"] is False
    assert out["base_action"].shape == out["vlog_action"].shape
    assert "router_idx" in out
