import copy

import pytest

torch = pytest.importorskip("torch")

from starVLA.model.vlog_vla.option_conditioner import OptionConditioner
from starVLA.model.vlog_vla.semimarkov_controller import SemiMarkovController
from starVLA.model.vlog_vla.universal_adapters import (
    EmbodimentAdapterBank,
    EmbodimentSpec,
)


def test_native_adapters_use_shared_dit_boundary_not_gr1_coordinates():
    specs = {
        "robocasa_gr1": EmbodimentSpec(state_dim=58, action_dim=29),
        "libero": EmbodimentSpec(state_dim=7, action_dim=7),
        "real_franka": EmbodimentSpec(state_dim=9, action_dim=7, state_tokens=2),
    }
    bank = EmbodimentAdapterBank(
        specs,
        base_embodiment="robocasa_gr1",
        dit_dim=32,
        dit_output_dim=32,
        hidden_dim=16,
    )
    t = torch.zeros(2, dtype=torch.long)
    libero_action = torch.randn(2, 4, 7)
    encoded = bank.action_encoders["libero"](libero_action, t)
    decoded = bank.action_decoders["libero"](encoded)
    assert encoded.shape == (2, 4, 32)
    assert decoded.shape == (2, 4, 7)
    assert bank.action_encoders["libero"].layer1.in_features == 7
    assert bank.action_decoders["libero"].layer2.out_features == 7
    assert "robocasa_gr1" not in bank.action_encoders

    real_state = torch.randn(2, 1, 9)
    assert bank.state_adapters["real_franka"](real_state).shape == (2, 2, 32)


def test_option_conditioner_zero_init_preserves_action_tokens():
    conditioner = OptionConditioner(8, 16, 16, 16, rho=0.05)
    action = torch.randn(3, 5, 16)
    output = conditioner(
        action,
        torch.randn(3, 8),
        torch.randn(3, 16),
        torch.randn(3, 16),
    )
    torch.testing.assert_close(output.action_tokens, action, atol=0, rtol=0)
    torch.testing.assert_close(
        output.option_token, torch.zeros_like(output.option_token), atol=0, rtol=0
    )


def test_option_conditioner_has_option_sensitive_causal_path_after_update():
    conditioner = OptionConditioner(8, 16, 16, 16, rho=0.05)
    with torch.no_grad():
        conditioner.film.weight.normal_(std=0.1)
        conditioner.to_option_token.weight.normal_(std=0.1)
    action = torch.randn(2, 5, 16)
    state = torch.randn(2, 16)
    embodiment = torch.randn(2, 16)
    option_a = torch.zeros(2, 8)
    option_b = torch.ones(2, 8)
    out_a = conditioner(action, option_a, state, embodiment)
    out_b = conditioner(action, option_b, state, embodiment)
    assert not torch.allclose(out_a.action_tokens, out_b.action_tokens)
    assert not torch.allclose(out_a.option_token, out_b.option_token)


def test_semimarkov_controller_is_independent_and_resets_on_episode_change():
    controller = SemiMarkovController(d_min=2, d_max=4, hysteresis=0.2)
    logits = torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    selected, _ = controller.update(logits, ["env0", "env1"], ["ep0", "ep0"])
    assert selected.tolist() == [1, 0]

    # d_min holds env0's option even though option 2 becomes best. env1 has an
    # independent state and is reset by its new episode id.
    selected, metadata = controller.update(
        torch.tensor([[0.0, 0.0, 5.0], [0.0, 4.0, 0.0]]),
        ["env0", "env1"],
        ["ep0", "ep1"],
    )
    assert selected.tolist() == [1, 1]
    assert metadata[0]["switch_reason"] == "d_min"
    assert metadata[1]["switch_reason"] == "episode_init"
    assert controller.state_for("env0").episode_id == "ep0"
    assert controller.state_for("env1").episode_id == "ep1"


def _small_action_config():
    omegaconf = pytest.importorskip("omegaconf")
    return omegaconf.OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 16,
                    "add_pos_embed": True,
                    "max_seq_len": 32,
                    "action_dim": 3,
                    "state_dim": 4,
                    "action_horizon": 3,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 2,
                    "num_target_vision_tokens": 2,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 5,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": False,
                        "norm_type": "ada_norm",
                        "num_layers": 1,
                        "output_dim": 768,
                        "positional_embeddings": None,
                    },
                }
            }
        }
    )


def _small_framework_config(stage="u1_oracle"):
    cfg = _small_action_config()
    cfg.framework.name = "QwenUniversalVLOG"
    cfg.framework.base_embodiment = "robocasa_gr1"
    cfg.framework.embodiments = {
        "robocasa_gr1": {"state_dim": 4, "action_dim": 3, "state_tokens": 1},
        "libero": {"state_dim": 2, "action_dim": 2, "state_tokens": 1},
    }
    cfg.framework.vlog = {
        "enabled": True,
        "train_stage": stage,
        "num_options": 4,
        "option_dim": 6,
        "fusion_enabled": True,
        "fusion_residual_scale": 0.05,
        "d_min": 2,
        "d_max": 4,
        "router_hysteresis": 0.2,
        "counterfactual_every": 4,
        "counterfactual_margin": 0.05,
        "improve_margin": 0.0,
        "commitment_cost": 0.25,
    }
    return cfg


def test_fusion_off_base_head_has_exact_original_velocity_and_sequence():
    pytest.importorskip("diffusers")
    from starVLA.model.modules.action_model.GR00T_ActionHeader import (
        FlowmatchingActionHead,
    )
    from starVLA.model.vlog_vla.qwen_universal_vlog import (
        UniversalFlowmatchingActionHead,
    )

    torch.manual_seed(7)
    config = _small_action_config()
    reference = FlowmatchingActionHead(copy.deepcopy(config)).eval()
    universal = UniversalFlowmatchingActionHead(
        copy.deepcopy(config),
        specs={
            "robocasa_gr1": EmbodimentSpec(4, 3),
            "libero": EmbodimentSpec(2, 2),
        },
        base_embodiment="robocasa_gr1",
        option_dim=6,
        fusion_residual_scale=0.05,
    ).eval()
    universal.load_state_dict(reference.state_dict(), strict=False)

    vl = torch.randn(2, 4, 5)
    actions = torch.randn(2, 3, 3)
    state = torch.randn(2, 1, 4)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.2, 0.7])
    t_discrete = (time * 1000).long()
    noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * actions

    action_tokens = reference.action_encoder(noisy, t_discrete)
    positions = reference.position_embedding(torch.arange(3))[None]
    sequence = torch.cat(
        [
            reference.state_encoder(state),
            reference.future_tokens.weight[None].expand(2, -1, -1),
            action_tokens + positions,
        ],
        dim=1,
    )
    reference_output = reference.model(
        hidden_states=sequence,
        encoder_hidden_states=vl,
        timestep=t_discrete,
    )
    reference_velocity = reference.action_decoder(reference_output)[:, -3:]
    actual = universal.flow_forward(
        vl,
        actions,
        state,
        "robocasa_gr1",
        t=time,
        noise=noise,
        fusion_enabled=False,
    )
    torch.testing.assert_close(
        actual["pred_velocity"], reference_velocity, atol=0, rtol=0
    )
    assert int(actual["dit_sequence_length"].item()) == 1 + 2 + 3


def test_variable_horizon_mask_excludes_padding_from_fm_loss():
    pytest.importorskip("diffusers")
    from starVLA.model.vlog_vla.qwen_universal_vlog import (
        UniversalFlowmatchingActionHead,
    )

    config = _small_action_config()
    head = UniversalFlowmatchingActionHead(
        config,
        specs={"robocasa_gr1": EmbodimentSpec(4, 3)},
        base_embodiment="robocasa_gr1",
        option_dim=6,
        fusion_residual_scale=0.05,
    ).eval()
    vl = torch.randn(1, 4, 5)
    actions = torch.randn(1, 3, 3)
    state = torch.randn(1, 1, 4)
    noise = torch.randn_like(actions)
    mask = torch.tensor([[True, True, False]])
    output = head.flow_forward(
        vl,
        actions,
        state,
        "robocasa_gr1",
        action_mask=mask,
        t=torch.tensor([0.4]),
        noise=noise,
    )
    expected = (
        (output["pred_velocity"] - output["target_velocity"])
        .square()
        .mean(-1)[:, :2]
        .mean()
    )
    torch.testing.assert_close(output["loss"], expected)


def test_usage_balance_is_differentiable_not_a_hard_histogram_only():
    pytest.importorskip("diffusers")
    from starVLA.model.vlog_vla.qwen_universal_vlog import UniversalVLOGCore

    core = UniversalVLOGCore(
        vl_dim=12,
        dit_dim=8,
        option_dim=6,
        num_options=4,
        commitment_cost=0.25,
    )
    state = torch.randn(3, 6)
    future = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)
    output = core.discover(state, future, mask)
    assert output["usage_loss"].requires_grad
    output["usage_loss"].backward()
    assert core.codebook.codebook.weight.grad is not None
    assert torch.isfinite(core.codebook.codebook.weight.grad).all()
    assert core.codebook.codebook.weight.grad.norm().item() > 0


def test_stage_freezing_keeps_u2_strictly_router_only(monkeypatch):
    pytest.importorskip("diffusers")
    from types import SimpleNamespace

    import starVLA.model.vlog_vla.qwen_universal_vlog as universal_module

    class FakeVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.ones(1))
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=5))

    monkeypatch.setattr(universal_module, "get_vlm_model", lambda config: FakeVLM())
    model = universal_module.QwenUniversalVLOG(_small_framework_config())

    u1_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any(name.startswith("vlog_core.posterior.") for name in u1_trainable)
    assert any(
        name.startswith("action_model.option_conditioner.") for name in u1_trainable
    )
    assert not any(name.startswith("vlog_core.router.") for name in u1_trainable)

    model.configure_train_stage("u2_router")
    u2_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert u2_trainable
    assert all(name.startswith("vlog_core.router.") for name in u2_trainable)

    model.configure_train_stage("u0_libero")
    u0l_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert u0l_trainable
    assert all(
        name.startswith("action_model.adapter_bank.")
        or name.startswith("action_model.embodiment_embedding.")
        for name in u0l_trainable
    )


def test_base_checkpoint_loader_rejects_missing_core_key(monkeypatch):
    pytest.importorskip("diffusers")
    from types import SimpleNamespace

    import starVLA.model.vlog_vla.qwen_universal_vlog as universal_module

    class FakeVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.ones(1))
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=5))

    monkeypatch.setattr(universal_module, "get_vlm_model", lambda config: FakeVLM())
    model = universal_module.QwenUniversalVLOG(_small_framework_config())
    current = model.state_dict()
    base_checkpoint = {key: current[key].clone() for key in model._base_core_keys()}
    incompatible = model.load_state_dict(base_checkpoint, strict=False)
    assert incompatible.missing_keys
    assert all(
        key.startswith(model._UNIVERSAL_MARKERS) for key in incompatible.missing_keys
    )

    broken = dict(base_checkpoint)
    broken.pop("qwen_vl_interface.dummy")
    with pytest.raises(RuntimeError, match="Refusing lossy GR00T warm-start"):
        model.load_state_dict(broken, strict=False)
