import inspect

import torch

from starVLA.model.vlog_vla import (
    LatentOptionCodebook,
    OptionAdapter,
    OptionCritic,
    OptionGraphLayer,
    PersistenceConfig,
    PersistentOptionRouter,
    PosteriorOptionEncoder,
    StateAggregator,
    TerminationHead,
    VLOGPolicyWrapper,
)


def test_vlog_module_shape_smoke():
    batch, tokens, hidden_dim = 2, 128, 256
    num_options, option_dim, action_dim, window = 16, 256, 7, 16
    hidden = torch.randn(batch, tokens, hidden_dim)
    robot_state = torch.randn(batch, 8)
    future_actions = torch.randn(batch, window, action_dim)

    aggregator = StateAggregator(hidden_dim, state_dim=8, out_dim=option_dim)
    state = aggregator(hidden, robot_state)
    assert state.shape == (batch, option_dim)

    posterior = PosteriorOptionEncoder(option_dim, action_dim, option_dim, window)
    e_post = posterior(state, future_actions)
    assert e_post.shape == (batch, option_dim)

    codebook = LatentOptionCodebook(num_options, option_dim)
    z_q, idx, vq_loss, commit_loss = codebook(e_post)
    assert z_q.shape == (batch, option_dim)
    assert idx.shape == (batch,)
    assert vq_loss.ndim == 0
    assert commit_loss.ndim == 0

    graph = OptionGraphLayer(option_dim, option_dim, num_options)
    nodes, edge_logits = graph(state, codebook.codebook.weight)
    assert nodes.shape == (batch, num_options, option_dim)
    assert edge_logits.shape == (batch, num_options, num_options)

    router = PersistentOptionRouter(option_dim, option_dim, num_options)
    z_soft, route_idx, probs, logits = router(state, nodes)
    assert z_soft.shape == (batch, option_dim)
    assert route_idx.shape == (batch,)
    assert probs.shape == (batch, num_options)
    assert logits.shape == (batch, num_options)

    critic = OptionCritic(option_dim, option_dim)
    assert critic.q_all(state, nodes).shape == (batch, num_options)

    termination = TerminationHead(option_dim, option_dim)
    assert termination(state, z_soft).shape == (batch,)

    adapter = OptionAdapter(hidden_dim, option_dim)
    assert adapter(hidden, z_soft).shape == hidden.shape


def test_zero_init_option_adapter_preserves_hidden_tokens():
    hidden = torch.randn(2, 8, 64)
    option = torch.randn(2, 32)
    adapter = OptionAdapter(hidden_dim=64, option_dim=32, alpha_init=0.0)
    adapted = adapter(hidden, option)
    assert torch.max(torch.abs(adapted - hidden)).item() < 1e-7
    assert adapter.alpha.item() == 0.0


def test_critic_is_option_level_not_action_level():
    critic = OptionCritic(state_dim=32, option_dim=16)
    sig = inspect.signature(critic.forward)
    assert list(sig.parameters) == ["state_feature", "option_embedding"]
    state = torch.randn(4, 32)
    options = torch.randn(4, 8, 16)
    q = critic.q_all(state, options)
    assert q.shape == (4, 8)


def test_persistence_logic_respects_min_and_max_duration():
    wrapper = VLOGPolicyWrapper(hidden_dim=32, state_dim=8, action_dim=7, option_dim=32, num_options=4, window_size=4, persistence=PersistenceConfig(d_min=2, d_max=8))
    wrapper.current_option_idx = torch.tensor([3])
    wrapper.option_age = 1
    assert wrapper._should_switch(beta=1.0, q_current=0.0, q_best=10.0) is False
    wrapper.option_age = 9
    assert wrapper._should_switch(beta=0.0, q_current=0.0, q_best=0.0) is True
