from starVLA.model.vlog_vla.dataset import VLOGJsonlWindowDataset
from scripts.vlog_vla.train_common import action_window_pseudo_options


def test_vlog_jsonl_window_dataset_loads_real_libero_rows():
    dataset = VLOGJsonlWindowDataset(
        "data/starvla_lerobot_standard_libero_spatial/train.jsonl",
        hidden_dim=128,
        num_tokens=64,
        window_size=10,
        action_dim=7,
        state_dim=8,
        max_samples=8,
    )
    sample = dataset[0]
    assert len(dataset) == 8
    assert sample["hidden_tokens"].shape == (64, 128)
    assert sample["robot_state"].shape == (8,)
    assert sample["future_actions"].shape == (10, 7)
    assert sample["reward"].ndim == 0
    assert sample["done"].ndim == 0


def test_action_window_pseudo_options_are_trajectory_derived():
    dataset = VLOGJsonlWindowDataset(
        "data/starvla_lerobot_standard_libero_spatial/train.jsonl",
        hidden_dim=128,
        num_tokens=64,
        window_size=10,
        action_dim=7,
        state_dim=8,
        max_samples=64,
    )
    actions = []
    for idx in range(64):
        actions.append(dataset[idx]["future_actions"])
    option_idx = action_window_pseudo_options(__import__("torch").stack(actions), 16)
    assert option_idx.min().item() >= 0
    assert option_idx.max().item() < 16
    assert len(set(option_idx.tolist())) >= 4
