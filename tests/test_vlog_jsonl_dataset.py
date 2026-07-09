import json

from starVLA.model.vlog_vla.dataset import VLOGJsonlWindowDataset
from scripts.vlog_vla.train_common import action_window_pseudo_options


def _write_libero_jsonl(path, num_rows=64):
    rows = []
    for idx in range(num_rows):
        episode_id = idx // 16
        step_id = idx % 16
        base = float(idx)
        axis = idx % 6
        sign = 1.0 if (idx // 6) % 2 == 0 else -1.0
        gripper = 1.0 if (idx // 12) % 2 == 0 else 0.0
        action_chunk = []
        for horizon in range(10):
            action = [0.001 * (dim + 1) for dim in range(7)]
            action[axis] = sign * (0.05 + 0.002 * horizon)
            action[6] = gripper
            action_chunk.append(action)
        rows.append(
            {
                "episode_id": episode_id,
                "step_id": step_id,
                "state": [base * 0.01 + j * 0.001 for j in range(8)],
                "action_chunk": action_chunk,
                "success": step_id == 15,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_vlog_jsonl_window_dataset_loads_real_libero_rows(tmp_path):
    jsonl_path = _write_libero_jsonl(tmp_path / "train.jsonl", num_rows=8)
    dataset = VLOGJsonlWindowDataset(
        jsonl_path,
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


def test_action_window_pseudo_options_are_trajectory_derived(tmp_path):
    jsonl_path = _write_libero_jsonl(tmp_path / "train.jsonl", num_rows=64)
    dataset = VLOGJsonlWindowDataset(
        jsonl_path,
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
