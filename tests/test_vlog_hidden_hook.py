import torch
import torch.nn as nn

from starVLA.model.vlog_vla.hidden_hook_utils import HiddenTokenHook


def test_hidden_hook_captures_input_and_does_not_change_output():
    module = nn.Linear(4, 3)
    x = torch.randn(2, 5, 4)
    with torch.no_grad():
        expected = module(x)
    hook = HiddenTokenHook(module)
    with hook:
        actual = module(x)
        assert hook.last_inputs is not None
        assert hook.last_inputs[0].shape == (2, 5, 4)
    assert torch.allclose(actual, expected)
    assert hook.handle is None
    hook.last_inputs = None
    module(x)
    assert hook.last_inputs is None
