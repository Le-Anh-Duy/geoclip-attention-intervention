from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from geoprobe.intervention.attention import InterventionState, _intervened_attention_forward, patch_vision_tower


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 1
        self.head_dim = 1
        self.scale = 1.0
        self.dropout = 0.0
        self.q_proj = nn.Linear(1, 1, bias=False)
        self.k_proj = nn.Linear(1, 1, bias=False)
        self.v_proj = nn.Linear(1, 1, bias=False)
        self.out_proj = nn.Identity()
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            layer.weight.data.fill_(1.0)

    def _shape(self, tensor, sequence_length, batch_size):
        return tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


def test_cls_key_is_neutral_and_patch_bias_precedes_softmax():
    attention = TinyAttention()
    state = InterventionState(np.ones((1, 1), dtype=bool), {0: 2.0}, patch_grid=1)
    forward = _intervened_attention_forward(state, 0).__get__(attention, TinyAttention)
    _, weights = forward(torch.zeros((1, 2, 1)), output_attentions=True)
    expected = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
    assert torch.equal(weights[0, 0, 0], expected)
    assert state.key_bias(0, device=torch.device("cpu"), dtype=torch.float32)[0].item() == 0.0


def test_patch_context_restores_forward_after_exception():
    attention = TinyAttention()
    tower = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    original = attention.forward
    with pytest.raises(RuntimeError, match="boom"):
        with patch_vision_tower(tower, InterventionState(np.ones((1, 1), bool), {0: 1.0}, 1), expected_layers=1, expected_heads=1):
            assert attention.forward.__func__ is not original.__func__
            raise RuntimeError("boom")
    assert attention.forward == original


@pytest.mark.parametrize("state", [
    InterventionState(np.zeros((1, 1), bool), {0: 2.0}, 1),
    InterventionState(np.ones((1, 1), bool), {}, 1),
    InterventionState(np.ones((1, 1), bool), {0: 0.0}, 1),
])
def test_empty_mask_no_layer_and_zero_bias_are_exact_noops(state):
    attention = TinyAttention()
    tower = SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    original = attention.forward
    with patch_vision_tower(tower, state, expected_layers=1, expected_heads=1):
        assert attention.forward == original
    assert attention.forward == original
