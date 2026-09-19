from __future__ import annotations

import math
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InterventionState:
    patch_mask: Any
    layer_biases: Mapping[int, float]
    patch_grid: int = 16

    def validated(self, layer_count: int = 24) -> "InterventionState":
        mask = torch.as_tensor(self.patch_mask, dtype=torch.bool)
        if tuple(mask.shape) != (self.patch_grid, self.patch_grid):
            raise ValueError(
                f"Patch mask must have shape {(self.patch_grid, self.patch_grid)}, got {tuple(mask.shape)}"
            )
        normalized: dict[int, float] = {}
        for raw_layer, raw_bias in self.layer_biases.items():
            layer, bias = int(raw_layer), float(raw_bias)
            if layer != raw_layer or not 0 <= layer < layer_count:
                raise ValueError(f"Layer index {raw_layer!r} is outside [0, {layer_count})")
            if not math.isfinite(bias):
                raise ValueError(f"Bias for layer {layer} must be finite")
            normalized[layer] = bias
        return InterventionState(mask, normalized, self.patch_grid)

    @property
    def active(self) -> bool:
        mask = torch.as_tensor(self.patch_mask, dtype=torch.bool)
        return bool(mask.any()) and bool(self.layer_biases) and any(float(value) != 0.0 for value in self.layer_biases.values())

    def key_bias(self, layer: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        bias = float(self.layer_biases[layer])
        mask = torch.as_tensor(self.patch_mask, device=device, dtype=torch.bool).reshape(-1)
        patch_bias = torch.where(
            mask,
            torch.tensor(bias, device=device, dtype=dtype),
            torch.tensor(-bias, device=device, dtype=dtype),
        )
        return torch.cat((torch.zeros(1, device=device, dtype=dtype), patch_bias))


def _intervened_attention_forward(state: InterventionState, layer_idx: int):
    def forward(
        self: Any,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, target_length, embed_dim = hidden_states.size()
        expected_tokens = state.patch_grid * state.patch_grid + 1
        if target_length != expected_tokens:
            raise ValueError(
                f"Layer {layer_idx} expected {expected_tokens} CLS+patch tokens, got {target_length}"
            )
        projection_shape = (batch_size * self.num_heads, target_length, self.head_dim)
        head_shape = (batch_size, target_length, self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).view(head_shape).transpose(1, 2).reshape(projection_shape) * self.scale
        key_states = self.k_proj(hidden_states).view(head_shape).transpose(1, 2).reshape(projection_shape)
        value_states = self.v_proj(hidden_states).view(head_shape).transpose(1, 2).reshape(projection_shape)
        source_length = key_states.size(1)
        attention_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        expected_shape = (batch_size * self.num_heads, target_length, source_length)
        if attention_weights.size() != expected_shape:
            raise ValueError(f"Attention weights should be {expected_shape}, got {tuple(attention_weights.size())}")
        if causal_attention_mask is not None:
            expected_mask = (batch_size, 1, target_length, source_length)
            if causal_attention_mask.size() != expected_mask:
                raise ValueError(f"Causal attention mask should be {expected_mask}")
            attention_weights = attention_weights.view(batch_size, self.num_heads, target_length, source_length)
            attention_weights = attention_weights + causal_attention_mask
            attention_weights = attention_weights.view(*expected_shape)
        if attention_mask is not None:
            expected_mask = (batch_size, 1, target_length, source_length)
            if attention_mask.size() != expected_mask:
                raise ValueError(f"Attention mask should be {expected_mask}")
            attention_weights = attention_weights.view(batch_size, self.num_heads, target_length, source_length)
            attention_weights = attention_weights + attention_mask
            attention_weights = attention_weights.view(*expected_shape)
        key_bias = state.key_bias(layer_idx, device=attention_weights.device, dtype=attention_weights.dtype)
        if key_bias.numel() != source_length:
            raise ValueError(f"Layer {layer_idx} bias has {key_bias.numel()} keys, attention has {source_length}")
        attention_weights = attention_weights + key_bias.view(1, 1, source_length)
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        returned_weights = None
        if output_attentions:
            returned_weights = attention_weights.view(batch_size, self.num_heads, target_length, source_length)
            attention_weights = returned_weights.view(*expected_shape)
        attention_probabilities = F.dropout(attention_weights, p=self.dropout, training=self.training)
        attention_output = torch.bmm(attention_probabilities, value_states)
        if attention_output.size() != (batch_size * self.num_heads, target_length, self.head_dim):
            raise ValueError("Unexpected attention output shape")
        attention_output = attention_output.view(batch_size, self.num_heads, target_length, self.head_dim)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, target_length, embed_dim)
        return self.out_proj(attention_output), returned_weights

    return forward


def _vision_layers(vision_tower: Any) -> Any:
    root = getattr(vision_tower, "vision_model", vision_tower)
    try:
        return root.encoder.layers
    except AttributeError as exc:
        raise TypeError("Vision tower must expose vision_model.encoder.layers or encoder.layers") from exc


@contextmanager
def patch_vision_tower(
    vision_tower: Any,
    state: InterventionState,
    *,
    expected_layers: int = 24,
    expected_heads: int = 16,
) -> Iterator[InterventionState]:
    validated = state.validated(expected_layers)
    if not validated.active:
        yield validated
        return
    layers = _vision_layers(vision_tower)
    if len(layers) != expected_layers:
        raise ValueError(f"Expected {expected_layers} vision layers, found {len(layers)}")
    root = getattr(vision_tower, "vision_model", vision_tower)
    config = getattr(root, "config", None)
    image_size = getattr(config, "image_size", None)
    patch_size = getattr(config, "patch_size", None)
    if image_size is not None and patch_size is not None:
        if image_size % patch_size or image_size // patch_size != validated.patch_grid:
            raise ValueError(
                f"Vision tower image/patch sizes {image_size}/{patch_size} do not produce "
                f"the configured {validated.patch_grid}x{validated.patch_grid} grid"
            )
    position_embedding = getattr(getattr(root, "embeddings", None), "position_embedding", None)
    token_count = getattr(position_embedding, "num_embeddings", None)
    expected_tokens = validated.patch_grid * validated.patch_grid + 1
    if token_count is not None and token_count != expected_tokens:
        raise ValueError(f"Vision tower has {token_count} tokens; expected {expected_tokens}")
    for index, layer in enumerate(layers):
        heads = getattr(layer.self_attn, "num_heads", None)
        if heads != expected_heads:
            raise ValueError(f"Layer {index} expected {expected_heads} attention heads, found {heads}")
    originals: list[tuple[Any, Any]] = []
    try:
        for index, bias in validated.layer_biases.items():
            if bias == 0.0:
                continue
            attention = layers[index].self_attn
            originals.append((attention, attention.forward))
            attention.forward = types.MethodType(_intervened_attention_forward(validated, index), attention)
        yield validated
    finally:
        for attention, original in originals:
            attention.forward = original
