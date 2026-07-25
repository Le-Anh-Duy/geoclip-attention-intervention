"""Region-scaled attention intervention for CLIP's vision tower.

The idea (as scoped with the user): after Q*K^T is normalized by softmax,
scale the attention probability that every query token assigns to each KEY
patch token by `a` if that patch falls inside a user-selected region, or by
`b` if it doesn't, then renormalize so each row still sums to 1. This nudges
every token's aggregated V (and, transitively, the final CLS representation)
to lean on the selected region a bit more (a > 1) and elsewhere a bit less
(b < 1). a = b = 1 is a no-op -> identical to the unmodified model.

Applies to every layer independently: each of the 24 CLIPEncoderLayer blocks
in ViT-L/14 gets its own (a, b), because a caller may want to intervene only
in a subset of layers.

The CLS token (sequence position 0) is never scaled -- it isn't a spatial
patch, so it has no "inside/outside region" meaning. This is a deliberate
default, not an oversight: see `_build_key_scale` below.
"""
import types
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class InterventionState:
    """Mutable, per-request config read by the patched attention forwards.

    One instance is attached to the model at load time and mutated (not
    replaced) before each forward pass -- see `run_with_intervention`.
    """
    # {layer_index: (a, b)}. A layer absent from this dict behaves as (1, 1).
    layer_ab: dict = field(default_factory=dict)
    # Bool tensor of length `grid_size * grid_size`, True = patch is inside
    # a user-selected region. None means "no regions selected" (all False).
    in_region_mask: torch.Tensor | None = None

    def key_scale_for_layer(self, layer_idx: int, num_positions: int) -> torch.Tensor | None:
        """Returns a (num_positions,) scale vector for this layer's keys, or
        None if this layer has no active intervention (fast no-op path)."""
        a, b = self.layer_ab.get(layer_idx, (1.0, 1.0))
        if a == 1.0 and b == 1.0:
            return None
        if self.in_region_mask is None:
            return None
        scale = torch.full((num_positions,), b, dtype=torch.float32)
        scale[0] = 1.0  # CLS token key: never scaled, see module docstring
        scale[1:] = torch.where(self.in_region_mask, torch.tensor(a), torch.tensor(b))
        return scale


def _intervened_attention_forward(state: InterventionState, layer_idx: int):
    """Builds a replacement for `CLIPAttention.forward` bound to one layer.

    Reimplements the same q/k/v projection + eager attention math as
    transformers' `eager_attention_forward` (see
    transformers/models/clip/modeling_clip.py), except the scaling step is
    inserted AFTER softmax and BEFORE the matmul with V -- exactly the point
    in the pipeline the user asked for. We bypass `ALL_ATTENTION_FUNCTIONS`
    entirely so this keeps working regardless of the configured attn
    implementation (eager/sdpa/flash).
    """

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        queries = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)

        key_scale = state.key_scale_for_layer(layer_idx, attn_weights.shape[-1])
        if key_scale is not None:
            # <-- the actual intervention: scale attention probs per-key,
            # a inside the selected region(s), b outside, then renormalize.
            attn_weights = attn_weights * key_scale.to(attn_weights.dtype)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

    return forward


def patch_vision_tower(clip_model) -> InterventionState:
    """Binds the intervention-aware forward onto every vision encoder layer.

    Call once, right after the model is loaded. Returns the shared
    InterventionState object callers mutate before each forward pass.
    """
    state = InterventionState()
    layers = clip_model.vision_model.encoder.layers
    for layer_idx, layer in enumerate(layers):
        layer.self_attn.forward = types.MethodType(
            _intervened_attention_forward(state, layer_idx), layer.self_attn
        )
    return state


def grid_size(clip_model) -> int:
    """Patches-per-side for the vision tower (16 for ViT-L/14 @ 224/14)."""
    cfg = clip_model.vision_model.config
    return cfg.image_size // cfg.patch_size


def num_layers(clip_model) -> int:
    return len(clip_model.vision_model.encoder.layers)


def build_in_region_mask(regions, orig_w: int, orig_h: int, grid: int) -> torch.Tensor:
    """Maps user-drawn regions (normalized x,y,w,h in [0,1] against the
    ORIGINAL uploaded image) to a boolean mask over the `grid x grid` patch
    tokens fed to CLIP.

    CLIP's own preprocessor resizes the shortest side to 224 then
    center-crops to 224x224 (see AutoProcessor for openai/clip-vit-large-patch14:
    do_resize=True/shortest_edge=224, do_center_crop=True/crop_size=224).
    We replicate that exact transform on the region coordinates so a patch
    is marked "in region" iff it actually is, post-preprocessing -- getting
    this wrong would silently misalign every region the user draws.
    """
    image_size = 224
    scale = image_size / min(orig_w, orig_h)
    resized_w, resized_h = orig_w * scale, orig_h * scale
    crop_x0 = (resized_w - image_size) / 2
    crop_y0 = (resized_h - image_size) / 2

    mask = torch.zeros((grid, grid), dtype=torch.bool)
    patch = image_size / grid

    for region in regions:
        x0 = region["x"] * orig_w * scale - crop_x0
        y0 = region["y"] * orig_h * scale - crop_y0
        x1 = (region["x"] + region["w"]) * orig_w * scale - crop_x0
        y1 = (region["y"] + region["h"]) * orig_h * scale - crop_y0
        # clip to the visible 224x224 crop
        x0, x1 = max(x0, 0), min(x1, image_size)
        y0, y1 = max(y0, 0), min(y1, image_size)
        if x1 <= x0 or y1 <= y0:
            continue  # region fell entirely outside the center-cropped area

        col0, col1 = int(x0 // patch), min(int((x1 - 1e-6) // patch) + 1, grid)
        row0, row1 = int(y0 // patch), min(int((y1 - 1e-6) // patch) + 1, grid)
        mask[row0:row1, col0:col1] = True

    return mask.flatten()
