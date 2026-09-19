from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class LayerSpec:
    """One zero-based visual-transformer layer and its paper-search position."""

    index: int
    name: str
    search_order: int | None


@dataclass(frozen=True)
class MaskSpec:
    score_threshold: float
    max_proposals: int
    crop_size: int
    patch_grid: int


@dataclass(frozen=True)
class SearchSpec:
    layers: tuple[int, ...]
    schedules: tuple[str, ...]
    base_bias: float


@dataclass(frozen=True)
class VisionSpec:
    layers: tuple[LayerSpec, ...]
    expected_heads: int

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def layer(self, index: int) -> LayerSpec:
        if not 0 <= index < len(self.layers):
            raise IndexError(f"Vision layer {index} is outside [0, {len(self.layers)})")
        return self.layers[index]


@dataclass(frozen=True)
class InterventionConfig:
    """Validated block-level view of the editable intervention configuration."""

    mask: MaskSpec
    vision: VisionSpec
    search: SearchSpec

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InterventionConfig":
        if set(value) != {"mask", "vision", "search"}:
            raise ValueError("intervention must contain exactly mask, vision, and search blocks")

        raw_mask = _mapping(value["mask"], "intervention.mask")
        _require_keys(raw_mask, {"score_threshold", "max_proposals", "crop_size", "patch_grid"}, "intervention.mask")
        mask = MaskSpec(
            score_threshold=float(raw_mask["score_threshold"]),
            max_proposals=int(raw_mask["max_proposals"]),
            crop_size=int(raw_mask["crop_size"]),
            patch_grid=int(raw_mask["patch_grid"]),
        )
        if not 0.0 <= mask.score_threshold <= 1.0:
            raise ValueError("intervention.mask.score_threshold must be in [0, 1]")
        if mask.max_proposals < 0 or mask.crop_size <= 0 or mask.patch_grid <= 0:
            raise ValueError("intervention mask sizes must be positive (max_proposals may be zero)")
        if mask.crop_size % mask.patch_grid:
            raise ValueError("intervention.mask.crop_size must be divisible by patch_grid")

        raw_vision = _mapping(value["vision"], "intervention.vision")
        _require_keys(raw_vision, {"expected_heads", "layers"}, "intervention.vision")
        raw_layers = raw_vision["layers"]
        if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes)) or not raw_layers:
            raise TypeError("intervention.vision.layers must be a non-empty sequence")
        layers: list[LayerSpec] = []
        for position, raw in enumerate(raw_layers):
            layer = _mapping(raw, f"intervention.vision.layers[{position}]")
            _require_keys(layer, {"index", "name", "search_order"}, f"intervention.vision.layers[{position}]")
            index = int(layer["index"])
            name = str(layer["name"])
            raw_order = layer["search_order"]
            search_order = None if raw_order is None else int(raw_order)
            if index != layer["index"] or not name:
                raise ValueError(f"Invalid layer specification at position {position}")
            layers.append(LayerSpec(index, name, search_order))
        indices = [layer.index for layer in layers]
        if indices != list(range(len(layers))):
            raise ValueError(f"Vision layer indices must be unique and contiguous from zero; found {indices}")
        names = [layer.name for layer in layers]
        if len(names) != len(set(names)):
            raise ValueError("Vision layer names must be unique")
        ranked = [layer for layer in layers if layer.search_order is not None]
        orders = [layer.search_order for layer in ranked]
        if sorted(orders) != list(range(len(ranked))):
            raise ValueError(f"Layer search_order values must be unique and contiguous from zero; found {orders}")
        ranked.sort(key=lambda layer: layer.search_order)
        expected_heads = int(raw_vision["expected_heads"])
        if expected_heads <= 0:
            raise ValueError("intervention.vision.expected_heads must be positive")
        vision = VisionSpec(tuple(layers), expected_heads)

        raw_search = _mapping(value["search"], "intervention.search")
        _require_keys(raw_search, {"schedules", "base_bias"}, "intervention.search")
        schedules = tuple(str(name) for name in raw_search["schedules"])
        if not schedules or len(schedules) != len(set(schedules)):
            raise ValueError("intervention.search.schedules must be non-empty and unique")
        allowed = {"fixed", "sqrt", "linear"}
        if any(name not in allowed for name in schedules):
            raise ValueError(f"Unknown intervention schedule; allowed: {sorted(allowed)}")
        base_bias = float(raw_search["base_bias"])
        if not math.isfinite(base_bias):
            raise ValueError("intervention.search.base_bias must be finite")
        search = SearchSpec(tuple(layer.index for layer in ranked), schedules, base_bias)
        if not search.layers:
            raise ValueError("At least one vision layer must have a search_order")
        return cls(mask, vision, search)


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{location} must contain exactly {sorted(expected)}; found {sorted(value)}")
