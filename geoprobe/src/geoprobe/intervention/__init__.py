from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "InterventionConfig": ("geoprobe.intervention.config", "InterventionConfig"),
    "LayerSpec": ("geoprobe.intervention.config", "LayerSpec"),
    "MaskSpec": ("geoprobe.intervention.config", "MaskSpec"),
    "SearchSpec": ("geoprobe.intervention.config", "SearchSpec"),
    "VisionSpec": ("geoprobe.intervention.config", "VisionSpec"),
    "InterventionState": ("geoprobe.intervention.attention", "InterventionState"),
    "patch_vision_tower": ("geoprobe.intervention.attention", "patch_vision_tower"),
    "boxes_to_patch_mask": ("geoprobe.intervention.masks", "boxes_to_patch_mask"),
    "proposal_union_mask": ("geoprobe.intervention.masks", "proposal_union_mask"),
    "schedule_biases": ("geoprobe.intervention.schedules", "schedule_biases"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value
