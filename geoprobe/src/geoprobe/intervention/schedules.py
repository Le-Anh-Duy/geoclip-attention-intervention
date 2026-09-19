from __future__ import annotations

import math


def schedule_biases(name: str, base_bias: float, layer_count: int) -> tuple[float, ...]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if not math.isfinite(base_bias):
        raise ValueError("base_bias must be finite")
    if name == "fixed":
        value = base_bias
    elif name == "sqrt":
        value = base_bias / math.sqrt(layer_count)
    elif name == "linear":
        value = base_bias / layer_count
    else:
        raise ValueError(f"Unknown schedule {name!r}; expected fixed, sqrt, or linear")
    return (value,) * layer_count
