from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, target: T | None = None):
        def add(value: T) -> T:
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in {self.name}")
            if not callable(value):
                raise TypeError(f"Registered value {name!r} must be callable")
            self._items[name] = value  # type: ignore[assignment]
            return value

        return add if target is None else add(target)

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name} type {name!r}; available: {choices}") from exc


def build(config: Mapping[str, Any], registry: Registry) -> Any:
    if not isinstance(config, Mapping):
        raise TypeError(f"Build config must be a mapping, got {type(config).__name__}")
    args = copy.deepcopy(dict(config))
    try:
        type_name = args.pop("type")
    except KeyError as exc:
        raise KeyError(f"Build config for {registry.name} requires 'type'") from exc
    if not isinstance(type_name, str):
        raise TypeError("Build config 'type' must be a string")
    return registry.get(type_name)(**args)
