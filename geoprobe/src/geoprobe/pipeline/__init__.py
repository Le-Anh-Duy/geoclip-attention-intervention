from __future__ import annotations

from typing import Any, Mapping

from geoprobe.registry import build

from .registry import PIPELINE_BLOCKS, load_pipeline_block


def build_pipeline_block(command: str, config: Mapping[str, Any]) -> Any:
    try:
        block_config = config["pipeline"][command]
    except KeyError as exc:
        raise KeyError(f"No pipeline block configured for command {command!r}") from exc
    type_name = block_config.get("type")
    if not isinstance(type_name, str):
        raise TypeError(f"Pipeline block for {command!r} requires a string 'type'")
    load_pipeline_block(type_name)
    return build({**block_config, "config": config}, PIPELINE_BLOCKS)


__all__ = [
    "PIPELINE_BLOCKS", "build_pipeline_block",
]
