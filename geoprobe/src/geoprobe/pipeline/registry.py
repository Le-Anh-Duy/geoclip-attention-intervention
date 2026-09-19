from __future__ import annotations

import importlib

from geoprobe.registry import Registry

PIPELINE_BLOCKS = Registry("pipeline block")

_BLOCK_MODULES = {
    "CacheProposalsBlock": "geoprobe.pipeline.cache_proposals",
    "SearchBlock": "geoprobe.pipeline.search",
    "EvaluateBlock": "geoprobe.pipeline.evaluate",
    "ExportBlock": "geoprobe.pipeline.export",
}


def load_pipeline_block(type_name: str) -> None:
    try:
        module = _BLOCK_MODULES[type_name]
    except KeyError as exc:
        raise KeyError(f"Unknown pipeline block type {type_name!r}; available: {sorted(_BLOCK_MODULES)}") from exc
    importlib.import_module(module)
