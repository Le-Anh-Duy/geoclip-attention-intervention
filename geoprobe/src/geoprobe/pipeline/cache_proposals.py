from __future__ import annotations

from typing import Any, Mapping

from geoprobe.proposals import PROPOSERS
from geoprobe.proposals.wedetect import cache_proposals
from geoprobe.registry import build

from .context import PipelineContext
from .registry import PIPELINE_BLOCKS


@PIPELINE_BLOCKS.register("CacheProposalsBlock")
class CacheProposalsBlock:
    """Stage 1: run WeDetect once and freeze original-coordinate proposals."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.context = PipelineContext.create(config)

    def run(self) -> None:
        context = self.context
        context.write_resolved_config()
        proposer = build(context.config["proposer"], PROPOSERS)
        cache_proposals(
            proposer, context.dataset,
            context.output_dir / "proposals.jsonl",
            context.output_dir / "proposals.metadata.json",
            metadata_extra={
                "configuration_fingerprint": context.configuration_fingerprint,
                "split_fingerprint": context.split_fingerprint,
            },
        )
