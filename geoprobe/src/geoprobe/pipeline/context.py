from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

from geoprobe.config import config_fingerprint
from geoprobe.datasets import DATASETS, split_fingerprint
from geoprobe.intervention.config import InterventionConfig
from geoprobe.io import SCHEMA_VERSION, atomic_write_json
from geoprobe.registry import build


@dataclass(frozen=True)
class PipelineContext:
    """Shared immutable inputs; expensive dataset/model objects stay lazy."""

    config: Mapping[str, Any]
    output_dir: Path
    configuration_fingerprint: str
    intervention: InterventionConfig

    @classmethod
    def create(cls, config: Mapping[str, Any]) -> "PipelineContext":
        output = Path(config["runtime"]["output_dir"]).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        return cls(
            config=copy.deepcopy(dict(config)),
            output_dir=output,
            configuration_fingerprint=config_fingerprint(dict(config)),
            intervention=InterventionConfig.from_mapping(config["intervention"]),
        )

    @cached_property
    def dataset(self) -> Sequence[Any]:
        return build(self.config["dataset"], DATASETS)

    @cached_property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.dataset)

    @cached_property
    def split_fingerprint(self) -> str:
        return split_fingerprint(self.dataset)

    @cached_property
    def proposal_rows(self) -> dict[str, dict[str, Any]]:
        from geoprobe.proposals import read_proposal_cache

        _, rows = read_proposal_cache(
            self.output_dir / "proposals.jsonl",
            self.output_dir / "proposals.metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "configuration_fingerprint": self.configuration_fingerprint,
                "split_fingerprint": self.split_fingerprint,
            },
        )
        indexed = {str(row["sample_id"]): row for row in rows}
        if set(indexed) != set(self.sample_ids):
            raise ValueError("Proposal cache sample IDs do not match the configured dataset")
        return indexed

    @cached_property
    def localizer(self) -> Any:
        from geoprobe.models import LOCALIZERS

        settings = copy.deepcopy(self.config["geoclip"])
        cache = Path(settings["gallery_cache"])
        if not cache.is_absolute():
            settings["gallery_cache"] = str(self.output_dir / cache)
        settings.update(
            device=self.config["runtime"]["device"],
            deterministic=self.config["runtime"]["deterministic"],
            dtype=self.config["runtime"]["dtype"],
        )
        return build(settings, LOCALIZERS)

    def write_resolved_config(self, sample_ids: Sequence[str] | None = None) -> None:
        atomic_write_json(self.output_dir / "resolved_config.json", {
            "schema_version": SCHEMA_VERSION,
            "configuration_fingerprint": self.configuration_fingerprint,
            "sample_ids": list(self.sample_ids if sample_ids is None else sample_ids),
            "config": dict(self.config),
        })
