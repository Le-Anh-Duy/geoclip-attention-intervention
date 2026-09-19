from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
from tqdm import tqdm

from geoprobe.evaluation.search import Candidate
from geoprobe.io import SCHEMA_VERSION, atomic_write_jsonl, read_jsonl

from .context import PipelineContext
from .inference import PairInferenceBlock
from .registry import PIPELINE_BLOCKS

_REQUIRED_PAIR_FIELDS = {
    "baseline_prediction", "baseline_score", "baseline_error_km", "baseline_embedding",
    "intervened_prediction", "intervened_score", "intervened_error_km", "intervened_embedding",
}


@PIPELINE_BLOCKS.register("EvaluateBlock")
class EvaluateBlock:
    """Stage 3: run one frozen candidate on holdout or all samples."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.context = PipelineContext.create(config)
        self.pairs = PairInferenceBlock(self.context)

    def run(self, *, selection_path: Path, split: str) -> None:
        if split not in {"holdout", "all"}:
            raise ValueError("Evaluation split must be 'holdout' or 'all'")
        context = self.context
        context.write_resolved_config()
        with Path(selection_path).open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        self._validate_selection(selection)
        candidate = Candidate.from_mapping(selection["winner"])
        requested = list(context.dataset) if split == "all" else [
            sample for sample in context.dataset if sample.split == "holdout"
        ]
        path = context.output_dir / f"evaluation_{split}.jsonl"
        rows = self._resume_rows(path, candidate)
        by_id = {row["sample_id"]: row for row in rows}
        expected_ids = {sample.sample_id for sample in requested}
        if not set(by_id).issubset(expected_ids):
            raise ValueError(f"Cannot resume {path}: contains samples outside requested split")
        for sample in tqdm(requested, desc=f"GeoProbe evaluate {split}", unit="image"):
            if sample.sample_id in by_id:
                continue
            mask = self.pairs.patch_mask(context.proposal_rows[sample.sample_id])
            with Image.open(sample.image_path) as opened:
                image = opened.convert("RGB")
                baseline = self.pairs.baseline(image)
                intervened = self.pairs.intervened(image, mask, candidate, baseline)
            row = self.pairs.paired_row(sample, mask, baseline, intervened, candidate)
            rows.append(row)
            by_id[sample.sample_id] = row
            atomic_write_jsonl(path, rows)
        if set(by_id) != expected_ids:
            raise AssertionError("Evaluation did not produce a complete set of paired rows")
        order = {sample.sample_id: index for index, sample in enumerate(requested)}
        rows.sort(key=lambda row: order[row["sample_id"]])
        atomic_write_jsonl(path, rows)

    def _validate_selection(self, selection: Mapping[str, Any]) -> None:
        context = self.context
        if (
            selection.get("schema_version") != SCHEMA_VERSION
            or selection.get("configuration_fingerprint") != context.configuration_fingerprint
            or selection.get("split_fingerprint") != context.split_fingerprint
        ):
            raise ValueError("Search selection belongs to a different configuration or split")
        discovery_ids = [sample.sample_id for sample in context.dataset if sample.split == "discovery"]
        if selection.get("sample_ids") != discovery_ids:
            raise ValueError("Search selection discovery sample IDs do not match the dataset")

    def _resume_rows(self, path: Path, candidate: Candidate) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows = read_jsonl(path)
        ids = [row.get("sample_id") for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Cannot resume {path}: duplicate sample IDs")
        context = self.context
        for row in rows:
            if (
                not _REQUIRED_PAIR_FIELDS.issubset(row)
                or row.get("configuration_fingerprint") != context.configuration_fingerprint
                or row.get("split_fingerprint") != context.split_fingerprint
            ):
                raise ValueError(f"Cannot resume {path}: incomplete or mismatched paired row {row.get('sample_id')!r}")
            if (
                row.get("selected_layers") != list(candidate.layers)
                or row.get("schedule") != candidate.schedule
                or row.get("base_bias") != candidate.base_bias
            ):
                raise ValueError(f"Cannot resume {path}: intervention selection changed")
        return rows
