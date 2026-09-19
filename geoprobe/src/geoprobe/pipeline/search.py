from __future__ import annotations

from typing import Any, Mapping

from PIL import Image
from tqdm import tqdm

from geoprobe.evaluation.metrics import clipped_improvement
from geoprobe.evaluation.search import generate_candidates, select_candidate
from geoprobe.io import SCHEMA_VERSION, atomic_write_json

from .context import PipelineContext
from .inference import PairInferenceBlock
from .registry import PIPELINE_BLOCKS


@PIPELINE_BLOCKS.register("SearchBlock")
class SearchBlock:
    """Stage 2: evaluate the explicit layer search order on discovery rows."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.context = PipelineContext.create(config)
        self.pairs = PairInferenceBlock(self.context)

    def run(self) -> None:
        context = self.context
        context.write_resolved_config()
        search = context.intervention.search
        candidates = generate_candidates(search.layers, search.schedules, search.base_bias)
        candidate_rows: list[list[dict[str, Any]]] = [[] for _ in candidates]
        discovery = [sample for sample in context.dataset if sample.split == "discovery"]
        clip_km = float(context.config["metrics"]["clip_improvement_km"])
        for sample in tqdm(discovery, desc="GeoProbe search", unit="image"):
            mask = self.pairs.patch_mask(context.proposal_rows[sample.sample_id])
            with Image.open(sample.image_path) as opened:
                image = opened.convert("RGB")
                baseline = self.pairs.baseline(image)
                baseline_error = self.pairs.error_km(sample, baseline)
                for index, candidate in enumerate(candidates):
                    intervened = self.pairs.intervened(image, mask, candidate, baseline)
                    intervened_error = self.pairs.error_km(sample, intervened)
                    candidate_rows[index].append({
                        "split": sample.split,
                        "baseline_error_km": baseline_error,
                        "intervened_error_km": intervened_error,
                        "clipped_improvement_km": float(clipped_improvement(
                            baseline_error, intervened_error, clip_km
                        )),
                    })
        winner, scores = select_candidate(list(zip(candidates, candidate_rows)), clip_km)
        winner_index = candidates.index(winner)
        artifact_candidates = [
            {
                **candidate.to_dict(), "discovery_score": score,
                "sample_improvements_km": [row["clipped_improvement_km"] for row in rows],
            }
            for candidate, score, rows in zip(candidates, scores, candidate_rows)
        ]
        atomic_write_json(context.output_dir / "search.json", {
            "schema_version": SCHEMA_VERSION,
            "configuration_fingerprint": context.configuration_fingerprint,
            "split_fingerprint": context.split_fingerprint,
            "sample_ids": [sample.sample_id for sample in discovery],
            "candidates": artifact_candidates,
            "winner_index": winner_index,
            "winner": winner.to_dict(),
        })
