from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image

from geoprobe.evaluation.metrics import clipped_improvement, embedding_cosine, haversine_distance_km
from geoprobe.evaluation.search import Candidate
from geoprobe.intervention import InterventionState, patch_vision_tower, proposal_union_mask
from geoprobe.io import SCHEMA_VERSION

from .context import PipelineContext


class PairInferenceBlock:
    """The only block that knows how baseline/intervention forwards are paired."""

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def patch_mask(self, proposal_row: Mapping[str, Any]) -> np.ndarray:
        spec = self.context.intervention.mask
        return proposal_union_mask(
            proposal_row["proposals"], int(proposal_row["width"]), int(proposal_row["height"]),
            score_threshold=spec.score_threshold, max_proposals=spec.max_proposals,
            crop_size=spec.crop_size, patch_grid=spec.patch_grid,
        )

    def baseline(self, image: Image.Image) -> Any:
        return self.context.localizer.localize(image)

    def intervened(self, image: Image.Image, mask: np.ndarray, candidate: Candidate, baseline: Any) -> Any:
        if not mask.any():
            return baseline
        spec = self.context.intervention
        state = InterventionState(mask, dict(zip(candidate.layers, candidate.biases)), spec.mask.patch_grid)
        with patch_vision_tower(
            self.context.localizer.vision_tower, state,
            expected_layers=spec.vision.layer_count, expected_heads=spec.vision.expected_heads,
        ):
            return self.context.localizer.localize(image)

    @staticmethod
    def error_km(sample: Any, prediction: Any) -> float:
        return float(haversine_distance_km(
            sample.latitude, sample.longitude, prediction.latitude, prediction.longitude
        ))

    def paired_row(
        self, sample: Any, mask: np.ndarray, baseline: Any, intervened: Any, candidate: Candidate
    ) -> dict[str, Any]:
        baseline_error = self.error_km(sample, baseline)
        intervened_error = self.error_km(sample, intervened)
        displacement = float(haversine_distance_km(
            baseline.latitude, baseline.longitude, intervened.latitude, intervened.longitude
        ))
        cosine = float(embedding_cosine(np.asarray(baseline.embedding), np.asarray(intervened.embedding)))
        improvement = float(clipped_improvement(baseline_error, intervened_error))
        return {
            "schema_version": SCHEMA_VERSION,
            "configuration_fingerprint": self.context.configuration_fingerprint,
            "split_fingerprint": self.context.split_fingerprint,
            "gallery_fingerprint": self.context.localizer.gallery_fingerprint,
            "sample_id": sample.sample_id, "split": sample.split,
            "target": [sample.latitude, sample.longitude],
            "active_mask": bool(mask.any()), "mask_patch_count": int(mask.sum()),
            "baseline_prediction": [baseline.latitude, baseline.longitude],
            "baseline_score": baseline.score, "baseline_error_km": baseline_error,
            "baseline_embedding": list(baseline.embedding),
            "intervened_prediction": [intervened.latitude, intervened.longitude],
            "intervened_score": intervened.score, "intervened_error_km": intervened_error,
            "intervened_embedding": list(intervened.embedding),
            "selected_layers": list(candidate.layers), "schedule": candidate.schedule,
            "base_bias": candidate.base_bias, "biases": list(candidate.biases),
            "prediction_displacement_km": displacement, "embedding_cosine": cosine,
            "clipped_improvement_km": improvement,
        }
