from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from geoprobe.intervention.schedules import schedule_biases

PAPER_LAYERS = (2, 4, 7, 12, 3)
PAPER_SCHEDULES = ("fixed", "sqrt", "linear")


@dataclass(frozen=True)
class Candidate:
    layers: tuple[int, ...]
    schedule: str
    base_bias: float = 2.0

    @property
    def biases(self) -> tuple[float, ...]:
        return schedule_biases(self.schedule, self.base_bias, len(self.layers))

    def to_dict(self) -> dict[str, Any]:
        return {"layers": list(self.layers), "schedule": self.schedule, "base_bias": self.base_bias}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Candidate":
        return cls(tuple(int(layer) for layer in value["layers"]), str(value["schedule"]), float(value["base_bias"]))


def generate_candidates(
    layers: Sequence[int] = PAPER_LAYERS, schedules: Sequence[str] = PAPER_SCHEDULES, base_bias: float = 2.0
) -> list[Candidate]:
    ordered = tuple(layers)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Candidate layer source must be unique")
    result = [Candidate((layer,), "fixed", base_bias) for layer in ordered]
    for size in range(2, len(ordered) + 1):
        for subset in combinations(ordered, size):
            for schedule in schedules:
                result.append(Candidate(subset, schedule, base_bias))
    return result


def score_candidate(rows: Sequence[Mapping[str, Any]], clip_km: float = 2500.0) -> float:
    if not rows:
        raise ValueError("Candidate score requires at least one discovery row")
    if any(row.get("split") != "discovery" for row in rows):
        raise ValueError("Search accepts only rows marked 'discovery'")
    baseline = np.asarray([row["baseline_error_km"] for row in rows], dtype=np.float64)
    intervened = np.asarray([row["intervened_error_km"] for row in rows], dtype=np.float64)
    if not np.isfinite(baseline).all() or not np.isfinite(intervened).all():
        raise ValueError("Candidate rows contain non-finite errors")
    return float(np.mean(np.clip(baseline - intervened, -clip_km, clip_km)))


def select_candidate(
    candidate_rows: Sequence[tuple[Candidate, Sequence[Mapping[str, Any]]]], clip_km: float = 2500.0
) -> tuple[Candidate, list[float]]:
    if not candidate_rows:
        raise ValueError("No candidates were evaluated")
    scores = [score_candidate(rows, clip_km) for _, rows in candidate_rows]
    winner_index = max(range(len(scores)), key=scores.__getitem__)
    return candidate_rows[winner_index][0], scores
