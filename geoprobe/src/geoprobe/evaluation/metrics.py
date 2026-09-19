from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0088
THRESHOLDS_KM = (1.0, 25.0, 200.0, 750.0, 2500.0)


def haversine_distance_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    lat1a, lon1a, lat2a, lon2a = np.broadcast_arrays(
        np.asarray(lat1, dtype=np.float64), np.asarray(lon1, dtype=np.float64),
        np.asarray(lat2, dtype=np.float64), np.asarray(lon2, dtype=np.float64),
    )
    values = np.stack((lat1a, lon1a, lat2a, lon2a))
    if not np.isfinite(values).all():
        raise ValueError("Haversine inputs must be finite")
    if np.any(np.abs(lat1a) > 90) or np.any(np.abs(lat2a) > 90):
        raise ValueError("Latitude must be in [-90, 90]")
    if np.any(np.abs(lon1a) > 180) or np.any(np.abs(lon2a) > 180):
        raise ValueError("Longitude must be in [-180, 180]")
    phi1, phi2 = np.radians(lat1a), np.radians(lat2a)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2a - lon1a)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def embedding_cosine(left: Any, right: Any) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1:
        raise ValueError(f"Embedding shapes must match, got {a.shape} and {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Embeddings must be finite")
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denominator == 0):
        raise ValueError("Embedding cosine is undefined for zero vectors")
    return np.sum(a * b, axis=-1) / denominator


def clipped_improvement(baseline_error: Any, intervened_error: Any, clip_km: float = 2500.0) -> np.ndarray:
    baseline, intervened = np.broadcast_arrays(
        np.asarray(baseline_error, dtype=np.float64), np.asarray(intervened_error, dtype=np.float64)
    )
    if not np.isfinite(baseline).all() or not np.isfinite(intervened).all():
        raise ValueError("Errors must be finite")
    if not np.isfinite(clip_km) or clip_km <= 0:
        raise ValueError("clip_km must be positive and finite")
    return np.clip(baseline - intervened, -clip_km, clip_km)


def method_metrics(errors_km: Any, thresholds_km: Sequence[float] = THRESHOLDS_KM) -> dict[str, float]:
    errors = np.asarray(errors_km, dtype=np.float64)
    if errors.ndim != 1 or errors.size == 0 or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("errors_km must be a non-empty finite non-negative vector")
    result = {f"accuracy@{int(t) if float(t).is_integer() else t}km": float(np.mean(errors <= t)) for t in thresholds_km}
    result["mean_error_km"] = float(np.mean(errors))
    return result


def paired_bootstrap_mean(
    baseline_values: Any,
    intervened_values: Any,
    *,
    seed: int = 42,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_values, dtype=np.float64)
    intervened = np.asarray(intervened_values, dtype=np.float64)
    if baseline.ndim != 1 or baseline.shape != intervened.shape or baseline.size == 0:
        raise ValueError("Paired bootstrap inputs must be equal non-empty vectors")
    if not np.isfinite(baseline).all() or not np.isfinite(intervened).all():
        raise ValueError("Paired bootstrap inputs must be finite")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    base_samples = np.empty(resamples, dtype=np.float64)
    intervention_samples = np.empty(resamples, dtype=np.float64)
    chunk = max(1, min(resamples, 1_000_000 // baseline.size))
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, baseline.size, size=(stop - start, baseline.size))
        base_samples[start:stop] = baseline[indices].mean(axis=1)
        intervention_samples[start:stop] = intervened[indices].mean(axis=1)
    delta_samples = intervention_samples - base_samples
    alpha = (1.0 - confidence) / 2.0
    interval = lambda values: [float(x) for x in np.quantile(values, [alpha, 1.0 - alpha])]
    return {
        "baseline": float(np.mean(baseline)),
        "intervened": float(np.mean(intervened)),
        "delta": float(np.mean(intervened - baseline)),
        "baseline_ci": interval(base_samples),
        "intervened_ci": interval(intervention_samples),
        "delta_ci": interval(delta_samples),
    }


def paired_metrics_with_intervals(
    baseline_errors: Any,
    intervened_errors: Any,
    *,
    thresholds_km: Sequence[float] = THRESHOLDS_KM,
    seed: int = 42,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, dict[str, Any]]:
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    intervened = np.asarray(intervened_errors, dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    for threshold in thresholds_km:
        key = f"accuracy@{int(threshold) if float(threshold).is_integer() else threshold}km"
        result[key] = paired_bootstrap_mean(
            (baseline <= threshold).astype(np.float64), (intervened <= threshold).astype(np.float64),
            seed=seed, resamples=resamples, confidence=confidence,
        )
    result["mean_error_km"] = paired_bootstrap_mean(
        baseline, intervened, seed=seed, resamples=resamples, confidence=confidence
    )
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty row group")
    baseline = np.asarray([row["baseline_error_km"] for row in rows], dtype=np.float64)
    intervened = np.asarray([row["intervened_error_km"] for row in rows], dtype=np.float64)
    displacement = np.asarray([row["prediction_displacement_km"] for row in rows], dtype=np.float64)
    cosine = np.asarray([row["embedding_cosine"] for row in rows], dtype=np.float64)
    improvement = clipped_improvement(baseline, intervened)
    return {
        "count": len(rows),
        "baseline": method_metrics(baseline),
        "intervened": method_metrics(intervened),
        "mean_prediction_displacement_km": float(np.mean(displacement)),
        "mean_embedding_cosine": float(np.mean(cosine)),
        "mean_clipped_improvement_km": float(np.mean(improvement)),
    }


def selected_vs_best_single_advantage(selected_improvement: Any, best_single_improvement: Any) -> float:
    selected, single = np.broadcast_arrays(
        np.asarray(selected_improvement, dtype=np.float64), np.asarray(best_single_improvement, dtype=np.float64)
    )
    if selected.ndim != 1 or selected.size == 0 or not np.isfinite(selected).all() or not np.isfinite(single).all():
        raise ValueError("Paired candidate improvements must be finite non-empty vectors")
    return float(np.mean(selected - single))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def candidate_rank_correlation(discovery_scores: Any, holdout_scores: Any) -> float:
    discovery = np.asarray(discovery_scores, dtype=np.float64)
    holdout = np.asarray(holdout_scores, dtype=np.float64)
    if discovery.ndim != 1 or discovery.shape != holdout.shape or discovery.size < 2:
        raise ValueError("Candidate score arrays must be equal vectors of length at least two")
    if not np.isfinite(discovery).all() or not np.isfinite(holdout).all():
        raise ValueError("Candidate scores must be finite")
    left, right = _average_ranks(discovery), _average_ranks(holdout)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else float("nan")
