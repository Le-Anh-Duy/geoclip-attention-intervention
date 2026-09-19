import numpy as np

from geoprobe.evaluation.metrics import clipped_improvement, haversine_distance_km, paired_bootstrap_mean
from geoprobe.evaluation.search import generate_candidates, select_candidate


def test_haversine_known_quarter_circumference():
    assert float(haversine_distance_km(0, 0, 0, 0)) == 0.0
    assert np.isclose(float(haversine_distance_km(0, 0, 0, 90)), 10007.557221017962)


def test_exact_83_candidate_order_and_first_tie_wins():
    candidates = generate_candidates()
    assert len(candidates) == 83
    assert [candidate.layers for candidate in candidates[:5]] == [(2,), (4,), (7,), (12,), (3,)]
    assert [(candidate.layers, candidate.schedule) for candidate in candidates[5:8]] == [
        ((2, 4), "fixed"), ((2, 4), "sqrt"), ((2, 4), "linear")
    ]
    assert (candidates[-1].layers, candidates[-1].schedule) == ((2, 4, 7, 12, 3), "linear")
    rows = [{"split": "discovery", "baseline_error_km": 1.0, "intervened_error_km": 1.0}]
    winner, _ = select_candidate([(candidate, rows) for candidate in candidates])
    assert winner is candidates[0]


def test_clipping_and_paired_bootstrap_are_deterministic():
    assert clipped_improvement([5000, 0], [0, 5000]).tolist() == [2500, -2500]
    first = paired_bootstrap_mean([1, 10, 100], [2, 8, 90], seed=42, resamples=1000)
    second = paired_bootstrap_mean([1, 10, 100], [2, 8, 90], seed=42, resamples=1000)
    assert first == second
    assert first["delta"] == np.mean([1, -2, -10])
