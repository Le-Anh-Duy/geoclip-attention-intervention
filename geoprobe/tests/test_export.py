import csv
import json

from geoprobe.config import config_fingerprint
from geoprobe.evaluation.metrics import haversine_distance_km
from geoprobe.io import SCHEMA_VERSION, atomic_write_json, atomic_write_jsonl
from geoprobe.pipeline.export import ExportBlock


def test_export_recomputes_paired_metrics_from_synthetic_rows(tmp_path):
    config = {
        "pipeline": {"export": {"type": "ExportBlock"}},
        "runtime": {"output_dir": str(tmp_path), "seed": 42},
        "intervention": {
            "mask": {"score_threshold": 0.4, "max_proposals": 100, "crop_size": 224, "patch_grid": 16},
            "vision": {
                "expected_heads": 1,
                "layers": [{"index": 0, "name": "synthetic", "search_order": 0}],
            },
            "search": {"schedules": ["fixed", "sqrt", "linear"], "base_bias": 2.0},
        },
        "metrics": {
            "thresholds_km": [1, 25, 200, 750, 2500],
            "clip_improvement_km": 2500.0,
            "bootstrap_resamples": 200,
            "bootstrap_confidence": 0.95,
        },
    }
    fingerprint = config_fingerprint(config)
    split_id = "synthetic-split"
    candidate = {
        "layers": [2, 7, 12], "schedule": "linear", "base_bias": 2.0,
        "discovery_score": 1.0, "sample_improvements_km": [1.0],
    }
    atomic_write_json(tmp_path / "search.json", {
        "schema_version": SCHEMA_VERSION,
        "configuration_fingerprint": fingerprint,
        "split_fingerprint": split_id,
        "sample_ids": ["discovery.jpg"],
        "candidates": [candidate],
        "winner_index": 0,
        "winner": {key: candidate[key] for key in ("layers", "schedule", "base_bias")},
    })
    common = {
        "schema_version": SCHEMA_VERSION,
        "configuration_fingerprint": fingerprint,
        "split_fingerprint": split_id,
        "gallery_fingerprint": "gallery",
        "target": [0.0, 0.0],
        "baseline_score": 1.0,
        "intervened_score": 1.0,
        "selected_layers": [2, 7, 12],
        "schedule": "linear",
        "base_bias": 2.0,
        "biases": [2 / 3, 2 / 3, 2 / 3],
        "mask_patch_count": 1,
    }
    rows = [
        {
            **common, "sample_id": "discovery.jpg", "split": "discovery", "active_mask": True,
            "baseline_prediction": [0.0, 1.0], "intervened_prediction": [0.0, 0.5],
            "baseline_embedding": [1.0, 0.0], "intervened_embedding": [1.0, 0.0],
            "baseline_error_km": 999.0, "intervened_error_km": 999.0,
            "prediction_displacement_km": 999.0, "embedding_cosine": -1.0,
            "clipped_improvement_km": -999.0,
        },
        {
            **common, "sample_id": "holdout.jpg", "split": "holdout", "active_mask": False,
            "mask_patch_count": 0,
            "baseline_prediction": [0.0, 0.0], "intervened_prediction": [0.0, 0.0],
            "baseline_embedding": [0.0, 1.0], "intervened_embedding": [0.0, 1.0],
            "baseline_error_km": 999.0, "intervened_error_km": 999.0,
            "prediction_displacement_km": 999.0, "embedding_cosine": -1.0,
            "clipped_improvement_km": -999.0,
        },
    ]
    atomic_write_jsonl(tmp_path / "evaluation_all.jsonl", rows)

    ExportBlock(config).run()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    expected_mean = float(haversine_distance_km(0, 0, 0, 1)) / 2
    assert summary["sample_ids"] == ["discovery.jpg", "holdout.jpg"]
    assert summary["groups"]["overall"]["baseline"]["mean_error_km"] == expected_mean
    assert summary["groups"]["active_mask"]["count"] == 1
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert {row["method"] for row in metrics} == {"baseline", "intervened"}
    assert {row["subset"] for row in metrics} == {"overall", "discovery", "holdout", "active_mask"}
    with (tmp_path / "search.csv").open(newline="", encoding="utf-8") as handle:
        search_rows = list(csv.DictReader(handle))
    assert search_rows[0]["layers"] == "[2,7,12]"
    assert search_rows[0]["winner"] == "True"
