from __future__ import annotations

import csv
import io
import json
from typing import Any, Mapping, Sequence

import numpy as np

from geoprobe.evaluation.metrics import (
    candidate_rank_correlation,
    clipped_improvement,
    embedding_cosine,
    haversine_distance_km,
    paired_metrics_with_intervals,
    selected_vs_best_single_advantage,
    summarize_rows,
)
from geoprobe.io import SCHEMA_VERSION, atomic_write_json, atomic_write_text, read_jsonl

from .context import PipelineContext
from .registry import PIPELINE_BLOCKS


@PIPELINE_BLOCKS.register("ExportBlock")
class ExportBlock:
    """Stage 4: recompute every aggregate from serialized paired rows."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.context = PipelineContext.create(config)

    def run(self) -> None:
        context = self.context
        search_artifact = self._read_search()
        raw_rows = self._read_evaluation_rows()
        split_id = search_artifact.get("split_fingerprint")
        if any(
            row.get("configuration_fingerprint") != context.configuration_fingerprint
            or row.get("split_fingerprint") != split_id
            for row in raw_rows
        ):
            raise ValueError("Evaluation rows belong to a different configuration or split")
        ids = [row.get("sample_id") for row in raw_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation rows contain duplicate sample IDs")
        clip_km = float(context.config["metrics"]["clip_improvement_km"])
        rows = [self._recompute(row, clip_km) for row in raw_rows]
        groups = self._groups(rows)
        summary = self._summary(search_artifact, groups, ids, split_id)
        settings = {
            "thresholds_km": context.config["metrics"]["thresholds_km"],
            "seed": context.config["runtime"]["seed"],
            "resamples": context.config["metrics"]["bootstrap_resamples"],
            "confidence": context.config["metrics"]["bootstrap_confidence"],
        }
        atomic_write_json(context.output_dir / "summary.json", summary)
        atomic_write_text(
            context.output_dir / "metrics.csv",
            self._metrics_csv(groups, settings, context.configuration_fingerprint, ids),
        )
        self._write_search_csv(search_artifact, rows, summary, ids)
        context.write_resolved_config(ids)

    def _read_search(self) -> dict[str, Any]:
        path = self.context.output_dir / "search.json"
        if not path.is_file():
            raise FileNotFoundError(f"Search artifact does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("configuration_fingerprint") != self.context.configuration_fingerprint:
            raise ValueError("Search artifact configuration does not match export configuration")
        return artifact

    def _read_evaluation_rows(self) -> list[dict[str, Any]]:
        path = self.context.output_dir / "evaluation_all.jsonl"
        if not path.is_file():
            path = self.context.output_dir / "evaluation_holdout.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"No evaluation_all.jsonl or evaluation_holdout.jsonl in {self.context.output_dir}"
            )
        return read_jsonl(path)

    @staticmethod
    def _recompute(row: Mapping[str, Any], clip_km: float) -> dict[str, Any]:
        result = dict(row)
        target = row["target"]
        baseline = row["baseline_prediction"]
        intervened = row["intervened_prediction"]
        result["baseline_error_km"] = float(
            haversine_distance_km(target[0], target[1], baseline[0], baseline[1])
        )
        result["intervened_error_km"] = float(
            haversine_distance_km(target[0], target[1], intervened[0], intervened[1])
        )
        result["prediction_displacement_km"] = float(
            haversine_distance_km(baseline[0], baseline[1], intervened[0], intervened[1])
        )
        result["embedding_cosine"] = float(
            embedding_cosine(row["baseline_embedding"], row["intervened_embedding"])
        )
        result["clipped_improvement_km"] = float(
            clipped_improvement(result["baseline_error_km"], result["intervened_error_km"], clip_km)
        )
        return result

    @staticmethod
    def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups = {"overall": rows}
        for split_name in ("discovery", "holdout"):
            selected = [row for row in rows if row["split"] == split_name]
            if selected:
                groups[split_name] = selected
        active = [row for row in rows if row["active_mask"]]
        if active:
            groups["active_mask"] = active
        return groups

    def _summary(
        self,
        search_artifact: Mapping[str, Any],
        groups: Mapping[str, Sequence[Mapping[str, Any]]],
        sample_ids: Sequence[str],
        split_id: str,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "configuration_fingerprint": self.context.configuration_fingerprint,
            "split_fingerprint": split_id,
            "sample_ids": list(sample_ids),
            "groups": {name: summarize_rows(rows) for name, rows in groups.items()},
            "winner": search_artifact["winner"],
        }
        candidates = search_artifact["candidates"]
        winner_index = int(search_artifact["winner_index"])
        singles = list(range(min(5, len(candidates))))
        required = singles + [winner_index]
        if candidates and singles and all("sample_improvements_km" in candidates[index] for index in required):
            best_single = max(singles, key=lambda index: candidates[index]["discovery_score"])
            summary["selected_vs_best_single_advantage_km"] = selected_vs_best_single_advantage(
                candidates[winner_index]["sample_improvements_km"],
                candidates[best_single]["sample_improvements_km"],
            )
        return summary

    @staticmethod
    def _metrics_csv(
        groups: Mapping[str, Sequence[Mapping[str, Any]]],
        settings: Mapping[str, Any],
        fingerprint: str,
        sample_ids: Sequence[str],
    ) -> str:
        stream = io.StringIO(newline="")
        fields = [
            "schema_version", "configuration_fingerprint", "sample_ids", "subset", "method",
            "metric", "estimate", "ci_low", "ci_high", "paired_delta",
            "paired_delta_ci_low", "paired_delta_ci_high",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        encoded_ids = json.dumps(list(sample_ids), separators=(",", ":"))
        for name, rows in groups.items():
            baseline = np.asarray([row["baseline_error_km"] for row in rows])
            intervened = np.asarray([row["intervened_error_km"] for row in rows])
            intervals = paired_metrics_with_intervals(
                baseline, intervened, thresholds_km=settings["thresholds_km"],
                seed=int(settings["seed"]), resamples=int(settings["resamples"]),
                confidence=float(settings["confidence"]),
            )
            for metric, values in intervals.items():
                for method in ("baseline", "intervened"):
                    ci = values[f"{method}_ci"]
                    writer.writerow({
                        "schema_version": SCHEMA_VERSION,
                        "configuration_fingerprint": fingerprint,
                        "sample_ids": encoded_ids,
                        "subset": name, "method": method, "metric": metric,
                        "estimate": values[method], "ci_low": ci[0], "ci_high": ci[1],
                        "paired_delta": values["delta"],
                        "paired_delta_ci_low": values["delta_ci"][0],
                        "paired_delta_ci_high": values["delta_ci"][1],
                    })
        return stream.getvalue()

    def _write_search_csv(
        self,
        search_artifact: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        summary: dict[str, Any],
        sample_ids: Sequence[str],
    ) -> None:
        candidates = search_artifact["candidates"]
        winner_index = int(search_artifact["winner_index"])
        holdout_rows = [row for row in rows if row["split"] == "holdout"]
        winner_holdout = (
            float(np.mean([row["clipped_improvement_km"] for row in holdout_rows]))
            if holdout_rows else None
        )
        holdout_scores = [candidate.get("holdout_score") for candidate in candidates]
        if winner_holdout is not None:
            holdout_scores[winner_index] = winner_holdout
        finite = [(index, float(score)) for index, score in enumerate(holdout_scores) if score is not None]
        ranks = {
            index: rank + 1
            for rank, (index, _) in enumerate(sorted(finite, key=lambda item: (-item[1], item[0])))
        }
        if len(finite) >= 2:
            indices = [index for index, _ in finite]
            summary["candidate_rank_correlation"] = candidate_rank_correlation(
                [candidates[index]["discovery_score"] for index in indices],
                [holdout_scores[index] for index in indices],
            )
            atomic_write_json(self.context.output_dir / "summary.json", summary)

        stream = io.StringIO(newline="")
        fields = [
            "schema_version", "configuration_fingerprint", "sample_ids", "order", "layers",
            "schedule", "base_bias", "discovery_score", "holdout_score", "holdout_rank", "winner",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        encoded_ids = json.dumps(list(sample_ids), separators=(",", ":"))
        for index, candidate in enumerate(candidates):
            writer.writerow({
                "schema_version": SCHEMA_VERSION,
                "configuration_fingerprint": self.context.configuration_fingerprint,
                "sample_ids": encoded_ids,
                "order": index,
                "layers": json.dumps(candidate["layers"], separators=(",", ":")),
                "schedule": candidate["schedule"],
                "base_bias": candidate["base_bias"],
                "discovery_score": candidate["discovery_score"],
                "holdout_score": holdout_scores[index],
                "holdout_rank": ranks.get(index),
                "winner": index == winner_index,
            })
        atomic_write_text(self.context.output_dir / "search.csv", stream.getvalue())
