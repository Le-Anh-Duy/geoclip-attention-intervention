from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from geoprobe.io import SCHEMA_VERSION, atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from geoprobe.registry import Registry

PROPOSERS = Registry("proposer")


@dataclass(frozen=True)
class Proposal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"xyxy": [self.x1, self.y1, self.x2, self.y2], "score": self.score}


def letterbox(image: Image.Image, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    if size <= 0:
        raise ValueError("Letterbox size must be positive")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = min(size / width, size / height)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = np.asarray(rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR), dtype=np.uint8)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, (left, top)


def rescale_boxes(boxes: np.ndarray, scale: float, padding: tuple[int, int], width: int, height: int) -> np.ndarray:
    result = np.asarray(boxes, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"Boxes must have shape (N, 4), got {result.shape}")
    left, top = padding
    result[:, [0, 2]] = (result[:, [0, 2]] - left) / scale
    result[:, [1, 3]] = (result[:, [1, 3]] - top) / scale
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0.0, float(width))
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0.0, float(height))
    return result


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),):
        raise ValueError("NMS expects boxes (N,4) and scores (N,)")
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[current], x1[rest]), np.maximum(y1[current], y1[rest])
        xx2, yy2 = np.minimum(x2[current], x2[rest]), np.minimum(y2[current], y2[rest])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[overlap <= iou_threshold]
    return keep


@PROPOSERS.register("WeDetectUniONNX")
class WeDetectUniONNX:
    def __init__(self, model_path: str | Path, input_size: int = 640, score_threshold: float = 0.15, nms_threshold: float = 0.7) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"WeDetect ONNX model does not exist: {self.model_path}")
        if input_size <= 0 or not 0.0 <= score_threshold <= 1.0 or not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("Invalid WeDetect input size or thresholds")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("WeDetect requires onnxruntime; install the geoprobe package dependencies") from exc
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"] if not ort.get_available_providers() else ort.get_available_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4:
            raise ValueError(f"WeDetect model {self.model_path} must expose one rank-4 input")
        if len(outputs) != 2 or any(len(output.shape) not in (2, 3) for output in outputs):
            raise ValueError(f"WeDetect model {self.model_path} must expose two rank-2/3 outputs")
        self.input_name = inputs[0].name
        names = [output.name.lower() for output in outputs]
        box_indices = [i for i, name in enumerate(names) if "box" in name]
        score_indices = [i for i, name in enumerate(names) if "score" in name or "logit" in name]
        if len(box_indices) == len(score_indices) == 1 and box_indices[0] != score_indices[0]:
            self.box_index, self.score_index = box_indices[0], score_indices[0]
        else:
            self.box_index = next((i for i, output in enumerate(outputs) if output.shape[-1] == 4), -1)
            self.score_index = 1 - self.box_index if self.box_index in (0, 1) else -1
        if self.box_index < 0:
            raise ValueError(f"Cannot identify WeDetect box output in model {self.model_path}")

    def __call__(self, image: Image.Image, sample_id: str = "<unknown>") -> list[Proposal]:
        width, height = image.size
        tensor, scale, padding = letterbox(image, self.input_size)
        try:
            outputs = self.session.run(None, {self.input_name: tensor})
        except Exception as exc:
            raise RuntimeError(f"WeDetect inference failed for sample {sample_id!r} with {self.model_path}: {exc}") from exc
        raw_boxes, raw_scores = np.asarray(outputs[self.box_index]), np.asarray(outputs[self.score_index])
        context = f"sample {sample_id!r}, model {self.model_path}"
        if raw_boxes.ndim != 3 or raw_boxes.shape[0] != 1 or raw_boxes.shape[-1] != 4:
            raise ValueError(f"Malformed WeDetect boxes for {context}: shape {raw_boxes.shape}")
        boxes = np.asarray(raw_boxes[0], dtype=np.float64)
        if raw_scores.ndim == 3 and raw_scores.shape[0] == 1:
            scores = np.max(raw_scores[0], axis=-1)
        elif raw_scores.ndim == 2 and raw_scores.shape[0] == 1:
            scores = raw_scores[0]
        else:
            raise ValueError(f"Malformed WeDetect scores for {context}: shape {raw_scores.shape}")
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(boxes),):
            raise ValueError(f"WeDetect box/score row mismatch for {context}: {len(boxes)} vs {scores.shape}")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f"Non-finite WeDetect output for {context}")
        if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
            raise ValueError(f"Malformed WeDetect xyxy box ordering for {context}")
        boxes = rescale_boxes(boxes, scale, padding, width, height)
        valid = (scores >= self.score_threshold) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores = boxes[valid], scores[valid]
        kept = nms(boxes, scores, self.nms_threshold)
        proposals = [Proposal(*map(float, boxes[index]), float(scores[index])) for index in kept]
        proposals.sort(key=lambda proposal: proposal.score, reverse=True)
        return proposals

    def metadata(self, dataset_count: int) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "dataset_count": int(dataset_count),
            "model_path": str(self.model_path), "model_sha256": sha256_file(self.model_path),
            "input_size": self.input_size, "score_threshold": self.score_threshold,
            "nms_threshold": self.nms_threshold,
        }


def cache_proposals(
    proposer: WeDetectUniONNX, samples: Sequence[Any], rows_path: Path, metadata_path: Path, *, metadata_extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    ids = [str(sample.sample_id) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot cache proposals for duplicate sample IDs")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        with Image.open(sample.image_path) as image:
            width, height = image.size
            proposals = proposer(image, str(sample.sample_id))
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "configuration_fingerprint": (metadata_extra or {}).get("configuration_fingerprint"),
            "split_fingerprint": (metadata_extra or {}).get("split_fingerprint"),
            "sample_id": str(sample.sample_id), "image_path": str(sample.image_path),
            "width": width, "height": height,
            "proposals": [item.to_dict() for item in proposals],
        })
    if {row["sample_id"] for row in rows} != set(ids) or len(rows) != len(ids):
        raise AssertionError("Proposal cache omitted or duplicated sample IDs")
    metadata = proposer.metadata(len(samples))
    metadata.update(dict(metadata_extra or {}))
    metadata["sample_ids"] = ids
    atomic_write_jsonl(rows_path, rows)
    atomic_write_json(metadata_path, metadata)
    return metadata


def read_proposal_cache(rows_path: Path, metadata_path: Path, expected_metadata: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Proposal cache metadata mismatch for {key}: expected {expected!r}, found {metadata.get(key)!r}")
    rows = read_jsonl(rows_path)
    ids = [row.get("sample_id") for row in rows]
    if len(ids) != len(set(ids)) or len(rows) != metadata.get("dataset_count"):
        raise ValueError("Proposal cache contains duplicate or missing sample IDs")
    if ids != metadata.get("sample_ids"):
        raise ValueError("Proposal cache row IDs do not match metadata sample IDs/order")
    return metadata, rows
