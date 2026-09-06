"""Prompt-free object proposals from an exported WeDetect-Base-Uni model.

The official WeDetect repository provides a self-contained ONNX export for
WeDetect-Uni.  This adapter intentionally consumes that visual-only export:
the text tower and the large vocabulary used by WeDetect-Anything are not
needed for attention intervention.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_IMAGE_SIZE = 640
DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_TOP_K = 100


class WeDetectUnavailable(RuntimeError):
    """Raised when the optional WeDetect runtime has not been configured."""


@dataclass(frozen=True)
class Proposal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def normalized_region(self, width: int, height: int) -> dict[str, float]:
        return {
            "x": self.x1 / width,
            "y": self.y1 / height,
            "w": (self.x2 - self.x1) / width,
            "h": (self.y2 - self.y1) / height,
        }


def default_model_path() -> Path:
    configured = os.getenv("WEDETECT_UNI_ONNX")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "wedetect" / "wedetect_anything_base.onnx"


def letterbox(image: Image.Image, size: int = DEFAULT_IMAGE_SIZE) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Reproduce WeDetect's square RGB letterbox preprocessing."""
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    ratio = min(size / orig_h, size / orig_w)
    resized_w = int(round(orig_w * ratio))
    resized_h = int(round(orig_h * ratio))
    resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

    dw = (size - resized_w) / 2
    dh = (size - resized_h) / 2
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_h, left : left + resized_w] = np.asarray(resized)
    return canvas, ratio, (dw, dh)


def rescale_boxes(
    boxes: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Map xyxy boxes from letterbox space back to the uploaded image."""
    result = boxes.astype(np.float32, copy=True)
    result[:, [0, 2]] = (result[:, [0, 2]] - pad[0]) / ratio
    result[:, [1, 3]] = (result[:, [1, 3]] - pad[1]) / ratio
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, orig_w)
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, orig_h)
    return result


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, top_k: int) -> np.ndarray:
    """Small NumPy NMS implementation; avoids requiring torchvision C++ ops."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    kept: list[int] = []

    while order.size and len(kept) < top_k:
        idx = int(order[0])
        kept.append(idx)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[idx] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= iou_threshold]

    return np.asarray(kept, dtype=np.int64)


class WeDetectUniONNX:
    """Lazy ONNXRuntime wrapper around the official visual-only export."""

    def __init__(self, model_path: str | os.PathLike | None = None, image_size: int = DEFAULT_IMAGE_SIZE):
        self.model_path = Path(model_path).expanduser() if model_path else default_model_path()
        self.image_size = image_size
        self._session = None
        self._session_lock = threading.Lock()
        self.providers: list[str] = []

    @property
    def configured(self) -> bool:
        return self.model_path.is_file()

    def _get_session(self):
        if self._session is not None:
            return self._session
        if not self.configured:
            raise WeDetectUnavailable(
                f"WeDetect-Uni ONNX not found at {self.model_path}. "
                "Run `python scripts/setup_wedetect_uni.py` from test-intervention/backend "
                "or set WEDETECT_UNI_ONNX to an exported model."
            )
        with self._session_lock:
            if self._session is None:
                try:
                    import onnxruntime as ort
                except ImportError as exc:
                    raise WeDetectUnavailable(
                        "onnxruntime is not installed; install backend requirements first"
                    ) from exc

                available = ort.get_available_providers()
                preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
                self._session = ort.InferenceSession(str(self.model_path), providers=preferred)
                self.providers = list(self._session.get_providers())
        return self._session

    def info(self) -> dict:
        return {
            "configured": self.configured,
            "model_path": str(self.model_path),
            "loaded": self._session is not None,
            "providers": self.providers,
            "image_size": self.image_size,
        }

    def generate(
        self,
        image: Image.Image,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[Proposal]:
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 1 <= top_k <= 1000:
            raise ValueError("top_k must be in [1, 1000]")

        session = self._get_session()
        orig_w, orig_h = image.size
        padded, ratio, pad = letterbox(image, self.image_size)
        tensor = np.ascontiguousarray((padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])

        # Proposal generation only needs objectness and boxes. Avoid copying
        # the large per-box embedding output from ONNXRuntime on every image.
        scores, boxes = session.run(
            ["scores", "bboxes"], {"input_image": tensor}
        )
        scores = np.asarray(scores[0])
        boxes = np.asarray(boxes[0], dtype=np.float32)
        if scores.ndim != 2 or boxes.ndim != 2 or boxes.shape[1] < 4 or len(scores) != len(boxes):
            raise RuntimeError(
                f"Unexpected WeDetect output shapes: scores={scores.shape}, bboxes={boxes.shape}"
            )

        # The 256 internal prompts are all generic objectness prompts.  The
        # official prompt-free inference takes the maximum score per box.
        objectness = scores.max(axis=1)
        valid = np.flatnonzero(objectness > score_threshold)
        if not len(valid):
            return []

        kept_local = _nms(boxes[valid, :4], objectness[valid], iou_threshold, top_k)
        kept = valid[kept_local]
        restored = rescale_boxes(boxes[kept, :4], ratio, pad, orig_w, orig_h)

        proposals = []
        for box, score in zip(restored, objectness[kept]):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            proposals.append(Proposal(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score)))
        return proposals


_proposer: WeDetectUniONNX | None = None
_proposer_lock = threading.Lock()


def get_proposer() -> WeDetectUniONNX:
    global _proposer
    if _proposer is None:
        with _proposer_lock:
            if _proposer is None:
                _proposer = WeDetectUniONNX()
    return _proposer
