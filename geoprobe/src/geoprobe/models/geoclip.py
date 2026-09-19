from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from geoprobe.io import SCHEMA_VERSION, sha256_file
from geoprobe.registry import Registry

LOCALIZERS = Registry("localizer")


@dataclass(frozen=True)
class LocalizationResult:
    latitude: float
    longitude: float
    score: float
    embedding: tuple[float, ...]


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _enable_determinism() -> None:
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


@LOCALIZERS.register("GeoCLIPLocalizer")
class GeoCLIPLocalizer:
    def __init__(
        self,
        weights_path: str | Path,
        gallery_path: str | Path,
        gallery_cache: str | Path,
        processor_model: str = "openai/clip-vit-large-patch14",
        expected_gallery_count: int | None = 100000,
        device: str = "cpu",
        deterministic: bool = True,
        dtype: str = "float32",
    ) -> None:
        if dtype != "float32":
            raise ValueError("Confirmatory GeoCLIP runs require dtype='float32'")
        if processor_model != "openai/clip-vit-large-patch14":
            raise ValueError("GeoProbe requires the GeoCLIP ViT-L/14 processor model")
        self.weights_path = Path(weights_path).expanduser().resolve()
        self.gallery_path = Path(gallery_path).expanduser().resolve()
        self.gallery_cache = Path(gallery_cache).expanduser().resolve()
        required_weights = [self.weights_path / name for name in ("image_encoder_mlp_weights.pth", "location_encoder_weights.pth", "logit_scale_weights.pth")]
        missing = [str(path) for path in required_weights if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"GeoCLIP weight files are missing: {missing}")
        if not self.gallery_path.is_file():
            raise FileNotFoundError(f"GeoCLIP gallery coordinates do not exist: {self.gallery_path}")
        if deterministic:
            _enable_determinism()
        self.device = torch.device(device)
        try:
            from geoclip import GeoCLIP
        except ImportError as exc:
            raise ImportError("GeoCLIPLocalizer requires separately installed geoclip==1.2.0; run 'python -m pip install -e ./geo-clip'") from exc
        try:
            self.model = GeoCLIP(from_pretrained=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GeoCLIP ViT-L/14 ({processor_model}); ensure its Hugging Face files are locally available: {exc}") from exc
        self.model.image_encoder.mlp.load_state_dict(torch.load(required_weights[0], map_location="cpu", weights_only=True))
        self.model.location_encoder.load_state_dict(torch.load(required_weights[1], map_location="cpu", weights_only=True))
        self.model.logit_scale = torch.nn.Parameter(torch.load(required_weights[2], map_location="cpu", weights_only=True).float())
        self.model.float().eval().to(self.device)
        frame = pd.read_csv(self.gallery_path)
        if not {"LAT", "LON"}.issubset(frame.columns):
            raise ValueError(f"Gallery {self.gallery_path} must contain LAT and LON columns")
        coordinates = frame[["LAT", "LON"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if expected_gallery_count is not None and len(coordinates) != expected_gallery_count:
            raise ValueError(f"Expected {expected_gallery_count} gallery rows, found {len(coordinates)}")
        if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates[:, 0]) > 90) or np.any(np.abs(coordinates[:, 1]) > 180):
            raise ValueError("Gallery contains invalid coordinates")
        self.coordinates = coordinates
        weight_hashes = {path.name: sha256_file(path) for path in required_weights}
        self.model_fingerprint = hashlib.sha256(json.dumps({"processor": processor_model, "weights": weight_hashes}, sort_keys=True).encode()).hexdigest()
        self.gallery_fingerprint = sha256_file(self.gallery_path)
        metadata = {
            "schema_version": SCHEMA_VERSION, "gallery_sha256": self.gallery_fingerprint,
            "gallery_count": len(coordinates), "model_fingerprint": self.model_fingerprint,
            "embedding_dimension": 512, "dtype": "float32",
        }
        self.gallery_embeddings = self._load_or_compute_gallery(metadata)
        self.cache_metadata = metadata

    @property
    def vision_tower(self) -> Any:
        return self.model.image_encoder.CLIP

    def _load_or_compute_gallery(self, metadata: dict[str, Any]) -> torch.Tensor:
        if self.gallery_cache.is_file():
            payload = torch.load(self.gallery_cache, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("metadata") != metadata:
                raise ValueError(f"Stale GeoCLIP gallery cache metadata: {self.gallery_cache}")
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, torch.Tensor):
                raise ValueError("Gallery cache embeddings are not a tensor")
        else:
            parts: list[torch.Tensor] = []
            with torch.inference_mode():
                for start in range(0, len(self.coordinates), 4096):
                    locations = torch.from_numpy(self.coordinates[start : start + 4096]).to(self.device, dtype=torch.float32)
                    parts.append(F.normalize(self.model.location_encoder(locations).float(), dim=-1).cpu())
            embeddings = torch.cat(parts, dim=0)
            _atomic_torch_save(self.gallery_cache, {"metadata": metadata, "embeddings": embeddings})
        if embeddings.ndim != 2 or embeddings.shape != (len(self.coordinates), metadata["embedding_dimension"]):
            raise ValueError(f"Gallery cache shape mismatch: expected {(len(self.coordinates), metadata['embedding_dimension'])}, found {tuple(embeddings.shape)}")
        if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
            raise ValueError("Gallery embeddings must be finite FP32")
        norms = torch.linalg.vector_norm(embeddings, dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise ValueError("Gallery cache embeddings are not normalized")
        return embeddings.to(self.device, dtype=torch.float32)

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        pixels = self.model.image_encoder.preprocess_image(image).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            embedding = F.normalize(self.model.image_encoder(pixels).float(), dim=-1)
        if embedding.shape != (1, self.gallery_embeddings.shape[1]) or not torch.isfinite(embedding).all():
            raise ValueError(f"Invalid GeoCLIP image embedding shape/value: {tuple(embedding.shape)}")
        return embedding

    def retrieve(self, embedding: torch.Tensor) -> LocalizationResult:
        if embedding.shape != (1, self.gallery_embeddings.shape[1]):
            raise ValueError("Image embedding dimension does not match the gallery")
        with torch.inference_mode():
            similarities = self.model.logit_scale.exp().float() * (embedding.float() @ self.gallery_embeddings.T)
            index = int(torch.argmax(similarities, dim=1).item())
            score = float(similarities[0, index].item())
        coordinate = self.coordinates[index]
        return LocalizationResult(float(coordinate[0]), float(coordinate[1]), score, tuple(float(x) for x in embedding[0].cpu().tolist()))

    def localize(self, image: Image.Image) -> LocalizationResult:
        return self.retrieve(self.encode_image(image))
