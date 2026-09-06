"""Evaluate GeoCLIP using the union of all retained WeDetect-Uni proposals.

This script is intended for ``colab exec -f``. Expected remote assets:

* /content/geoclip_source.zip
* /content/intervention.py
* /content/wedetect_proposals.py

WeDetect-Uni's official checkpoint is downloaded from Hugging Face and exported
to ONNX inside the Colab VM. The large model never needs to be uploaded from
the client machine.

It checkpoints per-image results so a second execution can resume after a
runtime interruption. Set MAX_IMAGES in the remote kernel for a smoke test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile
from transformers import AutoProcessor


DATASET_HANDLE = "lbgan2000/imgps3k-yfcc4k-cleaned"
SCORE_THRESHOLD = float(os.getenv("WEDETECT_SCORE_THRESHOLD", "0.4"))
IOU_THRESHOLD = float(os.getenv("WEDETECT_IOU_THRESHOLD", "0.7"))
PROPOSAL_TOP_K = int(os.getenv("WEDETECT_PROPOSAL_TOP_K", "1000"))
LAYER_SPEC = os.getenv("INTERVENTION_LAYERS", "all")
ATTENTION_A = float(os.getenv("INTERVENTION_A", "2.0"))
ATTENTION_B = float(os.getenv("INTERVENTION_B", "-2.0"))
BASELINE_BATCH_SIZE = int(os.getenv("BASELINE_BATCH_SIZE", "64"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "25"))

CONTENT = Path("/content")
SOURCE_ZIP = CONTENT / "geoclip_source.zip"
SOURCE_DIR = CONTENT / "geoclip_source"
WEDETECT_DIR = CONTENT / "wedetect"
ONNX_PATH = WEDETECT_DIR / "wedetect_anything_base.onnx"
WEDETECT_SOURCE_DIR = CONTENT / "wedetect_source"
WEDETECT_REPOSITORY = "https://github.com/WeChatCV/WeDetect.git"
WEDETECT_REVISION = "dd302dba0069ace1b05816bafbc3fa1dbd6aa68c"
WEDETECT_HF_REPOSITORY = "fushh7/WeDetect"
WEDETECT_HF_REVISION = "125b98f6807eb3459b57a43497d220ae4096f5c7"
WEDETECT_CHECKPOINT = "wedetect_base_uni.pth"
OUTPUT_DIR = CONTENT / "img2gps3k_wedetect_union_all_layers_a2_bneg2_eval"
PER_IMAGE_PATH = OUTPUT_DIR / "per_image.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
THRESHOLDS_KM = (1, 25, 200, 750, 2500)

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_source() -> None:
    package_marker = SOURCE_DIR / "geoclip" / "__init__.py"
    if not package_marker.is_file():
        if not SOURCE_ZIP.is_file():
            raise FileNotFoundError(f"Missing uploaded source archive: {SOURCE_ZIP}")
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(SOURCE_ZIP) as archive:
            # PowerShell Compress-Archive stores Windows backslashes in member
            # names. Linux's zipfile otherwise extracts those as literal
            # characters instead of directories.
            for member in archive.infolist():
                relative = Path(*member.filename.replace("\\", "/").split("/"))
                target = SOURCE_DIR / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
    for local_module in ("intervention.py", "wedetect_proposals.py"):
        source = CONTENT / local_module
        target = SOURCE_DIR / local_module
        if source.is_file() and not target.is_file():
            target.write_bytes(source.read_bytes())
    sys.path.insert(0, str(SOURCE_DIR))


def ensure_wedetect_model() -> None:
    """Download the official HF checkpoint and export ONNX on the Colab VM."""
    data_path = ONNX_PATH.with_suffix(ONNX_PATH.suffix + ".data")
    if ONNX_PATH.is_file() and data_path.is_file():
        print(f"WEDETECT_MODEL cached={ONNX_PATH}")
        return

    WEDETECT_DIR.mkdir(parents=True, exist_ok=True)
    export_script = WEDETECT_SOURCE_DIR / "wedetect_anything" / "export_onnx.py"
    if not export_script.is_file():
        if WEDETECT_SOURCE_DIR.exists():
            raise RuntimeError(
                f"Incomplete WeDetect checkout at {WEDETECT_SOURCE_DIR}; "
                "remove it before retrying"
            )
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                WEDETECT_REPOSITORY,
                str(WEDETECT_SOURCE_DIR),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(WEDETECT_SOURCE_DIR),
                "sparse-checkout",
                "set",
                "wedetect_anything",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(WEDETECT_SOURCE_DIR),
                "fetch",
                "--depth",
                "1",
                "origin",
                WEDETECT_REVISION,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(WEDETECT_SOURCE_DIR),
                "checkout",
                "--detach",
                "FETCH_HEAD",
            ],
            check=True,
        )

    checkpoint = hf_hub_download(
        repo_id=WEDETECT_HF_REPOSITORY,
        filename=WEDETECT_CHECKPOINT,
        revision=WEDETECT_HF_REVISION,
        cache_dir=str(CONTENT / "hf_cache"),
    )
    export_start = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            str(export_script),
            "--variant",
            "base",
            "--checkpoint",
            checkpoint,
            "--img-size",
            "640",
            "--device",
            "cpu",
            "--output-dir",
            str(WEDETECT_DIR),
        ],
        cwd=WEDETECT_SOURCE_DIR,
        check=True,
    )
    if not ONNX_PATH.is_file() or not data_path.is_file():
        raise RuntimeError(
            f"WeDetect export did not create both {ONNX_PATH.name} and {data_path.name}"
        )
    print(
        "WEDETECT_MODEL",
        json.dumps(
            {
                "checkpoint_source": f"{WEDETECT_HF_REPOSITORY}/{WEDETECT_CHECKPOINT}",
                "onnx": str(ONNX_PATH),
                "onnx_bytes": ONNX_PATH.stat().st_size,
                "external_data_bytes": data_path.stat().st_size,
                "export_seconds": time.perf_counter() - export_start,
            },
            sort_keys=True,
        ),
    )


def haversine_km(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    lat1, lon1 = np.radians(target[..., 0]), np.radians(target[..., 1])
    lat2, lon2 = np.radians(predicted[..., 0]), np.radians(predicted[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0, 1 - value)))


def metric_summary(distances: np.ndarray) -> dict:
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "mean_distance_km": float(np.mean(distances)),
        "median_distance_km": float(np.median(distances)),
        "accuracy": {
            f"acc_{threshold}_km": float(np.mean(distances <= threshold))
            for threshold in THRESHOLDS_KM
        },
    }


def enable_batched_masks(state) -> None:
    """Allow one independent proposal mask per image in an inference batch."""

    def key_bias_for_layer(self, layer_idx: int, num_positions: int):
        a, b = self.layer_ab.get(layer_idx, (0.0, 0.0))
        if (a == 0.0 and b == 0.0) or self.in_region_mask is None:
            return None
        mask = self.in_region_mask
        if mask.ndim == 1:
            bias = torch.full((num_positions,), b, dtype=torch.float32, device=mask.device)
            bias[0] = 0.0
            bias[1:].masked_fill_(mask, a)
            return bias
        if mask.ndim != 2 or mask.shape[1] != num_positions - 1:
            raise ValueError(f"Unexpected batched mask shape {tuple(mask.shape)}")
        bias = torch.full(
            (mask.shape[0], num_positions), b, dtype=torch.float32, device=mask.device
        )
        bias[:, 0] = 0.0
        bias[:, 1:].masked_fill_(mask, a)
        return bias[:, None, None, :]

    state.key_bias_for_layer = types.MethodType(key_bias_for_layer, state)


ensure_source()
ensure_wedetect_model()
from geoclip import GeoCLIP  # noqa: E402
import intervention  # noqa: E402
import wedetect_proposals  # noqa: E402


if not torch.cuda.is_available():
    raise RuntimeError("This full evaluation requires a Colab GPU runtime")
device = torch.device("cuda")
print("CONFIG", json.dumps({
    "dataset": DATASET_HANDLE,
    "score_threshold": SCORE_THRESHOLD,
    "iou_threshold": IOU_THRESHOLD,
    "proposal_top_k": PROPOSAL_TOP_K,
    "layers": LAYER_SPEC,
    "a": ATTENTION_A,
    "b": ATTENTION_B,
    "proposal_mode": "union_all",
    "max_images": MAX_IMAGES,
}, sort_keys=True))
print("GPU", torch.cuda.get_device_name(0))

dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
csv_path = dataset_path / "test_set" / "im2gps3k_places365.csv"
image_dir = dataset_path / "test_set" / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame["name"].map(lambda name: str(image_dir / name))
missing = frame.loc[~frame["image_path"].map(os.path.isfile), "name"].tolist()
if missing:
    raise RuntimeError(f"Missing labeled images: {missing[:10]}")
if MAX_IMAGES > 0:
    frame = frame.iloc[:MAX_IMAGES].copy()
frame = frame.reset_index(drop=True)
targets = frame[["LAT", "LON"]].to_numpy(dtype=np.float64)
print(f"DATASET path={dataset_path} labeled_images={len(frame)}")

setup_start = time.perf_counter()
model = GeoCLIP().to(device).eval()
model.image_encoder.image_processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14", use_fast=False
)
state = intervention.patch_vision_tower(model.image_encoder.CLIP)
enable_batched_masks(state)
num_attention_layers = intervention.num_layers(model.image_encoder.CLIP)
if LAYER_SPEC.strip().lower() == "all":
    active_layers = list(range(num_attention_layers))
else:
    active_layers = [int(value.strip()) for value in LAYER_SPEC.split(",") if value.strip()]
    invalid_layers = [layer for layer in active_layers if not 0 <= layer < num_attention_layers]
    if invalid_layers:
        raise ValueError(f"Invalid attention layers: {invalid_layers}")
with torch.inference_mode():
    gallery_features = F.normalize(
        model.location_encoder(model.gps_gallery.to(device)), dim=1
    )
    logit_scale = model.logit_scale.exp()
gallery_cpu = model.gps_gallery.cpu().numpy()
print(f"GEOCLIP_READY seconds={time.perf_counter() - setup_start:.2f}")

proposer = wedetect_proposals.WeDetectUniONNX(model_path=ONNX_PATH)
# Force session creation now so provider failures surface before baseline work.
proposer._get_session()
print("WEDETECT_READY providers=", proposer.providers)


def predict_pixels(pixel_values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        features = F.normalize(model.image_encoder(pixel_values.to(device)), dim=1)
        logits = logit_scale * (features @ gallery_features.T)
        probabilities = logits.softmax(dim=1)
        confidence, indices = probabilities.max(dim=1)
    return indices.cpu().numpy(), confidence.cpu().numpy()


baseline_start = time.perf_counter()
baseline_indices: list[int] = []
baseline_confidence: list[float] = []
state.layer_ab = {}
state.in_region_mask = None
for start in range(0, len(frame), BASELINE_BATCH_SIZE):
    batch = frame.iloc[start : start + BASELINE_BATCH_SIZE]
    images = []
    for image_path in batch["image_path"]:
        with Image.open(image_path) as source:
            images.append(source.convert("RGB"))
    pixels = model.image_encoder.image_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]
    indices, confidence = predict_pixels(pixels)
    baseline_indices.extend(indices.tolist())
    baseline_confidence.extend(confidence.tolist())
    done = min(start + BASELINE_BATCH_SIZE, len(frame))
    if done == len(frame) or done % (BASELINE_BATCH_SIZE * 5) == 0:
        print(f"BASELINE {done}/{len(frame)}")

baseline_indices_array = np.asarray(baseline_indices, dtype=np.int64)
baseline_gps = gallery_cpu[baseline_indices_array]
baseline_distances = haversine_km(baseline_gps, targets)
baseline_seconds = time.perf_counter() - baseline_start
print(f"BASELINE_DONE seconds={baseline_seconds:.2f}")

existing_records: dict[str, dict] = {}
if PER_IMAGE_PATH.is_file():
    previous = pd.read_csv(PER_IMAGE_PATH)
    existing_records = {str(row["name"]): row.to_dict() for _, row in previous.iterrows()}
    print(f"RESUME existing={len(existing_records)}")

records: dict[str, dict] = dict(existing_records)
proposal_start = time.perf_counter()
processed_this_run = 0


def save_checkpoint() -> None:
    ordered = [records[name] for name in frame["name"] if name in records]
    pd.DataFrame(ordered).to_csv(PER_IMAGE_PATH, index=False)
    print(f"CHECKPOINT rows={len(ordered)} path={PER_IMAGE_PATH}")


for row_index, row in frame.iterrows():
    name = str(row["name"])
    if name in records:
        continue
    base_gps = baseline_gps[row_index]
    base_distance = float(baseline_distances[row_index])
    base_confidence = float(baseline_confidence[row_index])
    target = targets[row_index]
    record = {
        "name": name,
        "target_lat": float(target[0]),
        "target_lon": float(target[1]),
        "baseline_lat": float(base_gps[0]),
        "baseline_lon": float(base_gps[1]),
        "baseline_confidence": base_confidence,
        "baseline_distance_km": base_distance,
        "proposal_count": 0,
        "union_patch_count": 0,
        "union_coverage": 0.0,
        "union_lat": float(base_gps[0]),
        "union_lon": float(base_gps[1]),
        "union_confidence": base_confidence,
        "union_distance_km": base_distance,
        "union_delta_km": 0.0,
        "error": "",
    }
    try:
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        raw = proposer.generate(
            image,
            score_threshold=SCORE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            top_k=PROPOSAL_TOP_K,
        )
        width, height = image.size
        proposals = []
        masks = []
        signatures = set()
        for proposal in raw:
            region = proposal.normalized_region(width, height)
            mask = intervention.build_in_region_mask([region], width, height, 16)
            if not mask.any():
                continue
            signature = mask.numpy().tobytes()
            if signature in signatures:
                continue
            signatures.add(signature)
            proposals.append(proposal)
            masks.append(mask)

        record["proposal_count"] = len(masks)
        if masks:
            union_mask = torch.stack(masks).any(dim=0)
            union_patch_count = int(union_mask.sum().item())
            state.layer_ab = {layer: (ATTENTION_A, ATTENTION_B) for layer in active_layers}
            state.in_region_mask = union_mask.to(device)
            pixels = model.image_encoder.image_processor(
                images=[image], return_tensors="pt"
            )["pixel_values"]
            indices, confidence = predict_pixels(pixels)
            state.layer_ab = {}
            state.in_region_mask = None

            union_gps = gallery_cpu[int(indices[0])]
            union_distance = float(haversine_km(union_gps, target))

            record.update({
                "union_patch_count": union_patch_count,
                "union_coverage": union_patch_count / 256.0,
                "union_lat": float(union_gps[0]),
                "union_lon": float(union_gps[1]),
                "union_confidence": float(confidence[0]),
                "union_distance_km": union_distance,
                "union_delta_km": base_distance - union_distance,
            })
    except Exception as exc:  # Continue the benchmark and report per-image failures.
        state.layer_ab = {}
        state.in_region_mask = None
        record["error"] = repr(exc)
        print(f"IMAGE_ERROR name={name} error={exc!r}")

    records[name] = record
    processed_this_run += 1
    total_done = sum(name in records for name in frame["name"])
    if processed_this_run % CHECKPOINT_EVERY == 0 or total_done == len(frame):
        save_checkpoint()
        elapsed = time.perf_counter() - proposal_start
        print(f"PROPOSALS {total_done}/{len(frame)} elapsed={elapsed:.1f}s")

save_checkpoint()
result_frame = pd.read_csv(PER_IMAGE_PATH)
proposal_seconds = time.perf_counter() - proposal_start
errors = result_frame["error"].fillna("").astype(str)
counts = result_frame["proposal_count"].to_numpy(dtype=np.int64)
union_patch_counts = result_frame["union_patch_count"].to_numpy(dtype=np.int64)
summary = {
    "dataset": DATASET_HANDLE,
    "evaluated_images": int(len(result_frame)),
    "failed_images": int(np.sum(errors != "")),
    "device": str(device),
    "gpu": torch.cuda.get_device_name(0),
    "wedetect_provider": proposer.providers[0] if proposer.providers else None,
    "config": {
        "score_threshold": SCORE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "proposal_top_k": PROPOSAL_TOP_K,
        "layers": active_layers,
        "a": ATTENTION_A,
        "b": ATTENTION_B,
        "proposal_mode": "union_all",
        "baseline_batch_size": BASELINE_BATCH_SIZE,
    },
    "proposal_statistics": {
        "images_with_proposals": int(np.sum(counts > 0)),
        "images_without_proposals": int(np.sum(counts == 0)),
        "total_unique_patch_masks": int(np.sum(counts)),
        "mean_per_image": float(np.mean(counts)),
        "median_per_image": float(np.median(counts)),
        "max_per_image": int(np.max(counts)),
        "mean_union_patch_count": float(np.mean(union_patch_counts)),
        "median_union_patch_count": float(np.median(union_patch_counts)),
        "max_union_patch_count": int(np.max(union_patch_counts)),
        "mean_union_coverage": float(np.mean(union_patch_counts / 256.0)),
        "images_with_full_patch_coverage": int(np.sum(union_patch_counts == 256)),
    },
    "metrics": {
        "baseline": metric_summary(result_frame["baseline_distance_km"].to_numpy()),
        "union_all": metric_summary(result_frame["union_distance_km"].to_numpy()),
    },
    "timing_seconds": {
        "baseline": baseline_seconds,
        "proposal_stage_this_run": proposal_seconds,
    },
    "artifacts": {
        "per_image": str(PER_IMAGE_PATH),
        "summary": str(SUMMARY_PATH),
    },
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("FINAL_SUMMARY")
print(json.dumps(summary, indent=2))
