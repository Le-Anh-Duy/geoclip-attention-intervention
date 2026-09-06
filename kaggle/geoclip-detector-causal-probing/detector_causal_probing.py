"""Offline G4 evaluation of detector-derived causal attention interventions."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel


SEED = 42
N_IMAGES = 300
DINO_BOX_THRESHOLD = 0.35
DINO_TEXT_THRESHOLD = 0.25
WEDETECT_THRESHOLD = 0.40
WEDETECT_NMS_IOU = 0.70
WEDETECT_TOP_K = 100
BOOST = (2.0, -2.0)
SUPPRESS = (-2.0, 2.0)
THRESHOLDS_KM = (1, 25, 200, 750, 2500)
PROMPTS = [
    "road sign", "storefront text", "license plate", "traffic light", "vehicle",
    "building", "bridge", "tower", "utility pole", "road marking", "vegetation",
    "mountain", "water", "distinctive architecture",
]

INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/detector_causal_probing")
OUTPUT.mkdir(parents=True, exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(SEED)
np.random.seed(SEED)


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern!r} below {INPUT}")
    print(f"INPUT {pattern}: {matches[0]}")
    return matches[0]


if not torch.cuda.is_available():
    raise RuntimeError("This experiment requires the requested G4 GPU")
GPU_NAME = torch.cuda.get_device_name(0)
if "RTX PRO 6000" not in GPU_NAME.upper():
    raise RuntimeError(f"Expected RTX PRO 6000 Blackwell, received {GPU_NAME}")
print("HARDWARE", GPU_NAME, torch.cuda.get_device_capability(0), torch.__version__, torch.version.cuda)
print("OFFLINE", os.environ["HF_HUB_OFFLINE"], "ARC mounted=", any(INPUT.glob("arc-prize-2026-arc-agi-3*")))
DEVICE = torch.device("cuda")

asset_manifest_path = find_one("asset_manifest.json")
ASSETS = asset_manifest_path.parent
asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
DINO_DIR = ASSETS / "grounding-dino-base"
CLIP_DIR = ASSETS / "clip-vit-large-patch14"
GEOCLIP_DIR = ASSETS / "geo-clip"
WEDETECT_ONNX = ASSETS / "wedetect" / "wedetect_anything_base.onnx"

csv_path = find_one("im2gps3k_places365.csv")
if csv_path.parent.name == "test_set":
    image_dir = csv_path.parent / "im2gps3ktest" / "im2gps3ktest"
else:
    image_dir = csv_path.parent / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame["name"].map(lambda name: str(image_dir / name))
missing = frame.loc[~frame["image_path"].map(lambda value: Path(value).is_file()), "name"].tolist()
if missing:
    raise RuntimeError(f"Missing Img2GPS3K files: {missing[:5]}")


def spatial_sample(data: pd.DataFrame, count: int) -> pd.DataFrame:
    """Balance the pilot across coarse world cells instead of taking first N."""
    data = data.copy()
    data["geo_cell"] = (
        pd.cut(data["LAT"], bins=np.linspace(-90, 90, 7), labels=False, include_lowest=True).astype(str)
        + "_"
        + pd.cut(data["LON"], bins=np.linspace(-180, 180, 13), labels=False, include_lowest=True).astype(str)
    )
    groups = [group.sample(frac=1, random_state=SEED) for _, group in data.groupby("geo_cell", sort=True)]
    selected = []
    cursor = 0
    while len(selected) < min(count, len(data)):
        added = False
        for group in groups:
            if cursor < len(group) and len(selected) < count:
                selected.append(group.iloc[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return pd.DataFrame(selected).reset_index(drop=True)


frame = spatial_sample(frame, N_IMAGES)
targets = frame[["LAT", "LON"]].to_numpy(dtype=np.float64)
print("SAMPLE", len(frame), "images, geo cells=", frame["geo_cell"].nunique())


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, top_k: int) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size and len(keep) < top_k:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[index], x1[rest]), np.maximum(y1[index], y1[rest])
        xx2, yy2 = np.minimum(x2[index], x2[rest]), np.minimum(y2[index], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[index] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)


def letterbox(image: Image.Image, size: int = 640):
    width, height = image.size
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = np.asarray(resized)
    return canvas, ratio, (dw, dh)


# Install the staged CUDA runtime without resolving dependencies (the Kaggle
# image already supplies them). This avoids silently falling back to CPU ORT.
wheel = next((ASSETS / "wheels").glob("onnxruntime_gpu-*.whl"))
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall", str(wheel)],
    check=True,
)
import onnxruntime as ort


providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in ort.get_available_providers()]
wedetect_session = ort.InferenceSession(str(WEDETECT_ONNX), providers=providers)
print("WEDETECT providers=", wedetect_session.get_providers())


def wedetect(image: Image.Image) -> list[dict]:
    orig_w, orig_h = image.size
    padded, ratio, (dw, dh) = letterbox(image)
    tensor = np.ascontiguousarray((padded.astype(np.float32) / 255).transpose(2, 0, 1)[None])
    scores, boxes = wedetect_session.run(["scores", "bboxes"], {"input_image": tensor})
    scores, boxes = np.asarray(scores[0]), np.asarray(boxes[0, :, :4], dtype=np.float32)
    objectness = scores.max(axis=1)
    valid = np.flatnonzero(objectness > WEDETECT_THRESHOLD)
    if not len(valid):
        return []
    kept = valid[nms(boxes[valid], objectness[valid], WEDETECT_NMS_IOU, WEDETECT_TOP_K)]
    boxes = boxes[kept].copy()
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / ratio, 0, orig_w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / ratio, 0, orig_h)
    return [
        {"box": box.tolist(), "score": float(score)}
        for box, score in zip(boxes, objectness[kept]) if box[2] > box[0] and box[3] > box[1]
    ]


print("Loading Grounding DINO from local Kaggle input")
dino_processor = AutoProcessor.from_pretrained(DINO_DIR, local_files_only=True)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_DIR, local_files_only=True).to(DEVICE).eval()


@torch.inference_mode()
def dino_detect(image: Image.Image) -> list[dict]:
    inputs = dino_processor(images=image, text=[PROMPTS], return_tensors="pt").to(DEVICE)
    outputs = dino_model(**inputs)
    result = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=DINO_BOX_THRESHOLD,
        text_threshold=DINO_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]
    labels = result.get("text_labels", result.get("labels", []))
    detections = []
    for box, score, label in zip(result["boxes"], result["scores"], labels):
        values = box.detach().float().cpu().tolist()
        if values[2] > values[0] and values[3] > values[1]:
            detections.append({"box": values, "score": float(score.detach().cpu()), "label": str(label)})
    return sorted(detections, key=lambda item: item["score"], reverse=True)


detector_records = []
detector_started = time.perf_counter()
for index, row in frame.iterrows():
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    dino_boxes = dino_detect(image)
    wedetect_boxes = wedetect(image)
    detector_records.append({
        "name": row["name"], "width": image.width, "height": image.height,
        "dino": dino_boxes, "wedetect": wedetect_boxes,
    })
    if (index + 1) % 10 == 0 or index + 1 == len(frame):
        print(f"DETECTORS {index + 1}/{len(frame)} elapsed={time.perf_counter() - detector_started:.1f}s", flush=True)
(OUTPUT / "detector_records.json").write_text(json.dumps(detector_records), encoding="utf-8")

del dino_model, dino_processor, wedetect_session
gc.collect()
torch.cuda.empty_cache()


def boxes_to_mask(boxes: list[dict], width: int, height: int, limit: int | None = None) -> torch.Tensor:
    """Map original-image boxes through CLIP resize/center-crop to 16x16 patches."""
    mask = torch.zeros((16, 16), dtype=torch.bool)
    scale = 224 / min(width, height)
    resized_w, resized_h = width * scale, height * scale
    crop_x, crop_y = (resized_w - 224) / 2, (resized_h - 224) / 2
    for detection in boxes[:limit]:
        x1, y1, x2, y2 = detection["box"]
        x1, x2 = max(0, x1 * scale - crop_x), min(224, x2 * scale - crop_x)
        y1, y2 = max(0, y1 * scale - crop_y), min(224, y2 * scale - crop_y)
        if x2 <= x1 or y2 <= y1:
            continue
        c0, c1 = int(x1 // 14), min(16, int((x2 - 1e-6) // 14) + 1)
        r0, r1 = int(y1 // 14), min(16, int((y2 - 1e-6) // 14) + 1)
        mask[r0:r1, c0:c1] = True
    return mask.flatten()


@dataclass
class InterventionState:
    layer_ab: dict[int, tuple[float, float]] = field(default_factory=dict)
    mask: torch.Tensor | None = None

    def bias(self, layer: int, positions: int):
        if self.mask is None or layer not in self.layer_ab:
            return None
        a, b = self.layer_ab[layer]
        masks = self.mask[None] if self.mask.ndim == 1 else self.mask
        result = torch.full((len(masks), positions), b, dtype=torch.float32, device=masks.device)
        result[:, 0] = 0
        result[:, 1:].masked_fill_(masks, a)
        return result[:, None, None, :]


def attention_forward(state: InterventionState, layer_index: int):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        shape = hidden_states.shape[:-1]
        hidden_shape = (*shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            weights = weights + attention_mask
        bias = state.bias(layer_index, weights.shape[-1])
        if bias is not None:
            weights = weights + bias.to(weights)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(*shape, -1)
        return self.out_proj(output), weights
    return forward


def patch_clip(clip: CLIPModel) -> InterventionState:
    state = InterventionState()
    for index, layer in enumerate(clip.vision_model.encoder.layers):
        layer.self_attn.forward = types.MethodType(attention_forward(state, index), layer.self_attn)
    return state


sys.path.insert(0, str(GEOCLIP_DIR))
from geoclip.model.location_encoder import LocationEncoder  # noqa: E402
from geoclip.model.misc import load_gps_data  # noqa: E402

print("Loading offline GeoCLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_DIR, local_files_only=True, use_fast=False)
clip = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(DEVICE).eval()
mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512)).to(DEVICE).eval()
weights_dir = GEOCLIP_DIR / "geoclip" / "model" / "weights"
mlp.load_state_dict(torch.load(weights_dir / "image_encoder_mlp_weights.pth", map_location=DEVICE))
location_encoder = LocationEncoder().to(DEVICE).eval()
gps_gallery = load_gps_data(GEOCLIP_DIR / "geoclip" / "model" / "gps_gallery" / "coordinates_100K.csv")
logit_scale = torch.load(weights_dir / "logit_scale_weights.pth", map_location=DEVICE).exp()
state = patch_clip(clip)
with torch.inference_mode():
    gallery_parts = []
    for start in range(0, len(gps_gallery), 4096):
        gallery_parts.append(F.normalize(location_encoder(gps_gallery[start:start + 4096].to(DEVICE)), dim=1))
    gallery_features = torch.cat(gallery_parts)
gallery_np = gps_gallery.numpy()


@torch.inference_mode()
def predict(pixel_values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    features = clip.get_image_features(pixel_values=pixel_values.to(DEVICE))
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    features = F.normalize(mlp(features), dim=1)
    probabilities = (logit_scale * features @ gallery_features.T).softmax(dim=1)
    confidence, indices = probabilities.max(dim=1)
    return indices.cpu().numpy(), confidence.cpu().numpy()


def haversine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    lat1, lon1 = np.radians(first[..., 0]), np.radians(first[..., 1])
    lat2, lon2 = np.radians(second[..., 0]), np.radians(second[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0, 1 - value)))


print("Computing baseline")
baseline_indices, baseline_confidences = [], []
state.mask, state.layer_ab = None, {}
for start in range(0, len(frame), 32):
    images = []
    for path in frame.iloc[start:start + 32]["image_path"]:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    pixels = clip_processor(images=images, return_tensors="pt")["pixel_values"]
    indices, confidences = predict(pixels)
    baseline_indices.extend(indices.tolist())
    baseline_confidences.extend(confidences.tolist())
baseline_gps = gallery_np[np.asarray(baseline_indices)]
baseline_error = haversine(baseline_gps, targets)

boost_names = ["wedetect_top1", "wedetect_top3", "wedetect_all", "dino_all", "agreement", "union", "disagreement"]
suppress_names = ["agreement", "union"]
layer_groups = {"early": range(0, 8), "middle": range(8, 16), "late": range(16, 24)}
long_records = []

for image_index, (row, detections) in enumerate(zip(frame.itertuples(index=False), detector_records)):
    with Image.open(row.image_path) as source:
        image = source.convert("RGB")
    dino_mask = boxes_to_mask(detections["dino"], image.width, image.height)
    wedetect_all = boxes_to_mask(detections["wedetect"], image.width, image.height)
    masks = {
        "wedetect_top1": boxes_to_mask(detections["wedetect"], image.width, image.height, 1),
        "wedetect_top3": boxes_to_mask(detections["wedetect"], image.width, image.height, 3),
        "wedetect_all": wedetect_all,
        "dino_all": dino_mask,
        "agreement": wedetect_all & dino_mask,
        "union": wedetect_all | dino_mask,
        "disagreement": wedetect_all ^ dino_mask,
    }
    base_common = {
        "name": row.name, "target_lat": row.LAT, "target_lon": row.LON,
        "dino_count": len(detections["dino"]), "wedetect_count": len(detections["wedetect"]),
        "agreement_iou": float((masks["agreement"].sum() / masks["union"].sum()).item()) if masks["union"].any() else 0.0,
    }
    long_records.append({
        **base_common, "config": "baseline", "mask_coverage": 0.0,
        "pred_lat": baseline_gps[image_index, 0], "pred_lon": baseline_gps[image_index, 1],
        "confidence": baseline_confidences[image_index], "error_km": baseline_error[image_index],
        "delta_km": 0.0, "shift_km": 0.0,
    })
    pixels_one = clip_processor(images=[image], return_tensors="pt")["pixel_values"]

    def evaluate_batch(names: list[str], ab: tuple[float, float], prefix: str, layers=range(24)):
        nonempty = [name for name in names if masks[name].any()]
        predictions = {}
        if nonempty:
            state.mask = torch.stack([masks[name] for name in nonempty]).to(DEVICE)
            state.layer_ab = {layer: ab for layer in layers}
            indices, confidences = predict(pixels_one.repeat(len(nonempty), 1, 1, 1))
            for name, index, confidence in zip(nonempty, indices, confidences):
                predictions[name] = (gallery_np[index], confidence)
        state.mask, state.layer_ab = None, {}
        for name in names:
            gps, confidence = predictions.get(name, (baseline_gps[image_index], baseline_confidences[image_index]))
            error = float(haversine(gps, targets[image_index]))
            shift = float(haversine(gps, baseline_gps[image_index]))
            long_records.append({
                **base_common, "config": f"{prefix}_{name}", "mask_coverage": float(masks[name].float().mean()),
                "pred_lat": gps[0], "pred_lon": gps[1], "confidence": confidence,
                "error_km": error, "delta_km": float(baseline_error[image_index] - error), "shift_km": shift,
            })

    evaluate_batch(boost_names, BOOST, "boost")
    evaluate_batch(suppress_names, SUPPRESS, "suppress")
    for group_name, layers in layer_groups.items():
        evaluate_batch(["agreement"], BOOST, f"boost_{group_name}", layers)

    if (image_index + 1) % 10 == 0 or image_index + 1 == len(frame):
        pd.DataFrame(long_records).to_csv(OUTPUT / "results_long.csv", index=False)
        print(f"GEOCLIP {image_index + 1}/{len(frame)}", flush=True)

results = pd.DataFrame(long_records)
results.to_csv(OUTPUT / "results_long.csv", index=False)


def bootstrap_ci(values: np.ndarray, rounds: int = 2000) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(rounds)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


metric_rows = []
for config, group in results.groupby("config", sort=False):
    errors, deltas = group["error_km"].to_numpy(), group["delta_km"].to_numpy()
    metric_rows.append({
        "config": config, "n": len(group), "mean_error_km": errors.mean(), "median_error_km": np.median(errors),
        "mean_delta_km": deltas.mean(), "delta_ci95_low": bootstrap_ci(deltas)[0], "delta_ci95_high": bootstrap_ci(deltas)[1],
        "win_rate": np.mean(deltas > 1e-6), "loss_rate": np.mean(deltas < -1e-6), "mean_shift_km": group["shift_km"].mean(),
        **{f"acc_{threshold}_km": np.mean(errors <= threshold) for threshold in THRESHOLDS_KM},
    })
metrics = pd.DataFrame(metric_rows).sort_values("mean_delta_km", ascending=False)
metrics.to_csv(OUTPUT / "metrics.csv", index=False)
display(metrics)

pivot = results.pivot(index="name", columns="config", values="delta_km")
metadata = results.query("config == 'baseline'").set_index("name")
summary = {
    "experiment": "counterfactual_detector_agreement",
    "gpu": GPU_NAME,
    "offline": True,
    "images": len(frame),
    "spatial_cells": int(frame["geo_cell"].nunique()),
    "prompts": PROMPTS,
    "thresholds": {"dino_box": DINO_BOX_THRESHOLD, "dino_text": DINO_TEXT_THRESHOLD, "wedetect": WEDETECT_THRESHOLD},
    "bias": {"boost": BOOST, "suppress": SUPPRESS},
    "best_mean_delta_config": str(metrics.query("config != 'baseline'").iloc[0]["config"]),
    "best_mean_delta_km": float(metrics.query("config != 'baseline'").iloc[0]["mean_delta_km"]),
    "mean_detector_patch_iou": float(metadata["agreement_iou"].mean()),
    "proposal_delta_correlation": float(metadata["wedetect_count"].corr(pivot["boost_wedetect_all"])),
    "asset_manifest": asset_manifest,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Paper-ready plots: effect distribution, dose-response, agreement, and geography.
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
plot_configs = ["boost_wedetect_all", "boost_dino_all", "boost_agreement", "boost_union", "suppress_agreement"]
results[results.config.isin(plot_configs)].boxplot(column="delta_km", by="config", ax=axes[0, 0], rot=25, showfliers=False)
axes[0, 0].axhline(0, color="black", linewidth=1); axes[0, 0].set_title("GPS error improvement"); axes[0, 0].set_ylabel("baseline - intervention (km)")

dose = metrics[metrics.config.isin(["boost_wedetect_top1", "boost_wedetect_top3", "boost_wedetect_all"])]
axes[0, 1].errorbar(dose.config, dose.mean_delta_km, yerr=[dose.mean_delta_km-dose.delta_ci95_low, dose.delta_ci95_high-dose.mean_delta_km], marker="o")
axes[0, 1].axhline(0, color="black", linewidth=1); axes[0, 1].set_title("WeDetect proposal dose-response"); axes[0, 1].tick_params(axis="x", rotation=20)

joined = metadata.join(pivot[["boost_agreement", "boost_union"]])
axes[1, 0].scatter(joined.agreement_iou, joined.boost_agreement - joined.boost_union, s=16, alpha=.6)
axes[1, 0].axhline(0, color="black", linewidth=1); axes[1, 0].set_xlabel("DINO-WeDetect patch IoU"); axes[1, 0].set_ylabel("agreement gain over union (km)"); axes[1, 0].set_title("Does detector consensus beat coverage?")

geo_delta = pivot["boost_agreement"]
scatter = axes[1, 1].scatter(metadata.target_lon, metadata.target_lat, c=geo_delta, cmap="coolwarm", vmin=-np.quantile(abs(geo_delta), .9), vmax=np.quantile(abs(geo_delta), .9), s=22)
axes[1, 1].set_xlim(-180, 180); axes[1, 1].set_ylim(-90, 90); axes[1, 1].set_xlabel("longitude"); axes[1, 1].set_ylabel("latitude"); axes[1, 1].set_title("Geographic distribution: agreement boost")
fig.colorbar(scatter, ax=axes[1, 1], label="improvement (km)")
fig.suptitle(""); plt.tight_layout(); plt.savefig(OUTPUT / "experiment_overview.png", dpi=180); plt.show()

# Qualitative detector overlays for the most responsive images.
example_names = pivot["boost_agreement"].abs().nlargest(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
record_by_name = {item["name"]: item for item in detector_records}
row_by_name = frame.set_index("name")
for axis, name in zip(axes.flat, example_names):
    with Image.open(row_by_name.loc[name, "image_path"]) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in record_by_name[name]["wedetect"]:
        draw.rectangle(item["box"], outline="cyan", width=3)
    for item in record_by_name[name]["dino"]:
        draw.rectangle(item["box"], outline="red", width=3)
    axis.imshow(image); axis.axis("off"); axis.set_title(f"delta={pivot.loc[name, 'boost_agreement']:.0f} km\ncyan=WeDetect red=DINO")
plt.tight_layout(); plt.savefig(OUTPUT / "qualitative_detector_agreement.png", dpi=160); plt.show()

print("FINAL_SUMMARY")
print(json.dumps(summary, indent=2))
