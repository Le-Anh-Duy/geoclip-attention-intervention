"""Test whether discovery-positive GeoCLIP layers have a combined causal benefit."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import AutoProcessor, CLIPModel


SEED = 42
NUM_LAYERS = 24
NUM_HEADS = 16
THRESHOLD = 0.40
TOP_K = 100
BASE_MAGNITUDE = 2.0
CLIP_KM = 2500.0
EFFECTIVE_BATCH = 160
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/wedetect_geoclip_layer_synergy")
OUTPUT.mkdir(parents=True, exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {pattern!r} under {INPUT}")
    print("INPUT", pattern, matches[0])
    return matches[0]


if not torch.cuda.is_available():
    raise RuntimeError("Layer synergy search requires the requested G4")
DEVICE = torch.device("cuda")
GPU = torch.cuda.get_device_name(0)
if "RTX PRO 6000" not in GPU.upper():
    raise RuntimeError(f"Expected RTX PRO 6000 Blackwell, got {GPU}")
print("HARDWARE", GPU, torch.cuda.get_device_capability(0), torch.__version__, torch.version.cuda)

asset_manifest_path = find_one("asset_manifest.json")
ASSETS = asset_manifest_path.parent
CLIP_DIR = ASSETS / "clip-vit-large-patch14"
GEOCLIP_DIR = ASSETS / "geo-clip"
proposals = [json.loads(line) for line in find_one("proposals.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
prior_summary = pd.read_csv(find_one("search_summary.csv"))
prior_arrays = np.load(find_one("search_arrays.npz"), allow_pickle=True)

csv_path = find_one("im2gps3k_places365.csv")
image_dir = csv_path.parent / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame.name.map(lambda name: str(image_dir / name))
if frame.name.tolist() != [item["name"] for item in proposals]:
    raise RuntimeError("WeDetect cache does not exactly match the Img2GPS3K CSV order")
if frame.name.tolist() != prior_arrays["names"].tolist():
    raise RuntimeError("Prior signal-search arrays do not match the current image order")
N = len(frame)
targets = frame[["LAT", "LON"]].to_numpy(dtype=np.float64)
discovery = np.asarray([
    int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 3 == 0 for name in frame.name
], dtype=bool)
holdout = ~discovery
if not np.array_equal(discovery, prior_arrays["discovery"]):
    raise RuntimeError("Discovery split differs from the prior search")

single_discovery = prior_summary.query("family == 'single_layer' and split == 'discovery'").copy()
single_discovery["layer"] = single_discovery.config.str.extract(r"(\d+)$").astype(int)
positive = single_discovery[single_discovery.clipped_mean_delta_km > 0].sort_values(
    "clipped_mean_delta_km", ascending=False
)
positive_layers = positive.layer.astype(int).tolist()
best_layer = positive_layers[0]
if len(positive_layers) < 2:
    raise RuntimeError(f"Need at least two discovery-positive layers, got {positive_layers}")
print("SPLIT", {"all": N, "discovery": int(discovery.sum()), "holdout": int(holdout.sum())})
print("DISCOVERY_POSITIVE_LAYERS", positive_layers)


def boxes_to_mask(record: dict) -> torch.Tensor:
    width, height = record["width"], record["height"]
    selected = [item for item in record["proposals"] if item["score"] >= THRESHOLD][:TOP_K]
    mask = torch.zeros((16, 16), dtype=torch.bool)
    scale = 224 / min(width, height)
    crop_x, crop_y = (width * scale - 224) / 2, (height * scale - 224) / 2
    for item in selected:
        x1, y1, x2, y2 = item["box"]
        x1, x2 = max(0, x1 * scale - crop_x), min(224, x2 * scale - crop_x)
        y1, y2 = max(0, y1 * scale - crop_y), min(224, y2 * scale - crop_y)
        if x2 <= x1 or y2 <= y1:
            continue
        c0, c1 = int(x1 // 14), min(16, int((x2 - 1e-6) // 14) + 1)
        r0, r1 = int(y1 // 14), min(16, int((y2 - 1e-6) // 14) + 1)
        mask[r0:r1, c0:c1] = True
    return mask.flatten()


masks = torch.stack([boxes_to_mask(record) for record in proposals])
print("MASKS", {"nonempty": int(masks.any(1).sum()), "mean_coverage": float(masks.float().mean())})


@dataclass
class State:
    masks: torch.Tensor | None = None
    schedule_a: torch.Tensor | None = None
    schedule_b: torch.Tensor | None = None


state = State()


def attention_forward(state: State, layer_index: int):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        shape = hidden_states.shape[:-1]
        hidden_shape = (*shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            weights = weights + attention_mask
        if state.schedule_a is not None:
            a = state.schedule_a[:, layer_index, :, None]
            b = state.schedule_b[:, layer_index, :, None]
            spatial_bias = torch.where(state.masks[:, None, :], a, b)
            key_bias = F.pad(spatial_bias, (1, 0), value=0.0).to(weights)
            weights = weights + key_bias[:, :, None, :]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(*shape, -1)
        return self.out_proj(output), weights
    return forward


processor = AutoProcessor.from_pretrained(CLIP_DIR, local_files_only=True, use_fast=False)
clip = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(DEVICE).eval()
for layer_index, layer in enumerate(clip.vision_model.encoder.layers):
    layer.self_attn.forward = types.MethodType(attention_forward(state, layer_index), layer.self_attn)

sys.path.insert(0, str(GEOCLIP_DIR))
from geoclip.model.location_encoder import LocationEncoder  # noqa: E402
from geoclip.model.misc import load_gps_data  # noqa: E402

weights_dir = GEOCLIP_DIR / "geoclip" / "model" / "weights"
mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 512)).to(DEVICE).eval()
mlp.load_state_dict(torch.load(weights_dir / "image_encoder_mlp_weights.pth", map_location=DEVICE))
location_encoder = LocationEncoder().to(DEVICE).eval()
gps_gallery = load_gps_data(GEOCLIP_DIR / "geoclip" / "model" / "gps_gallery" / "coordinates_100K.csv")
gallery_np = gps_gallery.numpy()
logit_scale = torch.load(weights_dir / "logit_scale_weights.pth", map_location=DEVICE).exp()
with torch.inference_mode():
    gallery_features = torch.cat([
        F.normalize(location_encoder(gps_gallery[start:start + 4096].to(DEVICE)), dim=1)
        for start in range(0, len(gps_gallery), 4096)
    ])


def load_pixels(indices) -> torch.Tensor:
    images = []
    for index in indices:
        with Image.open(frame.iloc[index].image_path) as source:
            images.append(source.convert("RGB"))
    return processor(images=images, return_tensors="pt")["pixel_values"]


@torch.inference_mode()
def encode(pixels: torch.Tensor) -> torch.Tensor:
    raw = clip.get_image_features(pixel_values=pixels.to(DEVICE, non_blocking=True))
    if hasattr(raw, "pooler_output"):
        raw = raw.pooler_output
    return F.normalize(mlp(raw), dim=1)


def predict(features: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        logits = logit_scale * (features.to(gallery_features.dtype) @ gallery_features.T)
        return logits.argmax(1).cpu().numpy()


def haversine(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    lat1, lon1 = np.radians(first[..., 0]), np.radians(first[..., 1])
    lat2, lon2 = np.radians(second[..., 0]), np.radians(second[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0, 1 - value)))


# Verify deterministic feature extraction before the full paired experiment.
probe_pixels = load_pixels(np.arange(min(8, N)))
probe_first = encode(probe_pixels)
probe_second = encode(probe_pixels)
determinism_max_abs_diff = float((probe_first - probe_second).abs().max().cpu())
if determinism_max_abs_diff != 0.0:
    raise RuntimeError(f"FP32 feature extraction is not deterministic: max diff {determinism_max_abs_diff}")
print("DETERMINISM_MAX_ABS_DIFF", determinism_max_abs_diff)
del probe_pixels, probe_first, probe_second

# Recompute baseline features for paired feature and prediction-shift diagnostics.
baseline_features = np.empty((N, 512), np.float32)
baseline_gps = np.empty((N, 2), np.float32)
for start in range(0, N, 32):
    indices = np.arange(start, min(start + 32, N))
    features = encode(load_pixels(indices))
    baseline_features[indices] = features.cpu().numpy()
    baseline_gps[indices] = gallery_np[predict(features)]
baseline_error = haversine(baseline_gps, targets)
baseline_difference = baseline_error - prior_arrays["baseline_error"]
print("BASELINE_REPRODUCIBILITY", {
    "mean_abs_error_difference_km": float(np.abs(baseline_difference).mean()),
    "max_abs_error_difference_km": float(np.abs(baseline_difference).max()),
    "changed_error_count": int(np.sum(np.abs(baseline_difference) > 0.1)),
})


def make_config(layers, scale_name, magnitude):
    layer_text = "_".join(f"{layer:02d}" for layer in layers)
    return {
        "name": f"{scale_name}_layers_{layer_text}",
        "layers": list(layers),
        "scale": scale_name,
        "magnitude": magnitude,
        "kind": "single_control" if len(layers) == 1 else "layer_combination",
    }


single_controls = [make_config((layer,), "fixed", BASE_MAGNITUDE) for layer in positive_layers]
combination_configs = []
for count in range(2, len(positive_layers) + 1):
    for subset in itertools.combinations(positive_layers, count):
        combination_configs.extend([
            make_config(subset, "fixed", BASE_MAGNITUDE),
            make_config(subset, "sqrt", BASE_MAGNITUDE / math.sqrt(count)),
            make_config(subset, "linear", BASE_MAGNITUDE / count),
        ])
configs = single_controls + combination_configs
control_index = {cfg["layers"][0]: index for index, cfg in enumerate(single_controls)}
print("CANDIDATES", {"single_controls": len(single_controls), "layer_combinations": len(combination_configs)})


def run_configs(configs):
    count = len(configs)
    outputs = {key: np.empty((N, count), np.float32) for key in ("error", "delta", "shift", "feature_cosine")}
    base_batch = max(1, EFFECTIVE_BATCH // count)
    for start in range(0, N, base_batch):
        image_indices = np.arange(start, min(start + base_batch, N))
        batch = len(image_indices)
        pixels = load_pixels(image_indices).to(DEVICE)
        expanded = pixels[:, None].expand(batch, count, *pixels.shape[1:]).reshape(batch * count, *pixels.shape[1:]).contiguous()
        batch_masks = masks[image_indices, None].expand(batch, count, 256).reshape(batch * count, 256).to(DEVICE)
        schedule_a = torch.zeros((batch * count, NUM_LAYERS, NUM_HEADS), device=DEVICE)
        schedule_b = torch.zeros_like(schedule_a)
        for image_offset in range(batch):
            for config_offset, cfg in enumerate(configs):
                flat = image_offset * count + config_offset
                if batch_masks[flat].any():
                    schedule_a[flat, cfg["layers"], :] = cfg["magnitude"]
                    schedule_b[flat, cfg["layers"], :] = -cfg["magnitude"]
        state.masks, state.schedule_a, state.schedule_b = batch_masks, schedule_a, schedule_b
        features = encode(expanded)
        state.schedule_a = state.schedule_b = None
        gps = gallery_np[predict(features)].reshape(batch, count, 2)
        errors = haversine(gps, targets[image_indices, None, :])
        outputs["error"][image_indices] = errors
        outputs["delta"][image_indices] = baseline_error[image_indices, None] - errors
        outputs["shift"][image_indices] = haversine(gps, baseline_gps[image_indices, None, :])
        features_np = features.reshape(batch, count, -1).cpu().numpy()
        outputs["feature_cosine"][image_indices] = np.sum(features_np * baseline_features[image_indices, None, :], axis=-1)
        if image_indices[-1] % 200 < base_batch:
            print(f"SYNERGY {image_indices[-1] + 1}/{N}", flush=True)
    return outputs


started = time.perf_counter()
outputs = run_configs(configs)
elapsed = time.perf_counter() - started
rows = []
for split_name, split in (("discovery", discovery), ("holdout", holdout), ("all", np.ones(N, bool))):
    for index, cfg in enumerate(configs):
        delta = outputs["delta"][split, index]
        clipped = np.clip(delta, -CLIP_KM, CLIP_KM)
        constituent_scores = [float(np.clip(
            outputs["delta"][split, control_index[layer]], -CLIP_KM, CLIP_KM
        ).mean()) for layer in cfg["layers"]]
        rows.append({
            "split": split_name,
            "config": cfg["name"],
            "kind": cfg["kind"],
            "layers": ",".join(map(str, cfg["layers"])),
            "layer_count": len(cfg["layers"]),
            "scale": cfg["scale"],
            "magnitude": cfg["magnitude"],
            "n": int(split.sum()),
            "mean_error_km": float(outputs["error"][split, index].mean()),
            "mean_delta_km": float(delta.mean()),
            "clipped_mean_delta_km": float(clipped.mean()),
            "gain_over_best_constituent_km": float(clipped.mean() - max(constituent_scores)),
            "win_rate": float(np.mean(delta > 1e-6)),
            "loss_rate": float(np.mean(delta < -1e-6)),
            "mean_shift_km": float(outputs["shift"][split, index].mean()),
            "mean_feature_cosine": float(outputs["feature_cosine"][split, index].mean()),
            **{f"acc_{km}_km": float(np.mean(outputs["error"][split, index] <= km)) for km in (1, 25, 200, 750, 2500)},
        })
summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT / "synergy_summary.csv", index=False)

best_discovery = summary.query(
    "split == 'discovery' and kind == 'layer_combination'"
).sort_values("clipped_mean_delta_km", ascending=False).iloc[0]
selected_index = next(index for index, cfg in enumerate(configs) if cfg["name"] == best_discovery.config)
locked = summary.query("split == 'holdout' and config == @best_discovery.config").iloc[0]
best_layer_config = f"fixed_layers_{best_layer:02d}"
locked_single = summary.query("split == 'holdout' and config == @best_layer_config").iloc[0]
selected_delta = outputs["delta"][holdout, selected_index]
single_delta = outputs["delta"][holdout, control_index[best_layer]]
paired_gain = np.clip(selected_delta, -CLIP_KM, CLIP_KM) - np.clip(single_delta, -CLIP_KM, CLIP_KM)


def bootstrap_ci(values, repeats=10000):
    rng = np.random.default_rng(SEED)
    means = np.empty(repeats)
    for index in range(repeats):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


result = {
    "gpu": GPU,
    "images": N,
    "discovery_images": int(discovery.sum()),
    "holdout_images": int(holdout.sum()),
    "positive_layers_discovery_ranked": positive_layers,
    "evaluation_precision": "deterministic_fp32_no_tf32",
    "determinism_max_abs_feature_difference": determinism_max_abs_diff,
    "single_control_count": len(single_controls),
    "combination_candidate_count": len(combination_configs),
    "baseline_reproducibility": {
        "mean_abs_error_difference_km": float(np.abs(baseline_difference).mean()),
        "max_abs_error_difference_km": float(np.abs(baseline_difference).max()),
        "changed_error_count": int(np.sum(np.abs(baseline_difference) > 0.1)),
    },
    "selected_on_discovery": best_discovery.to_dict(),
    "locked_holdout": locked.to_dict(),
    "locked_best_single_layer": locked_single.to_dict(),
    "locked_synergy_minus_single_clipped_km": float(paired_gain.mean()),
    "locked_synergy_delta_ci95_km": bootstrap_ci(selected_delta),
    "locked_synergy_clipped_delta_ci95_km": bootstrap_ci(np.clip(selected_delta, -CLIP_KM, CLIP_KM)),
    "locked_synergy_minus_single_ci95_km": bootstrap_ci(paired_gain),
    "elapsed_seconds": elapsed,
}
(OUTPUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
np.savez_compressed(
    OUTPUT / "synergy_arrays.npz",
    names=frame.name.to_numpy(),
    discovery=discovery,
    config_names=np.asarray([cfg["name"] for cfg in configs]),
    delta=outputs["delta"],
    shift=outputs["shift"],
    feature_cosine=outputs["feature_cosine"],
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for scale, group in summary.query("split == 'holdout' and kind == 'layer_combination'").groupby("scale"):
    axes[0].scatter(group.layer_count, group.clipped_mean_delta_km, alpha=0.7, label=scale)
axes[0].axhline(float(locked_single.clipped_mean_delta_km), color="black", linestyle="--", label=f"single layer {best_layer}")
axes[0].set(title="Locked holdout: combinations vs best single", xlabel="number of layers", ylabel="clipped mean improvement (km)")
axes[0].legend()
disc = summary.query("split == 'discovery' and kind == 'layer_combination'")[["config", "clipped_mean_delta_km"]].rename(columns={"clipped_mean_delta_km": "discovery_score"})
held = summary.query("split == 'holdout' and kind == 'layer_combination'")[["config", "clipped_mean_delta_km"]].rename(columns={"clipped_mean_delta_km": "holdout_score"})
joined = disc.merge(held, on="config")
axes[1].scatter(joined.discovery_score, joined.holdout_score, alpha=0.7)
axes[1].axhline(0, color="black", linewidth=1); axes[1].axvline(0, color="black", linewidth=1)
axes[1].set(title=f"Discovery-to-holdout transfer (r={joined.discovery_score.corr(joined.holdout_score):.3f})", xlabel="discovery clipped improvement (km)", ylabel="holdout clipped improvement (km)")
plt.tight_layout(); plt.savefig(OUTPUT / "layer_synergy_overview.png", dpi=180); plt.show()

print(json.dumps(result, indent=2))
display(summary.query("split == 'holdout'").sort_values("clipped_mean_delta_km", ascending=False).head(20))
