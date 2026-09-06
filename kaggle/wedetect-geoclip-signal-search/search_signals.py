"""Search GeoCLIP layers/heads using cached WeDetect proposals on all Img2GPS3K."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
from PIL import Image, ImageFile
from transformers import AutoProcessor, CLIPModel


SEED = 42
DEFAULT_THRESHOLD = 0.40
DEFAULT_TOP_K = 100
BOOST = (2.0, -2.0)
SUPPRESS = (-2.0, 2.0)
NUM_LAYERS = 24
NUM_HEADS = 16
EFFECTIVE_BATCH = 160
THRESHOLDS_KM = (1, 25, 200, 750, 2500)
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/wedetect_geoclip_signal_search")
OUTPUT.mkdir(parents=True, exist_ok=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {pattern!r} under {INPUT}")
    print("INPUT", pattern, matches[0])
    return matches[0]


if not torch.cuda.is_available():
    raise RuntimeError("Signal search requires the requested G4")
DEVICE = torch.device("cuda")
GPU = torch.cuda.get_device_name(0)
if "RTX PRO 6000" not in GPU.upper():
    raise RuntimeError(f"Expected RTX PRO 6000 Blackwell, got {GPU}")
print("HARDWARE", GPU, torch.cuda.get_device_capability(0), torch.__version__, torch.version.cuda)

asset_manifest_path = find_one("asset_manifest.json")
ASSETS = asset_manifest_path.parent
CLIP_DIR = ASSETS / "clip-vit-large-patch14"
GEOCLIP_DIR = ASSETS / "geo-clip"
cache_path = find_one("proposals.jsonl")
cache_summary = json.loads(find_one("cache_summary.json").read_text(encoding="utf-8"))
proposals = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]

csv_path = find_one("im2gps3k_places365.csv")
image_dir = csv_path.parent / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame.name.map(lambda name: str(image_dir / name))
if frame.name.tolist() != [item["name"] for item in proposals]:
    raise RuntimeError("WeDetect cache does not exactly match the Img2GPS3K CSV order")
N = len(frame)
targets = frame[["LAT", "LON"]].to_numpy(dtype=np.float64)
discovery = np.asarray([
    int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 3 == 0 for name in frame.name
], dtype=bool)
holdout = ~discovery
print("SPLIT", {"all": N, "discovery": int(discovery.sum()), "holdout": int(holdout.sum())})


def boxes_to_mask(record: dict, threshold: float, top_k: int) -> torch.Tensor:
    width, height = record["width"], record["height"]
    selected = [item for item in record["proposals"] if item["score"] >= threshold][:top_k]
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


mask_keys = [(threshold, top_k) for threshold in (0.20, 0.30, 0.40, 0.50) for top_k in (1, 3, 100)]
mask_bank = {
    key: torch.stack([boxes_to_mask(record, *key) for record in proposals])
    for key in mask_keys
}
default_masks = mask_bank[(DEFAULT_THRESHOLD, DEFAULT_TOP_K)]
proposal_counts = np.asarray([sum(item["score"] >= DEFAULT_THRESHOLD for item in record["proposals"]) for record in proposals])
coverage = default_masks.float().mean(1).numpy()
print("MASKS", {"nonempty": int(default_masks.any(1).sum()), "mean_coverage": float(coverage.mean()), "mean_count": float(proposal_counts.mean())})


@dataclass
class State:
    masks: torch.Tensor | None = None
    schedule_a: torch.Tensor | None = None
    schedule_b: torch.Tensor | None = None
    query_scope: str = "all"
    capture: bool = False
    captured: list = field(default_factory=list)


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
            if state.query_scope == "cls":
                weights[:, :, 0, :] = weights[:, :, 0, :] + key_bias
            else:
                weights = weights + key_bias[:, :, None, :]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(*shape, -1)
        output = self.out_proj(output)
        if state.capture:
            spatial = weights[:, :, 0, 1:].float()
            normalized = spatial / spatial.sum(-1, keepdim=True).clamp_min(1e-12)
            region_mass = (normalized * state.masks[:, None, :]).sum(-1)
            entropy = -(normalized.clamp_min(1e-12).log() * normalized).sum(-1) / math.log(256)
            update_ratio = output[:, 0].float().norm(dim=-1) / hidden_states[:, 0].float().norm(dim=-1).clamp_min(1e-12)
            state.captured.append((layer_index, region_mass.detach(), entropy.detach(), update_ratio.detach()))
        return output, weights
    return forward


print("Loading GeoCLIP")
processor = AutoProcessor.from_pretrained(CLIP_DIR, local_files_only=True, use_fast=False)
clip = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(DEVICE).eval()
state = State()
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
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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
def encode(pixels: torch.Tensor):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw = clip.get_image_features(pixel_values=pixels.to(DEVICE, non_blocking=True))
        if hasattr(raw, "pooler_output"):
            raw = raw.pooler_output
        features = F.normalize(mlp(raw), dim=1)
    return raw.float(), features.float()


def predict(features: torch.Tensor):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = logit_scale * (features.to(gallery_features.dtype) @ gallery_features.T)
        top = logits.topk(2, dim=1)
        confidence = (top.values[:, 0].float() - torch.logsumexp(logits.float(), dim=1)).exp()
    return top.indices[:, 0].cpu().numpy(), confidence.cpu().numpy(), (top.values[:, 0] - top.values[:, 1]).float().cpu().numpy()


def haversine(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    lat1, lon1 = np.radians(first[..., 0]), np.radians(first[..., 1])
    lat2, lon2 = np.radians(second[..., 0]), np.radians(second[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0, 1 - value)))


# Baseline pass also records head-level CLS attention mass, entropy, and update norm.
baseline_features = np.empty((N, 512), np.float32)
baseline_raw_norm = np.empty(N, np.float32)
baseline_confidence = np.empty(N, np.float32)
baseline_margin = np.empty(N, np.float32)
baseline_gps = np.empty((N, 2), np.float32)
attention_mass = np.empty((N, NUM_LAYERS, NUM_HEADS), np.float32)
attention_entropy = np.empty_like(attention_mass)
update_ratio = np.empty((N, NUM_LAYERS), np.float32)
started = time.perf_counter()
for start in range(0, N, 32):
    indices = np.arange(start, min(start + 32, N))
    state.masks = default_masks[indices].to(DEVICE)
    state.capture, state.captured = True, []
    raw, features = encode(load_pixels(indices))
    state.capture = False
    pred, conf, margin = predict(features)
    baseline_features[indices] = features.cpu().numpy()
    baseline_raw_norm[indices] = raw.norm(dim=1).cpu().numpy()
    baseline_gps[indices] = gallery_np[pred]
    baseline_confidence[indices], baseline_margin[indices] = conf, margin
    for layer, mass, entropy, ratio in state.captured:
        attention_mass[indices, layer] = mass.cpu().numpy()
        attention_entropy[indices, layer] = entropy.cpu().numpy()
        update_ratio[indices, layer] = ratio.cpu().numpy()
    if len(indices) and indices[-1] % 320 < 32:
        print(f"BASELINE {indices[-1] + 1}/{N}", flush=True)
baseline_error = haversine(baseline_gps, targets)


def config(name, layers, heads=range(NUM_HEADS), ab=BOOST, threshold=DEFAULT_THRESHOLD, top_k=DEFAULT_TOP_K):
    return {"name": name, "layers": list(layers), "heads": list(heads), "a": ab[0], "b": ab[1], "threshold": threshold, "top_k": top_k}


def run_configs(configs, scope="all"):
    count = len(configs)
    outputs = {
        key: np.empty((N, count), np.float32)
        for key in ("error", "delta", "shift", "feature_cosine", "confidence", "margin")
    }
    base_batch = max(1, EFFECTIVE_BATCH // count)
    for start in range(0, N, base_batch):
        image_indices = np.arange(start, min(start + base_batch, N))
        batch = len(image_indices)
        pixels = load_pixels(image_indices).to(DEVICE)
        expanded_pixels = pixels[:, None].expand(batch, count, *pixels.shape[1:]).reshape(batch * count, *pixels.shape[1:]).contiguous()
        masks = torch.stack([
            mask_bank[(cfg["threshold"], cfg["top_k"])][image_index]
            for image_index in image_indices for cfg in configs
        ]).to(DEVICE)
        schedule_a = torch.zeros((batch * count, NUM_LAYERS, NUM_HEADS), device=DEVICE)
        schedule_b = torch.zeros_like(schedule_a)
        for image_offset in range(batch):
            for config_offset, cfg in enumerate(configs):
                flat = image_offset * count + config_offset
                if masks[flat].any():
                    for layer in cfg["layers"]:
                        schedule_a[flat, layer, cfg["heads"]] = cfg["a"]
                        schedule_b[flat, layer, cfg["heads"]] = cfg["b"]
        state.masks, state.schedule_a, state.schedule_b, state.query_scope = masks, schedule_a, schedule_b, scope
        _, features = encode(expanded_pixels)
        state.schedule_a = state.schedule_b = None
        pred, conf, margin = predict(features)
        gps = gallery_np[pred].reshape(batch, count, 2)
        errors = haversine(gps, targets[image_indices, None, :])
        shifts = haversine(gps, baseline_gps[image_indices, None, :])
        features = features.reshape(batch, count, -1).float().cpu().numpy()
        cosines = np.sum(features * baseline_features[image_indices, None, :], axis=-1)
        outputs["error"][image_indices] = errors
        outputs["delta"][image_indices] = baseline_error[image_indices, None] - errors
        outputs["shift"][image_indices] = shifts
        outputs["feature_cosine"][image_indices] = cosines
        outputs["confidence"][image_indices] = conf.reshape(batch, count)
        outputs["margin"][image_indices] = margin.reshape(batch, count)
        if image_indices[-1] % 250 < base_batch:
            print(f"SEARCH[{scope},{count}] {image_indices[-1] + 1}/{N}", flush=True)
    return outputs


def summarize(configs, outputs, family):
    rows = []
    for split_name, split in (("discovery", discovery), ("holdout", holdout), ("all", np.ones(N, bool))):
        for index, cfg in enumerate(configs):
            errors, deltas = outputs["error"][split, index], outputs["delta"][split, index]
            rows.append({
                "family": family, "split": split_name, "config": cfg["name"], "n": int(split.sum()),
                "mean_error_km": float(errors.mean()), "median_error_km": float(np.median(errors)),
                "mean_delta_km": float(deltas.mean()), "clipped_mean_delta_km": float(np.clip(deltas, -2500, 2500).mean()),
                "win_rate": float(np.mean(deltas > 1e-6)), "loss_rate": float(np.mean(deltas < -1e-6)),
                "mean_shift_km": float(outputs["shift"][split, index].mean()),
                "mean_feature_cosine": float(outputs["feature_cosine"][split, index].mean()),
                **{f"acc_{threshold}_km": float(np.mean(errors <= threshold)) for threshold in THRESHOLDS_KM},
            })
    return pd.DataFrame(rows)


# 1) Exhaustive single-layer scan on every test image.
layer_configs = [config(f"layer_{layer:02d}", [layer]) for layer in range(NUM_LAYERS)]
layer_outputs = run_configs(layer_configs)
layer_summary = summarize(layer_configs, layer_outputs, "single_layer")
best_layer_name = layer_summary.query("split == 'discovery'").sort_values("clipped_mean_delta_km", ascending=False).iloc[0].config
best_layer = int(best_layer_name.split("_")[-1])
print("BEST_LAYER_DISCOVERY", best_layer)

# 2) Search all heads within the discovery-selected layer, again on all images.
head_configs = [config(f"layer_{best_layer:02d}_head_{head:02d}", [best_layer], [head]) for head in range(NUM_HEADS)]
head_outputs = run_configs(head_configs)
head_summary = summarize(head_configs, head_outputs, "single_head")
best_head_name = head_summary.query("split == 'discovery'").sort_values("clipped_mean_delta_km", ascending=False).iloc[0].config
best_head = int(best_head_name.split("_")[-1])
print("BEST_HEAD_DISCOVERY", best_head)

# 3) Structured hypotheses: windows, layer groups, polarity, and proposal dose.
structured = [
    config("best_layer_all_heads", [best_layer]),
    config("best_layer_best_head", [best_layer], [best_head]),
    config("best_window_3", range(max(0, best_layer - 1), min(NUM_LAYERS, best_layer + 2))),
    config("best_window_5", range(max(0, best_layer - 2), min(NUM_LAYERS, best_layer + 3))),
    config("early_0_7", range(0, 8)), config("middle_8_15", range(8, 16)), config("late_16_23", range(16, 24)),
    config("all_layers_boost", range(24)), config("all_layers_suppress", range(24), ab=SUPPRESS),
]
for threshold in (0.20, 0.30, 0.40, 0.50):
    for top_k in (1, 3, 100):
        structured.append(config(f"dose_t{threshold:.1f}_k{top_k}", [best_layer], threshold=threshold, top_k=top_k))
structured_outputs = run_configs(structured)
structured_summary = summarize(structured, structured_outputs, "structured")

# CLS-only bias separates direct aggregation into CLS from patch-to-patch effects.
cls_configs = [
    config("cls_only_best_layer", [best_layer]),
    config("cls_only_window_3", range(max(0, best_layer - 1), min(NUM_LAYERS, best_layer + 2))),
    config("cls_only_all_layers", range(24)),
]
cls_outputs = run_configs(cls_configs, scope="cls")
cls_summary = summarize(cls_configs, cls_outputs, "query_scope")

summary_table = pd.concat([layer_summary, head_summary, structured_summary, cls_summary], ignore_index=True)
summary_table.to_csv(OUTPUT / "search_summary.csv", index=False)
np.savez_compressed(
    OUTPUT / "search_arrays.npz",
    names=frame.name.to_numpy(), discovery=discovery, baseline_error=baseline_error,
    baseline_confidence=baseline_confidence, baseline_margin=baseline_margin,
    proposal_counts=proposal_counts, coverage=coverage,
    layer_delta=layer_outputs["delta"], layer_shift=layer_outputs["shift"],
    head_delta=head_outputs["delta"], structured_delta=structured_outputs["delta"], cls_delta=cls_outputs["delta"],
    attention_mass=attention_mass, attention_entropy=attention_entropy, update_ratio=update_ratio,
)

# Relate observational attention signals to causal head interventions.
head_signals = []
for head in range(NUM_HEADS):
    mask = holdout & (coverage > 0)
    mass = attention_mass[mask, best_layer, head]
    enrichment = mass / np.maximum(coverage[mask], 1 / 256)
    delta = head_outputs["delta"][mask, head]
    head_signals.append({
        "head": head, "attention_mass": float(mass.mean()), "attention_enrichment": float(enrichment.mean()),
        "attention_entropy": float(attention_entropy[mask, best_layer, head].mean()),
        "causal_delta_km": float(delta.mean()), "causal_clipped_delta_km": float(np.clip(delta, -2500, 2500).mean()),
        "mean_abs_shift_km": float(head_outputs["shift"][mask, head].mean()),
    })
head_signals = pd.DataFrame(head_signals)
head_signals.to_csv(OUTPUT / "head_signals.csv", index=False)

best_structured = structured_summary.query("split == 'discovery'").sort_values("clipped_mean_delta_km", ascending=False).iloc[0]
locked = summary_table[(summary_table.split == "holdout") & (summary_table.config == best_structured.config)].iloc[0]
baseline_metrics = {
    "mean_error_km": float(baseline_error[holdout].mean()), "median_error_km": float(np.median(baseline_error[holdout])),
    **{f"acc_{threshold}_km": float(np.mean(baseline_error[holdout] <= threshold)) for threshold in THRESHOLDS_KM},
}
result = {
    "gpu": GPU, "images": N, "discovery_images": int(discovery.sum()), "holdout_images": int(holdout.sum()),
    "wedetect_cache": cache_summary, "best_layer_discovery": best_layer, "best_head_discovery": best_head,
    "selected_structured_config": best_structured.config,
    "locked_holdout": locked.to_dict(), "holdout_baseline": baseline_metrics,
    "signal_correlations": {
        "head_attention_enrichment_vs_causal_delta": float(head_signals.attention_enrichment.corr(head_signals.causal_clipped_delta_km)),
        "proposal_count_vs_best_layer_delta": float(np.corrcoef(proposal_counts[holdout], layer_outputs["delta"][holdout, best_layer])[0, 1]),
        "baseline_margin_vs_intervention_shift": float(np.corrcoef(baseline_margin[holdout], layer_outputs["shift"][holdout, best_layer])[0, 1]),
    },
    "elapsed_seconds": time.perf_counter() - started,
}
(OUTPUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

# Compact diagnostic figure for layer, head, dose, and observational/causal mismatch.
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
layer_plot = layer_summary.query("split == 'holdout'").sort_values("config")
axes[0, 0].plot(range(24), layer_plot.clipped_mean_delta_km, marker="o")
axes[0, 0].axhline(0, color="black", linewidth=1); axes[0, 0].axvline(best_layer, color="red", linestyle="--")
axes[0, 0].set(title="Locked-holdout single-layer response", xlabel="layer", ylabel="clipped mean improvement (km)")

head_plot = head_summary.query("split == 'holdout'").sort_values("config")
axes[0, 1].bar(range(16), head_plot.clipped_mean_delta_km)
axes[0, 1].axhline(0, color="black", linewidth=1); axes[0, 1].axvline(best_head, color="red", linestyle="--")
axes[0, 1].set(title=f"Head response at layer {best_layer}", xlabel="head", ylabel="clipped mean improvement (km)")

dose = structured_summary[
    (structured_summary.split == "holdout") & structured_summary.config.str.startswith("dose")
].copy()
for threshold, group in dose.groupby(dose.config.str.extract(r"t([0-9.]+)")[0]):
    group = group.assign(k=group.config.str.extract(r"k(\d+)")[0].astype(int)).sort_values("k")
    axes[1, 0].plot(group.k, group.clipped_mean_delta_km, marker="o", label=f"threshold {threshold}")
axes[1, 0].axhline(0, color="black", linewidth=1); axes[1, 0].legend(); axes[1, 0].set(title="Proposal dose on holdout", xlabel="top-k (100=all)", ylabel="clipped mean improvement (km)")

axes[1, 1].scatter(head_signals.attention_enrichment, head_signals.causal_clipped_delta_km)
for row in head_signals.itertuples(): axes[1, 1].annotate(str(row.head), (row.attention_enrichment, row.causal_clipped_delta_km), fontsize=8)
axes[1, 1].axhline(0, color="black", linewidth=1); axes[1, 1].set(title="Attention is not necessarily causal", xlabel="baseline proposal-attention enrichment", ylabel="causal head improvement (km)")
plt.tight_layout(); plt.savefig(OUTPUT / "signal_search_overview.png", dpi=180); plt.show()

display(summary_table.query("split == 'holdout'").sort_values("clipped_mean_delta_km", ascending=False).head(20))
display(head_signals.sort_values("causal_clipped_delta_km", ascending=False))
print("FINAL_RESULT")
print(json.dumps(result, indent=2))
