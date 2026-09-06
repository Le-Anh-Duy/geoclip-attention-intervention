"""Run WeDetect-Uni once on the complete Img2GPS3K test set and cache proposals."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms


SEED = 42
BATCH_SIZE = 32
CACHE_THRESHOLD = 0.15
NMS_IOU = 0.70
TOP_K = 100
INPUT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/wedetect_img2gps3k_cache")
OUTPUT.mkdir(parents=True, exist_ok=True)
JSONL = OUTPUT / "proposals.jsonl"
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {pattern!r} under {INPUT}")
    print("INPUT", pattern, matches[0])
    return matches[0]


if not torch.cuda.is_available():
    raise RuntimeError("WeDetect cache requires the requested G4")
DEVICE = torch.device("cuda")
GPU = torch.cuda.get_device_name(0)
if "RTX PRO 6000" not in GPU.upper():
    raise RuntimeError(f"Expected RTX PRO 6000 Blackwell, got {GPU}")
print("HARDWARE", GPU, torch.cuda.get_device_capability(0), torch.__version__, torch.version.cuda)

csv_path = find_one("im2gps3k_places365.csv")
image_dir = csv_path.parent / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame["name"].map(lambda name: str(image_dir / name))
missing = frame.loc[~frame.image_path.map(lambda value: Path(value).is_file()), "name"].tolist()
if missing:
    raise RuntimeError(f"Missing images: {missing[:5]}")
print("DATASET", len(frame), "images")

checkpoint = find_one("wedetect_base_uni.pth")
models_py = find_one("wedetect_anything/models.py")
sys.path.insert(0, str(models_py.parent))
from models import SimpleYOLOWorldProposalDetector, VisionProposalDeployModel, load_proposal_checkpoint  # noqa: E402

detector = SimpleYOLOWorldProposalDetector("base", prompt_dim=768, num_prompts=256)
load_proposal_checkpoint(detector, str(checkpoint))
deploy = VisionProposalDeployModel(detector, img_size=640)


class FastProposalModel(nn.Module):
    """The official deploy forward without returning unused 8400x768 embeds."""

    def __init__(self, source):
        super().__init__()
        for name in ("backbone", "neck", "cls_preds", "reg_preds", "cls_contrasts"):
            setattr(self, name, getattr(source, name))
        self.reg_max = source.reg_max
        self.num_levels = source.num_levels
        self.register_buffer("embeddings", source.embeddings)
        self.register_buffer("proj", source.proj)
        self.register_buffer("points", source.points)
        self.register_buffer("strides", source.strides)

    def forward(self, image):
        features = self.neck(self.backbone(image))
        score_parts, box_parts = [], []
        for level in range(self.num_levels):
            feature = features[level]
            batch, _, height, width = feature.shape
            contrast = self.cls_contrasts[level]
            embedding = contrast.norm(self.cls_preds[level](feature))
            logits = torch.einsum("bchw,kc->bkhw", embedding, self.embeddings)
            logits = logits * contrast.logit_scale.exp() + contrast.bias
            regression = self.reg_preds[level](feature)
            regression = regression.reshape(-1, 4, self.reg_max, height * width).permute(0, 3, 1, 2)
            regression = regression.softmax(3).matmul(self.proj).squeeze(-1)
            score_parts.append(logits.permute(0, 2, 3, 1).reshape(batch, height * width, -1))
            box_parts.append(regression)
        scores = torch.cat(score_parts, dim=1).sigmoid()
        distances = torch.cat(box_parts, dim=1)
        boxes = torch.cat(
            [self.points - distances[..., :2] * self.strides, self.points + distances[..., 2:] * self.strides],
            dim=-1,
        )
        return scores, boxes


model = FastProposalModel(deploy).to(DEVICE).eval().to(memory_format=torch.channels_last)
del detector, deploy


class ImageDataset(Dataset):
    def __len__(self):
        return len(frame)

    def __getitem__(self, index):
        row = frame.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        ratio = min(640 / height, 640 / width)
        resized_w, resized_h = round(width * ratio), round(height * ratio)
        resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        dw, dh = (640 - resized_w) / 2, (640 - resized_h) / 2
        left, top = round(dw - 0.1), round(dh - 0.1)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = np.asarray(resized)
        tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float().div_(255)
        return tensor, row["name"], width, height, ratio, dw, dh


loader = DataLoader(
    ImageDataset(), batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    pin_memory=True, persistent_workers=True, prefetch_factor=3,
)

records = []
started = time.perf_counter()
with torch.inference_mode(), JSONL.open("w", encoding="utf-8") as output:
    for batch_index, (images, names, widths, heights, ratios, dws, dhs) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores, boxes = model(images)
            objectness = scores.amax(dim=-1).float()
        boxes = boxes.float()
        for item in range(len(names)):
            valid = torch.where(objectness[item] >= CACHE_THRESHOLD)[0]
            if len(valid):
                keep = nms(boxes[item, valid], objectness[item, valid], NMS_IOU)[:TOP_K]
                indices = valid[keep]
                selected_boxes = boxes[item, indices].clone()
                selected_scores = objectness[item, indices]
                selected_boxes[:, [0, 2]] = ((selected_boxes[:, [0, 2]] - dws[item].to(DEVICE)) / ratios[item].to(DEVICE)).clamp(0, widths[item].item())
                selected_boxes[:, [1, 3]] = ((selected_boxes[:, [1, 3]] - dhs[item].to(DEVICE)) / ratios[item].to(DEVICE)).clamp(0, heights[item].item())
                boxes_list = selected_boxes.cpu().tolist()
                scores_list = selected_scores.cpu().tolist()
            else:
                boxes_list, scores_list = [], []
            proposals = [
                {"box": box, "score": score}
                for box, score in zip(boxes_list, scores_list)
                if box[2] > box[0] and box[3] > box[1]
            ]
            record = {
                "name": names[item], "width": int(widths[item]), "height": int(heights[item]),
                "lat": float(frame.iloc[len(records)].LAT), "lon": float(frame.iloc[len(records)].LON),
                "proposals": proposals,
            }
            records.append(record)
            output.write(json.dumps(record) + "\n")
        output.flush()
        done = len(records)
        if done % 256 < BATCH_SIZE or done == len(frame):
            print(f"CACHE {done}/{len(frame)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

if [record["name"] for record in records] != frame.name.tolist():
    raise RuntimeError("Cache order/completeness validation failed")
counts_by_threshold = {
    str(threshold): [sum(item["score"] >= threshold for item in record["proposals"]) for record in records]
    for threshold in (0.15, 0.20, 0.30, 0.40, 0.50)
}
summary = {
    "dataset": "Img2GPS3K", "images": len(records), "gpu": GPU,
    "cache_threshold": CACHE_THRESHOLD, "nms_iou": NMS_IOU, "top_k": TOP_K,
    "elapsed_seconds": time.perf_counter() - started,
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "counts": {
        threshold: {
            "total": int(np.sum(counts)), "mean": float(np.mean(counts)), "median": float(np.median(counts)),
            "max": int(np.max(counts)), "images_with_proposals": int(np.sum(np.asarray(counts) > 0)),
        }
        for threshold, counts in counts_by_threshold.items()
    },
}
(OUTPUT / "cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("CACHE_COMPLETE")
print(json.dumps(summary, indent=2))
