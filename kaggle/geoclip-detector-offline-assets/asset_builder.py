"""Build cloud-side model assets for the offline detector probing notebook."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download


OUTPUT = Path("/kaggle/working/offline_assets")
OUTPUT.mkdir(parents=True, exist_ok=True)

REVISIONS = {
    "grounding_dino": "IDEA-Research/grounding-dino-base",
    "clip": "openai/clip-vit-large-patch14",
    "geoclip_git": "https://github.com/VicenteVivan/geo-clip.git",
    "wedetect_git": "https://github.com/WeChatCV/WeDetect.git",
    "wedetect_commit": "dd302dba0069ace1b05816bafbc3fa1dbd6aa68c",
    "wedetect_hf": "fushh7/WeDetect",
    "wedetect_hf_commit": "125b98f6807eb3459b57a43497d220ae4096f5c7",
}


def run(command: list[str], cwd: Path | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


print("CUDA", torch.cuda.is_available(), torch.__version__, torch.version.cuda)
started = time.perf_counter()

# Hugging Face snapshots are materialized as ordinary files, not cache symlinks,
# so a downstream Kaggle kernel can consume them with local_files_only=True.
snapshot_download(
    REVISIONS["grounding_dino"],
    local_dir=OUTPUT / "grounding-dino-base",
    allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.py"],
)
snapshot_download(
    REVISIONS["clip"],
    local_dir=OUTPUT / "clip-vit-large-patch14",
    allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
)

geoclip_dir = OUTPUT / "geo-clip"
if not geoclip_dir.exists():
    run(["git", "clone", "--depth", "1", REVISIONS["geoclip_git"], str(geoclip_dir)])
if (geoclip_dir / ".git").exists():
    shutil.rmtree(geoclip_dir / ".git")

# Export WeDetect-Uni once on Kaggle. Only the prompt-free ONNX graph and its
# external tensor data are retained for the offline experiment.
run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
wedetect_source = Path("/kaggle/working/wedetect-source")
if not wedetect_source.exists():
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REVISIONS["wedetect_git"], str(wedetect_source)])
    run(["git", "-C", str(wedetect_source), "sparse-checkout", "set", "wedetect_anything"])
    run(["git", "-C", str(wedetect_source), "fetch", "--depth", "1", "origin", REVISIONS["wedetect_commit"]])
    run(["git", "-C", str(wedetect_source), "checkout", "--detach", "FETCH_HEAD"])

checkpoint = hf_hub_download(
    repo_id=REVISIONS["wedetect_hf"],
    filename="wedetect_base_uni.pth",
    revision=REVISIONS["wedetect_hf_commit"],
    cache_dir="/kaggle/working/hf-cache",
)
wedetect_dir = OUTPUT / "wedetect"
wedetect_dir.mkdir(exist_ok=True)
run(
    [
        sys.executable,
        str(wedetect_source / "wedetect_anything" / "export_onnx.py"),
        "--variant", "base",
        "--checkpoint", checkpoint,
        "--img-size", "640",
        "--device", "cpu",
        "--output-dir", str(wedetect_dir),
    ],
    cwd=wedetect_source,
)

# Keep a compatible wheel as a fallback if a future Kaggle image omits ORT.
wheels = OUTPUT / "wheels"
wheels.mkdir(exist_ok=True)
run([sys.executable, "-m", "pip", "download", "-q", "--only-binary=:all:", "--dest", str(wheels), "onnxruntime-gpu"])

onnx_path = wedetect_dir / "wedetect_anything_base.onnx"
data_path = onnx_path.with_suffix(".onnx.data")
required = [
    onnx_path,
    data_path,
    OUTPUT / "grounding-dino-base" / "config.json",
    OUTPUT / "clip-vit-large-patch14" / "config.json",
    geoclip_dir / "geoclip" / "model" / "weights" / "location_encoder_weights.pth",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"Asset build incomplete: {missing}")

manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "elapsed_seconds": time.perf_counter() - started,
    "sources": REVISIONS,
    "torch": torch.__version__,
    "files": {
        str(path.relative_to(OUTPUT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in required
    },
}
(OUTPUT / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("ASSET_BUILD_COMPLETE")
print(json.dumps(manifest, indent=2))
