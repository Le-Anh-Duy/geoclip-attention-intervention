"""Loads the GeoCLIP model straight from the vendored `geo-clip` source tree.

We don't pip-install the geoclip package; we just point sys.path at the
sibling `geo-clip/` checkout and import it. `GeoCLIP(from_pretrained=True)`
downloads the CLIP ViT-L/14 weights from the HF Hub on first run (cached
under ~/.cache/huggingface afterwards) and loads the GeoCLIP-specific
location-encoder / MLP / logit_scale weights that ship in that repo.
"""
import os
import sys
import threading

import torch
import torch.nn.functional as F
from PIL import Image

# Bound PyTorch's intra-op thread pool. Left at its default (= all logical
# cores), every matmul/softmax barrier-syncs across ALL of them; on a
# many-core machine that's also running a browser/editor/etc, a few
# preempted threads stall the whole barrier and single-image CPU inference
# balloons from ~seconds to many minutes. Capping it well under the core
# count keeps each op's sync cheap and leaves cores free for the OS to
# schedule the request-handling threads promptly.
torch.set_num_threads(min(8, os.cpu_count() or 8))

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEOCLIP_SRC = os.path.join(os.path.dirname(_BACKEND_DIR), "geo-clip")

if _GEOCLIP_SRC not in sys.path:
    sys.path.insert(0, _GEOCLIP_SRC)

from geoclip import GeoCLIP  # noqa: E402

_model = None
_model_lock = threading.Lock()
_gallery_location_features = None


def get_model() -> GeoCLIP:
    """Singleton accessor -- the model is loaded once at process start."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model = GeoCLIP(from_pretrained=True)
                model.eval()
                _model = model
    return _model


@torch.no_grad()
def _gallery_features(model: GeoCLIP) -> torch.Tensor:
    """GeoCLIP.predict() re-runs the 100K-row GPS gallery through the
    location-encoder MLP (3x Linear(1024,1024)-stack) on every single call --
    it never changes (depends only on frozen weights, not the query image),
    so that's pure waste, and our /predict calls predict() twice per request
    (baseline + intervention). Caching it once after load turns 2 gallery
    passes per request into 0, which was the single largest cost per
    request, well above the ViT forward pass on one image."""
    global _gallery_location_features
    if _gallery_location_features is None:
        feats = model.location_encoder(model.gps_gallery.to(model.device))
        _gallery_location_features = F.normalize(feats, dim=1)
    return _gallery_location_features


def warm_gallery_cache(model: GeoCLIP) -> None:
    """Call once right after load so the 100K-point gallery pass happens
    during startup (already shown as 'loading' in the UI) instead of
    stalling the first user-triggered /predict call."""
    _gallery_features(model)


@torch.no_grad()
def predict_cached(model: GeoCLIP, image_path: str, top_k: int):
    """Drop-in replacement for GeoCLIP.predict() that reuses the cached
    gallery embedding instead of recomputing it -- same math, same output,
    just skips the redundant location-encoder pass over all 100K points."""
    image = Image.open(image_path)
    image = model.image_encoder.preprocess_image(image).to(model.device)

    image_features = F.normalize(model.image_encoder(image), dim=1)
    location_features = _gallery_features(model)
    logits_per_image = model.logit_scale.exp() * (image_features @ location_features.t())
    probs_per_image = logits_per_image.softmax(dim=-1).cpu()

    top_pred = torch.topk(probs_per_image, top_k, dim=1)
    top_pred_gps = model.gps_gallery[top_pred.indices[0]]
    top_pred_prob = top_pred.values[0]
    return top_pred_gps, top_pred_prob
