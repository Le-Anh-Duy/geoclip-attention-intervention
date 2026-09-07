"""Generate Kaggle .ipynb files from the reviewable Python sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build(folder: str, script: str, notebook: str, title: str, description: str) -> None:
    source = (ROOT / folder / script).read_text(encoding="utf-8")
    payload = {
        "cells": [
            {
                "id": "overview",
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", description],
            },
            {
                "id": "experiment",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / folder / notebook).write_text(json.dumps(payload, indent=1), encoding="utf-8")


build(
    "geoclip-detector-offline-assets",
    "asset_builder.py",
    "geoclip-detector-offline-assets.ipynb",
    "GeoCLIP detector offline assets",
    "Cloud-side preparation of Grounding DINO, CLIP, GeoCLIP, and WeDetect assets for an Internet-disabled downstream notebook.",
)
build(
    "geoclip-detector-causal-probing",
    "detector_causal_probing.py",
    "geoclip-detector-causal-probing.ipynb",
    "Counterfactual detector agreement for GeoCLIP",
    "Offline G4 experiment comparing Grounding DINO and WeDetect masks through causal attention interventions.",
)
build(
    "wedetect-img2gps3k-cache",
    "cache_wedetect.py",
    "wedetect-img2gps3k-cache.ipynb",
    "WeDetect cache for the complete Img2GPS3K test set",
    "One CUDA pass over all 2,997 images. The reusable cache retains low-threshold NMS proposals so downstream searches never rerun WeDetect.",
)
build(
    "wedetect-geoclip-signal-search",
    "search_signals.py",
    "wedetect-geoclip-signal-search.ipynb",
    "WeDetect-guided GeoCLIP layer, head, and signal search",
    "Full-test offline search over all 24 layers, all 16 attention heads, query scope, proposal dose, and baseline internal signals. Configuration selection uses a deterministic discovery split and reports a locked holdout.",
)
build(
    "wedetect-geoclip-layer-synergy",
    "search_synergy.py",
    "wedetect-geoclip-layer-synergy.ipynb",
    "WeDetect-guided GeoCLIP layer synergy test",
    "Confirmatory full-test search over every combination of discovery-positive layers, with fixed and layer-count-normalized intervention strength. The prior search selects candidates; a locked holdout tests whether combining layers beats the best single layer.",
)
