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
