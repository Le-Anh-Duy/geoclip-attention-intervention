# GeoCLIP attention-intervention sandbox

Manual-region sandbox for testing whether scaling CLIP's vision-tower
attention toward/away from a chosen image region shifts GeoCLIP's predicted
GPS. Stands in for Grounding DINO boxes until that's wired up — regions are
drawn by hand in the UI instead of detected.

## Layout

```
geo-clip/          # vendored GeoCLIP source (unmodified) — model + weights + gallery
backend/           # FastAPI server, loads GeoCLIP once, applies the intervention
frontend/          # React (Vite) UI: upload image, draw regions, tune per-layer a/b
```

`geo-clip/` is git-ignored here (weights + GPS gallery are data, not code) — clone it
separately before running the backend:

```
git clone https://github.com/Le-Anh-Duy/geo-clip
```

## Running it

**Backend** (downloads CLIP ViT-L/14, ~1.7GB, on first run — cached under
`~/.cache/huggingface` after that):

```
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

**Frontend**:

```
cd frontend
npm run dev
```

Open `http://localhost:5173`. Backend must be reachable at
`http://127.0.0.1:8000` (hardcoded in `frontend/src/api.js`).

## The intervention, and where it lives

Agreed formula: after `softmax(QK^T / sqrt(d))` produces attention
probabilities, scale the probability every query token assigns to each KEY
patch — `a` if that patch is inside a user-selected region, `b` if it isn't —
then renormalize each row back to sum 1. This happens *before* the result is
multiplied by V, and independently per layer (24 layers in ViT-L/14, each
with its own `a`/`b`). `a = b = 1` is a no-op.

- **The actual scale-and-renormalize step**: `backend/app/intervention.py:78-83`
  (`_intervened_attention_forward`, inside `key_scale_for_layer` +
  the multiply/renormalize lines right after it's called).
- **How `a`/`b` are picked per layer, and the CLS-token exemption**:
  `backend/app/intervention.py:39-50` (`InterventionState.key_scale_for_layer`).
  Position 0 (CLS) is hardcoded to scale 1 always — it isn't a spatial patch,
  so "inside/outside region" doesn't apply to it. Change line 48 if you want
  to experiment with scaling CLS too.
- **Region → patch-grid mapping** (drawn rectangle → which of the 256 patch
  tokens count as "in region"): `backend/app/intervention.py:118-154`
  (`build_in_region_mask`). This replicates CLIP's own preprocessing
  (resize-shortest-side-to-224 + center-crop-224) on the region coordinates,
  so a patch is only marked "in region" if it actually is post-preprocessing.
- **Binding the patched forward onto all 24 layers**:
  `backend/app/intervention.py:93-105` (`patch_vision_tower`), called once at
  server startup from `backend/app/main.py` (`lifespan`).
- **Per-request wiring** (baseline run vs. intervention run, both against the
  same loaded model): `backend/app/main.py` (`predict` endpoint) — sets
  `_state.layer_ab` / `_state.in_region_mask` before each of the two calls to
  `model.predict(...)`, and resets them after.

## UI ↔ backend config mapping

- Drawn rectangles in `ImageRegionSelector.jsx` → `regions: [{x,y,w,h}]`
  (fractions of the original image) → `build_in_region_mask`.
- Each row in `LayerControls.jsx` → `layer_configs: {layerIdx: [a, b]}` →
  `InterventionState.layer_ab`.
- Ground-truth lat/lon inputs in `App.jsx` → `ground_truth: {lat, lon}` →
  compared against top-1 prediction via `backend/app/geo_utils.py`
  (Haversine/geodesic distance + the 1/25/200/750/2500 km threshold hits used
  elsewhere in this project's evaluation).

## Known simplifications (sandbox scope, not production)

- `backend/app/main.py`: one global lock serializes all `/predict` calls
  against the shared `InterventionState` — fine for one person testing
  locally, would need per-request state for concurrent users.
- No auth, no HTTPS, CORS locked to `localhost:5173` only.
- Region intervention targets the **image encoder** only (`vision_model`).
  The location encoder (GPS → embedding) is untouched — see project notes on
  why that's out of scope.
