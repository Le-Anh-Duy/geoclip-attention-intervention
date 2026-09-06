# GeoCLIP attention-intervention sandbox

Sandbox for testing whether biasing CLIP's vision-tower attention toward or
away from a chosen image region shifts GeoCLIP's predicted GPS. Regions can be
drawn manually or generated without a text prompt by WeDetect-Base-Uni.

## Layout

```
geo-clip/          # vendored GeoCLIP source (unmodified) — model + weights + gallery
backend/           # FastAPI server, loads GeoCLIP once, applies the intervention
frontend/          # React (Vite) UI: upload image, draw regions, tune per-layer a/b
```

`geo-clip/` is included in the parent experiment repository so this sandbox can
run from the checked-out workspace without a second clone.

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

## Optional WeDetect-Base-Uni proposals

Install the backend requirements, then download the official Base-Uni
checkpoint and export its visual-only proposal graph to ONNX:

```
cd backend
python scripts/setup_wedetect_uni.py
```

The generated model is stored outside the repository at
`~/.cache/wedetect/wedetect_anything_base.onnx`. To use another export, set:

```
WEDETECT_UNI_ONNX=/absolute/path/to/wedetect_anything_base.onnx
```

The UI button **Sinh proposal bằng WeDetect-Uni** calls `POST /proposals`.
Returned boxes are deduplicated after projection to GeoCLIP's patch grid;
click a blue box to include it in the next intervention run. Manual regions
remain available and can be combined with selected proposals.
The default WeDetect objectness threshold is `0.4` and the NMS IoU threshold
is `0.7`.

**Đánh giá tất cả proposal độc lập** calls `POST /evaluate-proposals`. It runs
the baseline once, then applies the current attention configuration to each
retained proposal separately; proposal boxes are never unioned in this batch.
When ground-truth coordinates are supplied, results are ranked by top-1
distance improvement. The proposal limit is exposed in the UI (20 by default,
up to 1000) because a run costs one GeoCLIP inference per unique patch mask.

Only the prompt-free visual proposal stage is loaded. The WeDetect text tower,
large vocabulary, and WeDetect-Ref are not part of this pipeline. The upstream
WeDetect source and weights are GPL-v3 and remain in the user cache.

## The intervention, and where it lives

Agreed formula: add a per-key bias to `QK^T / sqrt(d)` before softmax — `a`
for patches inside a user-selected region and `b` outside it. This happens
independently per layer (24 layers in ViT-L/14). Positive bias attracts
attention, negative bias repels it, and `a = b = 0` is a no-op.

- **The pre-softmax bias step**: `backend/app/intervention.py`
  (`_intervened_attention_forward`, using `key_bias_for_layer`).
- **How `a`/`b` are picked per layer, and the CLS-token exemption**:
  `backend/app/intervention.py` (`InterventionState.key_bias_for_layer`).
  Position 0 (CLS) is hardcoded to bias 0 always — it isn't a spatial patch,
  so "inside/outside region" doesn't apply to it.
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
  `InterventionState.layer_ab`. These values are always stored as logits
  biases. The UI's Scale view displays the exact equivalent `exp(bias)`.
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
