import io
import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import geo_utils, intervention, wedetect_proposals
from .geoclip_loader import get_model, predict_cached, warm_gallery_cache
from .schemas import (
    ModelInfo,
    PredictionItem,
    PredictResponse,
    ProposalEvaluationItem,
    ProposalEvaluationResponse,
    ProposalItem,
    ProposalResponse,
    RunResult,
)

_state = None  # intervention.InterventionState, set once the model finishes loading
# ponytail: single global lock serializes requests against the shared
# InterventionState instead of making it request-scoped; fine for a local
# single-user sandbox tool, revisit with per-request state if this ever
# serves concurrent users.
_predict_lock = threading.Lock()

# Model loading (HF download + weight load, ~1-2min on first run) happens in
# a background thread instead of blocking FastAPI's startup, so /health is
# reachable immediately and the UI can show "loading" instead of looking dead.
_status_lock = threading.Lock()
_status = {"state": "loading", "detail": None}


def _set_status(state: str, detail: str | None = None):
    with _status_lock:
        _status["state"] = state
        _status["detail"] = detail


def _load_model_background():
    global _state
    try:
        model = get_model()  # triggers HF download of CLIP ViT-L/14 on first run
        _state = intervention.patch_vision_tower(model.image_encoder.CLIP)
        warm_gallery_cache(model)  # pay the 100K-point gallery pass once here, not on the first request
        _set_status("ready")
    except Exception as e:  # noqa: BLE001 -- surface any load failure to the UI
        _set_status("error", str(e))


def _require_ready():
    with _status_lock:
        state, detail = _status["state"], _status["detail"]
    if state != "ready":
        raise HTTPException(503, detail or "Model chưa sẵn sàng (đang tải/khởi tạo)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model_background, daemon=True).start()
    yield


app = FastAPI(title="GeoCLIP attention-intervention sandbox", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Polled by the frontend status badge: loading -> ready, or busy while
    a /predict call holds `_predict_lock`, or error with a detail message."""
    with _status_lock:
        state, detail = _status["state"], _status["detail"]
    if state == "ready" and _predict_lock.locked():
        state = "busy"
    return {"state": state, "detail": detail}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    _require_ready()
    model = get_model()
    clip = model.image_encoder.CLIP
    cfg = clip.vision_model.config
    return ModelInfo(
        num_layers=intervention.num_layers(clip),
        grid_size=intervention.grid_size(clip),
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
    )


@app.get("/proposal-info")
def proposal_info():
    """Report whether the optional visual-only WeDetect-Uni export is ready."""
    return wedetect_proposals.get_proposer().info()


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            return source.convert("RGB")
    except Exception as exc:  # noqa: BLE001 -- Pillow exposes several decode errors
        raise HTTPException(422, f"Invalid image: {exc}") from exc


def _prepare_proposals(
    pil_image: Image.Image,
    score_threshold: float,
    iou_threshold: float,
    top_k: int,
):
    """Generate boxes and collapse boxes equivalent on GeoCLIP's patch grid."""
    proposer = wedetect_proposals.get_proposer()
    try:
        raw = proposer.generate(
            pil_image,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
    except wedetect_proposals.WeDetectUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    model = get_model()
    grid = intervention.grid_size(model.image_encoder.CLIP)
    width, height = pil_image.size
    unique_masks: set[bytes] = set()
    items: list[ProposalItem] = []
    masks = []

    for proposal in raw:
        region = proposal.normalized_region(width, height)
        patch_mask = intervention.build_in_region_mask([region], width, height, grid)
        patch_count = int(patch_mask.sum().item())
        if patch_count == 0:
            continue
        signature = patch_mask.numpy().tobytes()
        if signature in unique_masks:
            continue
        unique_masks.add(signature)
        items.append(
            ProposalItem(
                index=len(items),
                region=region,
                objectness=proposal.score,
                patch_count=patch_count,
            )
        )
        masks.append(patch_mask)

    return proposer, raw, items, masks, grid


@app.post("/proposals", response_model=ProposalResponse)
def proposals(
    image: UploadFile = File(...),
    score_threshold: float = Form(wedetect_proposals.DEFAULT_SCORE_THRESHOLD),
    iou_threshold: float = Form(wedetect_proposals.DEFAULT_IOU_THRESHOLD),
    top_k: int = Form(50),
):
    """Generate prompt-free WeDetect-Uni boxes and deduplicate intervention masks.

    Deduplication happens after mapping each box through CLIP's resize and
    center-crop.  Pixel-space boxes that activate the same GeoCLIP patch mask
    are equivalent for this experiment, so only the highest-objectness one is
    returned.
    """
    _require_ready()
    pil_image = _decode_image(image.file.read())
    proposer, raw, items, _masks, grid = _prepare_proposals(
        pil_image, score_threshold, iou_threshold, top_k
    )

    return ProposalResponse(
        proposals=items,
        raw_proposal_count=len(raw),
        unique_patch_mask_count=len(items),
        grid_size=grid,
        provider=proposer.providers[0] if proposer.providers else None,
    )


def _run(model, image_path: str, top_k: int, ground_truth: dict | None) -> RunResult:
    top_gps, top_prob = predict_cached(model, image_path, top_k)
    predictions = [
        PredictionItem(lat=float(gps[0]), lon=float(gps[1]), prob=float(p))
        for gps, p in zip(top_gps, top_prob)
    ]
    result = RunResult(predictions=predictions)
    if ground_truth is not None and predictions:
        top1 = predictions[0]
        dist = geo_utils.distance_km(top1.lat, top1.lon, ground_truth["lat"], ground_truth["lon"])
        result.top1_distance_km = dist
        result.threshold_hits = geo_utils.threshold_hits(dist)
    return result


@app.post("/evaluate-proposals", response_model=ProposalEvaluationResponse)
def evaluate_proposals(
    image: UploadFile = File(...),
    layer_configs: str = Form("{}"),
    ground_truth: str | None = Form(None),
    top_k: int = Form(5),
    score_threshold: float = Form(wedetect_proposals.DEFAULT_SCORE_THRESHOLD),
    iou_threshold: float = Form(wedetect_proposals.DEFAULT_IOU_THRESHOLD),
    proposal_top_k: int = Form(20),
):
    """Run one independent GeoCLIP intervention for every WeDetect proposal.

    Proposal masks are deliberately not unioned. The baseline is computed once,
    then the same attention configuration is evaluated with each proposal mask.
    """
    _require_ready()
    try:
        layer_configs_parsed = {int(k): tuple(v) for k, v in json.loads(layer_configs).items()}
        ground_truth_parsed = json.loads(ground_truth) if ground_truth else None
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(422, f"Malformed layer_configs/ground_truth: {exc}") from exc

    image_bytes = image.file.read()
    pil_image = _decode_image(image_bytes)
    proposer, raw, items, masks, grid = _prepare_proposals(
        pil_image, score_threshold, iou_threshold, proposal_top_k
    )
    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        with _predict_lock:
            _state.layer_ab = {}
            _state.in_region_mask = None
            baseline = _run(model, tmp_path, top_k, ground_truth_parsed)

            evaluations = []
            for item, patch_mask in zip(items, masks):
                _state.layer_ab = layer_configs_parsed
                _state.in_region_mask = patch_mask
                run = _run(model, tmp_path, top_k, ground_truth_parsed)
                improvement = None
                if baseline.top1_distance_km is not None and run.top1_distance_km is not None:
                    improvement = baseline.top1_distance_km - run.top1_distance_km
                evaluations.append(
                    ProposalEvaluationItem(
                        proposal=item,
                        intervention=run,
                        distance_improvement_km=improvement,
                    )
                )
    finally:
        if _state is not None:
            _state.layer_ab = {}
            _state.in_region_mask = None
        os.unlink(tmp_path)

    return ProposalEvaluationResponse(
        baseline=baseline,
        evaluations=evaluations,
        raw_proposal_count=len(raw),
        unique_patch_mask_count=len(items),
        grid_size=grid,
        provider=proposer.providers[0] if proposer.providers else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    image: UploadFile = File(...),
    regions: str = Form("[]"),
    layer_configs: str = Form("{}"),
    ground_truth: str | None = Form(None),
    top_k: int = Form(5),
):
    # Sync `def`, not `async def`: model.predict() below is a long, blocking,
    # CPU-bound call. FastAPI runs sync endpoints in a worker thread
    # (starlette's threadpool) instead of on the asyncio event loop, so a
    # slow inference here doesn't freeze /health or any other request --
    # an `async def` version of this endpoint blocked the whole server for
    # the full duration of every predict call.
    _require_ready()
    try:
        regions_parsed = json.loads(regions)
        layer_configs_parsed = {int(k): tuple(v) for k, v in json.loads(layer_configs).items()}
        ground_truth_parsed = json.loads(ground_truth) if ground_truth else None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(422, f"Malformed regions/layer_configs/ground_truth: {e}")

    image_bytes = image.file.read()
    orig_w, orig_h = Image.open(io.BytesIO(image_bytes)).size

    model = get_model()
    grid = intervention.grid_size(model.image_encoder.CLIP)
    in_region_mask = intervention.build_in_region_mask(regions_parsed, orig_w, orig_h, grid)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        with _predict_lock:
            _state.layer_ab = {}
            _state.in_region_mask = None
            baseline = _run(model, tmp_path, top_k, ground_truth_parsed)

            _state.layer_ab = layer_configs_parsed
            _state.in_region_mask = in_region_mask
            intervened = _run(model, tmp_path, top_k, ground_truth_parsed)

            _state.layer_ab = {}
            _state.in_region_mask = None
    finally:
        os.unlink(tmp_path)

    return PredictResponse(
        baseline=baseline,
        intervention=intervened,
        in_region_patch_count=int(in_region_mask.sum().item()),
        grid_size=grid,
    )
