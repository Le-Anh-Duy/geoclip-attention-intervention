from pydantic import BaseModel


class ModelInfo(BaseModel):
    num_layers: int
    grid_size: int  # patches per side (e.g. 16 -> 256 patch tokens)
    patch_size: int
    image_size: int


class PredictionItem(BaseModel):
    lat: float
    lon: float
    prob: float


class RunResult(BaseModel):
    predictions: list[PredictionItem]
    top1_distance_km: float | None = None
    threshold_hits: dict | None = None


class PredictResponse(BaseModel):
    baseline: RunResult
    intervention: RunResult
    in_region_patch_count: int
    grid_size: int
