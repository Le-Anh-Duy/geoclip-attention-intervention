from pydantic import BaseModel


class ModelInfo(BaseModel):
    num_layers: int
    grid_size: int  # patches per side (e.g. 16 -> 256 patch tokens)
    patch_size: int
    image_size: int


class Region(BaseModel):
    x: float
    y: float
    w: float
    h: float


class ProposalItem(BaseModel):
    index: int
    region: Region
    objectness: float
    patch_count: int


class ProposalResponse(BaseModel):
    proposals: list[ProposalItem]
    raw_proposal_count: int
    unique_patch_mask_count: int
    grid_size: int
    provider: str | None = None


class PredictionItem(BaseModel):
    lat: float
    lon: float
    prob: float


class RunResult(BaseModel):
    predictions: list[PredictionItem]
    top1_distance_km: float | None = None
    threshold_hits: dict | None = None


class ProposalEvaluationItem(BaseModel):
    proposal: ProposalItem
    intervention: RunResult
    distance_improvement_km: float | None = None


class ProposalEvaluationResponse(BaseModel):
    baseline: RunResult
    evaluations: list[ProposalEvaluationItem]
    raw_proposal_count: int
    unique_patch_mask_count: int
    grid_size: int
    provider: str | None = None


class PredictResponse(BaseModel):
    baseline: RunResult
    intervention: RunResult
    in_region_patch_count: int
    grid_size: int
