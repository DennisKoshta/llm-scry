from fastapi import APIRouter, HTTPException

from ..engine import engine
from ..schemas import ModelInfo, ModelLoadRequest

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/load", response_model=ModelInfo)
def load_model(req: ModelLoadRequest) -> ModelInfo:
    try:
        return engine.load(req.name, req.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model load failed: {e}") from e


@router.get("/info", response_model=ModelInfo)
def get_info() -> ModelInfo:
    if engine.model is None:
        raise HTTPException(status_code=404, detail="no model loaded")
    return engine.info()
