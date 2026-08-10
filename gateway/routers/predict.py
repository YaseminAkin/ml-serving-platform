"""POST /predict endpoint."""

import hashlib

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Prediction
from db.session import get_db
from gateway.auth import require_api_key
from gateway.models.model import predict
from gateway.schemas import PredictRequest, PredictResponse
from registry import shadow
from registry.registry import get_production_model_version, get_shadow_model_version

router = APIRouter()

DEFAULT_MODEL_VERSION = "v1"


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: PredictRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> PredictResponse:
    """Run sentiment prediction, logging it and mirroring to shadow if enabled."""
    result = predict(request.text)
    input_hash = hashlib.sha256(request.text.encode()).hexdigest()

    production = await get_production_model_version(db)
    model_version = production.version if production else DEFAULT_MODEL_VERSION

    db.add(
        Prediction(
            model_version=model_version,
            input_hash=input_hash,
            output={"label": result["label"], "score": result["score"]},
            latency_ms=result["latency_ms"],
            cached=False,
        )
    )
    await db.commit()

    if shadow.is_shadow_enabled():
        shadow_version = await get_shadow_model_version(db)
        if shadow_version is not None:
            shadow.schedule_shadow_prediction(
                request.text, input_hash, shadow_version.version
            )

    return PredictResponse(
        label=result["label"],
        score=result["score"],
        model_version=model_version,
        cached=False,
        latency_ms=result["latency_ms"],
    )
