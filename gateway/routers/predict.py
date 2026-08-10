"""POST /predict endpoint."""

import hashlib
import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Prediction
from db.session import get_db
from gateway import cache, metrics
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
    """Run sentiment prediction, using cache, logging, and shadow mirroring."""
    input_hash = hashlib.sha256(request.text.encode()).hexdigest()

    production = await get_production_model_version(db)
    model_version = production.version if production else DEFAULT_MODEL_VERSION

    start = time.perf_counter()
    cached_result = await cache.get_cached_prediction(model_version, input_hash)
    is_cached = cached_result is not None

    if is_cached:
        result = {**cached_result, "latency_ms": (time.perf_counter() - start) * 1000}
        metrics.cache_hit_total.inc()
    else:
        result = predict(request.text)
        await cache.set_cached_prediction(
            model_version,
            input_hash,
            {"label": result["label"], "score": result["score"]},
        )
        metrics.inference_latency_seconds.observe(result["latency_ms"] / 1000)

    metrics.request_count.labels(model_version=model_version, status_code="200").inc()

    db.add(
        Prediction(
            model_version=model_version,
            input_hash=input_hash,
            text=request.text,
            output={"label": result["label"], "score": result["score"]},
            latency_ms=result["latency_ms"],
            cached=is_cached,
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
        cached=is_cached,
        latency_ms=result["latency_ms"],
    )
