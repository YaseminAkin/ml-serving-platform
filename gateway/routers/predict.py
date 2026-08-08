"""POST /predict endpoint."""

from fastapi import APIRouter, Depends

from gateway.auth import require_api_key
from gateway.models.model import predict
from gateway.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: PredictRequest, _: None = Depends(require_api_key)
) -> PredictResponse:
    """Run sentiment prediction on the given text."""
    result = predict(request.text)

    return PredictResponse(
        label=result["label"],
        score=result["score"],
        model_version="v1",
        cached=False,
        latency_ms=result["latency_ms"],
    )
