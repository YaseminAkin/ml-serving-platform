"""Pydantic request/response schemas for the gateway API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    label: str
    score: float
    model_version: str
    cached: bool
    latency_ms: float
