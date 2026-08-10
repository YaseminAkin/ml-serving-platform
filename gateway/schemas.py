"""Pydantic request/response schemas for the gateway API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from db.models import ModelStatus


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


class ModelCreateRequest(BaseModel):
    """Request body for POST /models."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ModelResponse(BaseModel):
    """Response body for model registry endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: str
    path: str
    status: ModelStatus
    created_at: datetime
