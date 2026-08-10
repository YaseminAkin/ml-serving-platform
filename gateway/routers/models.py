"""Model registry endpoints: register, list, and promote model versions."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ModelVersion
from db.session import get_db
from gateway.auth import require_api_key
from gateway.schemas import ModelCreateRequest, ModelResponse
from registry.registry import (
    DuplicateModelVersionError,
    InvalidStateTransitionError,
    ModelVersionNotFoundError,
    create_model_version,
    list_model_versions,
    promote_model_version,
    set_shadow_model_version,
)

router = APIRouter()


@router.post(
    "/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED
)
async def register_model(
    request: ModelCreateRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> ModelVersion:
    """Register a new model version with status=staging."""
    try:
        return await create_model_version(
            db, request.name, request.version, request.path
        )
    except DuplicateModelVersionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.put("/models/{model_id}/shadow", response_model=ModelResponse)
async def shadow_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> ModelVersion:
    """Mark a model version as the shadow candidate, demoting the incumbent."""
    try:
        return await set_shadow_model_version(db, model_id)
    except ModelVersionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except (DuplicateModelVersionError, InvalidStateTransitionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get("/models", response_model=list[ModelResponse])
async def list_models(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> list[ModelVersion]:
    """List all registered model versions, newest first."""
    return await list_model_versions(db)


@router.put("/models/{model_id}/promote", response_model=ModelResponse)
async def promote_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> ModelVersion:
    """Promote a model version to production, demoting the incumbent."""
    try:
        return await promote_model_version(db, model_id)
    except ModelVersionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except DuplicateModelVersionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
