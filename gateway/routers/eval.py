"""Eval-related endpoints: drift alert history."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DriftAlert
from db.session import get_db
from gateway.auth import require_api_key
from gateway.schemas import AlertResponse

router = APIRouter()


@router.get("/alerts", response_model=list[AlertResponse])
async def list_alerts(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> list[DriftAlert]:
    """Return drift alert history, newest first."""
    result = await db.execute(
        select(DriftAlert).order_by(DriftAlert.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())
