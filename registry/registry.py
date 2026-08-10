"""Model version registry — register, list, and promote model versions."""

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ModelStatus, ModelVersion


class DuplicateModelVersionError(Exception):
    """Raised when a (name, version) pair is already registered."""


class ModelVersionNotFoundError(Exception):
    """Raised when a model version id does not exist."""


async def create_model_version(
    db: AsyncSession, name: str, version: str, path: str
) -> ModelVersion:
    """Register a new model version with status=staging."""
    model = ModelVersion(
        name=name, version=version, path=path, status=ModelStatus.STAGING
    )
    db.add(model)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateModelVersionError(
            f"Model version '{name}:{version}' is already registered"
        ) from exc
    await db.refresh(model)
    return model


async def list_model_versions(db: AsyncSession) -> list[ModelVersion]:
    """List all registered model versions, newest first."""
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    )
    return list(result.scalars().all())


async def promote_model_version(db: AsyncSession, model_id: int) -> ModelVersion:
    """Promote a model version to production, demoting the current one.

    Only one version may hold status=production at a time — enforced here
    by demoting the incumbent, and backstopped by a partial unique index
    on the DB (see db/models.py) in case of a race between concurrent
    promote calls.
    """
    target = await db.get(ModelVersion, model_id)
    if target is None:
        raise ModelVersionNotFoundError(f"Model version {model_id} not found")

    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.status == ModelStatus.PRODUCTION)
        .with_for_update()
    )
    current_production = result.scalar_one_or_none()
    if current_production is not None and current_production.id != target.id:
        current_production.status = ModelStatus.ARCHIVED

    target.status = ModelStatus.PRODUCTION

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateModelVersionError(
            "Another version was promoted to production concurrently"
        ) from exc

    await db.refresh(target)
    return target
