"""SQLAlchemy ORM models for gateway database tables."""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class ModelStatus(enum.StrEnum):
    """Lifecycle status of a registered model version."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    SHADOW = "shadow"


class ModelVersion(Base):
    """A registered model artefact (e.g. distilbert v1)."""

    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_models_name_version"),
        Index(
            "uq_models_single_production",
            "status",
            unique=True,
            postgresql_where=text("status = 'production'"),
        ),
        Index(
            "uq_models_single_shadow",
            "status",
            unique=True,
            postgresql_where=text("status = 'shadow'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[ModelStatus] = mapped_column(
        Enum(
            ModelStatus,
            name="model_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=ModelStatus.STAGING,
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )


class Prediction(Base):
    """A single inference request/response, logged for eval and caching."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_input_hash", "input_hash"),
        Index("ix_predictions_model_version", "model_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    output: Mapped[dict] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    timestamp: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )


class EvalResult(Base):
    """An eval score computed for a logged prediction."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id"), nullable=False
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
