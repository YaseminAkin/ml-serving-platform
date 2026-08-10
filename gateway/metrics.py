"""Prometheus metrics for the gateway."""

from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ModelStatus, ModelVersion

request_count = Counter(
    "request_count",
    "Total number of prediction requests",
    ["model_version", "status_code"],
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds (cache hits are not inference)",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

cache_hit_total = Counter(
    "cache_hit_total",
    "Total number of prediction cache hits",
)

active_model_version = Gauge(
    "active_model_version",
    "1 for the currently active (production) model version, 0 otherwise",
    ["model_version"],
)


async def sync_active_model_version_gauge(db: AsyncSession) -> None:
    """Resync active_model_version to match the registry's current state.

    A gauge can't say *which* version is active on its own, so every
    registered version gets its own labelled series set to 1 or 0 —
    filtering on `active_model_version == 1` finds the live one.
    """
    result = await db.execute(select(ModelVersion))
    for version in result.scalars().all():
        active_model_version.labels(model_version=version.version).set(
            1 if version.status == ModelStatus.PRODUCTION else 0
        )
