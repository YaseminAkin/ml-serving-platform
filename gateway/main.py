"""FastAPI application entry point for the ML model serving gateway."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from db.session import SessionLocal
from gateway import cache
from gateway.metrics import sync_active_model_version_gauge
from gateway.routers import eval as eval_router
from gateway.routers import models, predict
from gateway.schemas import CacheMetricsResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with SessionLocal() as db:
        await sync_active_model_version_gauge(db)
    yield


app = FastAPI(title="ML Serving Gateway", version="0.1.0", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(models.router)
app.include_router(eval_router.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}


@app.get("/metrics/cache", response_model=CacheMetricsResponse)
async def cache_metrics() -> dict:
    """Return prediction cache hit-rate metrics."""
    return await cache.get_cache_stats()


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose metrics in Prometheus text exposition format for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
