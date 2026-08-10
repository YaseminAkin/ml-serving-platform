"""FastAPI application entry point for the ML model serving gateway."""

from fastapi import FastAPI

from gateway import cache
from gateway.routers import eval as eval_router
from gateway.routers import models, predict
from gateway.schemas import CacheMetricsResponse

app = FastAPI(title="ML Serving Gateway", version="0.1.0")

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
