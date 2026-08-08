"""FastAPI application entry point for the ML model serving gateway."""

from fastapi import FastAPI

from gateway.routers import predict

app = FastAPI(title="ML Serving Gateway", version="0.1.0")

app.include_router(predict.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}
