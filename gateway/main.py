"""FastAPI application entry point for the ML model serving gateway."""

from fastapi import FastAPI

app = FastAPI(title="ML Serving Gateway", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}
