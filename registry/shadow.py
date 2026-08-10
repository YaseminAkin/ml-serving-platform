"""Shadow mode: mirror live predictions to a shadow model version.

The shadow call runs fire-and-forget via asyncio.create_task() so it never
adds latency to the caller's response, and it must never raise into the
request path — any failure is logged and swallowed.
"""

import asyncio
import logging
import os

from db.models import Prediction
from db.session import SessionLocal
from gateway.models.model import predict

logger = logging.getLogger(__name__)


def is_shadow_enabled() -> bool:
    """Whether shadow mode is on, read fresh from env on every call."""
    return os.environ.get("SHADOW_ENABLED", "false").lower() == "true"


# asyncio only holds a weak reference to tasks created via create_task; keep
# a strong reference here so they aren't garbage-collected mid-flight.
_background_tasks: set = set()


def schedule_shadow_prediction(text: str, input_hash: str, shadow_version: str) -> None:
    """Fire off a shadow prediction in the background, if shadow mode is on."""
    task = asyncio.create_task(_run_shadow_prediction(text, input_hash, shadow_version))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_shadow_prediction(
    text: str, input_hash: str, shadow_version: str
) -> None:
    """Run inference against the shadow model and log the result."""
    try:
        result = predict(text)
        async with SessionLocal() as db:
            db.add(
                Prediction(
                    model_version=shadow_version,
                    input_hash=input_hash,
                    output={"label": result["label"], "score": result["score"]},
                    latency_ms=result["latency_ms"],
                    cached=False,
                )
            )
            await db.commit()
    except Exception:
        logger.exception("shadow prediction failed for version=%s", shadow_version)
