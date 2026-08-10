"""Redis-backed response cache for inference results."""

import json
import logging
import os

import redis.asyncio as redis

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_SECONDS = 3600

_HITS_KEY = "cache:hits"
_MISSES_KEY = "cache:misses"

_redis = redis.from_url(REDIS_URL, decode_responses=True)


def _prediction_key(model_version: str, input_hash: str) -> str:
    # Scoped by model_version so promoting a new version can't serve stale
    # predictions cached under the old one for the same input text.
    return f"prediction:{model_version}:{input_hash}"


async def get_cached_prediction(model_version: str, input_hash: str) -> dict | None:
    """Look up a cached prediction, recording a hit or miss. None if unavailable."""
    try:
        raw = await _redis.get(_prediction_key(model_version, input_hash))
        await _redis.incr(_HITS_KEY if raw is not None else _MISSES_KEY)
    except redis.RedisError:
        logger.exception("cache lookup failed, treating as miss")
        return None

    return json.loads(raw) if raw is not None else None


async def set_cached_prediction(
    model_version: str, input_hash: str, result: dict
) -> None:
    """Cache a prediction result with a 1-hour TTL. Failures are swallowed."""
    try:
        await _redis.setex(
            _prediction_key(model_version, input_hash),
            CACHE_TTL_SECONDS,
            json.dumps(result),
        )
    except redis.RedisError:
        logger.exception("cache write failed")


async def get_cache_stats() -> dict:
    """Return hit/miss counters and the derived hit rate."""
    try:
        hits = int(await _redis.get(_HITS_KEY) or 0)
        misses = int(await _redis.get(_MISSES_KEY) or 0)
    except redis.RedisError:
        logger.exception("cache stats lookup failed")
        hits = misses = 0

    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    return {"hit_rate": hit_rate, "total_hits": hits, "total_misses": misses}
