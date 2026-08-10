"""Drift detection — compares rolling vs baseline confidence per model version.

Runs on an hourly cadence, separate from eval/worker.py, since drift is a
statistical check over already-scored predictions rather than a per-request
scoring step. The baseline/rolling means are computed with SQL AVG() over
ordered, limited subqueries rather than pulling rows into Python — cheaper,
and it's just two scalars per model version either way.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import DriftAlert, EvalResult, Prediction
from db.session import SessionLocal
from eval.scorer import CONFIDENCE_METHOD

logger = logging.getLogger(__name__)

load_dotenv()

CHECK_INTERVAL_SECONDS = float(os.environ.get("DRIFT_CHECK_INTERVAL_SECONDS", "3600"))
WINDOW_SIZE = int(os.environ.get("DRIFT_WINDOW_SIZE", "500"))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.1"))


async def _model_versions_with_scores(db: AsyncSession) -> list[str]:
    """Distinct model versions that have at least one confidence score."""
    result = await db.execute(
        select(Prediction.model_version)
        .join(EvalResult, EvalResult.prediction_id == Prediction.id)
        .where(EvalResult.method == CONFIDENCE_METHOD)
        .distinct()
    )
    return [row[0] for row in result.all()]


async def _scored_count(db: AsyncSession, model_version: str) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(EvalResult)
        .join(Prediction, EvalResult.prediction_id == Prediction.id)
        .where(
            Prediction.model_version == model_version,
            EvalResult.method == CONFIDENCE_METHOD,
        )
    )
    return result.scalar_one()


async def _windowed_mean(
    db: AsyncSession, model_version: str, *, ascending: bool
) -> float:
    """Mean confidence over the first (ascending) or last WINDOW_SIZE scores."""
    order = Prediction.id.asc() if ascending else Prediction.id.desc()
    subquery = (
        select(EvalResult.score)
        .join(Prediction, EvalResult.prediction_id == Prediction.id)
        .where(
            Prediction.model_version == model_version,
            EvalResult.method == CONFIDENCE_METHOD,
        )
        .order_by(order)
        .limit(WINDOW_SIZE)
        .subquery()
    )
    result = await db.execute(select(func.avg(subquery.c.score)))
    return float(result.scalar_one())


async def _check_model_version(
    db: AsyncSession, model_version: str
) -> DriftAlert | None:
    """Check one model version for drift; returns the alert row if one fired."""
    # Need at least a full window before a baseline is meaningful. Below
    # 2x WINDOW_SIZE the baseline and rolling windows overlap, which just
    # makes the check more conservative until enough data accumulates.
    if await _scored_count(db, model_version) < WINDOW_SIZE:
        return None

    baseline_mean = await _windowed_mean(db, model_version, ascending=True)
    rolling_mean = await _windowed_mean(db, model_version, ascending=False)
    delta = abs(rolling_mean - baseline_mean)

    if delta <= DRIFT_THRESHOLD:
        return None

    alert = DriftAlert(
        model_version=model_version,
        baseline_mean=baseline_mean,
        rolling_mean=rolling_mean,
        delta=delta,
        threshold=DRIFT_THRESHOLD,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    logger.warning(
        "drift detected for model_version=%s: baseline=%.4f rolling=%.4f "
        "delta=%.4f (threshold=%.4f)",
        model_version,
        baseline_mean,
        rolling_mean,
        delta,
        DRIFT_THRESHOLD,
    )
    return alert


async def run_once(db: AsyncSession) -> list[DriftAlert]:
    """Check every model version with scored predictions for drift."""
    alerts = []
    for model_version in await _model_versions_with_scores(db):
        alert = await _check_model_version(db, model_version)
        if alert is not None:
            alerts.append(alert)
    return alerts


async def run_forever() -> None:
    """Check for drift on a fixed interval (default hourly)."""
    logger.info("drift detector starting, interval=%ss", CHECK_INTERVAL_SECONDS)
    while True:
        try:
            async with SessionLocal() as db:
                alerts = await run_once(db)
            if alerts:
                logger.warning("%d drift alert(s) raised this cycle", len(alerts))
        except Exception:
            logger.exception("drift check failed")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_forever())
