"""Async eval worker — polls predictions and writes confidence/LLM-judge scores.

Runs as a separate process from the gateway so eval never sits on the
inference critical path, and can be re-run retroactively over old
predictions since scoring reads from the DB rather than the live request.
"""

import asyncio
import logging
import os

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import EvalResult, Prediction
from db.session import SessionLocal
from eval.scorer import (
    CONFIDENCE_METHOD,
    LLM_JUDGE_METHOD,
    score_confidence,
    score_llm_judge,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = float(os.environ.get("EVAL_POLL_INTERVAL_SECONDS", "5"))
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "20"))


async def _unscored_predictions(db: AsyncSession) -> list[Prediction]:
    """Predictions with no eval_results row at all yet."""
    result = await db.execute(
        select(Prediction)
        .outerjoin(EvalResult, EvalResult.prediction_id == Prediction.id)
        .where(EvalResult.id.is_(None))
        .order_by(Prediction.id)
        .limit(BATCH_SIZE)
    )
    return list(result.scalars().all())


async def _score_prediction(db: AsyncSession, prediction: Prediction) -> None:
    """Write confidence (always) and LLM-judge (if configured) eval scores."""
    db.add(
        EvalResult(
            prediction_id=prediction.id,
            score=score_confidence(prediction),
            method=CONFIDENCE_METHOD,
        )
    )

    llm_score = await score_llm_judge(prediction)
    if llm_score is not None:
        db.add(
            EvalResult(
                prediction_id=prediction.id,
                score=llm_score,
                method=LLM_JUDGE_METHOD,
            )
        )

    await db.commit()


async def run_once(db: AsyncSession) -> int:
    """Score one batch of unscored predictions. Returns how many were scored."""
    predictions = await _unscored_predictions(db)
    for prediction in predictions:
        await _score_prediction(db, prediction)
    return len(predictions)


async def run_forever() -> None:
    """Poll indefinitely, scoring new predictions as they appear."""
    logger.info("eval worker starting, poll interval=%ss", POLL_INTERVAL_SECONDS)
    while True:
        try:
            async with SessionLocal() as db:
                scored = await run_once(db)
            if scored:
                logger.info("scored %d prediction(s)", scored)
        except Exception:
            logger.exception("eval worker poll failed")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_forever())
