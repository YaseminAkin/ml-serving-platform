"""Eval-related endpoints: drift alert history and the eval API surface."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from db.models import DriftAlert, EvalResult, Prediction
from db.session import get_db
from eval import drift
from eval.scorer import CONFIDENCE_METHOD, LLM_JUDGE_METHOD
from gateway.auth import require_api_key
from gateway.schemas import AlertResponse, EvalPredictionEntry, EvalSummaryEntry

router = APIRouter()


@router.get("/alerts", response_model=list[AlertResponse])
async def list_alerts(
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> list[DriftAlert]:
    """Return drift alert history, newest first."""
    result = await db.execute(
        select(DriftAlert).order_by(DriftAlert.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


@router.get("/eval/summary", response_model=list[EvalSummaryEntry])
async def eval_summary(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> list[EvalSummaryEntry]:
    """Per-version prediction volume, average confidence, and drift status."""
    counts_result = await db.execute(
        select(Prediction.model_version, func.count(Prediction.id)).group_by(
            Prediction.model_version
        )
    )
    counts = dict(counts_result.all())

    avg_result = await db.execute(
        select(Prediction.model_version, func.avg(EvalResult.score))
        .join(EvalResult, EvalResult.prediction_id == Prediction.id)
        .where(EvalResult.method == CONFIDENCE_METHOD)
        .group_by(Prediction.model_version)
    )
    avg_scores = dict(avg_result.all())

    entries = []
    for model_version, prediction_count in counts.items():
        status, *_ = await drift.get_drift_status(db, model_version)
        entries.append(
            EvalSummaryEntry(
                model_version=model_version,
                prediction_count=prediction_count,
                avg_confidence_score=avg_scores.get(model_version),
                drift_status=status,
            )
        )
    return entries


@router.get("/eval/predictions", response_model=list[EvalPredictionEntry])
async def eval_predictions(
    version: str,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(require_api_key),
) -> list[EvalPredictionEntry]:
    """Paginated prediction browser with confidence and LLM-judge scores."""
    confidence = aliased(EvalResult)
    judge = aliased(EvalResult)

    result = await db.execute(
        select(Prediction, confidence.score, judge.score)
        .outerjoin(
            confidence,
            (confidence.prediction_id == Prediction.id)
            & (confidence.method == CONFIDENCE_METHOD),
        )
        .outerjoin(
            judge,
            (judge.prediction_id == Prediction.id) & (judge.method == LLM_JUDGE_METHOD),
        )
        .where(Prediction.model_version == version)
        .order_by(Prediction.id.desc())
        .limit(limit)
        .offset(offset)
    )

    return [
        EvalPredictionEntry(
            id=prediction.id,
            model_version=prediction.model_version,
            text=prediction.text,
            label=prediction.output["label"],
            score=prediction.output["score"],
            cached=prediction.cached,
            timestamp=prediction.timestamp,
            confidence_score=confidence_score,
            llm_judge_score=llm_judge_score,
        )
        for prediction, confidence_score, llm_judge_score in result.all()
    ]
