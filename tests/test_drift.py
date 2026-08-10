"""Tests for eval/drift.py and GET /alerts."""

import pytest

from db.models import EvalResult, Prediction
from eval import drift

API_KEY_HEADER = {"X-API-Key": "dev"}


async def _seed_scored_predictions(db_session, model_version, scores):
    """Insert predictions + confidence eval_results in list order (id ascending)."""
    for i, score in enumerate(scores):
        prediction = Prediction(
            model_version=model_version,
            input_hash=f"hash-{model_version}-{i}",
            text=f"text {i}",
            output={"label": "POSITIVE", "score": score},
            latency_ms=1.0,
            cached=False,
        )
        db_session.add(prediction)
        await db_session.flush()
        db_session.add(
            EvalResult(
                prediction_id=prediction.id, score=score, method=drift.CONFIDENCE_METHOD
            )
        )
    await db_session.commit()


@pytest.mark.asyncio
async def test_run_once_skips_below_window_size(db_session, monkeypatch):
    monkeypatch.setattr(drift, "WINDOW_SIZE", 5)
    await _seed_scored_predictions(db_session, "v1", [0.9, 0.9, 0.9])

    alerts = await drift.run_once(db_session)

    assert alerts == []


@pytest.mark.asyncio
async def test_run_once_no_alert_within_threshold(db_session, monkeypatch):
    monkeypatch.setattr(drift, "WINDOW_SIZE", 5)
    monkeypatch.setattr(drift, "DRIFT_THRESHOLD", 0.1)
    await _seed_scored_predictions(db_session, "v1", [0.9] * 10)

    alerts = await drift.run_once(db_session)

    assert alerts == []


@pytest.mark.asyncio
async def test_run_once_detects_drift_beyond_threshold(db_session, monkeypatch):
    monkeypatch.setattr(drift, "WINDOW_SIZE", 5)
    monkeypatch.setattr(drift, "DRIFT_THRESHOLD", 0.05)
    scores = [0.95] * 5 + [0.60] * 5
    await _seed_scored_predictions(db_session, "v1", scores)

    alerts = await drift.run_once(db_session)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.model_version == "v1"
    assert alert.baseline_mean == pytest.approx(0.95)
    assert alert.rolling_mean == pytest.approx(0.60)
    assert alert.delta == pytest.approx(0.35)
    assert alert.threshold == 0.05


@pytest.mark.asyncio
async def test_run_once_checks_each_model_version_independently(
    db_session, monkeypatch
):
    monkeypatch.setattr(drift, "WINDOW_SIZE", 5)
    monkeypatch.setattr(drift, "DRIFT_THRESHOLD", 0.05)
    await _seed_scored_predictions(db_session, "v1", [0.95] * 5 + [0.60] * 5)
    await _seed_scored_predictions(db_session, "v2", [0.9] * 10)

    alerts = await drift.run_once(db_session)

    assert {a.model_version for a in alerts} == {"v1"}


def test_alerts_endpoint_returns_seeded_alerts(client, sync_engine):
    from sqlalchemy import text as sa_text

    with sync_engine.begin() as conn:
        conn.execute(
            sa_text(
                "INSERT INTO drift_alerts "
                "(model_version, baseline_mean, rolling_mean, delta, threshold) "
                "VALUES ('v1', 0.95, 0.60, 0.35, 0.1)"
            )
        )

    response = client.get("/alerts", headers=API_KEY_HEADER)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["model_version"] == "v1"
    assert body[0]["delta"] == 0.35


def test_alerts_endpoint_requires_api_key(client):
    response = client.get("/alerts")

    assert response.status_code == 401
