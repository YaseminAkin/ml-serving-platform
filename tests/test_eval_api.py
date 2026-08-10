"""Tests for GET /eval/summary and GET /eval/predictions."""

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


def test_eval_summary_reports_count_and_avg_score(client):
    client.post("/predict", json={"text": "loved it"}, headers=API_KEY_HEADER)
    client.post("/predict", json={"text": "hated it"}, headers=API_KEY_HEADER)

    response = client.get("/eval/summary", headers=API_KEY_HEADER)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    entry = body[0]
    assert entry["model_version"] == "v1"
    assert entry["prediction_count"] == 2
    assert entry["drift_status"] == "ok"


@pytest.mark.asyncio
async def test_eval_summary_drift_status_reflects_live_drift(
    client, db_session, monkeypatch
):
    monkeypatch.setattr(drift, "WINDOW_SIZE", 5)
    monkeypatch.setattr(drift, "DRIFT_THRESHOLD", 0.05)
    await _seed_scored_predictions(db_session, "v1", [0.95] * 5 + [0.60] * 5)

    response = client.get("/eval/summary", headers=API_KEY_HEADER)

    body = response.json()
    assert body[0]["drift_status"] == "alert"
    assert body[0]["avg_confidence_score"] == pytest.approx(
        sum([0.95] * 5 + [0.60] * 5) / 10
    )


def test_eval_summary_requires_api_key(client):
    response = client.get("/eval/summary")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_eval_predictions_returns_scored_rows_for_version(client, db_session):
    await _seed_scored_predictions(db_session, "v1", [0.9, 0.8])
    await _seed_scored_predictions(db_session, "v2", [0.7])

    response = client.get(
        "/eval/predictions", params={"version": "v1"}, headers=API_KEY_HEADER
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(row["model_version"] == "v1" for row in body)
    assert {row["confidence_score"] for row in body} == {0.9, 0.8}
    # newest first
    assert body[0]["text"] == "text 1"


@pytest.mark.asyncio
async def test_eval_predictions_respects_limit_and_offset(client, db_session):
    await _seed_scored_predictions(db_session, "v1", [0.1, 0.2, 0.3, 0.4])

    page1 = client.get(
        "/eval/predictions",
        params={"version": "v1", "limit": 2, "offset": 0},
        headers=API_KEY_HEADER,
    ).json()
    page2 = client.get(
        "/eval/predictions",
        params={"version": "v1", "limit": 2, "offset": 2},
        headers=API_KEY_HEADER,
    ).json()

    assert len(page1) == 2
    assert len(page2) == 2
    assert {row["id"] for row in page1}.isdisjoint({row["id"] for row in page2})


def test_eval_predictions_requires_version_query_param(client):
    response = client.get("/eval/predictions", headers=API_KEY_HEADER)

    assert response.status_code == 422


def test_eval_predictions_requires_api_key(client):
    response = client.get("/eval/predictions", params={"version": "v1"})

    assert response.status_code == 401
