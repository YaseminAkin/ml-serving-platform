"""Tests for eval/scorer.py and eval/worker.py."""

import pytest
from sqlalchemy import text

from db.models import Prediction
from eval import scorer, worker

API_KEY_HEADER = {"X-API-Key": "dev"}


def test_score_confidence_reads_model_output_score():
    prediction = Prediction(output={"label": "POSITIVE", "score": 0.87})

    assert scorer.score_confidence(prediction) == 0.87


@pytest.mark.asyncio
async def test_score_llm_judge_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    prediction = Prediction(
        text="great product", output={"label": "POSITIVE", "score": 0.9}
    )

    result = await scorer.score_llm_judge(prediction)

    assert result is None


@pytest.mark.asyncio
async def test_score_llm_judge_parses_openai_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"score": 0.75}'}}]}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(scorer.httpx, "AsyncClient", lambda **kwargs: FakeAsyncClient())

    prediction = Prediction(
        text="great product", output={"label": "POSITIVE", "score": 0.9}
    )
    result = await scorer.score_llm_judge(prediction)

    assert result == 0.75


@pytest.mark.asyncio
async def test_score_llm_judge_returns_none_on_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FailingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise RuntimeError("network error")

    monkeypatch.setattr(
        scorer.httpx, "AsyncClient", lambda **kwargs: FailingAsyncClient()
    )

    prediction = Prediction(
        text="great product", output={"label": "POSITIVE", "score": 0.9}
    )
    result = await scorer.score_llm_judge(prediction)

    assert result is None


@pytest.mark.asyncio
async def test_run_once_scores_unscored_predictions(client, sync_engine, db_session):
    client.post("/predict", json={"text": "loved it"}, headers=API_KEY_HEADER)

    scored_count = await worker.run_once(db_session)

    assert scored_count == 1
    with sync_engine.connect() as conn:
        rows = conn.execute(text("SELECT method, score FROM eval_results")).fetchall()
    assert len(rows) == 1
    assert rows[0].method == "confidence"


@pytest.mark.asyncio
async def test_run_once_does_not_rescore_already_scored_predictions(client, db_session):
    client.post("/predict", json={"text": "loved it"}, headers=API_KEY_HEADER)

    first_pass = await worker.run_once(db_session)
    second_pass = await worker.run_once(db_session)

    assert first_pass == 1
    assert second_pass == 0


@pytest.mark.asyncio
async def test_run_once_skips_llm_judge_without_api_key(
    client, sync_engine, db_session, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client.post("/predict", json={"text": "loved it"}, headers=API_KEY_HEADER)

    await worker.run_once(db_session)

    with sync_engine.connect() as conn:
        methods = {
            row.method
            for row in conn.execute(text("SELECT method FROM eval_results")).fetchall()
        }
    assert methods == {"confidence"}
