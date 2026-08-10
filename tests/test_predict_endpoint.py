"""Integration tests for POST /predict: DB logging and shadow mirroring."""

import hashlib
import time

from sqlalchemy import text

API_KEY_HEADER = {"X-API-Key": "dev"}


def _register(client, name="distilbert-sst2", version="v1", path="/models/v1"):
    return client.post(
        "/models",
        json={"name": name, "version": version, "path": path},
        headers=API_KEY_HEADER,
    )


def _predictions_for_version(sync_engine, model_version):
    with sync_engine.connect() as conn:
        return conn.execute(
            text("SELECT * FROM predictions WHERE model_version = :v"),
            {"v": model_version},
        ).fetchall()


def test_predict_logs_to_predictions_table(client, sync_engine):
    response = client.post(
        "/predict", json={"text": "great product"}, headers=API_KEY_HEADER
    )

    assert response.status_code == 200
    rows = _predictions_for_version(sync_engine, response.json()["model_version"])
    assert len(rows) == 1
    assert rows[0].input_hash == hashlib.sha256(b"great product").hexdigest()


def test_predict_falls_back_to_default_version_with_no_registry(client):
    response = client.post(
        "/predict", json={"text": "great product"}, headers=API_KEY_HEADER
    )

    assert response.json()["model_version"] == "v1"


def test_predict_uses_current_production_version(client):
    created = _register(client, version="v2").json()
    client.put(f"/models/{created['id']}/promote", headers=API_KEY_HEADER)

    response = client.post(
        "/predict", json={"text": "great product"}, headers=API_KEY_HEADER
    )

    assert response.json()["model_version"] == "v2"


def test_predict_does_not_mirror_when_shadow_disabled(client, sync_engine, monkeypatch):
    monkeypatch.setenv("SHADOW_ENABLED", "false")
    shadow_created = _register(client, version="v-shadow").json()
    client.put(f"/models/{shadow_created['id']}/shadow", headers=API_KEY_HEADER)

    client.post("/predict", json={"text": "great product"}, headers=API_KEY_HEADER)
    time.sleep(0.5)

    assert _predictions_for_version(sync_engine, "v-shadow") == []


def test_predict_mirrors_to_shadow_when_enabled(client, sync_engine, monkeypatch):
    monkeypatch.setenv("SHADOW_ENABLED", "true")
    shadow_created = _register(client, version="v-shadow").json()
    client.put(f"/models/{shadow_created['id']}/shadow", headers=API_KEY_HEADER)

    client.post("/predict", json={"text": "great product"}, headers=API_KEY_HEADER)

    deadline = time.monotonic() + 5
    rows = []
    while time.monotonic() < deadline:
        rows = _predictions_for_version(sync_engine, "v-shadow")
        if rows:
            break
        time.sleep(0.1)

    assert len(rows) == 1
    assert rows[0].cached is False
