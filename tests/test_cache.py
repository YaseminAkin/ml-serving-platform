"""Integration tests for the Redis prediction cache and /metrics/cache."""

API_KEY_HEADER = {"X-API-Key": "dev"}


def _register(client, name="distilbert-sst2", version="v1", path="/models/v1"):
    return client.post(
        "/models",
        json={"name": name, "version": version, "path": path},
        headers=API_KEY_HEADER,
    )


def test_first_predict_is_a_cache_miss(client):
    response = client.post(
        "/predict",
        json={"text": "unique text for cache miss test"},
        headers=API_KEY_HEADER,
    )

    assert response.status_code == 200
    assert response.json()["cached"] is False


def test_repeated_predict_is_a_cache_hit(client):
    payload = {"text": "unique text for cache hit test"}

    first = client.post("/predict", json=payload, headers=API_KEY_HEADER)
    second = client.post("/predict", json=payload, headers=API_KEY_HEADER)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["label"] == first.json()["label"]
    assert second.json()["score"] == first.json()["score"]


def test_cache_is_scoped_by_model_version(client):
    payload = {"text": "text used across model versions"}

    client.post("/predict", json=payload, headers=API_KEY_HEADER)

    created = _register(client, version="v2").json()
    client.put(f"/models/{created['id']}/promote", headers=API_KEY_HEADER)

    response = client.post("/predict", json=payload, headers=API_KEY_HEADER)

    assert response.json()["model_version"] == "v2"
    assert response.json()["cached"] is False


def test_metrics_cache_reflects_hits_and_misses(client):
    payload = {"text": "text for metrics endpoint test"}
    client.post("/predict", json=payload, headers=API_KEY_HEADER)
    client.post("/predict", json=payload, headers=API_KEY_HEADER)
    client.post("/predict", json=payload, headers=API_KEY_HEADER)

    response = client.get("/metrics/cache")

    assert response.status_code == 200
    body = response.json()
    assert body["total_misses"] == 1
    assert body["total_hits"] == 2
    assert body["hit_rate"] == 2 / 3
