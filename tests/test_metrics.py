"""Tests for gateway/metrics.py and GET /metrics.

The Prometheus registry is a process-global singleton, not reset between
tests (unlike the DB/Redis fixtures), so assertions here compare before/after
deltas rather than absolute values — other tests in the same session also
hit /predict and bump the same counters.
"""

from prometheus_client import REGISTRY

API_KEY_HEADER = {"X-API-Key": "dev"}


def _sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_returns_prometheus_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "request_count_total" in response.text
    assert "inference_latency_seconds" in response.text
    assert "cache_hit_total" in response.text
    assert "active_model_version" in response.text


def test_predict_increments_request_count(client):
    labels = {"model_version": "v1", "status_code": "200"}
    before = _sample("request_count_total", labels)

    client.post(
        "/predict", json={"text": "metrics request count test"}, headers=API_KEY_HEADER
    )

    assert _sample("request_count_total", labels) == before + 1


def test_predict_cache_miss_records_inference_latency(client):
    before = _sample("inference_latency_seconds_count")

    client.post(
        "/predict",
        json={"text": "metrics inference latency test unique"},
        headers=API_KEY_HEADER,
    )

    assert _sample("inference_latency_seconds_count") == before + 1


def test_predict_cache_hit_increments_cache_hit_total_not_latency(client):
    payload = {"text": "metrics cache hit test unique"}
    client.post("/predict", json=payload, headers=API_KEY_HEADER)

    hits_before = _sample("cache_hit_total")
    latency_count_before = _sample("inference_latency_seconds_count")

    client.post("/predict", json=payload, headers=API_KEY_HEADER)

    assert _sample("cache_hit_total") == hits_before + 1
    assert _sample("inference_latency_seconds_count") == latency_count_before


def test_promote_sets_active_model_version_gauge(client):
    created = client.post(
        "/models",
        json={"name": "distilbert-sst2", "version": "metrics-v", "path": "/models/v"},
        headers=API_KEY_HEADER,
    ).json()

    client.put(f"/models/{created['id']}/promote", headers=API_KEY_HEADER)

    assert _sample("active_model_version", {"model_version": "metrics-v"}) == 1.0
