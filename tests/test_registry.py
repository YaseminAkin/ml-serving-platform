"""Integration tests for the model registry endpoints."""

API_KEY_HEADER = {"X-API-Key": "dev"}


def _register(client, name="distilbert-sst2", version="v1", path="/models/v1"):
    return client.post(
        "/models",
        json={"name": name, "version": version, "path": path},
        headers=API_KEY_HEADER,
    )


def test_register_model_returns_staging_status(client):
    response = _register(client)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "distilbert-sst2"
    assert body["version"] == "v1"
    assert body["status"] == "staging"


def test_register_duplicate_name_version_returns_409(client):
    _register(client)
    response = _register(client)

    assert response.status_code == 409


def test_list_models_returns_registered_versions(client):
    _register(client, version="v1")
    _register(client, version="v2")

    response = client.get("/models", headers=API_KEY_HEADER)

    assert response.status_code == 200
    versions = {m["version"] for m in response.json()}
    assert versions == {"v1", "v2"}


def test_promote_sets_status_to_production(client):
    created = _register(client).json()

    response = client.put(f"/models/{created['id']}/promote", headers=API_KEY_HEADER)

    assert response.status_code == 200
    assert response.json()["status"] == "production"


def test_promote_demotes_previous_production_version(client):
    v1 = _register(client, version="v1").json()
    v2 = _register(client, version="v2").json()

    client.put(f"/models/{v1['id']}/promote", headers=API_KEY_HEADER)
    client.put(f"/models/{v2['id']}/promote", headers=API_KEY_HEADER)

    statuses = {
        m["version"]: m["status"]
        for m in client.get("/models", headers=API_KEY_HEADER).json()
    }
    assert statuses["v1"] == "archived"
    assert statuses["v2"] == "production"


def test_promote_nonexistent_id_returns_404(client):
    response = client.put("/models/999999/promote", headers=API_KEY_HEADER)

    assert response.status_code == 404


def test_models_endpoints_require_api_key(client):
    response = client.get("/models")

    assert response.status_code == 401
