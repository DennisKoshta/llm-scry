from fastapi.testclient import TestClient

from llm_scry.main import app
from llm_scry.sessions import store


def setup_function() -> None:
    # Start each test from a clean store
    store._items.clear()


def test_list_sessions_empty():
    with TestClient(app) as client:
        res = client.get("/sessions")
        assert res.status_code == 200
        assert res.json() == []


def test_get_session_not_found():
    with TestClient(app) as client:
        res = client.get("/session/does-not-exist")
        assert res.status_code == 404


def test_generate_without_model():
    with TestClient(app) as client:
        res = client.post("/generate", json={"prompt": "hi"})
        assert res.status_code == 400
        assert "no model loaded" in res.json()["detail"].lower()


def test_model_info_without_model():
    with TestClient(app) as client:
        res = client.get("/model/info")
        assert res.status_code == 404
