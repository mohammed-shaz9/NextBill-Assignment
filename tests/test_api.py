"""Integration tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """Enable auth for tests (it is off by default, per assignment spec)."""
    monkeypatch.setattr(settings, "API_KEY", "test_secret")


@pytest.fixture
def auth_headers():
    return {settings.API_KEY_NAME: "test_secret"}


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_docs_link(self, client):
        response = client.get("/")
        assert "docs" in response.text


class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_has_categories(self, client):
        data = client.get("/health").json()
        assert len(data["categories"]) == 6


class TestCategoriesEndpoint:
    """Tests for GET /categories"""

    def test_categories_returns_200(self, client):
        response = client.get("/categories")
        assert response.status_code == 200

    def test_all_categories_present(self, client):
        data = client.get("/categories").json()
        for cat in settings.CATEGORIES:
            assert cat in data["categories"]


class TestPredictEndpoint:
    """Tests for POST /predict"""

    def test_predict_without_api_key_forbidden(self, client):
        """Requests without x-api-key should be forbidden (403)."""
        response = client.post("/predict", json={"text": "Blue Dart courier charges"})
        assert response.status_code == 401
        assert response.json()["success"] is False

    def test_predict_invalid_api_key_forbidden(self, client):
        """Requests with incorrect API key should be forbidden (403)."""
        response = client.post(
            "/predict",
            json={"text": "Blue Dart courier charges"},
            headers={settings.API_KEY_NAME: "wrong_key"},
        )
        assert response.status_code == 401

    def test_predict_returns_200_with_auth(self, client, auth_headers):
        response = client.post(
            "/predict", json={"text": "Blue Dart courier charges"}, headers=auth_headers
        )
        assert response.status_code == 200

    def test_predict_has_category(self, client, auth_headers):
        response = client.post(
            "/predict", json={"text": "AWS cloud hosting bill"}, headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "category" in data
        assert data["category"] in settings.CATEGORIES or data["category"] == "Unknown"

    def test_predict_has_confidence(self, client, auth_headers):
        response = client.post(
            "/predict", json={"text": "Monthly electricity bill"}, headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_has_processing_time(self, client, auth_headers):
        response = client.post("/predict", json={"text": "Office stationery"}, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_predict_empty_text_rejected(self, client, auth_headers):
        """Empty or too-short text should be rejected by validation."""
        response = client.post("/predict", json={"text": ""}, headers=auth_headers)
        assert response.status_code == 422

    def test_predict_short_text_rejected(self, client, auth_headers):
        """Text shorter than 3 chars should be rejected."""
        response = client.post("/predict", json={"text": "ab"}, headers=auth_headers)
        assert response.status_code == 422

    def test_predict_missing_text_field(self, client, auth_headers):
        """Missing 'text' field should return 422."""
        response = client.post("/predict", json={}, headers=auth_headers)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for POST /predict/batch"""

    def test_batch_without_api_key_forbidden(self, client):
        response = client.post(
            "/predict/batch", json={"texts": ["courier delivery", "AWS hosting"]}
        )
        assert response.status_code == 401

    def test_batch_returns_200_with_auth(self, client, auth_headers):
        response = client.post(
            "/predict/batch",
            json={"texts": ["courier delivery", "AWS hosting", "electricity bill"]},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_batch_returns_correct_count(self, client, auth_headers):
        texts = ["courier delivery", "AWS hosting", "electricity bill"]
        response = client.post("/predict/batch", json={"texts": texts}, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 3

    def test_batch_has_total_time(self, client, auth_headers):
        response = client.post(
            "/predict/batch", json={"texts": ["test invoice"]}, headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_processing_time_ms" in data

    def test_batch_empty_list_rejected(self, client, auth_headers):
        """Empty texts list should be rejected."""
        response = client.post("/predict/batch", json={"texts": []}, headers=auth_headers)
        assert response.status_code == 422


class TestSpecContract:
    def test_spec_example_without_key_when_auth_disabled(self, client, monkeypatch):
        monkeypatch.setattr(settings, "API_KEY", "")
        r = client.post(
            "/predict",
            json={"text": "Blue Dart courier charges for warehouse delivery"},
        )
        assert r.status_code == 200 and r.json()["category"] == "Logistics"
        r = client.post("/predict", json={"text": "AWS monthly cloud hosting bill"})
        assert r.json()["category"] == "Cloud/Software"
        assert len(r.json()["alternatives"]) == 3

    def test_metrics_are_real(self, client, auth_headers):
        before = client.get("/metrics", headers=auth_headers).json()["inferences_processed"]
        client.post(
            "/predict/batch",
            json={"texts": ["courier", "AWS hosting"]},
            headers=auth_headers,
        )
        m = client.get("/metrics", headers=auth_headers).json()
        assert m["inferences_processed"] == before + 2
        assert "test_macro_f1" in m["model_evaluation"]

    def test_feedback_rejects_unknown_category(self, client, auth_headers):
        r = client.post(
            "/feedback",
            headers=auth_headers,
            json={
                "original_text": "x",
                "predicted_category": "Travel",
                "actual_category": "Nope",
            },
        )
        assert r.status_code == 422
