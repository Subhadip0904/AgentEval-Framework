# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_basic():
    """Test /ask endpoint accepts request and returns response"""
    payload = {"question": "What is NVMe?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)


def test_ask_endpoint_with_context():
    """Test /ask endpoint handles context parameter"""
    payload = {
        "question": "How does FTL work?",
        "context": "SSD architecture",
        "user_role": "engineer"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_ask_endpoint_required_field():
    """Test /ask endpoint validates required fields"""
    payload = {}  # Missing 'question'
    response = client.post("/ask", json=payload)
    assert response.status_code == 422  # Validation error


def test_ask_endpoint_response_structure():
    """Test /ask endpoint returns correct response structure"""
    payload = {"question": "Tell me about PCIe"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert data["confidence"] in ["high", "medium", "low"]


def test_ask_endpoint_handles_errors():
    """Test /ask endpoint handles errors gracefully"""
    payload = {"question": "What is X?"}
    response = client.post("/ask", json=payload)
    # Should not crash
    assert response.status_code in [200, 500]
