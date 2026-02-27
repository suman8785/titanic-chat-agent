"""
Tests for the FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data


class TestDatasetEndpoints:
    """Tests for dataset endpoints."""
    
    def test_get_stats(self, client):
        """Test dataset stats endpoint."""
        response = client.get("/api/dataset/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_passengers" in data
        assert "survival_rate" in data
    
    def test_get_columns(self, client):
        """Test dataset columns endpoint."""
        response = client.get("/api/dataset/columns")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) > 0
    
    def test_get_sample(self, client):
        """Test dataset sample endpoint."""
        response = client.get("/api/dataset/sample?n=3")
        assert response.status_code == 200
        
        data = response.json()
        assert "sample" in data
        assert len(data["sample"]) <= 3


class TestChatEndpoint:
    """Tests for chat endpoint."""
    
    def test_simple_chat(self, client):
        """Test simple chat message."""
        response = client.post(
            "/api/chat",
            json={
                "message": "What is the total number of passengers?",
                "session_id": "test-session-123",
                "include_reasoning": False
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "execution_time" in data
    
    def test_chat_with_visualization(self, client):
        """Test chat that should generate visualization."""
        response = client.post(
            "/api/chat",
            json={
                "message": "Show me survival rates by class as a chart",
                "session_id": "test-session-456",
                "include_reasoning": True
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        # Note: Visualization may or may not be present depending on agent decision
    
    def test_empty_message(self, client):
        """Test empty message validation."""
        response = client.post(
            "/api/chat",
            json={
                "message": "",
                "session_id": "test-session-789"
            }
        )
        assert response.status_code == 422  # Validation error


class TestSessionEndpoints:
    """Tests for session management endpoints."""
    
    def test_get_session_info(self, client):
        """Test get session info."""
        # First create a session by sending a message
        client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "session_id": "info-test-session"
            }
        )
        
        response = client.get("/api/session/info-test-session")
        assert response.status_code == 200
        
        data = response.json()
        assert "session_id" in data
    
    def test_clear_session(self, client):
        """Test clear session."""
        response = client.delete("/api/session/test-clear-session")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"