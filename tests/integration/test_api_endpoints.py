"""
Integration tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.api.main import app
from src.api.dependencies import app_state


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client to avoid external dependencies"""
    mock_client = Mock()
    mock_client.get_response.return_value = "This is a mocked response from the LLM."
    return mock_client


@pytest.fixture
def mock_router():
    """Mock router for testing"""
    mock = Mock()
    # Default routing behavior
    mock.route.return_value = ("professor", 0.95, 2.5)
    return mock


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    mock = Mock()
    mock.log_interaction.return_value = None
    return mock


@pytest.fixture
def mock_config():
    """Mock Hydra config"""
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "app": {
            "title": "LLM-Router",
            "version": "1.0.0",
            "host": "0.0.0.0",
            "port": 8000,
            "debug": True
        },
        "forensics": {
            "log_path": "data/logs/routing_events.jsonl",
            "enabled": True
        },
        "router": {
            "name": "hybrid",
            "parameters": {
                "semantic_model": "all-MiniLM-L6-v2",
                "semantic_threshold": 0.45,
                "fallback_expert": "professor"
            }
        }
    })
    return config


@pytest.fixture
def mock_experts_config():
    """Mock experts configuration"""
    return {
        "professor": {
            "id": "professor",
            "model_name": "llama3.2:3b",
            "system_prompt": "You are a strict, formal Professor of Computer Science.",
            "description": "Handles technical queries, code debugging, and math."
        },
        "zoomer": {
            "id": "zoomer",
            "model_name": "tinyllama",
            "system_prompt": "You are a Gen-Z skateboarder.",
            "description": "Handles casual conversation, jokes, and vibes."
        }
    }


@pytest.fixture
def client(mock_router, mock_ollama_client, mock_logger, mock_config, mock_experts_config):
    """Create a test client with mocked dependencies"""
    # Mock the app state with our test fixtures
    app_state.router = mock_router
    app_state.llm_client = mock_ollama_client
    app_state.logger = mock_logger
    app_state.config = mock_config
    app_state.experts_config = mock_experts_config

    with TestClient(app) as test_client:
        yield test_client


class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert data["name"] == "LLM Router API"


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    @patch('requests.get')
    def test_health_check_ollama_healthy(self, mock_get, client):
        """Test health check when Ollama is healthy"""
        # Mock Ollama health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["ollama_status"] == "healthy"
        assert "routing_strategies" in data
        assert "experts" in data

    @patch('requests.get')
    def test_health_check_ollama_unreachable(self, mock_get, client):
        """Test health check when Ollama is unreachable"""
        # Mock Ollama connection failure
        mock_get.side_effect = Exception("Connection failed")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama_status"] == "unreachable"


class TestRouteEndpoint:
    """Tests for /v1/route endpoint"""

    def test_route_query_success(self, client, mock_router):
        """Test successful query routing"""
        mock_router.route.return_value = ("professor", 0.95, 2.5)

        response = client.post(
            "/v1/route",
            json={"query": "explain the def keyword in Python"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["expert"] == "professor"
        assert data["confidence"] == 0.95
        assert data["latency_ms"] == 2.5
        assert "interaction_id" in data
        assert "timestamp" in data
        assert "strategy_used" in data

    def test_route_query_validation_empty_query(self, client):
        """Test validation for empty query"""
        response = client.post(
            "/v1/route",
            json={"query": "   "}  # Whitespace only
        )

        assert response.status_code == 422  # Validation error

    def test_route_query_validation_missing_query(self, client):
        """Test validation for missing query field"""
        response = client.post(
            "/v1/route",
            json={}
        )

        assert response.status_code == 422  # Validation error

    def test_route_query_with_metadata(self, client, mock_router):
        """Test routing with optional metadata"""
        mock_router.route.return_value = ("zoomer", 1.0, 1.2)

        response = client.post(
            "/v1/route",
            json={
                "query": "yo what's up",
                "user_id": "user123",
                "session_id": "session456"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["expert"] == "zoomer"


class TestChatEndpoint:
    """Tests for /v1/chat endpoint"""

    def test_chat_with_auto_routing(self, client, mock_router, mock_ollama_client, mock_logger):
        """Test chat with automatic routing"""
        mock_router.route.return_value = ("professor", 0.85, 3.5)
        mock_ollama_client.get_response.return_value = "The def keyword is used to define functions in Python."

        response = client.post(
            "/v1/chat",
            json={"query": "explain the def keyword"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["expert"] == "professor"
        assert data["response"] == "The def keyword is used to define functions in Python."
        assert data["routing_confidence"] == 0.85
        assert "total_latency_ms" in data
        assert "interaction_id" in data

        # Verify logger was called
        mock_logger.log_interaction.assert_called_once()

    def test_chat_with_manual_expert_selection(self, client, mock_ollama_client, mock_logger):
        """Test chat with manually specified expert"""
        mock_ollama_client.get_response.return_value = "Yo, that's lit fam!"

        response = client.post(
            "/v1/chat",
            json={
                "query": "tell me something cool",
                "expert": "zoomer"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["expert"] == "zoomer"
        assert data["routing_confidence"] == 1.0  # Manual selection = 100% confidence
        assert data["strategy_used"] == "manual"

    def test_chat_with_custom_parameters(self, client, mock_router, mock_ollama_client):
        """Test chat with custom temperature and max_tokens"""
        mock_router.route.return_value = ("professor", 0.9, 2.0)
        mock_ollama_client.get_response.return_value = "Custom response"

        response = client.post(
            "/v1/chat",
            json={
                "query": "explain recursion",
                "temperature": 0.5,
                "max_tokens": 256
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Custom response"

    def test_chat_ollama_error(self, client, mock_router, mock_ollama_client):
        """Test chat when Ollama returns an error"""
        mock_router.route.return_value = ("professor", 0.9, 2.0)
        mock_ollama_client.get_response.return_value = "[System Error]: Could not connect to Ollama"

        response = client.post(
            "/v1/chat",
            json={"query": "test query"}
        )

        assert response.status_code == 503  # Service unavailable

    def test_chat_invalid_expert(self, client, mock_experts_config):
        """Test chat with invalid expert name"""
        response = client.post(
            "/v1/chat",
            json={
                "query": "test query",
                "expert": "invalid_expert"
            }
        )

        assert response.status_code == 422  # Validation error


class TestFeedbackEndpoint:
    """Tests for /v1/feedback endpoint"""

    def test_submit_positive_feedback(self, client):
        """Test submitting positive feedback"""
        response = client.post(
            "/v1/feedback",
            json={
                "interaction_id": "test-interaction-123",
                "feedback": 1
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["interaction_id"] == "test-interaction-123"

    def test_submit_negative_feedback(self, client):
        """Test submitting negative feedback"""
        response = client.post(
            "/v1/feedback",
            json={
                "interaction_id": "test-interaction-456",
                "feedback": 0
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_submit_feedback_with_comment(self, client):
        """Test submitting feedback with optional comment"""
        response = client.post(
            "/v1/feedback",
            json={
                "interaction_id": "test-interaction-789",
                "feedback": 1,
                "comment": "Great response!"
            }
        )

        assert response.status_code == 200

    def test_feedback_validation_invalid_score(self, client):
        """Test validation for invalid feedback score"""
        response = client.post(
            "/v1/feedback",
            json={
                "interaction_id": "test-interaction-999",
                "feedback": 5  # Invalid: must be 0 or 1
            }
        )

        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Tests for error handling"""

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.get("/v1/chat")  # Should be POST
        assert response.status_code == 405
