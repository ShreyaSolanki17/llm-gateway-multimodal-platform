import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI test client."""
    return TestClient(app)
