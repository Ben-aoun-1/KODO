"""Tests for the Kodo API module.

This module contains comprehensive tests for:
- Settings configuration
- Health check endpoints
- Repository management endpoints
- Code query endpoints
- Middleware behavior

All tests use httpx.AsyncClient for async testing with FastAPI.
Dependencies (GraphStore, IngestionPipeline) are mocked.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.config import Settings, get_settings
from api.dependencies import (
    get_graph_connection,
    get_graph_store,
    get_ingestion_pipeline,
)
from api.main import create_app
from api.middleware import RequestLoggingMiddleware, TimingMiddleware
from core.graph.models import (
    ClassNode,
    FileNode,
    FunctionNode,
    ImpactResult,
    MethodNode,
    RepositoryNode,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_store() -> MagicMock:
    """Create a mock GraphStore instance."""
    mock = MagicMock()
    mock.get_repository = AsyncMock(return_value=None)
    mock.create_repository = AsyncMock()
    mock.delete_repository = AsyncMock(return_value=True)
    mock.get_repository_statistics = AsyncMock(
        return_value={
            "file_count": 10,
            "function_count": 25,
            "class_count": 5,
            "method_count": 40,
        }
    )
    mock.get_files_in_repository = AsyncMock(return_value=[])
    mock.find_function_by_name = AsyncMock(return_value=[])
    mock.find_class_by_name = AsyncMock(return_value=[])
    mock.get_methods_of_class = AsyncMock(return_value=[])
    mock.find_callers = AsyncMock(return_value=[])
    mock.find_callers_recursive = AsyncMock(return_value=[])
    mock.find_callees = AsyncMock(return_value=[])
    mock.analyze_impact = AsyncMock(
        return_value=ImpactResult(
            source_id="test",
            affected_nodes=[],
            affected_files=[],
            depth=0,
            total_affected=0,
        )
    )
    return mock


@pytest.fixture
def mock_graph_connection() -> MagicMock:
    """Create a mock GraphConnection instance."""
    mock = MagicMock()
    mock.health_check = AsyncMock(return_value={"status": "healthy"})
    return mock


@pytest.fixture
def mock_ingestion_pipeline() -> MagicMock:
    """Create a mock IngestionPipeline instance."""
    mock = MagicMock()
    mock.ingest = AsyncMock()
    return mock


@pytest.fixture
def sample_repository_node() -> RepositoryNode:
    """Create a sample repository node."""
    return RepositoryNode(
        id="repo:test-repo",
        name="test-repo",
        url="https://github.com/test/test-repo",
        default_branch="main",
        last_indexed=datetime(2024, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_file_nodes() -> list[FileNode]:
    """Create sample file nodes."""
    return [
        FileNode(
            id="repo:test-repo:/src/main.py",
            path="/src/main.py",
            language="python",
            hash="abc123",
            size=1024,
            repo_id="repo:test-repo",
        ),
        FileNode(
            id="repo:test-repo:/src/utils.py",
            path="/src/utils.py",
            language="python",
            hash="def456",
            size=512,
            repo_id="repo:test-repo",
        ),
    ]


@pytest.fixture
def sample_function_nodes() -> list[FunctionNode]:
    """Create sample function nodes."""
    return [
        FunctionNode(
            id="/src/main.py:process_data:10",
            name="process_data",
            file_path="/src/main.py",
            start_line=10,
            end_line=30,
            is_async=True,
            docstring="Process input data.",
            parameters=["data", "options"],
            return_type="dict",
            repo_id="repo:test-repo",
        ),
        FunctionNode(
            id="/src/utils.py:helper:5",
            name="helper",
            file_path="/src/utils.py",
            start_line=5,
            end_line=15,
            is_async=False,
            docstring="A helper function.",
            parameters=["x"],
            return_type="int",
            repo_id="repo:test-repo",
        ),
    ]


@pytest.fixture
def sample_class_nodes() -> list[ClassNode]:
    """Create sample class nodes."""
    return [
        ClassNode(
            id="/src/main.py:DataProcessor:50",
            name="DataProcessor",
            file_path="/src/main.py",
            start_line=50,
            end_line=100,
            docstring="A data processing class.",
            bases=["BaseProcessor"],
            decorators=["dataclass"],
            repo_id="repo:test-repo",
        ),
    ]


@pytest.fixture
def sample_method_nodes() -> list[MethodNode]:
    """Create sample method nodes."""
    return [
        MethodNode(
            id="/src/main.py:DataProcessor.run:60",
            name="run",
            file_path="/src/main.py",
            start_line=60,
            end_line=80,
            is_async=True,
            docstring="Run the processor.",
            parameters=["self", "input_data"],
            return_type="dict",
            class_id="/src/main.py:DataProcessor:50",
            repo_id="repo:test-repo",
        ),
    ]


@pytest.fixture
def app_with_mocks(
    mock_graph_store: MagicMock,
    mock_graph_connection: MagicMock,
    mock_ingestion_pipeline: MagicMock,
) -> FastAPI:
    """Create a FastAPI app with mocked dependencies."""
    app = create_app()

    async def override_graph_store() -> AsyncGenerator[MagicMock, None]:
        yield mock_graph_store

    async def override_graph_connection() -> AsyncGenerator[MagicMock, None]:
        yield mock_graph_connection

    async def override_ingestion_pipeline() -> AsyncGenerator[MagicMock, None]:
        yield mock_ingestion_pipeline

    app.dependency_overrides[get_graph_store] = override_graph_store
    app.dependency_overrides[get_graph_connection] = override_graph_connection
    app.dependency_overrides[get_ingestion_pipeline] = override_ingestion_pipeline

    return app


@pytest_asyncio.fixture
async def client(app_with_mocks: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mocks),
        base_url="http://test",
    ) as ac:
        yield ac


# =============================================================================
# Config Tests
# =============================================================================


class TestSettings:
    """Tests for Settings configuration."""

    def test_settings_default_values(self):
        """Test that Settings has correct default values."""
        # Clear cache to ensure fresh settings
        get_settings.cache_clear()

        settings = Settings()

        assert settings.app_name == "Kodo API"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.neo4j_uri == "bolt://localhost:7687"
        assert settings.neo4j_user == "neo4j"
        assert settings.neo4j_password == "password"
        assert settings.neo4j_database == "neo4j"
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.qdrant_api_key is None
        assert settings.api_prefix == "/api/v1"
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:5173" in settings.cors_origins

    def test_settings_from_environment_variables(self):
        """Test that Settings loads from environment variables."""
        get_settings.cache_clear()

        with patch.dict(
            os.environ,
            {
                "APP_NAME": "Custom API",
                "APP_VERSION": "1.0.0",
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "NEO4J_URI": "bolt://custom:7687",
                "NEO4J_USER": "admin",
                "NEO4J_PASSWORD": "secret",
                "API_PREFIX": "/api/v2",
            },
            clear=False,
        ):
            settings = Settings()

            assert settings.app_name == "Custom API"
            assert settings.app_version == "1.0.0"
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.neo4j_uri == "bolt://custom:7687"
            assert settings.neo4j_user == "admin"
            assert settings.neo4j_password == "secret"
            assert settings.api_prefix == "/api/v2"

    def test_get_settings_caching(self):
        """Test that get_settings() caches the settings instance."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance due to caching
        assert settings1 is settings2

    def test_settings_optional_api_keys(self):
        """Test that optional API keys default to None."""
        get_settings.cache_clear()

        settings = Settings()

        assert settings.anthropic_api_key is None
        assert settings.voyage_api_key is None
        assert settings.openai_api_key is None
        assert settings.github_app_id is None
        assert settings.github_private_key is None
        assert settings.postgres_url is None


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self, client: AsyncClient):
        """Test GET /health returns 200 with healthy status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data

    @pytest.mark.asyncio
    async def test_health_ready_checks_dependencies(
        self,
        client: AsyncClient,
        mock_graph_connection: MagicMock,
    ):
        """Test GET /health/ready checks Neo4j connection."""
        mock_graph_connection.health_check.return_value = {"status": "healthy"}

        response = await client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "checks" in data
        assert "neo4j" in data["checks"]
        assert data["checks"]["neo4j"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_ready_unhealthy_neo4j(
        self,
        client: AsyncClient,
        mock_graph_connection: MagicMock,
    ):
        """Test GET /health/ready returns unhealthy when Neo4j is down."""
        mock_graph_connection.health_check.return_value = {
            "status": "unhealthy",
            "message": "Connection failed",
        }

        response = await client.get("/health/ready")

        assert response.status_code == 200  # Still returns 200, but status is unhealthy
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["neo4j"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_ready_neo4j_exception(
        self,
        client: AsyncClient,
        mock_graph_connection: MagicMock,
    ):
        """Test GET /health/ready handles Neo4j exceptions."""
        mock_graph_connection.health_check.side_effect = Exception("Connection error")

        response = await client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["neo4j"]["status"] == "unhealthy"
        assert "Connection error" in data["checks"]["neo4j"]["message"]

    @pytest.mark.asyncio
    async def test_liveness_check_returns_200(self, client: AsyncClient):
        """Test GET /health/live returns 200."""
        response = await client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


# =============================================================================
# Repository Endpoint Tests
# =============================================================================


class TestRepositoryEndpoints:
    """Tests for repository management endpoints."""

    @pytest.mark.asyncio
    async def test_create_repository(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test POST /api/v1/repos creates a repository."""
        mock_graph_store.get_repository.return_value = None  # No existing repo
        mock_graph_store.create_repository.return_value = None

        response = await client.post(
            "/api/v1/repos",
            json={
                "url": "https://github.com/test/new-repo",
                "name": "new-repo",
                "branch": "main",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "new-repo"
        assert data["url"] == "https://github.com/test/new-repo"
        assert data["default_branch"] == "main"
        assert data["status"] == "pending"
        mock_graph_store.create_repository.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_repository_derives_name_from_url(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test POST /api/v1/repos derives name from URL if not provided."""
        mock_graph_store.get_repository.return_value = None

        response = await client.post(
            "/api/v1/repos",
            json={"url": "https://github.com/owner/my-project.git"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "my-project"  # .git stripped, name derived

    @pytest.mark.asyncio
    async def test_create_repository_conflict(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test POST /api/v1/repos returns 409 if repository exists."""
        mock_graph_store.get_repository.return_value = sample_repository_node

        response = await client.post(
            "/api/v1/repos",
            json={"url": "https://github.com/test/test-repo"},
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_repository_invalid_url(
        self,
        client: AsyncClient,
    ):
        """Test POST /api/v1/repos validates URL format."""
        response = await client.post(
            "/api/v1/repos",
            json={"url": "not-a-valid-url"},
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_list_repositories(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos lists repositories."""
        response = await client.get("/api/v1/repos")

        assert response.status_code == 200
        data = response.json()
        assert "repositories" in data
        assert "total" in data
        assert isinstance(data["repositories"], list)

    @pytest.mark.asyncio
    async def test_get_repository(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test GET /api/v1/repos/{repo_id} returns repository details."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.get_repository_statistics.return_value = {
            "file_count": 10,
            "function_count": 25,
            "class_count": 5,
        }

        response = await client.get("/api/v1/repos/repo:test-repo")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "repo:test-repo"
        assert data["name"] == "test-repo"
        assert data["file_count"] == 10
        assert data["function_count"] == 25
        assert data["class_count"] == 5

    @pytest.mark.asyncio
    async def test_get_repository_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id} returns 404 for non-existent repo."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_sync_repository(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test POST /api/v1/repos/{repo_id}/sync triggers sync."""
        mock_graph_store.get_repository.return_value = sample_repository_node

        response = await client.post("/api/v1/repos/repo:test-repo/sync")

        assert response.status_code == 200
        data = response.json()
        assert data["repo_id"] == "repo:test-repo"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_sync_repository_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test POST /api/v1/repos/{repo_id}/sync returns 404 for non-existent repo."""
        mock_graph_store.get_repository.return_value = None

        response = await client.post("/api/v1/repos/nonexistent/sync")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_repository(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test DELETE /api/v1/repos/{repo_id} deletes repository."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.delete_repository.return_value = True

        response = await client.delete("/api/v1/repos/repo:test-repo")

        assert response.status_code == 200
        data = response.json()
        assert data["repo_id"] == "repo:test-repo"
        assert data["deleted"] is True
        mock_graph_store.delete_repository.assert_called_once_with("repo:test-repo")

    @pytest.mark.asyncio
    async def test_delete_repository_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test DELETE /api/v1/repos/{repo_id} returns 404 for non-existent repo."""
        mock_graph_store.get_repository.return_value = None

        response = await client.delete("/api/v1/repos/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_repository_failure(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test DELETE /api/v1/repos/{repo_id} handles deletion failure."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.delete_repository.return_value = False

        response = await client.delete("/api/v1/repos/repo:test-repo")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is False


# =============================================================================
# Query Endpoint Tests
# =============================================================================


class TestQueryEndpoints:
    """Tests for code query endpoints."""

    @pytest.mark.asyncio
    async def test_list_files(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_file_nodes: list[FileNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/files lists files."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.get_files_in_repository.return_value = sample_file_nodes

        response = await client.get("/api/v1/repos/repo:test-repo/files")

        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "total" in data
        assert len(data["files"]) == 2
        assert data["files"][0]["path"] == "/src/main.py"
        assert data["files"][0]["language"] == "python"

    @pytest.mark.asyncio
    async def test_list_files_with_language_filter(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_file_nodes: list[FileNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/files with language filter."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.get_files_in_repository.return_value = sample_file_nodes

        response = await client.get(
            "/api/v1/repos/repo:test-repo/files",
            params={"language": "python"},
        )

        assert response.status_code == 200
        data = response.json()
        # All sample files are Python, so should return all
        assert len(data["files"]) == 2

    @pytest.mark.asyncio
    async def test_list_files_with_pagination(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_file_nodes: list[FileNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/files with pagination."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.get_files_in_repository.return_value = sample_file_nodes

        response = await client.get(
            "/api/v1/repos/repo:test-repo/files",
            params={"limit": 1, "offset": 0},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["files"]) == 1
        assert data["total"] == 2  # Total count before pagination

    @pytest.mark.asyncio
    async def test_list_files_repo_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id}/files returns 404 for non-existent repo."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent/files")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_functions(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_function_nodes: list[FunctionNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/functions lists functions."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = sample_function_nodes

        response = await client.get(
            "/api/v1/repos/repo:test-repo/functions",
            params={"name": "process"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "functions" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_list_functions_repo_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id}/functions returns 404."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent/functions")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_classes(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_class_nodes: list[ClassNode],
        sample_method_nodes: list[MethodNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/classes lists classes."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_class_by_name.return_value = sample_class_nodes
        mock_graph_store.get_methods_of_class.return_value = sample_method_nodes

        response = await client.get(
            "/api/v1/repos/repo:test-repo/classes",
            params={"name": "DataProcessor"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_list_classes_repo_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id}/classes returns 404."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent/classes")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_find_callers(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_function_nodes: list[FunctionNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callers/{name} finds callers."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = [sample_function_nodes[0]]
        mock_graph_store.find_callers.return_value = [sample_function_nodes[1]]

        response = await client.get("/api/v1/repos/repo:test-repo/graph/callers/process_data")

        assert response.status_code == 200
        data = response.json()
        assert "entity_id" in data
        assert "entity_name" in data
        assert "results" in data
        assert "total" in data
        assert data["entity_name"] == "process_data"

    @pytest.mark.asyncio
    async def test_find_callers_with_depth(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_function_nodes: list[FunctionNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callers/{name} with max_depth."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = [sample_function_nodes[0]]
        mock_graph_store.find_callers_recursive.return_value = [(sample_function_nodes[1], 1)]

        response = await client.get(
            "/api/v1/repos/repo:test-repo/graph/callers/process_data",
            params={"max_depth": 3},
        )

        assert response.status_code == 200
        mock_graph_store.find_callers_recursive.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_callers_repo_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callers/{name} returns 404."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent/graph/callers/some_func")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_find_callers_function_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callers/{name} returns 404 for unknown function."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = []

        response = await client.get("/api/v1/repos/repo:test-repo/graph/callers/unknown_func")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_find_callees(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_function_nodes: list[FunctionNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callees/{name} finds callees."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = [sample_function_nodes[0]]
        mock_graph_store.find_callees.return_value = [sample_function_nodes[1]]

        response = await client.get("/api/v1/repos/repo:test-repo/graph/callees/process_data")

        assert response.status_code == 200
        data = response.json()
        assert "entity_id" in data
        assert "entity_name" in data
        assert "results" in data
        assert data["entity_name"] == "process_data"

    @pytest.mark.asyncio
    async def test_find_callees_repo_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callees/{name} returns 404."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent/graph/callees/some_func")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_find_callees_function_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/callees/{name} returns 404 for unknown function."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = []

        response = await client.get("/api/v1/repos/repo:test-repo/graph/callees/unknown_func")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_impact_analysis_function(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_function_nodes: list[FunctionNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/impact/{name} for function."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = [sample_function_nodes[0]]
        mock_graph_store.analyze_impact.return_value = ImpactResult(
            source_id=sample_function_nodes[0].id,
            affected_nodes=["node1", "node2"],
            affected_files=["/src/other.py"],
            depth=2,
            total_affected=2,
        )

        response = await client.get(
            "/api/v1/repos/repo:test-repo/graph/impact/process_data",
            params={"entity_type": "function"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "source_id" in data
        assert "source_name" in data
        assert "affected_entities" in data
        assert "affected_files" in data
        assert "impact_depth" in data
        assert "total_affected" in data
        assert data["source_name"] == "process_data"
        assert data["total_affected"] == 2

    @pytest.mark.asyncio
    async def test_impact_analysis_class(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_class_nodes: list[ClassNode],
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/impact/{name} for class."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_class_by_name.return_value = [sample_class_nodes[0]]
        mock_graph_store.analyze_impact.return_value = ImpactResult(
            source_id=sample_class_nodes[0].id,
            affected_nodes=["node1"],
            affected_files=["/src/other.py"],
            depth=1,
            total_affected=1,
        )

        response = await client.get(
            "/api/v1/repos/repo:test-repo/graph/impact/DataProcessor",
            params={"entity_type": "class"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_name"] == "DataProcessor"

    @pytest.mark.asyncio
    async def test_impact_analysis_repo_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/impact/{name} returns 404."""
        mock_graph_store.get_repository.return_value = None

        response = await client.get("/api/v1/repos/nonexistent/graph/impact/some_entity")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_impact_analysis_entity_not_found(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/impact/{name} returns 404 for unknown entity."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.find_function_by_name.return_value = []

        response = await client.get(
            "/api/v1/repos/repo:test-repo/graph/impact/unknown_func",
            params={"entity_type": "function"},
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_impact_analysis_unsupported_entity_type(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
    ):
        """Test GET /api/v1/repos/{repo_id}/graph/impact/{name} returns 400 for unsupported type."""
        mock_graph_store.get_repository.return_value = sample_repository_node

        response = await client.get(
            "/api/v1/repos/repo:test-repo/graph/impact/some_var",
            params={"entity_type": "variable"},
        )

        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]


# =============================================================================
# Middleware Tests
# =============================================================================


class TestMiddleware:
    """Tests for custom middleware."""

    @pytest.mark.asyncio
    async def test_timing_middleware_adds_header(self, client: AsyncClient):
        """Test that X-Response-Time header is added to responses."""
        response = await client.get("/health")

        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
        # Verify format is like "X.XXms"
        time_header = response.headers["X-Response-Time"]
        assert time_header.endswith("ms")

    @pytest.mark.asyncio
    async def test_request_logging_middleware_adds_request_id(
        self,
        client: AsyncClient,
    ):
        """Test that X-Request-ID header is added to responses."""
        response = await client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        # Request ID should be an 8-character string
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 8

    @pytest.mark.asyncio
    async def test_middleware_applied_to_all_routes(self, client: AsyncClient):
        """Test that middleware is applied to various routes."""
        routes_to_test = [
            "/health",
            "/health/live",
            "/api/v1/repos",
        ]

        for route in routes_to_test:
            response = await client.get(route)
            assert "X-Response-Time" in response.headers, f"Missing header on {route}"
            assert "X-Request-ID" in response.headers, f"Missing header on {route}"


# =============================================================================
# Root Endpoint Tests
# =============================================================================


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.asyncio
    async def test_root_returns_api_info(self, client: AsyncClient):
        """Test GET / returns API information."""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_404_for_unknown_route(self, client: AsyncClient):
        """Test that unknown routes return 404."""
        response = await client.get("/api/v1/unknown/route")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_422_for_invalid_request_body(self, client: AsyncClient):
        """Test that invalid request bodies return 422."""
        response = await client.post(
            "/api/v1/repos",
            json={"invalid": "data"},  # Missing required 'url' field
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_500_on_internal_error(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test that internal errors return 500."""
        mock_graph_store.get_repository.return_value = None
        mock_graph_store.create_repository.side_effect = Exception("Database error")

        response = await client.post(
            "/api/v1/repos",
            json={"url": "https://github.com/test/error-repo"},
        )

        assert response.status_code == 500
        assert "Failed to create repository" in response.json()["detail"]


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API workflows."""

    @pytest.mark.asyncio
    async def test_repository_lifecycle(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
    ):
        """Test complete repository lifecycle: create, get, sync, delete."""
        # Setup mocks for the lifecycle
        repo_node = RepositoryNode(
            id="repo:lifecycle-test",
            name="lifecycle-test",
            url="https://github.com/test/lifecycle-test",
            default_branch="main",
        )

        # Create
        mock_graph_store.get_repository.return_value = None
        response = await client.post(
            "/api/v1/repos",
            json={"url": "https://github.com/test/lifecycle-test"},
        )
        assert response.status_code == 201
        created_id = response.json()["id"]

        # Get
        mock_graph_store.get_repository.return_value = repo_node
        response = await client.get(f"/api/v1/repos/{created_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "lifecycle-test"

        # Sync
        response = await client.post(f"/api/v1/repos/{created_id}/sync")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"

        # Delete
        mock_graph_store.delete_repository.return_value = True
        response = await client.delete(f"/api/v1/repos/{created_id}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_query_workflow(
        self,
        client: AsyncClient,
        mock_graph_store: MagicMock,
        sample_repository_node: RepositoryNode,
        sample_file_nodes: list[FileNode],
        sample_function_nodes: list[FunctionNode],
        sample_class_nodes: list[ClassNode],
    ):
        """Test querying repository contents."""
        mock_graph_store.get_repository.return_value = sample_repository_node
        mock_graph_store.get_files_in_repository.return_value = sample_file_nodes
        mock_graph_store.find_function_by_name.return_value = sample_function_nodes
        mock_graph_store.find_class_by_name.return_value = sample_class_nodes
        mock_graph_store.get_methods_of_class.return_value = []

        repo_id = "repo:test-repo"

        # List files
        response = await client.get(f"/api/v1/repos/{repo_id}/files")
        assert response.status_code == 200
        assert response.json()["total"] == 2

        # List functions
        response = await client.get(
            f"/api/v1/repos/{repo_id}/functions",
            params={"name": "process"},
        )
        assert response.status_code == 200

        # List classes
        response = await client.get(
            f"/api/v1/repos/{repo_id}/classes",
            params={"name": "Data"},
        )
        assert response.status_code == 200
