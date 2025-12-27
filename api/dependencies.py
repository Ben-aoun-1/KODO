"""Dependency injection setup for the Kodo API.

This module provides FastAPI dependency functions for injecting
services and resources into route handlers.
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends

from core.graph.connection import GraphConnection
from core.graph.store import GraphStore
from core.ingestion.pipeline import IngestionPipeline

from .config import Settings, get_settings

# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]


# Global instances for connection management
_graph_connection: GraphConnection | None = None
_graph_store: GraphStore | None = None
_ingestion_pipeline: IngestionPipeline | None = None


async def init_dependencies(settings: Settings) -> None:
    """Initialize global dependencies on application startup.

    Creates and connects to database instances that will be shared
    across all requests.

    Args:
        settings: Application settings instance.
    """
    global _graph_connection, _graph_store, _ingestion_pipeline

    # Initialize Neo4j connection
    _graph_connection = GraphConnection(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    await _graph_connection.connect()

    # Create graph store with the connection
    _graph_store = GraphStore(_graph_connection)

    # Initialize ingestion pipeline
    _ingestion_pipeline = IngestionPipeline()


async def shutdown_dependencies() -> None:
    """Cleanup dependencies on application shutdown.

    Closes database connections and releases resources.
    """
    global _graph_connection, _graph_store, _ingestion_pipeline

    if _graph_connection is not None:
        await _graph_connection.close()
        _graph_connection = None

    _graph_store = None
    _ingestion_pipeline = None


async def get_graph_connection() -> AsyncGenerator[GraphConnection, None]:
    """Get the Neo4j graph connection.

    Yields:
        The shared GraphConnection instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _graph_connection is None:
        raise RuntimeError(
            "Graph connection not initialized. Ensure init_dependencies() was called on startup."
        )
    yield _graph_connection


async def get_graph_store() -> AsyncGenerator[GraphStore, None]:
    """Get the graph store for Neo4j operations.

    Yields:
        The shared GraphStore instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _graph_store is None:
        raise RuntimeError(
            "Graph store not initialized. Ensure init_dependencies() was called on startup."
        )
    yield _graph_store


async def get_ingestion_pipeline() -> AsyncGenerator[IngestionPipeline, None]:
    """Get the ingestion pipeline for repository processing.

    Yields:
        The shared IngestionPipeline instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _ingestion_pipeline is None:
        raise RuntimeError(
            "Ingestion pipeline not initialized. Ensure init_dependencies() was called on startup."
        )
    yield _ingestion_pipeline


# Type aliases for commonly used dependencies
GraphConnectionDep = Annotated[GraphConnection, Depends(get_graph_connection)]
GraphStoreDep = Annotated[GraphStore, Depends(get_graph_store)]
IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
