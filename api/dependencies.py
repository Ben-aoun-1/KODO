"""Dependency injection setup for the Kodo API.

This module provides FastAPI dependency functions for injecting
services and resources into route handlers.
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends

from core.embeddings.store import VectorStore, VectorStoreConfig
from core.graph.connection import GraphConnection
from core.graph.store import GraphStore
from core.ingestion.pipeline import IngestionPipeline
from core.query.engine import QueryEngine, QueryEngineConfig

from .config import Settings, get_settings

# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]


# Global instances for connection management
_graph_connection: GraphConnection | None = None
_graph_store: GraphStore | None = None
_ingestion_pipeline: IngestionPipeline | None = None
_vector_store: VectorStore | None = None
_query_engine: QueryEngine | None = None


async def init_dependencies(settings: Settings) -> None:
    """Initialize global dependencies on application startup.

    Creates and connects to database instances that will be shared
    across all requests.

    Args:
        settings: Application settings instance.
    """
    global _graph_connection, _graph_store, _ingestion_pipeline, _vector_store, _query_engine

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

    # Initialize vector store for semantic search
    vector_config = VectorStoreConfig(
        url=settings.qdrant_url,
        collection_name="code_chunks",
    )
    _vector_store = VectorStore(config=vector_config)

    # Initialize query engine
    query_config = QueryEngineConfig(
        max_context_tokens=settings.max_context_tokens,
        max_response_tokens=settings.max_response_tokens,
    )
    _query_engine = QueryEngine(config=query_config)


async def shutdown_dependencies() -> None:
    """Cleanup dependencies on application shutdown.

    Closes database connections and releases resources.
    """
    global _graph_connection, _graph_store, _ingestion_pipeline, _vector_store, _query_engine

    if _graph_connection is not None:
        await _graph_connection.close()
        _graph_connection = None

    _graph_store = None
    _ingestion_pipeline = None
    _vector_store = None
    _query_engine = None


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


async def get_vector_store() -> AsyncGenerator[VectorStore, None]:
    """Get the vector store for semantic search.

    Yields:
        The shared VectorStore instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _vector_store is None:
        raise RuntimeError(
            "Vector store not initialized. Ensure init_dependencies() was called on startup."
        )
    yield _vector_store


async def get_query_engine() -> AsyncGenerator[QueryEngine, None]:
    """Get the query engine for natural language queries.

    Yields:
        The shared QueryEngine instance.

    Raises:
        RuntimeError: If dependencies have not been initialized.
    """
    if _query_engine is None:
        raise RuntimeError(
            "Query engine not initialized. Ensure init_dependencies() was called on startup."
        )
    yield _query_engine


# Type aliases for commonly used dependencies
GraphConnectionDep = Annotated[GraphConnection, Depends(get_graph_connection)]
GraphStoreDep = Annotated[GraphStore, Depends(get_graph_store)]
IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
QueryEngineDep = Annotated[QueryEngine, Depends(get_query_engine)]
