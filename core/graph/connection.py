"""Neo4j connection management.

This module provides the GraphConnection class for managing connections to
the Neo4j database. It supports async context management, connection pooling,
and health checks.
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError, ServiceUnavailable


class GraphConnectionError(Exception):
    """Exception raised for graph connection errors."""

    pass


class GraphConnection:
    """Manages connections to the Neo4j graph database.

    This class provides async connection management with connection pooling,
    health checks, and convenient query execution helpers.

    Attributes:
        uri: Neo4j connection URI.
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: float = 60.0,
    ) -> None:
        """Initialize the graph connection.

        Args:
            uri: Neo4j connection URI. Defaults to NEO4J_URI env var.
            user: Neo4j username. Defaults to NEO4J_USER env var.
            password: Neo4j password. Defaults to NEO4J_PASSWORD env var.
            database: Neo4j database name.
            max_connection_pool_size: Maximum connections in the pool.
            connection_acquisition_timeout: Timeout for acquiring a connection.
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self._max_pool_size = max_connection_pool_size
        self._acquisition_timeout = connection_acquisition_timeout
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j.

        Creates the async driver with connection pooling.

        Raises:
            GraphConnectionError: If connection fails.
        """
        if self._driver is not None:
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self._max_pool_size,
                connection_acquisition_timeout=self._acquisition_timeout,
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            raise GraphConnectionError(f"Failed to connect to Neo4j: {e}") from e
        except Exception as e:
            raise GraphConnectionError(f"Unexpected error connecting to Neo4j: {e}") from e

    async def close(self) -> None:
        """Close the Neo4j connection.

        Closes the driver and releases all resources.
        """
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    async def __aenter__(self) -> "GraphConnection":
        """Async context manager entry.

        Returns:
            Self after establishing connection.
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit.

        Closes the connection.
        """
        await self.close()

    @property
    def driver(self) -> AsyncDriver:
        """Get the Neo4j driver.

        Returns:
            The async Neo4j driver.

        Raises:
            GraphConnectionError: If not connected.
        """
        if self._driver is None:
            raise GraphConnectionError("Not connected to Neo4j. Call connect() first.")
        return self._driver

    @asynccontextmanager
    async def session(self, **kwargs: Any) -> AsyncGenerator[AsyncSession, None]:
        """Get a Neo4j session as an async context manager.

        Args:
            **kwargs: Additional session configuration.

        Yields:
            An async Neo4j session.

        Raises:
            GraphConnectionError: If not connected.
        """
        if self._driver is None:
            raise GraphConnectionError("Not connected to Neo4j. Call connect() first.")

        session = self._driver.session(database=self.database, **kwargs)
        try:
            yield session
        finally:
            await session.close()

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the Neo4j connection.

        Returns:
            Dictionary with health status information.
        """
        try:
            if self._driver is None:
                return {
                    "status": "disconnected",
                    "message": "Driver not initialized",
                }

            await self._driver.verify_connectivity()

            async with self.session() as session:
                result = await session.run("RETURN 1 as n")
                record = await result.single()

                if record and record["n"] == 1:
                    return {
                        "status": "healthy",
                        "uri": self.uri,
                        "database": self.database,
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": "Query returned unexpected result",
                    }
        except ServiceUnavailable as e:
            return {
                "status": "unhealthy",
                "message": f"Service unavailable: {e}",
            }
        except Neo4jError as e:
            return {
                "status": "unhealthy",
                "message": f"Neo4j error: {e}",
            }

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.

        This is a convenience method for simple queries. For transactions
        or more complex operations, use the session() context manager.

        Args:
            query: Cypher query string.
            parameters: Query parameters (parameterized queries for safety).
            **kwargs: Additional session configuration.

        Returns:
            List of result records as dictionaries.

        Raises:
            GraphConnectionError: If not connected.
            Neo4jError: If query execution fails.
        """
        if parameters is None:
            parameters = {}

        async with self.session(**kwargs) as session:
            result = await session.run(query, parameters)
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute a write query within a transaction.

        Wraps the query in an explicit write transaction for safety.

        Args:
            query: Cypher query string.
            parameters: Query parameters.
            **kwargs: Additional session configuration.

        Returns:
            List of result records as dictionaries.
        """
        if parameters is None:
            parameters = {}

        async with self.session(**kwargs) as session:

            async def _write_tx(tx: Any) -> list[dict[str, Any]]:
                result = await tx.run(query, parameters)
                return await result.data()

            return await session.execute_write(_write_tx)

    async def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute a read query within a transaction.

        Wraps the query in an explicit read transaction for optimization.

        Args:
            query: Cypher query string.
            parameters: Query parameters.
            **kwargs: Additional session configuration.

        Returns:
            List of result records as dictionaries.
        """
        if parameters is None:
            parameters = {}

        async with self.session(**kwargs) as session:

            async def _read_tx(tx: Any) -> list[dict[str, Any]]:
                result = await tx.run(query, parameters)
                return await result.data()

            return await session.execute_read(_read_tx)

    async def create_indexes(self) -> None:
        """Create required indexes for optimal query performance.

        Creates indexes on commonly queried properties as defined in the schema.
        """
        indexes = [
            # Repository indexes
            "CREATE INDEX repo_id IF NOT EXISTS FOR (r:Repository) ON (r.id)",
            "CREATE INDEX repo_name IF NOT EXISTS FOR (r:Repository) ON (r.name)",
            # File indexes
            "CREATE INDEX file_id IF NOT EXISTS FOR (f:File) ON (f.id)",
            "CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX file_repo IF NOT EXISTS FOR (f:File) ON (f.repo_id)",
            # Module indexes
            "CREATE INDEX module_id IF NOT EXISTS FOR (m:Module) ON (m.id)",
            "CREATE INDEX module_name IF NOT EXISTS FOR (m:Module) ON (m.name)",
            # Class indexes
            "CREATE INDEX class_id IF NOT EXISTS FOR (c:Class) ON (c.id)",
            "CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)",
            "CREATE INDEX class_file IF NOT EXISTS FOR (c:Class) ON (c.file_path)",
            # Function indexes
            "CREATE INDEX func_id IF NOT EXISTS FOR (f:Function) ON (f.id)",
            "CREATE INDEX func_name IF NOT EXISTS FOR (f:Function) ON (f.name)",
            "CREATE INDEX func_file IF NOT EXISTS FOR (f:Function) ON (f.file_path)",
            # Method indexes
            "CREATE INDEX method_id IF NOT EXISTS FOR (m:Method) ON (m.id)",
            "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)",
            "CREATE INDEX method_file IF NOT EXISTS FOR (m:Method) ON (m.file_path)",
            # Variable indexes
            "CREATE INDEX var_id IF NOT EXISTS FOR (v:Variable) ON (v.id)",
            "CREATE INDEX var_name IF NOT EXISTS FOR (v:Variable) ON (v.name)",
        ]

        for index_query in indexes:
            try:
                await self.execute_write(index_query)
            except Neo4jError as e:
                logger.warning(f"Failed to create index: {e}")

        logger.info("Graph indexes created/verified")

    async def create_constraints(self) -> None:
        """Create uniqueness constraints for node IDs.

        Ensures that node IDs are unique within their label.
        """
        constraints = [
            "CREATE CONSTRAINT repo_id_unique IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT module_id_unique IF NOT EXISTS FOR (m:Module) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT class_id_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT func_id_unique IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT method_id_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT var_id_unique IF NOT EXISTS FOR (v:Variable) REQUIRE v.id IS UNIQUE",
        ]

        for constraint_query in constraints:
            try:
                await self.execute_write(constraint_query)
            except Neo4jError as e:
                logger.warning(f"Failed to create constraint: {e}")

        logger.info("Graph constraints created/verified")
