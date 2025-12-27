"""Qdrant vector store for semantic code search.

This module provides the VectorStore class for storing and
searching code embeddings using Qdrant.
"""

import contextlib
from typing import Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from core.embeddings.chunker import ChunkType, CodeChunk

logger = structlog.get_logger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store errors."""

    pass


class CollectionNotFoundError(VectorStoreError):
    """Collection does not exist."""

    pass


class SearchResult(BaseModel):
    """Result from semantic search.

    Attributes:
        chunk_id: ID of the matched chunk.
        score: Similarity score (0-1).
        content: The matched text content.
        chunk_type: Type of chunk.
        repo_id: Repository ID.
        file_path: File path.
        entity_name: Entity name.
        entity_type: Entity type.
        start_line: Start line.
        end_line: End line.
        language: Programming language.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Chunk ID")
    score: float = Field(..., description="Similarity score")
    content: str = Field(default="", description="Text content")
    chunk_type: str = Field(..., description="Chunk type")
    repo_id: str = Field(..., description="Repository ID")
    file_path: str = Field(..., description="File path")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    language: str = Field(..., description="Language")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store.

    Attributes:
        host: Qdrant host address.
        port: Qdrant port.
        collection_name: Name of the collection.
        vector_size: Size of embedding vectors.
        distance: Distance metric for similarity.
        on_disk: Store vectors on disk.
    """

    model_config = ConfigDict(frozen=True)

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(default="code_chunks", description="Collection name")
    vector_size: int = Field(default=1024, description="Vector dimension")
    distance: str = Field(default="Cosine", description="Distance metric")
    on_disk: bool = Field(default=False, description="Store on disk")


class VectorStore:
    """Qdrant-based vector store for code embeddings.

    This class manages the storage and retrieval of code chunk
    embeddings using Qdrant vector database.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        """Initialize vector store.

        Args:
            config: Store configuration.
            client: Optional pre-configured Qdrant client.
        """
        self.config = config or VectorStoreConfig()
        self._client = client
        self._logger = logger.bind(component="vector_store")

    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create Qdrant client.

        Returns:
            Qdrant async client.
        """
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
            )
        return self._client

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            client = await self._get_client()
            # Try to get collections to verify connection
            await client.get_collections()
            return True
        except Exception as e:
            self._logger.error("health_check_failed", error=str(e))
            return False

    async def create_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """Create the code chunks collection.

        Args:
            recreate: If True, delete and recreate if exists.

        Raises:
            VectorStoreError: If creation fails.
        """
        client = await self._get_client()

        try:
            # Check if collection exists
            collections = await client.get_collections()
            exists = any(c.name == self.config.collection_name for c in collections.collections)

            if exists:
                if recreate:
                    await client.delete_collection(self.config.collection_name)
                    self._logger.info("collection_deleted", name=self.config.collection_name)
                else:
                    self._logger.info("collection_exists", name=self.config.collection_name)
                    return

            # Create collection with appropriate distance metric
            distance = models.Distance.COSINE
            if self.config.distance.upper() == "EUCLID":
                distance = models.Distance.EUCLID
            elif self.config.distance.upper() == "DOT":
                distance = models.Distance.DOT

            await client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.vector_size,
                    distance=distance,
                    on_disk=self.config.on_disk,
                ),
            )

            # Create payload indices for filtering
            await self._create_indices(client)

            self._logger.info(
                "collection_created",
                name=self.config.collection_name,
                vector_size=self.config.vector_size,
            )

        except Exception as e:
            raise VectorStoreError(f"Failed to create collection: {e}")

    async def _create_indices(self, client: AsyncQdrantClient) -> None:
        """Create payload indices for efficient filtering.

        Args:
            client: Qdrant client.
        """
        indices = [
            ("repo_id", models.PayloadSchemaType.KEYWORD),
            ("file_path", models.PayloadSchemaType.KEYWORD),
            ("entity_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
            ("language", models.PayloadSchemaType.KEYWORD),
        ]

        for field, schema_type in indices:
            with contextlib.suppress(UnexpectedResponse):
                await client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )

    async def delete_collection(self) -> None:
        """Delete the code chunks collection.

        Raises:
            VectorStoreError: If deletion fails.
        """
        client = await self._get_client()
        try:
            await client.delete_collection(self.config.collection_name)
            self._logger.info("collection_deleted", name=self.config.collection_name)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection: {e}")

    async def collection_exists(self) -> bool:
        """Check if the collection exists.

        Returns:
            True if collection exists.
        """
        client = await self._get_client()
        try:
            collections = await client.get_collections()
            return any(c.name == self.config.collection_name for c in collections.collections)
        except Exception:
            return False

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection information.

        Returns:
            Collection info dictionary.

        Raises:
            CollectionNotFoundError: If collection doesn't exist.
        """
        client = await self._get_client()
        try:
            info = await client.get_collection(self.config.collection_name)
            return {
                "name": self.config.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            }
        except UnexpectedResponse:
            raise CollectionNotFoundError(f"Collection '{self.config.collection_name}' not found")

    async def upsert_chunk(
        self,
        chunk: CodeChunk,
        vector: list[float],
    ) -> str:
        """Insert or update a single chunk.

        Args:
            chunk: Code chunk to store.
            vector: Embedding vector.

        Returns:
            Point ID.

        Raises:
            VectorStoreError: If upsert fails.
        """
        client = await self._get_client()

        # Generate deterministic ID from chunk ID
        point_id = str(uuid4())
        payload = chunk.to_payload()
        payload["content"] = chunk.content

        try:
            await client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            return point_id
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunk: {e}")

    async def upsert_chunks(
        self,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
        batch_size: int = 100,
    ) -> list[str]:
        """Insert or update multiple chunks.

        Args:
            chunks: Code chunks to store.
            vectors: Embedding vectors (same order as chunks).
            batch_size: Number of points per batch.

        Returns:
            List of point IDs.

        Raises:
            VectorStoreError: If upsert fails.
            ValueError: If chunks and vectors have different lengths.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors must have same length")

        if not chunks:
            return []

        client = await self._get_client()
        point_ids: list[str] = []

        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_vectors = vectors[batch_start:batch_end]

            points = []
            for chunk, vector in zip(batch_chunks, batch_vectors, strict=True):
                point_id = str(uuid4())
                payload = chunk.to_payload()
                payload["content"] = chunk.content
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )
                point_ids.append(point_id)

            try:
                await client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                )
            except Exception as e:
                raise VectorStoreError(f"Failed to upsert batch: {e}")

        self._logger.info(
            "chunks_upserted",
            count=len(chunks),
            batches=(len(chunks) + batch_size - 1) // batch_size,
        )
        return point_ids

    async def delete_by_repo(self, repo_id: str) -> int:
        """Delete all chunks for a repository.

        Args:
            repo_id: Repository identifier.

        Returns:
            Number of deleted points (approximate).

        Raises:
            VectorStoreError: If deletion fails.
        """
        client = await self._get_client()

        try:
            # Get count before deletion
            count_before = await self._count_by_filter(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_id",
                            match=models.MatchValue(value=repo_id),
                        )
                    ]
                )
            )

            # Delete by filter
            await client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="repo_id",
                                match=models.MatchValue(value=repo_id),
                            )
                        ]
                    )
                ),
            )

            self._logger.info("repo_chunks_deleted", repo_id=repo_id, count=count_before)
            return count_before

        except Exception as e:
            raise VectorStoreError(f"Failed to delete by repo: {e}")

    async def delete_by_file(self, repo_id: str, file_path: str) -> int:
        """Delete all chunks for a file.

        Args:
            repo_id: Repository identifier.
            file_path: File path.

        Returns:
            Number of deleted points (approximate).

        Raises:
            VectorStoreError: If deletion fails.
        """
        client = await self._get_client()

        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="repo_id",
                        match=models.MatchValue(value=repo_id),
                    ),
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_path),
                    ),
                ]
            )

            count_before = await self._count_by_filter(filter_condition)

            await client.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(filter=filter_condition),
            )

            self._logger.info(
                "file_chunks_deleted",
                repo_id=repo_id,
                file_path=file_path,
                count=count_before,
            )
            return count_before

        except Exception as e:
            raise VectorStoreError(f"Failed to delete by file: {e}")

    async def _count_by_filter(self, filter_: models.Filter) -> int:
        """Count points matching a filter.

        Args:
            filter_: Qdrant filter.

        Returns:
            Count of matching points.
        """
        client = await self._get_client()
        try:
            result = await client.count(
                collection_name=self.config.collection_name,
                count_filter=filter_,
                exact=False,
            )
            return int(result.count)
        except Exception:
            return 0

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        repo_id: str | None = None,
        file_path: str | None = None,
        entity_type: str | None = None,
        chunk_type: ChunkType | None = None,
        language: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar code chunks.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            repo_id: Filter by repository.
            file_path: Filter by file path.
            entity_type: Filter by entity type (function, class, etc.).
            chunk_type: Filter by chunk type.
            language: Filter by programming language.
            score_threshold: Minimum similarity score.

        Returns:
            List of search results sorted by score.

        Raises:
            VectorStoreError: If search fails.
        """
        client = await self._get_client()

        # Build filter conditions
        must_conditions: list[models.Condition] = []

        if repo_id:
            must_conditions.append(
                models.FieldCondition(
                    key="repo_id",
                    match=models.MatchValue(value=repo_id),
                )
            )

        if file_path:
            must_conditions.append(
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchValue(value=file_path),
                )
            )

        if entity_type:
            must_conditions.append(
                models.FieldCondition(
                    key="entity_type",
                    match=models.MatchValue(value=entity_type),
                )
            )

        if chunk_type:
            must_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type.value),
                )
            )

        if language:
            must_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            results = await client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            search_results = []
            for point in results:
                payload = point.payload or {}
                search_results.append(
                    SearchResult(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        score=point.score,
                        content=payload.get("content", ""),
                        chunk_type=payload.get("chunk_type", "unknown"),
                        repo_id=payload.get("repo_id", ""),
                        file_path=payload.get("file_path", ""),
                        entity_name=payload.get("entity_name", ""),
                        entity_type=payload.get("entity_type", ""),
                        start_line=payload.get("start_line", 0),
                        end_line=payload.get("end_line", 0),
                        language=payload.get("language", ""),
                        metadata={
                            k: v
                            for k, v in payload.items()
                            if k
                            not in {
                                "chunk_id",
                                "content",
                                "chunk_type",
                                "repo_id",
                                "file_path",
                                "entity_name",
                                "entity_type",
                                "start_line",
                                "end_line",
                                "language",
                            }
                        },
                    )
                )

            self._logger.debug(
                "search_completed",
                results=len(search_results),
                repo_id=repo_id,
            )
            return search_results

        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                raise CollectionNotFoundError(
                    f"Collection '{self.config.collection_name}' not found"
                )
            raise VectorStoreError(f"Search failed: {e}")
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")

    async def search_by_entity(
        self,
        query_vector: list[float],
        repo_id: str,
        entity_name: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for chunks related to a specific entity.

        Args:
            query_vector: Query embedding vector.
            repo_id: Repository identifier.
            entity_name: Entity name to search for.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        client = await self._get_client()

        try:
            results = await client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_id",
                            match=models.MatchValue(value=repo_id),
                        ),
                        models.FieldCondition(
                            key="entity_name",
                            match=models.MatchValue(value=entity_name),
                        ),
                    ]
                ),
                limit=limit,
                with_payload=True,
            )

            return [
                SearchResult(
                    chunk_id=point.payload.get("chunk_id", str(point.id)),
                    score=point.score,
                    content=point.payload.get("content", ""),
                    chunk_type=point.payload.get("chunk_type", "unknown"),
                    repo_id=point.payload.get("repo_id", ""),
                    file_path=point.payload.get("file_path", ""),
                    entity_name=point.payload.get("entity_name", ""),
                    entity_type=point.payload.get("entity_type", ""),
                    start_line=point.payload.get("start_line", 0),
                    end_line=point.payload.get("end_line", 0),
                    language=point.payload.get("language", ""),
                )
                for point in results
                if point.payload
            ]

        except Exception as e:
            raise VectorStoreError(f"Search by entity failed: {e}")

    async def get_chunks_for_file(
        self,
        repo_id: str,
        file_path: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all chunks for a specific file.

        Args:
            repo_id: Repository identifier.
            file_path: File path.
            limit: Maximum results.

        Returns:
            List of chunk payloads.
        """
        client = await self._get_client()

        try:
            results, _ = await client.scroll(
                collection_name=self.config.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="repo_id",
                            match=models.MatchValue(value=repo_id),
                        ),
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        ),
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [point.payload for point in results if point.payload]

        except Exception as e:
            raise VectorStoreError(f"Failed to get chunks for file: {e}")
