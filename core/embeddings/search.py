"""Semantic search for code using embeddings.

This module provides high-level semantic search functionality
that combines embedding generation and vector search.
"""

from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.embeddings.chunker import ChunkerConfig, ChunkType, CodeChunk, CodeChunker
from core.embeddings.client import EmbeddingClient, create_embedding_client
from core.embeddings.models import EmbeddingConfig, EmbeddingProvider
from core.embeddings.store import VectorStore, VectorStoreConfig
from core.parser.models import CodeEntity

logger = structlog.get_logger(__name__)


class SemanticSearchConfig(BaseModel):
    """Configuration for semantic search.

    Attributes:
        embedding_provider: Embedding provider to use.
        embedding_model: Model name.
        vector_dimension: Vector dimension.
        qdrant_host: Qdrant host address.
        qdrant_port: Qdrant port.
        collection_name: Qdrant collection name.
        default_limit: Default number of results.
        score_threshold: Minimum similarity score.
    """

    model_config = ConfigDict(frozen=True)

    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.VOYAGE,
        description="Embedding provider",
    )
    embedding_model: str = Field(
        default="voyage-code-2",
        description="Model name",
    )
    vector_dimension: int = Field(
        default=1024,
        description="Vector dimension",
    )
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(default="code_chunks", description="Collection name")
    default_limit: int = Field(default=10, ge=1, le=100, description="Default limit")
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Score threshold")


class CodeSearchResult(BaseModel):
    """Enhanced search result with code context.

    Attributes:
        entity_name: Name of the code entity.
        entity_type: Type (function, class, method).
        file_path: Path to the file.
        start_line: Starting line number.
        end_line: Ending line number.
        language: Programming language.
        content: The matched code content.
        score: Similarity score.
        chunk_type: Type of chunk that matched.
        repo_id: Repository identifier.
        context: Additional context information.
    """

    model_config = ConfigDict(frozen=True)

    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    language: str = Field(..., description="Language")
    content: str = Field(..., description="Code content")
    score: float = Field(..., description="Similarity score")
    chunk_type: str = Field(..., description="Chunk type")
    repo_id: str = Field(..., description="Repository ID")
    context: dict[str, Any] = Field(default_factory=dict, description="Context")


class SemanticSearch:
    """High-level semantic code search.

    This class provides a unified interface for semantic code search,
    handling embedding generation, storage, and retrieval.
    """

    def __init__(
        self,
        api_key: str,
        config: SemanticSearchConfig | None = None,
    ) -> None:
        """Initialize semantic search.

        Args:
            api_key: API key for embedding provider.
            config: Search configuration.
        """
        self.config = config or SemanticSearchConfig()
        self._api_key = api_key
        self._embedding_client: EmbeddingClient | None = None
        self._vector_store: VectorStore | None = None
        self._chunker = CodeChunker(ChunkerConfig())
        self._logger = logger.bind(component="semantic_search")

    async def _get_embedding_client(self) -> EmbeddingClient:
        """Get or create embedding client.

        Returns:
            Embedding client.
        """
        if self._embedding_client is None:
            embedding_config = EmbeddingConfig(
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
                dimension=self.config.vector_dimension,
            )
            self._embedding_client = create_embedding_client(
                provider=self.config.embedding_provider,
                api_key=self._api_key,
                config=embedding_config,
            )
        return self._embedding_client

    async def _get_vector_store(self) -> VectorStore:
        """Get or create vector store.

        Returns:
            Vector store.
        """
        if self._vector_store is None:
            store_config = VectorStoreConfig(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                collection_name=self.config.collection_name,
                vector_size=self.config.vector_dimension,
            )
            self._vector_store = VectorStore(config=store_config)
        return self._vector_store

    async def close(self) -> None:
        """Close all clients and release resources."""
        if self._embedding_client:
            await self._embedding_client.close()
            self._embedding_client = None

        if self._vector_store:
            await self._vector_store.close()
            self._vector_store = None

    async def initialize(self, recreate: bool = False) -> None:
        """Initialize the search system.

        Creates the vector collection if it doesn't exist.

        Args:
            recreate: If True, recreate the collection.
        """
        store = await self._get_vector_store()
        await store.create_collection(recreate=recreate)
        self._logger.info("semantic_search_initialized")

    async def index_entities(
        self,
        entities: list[CodeEntity],
        repo_id: str,
    ) -> int:
        """Index code entities for semantic search.

        Args:
            entities: Code entities to index.
            repo_id: Repository identifier.

        Returns:
            Number of chunks indexed.
        """
        if not entities:
            return 0

        # Chunk entities
        chunks = self._chunker.chunk_entities(entities, repo_id)
        if not chunks:
            return 0

        # Generate embeddings
        client = await self._get_embedding_client()
        texts = [chunk.content for chunk in chunks]
        batch_result = await client.embed_batch(texts)

        # Store in vector database
        vectors = [emb.vector for emb in batch_result.embeddings]
        valid_chunks = [
            chunk for i, chunk in enumerate(chunks) if i not in batch_result.failed_indices
        ]

        if len(vectors) != len(valid_chunks):
            # Truncate to match
            min_len = min(len(vectors), len(valid_chunks))
            vectors = vectors[:min_len]
            valid_chunks = valid_chunks[:min_len]

        store = await self._get_vector_store()
        await store.upsert_chunks(valid_chunks, vectors)

        self._logger.info(
            "entities_indexed",
            entity_count=len(entities),
            chunk_count=len(valid_chunks),
            repo_id=repo_id,
        )
        return len(valid_chunks)

    async def index_chunk(
        self,
        chunk: CodeChunk,
    ) -> str:
        """Index a single code chunk.

        Args:
            chunk: Code chunk to index.

        Returns:
            Point ID.
        """
        client = await self._get_embedding_client()
        result = await client.embed(chunk.content)

        store = await self._get_vector_store()
        point_id = await store.upsert_chunk(chunk, result.vector)

        return point_id

    async def search(
        self,
        query: str,
        repo_id: str | None = None,
        limit: int | None = None,
        entity_type: str | None = None,
        chunk_type: ChunkType | None = None,
        language: str | None = None,
        file_path: str | None = None,
    ) -> list[CodeSearchResult]:
        """Search for code matching a natural language query.

        Args:
            query: Natural language search query.
            repo_id: Filter by repository.
            limit: Maximum number of results.
            entity_type: Filter by entity type.
            chunk_type: Filter by chunk type.
            language: Filter by programming language.
            file_path: Filter by file path.

        Returns:
            List of matching code results.
        """
        limit = limit or self.config.default_limit

        # Generate query embedding
        client = await self._get_embedding_client()
        query_result = await client.embed(query)

        # Search vector store
        store = await self._get_vector_store()
        results = await store.search(
            query_vector=query_result.vector,
            limit=limit,
            repo_id=repo_id,
            file_path=file_path,
            entity_type=entity_type,
            chunk_type=chunk_type,
            language=language,
            score_threshold=self.config.score_threshold,
        )

        # Convert to code search results
        code_results = [
            CodeSearchResult(
                entity_name=r.entity_name,
                entity_type=r.entity_type,
                file_path=r.file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                language=r.language,
                content=r.content,
                score=r.score,
                chunk_type=r.chunk_type,
                repo_id=r.repo_id,
                context=r.metadata,
            )
            for r in results
        ]

        self._logger.debug(
            "search_completed",
            query=query[:50],
            results=len(code_results),
        )
        return code_results

    async def search_similar_functions(
        self,
        function_code: str,
        repo_id: str,
        limit: int = 10,
    ) -> list[CodeSearchResult]:
        """Find functions similar to the given code.

        Args:
            function_code: Source code of the function.
            repo_id: Repository to search in.
            limit: Maximum results.

        Returns:
            List of similar functions.
        """
        return await self.search(
            query=function_code,
            repo_id=repo_id,
            limit=limit,
            entity_type="function",
            chunk_type=ChunkType.FUNCTION_FULL,
        )

    async def search_by_docstring(
        self,
        description: str,
        repo_id: str,
        limit: int = 10,
    ) -> list[CodeSearchResult]:
        """Find code by docstring/description similarity.

        Args:
            description: Natural language description.
            repo_id: Repository to search in.
            limit: Maximum results.

        Returns:
            List of matching code.
        """
        # Search docstring chunks specifically
        client = await self._get_embedding_client()
        query_result = await client.embed(description)

        store = await self._get_vector_store()

        # Search for docstring chunks
        docstring_results = await store.search(
            query_vector=query_result.vector,
            limit=limit,
            repo_id=repo_id,
            chunk_type=ChunkType.FUNCTION_DOCSTRING,
            score_threshold=self.config.score_threshold,
        )

        # Also search class docstrings
        class_docstring_results = await store.search(
            query_vector=query_result.vector,
            limit=limit,
            repo_id=repo_id,
            chunk_type=ChunkType.CLASS_DOCSTRING,
            score_threshold=self.config.score_threshold,
        )

        # Combine and sort by score
        all_results = list(docstring_results) + list(class_docstring_results)
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:limit]

        return [
            CodeSearchResult(
                entity_name=r.entity_name,
                entity_type=r.entity_type,
                file_path=r.file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                language=r.language,
                content=r.content,
                score=r.score,
                chunk_type=r.chunk_type,
                repo_id=r.repo_id,
                context=r.metadata,
            )
            for r in all_results
        ]

    async def delete_repository(self, repo_id: str) -> int:
        """Delete all indexed data for a repository.

        Args:
            repo_id: Repository identifier.

        Returns:
            Number of deleted chunks.
        """
        store = await self._get_vector_store()
        count = await store.delete_by_repo(repo_id)
        self._logger.info("repository_deleted", repo_id=repo_id, chunks_deleted=count)
        return count

    async def delete_file(self, repo_id: str, file_path: str) -> int:
        """Delete all indexed data for a file.

        Args:
            repo_id: Repository identifier.
            file_path: File path.

        Returns:
            Number of deleted chunks.
        """
        store = await self._get_vector_store()
        count = await store.delete_by_file(repo_id, file_path)
        self._logger.info(
            "file_deleted",
            repo_id=repo_id,
            file_path=file_path,
            chunks_deleted=count,
        )
        return count

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the search index.

        Returns:
            Statistics dictionary.
        """
        store = await self._get_vector_store()
        try:
            info = await store.get_collection_info()
            return {
                "collection": info["name"],
                "total_chunks": info["points_count"],
                "status": info["status"],
                "embedding_provider": self.config.embedding_provider.value,
                "embedding_model": self.config.embedding_model,
            }
        except Exception as e:
            return {"error": str(e)}
