"""Tests for the embeddings module.

This module tests chunking, embedding clients, vector store, and semantic search.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.embeddings.chunker import (
    ChunkerConfig,
    ChunkType,
    CodeChunk,
    CodeChunker,
)
from core.embeddings.client import (
    EmbeddingClient,
    EmbeddingClientError,
    OpenAIClient,
    VoyageClient,
    create_embedding_client,
)
from core.embeddings.models import (
    OPENAI_CONFIG,
    OPENAI_TEXT_EMBEDDING_3_LARGE,
    VOYAGE_CODE_2,
    VOYAGE_CODE_CONFIG,
    EmbeddingBatchResult,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingProvider,
    EmbeddingResult,
)
from core.embeddings.search import (
    CodeSearchResult,
    SemanticSearch,
    SemanticSearchConfig,
)
from core.embeddings.store import (
    CollectionNotFoundError,
    SearchResult,
    VectorStore,
    VectorStoreConfig,
    VectorStoreError,
)
from core.parser.models import ClassEntity, FunctionEntity, MethodEntity, Parameter

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_function_entity() -> FunctionEntity:
    """Create a sample function entity for testing."""
    return FunctionEntity(
        id="test.py:example_function:10",
        name="example_function",
        file_path="test.py",
        start_line=10,
        end_line=20,
        source_code="""def example_function(x: int, y: str) -> bool:
    '''Example function that does something.

    Args:
        x: An integer value.
        y: A string value.

    Returns:
        True if successful.
    '''
    return x > 0 and len(y) > 0""",
        docstring="Example function that does something.",
        language="python",
        is_async=False,
        parameters=[
            Parameter(name="x", type_annotation="int"),
            Parameter(name="y", type_annotation="str"),
        ],
        return_type="bool",
        decorators=[],
        calls=[],
    )


@pytest.fixture
def sample_async_function_entity() -> FunctionEntity:
    """Create a sample async function entity for testing."""
    return FunctionEntity(
        id="test.py:async_handler:30",
        name="async_handler",
        file_path="test.py",
        start_line=30,
        end_line=40,
        source_code="""async def async_handler(request: Request) -> Response:
    '''Handle an async request.'''
    data = await request.json()
    return Response(data)""",
        docstring="Handle an async request.",
        language="python",
        is_async=True,
        parameters=[
            Parameter(name="request", type_annotation="Request"),
        ],
        return_type="Response",
        decorators=["route('/handle')"],
        calls=["request.json", "Response"],
    )


@pytest.fixture
def sample_class_entity() -> ClassEntity:
    """Create a sample class entity for testing."""
    method = MethodEntity(
        id="test.py:Calculator.add:15",
        name="add",
        file_path="test.py",
        start_line=15,
        end_line=17,
        source_code="""def add(self, a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b""",
        docstring="Add two numbers.",
        language="python",
        is_async=False,
        parameters=[
            Parameter(name="self"),
            Parameter(name="a", type_annotation="int"),
            Parameter(name="b", type_annotation="int"),
        ],
        return_type="int",
        decorators=[],
        calls=[],
    )

    return ClassEntity(
        id="test.py:Calculator:10",
        name="Calculator",
        file_path="test.py",
        start_line=10,
        end_line=25,
        source_code="""class Calculator:
    '''A simple calculator class.'''

    def add(self, a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    def subtract(self, a: int, b: int) -> int:
        '''Subtract two numbers.'''
        return a - b""",
        docstring="A simple calculator class.",
        language="python",
        bases=["BaseCalculator"],
        decorators=["dataclass"],
        methods=[method],
    )


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    mock = AsyncMock()
    mock.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    mock.create_collection = AsyncMock()
    mock.delete_collection = AsyncMock()
    mock.upsert = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.delete = AsyncMock()
    mock.count = AsyncMock(return_value=MagicMock(count=0))
    mock.scroll = AsyncMock(return_value=([], None))
    mock.close = AsyncMock()
    return mock


# =============================================================================
# Embedding Models Tests
# =============================================================================


class TestEmbeddingModels:
    """Tests for embedding model definitions."""

    def test_embedding_provider_enum(self):
        """Test EmbeddingProvider enum values."""
        assert EmbeddingProvider.VOYAGE == "voyage"
        assert EmbeddingProvider.OPENAI == "openai"

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.provider == EmbeddingProvider.VOYAGE
        assert config.model == "voyage-code-2"
        assert config.dimension == 1024
        assert config.batch_size == 128
        assert config.max_retries == 3

    def test_embedding_config_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model="text-embedding-3-large",
            dimension=3072,
            batch_size=100,
            max_retries=5,
        )
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.model == "text-embedding-3-large"
        assert config.dimension == 3072

    def test_voyage_config_preset(self):
        """Test Voyage preset configuration."""
        assert VOYAGE_CODE_CONFIG.provider == EmbeddingProvider.VOYAGE
        assert VOYAGE_CODE_CONFIG.model == "voyage-code-2"
        assert VOYAGE_CODE_CONFIG.dimension == 1024

    def test_openai_config_preset(self):
        """Test OpenAI preset configuration."""
        assert OPENAI_CONFIG.provider == EmbeddingProvider.OPENAI
        assert OPENAI_CONFIG.model == "text-embedding-3-large"
        assert OPENAI_CONFIG.dimension == 3072

    def test_embedding_result(self):
        """Test EmbeddingResult model."""
        result = EmbeddingResult(
            text="test text",
            vector=[0.1, 0.2, 0.3],
            model="voyage-code-2",
            token_count=5,
            metadata={"key": "value"},
        )
        assert result.text == "test text"
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.model == "voyage-code-2"
        assert result.token_count == 5
        assert result.metadata == {"key": "value"}

    def test_embedding_batch_result(self):
        """Test EmbeddingBatchResult model."""
        result = EmbeddingBatchResult(
            embeddings=[
                EmbeddingResult(text="a", vector=[0.1], model="test"),
                EmbeddingResult(text="b", vector=[0.2], model="test"),
            ],
            total_tokens=10,
            model="test",
            failed_indices=[2],
        )
        assert len(result.embeddings) == 2
        assert result.total_tokens == 10
        assert result.failed_indices == [2]

    def test_embedding_model_info(self):
        """Test EmbeddingModel definitions."""
        assert VOYAGE_CODE_2.provider == EmbeddingProvider.VOYAGE
        assert VOYAGE_CODE_2.model_name == "voyage-code-2"
        assert VOYAGE_CODE_2.dimension == 1024
        assert VOYAGE_CODE_2.max_tokens == 16000

        assert OPENAI_TEXT_EMBEDDING_3_LARGE.provider == EmbeddingProvider.OPENAI
        assert OPENAI_TEXT_EMBEDDING_3_LARGE.dimension == 3072


# =============================================================================
# Code Chunker Tests
# =============================================================================


class TestChunkerConfig:
    """Tests for chunker configuration."""

    def test_default_config(self):
        """Test default chunker configuration."""
        config = ChunkerConfig()
        assert config.max_chunk_tokens == 2048
        assert config.include_signatures is True
        assert config.include_docstrings is True
        assert config.include_full_source is True
        assert config.overlap_lines == 0

    def test_custom_config(self):
        """Test custom chunker configuration."""
        config = ChunkerConfig(
            max_chunk_tokens=1024,
            include_signatures=False,
            include_docstrings=True,
            include_full_source=False,
        )
        assert config.max_chunk_tokens == 1024
        assert config.include_signatures is False


class TestCodeChunk:
    """Tests for CodeChunk model."""

    def test_code_chunk_creation(self):
        """Test creating a code chunk."""
        chunk = CodeChunk(
            id="repo:test.py:func:10:full",
            content="def func(): pass",
            chunk_type=ChunkType.FUNCTION_FULL,
            repo_id="repo",
            file_path="test.py",
            entity_name="func",
            entity_type="function",
            start_line=10,
            end_line=10,
            language="python",
        )
        assert chunk.id == "repo:test.py:func:10:full"
        assert chunk.chunk_type == ChunkType.FUNCTION_FULL

    def test_to_payload(self):
        """Test chunk to payload conversion."""
        chunk = CodeChunk(
            id="repo:test.py:func:10:full",
            content="def func(): pass",
            chunk_type=ChunkType.FUNCTION_FULL,
            repo_id="repo",
            file_path="test.py",
            entity_name="func",
            entity_type="function",
            start_line=10,
            end_line=15,
            language="python",
            metadata={"is_async": False},
        )
        payload = chunk.to_payload()

        assert payload["chunk_id"] == "repo:test.py:func:10:full"
        assert payload["chunk_type"] == "function_full"
        assert payload["repo_id"] == "repo"
        assert payload["file_path"] == "test.py"
        assert payload["entity_name"] == "func"
        assert payload["entity_type"] == "function"
        assert payload["start_line"] == 10
        assert payload["end_line"] == 15
        assert payload["language"] == "python"
        assert payload["is_async"] is False


class TestCodeChunker:
    """Tests for CodeChunker functionality."""

    def test_chunker_initialization(self):
        """Test chunker initialization."""
        chunker = CodeChunker()
        assert chunker.config is not None

    def test_chunker_with_custom_config(self):
        """Test chunker with custom config."""
        config = ChunkerConfig(include_signatures=False)
        chunker = CodeChunker(config)
        assert chunker.config.include_signatures is False

    def test_chunk_function_full(self, sample_function_entity):
        """Test chunking a function with full source."""
        chunker = CodeChunker()
        chunks = chunker.chunk_function(sample_function_entity, "test-repo")

        # Should have full, signature, and docstring chunks
        assert len(chunks) == 3

        chunk_types = {c.chunk_type for c in chunks}
        assert ChunkType.FUNCTION_FULL in chunk_types
        assert ChunkType.FUNCTION_SIGNATURE in chunk_types
        assert ChunkType.FUNCTION_DOCSTRING in chunk_types

    def test_chunk_function_only_full(self, sample_function_entity):
        """Test chunking with only full source enabled."""
        config = ChunkerConfig(
            include_signatures=False,
            include_docstrings=False,
            include_full_source=True,
        )
        chunker = CodeChunker(config)
        chunks = chunker.chunk_function(sample_function_entity, "test-repo")

        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.FUNCTION_FULL

    def test_chunk_async_function(self, sample_async_function_entity):
        """Test chunking an async function."""
        chunker = CodeChunker()
        chunks = chunker.chunk_function(sample_async_function_entity, "test-repo")

        # Find the full chunk
        full_chunk = next(c for c in chunks if c.chunk_type == ChunkType.FUNCTION_FULL)
        assert full_chunk.metadata["is_async"] is True

        # Find the signature chunk
        sig_chunk = next(c for c in chunks if c.chunk_type == ChunkType.FUNCTION_SIGNATURE)
        assert "async " in sig_chunk.content

    def test_chunk_class(self, sample_class_entity):
        """Test chunking a class entity."""
        chunker = CodeChunker()
        chunks = chunker.chunk_class(sample_class_entity, "test-repo")

        chunk_types = [c.chunk_type for c in chunks]

        # Should have class chunks
        assert ChunkType.CLASS_FULL in chunk_types
        assert ChunkType.CLASS_SIGNATURE in chunk_types
        assert ChunkType.CLASS_DOCSTRING in chunk_types

        # Should also have method chunks
        assert ChunkType.METHOD_FULL in chunk_types

    def test_chunk_class_signature_with_bases(self, sample_class_entity):
        """Test class signature includes inheritance."""
        chunker = CodeChunker()
        chunks = chunker.chunk_class(sample_class_entity, "test-repo")

        sig_chunk = next(c for c in chunks if c.chunk_type == ChunkType.CLASS_SIGNATURE)
        assert "BaseCalculator" in sig_chunk.content
        assert "@dataclass" in sig_chunk.content

    def test_chunk_method(self, sample_class_entity):
        """Test chunking a method."""
        chunker = CodeChunker()
        method = sample_class_entity.methods[0]
        chunks = chunker.chunk_method(method, "Calculator", "test-repo")

        assert len(chunks) >= 1
        full_chunk = next(c for c in chunks if c.chunk_type == ChunkType.METHOD_FULL)
        assert "Calculator" in full_chunk.entity_name
        assert full_chunk.metadata["class_name"] == "Calculator"

    def test_chunk_entity_function(self, sample_function_entity):
        """Test chunk_entity dispatches correctly for functions."""
        chunker = CodeChunker()
        chunks = chunker.chunk_entity(sample_function_entity, "test-repo")

        assert len(chunks) > 0
        assert any(c.chunk_type == ChunkType.FUNCTION_FULL for c in chunks)

    def test_chunk_entity_class(self, sample_class_entity):
        """Test chunk_entity dispatches correctly for classes."""
        chunker = CodeChunker()
        chunks = chunker.chunk_entity(sample_class_entity, "test-repo")

        assert len(chunks) > 0
        assert any(c.chunk_type == ChunkType.CLASS_FULL for c in chunks)

    def test_chunk_entities_multiple(self, sample_function_entity, sample_class_entity):
        """Test chunking multiple entities."""
        chunker = CodeChunker()
        entities = [sample_function_entity, sample_class_entity]
        chunks = chunker.chunk_entities(entities, "test-repo")

        # Should have chunks from both entities
        assert len(chunks) > 4
        assert any(c.chunk_type == ChunkType.FUNCTION_FULL for c in chunks)
        assert any(c.chunk_type == ChunkType.CLASS_FULL for c in chunks)

    def test_chunk_entities_empty_list(self):
        """Test chunking empty list."""
        chunker = CodeChunker()
        chunks = chunker.chunk_entities([], "test-repo")
        assert chunks == []

    def test_build_function_signature(self, sample_function_entity):
        """Test building function signature."""
        chunker = CodeChunker()
        signature = chunker._build_function_signature(sample_function_entity)

        assert "def example_function" in signature
        assert "x: int" in signature
        assert "y: str" in signature
        assert "-> bool" in signature

    def test_build_async_function_signature(self, sample_async_function_entity):
        """Test building async function signature."""
        chunker = CodeChunker()
        signature = chunker._build_function_signature(sample_async_function_entity)

        assert signature.startswith("async def")

    def test_build_class_signature(self, sample_class_entity):
        """Test building class signature."""
        chunker = CodeChunker()
        signature = chunker._build_class_signature(sample_class_entity)

        assert "class Calculator" in signature
        assert "BaseCalculator" in signature
        assert "@dataclass" in signature


# =============================================================================
# Embedding Client Tests
# =============================================================================


class TestVoyageClient:
    """Tests for Voyage embedding client."""

    def test_voyage_client_initialization(self):
        """Test Voyage client initialization."""
        client = VoyageClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client.config == VOYAGE_CODE_CONFIG

    def test_voyage_client_custom_config(self):
        """Test Voyage client with custom config."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.VOYAGE,
            model="voyage-2",
            batch_size=64,
        )
        client = VoyageClient("test-api-key", config)
        assert client.config.model == "voyage-2"
        assert client.config.batch_size == 64

    @pytest.mark.asyncio
    async def test_voyage_embed_batch_empty(self):
        """Test embedding empty batch."""
        client = VoyageClient("test-api-key")
        result = await client.embed_batch([])
        assert result.embeddings == []
        await client.close()

    @pytest.mark.asyncio
    async def test_voyage_client_close(self):
        """Test closing Voyage client."""
        client = VoyageClient("test-api-key")
        await client.close()
        # Should not raise


class TestOpenAIClient:
    """Tests for OpenAI embedding client."""

    def test_openai_client_initialization(self):
        """Test OpenAI client initialization."""
        client = OpenAIClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client.config == OPENAI_CONFIG

    def test_openai_count_tokens(self):
        """Test token counting."""
        client = OpenAIClient("test-api-key")
        count = client.count_tokens("Hello world")
        assert count >= 2

    @pytest.mark.asyncio
    async def test_openai_embed_batch_empty(self):
        """Test embedding empty batch."""
        client = OpenAIClient("test-api-key")
        result = await client.embed_batch([])
        assert result.embeddings == []
        await client.close()


class TestCreateEmbeddingClient:
    """Tests for embedding client factory."""

    def test_create_voyage_client(self):
        """Test creating Voyage client."""
        client = create_embedding_client(
            EmbeddingProvider.VOYAGE,
            "test-api-key",
        )
        assert isinstance(client, VoyageClient)

    def test_create_openai_client(self):
        """Test creating OpenAI client."""
        client = create_embedding_client(
            EmbeddingProvider.OPENAI,
            "test-api-key",
        )
        assert isinstance(client, OpenAIClient)

    def test_create_client_with_config(self):
        """Test creating client with custom config."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.VOYAGE,
            batch_size=50,
        )
        client = create_embedding_client(
            EmbeddingProvider.VOYAGE,
            "test-api-key",
            config,
        )
        assert client.config.batch_size == 50


# =============================================================================
# Vector Store Tests
# =============================================================================


class TestVectorStoreConfig:
    """Tests for vector store configuration."""

    def test_default_config(self):
        """Test default store configuration."""
        config = VectorStoreConfig()
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.collection_name == "code_chunks"
        assert config.vector_size == 1024
        assert config.distance == "Cosine"

    def test_custom_config(self):
        """Test custom store configuration."""
        config = VectorStoreConfig(
            host="qdrant.example.com",
            port=6334,
            collection_name="my_chunks",
            vector_size=3072,
            distance="Euclid",
        )
        assert config.host == "qdrant.example.com"
        assert config.port == 6334
        assert config.vector_size == 3072


class TestSearchResult:
    """Tests for search result model."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            chunk_id="test:id",
            score=0.95,
            content="def func(): pass",
            chunk_type="function_full",
            repo_id="repo",
            file_path="test.py",
            entity_name="func",
            entity_type="function",
            start_line=1,
            end_line=1,
            language="python",
        )
        assert result.score == 0.95
        assert result.chunk_type == "function_full"


class TestVectorStore:
    """Tests for vector store operations."""

    def test_store_initialization(self):
        """Test store initialization."""
        store = VectorStore()
        assert store.config is not None

    def test_store_custom_config(self):
        """Test store with custom config."""
        config = VectorStoreConfig(host="custom-host")
        store = VectorStore(config)
        assert store.config.host == "custom-host"

    @pytest.mark.asyncio
    async def test_store_with_mock_client(self, mock_qdrant_client):
        """Test store operations with mock client."""
        store = VectorStore(client=mock_qdrant_client)
        assert store._client is mock_qdrant_client

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_qdrant_client):
        """Test health check returns True on success."""
        store = VectorStore(client=mock_qdrant_client)
        result = await store.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_qdrant_client):
        """Test health check returns False on failure."""
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")
        store = VectorStore(client=mock_qdrant_client)
        result = await store.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_create_collection(self, mock_qdrant_client):
        """Test creating a collection."""
        store = VectorStore(client=mock_qdrant_client)
        await store.create_collection()
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, mock_qdrant_client):
        """Test creating collection that already exists."""
        mock_collection = MagicMock()
        mock_collection.name = "code_chunks"
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[mock_collection])

        store = VectorStore(client=mock_qdrant_client)
        await store.create_collection(recreate=False)

        # Should not create since it exists
        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_collection_recreate(self, mock_qdrant_client):
        """Test recreating an existing collection."""
        mock_collection = MagicMock()
        mock_collection.name = "code_chunks"
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[mock_collection])

        store = VectorStore(client=mock_qdrant_client)
        await store.create_collection(recreate=True)

        mock_qdrant_client.delete_collection.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_collection_exists(self, mock_qdrant_client):
        """Test checking if collection exists."""
        mock_collection = MagicMock()
        mock_collection.name = "code_chunks"
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[mock_collection])

        store = VectorStore(client=mock_qdrant_client)
        result = await store.collection_exists()
        assert result is True

    @pytest.mark.asyncio
    async def test_collection_not_exists(self, mock_qdrant_client):
        """Test collection doesn't exist."""
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])

        store = VectorStore(client=mock_qdrant_client)
        result = await store.collection_exists()
        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_chunk(self, mock_qdrant_client):
        """Test upserting a single chunk."""
        chunk = CodeChunk(
            id="test:id",
            content="def func(): pass",
            chunk_type=ChunkType.FUNCTION_FULL,
            repo_id="repo",
            file_path="test.py",
            entity_name="func",
            entity_type="function",
            start_line=1,
            end_line=1,
            language="python",
        )

        store = VectorStore(client=mock_qdrant_client)
        point_id = await store.upsert_chunk(chunk, [0.1, 0.2, 0.3])

        assert point_id is not None
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_chunks_batch(self, mock_qdrant_client):
        """Test upserting multiple chunks."""
        chunks = [
            CodeChunk(
                id=f"test:id:{i}",
                content=f"def func{i}(): pass",
                chunk_type=ChunkType.FUNCTION_FULL,
                repo_id="repo",
                file_path="test.py",
                entity_name=f"func{i}",
                entity_type="function",
                start_line=i,
                end_line=i,
                language="python",
            )
            for i in range(3)
        ]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(3)]

        store = VectorStore(client=mock_qdrant_client)
        point_ids = await store.upsert_chunks(chunks, vectors)

        assert len(point_ids) == 3
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_chunks_mismatched_lengths(self, mock_qdrant_client):
        """Test upserting with mismatched lengths raises error."""
        chunks = [
            CodeChunk(
                id="test:id",
                content="code",
                chunk_type=ChunkType.FUNCTION_FULL,
                repo_id="repo",
                file_path="test.py",
                entity_name="func",
                entity_type="function",
                start_line=1,
                end_line=1,
                language="python",
            )
        ]
        vectors = [[0.1], [0.2]]  # 2 vectors for 1 chunk

        store = VectorStore(client=mock_qdrant_client)
        with pytest.raises(ValueError):
            await store.upsert_chunks(chunks, vectors)

    @pytest.mark.asyncio
    async def test_delete_by_repo(self, mock_qdrant_client):
        """Test deleting chunks by repository."""
        store = VectorStore(client=mock_qdrant_client)
        count = await store.delete_by_repo("test-repo")

        assert count == 0  # Mock returns 0
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_file(self, mock_qdrant_client):
        """Test deleting chunks by file."""
        store = VectorStore(client=mock_qdrant_client)
        count = await store.delete_by_file("test-repo", "test.py")

        assert count == 0
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_basic(self, mock_qdrant_client):
        """Test basic search."""
        mock_point = MagicMock()
        mock_point.id = "point-1"
        mock_point.score = 0.95
        mock_point.payload = {
            "chunk_id": "test:id",
            "content": "def func(): pass",
            "chunk_type": "function_full",
            "repo_id": "repo",
            "file_path": "test.py",
            "entity_name": "func",
            "entity_type": "function",
            "start_line": 1,
            "end_line": 1,
            "language": "python",
        }
        mock_qdrant_client.search.return_value = [mock_point]

        store = VectorStore(client=mock_qdrant_client)
        results = await store.search([0.1, 0.2, 0.3], limit=10)

        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].entity_name == "func"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_qdrant_client):
        """Test search with filters."""
        mock_qdrant_client.search.return_value = []

        store = VectorStore(client=mock_qdrant_client)
        await store.search(
            [0.1, 0.2, 0.3],
            repo_id="repo",
            entity_type="function",
            chunk_type=ChunkType.FUNCTION_FULL,
            language="python",
        )

        # Verify search was called with filter
        call_args = mock_qdrant_client.search.call_args
        assert call_args.kwargs["query_filter"] is not None

    @pytest.mark.asyncio
    async def test_close(self, mock_qdrant_client):
        """Test closing the store."""
        store = VectorStore(client=mock_qdrant_client)
        await store.close()
        mock_qdrant_client.close.assert_called_once()


# =============================================================================
# Semantic Search Tests
# =============================================================================


class TestSemanticSearchConfig:
    """Tests for semantic search configuration."""

    def test_default_config(self):
        """Test default search configuration."""
        config = SemanticSearchConfig()
        assert config.embedding_provider == EmbeddingProvider.VOYAGE
        assert config.embedding_model == "voyage-code-2"
        assert config.vector_dimension == 1024
        assert config.qdrant_host == "localhost"
        assert config.default_limit == 10
        assert config.score_threshold == 0.5

    def test_custom_config(self):
        """Test custom search configuration."""
        config = SemanticSearchConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            embedding_model="text-embedding-3-large",
            vector_dimension=3072,
            score_threshold=0.7,
        )
        assert config.embedding_provider == EmbeddingProvider.OPENAI
        assert config.score_threshold == 0.7


class TestCodeSearchResult:
    """Tests for code search result model."""

    def test_code_search_result(self):
        """Test creating a code search result."""
        result = CodeSearchResult(
            entity_name="example_function",
            entity_type="function",
            file_path="test.py",
            start_line=10,
            end_line=20,
            language="python",
            content="def example_function(): pass",
            score=0.92,
            chunk_type="function_full",
            repo_id="test-repo",
            context={"is_async": False},
        )
        assert result.entity_name == "example_function"
        assert result.score == 0.92
        assert result.context["is_async"] is False


class TestSemanticSearch:
    """Tests for semantic search functionality."""

    @pytest.fixture
    def mock_embedding_client(self):
        """Create mock embedding client."""
        mock = AsyncMock()
        mock.embed = AsyncMock(
            return_value=EmbeddingResult(
                text="query",
                vector=[0.1] * 1024,
                model="test",
            )
        )
        mock.embed_batch = AsyncMock(
            return_value=EmbeddingBatchResult(
                embeddings=[
                    EmbeddingResult(text="a", vector=[0.1] * 1024, model="test"),
                ],
                total_tokens=10,
                model="test",
            )
        )
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock = AsyncMock()
        mock.create_collection = AsyncMock()
        mock.upsert_chunks = AsyncMock(return_value=["id1"])
        mock.upsert_chunk = AsyncMock(return_value="id1")
        mock.search = AsyncMock(return_value=[])
        mock.delete_by_repo = AsyncMock(return_value=5)
        mock.delete_by_file = AsyncMock(return_value=2)
        mock.get_collection_info = AsyncMock(
            return_value={
                "name": "code_chunks",
                "points_count": 100,
                "status": "green",
            }
        )
        mock.close = AsyncMock()
        return mock

    def test_semantic_search_initialization(self):
        """Test semantic search initialization."""
        search = SemanticSearch("test-api-key")
        assert search.config is not None

    def test_semantic_search_custom_config(self):
        """Test semantic search with custom config."""
        config = SemanticSearchConfig(
            embedding_provider=EmbeddingProvider.OPENAI,
            score_threshold=0.8,
        )
        search = SemanticSearch("test-api-key", config)
        assert search.config.embedding_provider == EmbeddingProvider.OPENAI

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing semantic search."""
        search = SemanticSearch("test-api-key")
        await search.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_index_entities_empty(self):
        """Test indexing empty list."""
        search = SemanticSearch("test-api-key")
        count = await search.index_entities([], "test-repo")
        assert count == 0
        await search.close()


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestEmbeddingsIntegration:
    """Integration tests for the embeddings module."""

    def test_chunk_and_convert(self, sample_function_entity):
        """Test chunking and converting to payloads."""
        chunker = CodeChunker()
        chunks = chunker.chunk_function(sample_function_entity, "test-repo")

        # Verify all chunks can convert to payloads
        for chunk in chunks:
            payload = chunk.to_payload()
            assert "chunk_id" in payload
            assert "repo_id" in payload
            assert payload["repo_id"] == "test-repo"

    def test_chunk_types_enum_values(self):
        """Test all chunk type enum values."""
        expected_types = {
            "function_full",
            "function_signature",
            "function_docstring",
            "class_full",
            "class_signature",
            "class_docstring",
            "method_full",
            "method_signature",
            "file_summary",
            "import_block",
        }
        actual_types = {ct.value for ct in ChunkType}
        assert expected_types == actual_types

    def test_embedding_providers_supported(self):
        """Test all embedding providers can create clients."""
        for provider in EmbeddingProvider:
            client = create_embedding_client(provider, "test-key")
            assert client is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_vector_store_error(self):
        """Test VectorStoreError exception."""
        error = VectorStoreError("Test error")
        assert str(error) == "Test error"

    def test_collection_not_found_error(self):
        """Test CollectionNotFoundError exception."""
        error = CollectionNotFoundError("Collection 'test' not found")
        assert "Collection" in str(error)

    def test_embedding_client_error(self):
        """Test EmbeddingClientError exception."""
        error = EmbeddingClientError("API error")
        assert str(error) == "API error"
