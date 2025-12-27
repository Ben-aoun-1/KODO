"""Code chunking strategies for embeddings.

This module provides chunking functionality to break code into
meaningful segments for embedding generation.
"""

from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.parser.models import ClassEntity, CodeEntity, FunctionEntity, MethodEntity

logger = structlog.get_logger(__name__)


class ChunkType(str, Enum):
    """Types of code chunks for embedding."""

    FUNCTION_FULL = "function_full"
    FUNCTION_SIGNATURE = "function_signature"
    FUNCTION_DOCSTRING = "function_docstring"
    CLASS_FULL = "class_full"
    CLASS_SIGNATURE = "class_signature"
    CLASS_DOCSTRING = "class_docstring"
    METHOD_FULL = "method_full"
    METHOD_SIGNATURE = "method_signature"
    FILE_SUMMARY = "file_summary"
    IMPORT_BLOCK = "import_block"


class CodeChunk(BaseModel):
    """A chunk of code ready for embedding.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content to embed.
        chunk_type: Type of chunk.
        repo_id: Repository identifier.
        file_path: Path to source file.
        entity_name: Name of the code entity.
        entity_type: Type of code entity (function, class, etc.).
        start_line: Starting line number.
        end_line: Ending line number.
        language: Programming language.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Text content")
    chunk_type: ChunkType = Field(..., description="Chunk type")
    repo_id: str = Field(..., description="Repository ID")
    file_path: str = Field(..., description="File path")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    language: str = Field(..., description="Language")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def to_payload(self) -> dict[str, Any]:
        """Convert chunk to Qdrant payload format.

        Returns:
            Dictionary suitable for Qdrant storage.
        """
        return {
            "chunk_id": self.id,
            "chunk_type": self.chunk_type.value,
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            **self.metadata,
        }


class ChunkerConfig(BaseModel):
    """Configuration for code chunking.

    Attributes:
        max_chunk_tokens: Maximum tokens per chunk.
        include_signatures: Include function/class signatures.
        include_docstrings: Include docstrings separately.
        include_full_source: Include full source code.
        overlap_lines: Lines of overlap between chunks.
    """

    model_config = ConfigDict(frozen=True)

    max_chunk_tokens: int = Field(
        default=2048,
        ge=256,
        le=8192,
        description="Max tokens per chunk",
    )
    include_signatures: bool = Field(
        default=True,
        description="Include signatures",
    )
    include_docstrings: bool = Field(
        default=True,
        description="Include docstrings",
    )
    include_full_source: bool = Field(
        default=True,
        description="Include full source",
    )
    overlap_lines: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Overlap lines",
    )


class CodeChunker:
    """Chunks code entities for embedding generation.

    This class provides methods to break code into meaningful chunks
    that can be embedded for semantic search.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialize the code chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config or ChunkerConfig()
        self._logger = logger.bind(component="chunker")

    def chunk_function(
        self,
        entity: FunctionEntity,
        repo_id: str,
    ) -> list[CodeChunk]:
        """Chunk a function entity.

        Args:
            entity: Function entity to chunk.
            repo_id: Repository identifier.

        Returns:
            List of code chunks for the function.
        """
        chunks: list[CodeChunk] = []
        base_id = f"{repo_id}:{entity.file_path}:{entity.name}:{entity.start_line}"

        # Full function source
        if self.config.include_full_source and entity.source_code:
            chunks.append(
                CodeChunk(
                    id=f"{base_id}:full",
                    content=entity.source_code,
                    chunk_type=ChunkType.FUNCTION_FULL,
                    repo_id=repo_id,
                    file_path=entity.file_path,
                    entity_name=entity.name,
                    entity_type="function",
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    language=entity.language,
                    metadata={
                        "is_async": entity.is_async,
                        "parameters": [p.name for p in entity.parameters],
                        "return_type": entity.return_type,
                    },
                )
            )

        # Function signature
        if self.config.include_signatures:
            signature = self._build_function_signature(entity)
            if signature:
                chunks.append(
                    CodeChunk(
                        id=f"{base_id}:sig",
                        content=signature,
                        chunk_type=ChunkType.FUNCTION_SIGNATURE,
                        repo_id=repo_id,
                        file_path=entity.file_path,
                        entity_name=entity.name,
                        entity_type="function",
                        start_line=entity.start_line,
                        end_line=entity.start_line,
                        language=entity.language,
                    )
                )

        # Function docstring
        if self.config.include_docstrings and entity.docstring:
            chunks.append(
                CodeChunk(
                    id=f"{base_id}:doc",
                    content=entity.docstring,
                    chunk_type=ChunkType.FUNCTION_DOCSTRING,
                    repo_id=repo_id,
                    file_path=entity.file_path,
                    entity_name=entity.name,
                    entity_type="function",
                    start_line=entity.start_line,
                    end_line=entity.start_line,
                    language=entity.language,
                )
            )

        return chunks

    def chunk_class(
        self,
        entity: ClassEntity,
        repo_id: str,
    ) -> list[CodeChunk]:
        """Chunk a class entity.

        Args:
            entity: Class entity to chunk.
            repo_id: Repository identifier.

        Returns:
            List of code chunks for the class.
        """
        chunks: list[CodeChunk] = []
        base_id = f"{repo_id}:{entity.file_path}:{entity.name}:{entity.start_line}"

        # Full class source
        if self.config.include_full_source and entity.source_code:
            chunks.append(
                CodeChunk(
                    id=f"{base_id}:full",
                    content=entity.source_code,
                    chunk_type=ChunkType.CLASS_FULL,
                    repo_id=repo_id,
                    file_path=entity.file_path,
                    entity_name=entity.name,
                    entity_type="class",
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    language=entity.language,
                    metadata={
                        "bases": entity.bases,
                        "decorators": entity.decorators,
                    },
                )
            )

        # Class signature
        if self.config.include_signatures:
            signature = self._build_class_signature(entity)
            if signature:
                chunks.append(
                    CodeChunk(
                        id=f"{base_id}:sig",
                        content=signature,
                        chunk_type=ChunkType.CLASS_SIGNATURE,
                        repo_id=repo_id,
                        file_path=entity.file_path,
                        entity_name=entity.name,
                        entity_type="class",
                        start_line=entity.start_line,
                        end_line=entity.start_line,
                        language=entity.language,
                    )
                )

        # Class docstring
        if self.config.include_docstrings and entity.docstring:
            chunks.append(
                CodeChunk(
                    id=f"{base_id}:doc",
                    content=entity.docstring,
                    chunk_type=ChunkType.CLASS_DOCSTRING,
                    repo_id=repo_id,
                    file_path=entity.file_path,
                    entity_name=entity.name,
                    entity_type="class",
                    start_line=entity.start_line,
                    end_line=entity.start_line,
                    language=entity.language,
                )
            )

        # Chunk methods separately
        for method in entity.methods:
            method_chunks = self.chunk_method(method, entity.name, repo_id)
            chunks.extend(method_chunks)

        return chunks

    def chunk_method(
        self,
        entity: MethodEntity,
        class_name: str,
        repo_id: str,
    ) -> list[CodeChunk]:
        """Chunk a method entity.

        Args:
            entity: Method entity to chunk.
            class_name: Name of the containing class.
            repo_id: Repository identifier.

        Returns:
            List of code chunks for the method.
        """
        chunks: list[CodeChunk] = []
        qualified_name = f"{class_name}.{entity.name}"
        base_id = f"{repo_id}:{entity.file_path}:{qualified_name}:{entity.start_line}"

        # Full method source
        if self.config.include_full_source and entity.source_code:
            chunks.append(
                CodeChunk(
                    id=f"{base_id}:full",
                    content=entity.source_code,
                    chunk_type=ChunkType.METHOD_FULL,
                    repo_id=repo_id,
                    file_path=entity.file_path,
                    entity_name=qualified_name,
                    entity_type="method",
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    language=entity.language,
                    metadata={
                        "class_name": class_name,
                        "is_async": entity.is_async,
                        "parameters": [p.name for p in entity.parameters],
                        "return_type": entity.return_type,
                    },
                )
            )

        # Method signature
        if self.config.include_signatures:
            signature = self._build_function_signature(entity)
            if signature:
                chunks.append(
                    CodeChunk(
                        id=f"{base_id}:sig",
                        content=f"{class_name}.{signature}",
                        chunk_type=ChunkType.METHOD_SIGNATURE,
                        repo_id=repo_id,
                        file_path=entity.file_path,
                        entity_name=qualified_name,
                        entity_type="method",
                        start_line=entity.start_line,
                        end_line=entity.start_line,
                        language=entity.language,
                    )
                )

        return chunks

    def chunk_entity(
        self,
        entity: CodeEntity,
        repo_id: str,
    ) -> list[CodeChunk]:
        """Chunk any code entity.

        Args:
            entity: Code entity to chunk.
            repo_id: Repository identifier.

        Returns:
            List of code chunks.
        """
        if isinstance(entity, FunctionEntity):
            return self.chunk_function(entity, repo_id)
        elif isinstance(entity, ClassEntity):
            return self.chunk_class(entity, repo_id)
        else:
            self._logger.warning(
                "unsupported_entity_type",
                entity_type=type(entity).__name__,
            )
            return []

    def chunk_entities(
        self,
        entities: list[CodeEntity],
        repo_id: str,
    ) -> list[CodeChunk]:
        """Chunk multiple code entities.

        Args:
            entities: List of code entities.
            repo_id: Repository identifier.

        Returns:
            List of all code chunks.
        """
        all_chunks: list[CodeChunk] = []
        for entity in entities:
            chunks = self.chunk_entity(entity, repo_id)
            all_chunks.extend(chunks)

        self._logger.info(
            "chunked_entities",
            entity_count=len(entities),
            chunk_count=len(all_chunks),
        )
        return all_chunks

    def _build_function_signature(self, entity: FunctionEntity | MethodEntity) -> str:
        """Build a function signature string.

        Args:
            entity: Function or method entity.

        Returns:
            Function signature string.
        """
        prefix = "async " if entity.is_async else ""

        # Build parameter strings from Parameter objects
        param_strs = []
        for param in entity.parameters:
            param_str = param.name
            if param.type_annotation:
                param_str = f"{param.name}: {param.type_annotation}"
            if param.default_value:
                param_str = f"{param_str} = {param.default_value}"
            param_strs.append(param_str)

        params = ", ".join(param_strs)
        return_type = f" -> {entity.return_type}" if entity.return_type else ""
        return f"{prefix}def {entity.name}({params}){return_type}"

    def _build_class_signature(self, entity: ClassEntity) -> str:
        """Build a class signature string.

        Args:
            entity: Class entity.

        Returns:
            Class signature string.
        """
        bases = ", ".join(entity.bases) if entity.bases else ""
        inheritance = f"({bases})" if bases else ""
        decorators = "\n".join(f"@{d}" for d in entity.decorators) if entity.decorators else ""
        prefix = f"{decorators}\n" if decorators else ""
        return f"{prefix}class {entity.name}{inheritance}"
