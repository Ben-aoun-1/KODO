"""Data models for the query engine.

This module defines the request and response models for code queries.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryStatus(str, Enum):
    """Status of a query execution."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceReference(BaseModel):
    """A reference to source code location.

    Attributes:
        file_path: Path to the source file.
        start_line: Starting line number.
        end_line: Ending line number.
        entity_name: Name of the referenced entity.
        entity_type: Type of entity (function, class, etc.).
        relevance: Relevance score (0-1).
        snippet: Code snippet preview.
    """

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(..., description="File path")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    entity_name: str = Field(default="", description="Entity name")
    entity_type: str = Field(default="code", description="Entity type")
    relevance: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance score")
    snippet: str = Field(default="", description="Code snippet preview")

    def format_location(self) -> str:
        """Format the location as a string.

        Returns:
            Formatted location string.
        """
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class QueryRequest(BaseModel):
    """A query request to the engine.

    Attributes:
        question: The natural language question.
        repo_id: Repository identifier.
        file_context: Optional file path for context.
        include_sources: Whether to include source references.
        max_tokens: Maximum tokens for response.
        stream: Whether to stream the response.
    """

    model_config = ConfigDict(frozen=True)

    question: str = Field(..., min_length=1, description="The question")
    repo_id: str = Field(..., description="Repository ID")
    file_context: str | None = Field(None, description="Optional file context")
    include_sources: bool = Field(default=True, description="Include source references")
    max_tokens: int = Field(default=4096, ge=100, le=8192, description="Max response tokens")
    stream: bool = Field(default=False, description="Stream response")


class QueryResult(BaseModel):
    """The result content of a query.

    Attributes:
        answer: The answer text.
        sources: Referenced source code locations.
        confidence: Confidence score (0-1).
        follow_up_questions: Suggested follow-up questions.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(..., description="The answer")
    sources: list[SourceReference] = Field(default_factory=list, description="Source references")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )


class QueryResponse(BaseModel):
    """Full response from the query engine.

    Attributes:
        request_id: Unique request identifier.
        status: Query status.
        query_type: Classified query type.
        result: Query result (if completed).
        error: Error message (if failed).
        processing_time_ms: Processing time in milliseconds.
        tokens_used: Total tokens used.
        created_at: Timestamp of creation.
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., description="Request ID")
    status: QueryStatus = Field(..., description="Query status")
    query_type: str = Field(..., description="Classified query type")
    result: QueryResult | None = Field(None, description="Query result")
    error: str | None = Field(None, description="Error message")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")
    tokens_used: int = Field(default=0, description="Tokens used")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Created timestamp")

    @property
    def is_success(self) -> bool:
        """Check if the query was successful."""
        return self.status == QueryStatus.COMPLETED and self.result is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "query_type": self.query_type,
            "result": (
                {
                    "answer": self.result.answer,
                    "sources": [
                        {
                            "file_path": s.file_path,
                            "start_line": s.start_line,
                            "end_line": s.end_line,
                            "entity_name": s.entity_name,
                            "entity_type": s.entity_type,
                            "relevance": s.relevance,
                        }
                        for s in self.result.sources
                    ],
                    "confidence": self.result.confidence,
                    "follow_up_questions": self.result.follow_up_questions,
                }
                if self.result
                else None
            ),
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat(),
        }
