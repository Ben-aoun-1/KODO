"""Natural language query endpoints for the Kodo API.

This module provides endpoints for asking natural language questions
about code, performing semantic search, and getting AI-powered responses.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import QueryEngineDep, VectorStoreDep
from core.query.models import QueryRequest, QueryResponse, QueryStatus

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/repos/{repo_id}/ask", tags=["Natural Language Query"])


class AskRequest(BaseModel):
    """Request model for asking questions.

    Attributes:
        question: The natural language question.
        file_context: Optional file path for focused context.
        include_sources: Whether to include source references.
        max_tokens: Maximum response tokens.
        stream: Whether to stream the response.
    """

    question: str = Field(..., min_length=1, max_length=2000, description="The question")
    file_context: str | None = Field(None, description="Optional file context")
    include_sources: bool = Field(default=True, description="Include sources")
    max_tokens: int = Field(default=4096, ge=100, le=8192, description="Max tokens")
    stream: bool = Field(default=False, description="Stream response")


class SourceInfo(BaseModel):
    """Source code reference in response.

    Attributes:
        file_path: Path to the source file.
        start_line: Starting line number.
        end_line: Ending line number.
        entity_name: Name of the code entity.
        entity_type: Type of entity.
        relevance: Relevance score.
    """

    file_path: str = Field(..., description="File path")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    entity_name: str = Field(default="", description="Entity name")
    entity_type: str = Field(default="code", description="Entity type")
    relevance: float = Field(default=1.0, description="Relevance score")


class AskResponse(BaseModel):
    """Response model for ask endpoint.

    Attributes:
        request_id: Unique request identifier.
        query_type: Classified query type.
        answer: The answer text.
        sources: Referenced source locations.
        confidence: Confidence score.
        follow_up_questions: Suggested follow-ups.
        processing_time_ms: Processing time.
        tokens_used: Total tokens used.
    """

    request_id: str = Field(..., description="Request ID")
    query_type: str = Field(..., description="Query type")
    answer: str = Field(..., description="The answer")
    sources: list[SourceInfo] = Field(default_factory=list, description="Sources")
    confidence: float = Field(default=1.0, description="Confidence")
    follow_up_questions: list[str] = Field(default_factory=list, description="Follow-ups")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    tokens_used: int = Field(default=0, description="Tokens used")


class SearchRequest(BaseModel):
    """Request model for semantic search.

    Attributes:
        query: The search query.
        limit: Maximum number of results.
        entity_types: Filter by entity types.
        min_score: Minimum relevance score.
    """

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Max results")
    entity_types: list[str] | None = Field(None, description="Filter by types")
    min_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Min score")


class SearchResultItem(BaseModel):
    """A single search result.

    Attributes:
        file_path: Path to the source file.
        entity_name: Name of the code entity.
        entity_type: Type of entity.
        start_line: Starting line number.
        end_line: Ending line number.
        content: Code content preview.
        score: Relevance score.
    """

    file_path: str = Field(..., description="File path")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    content: str = Field(..., description="Content preview")
    score: float = Field(..., description="Relevance score")


class SearchResponse(BaseModel):
    """Response model for semantic search.

    Attributes:
        query: Original query.
        results: Search results.
        total: Total matches found.
    """

    query: str = Field(..., description="Original query")
    results: list[SearchResultItem] = Field(..., description="Results")
    total: int = Field(..., description="Total results")


def _query_response_to_ask_response(response: QueryResponse) -> AskResponse:
    """Convert internal QueryResponse to API AskResponse."""
    sources = []
    if response.result and response.result.sources:
        for src in response.result.sources:
            sources.append(
                SourceInfo(
                    file_path=src.file_path,
                    start_line=src.start_line,
                    end_line=src.end_line,
                    entity_name=src.entity_name,
                    entity_type=src.entity_type,
                    relevance=src.relevance,
                )
            )

    return AskResponse(
        request_id=response.request_id,
        query_type=response.query_type,
        answer=response.result.answer if response.result else "",
        sources=sources,
        confidence=response.result.confidence if response.result else 0.0,
        follow_up_questions=response.result.follow_up_questions if response.result else [],
        processing_time_ms=response.processing_time_ms,
        tokens_used=response.tokens_used,
    )


@router.post(
    "",
    response_model=AskResponse,
    summary="Ask a question",
    description="Ask a natural language question about the codebase.",
    responses={
        200: {"description": "Successful response"},
        404: {"description": "Repository not found"},
        500: {"description": "Query processing failed"},
    },
)
async def ask_question(
    repo_id: str,
    request: AskRequest,
    query_engine: QueryEngineDep,
    vector_store: VectorStoreDep,
) -> AskResponse:
    """Ask a natural language question about the codebase.

    This endpoint processes natural language queries about code,
    automatically classifying the query type and building relevant context.

    Args:
        repo_id: Repository identifier.
        request: The ask request with question and options.
        query_engine: Query engine for processing.
        vector_store: Vector store for semantic search.

    Returns:
        AskResponse with answer and source references.

    Raises:
        HTTPException: If query processing fails.
    """
    logger.info(
        "ask_question_received",
        repo_id=repo_id,
        question_preview=request.question[:50],
    )

    # Build search results from vector store for context
    search_results: list[dict[str, Any]] = []
    try:
        # Search for relevant code
        results = await vector_store.search(
            query_vector=[],  # Will use embedding client in production
            limit=10,
            repo_id=repo_id,
        )
        search_results = [
            {
                "content": r.content,
                "file_path": r.payload.get("file_path", ""),
                "start_line": r.payload.get("start_line", 0),
                "end_line": r.payload.get("end_line", 0),
                "entity_name": r.payload.get("entity_name", ""),
                "entity_type": r.payload.get("entity_type", "code"),
                "language": r.payload.get("language", "python"),
                "score": r.score,
            }
            for r in results
        ]
    except Exception as e:
        logger.warning("vector_search_failed", error=str(e))
        # Continue without vector context

    # Create query request
    query_request = QueryRequest(
        question=request.question,
        repo_id=repo_id,
        file_context=request.file_context,
        include_sources=request.include_sources,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )

    # Execute query
    response = await query_engine.query(query_request, search_results=search_results)

    if response.status == QueryStatus.FAILED:
        logger.error("query_failed", error=response.error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {response.error}",
        )

    return _query_response_to_ask_response(response)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search",
    description="Search code using natural language.",
)
async def semantic_search(
    repo_id: str,
    request: SearchRequest,
    vector_store: VectorStoreDep,
) -> SearchResponse:
    """Search code using natural language.

    Performs semantic search over the codebase using embeddings.

    Args:
        repo_id: Repository identifier.
        request: The search request.
        vector_store: Vector store for search.

    Returns:
        SearchResponse with matching results.
    """
    logger.info(
        "semantic_search",
        repo_id=repo_id,
        query_preview=request.query[:50],
    )

    try:
        # In production, would generate embedding from query
        # For now, return empty results until embedding client is wired up
        results = await vector_store.search(
            query_vector=[],  # Will use embedding client
            limit=request.limit,
            repo_id=repo_id,
            min_score=request.min_score,
        )

        # Apply entity type filter if provided
        if request.entity_types:
            results = [r for r in results if r.payload.get("entity_type") in request.entity_types]

        # Convert to response format
        items = []
        for r in results:
            items.append(
                SearchResultItem(
                    file_path=r.payload.get("file_path", ""),
                    entity_name=r.payload.get("entity_name", ""),
                    entity_type=r.payload.get("entity_type", "code"),
                    start_line=r.payload.get("start_line", 0),
                    end_line=r.payload.get("end_line", 0),
                    content=r.content[:500] if r.content else "",
                    score=r.score,
                )
            )

        return SearchResponse(
            query=request.query,
            results=items,
            total=len(items),
        )

    except Exception as e:
        logger.error("semantic_search_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/classify",
    summary="Classify query",
    description="Classify a question into a query type without executing it.",
)
async def classify_query(
    repo_id: str,
    question: str = Query(..., min_length=1, description="The question to classify"),
    query_engine: QueryEngineDep = None,
) -> dict[str, Any]:
    """Classify a question without executing the query.

    Useful for understanding how the system interprets questions.

    Args:
        repo_id: Repository identifier.
        question: The question to classify.
        query_engine: Query engine for classification.

    Returns:
        Classification result with type and confidence.
    """
    result = query_engine.route(question)

    return {
        "question": question,
        "query_type": result.query_type.value,
        "confidence": result.confidence,
        "extracted_entities": result.extracted_entities,
        "suggested_context": result.suggested_context,
    }
