"""Tests for the query module.

This module contains comprehensive tests for query routing,
handlers, and the query engine.
"""

import pytest

from core.llm.context import CodeContext, CodeSnippet
from core.llm.prompts import QueryType
from core.query.engine import MockQueryEngine, QueryEngine, QueryEngineConfig
from core.query.handlers import (
    BaseHandler,
    DebugHandler,
    ExplainHandler,
    FindHandler,
    GeneralHandler,
    GenerateHandler,
    HandlerContext,
    TraceHandler,
    get_handler,
)
from core.query.models import (
    QueryRequest,
    QueryResponse,
    QueryResult,
    QueryStatus,
    SourceReference,
)
from core.query.router import QueryRouter, RouteResult

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def query_router() -> QueryRouter:
    """Create a query router instance."""
    return QueryRouter()


@pytest.fixture
def sample_handler_context() -> HandlerContext:
    """Create a sample handler context."""
    return HandlerContext(
        question="How does the login function work?",
        repo_id="test-repo",
        extracted_entities=["login"],
        file_context="auth/login.py",
    )


@pytest.fixture
def sample_code_context() -> CodeContext:
    """Create a sample code context."""
    snippet = CodeSnippet(
        content="def login(username, password):\n    return True",
        file_path="auth/login.py",
        start_line=10,
        end_line=12,
        entity_name="login",
        entity_type="function",
        language="python",
        relevance_score=0.9,
        token_count=20,
    )
    context = CodeContext(
        snippets=[snippet],
        total_tokens=20,
        max_tokens=8000,
        repo_id="test-repo",
        query="login function",
    )
    return context


@pytest.fixture
def sample_search_results() -> list[dict]:
    """Create sample search results."""
    return [
        {
            "content": "def login(username, password):\n    return True",
            "file_path": "auth/login.py",
            "start_line": 10,
            "end_line": 12,
            "entity_name": "login",
            "entity_type": "function",
            "language": "python",
            "score": 0.9,
        },
        {
            "content": "def logout(user):\n    session.clear()",
            "file_path": "auth/logout.py",
            "start_line": 5,
            "end_line": 7,
            "entity_name": "logout",
            "entity_type": "function",
            "language": "python",
            "score": 0.7,
        },
    ]


# =============================================================================
# Test SourceReference
# =============================================================================


class TestSourceReference:
    """Tests for SourceReference model."""

    def test_source_reference_creation(self):
        """Test creating a source reference."""
        ref = SourceReference(
            file_path="test.py",
            start_line=10,
            end_line=20,
            entity_name="my_func",
            entity_type="function",
            relevance=0.9,
            snippet="def my_func(): pass",
        )
        assert ref.file_path == "test.py"
        assert ref.start_line == 10
        assert ref.end_line == 20
        assert ref.entity_name == "my_func"
        assert ref.relevance == 0.9

    def test_format_location_single_line(self):
        """Test formatting single line location."""
        ref = SourceReference(file_path="test.py", start_line=10, end_line=10)
        assert ref.format_location() == "test.py:10"

    def test_format_location_multi_line(self):
        """Test formatting multi-line location."""
        ref = SourceReference(file_path="test.py", start_line=10, end_line=20)
        assert ref.format_location() == "test.py:10-20"

    def test_default_values(self):
        """Test default values for source reference."""
        ref = SourceReference(file_path="test.py", start_line=1, end_line=1)
        assert ref.entity_name == ""
        assert ref.entity_type == "code"
        assert ref.relevance == 1.0
        assert ref.snippet == ""


# =============================================================================
# Test QueryRequest
# =============================================================================


class TestQueryRequest:
    """Tests for QueryRequest model."""

    def test_request_creation(self):
        """Test creating a query request."""
        request = QueryRequest(
            question="How does this work?",
            repo_id="test-repo",
        )
        assert request.question == "How does this work?"
        assert request.repo_id == "test-repo"
        assert request.include_sources is True
        assert request.stream is False

    def test_request_with_options(self):
        """Test request with custom options."""
        request = QueryRequest(
            question="Explain login",
            repo_id="test-repo",
            file_context="auth/login.py",
            include_sources=False,
            max_tokens=2048,
            stream=True,
        )
        assert request.file_context == "auth/login.py"
        assert request.include_sources is False
        assert request.max_tokens == 2048
        assert request.stream is True

    def test_request_validation(self):
        """Test request validation."""
        # Empty question should fail
        with pytest.raises(ValueError):
            QueryRequest(question="", repo_id="test-repo")

        # Max tokens out of range should fail
        with pytest.raises(ValueError):
            QueryRequest(question="test", repo_id="repo", max_tokens=50)


# =============================================================================
# Test QueryResult
# =============================================================================


class TestQueryResult:
    """Tests for QueryResult model."""

    def test_result_creation(self):
        """Test creating a query result."""
        result = QueryResult(
            answer="The login function authenticates users.",
            sources=[SourceReference(file_path="auth.py", start_line=10, end_line=20)],
            confidence=0.9,
        )
        assert "login" in result.answer
        assert len(result.sources) == 1
        assert result.confidence == 0.9

    def test_result_with_follow_ups(self):
        """Test result with follow-up questions."""
        result = QueryResult(
            answer="Answer",
            follow_up_questions=["Question 1?", "Question 2?"],
        )
        assert len(result.follow_up_questions) == 2


# =============================================================================
# Test QueryResponse
# =============================================================================


class TestQueryResponse:
    """Tests for QueryResponse model."""

    def test_response_creation(self):
        """Test creating a query response."""
        response = QueryResponse(
            request_id="test-123",
            status=QueryStatus.COMPLETED,
            query_type="explain",
            result=QueryResult(answer="Test answer"),
        )
        assert response.request_id == "test-123"
        assert response.status == QueryStatus.COMPLETED
        assert response.is_success is True

    def test_failed_response(self):
        """Test failed query response."""
        response = QueryResponse(
            request_id="test-456",
            status=QueryStatus.FAILED,
            query_type="unknown",
            error="Something went wrong",
        )
        assert response.status == QueryStatus.FAILED
        assert response.is_success is False
        assert response.error == "Something went wrong"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        response = QueryResponse(
            request_id="test-789",
            status=QueryStatus.COMPLETED,
            query_type="explain",
            result=QueryResult(answer="Answer"),
            processing_time_ms=100.0,
            tokens_used=500,
        )
        d = response.to_dict()
        assert d["request_id"] == "test-789"
        assert d["status"] == "completed"
        assert d["result"]["answer"] == "Answer"
        assert d["processing_time_ms"] == 100.0


# =============================================================================
# Test QueryRouter
# =============================================================================


class TestQueryRouter:
    """Tests for QueryRouter."""

    def test_router_initialization(self, query_router):
        """Test router initializes correctly."""
        assert query_router is not None

    def test_route_explain_query(self, query_router):
        """Test routing an explain query."""
        result = query_router.route("Explain how the login function works")
        assert result.query_type == QueryType.EXPLAIN
        assert result.confidence >= 0.5

    def test_route_trace_query(self, query_router):
        """Test routing a trace query."""
        result = query_router.route("Trace the execution flow of process_request")
        assert result.query_type == QueryType.TRACE

    def test_route_find_query(self, query_router):
        """Test routing a find query."""
        result = query_router.route("Find all functions that use the database")
        assert result.query_type == QueryType.FIND

    def test_route_generate_query(self, query_router):
        """Test routing a generate query."""
        result = query_router.route("Generate a function to validate emails")
        assert result.query_type == QueryType.GENERATE

    def test_route_debug_query(self, query_router):
        """Test routing a debug query."""
        result = query_router.route("Debug the error in the authentication module")
        assert result.query_type == QueryType.DEBUG

    def test_route_review_query(self, query_router):
        """Test routing a review query."""
        result = query_router.route("Review this code for issues")
        assert result.query_type == QueryType.REVIEW

    def test_route_refactor_query(self, query_router):
        """Test routing a refactor query."""
        result = query_router.route("Refactor this function to be cleaner")
        assert result.query_type == QueryType.REFACTOR

    def test_route_document_query(self, query_router):
        """Test routing a documentation query."""
        result = query_router.route("Document this class with proper docstrings")
        assert result.query_type == QueryType.DOCUMENT

    def test_route_test_query(self, query_router):
        """Test routing a test query."""
        result = query_router.route("Need pytest tests for the calculator class")
        assert result.query_type == QueryType.TEST

    def test_route_general_query(self, query_router):
        """Test routing a general query."""
        result = query_router.route("Hello, can you help me today?")
        assert result.query_type == QueryType.GENERAL

    def test_entity_extraction(self, query_router):
        """Test entity extraction from queries."""
        result = query_router.route("Explain how `process_data` handles the input")
        assert "process_data" in result.extracted_entities

    def test_entity_extraction_function_call(self, query_router):
        """Test extraction of function call syntax."""
        result = query_router.route("What does calculate_total() return?")
        assert "calculate_total" in result.extracted_entities

    def test_context_suggestions(self, query_router):
        """Test context suggestions are included."""
        result = query_router.route("Explain the login function")
        assert len(result.suggested_context) > 0

    def test_get_handler_name(self, query_router):
        """Test getting handler name for query type."""
        assert query_router.get_handler_name(QueryType.EXPLAIN) == "ExplainHandler"
        assert query_router.get_handler_name(QueryType.TRACE) == "TraceHandler"
        assert query_router.get_handler_name(QueryType.GENERAL) == "GeneralHandler"


# =============================================================================
# Test Handlers
# =============================================================================


class TestHandlerContext:
    """Tests for HandlerContext model."""

    def test_context_creation(self):
        """Test creating handler context."""
        ctx = HandlerContext(
            question="Test question",
            repo_id="test-repo",
        )
        assert ctx.question == "Test question"
        assert ctx.repo_id == "test-repo"
        assert ctx.code_context is None

    def test_context_with_code_context(self, sample_code_context):
        """Test context with code context."""
        ctx = HandlerContext(
            question="Test",
            repo_id="repo",
            code_context=sample_code_context,
        )
        assert ctx.code_context is not None
        assert len(ctx.code_context.snippets) == 1


class TestBaseHandler:
    """Tests for BaseHandler functionality."""

    def test_get_handler_returns_correct_type(self):
        """Test get_handler returns correct handler types."""
        assert isinstance(get_handler(QueryType.EXPLAIN), ExplainHandler)
        assert isinstance(get_handler(QueryType.TRACE), TraceHandler)
        assert isinstance(get_handler(QueryType.FIND), FindHandler)
        assert isinstance(get_handler(QueryType.GENERATE), GenerateHandler)
        assert isinstance(get_handler(QueryType.DEBUG), DebugHandler)
        assert isinstance(get_handler(QueryType.GENERAL), GeneralHandler)


class TestExplainHandler:
    """Tests for ExplainHandler."""

    def test_query_type(self):
        """Test handler query type."""
        handler = ExplainHandler()
        assert handler.query_type == QueryType.EXPLAIN

    def test_build_context_query(self, sample_handler_context):
        """Test building context query."""
        handler = ExplainHandler()
        query = handler.build_context_query(sample_handler_context)
        assert "login" in query

    def test_enhance_prompt(self, sample_handler_context):
        """Test prompt enhancement."""
        handler = ExplainHandler()
        enhanced = handler.enhance_prompt(sample_handler_context)
        assert "login" in enhanced.lower()

    def test_process_response(self, sample_handler_context, sample_code_context):
        """Test response processing."""
        handler = ExplainHandler()
        sample_handler_context = HandlerContext(
            question=sample_handler_context.question,
            repo_id=sample_handler_context.repo_id,
            code_context=sample_code_context,
            extracted_entities=sample_handler_context.extracted_entities,
        )
        result = handler.process_response("Test answer", sample_handler_context)
        assert result.answer == "Test answer"
        assert len(result.sources) > 0

    def test_follow_up_suggestions(self, sample_handler_context):
        """Test follow-up question suggestions."""
        handler = ExplainHandler()
        suggestions = handler._suggest_follow_ups(sample_handler_context)
        assert len(suggestions) > 0


class TestTraceHandler:
    """Tests for TraceHandler."""

    def test_query_type(self):
        """Test handler query type."""
        handler = TraceHandler()
        assert handler.query_type == QueryType.TRACE

    def test_enhance_prompt(self, sample_handler_context):
        """Test trace prompt enhancement."""
        handler = TraceHandler()
        enhanced = handler.enhance_prompt(sample_handler_context)
        assert "step-by-step" in enhanced.lower()
        assert "entry point" in enhanced.lower()


class TestFindHandler:
    """Tests for FindHandler."""

    def test_query_type(self):
        """Test handler query type."""
        handler = FindHandler()
        assert handler.query_type == QueryType.FIND

    def test_enhance_prompt(self, sample_handler_context):
        """Test find prompt enhancement."""
        handler = FindHandler()
        enhanced = handler.enhance_prompt(sample_handler_context)
        assert "file path" in enhanced.lower()


class TestGenerateHandler:
    """Tests for GenerateHandler."""

    def test_query_type(self):
        """Test handler query type."""
        handler = GenerateHandler()
        assert handler.query_type == QueryType.GENERATE

    def test_enhance_prompt_with_file_context(self):
        """Test generate prompt includes file context."""
        handler = GenerateHandler()
        ctx = HandlerContext(
            question="Generate a function",
            repo_id="repo",
            file_context="utils/helpers.py",
        )
        enhanced = handler.enhance_prompt(ctx)
        assert "utils/helpers.py" in enhanced


class TestDebugHandler:
    """Tests for DebugHandler."""

    def test_query_type(self):
        """Test handler query type."""
        handler = DebugHandler()
        assert handler.query_type == QueryType.DEBUG

    def test_enhance_prompt(self, sample_handler_context):
        """Test debug prompt enhancement."""
        handler = DebugHandler()
        enhanced = handler.enhance_prompt(sample_handler_context)
        assert "root cause" in enhanced.lower()


class TestGeneralHandler:
    """Tests for GeneralHandler."""

    def test_query_type(self):
        """Test handler query type."""
        handler = GeneralHandler()
        assert handler.query_type == QueryType.GENERAL

    def test_enhance_prompt_no_modification(self, sample_handler_context):
        """Test general handler doesn't modify prompt."""
        handler = GeneralHandler()
        enhanced = handler.enhance_prompt(sample_handler_context)
        assert enhanced == sample_handler_context.question


# =============================================================================
# Test QueryEngineConfig
# =============================================================================


class TestQueryEngineConfig:
    """Tests for QueryEngineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QueryEngineConfig()
        assert config.max_context_tokens == 8000
        assert config.max_response_tokens == 4096
        assert config.include_sources is True
        assert config.min_relevance == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = QueryEngineConfig(
            max_context_tokens=4000,
            max_response_tokens=2048,
            include_sources=False,
            min_relevance=0.7,
        )
        assert config.max_context_tokens == 4000
        assert config.include_sources is False


# =============================================================================
# Test QueryEngine
# =============================================================================


class TestQueryEngine:
    """Tests for QueryEngine."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = QueryEngine()
        assert engine is not None
        assert engine.config is not None

    def test_engine_with_custom_config(self):
        """Test engine with custom configuration."""
        config = QueryEngineConfig(max_context_tokens=4000)
        engine = QueryEngine(config=config)
        assert engine.config.max_context_tokens == 4000

    def test_route_method(self):
        """Test route method exposes routing."""
        engine = QueryEngine()
        result = engine.route("Explain the login function")
        assert result.query_type == QueryType.EXPLAIN

    def test_get_handler_method(self):
        """Test get_handler method."""
        engine = QueryEngine()
        handler = engine.get_handler(QueryType.EXPLAIN)
        assert isinstance(handler, ExplainHandler)

    @pytest.mark.asyncio
    async def test_build_context(self, sample_search_results):
        """Test context building from search results."""
        engine = QueryEngine()
        context = await engine.build_context(
            question="Login function",
            repo_id="test-repo",
            search_results=sample_search_results,
        )
        assert context is not None
        assert len(context.snippets) > 0

    @pytest.mark.asyncio
    async def test_query_without_llm_client(self):
        """Test query returns placeholder without LLM client."""
        engine = QueryEngine()
        request = QueryRequest(
            question="How does login work?",
            repo_id="test-repo",
        )
        response = await engine.query(request)
        assert response.status == QueryStatus.COMPLETED
        assert "placeholder" in response.result.answer.lower()

    @pytest.mark.asyncio
    async def test_query_with_search_results(self, sample_search_results):
        """Test query with search results builds context."""
        engine = QueryEngine()
        request = QueryRequest(
            question="Explain the login function",
            repo_id="test-repo",
        )
        response = await engine.query(request, search_results=sample_search_results)
        assert response.status == QueryStatus.COMPLETED


class TestMockQueryEngine:
    """Tests for MockQueryEngine."""

    @pytest.mark.asyncio
    async def test_mock_engine_returns_responses(self):
        """Test mock engine returns predefined responses."""
        engine = MockQueryEngine(responses=["Test answer"])
        request = QueryRequest(
            question="Test question",
            repo_id="test-repo",
        )
        response = await engine.query(request)
        assert response.status == QueryStatus.COMPLETED
        assert response.result.answer == "Test answer"

    @pytest.mark.asyncio
    async def test_mock_engine_cycles_responses(self):
        """Test mock engine cycles through responses."""
        engine = MockQueryEngine(responses=["Answer 1", "Answer 2"])
        request = QueryRequest(question="Test", repo_id="repo")

        response1 = await engine.query(request)
        response2 = await engine.query(request)

        assert response1.result.answer == "Answer 1"
        assert response2.result.answer == "Answer 2"


# =============================================================================
# Test Integration
# =============================================================================


class TestQueryIntegration:
    """Integration tests for the query module."""

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self, sample_search_results):
        """Test the full query pipeline."""
        engine = MockQueryEngine(
            responses=["The login function authenticates users by checking credentials."]
        )

        request = QueryRequest(
            question="Explain how the login function works",
            repo_id="test-repo",
        )

        response = await engine.query(request, search_results=sample_search_results)

        assert response.status == QueryStatus.COMPLETED
        assert response.query_type == "explain"
        assert "login" in response.result.answer.lower()
        assert response.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_query_with_file_context(self):
        """Test query with file context."""
        engine = MockQueryEngine(responses=["File-specific answer"])

        request = QueryRequest(
            question="What does this file do?",
            repo_id="test-repo",
            file_context="auth/login.py",
        )

        response = await engine.query(request)
        assert response.status == QueryStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_different_query_types(self):
        """Test different query types route correctly."""
        engine = MockQueryEngine()

        test_cases = [
            ("Explain the login function", "explain"),
            ("Trace the data flow", "trace"),
            ("Find all database queries", "find"),
            ("Generate a helper function", "generate"),
            ("Debug this error", "debug"),
        ]

        for question, expected_type in test_cases:
            request = QueryRequest(question=question, repo_id="repo")
            response = await engine.query(request)
            assert response.query_type == expected_type, f"Failed for: {question}"


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_query_handles_errors_gracefully(self):
        """Test that query handles errors gracefully."""
        engine = QueryEngine()

        # Create an invalid request that might cause issues
        request = QueryRequest(
            question="Test question",
            repo_id="",  # Empty repo ID might cause issues
        )

        # Should not raise, should return error response
        response = await engine.query(request)
        # Either succeeds with placeholder or fails gracefully
        assert response.status in [QueryStatus.COMPLETED, QueryStatus.FAILED]

    def test_router_handles_empty_query(self, query_router):
        """Test router handles edge cases."""
        # Very short query
        result = query_router.route("hi")
        assert result.query_type == QueryType.GENERAL

    def test_router_handles_special_characters(self, query_router):
        """Test router handles special characters."""
        result = query_router.route("What does @decorator do in Python?")
        assert result is not None
