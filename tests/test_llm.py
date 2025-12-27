"""Tests for the LLM module.

This module tests prompt templates, context building, and Claude client.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.llm.client import (
    ClaudeClient,
    LLMClientError,
    LLMConfig,
    LLMResponse,
    MockClaudeClient,
)
from core.llm.context import (
    CodeContext,
    CodeSnippet,
    ContextBuilder,
    ContextConfig,
    TokenCounter,
)
from core.llm.prompts import (
    EXPLAIN_TEMPLATE,
    GENERAL_TEMPLATE,
    PromptTemplate,
    QueryType,
    classify_query,
    get_prompt_template,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_snippet() -> CodeSnippet:
    """Create a sample code snippet."""
    return CodeSnippet(
        content="""def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two integers.'''
    return a + b""",
        file_path="math_utils.py",
        start_line=10,
        end_line=13,
        entity_name="calculate_sum",
        entity_type="function",
        language="python",
        relevance_score=0.95,
        token_count=50,
    )


@pytest.fixture
def sample_snippets() -> list[CodeSnippet]:
    """Create multiple sample snippets."""
    return [
        CodeSnippet(
            content="def func1(): pass",
            file_path="a.py",
            start_line=1,
            end_line=1,
            entity_name="func1",
            entity_type="function",
            language="python",
            relevance_score=0.9,
            token_count=20,
        ),
        CodeSnippet(
            content="def func2(): pass",
            file_path="b.py",
            start_line=1,
            end_line=1,
            entity_name="func2",
            entity_type="function",
            language="python",
            relevance_score=0.8,
            token_count=20,
        ),
        CodeSnippet(
            content="def func3(): pass",
            file_path="c.py",
            start_line=1,
            end_line=1,
            entity_name="func3",
            entity_type="function",
            language="python",
            relevance_score=0.7,
            token_count=20,
        ),
        CodeSnippet(
            content="def low_relevance(): pass",
            file_path="d.py",
            start_line=1,
            end_line=1,
            entity_name="low_relevance",
            entity_type="function",
            language="python",
            relevance_score=0.3,
            token_count=20,
        ),
    ]


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestQueryType:
    """Tests for QueryType enum."""

    def test_query_type_values(self):
        """Test all query type values."""
        assert QueryType.EXPLAIN == "explain"
        assert QueryType.TRACE == "trace"
        assert QueryType.FIND == "find"
        assert QueryType.GENERATE == "generate"
        assert QueryType.REVIEW == "review"
        assert QueryType.REFACTOR == "refactor"
        assert QueryType.DEBUG == "debug"
        assert QueryType.DOCUMENT == "document"
        assert QueryType.TEST == "test"
        assert QueryType.GENERAL == "general"

    def test_all_query_types_have_templates(self):
        """Test that all query types have templates."""
        for query_type in QueryType:
            template = get_prompt_template(query_type)
            assert template is not None
            assert template.query_type == query_type


class TestPromptTemplate:
    """Tests for PromptTemplate model."""

    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            query_type=QueryType.EXPLAIN,
            system_prompt="You are a helpful assistant.",
            user_template="Question: {question}",
        )
        assert template.query_type == QueryType.EXPLAIN
        assert template.requires_context is True
        assert template.max_context_tokens == 8000

    def test_format_user_message(self):
        """Test formatting user message."""
        template = PromptTemplate(
            query_type=QueryType.GENERAL,
            system_prompt="System prompt",
            user_template="Context: {context}\n\nQuestion: {question}",
        )
        message = template.format_user_message(
            question="What does this do?",
            context="def foo(): pass",
        )
        assert "What does this do?" in message
        assert "def foo(): pass" in message

    def test_format_user_message_with_extra_kwargs(self):
        """Test formatting with additional kwargs."""
        template = PromptTemplate(
            query_type=QueryType.REVIEW,
            system_prompt="System prompt",
            user_template="Code ({language}):\n{code}\n\n{question}",
        )
        message = template.format_user_message(
            question="Review this",
            context="",
            language="python",
            code="def foo(): pass",
        )
        assert "python" in message
        assert "def foo(): pass" in message


class TestGetPromptTemplate:
    """Tests for get_prompt_template function."""

    def test_get_explain_template(self):
        """Test getting explain template."""
        template = get_prompt_template(QueryType.EXPLAIN)
        assert template.query_type == QueryType.EXPLAIN
        assert "explanation" in template.system_prompt.lower()

    def test_get_trace_template(self):
        """Test getting trace template."""
        template = get_prompt_template(QueryType.TRACE)
        assert template.query_type == QueryType.TRACE
        assert template.max_context_tokens == 10000

    def test_get_generate_template(self):
        """Test getting generate template."""
        template = get_prompt_template(QueryType.GENERATE)
        assert template.query_type == QueryType.GENERATE
        assert "generation" in template.system_prompt.lower()


class TestClassifyQuery:
    """Tests for query classification."""

    def test_classify_explain(self):
        """Test classifying explain queries."""
        assert classify_query("explain how this function works") == QueryType.EXPLAIN
        assert classify_query("what does this class do") == QueryType.EXPLAIN
        assert classify_query("help me understand the code") == QueryType.EXPLAIN

    def test_classify_trace(self):
        """Test classifying trace queries."""
        assert classify_query("trace the data flow") == QueryType.TRACE
        assert classify_query("what is the call chain for this") == QueryType.TRACE
        assert classify_query("follow the execution path") == QueryType.TRACE

    def test_classify_find(self):
        """Test classifying find queries."""
        assert classify_query("find all usages of this function") == QueryType.FIND
        assert classify_query("where is this defined") == QueryType.FIND
        assert classify_query("search for error handlers") == QueryType.FIND

    def test_classify_generate(self):
        """Test classifying generate queries."""
        assert classify_query("generate a function to parse JSON") == QueryType.GENERATE
        assert classify_query("write a test for this") == QueryType.GENERATE
        assert classify_query("create a new class") == QueryType.GENERATE

    def test_classify_debug(self):
        """Test classifying debug queries."""
        assert classify_query("fix this bug") == QueryType.DEBUG
        assert classify_query("why is this error happening") == QueryType.DEBUG
        assert classify_query("help me debug this issue") == QueryType.DEBUG

    def test_classify_general(self):
        """Test classifying general queries."""
        assert classify_query("hello") == QueryType.GENERAL
        assert classify_query("what is Python") == QueryType.GENERAL


# =============================================================================
# Context Tests
# =============================================================================


class TestCodeSnippet:
    """Tests for CodeSnippet model."""

    def test_snippet_creation(self, sample_snippet):
        """Test creating a code snippet."""
        assert sample_snippet.entity_name == "calculate_sum"
        assert sample_snippet.relevance_score == 0.95
        assert sample_snippet.token_count == 50

    def test_snippet_format(self, sample_snippet):
        """Test formatting a snippet."""
        formatted = sample_snippet.format()
        assert "calculate_sum" in formatted
        assert "math_utils.py:10-13" in formatted
        assert "python" in formatted
        assert "def calculate_sum" in formatted

    def test_snippet_default_values(self):
        """Test snippet default values."""
        snippet = CodeSnippet(
            content="code",
            file_path="test.py",
            start_line=1,
            end_line=1,
        )
        assert snippet.entity_name == ""
        assert snippet.entity_type == "code"
        assert snippet.language == "python"
        assert snippet.relevance_score == 1.0
        assert snippet.token_count == 0


class TestCodeContext:
    """Tests for CodeContext model."""

    def test_empty_context(self):
        """Test empty context."""
        context = CodeContext()
        assert context.is_empty
        assert context.total_tokens == 0
        assert "No relevant code" in context.format()

    def test_add_snippet(self, sample_snippet):
        """Test adding snippets to context."""
        context = CodeContext(max_tokens=100)
        assert context.add_snippet(sample_snippet) is True
        assert len(context.snippets) == 1
        assert context.total_tokens == 50

    def test_add_snippet_exceeds_limit(self, sample_snippet):
        """Test adding snippet that exceeds limit."""
        context = CodeContext(max_tokens=30)
        assert context.add_snippet(sample_snippet) is False
        assert context.is_empty

    def test_format_with_snippets(self, sample_snippet):
        """Test formatting context with snippets."""
        context = CodeContext(max_tokens=1000)
        context.add_snippet(sample_snippet)
        formatted = context.format()
        assert "calculate_sum" in formatted
        assert "def calculate_sum" in formatted


class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_tokens(self):
        """Test counting tokens."""
        counter = TokenCounter()
        count = counter.count("Hello world")
        assert count >= 2

    def test_count_empty_string(self):
        """Test counting empty string."""
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_count_messages(self):
        """Test counting messages."""
        counter = TokenCounter()
        count = counter.count_messages(
            system="You are helpful.",
            user="Hello!",
        )
        assert count > 0

    def test_truncate(self):
        """Test truncating text."""
        counter = TokenCounter()
        long_text = "word " * 1000
        truncated = counter.truncate(long_text, 50)
        # Should be shorter than original
        assert len(truncated) < len(long_text)


class TestContextConfig:
    """Tests for ContextConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ContextConfig()
        assert config.max_tokens == 8000
        assert config.max_snippets == 20
        assert config.min_relevance == 0.5
        assert config.include_imports is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextConfig(
            max_tokens=4000,
            max_snippets=10,
            min_relevance=0.7,
        )
        assert config.max_tokens == 4000
        assert config.min_relevance == 0.7


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = ContextBuilder()
        assert builder.config is not None

    def test_builder_with_custom_config(self):
        """Test builder with custom config."""
        config = ContextConfig(max_tokens=4000)
        builder = ContextBuilder(config)
        assert builder.config.max_tokens == 4000

    def test_build_context(self, sample_snippets):
        """Test building context from snippets."""
        builder = ContextBuilder()
        context = builder.build(
            snippets=sample_snippets,
            query="What do these functions do?",
            repo_id="test-repo",
        )
        assert not context.is_empty
        assert context.repo_id == "test-repo"

    def test_build_context_respects_relevance(self, sample_snippets):
        """Test that low relevance snippets are filtered."""
        config = ContextConfig(min_relevance=0.5)
        builder = ContextBuilder(config)
        context = builder.build(
            snippets=sample_snippets,
            query="test",
        )
        # Low relevance snippet (0.3) should be filtered
        for snippet in context.snippets:
            assert snippet.relevance_score >= 0.5

    def test_build_context_respects_token_limit(self, sample_snippets):
        """Test that context respects token limit."""
        # Use max_tokens override in build() since ContextConfig has min of 1000
        builder = ContextBuilder()
        context = builder.build(
            snippets=sample_snippets,
            query="test",
            max_tokens=50,  # Override to test small limit
        )
        assert context.total_tokens <= 50

    def test_build_context_sorts_by_relevance(self, sample_snippets):
        """Test that snippets are sorted by relevance."""
        builder = ContextBuilder()
        context = builder.build(
            snippets=sample_snippets,
            query="test",
        )
        # First snippet should have highest relevance
        if len(context.snippets) >= 2:
            assert context.snippets[0].relevance_score >= context.snippets[1].relevance_score

    def test_build_from_search_results(self):
        """Test building context from search results."""
        builder = ContextBuilder()
        results = [
            {
                "content": "def foo(): pass",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 1,
                "entity_name": "foo",
                "entity_type": "function",
                "language": "python",
                "score": 0.9,
            }
        ]
        context = builder.build_from_search_results(results, "test query")
        assert not context.is_empty

    def test_create_snippet(self):
        """Test creating snippet with token count."""
        builder = ContextBuilder()
        snippet = builder.create_snippet(
            content="def foo(): pass",
            file_path="test.py",
            start_line=1,
            end_line=1,
            entity_name="foo",
        )
        assert snippet.token_count > 0

    def test_estimate_tokens(self):
        """Test estimating tokens."""
        builder = ContextBuilder()
        tokens = builder.estimate_tokens("Hello world")
        assert tokens >= 2

    def test_truncate_to_fit(self):
        """Test truncating text."""
        builder = ContextBuilder()
        long_text = "word " * 100
        truncated = builder.truncate_to_fit(long_text, 20)
        tokens = builder.estimate_tokens(truncated)
        assert tokens <= 20


# =============================================================================
# LLM Client Tests
# =============================================================================


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = LLMConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096
        assert config.temperature == 0.0
        assert config.timeout == 120.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            model="claude-opus-4-20250514",
            max_tokens=8192,
            temperature=0.5,
        )
        assert config.model == "claude-opus-4-20250514"
        assert config.max_tokens == 8192


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_response_creation(self):
        """Test creating a response."""
        response = LLMResponse(
            content="This is the answer.",
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
        )
        assert response.content == "This is the answer."
        assert response.input_tokens == 100

    def test_response_with_query_type(self):
        """Test response with query type."""
        response = LLMResponse(
            content="Answer",
            model="claude",
            query_type=QueryType.EXPLAIN,
        )
        assert response.query_type == QueryType.EXPLAIN


class TestMockClaudeClient:
    """Tests for MockClaudeClient."""

    @pytest.mark.asyncio
    async def test_mock_complete(self):
        """Test mock completion."""
        client = MockClaudeClient(responses=["Test response"])
        response = await client.complete("system", "user")
        assert response.content == "Test response"
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_mock_complete_multiple_calls(self):
        """Test multiple calls cycle through responses."""
        client = MockClaudeClient(responses=["First", "Second"])
        r1 = await client.complete("system", "user")
        r2 = await client.complete("system", "user")
        assert r1.content == "First"
        assert r2.content == "Second"

    @pytest.mark.asyncio
    async def test_mock_stream(self):
        """Test mock streaming."""
        client = MockClaudeClient(responses=["Hello world"])
        chunks = []
        async for chunk in client.stream("system", "user"):
            chunks.append(chunk)
        result = "".join(chunks).strip()
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_mock_tracks_calls(self):
        """Test that mock tracks calls."""
        client = MockClaudeClient()
        await client.complete("System message", "User message")
        assert len(client.calls) == 1
        assert client.calls[0]["system"] == "System message"
        assert client.calls[0]["user"] == "User message"

    @pytest.mark.asyncio
    async def test_mock_reset(self):
        """Test resetting mock client."""
        client = MockClaudeClient()
        await client.complete("system", "user")
        client.reset()
        assert len(client.calls) == 0

    @pytest.mark.asyncio
    async def test_mock_close(self):
        """Test mock close is no-op."""
        client = MockClaudeClient()
        await client.close()  # Should not raise


class TestClaudeClient:
    """Tests for ClaudeClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = ClaudeClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client.config is not None

    def test_client_custom_config(self):
        """Test client with custom config."""
        config = LLMConfig(temperature=0.5)
        client = ClaudeClient("test-api-key", config)
        assert client.config.temperature == 0.5

    def test_build_payload(self):
        """Test building API payload."""
        client = ClaudeClient("test-api-key")
        payload = client._build_payload(
            system="System",
            user="User",
            stream=False,
        )
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["system"] == "System"
        assert payload["messages"][0]["content"] == "User"
        assert payload["stream"] is False

    def test_build_payload_with_overrides(self):
        """Test building payload with parameter overrides."""
        client = ClaudeClient("test-api-key")
        payload = client._build_payload(
            system="System",
            user="User",
            temperature=0.7,
            max_tokens=2000,
        )
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 2000

    def test_parse_response(self):
        """Test parsing API response."""
        client = ClaudeClient("test-api-key")
        data = {
            "content": [{"type": "text", "text": "Response text"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "stop_reason": "end_turn",
        }
        response = client._parse_response(data)
        assert response.content == "Response text"
        assert response.input_tokens == 100
        assert response.output_tokens == 50

    @pytest.mark.asyncio
    async def test_client_close(self):
        """Test closing client."""
        client = ClaudeClient("test-api-key")
        await client.close()  # Should not raise


# =============================================================================
# Integration Tests
# =============================================================================


class TestLLMIntegration:
    """Integration tests for LLM module."""

    @pytest.mark.asyncio
    async def test_query_with_context(self):
        """Test querying with code context."""
        # Create context
        builder = ContextBuilder()
        snippet = builder.create_snippet(
            content="def calculate(x): return x * 2",
            file_path="calc.py",
            start_line=1,
            end_line=1,
            entity_name="calculate",
            entity_type="function",
        )
        context = builder.build([snippet], "What does calculate do?")

        # Query mock client
        client = MockClaudeClient(responses=["The calculate function doubles its input."])
        response = await client.query(
            question="What does calculate do?",
            context=context,
            query_type=QueryType.EXPLAIN,
        )

        assert "doubles" in response.content

    @pytest.mark.asyncio
    async def test_auto_query_classification(self):
        """Test automatic query classification."""
        client = MockClaudeClient()

        # Test explain query
        response = await client.query(
            question="How does this function work?",
            query_type=None,  # Should auto-classify
        )
        assert response is not None

    def test_full_prompt_formatting(self):
        """Test full prompt formatting pipeline."""
        # Build context
        builder = ContextBuilder()
        snippet = builder.create_snippet(
            content="async def fetch(): return await get_data()",
            file_path="api.py",
            start_line=10,
            end_line=10,
            entity_name="fetch",
            entity_type="function",
        )
        context = builder.build([snippet], "Explain fetch")

        # Get template
        template = get_prompt_template(QueryType.EXPLAIN)

        # Format message
        message = template.format_user_message(
            question="Explain how the fetch function works",
            context=context.format(),
        )

        assert "fetch" in message
        assert "api.py" in message
        assert "await get_data()" in message


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_llm_client_error(self):
        """Test LLMClientError."""
        error = LLMClientError("Test error")
        assert str(error) == "Test error"

    def test_invalid_query_type(self):
        """Test handling invalid query type."""
        # This should not raise, just return GENERAL
        result = classify_query("some random text that doesn't match patterns")
        assert result == QueryType.GENERAL
