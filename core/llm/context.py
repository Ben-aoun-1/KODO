"""Context building for LLM queries.

This module provides functionality to assemble code context for LLM queries,
including token budgeting and prioritization.
"""

import contextlib
from typing import Any

import structlog
import tiktoken
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class CodeSnippet(BaseModel):
    """A snippet of code to include in context.

    Attributes:
        content: The code content.
        file_path: Path to the source file.
        start_line: Starting line number.
        end_line: Ending line number.
        entity_name: Name of the code entity.
        entity_type: Type of entity (function, class, etc.).
        language: Programming language.
        relevance_score: How relevant this snippet is (0-1).
        token_count: Estimated token count.
    """

    model_config = ConfigDict(frozen=True)

    content: str = Field(..., description="Code content")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    entity_name: str = Field(default="", description="Entity name")
    entity_type: str = Field(default="code", description="Entity type")
    language: str = Field(default="python", description="Language")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Relevance")
    token_count: int = Field(default=0, description="Token count")

    def format(self) -> str:
        """Format the snippet for inclusion in context.

        Returns:
            Formatted code snippet with metadata.
        """
        location = f"{self.file_path}:{self.start_line}-{self.end_line}"
        header = f"### {self.entity_type}: {self.entity_name} ({location})"
        return f"{header}\n\n```{self.language}\n{self.content}\n```"


class CodeContext(BaseModel):
    """Assembled code context for an LLM query.

    Attributes:
        snippets: List of code snippets included in context.
        total_tokens: Total token count.
        max_tokens: Maximum allowed tokens.
        repo_id: Repository identifier.
        query: Original query that this context is for.
        metadata: Additional context metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    snippets: list[CodeSnippet] = Field(default_factory=list, description="Code snippets")
    total_tokens: int = Field(default=0, description="Total tokens")
    max_tokens: int = Field(default=8000, description="Max tokens")
    repo_id: str = Field(default="", description="Repository ID")
    query: str = Field(default="", description="Original query")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def format(self) -> str:
        """Format the full context for LLM consumption.

        Returns:
            Formatted context string.
        """
        if not self.snippets:
            return "No relevant code found in the codebase."

        parts = []
        for snippet in self.snippets:
            parts.append(snippet.format())

        return "\n\n".join(parts)

    def add_snippet(self, snippet: CodeSnippet) -> bool:
        """Add a snippet to the context if it fits.

        Args:
            snippet: Snippet to add.

        Returns:
            True if snippet was added, False if it would exceed token limit.
        """
        if self.total_tokens + snippet.token_count > self.max_tokens:
            return False

        self.snippets.append(snippet)
        self.total_tokens += snippet.token_count
        return True

    @property
    def is_empty(self) -> bool:
        """Check if context has no snippets."""
        return len(self.snippets) == 0


class ContextConfig(BaseModel):
    """Configuration for context building.

    Attributes:
        max_tokens: Maximum tokens for context.
        max_snippets: Maximum number of snippets.
        min_relevance: Minimum relevance score to include.
        include_imports: Include import statements.
        include_docstrings: Include docstrings.
        include_related: Include related code (callers/callees).
        related_depth: Depth for related code traversal.
    """

    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(default=8000, ge=1000, le=32000, description="Max tokens")
    max_snippets: int = Field(default=20, ge=1, le=100, description="Max snippets")
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Min relevance")
    include_imports: bool = Field(default=True, description="Include imports")
    include_docstrings: bool = Field(default=True, description="Include docstrings")
    include_related: bool = Field(default=True, description="Include related code")
    related_depth: int = Field(default=1, ge=0, le=3, description="Related code depth")


class TokenCounter:
    """Utility for counting tokens in text.

    Uses tiktoken for accurate token counting with Claude's tokenizer.
    """

    def __init__(self, model: str = "cl100k_base") -> None:
        """Initialize token counter.

        Args:
            model: Tokenizer model to use.
        """
        self._encoder: tiktoken.Encoding | None = None
        with contextlib.suppress(Exception):
            self._encoder = tiktoken.get_encoding(model)
        self._logger = logger.bind(component="token_counter")

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        if self._encoder:
            return len(self._encoder.encode(text))
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4

    def count_messages(
        self,
        system: str,
        user: str,
        assistant: str = "",
    ) -> int:
        """Count tokens in a message set.

        Args:
            system: System message.
            user: User message.
            assistant: Assistant message (if any).

        Returns:
            Total token count including message overhead.
        """
        # Account for message formatting overhead (approx 4 tokens per message)
        overhead = 12
        return overhead + self.count(system) + self.count(user) + self.count(assistant)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text.
        """
        if self._encoder:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return str(self._encoder.decode(tokens[:max_tokens]))

        # Fallback: rough estimate
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


class ContextBuilder:
    """Builds code context for LLM queries.

    This class assembles relevant code snippets into a context
    that fits within token limits, prioritizing by relevance.
    """

    def __init__(
        self,
        config: ContextConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Initialize context builder.

        Args:
            config: Context configuration.
            token_counter: Token counter instance.
        """
        self.config = config or ContextConfig()
        self._token_counter = token_counter or TokenCounter()
        self._logger = logger.bind(component="context_builder")

    def build(
        self,
        snippets: list[CodeSnippet],
        query: str,
        repo_id: str = "",
        max_tokens: int | None = None,
    ) -> CodeContext:
        """Build context from code snippets.

        Args:
            snippets: Available code snippets.
            query: The user's query.
            repo_id: Repository identifier.
            max_tokens: Override max tokens (uses config if not provided).

        Returns:
            Assembled code context.
        """
        max_tokens = max_tokens or self.config.max_tokens

        # Filter by minimum relevance
        eligible = [s for s in snippets if s.relevance_score >= self.config.min_relevance]

        # Sort by relevance (highest first)
        eligible.sort(key=lambda s: s.relevance_score, reverse=True)

        # Limit to max snippets
        eligible = eligible[: self.config.max_snippets]

        # Calculate token counts if not already set
        snippets_with_counts = []
        for snippet in eligible:
            if snippet.token_count == 0:
                token_count = self._token_counter.count(snippet.format())
                snippet = CodeSnippet(
                    content=snippet.content,
                    file_path=snippet.file_path,
                    start_line=snippet.start_line,
                    end_line=snippet.end_line,
                    entity_name=snippet.entity_name,
                    entity_type=snippet.entity_type,
                    language=snippet.language,
                    relevance_score=snippet.relevance_score,
                    token_count=token_count,
                )
            snippets_with_counts.append(snippet)

        # Build context within token budget
        context = CodeContext(
            max_tokens=max_tokens,
            repo_id=repo_id,
            query=query,
        )

        for snippet in snippets_with_counts:
            if not context.add_snippet(snippet):
                # Token budget exhausted
                break

        self._logger.info(
            "context_built",
            query_length=len(query),
            snippets_included=len(context.snippets),
            total_tokens=context.total_tokens,
            max_tokens=max_tokens,
        )

        return context

    def build_from_search_results(
        self,
        results: list[dict[str, Any]],
        query: str,
        repo_id: str = "",
    ) -> CodeContext:
        """Build context from semantic search results.

        Args:
            results: Search results with score, content, and metadata.
            query: The user's query.
            repo_id: Repository identifier.

        Returns:
            Assembled code context.
        """
        snippets = []
        for result in results:
            snippet = CodeSnippet(
                content=result.get("content", ""),
                file_path=result.get("file_path", "unknown"),
                start_line=result.get("start_line", 0),
                end_line=result.get("end_line", 0),
                entity_name=result.get("entity_name", ""),
                entity_type=result.get("entity_type", "code"),
                language=result.get("language", "python"),
                relevance_score=result.get("score", 0.5),
            )
            snippets.append(snippet)

        return self.build(snippets, query, repo_id)

    def create_snippet(
        self,
        content: str,
        file_path: str,
        start_line: int,
        end_line: int,
        entity_name: str = "",
        entity_type: str = "code",
        language: str = "python",
        relevance_score: float = 1.0,
    ) -> CodeSnippet:
        """Create a code snippet with token count.

        Args:
            content: Code content.
            file_path: Source file path.
            start_line: Starting line.
            end_line: Ending line.
            entity_name: Entity name.
            entity_type: Entity type.
            language: Programming language.
            relevance_score: Relevance score.

        Returns:
            Code snippet with calculated token count.
        """
        snippet = CodeSnippet(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            entity_name=entity_name,
            entity_type=entity_type,
            language=language,
            relevance_score=relevance_score,
        )
        token_count = self._token_counter.count(snippet.format())
        return CodeSnippet(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            entity_name=entity_name,
            entity_type=entity_type,
            language=language,
            relevance_score=relevance_score,
            token_count=token_count,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        return self._token_counter.count(text)

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum tokens.

        Returns:
            Truncated text.
        """
        return self._token_counter.truncate(text, max_tokens)
