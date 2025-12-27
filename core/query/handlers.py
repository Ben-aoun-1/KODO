"""Query handlers for different query types.

This module provides handler classes that implement specific logic
for different types of code queries.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.llm.context import CodeContext, ContextBuilder
from core.llm.prompts import QueryType
from core.query.models import QueryResult, SourceReference

logger = structlog.get_logger(__name__)


class HandlerContext(BaseModel):
    """Context passed to query handlers.

    Attributes:
        question: The user's question.
        repo_id: Repository identifier.
        code_context: Assembled code context.
        extracted_entities: Entities extracted from the query.
        file_context: Optional file path for focus.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str = Field(..., description="The question")
    repo_id: str = Field(..., description="Repository ID")
    code_context: CodeContext | None = Field(None, description="Code context")
    extracted_entities: list[str] = Field(default_factory=list, description="Extracted entities")
    file_context: str | None = Field(None, description="File context")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class BaseHandler(ABC):
    """Base class for query handlers.

    Handlers implement query-type-specific logic for building context
    and processing LLM responses.
    """

    def __init__(self) -> None:
        """Initialize the handler."""
        self._logger = logger.bind(handler=self.__class__.__name__)
        self._context_builder = ContextBuilder()

    @property
    @abstractmethod
    def query_type(self) -> QueryType:
        """Get the query type this handler handles."""
        ...

    @abstractmethod
    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build a query string for context retrieval.

        Args:
            ctx: Handler context.

        Returns:
            Query string for semantic search.
        """
        ...

    @abstractmethod
    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """Enhance the user's question with handler-specific instructions.

        Args:
            ctx: Handler context.

        Returns:
            Enhanced question.
        """
        ...

    def process_response(
        self,
        answer: str,
        ctx: HandlerContext,
    ) -> QueryResult:
        """Process the LLM response into a QueryResult.

        Args:
            answer: Raw LLM response.
            ctx: Handler context.

        Returns:
            Processed query result.
        """
        # Extract source references from context
        sources = self._extract_sources(ctx.code_context)

        # Generate follow-up questions
        follow_ups = self._suggest_follow_ups(ctx)

        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=self._estimate_confidence(ctx),
            follow_up_questions=follow_ups,
        )

    def _extract_sources(self, context: CodeContext | None) -> list[SourceReference]:
        """Extract source references from code context.

        Args:
            context: The code context.

        Returns:
            List of source references.
        """
        if not context or not context.snippets:
            return []

        sources = []
        for snippet in context.snippets:
            sources.append(
                SourceReference(
                    file_path=snippet.file_path,
                    start_line=snippet.start_line,
                    end_line=snippet.end_line,
                    entity_name=snippet.entity_name,
                    entity_type=snippet.entity_type,
                    relevance=snippet.relevance_score,
                    snippet=snippet.content[:200] if snippet.content else "",
                )
            )

        return sources

    def _estimate_confidence(self, ctx: HandlerContext) -> float:
        """Estimate confidence based on context quality.

        Args:
            ctx: Handler context.

        Returns:
            Confidence score (0-1).
        """
        if not ctx.code_context or ctx.code_context.is_empty:
            return 0.5

        # Base confidence on context quality
        snippets = ctx.code_context.snippets
        if not snippets:
            return 0.5

        # Average relevance of top snippets
        top_relevances = [s.relevance_score for s in snippets[:3]]
        avg_relevance = sum(top_relevances) / len(top_relevances)

        # Adjust based on context coverage
        coverage_factor = min(len(snippets) / 5.0, 1.0)

        return min(avg_relevance * (0.7 + 0.3 * coverage_factor), 1.0)

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest follow-up questions based on query type.

        Args:
            ctx: Handler context.

        Returns:
            List of follow-up question suggestions.
        """
        # Override in subclasses for type-specific suggestions
        return []


class ExplainHandler(BaseHandler):
    """Handler for explanation queries."""

    @property
    def query_type(self) -> QueryType:
        return QueryType.EXPLAIN

    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build context query for explanation."""
        # Include entity names if extracted
        if ctx.extracted_entities:
            return f"{ctx.question} {' '.join(ctx.extracted_entities)}"
        return ctx.question

    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """Enhance prompt for explanation."""
        enhanced = ctx.question
        if ctx.extracted_entities:
            enhanced += f"\n\nFocus on explaining: {', '.join(ctx.extracted_entities)}"
        return enhanced

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest follow-ups for explanation queries."""
        suggestions = []
        if ctx.extracted_entities:
            entity = ctx.extracted_entities[0]
            suggestions.extend(
                [
                    f"What functions call {entity}?",
                    f"What are the edge cases for {entity}?",
                    f"How would you test {entity}?",
                ]
            )
        return suggestions[:3]


class TraceHandler(BaseHandler):
    """Handler for trace/flow queries."""

    @property
    def query_type(self) -> QueryType:
        return QueryType.TRACE

    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build context query for tracing."""
        # Include call graph context
        base = ctx.question
        if ctx.extracted_entities:
            base += f" call graph {' '.join(ctx.extracted_entities)}"
        return base

    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """Enhance prompt for tracing."""
        enhanced = ctx.question
        enhanced += "\n\nProvide a step-by-step trace showing:"
        enhanced += "\n- The entry point"
        enhanced += "\n- Each function/method called in order"
        enhanced += "\n- Data transformations at each step"
        enhanced += "\n- The final output or side effects"
        return enhanced

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest follow-ups for trace queries."""
        suggestions = []
        if ctx.extracted_entities:
            entity = ctx.extracted_entities[0]
            suggestions.extend(
                [
                    f"What data does {entity} modify?",
                    f"What happens if {entity} fails?",
                    f"Show the reverse trace - what calls {entity}?",
                ]
            )
        return suggestions[:3]


class FindHandler(BaseHandler):
    """Handler for find/search queries."""

    @property
    def query_type(self) -> QueryType:
        return QueryType.FIND

    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build context query for finding."""
        return ctx.question

    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """Enhance prompt for finding."""
        enhanced = ctx.question
        enhanced += "\n\nFor each match found, provide:"
        enhanced += "\n- File path and line numbers"
        enhanced += "\n- Brief description of what it does"
        enhanced += "\n- Why it matches the search criteria"
        return enhanced

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest follow-ups for find queries."""
        return [
            "Show me similar code patterns",
            "Which of these are most commonly used?",
            "Are any of these deprecated or need updating?",
        ]


class GenerateHandler(BaseHandler):
    """Handler for code generation queries."""

    @property
    def query_type(self) -> QueryType:
        return QueryType.GENERATE

    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build context query for generation."""
        # Look for similar existing implementations
        return f"similar implementation {ctx.question}"

    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """Enhance prompt for generation."""
        enhanced = ctx.question
        enhanced += "\n\nWhen generating code:"
        enhanced += "\n- Follow the existing code style in the codebase"
        enhanced += "\n- Include appropriate type hints"
        enhanced += "\n- Add a docstring explaining the purpose"
        enhanced += "\n- Consider error handling"
        if ctx.file_context:
            enhanced += f"\n- Place this in: {ctx.file_context}"
        return enhanced

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest follow-ups for generation queries."""
        return [
            "Generate tests for this code",
            "What edge cases should I handle?",
            "How can I optimize this?",
        ]


class DebugHandler(BaseHandler):
    """Handler for debugging queries."""

    @property
    def query_type(self) -> QueryType:
        return QueryType.DEBUG

    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build context query for debugging."""
        return ctx.question

    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """Enhance prompt for debugging."""
        enhanced = ctx.question
        enhanced += "\n\nHelp debug this by:"
        enhanced += "\n1. Analyzing the likely root cause"
        enhanced += "\n2. Suggesting specific debugging steps"
        enhanced += "\n3. Providing potential fixes"
        enhanced += "\n4. Explaining why the fix works"
        return enhanced

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest follow-ups for debug queries."""
        return [
            "What could cause similar bugs?",
            "How can I prevent this in the future?",
            "Should I add tests for this case?",
        ]


class GeneralHandler(BaseHandler):
    """Handler for general code questions."""

    @property
    def query_type(self) -> QueryType:
        return QueryType.GENERAL

    def build_context_query(self, ctx: HandlerContext) -> str:
        """Build context query for general questions."""
        return ctx.question

    def enhance_prompt(self, ctx: HandlerContext) -> str:
        """No enhancement for general questions."""
        return ctx.question

    def _suggest_follow_ups(self, ctx: HandlerContext) -> list[str]:
        """Suggest generic follow-ups."""
        return []


# Handler registry
HANDLERS: dict[QueryType, type[BaseHandler]] = {
    QueryType.EXPLAIN: ExplainHandler,
    QueryType.TRACE: TraceHandler,
    QueryType.FIND: FindHandler,
    QueryType.GENERATE: GenerateHandler,
    QueryType.DEBUG: DebugHandler,
    QueryType.GENERAL: GeneralHandler,
    # These use GeneralHandler as fallback until implemented
    QueryType.REVIEW: GeneralHandler,
    QueryType.REFACTOR: GeneralHandler,
    QueryType.DOCUMENT: GeneralHandler,
    QueryType.TEST: GeneralHandler,
}


def get_handler(query_type: QueryType) -> BaseHandler:
    """Get a handler instance for a query type.

    Args:
        query_type: The query type.

    Returns:
        Handler instance.
    """
    handler_class = HANDLERS.get(query_type, GeneralHandler)
    return handler_class()
