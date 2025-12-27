"""Prompt templates for code-related LLM queries.

This module defines prompt templates for different types of code queries
including explanation, tracing, finding, and generation.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryType(str, Enum):
    """Types of code queries supported by the system."""

    EXPLAIN = "explain"  # Explain how code works
    TRACE = "trace"  # Trace data/control flow
    FIND = "find"  # Find code matching criteria
    GENERATE = "generate"  # Generate code
    REVIEW = "review"  # Review code for issues
    REFACTOR = "refactor"  # Suggest refactoring
    DEBUG = "debug"  # Help debug issues
    DOCUMENT = "document"  # Generate documentation
    TEST = "test"  # Generate tests
    GENERAL = "general"  # General code questions


class PromptTemplate(BaseModel):
    """A prompt template for LLM queries.

    Attributes:
        query_type: Type of query this template handles.
        system_prompt: System message for the LLM.
        user_template: Template for user message with placeholders.
        requires_context: Whether code context is required.
        max_context_tokens: Maximum tokens for context.
    """

    model_config = ConfigDict(frozen=True)

    query_type: QueryType = Field(..., description="Query type")
    system_prompt: str = Field(..., description="System prompt")
    user_template: str = Field(..., description="User message template")
    requires_context: bool = Field(default=True, description="Requires context")
    max_context_tokens: int = Field(default=8000, description="Max context tokens")

    def format_user_message(
        self,
        question: str,
        context: str = "",
        **kwargs: Any,
    ) -> str:
        """Format the user message with provided values.

        Args:
            question: The user's question.
            context: Code context to include.
            **kwargs: Additional template variables.

        Returns:
            Formatted user message.
        """
        return self.user_template.format(
            question=question,
            context=context,
            **kwargs,
        )


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are Kodo, an expert code assistant that helps developers understand and work with codebases. You have deep knowledge of software engineering, design patterns, and best practices.

Key principles:
- Be precise and accurate in your explanations
- Reference specific code locations (file:line) when discussing code
- Explain the "why" behind code decisions, not just the "what"
- When uncertain, acknowledge limitations rather than guessing
- Use clear, concise language suitable for developers

You have access to the codebase context provided below. Use this context to give informed, specific answers."""

SYSTEM_PROMPT_EXPLAIN = (
    SYSTEM_PROMPT_BASE
    + """

For explanation queries:
- Start with a high-level overview before diving into details
- Explain the purpose and responsibility of each component
- Describe how different parts interact with each other
- Highlight any design patterns or architectural decisions
- Note any potential issues or areas of complexity"""
)

SYSTEM_PROMPT_TRACE = (
    SYSTEM_PROMPT_BASE
    + """

For tracing queries:
- Follow the execution path step by step
- Identify all functions/methods involved in the flow
- Note any data transformations along the way
- Highlight branching logic and conditions
- Track variable state changes where relevant"""
)

SYSTEM_PROMPT_FIND = (
    SYSTEM_PROMPT_BASE
    + """

For finding queries:
- Search through the provided context thoroughly
- List all matching items with their locations
- Explain why each match is relevant
- Suggest related items that might also be useful
- If nothing matches, suggest alternatives"""
)

SYSTEM_PROMPT_GENERATE = (
    SYSTEM_PROMPT_BASE
    + """

For code generation:
- Follow the existing code style and conventions in the codebase
- Use appropriate types and error handling
- Include docstrings and comments where helpful
- Consider edge cases and error conditions
- Make code testable and maintainable"""
)

SYSTEM_PROMPT_REVIEW = (
    SYSTEM_PROMPT_BASE
    + """

For code review:
- Check for bugs, logic errors, and edge cases
- Evaluate code quality and maintainability
- Identify security vulnerabilities if any
- Suggest improvements with specific examples
- Prioritize issues by severity"""
)

SYSTEM_PROMPT_REFACTOR = (
    SYSTEM_PROMPT_BASE
    + """

For refactoring suggestions:
- Identify code smells and anti-patterns
- Suggest specific refactoring techniques
- Explain the benefits of each suggestion
- Consider backward compatibility
- Provide before/after examples when helpful"""
)

SYSTEM_PROMPT_DEBUG = (
    SYSTEM_PROMPT_BASE
    + """

For debugging assistance:
- Analyze the problem systematically
- Identify potential root causes
- Suggest debugging steps to isolate the issue
- Provide potential fixes with explanations
- Consider related areas that might be affected"""
)

SYSTEM_PROMPT_DOCUMENT = (
    SYSTEM_PROMPT_BASE
    + """

For documentation:
- Write clear, comprehensive documentation
- Include usage examples
- Document parameters, returns, and exceptions
- Follow the project's documentation style
- Add helpful context for future maintainers"""
)

SYSTEM_PROMPT_TEST = (
    SYSTEM_PROMPT_BASE
    + """

For test generation:
- Cover happy paths and edge cases
- Use appropriate testing patterns
- Mock external dependencies properly
- Write clear test descriptions
- Follow the project's testing conventions"""
)

SYSTEM_PROMPT_GENERAL = SYSTEM_PROMPT_BASE


# =============================================================================
# User Message Templates
# =============================================================================

USER_TEMPLATE_WITH_CONTEXT = """## Code Context

{context}

## Question

{question}"""

USER_TEMPLATE_NO_CONTEXT = """## Question

{question}"""

USER_TEMPLATE_WITH_CODE = """## Code Context

{context}

## Code to Analyze

```{language}
{code}
```

## Question

{question}"""


# =============================================================================
# Prompt Templates
# =============================================================================

EXPLAIN_TEMPLATE = PromptTemplate(
    query_type=QueryType.EXPLAIN,
    system_prompt=SYSTEM_PROMPT_EXPLAIN,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=8000,
)

TRACE_TEMPLATE = PromptTemplate(
    query_type=QueryType.TRACE,
    system_prompt=SYSTEM_PROMPT_TRACE,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=10000,
)

FIND_TEMPLATE = PromptTemplate(
    query_type=QueryType.FIND,
    system_prompt=SYSTEM_PROMPT_FIND,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=12000,
)

GENERATE_TEMPLATE = PromptTemplate(
    query_type=QueryType.GENERATE,
    system_prompt=SYSTEM_PROMPT_GENERATE,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=6000,
)

REVIEW_TEMPLATE = PromptTemplate(
    query_type=QueryType.REVIEW,
    system_prompt=SYSTEM_PROMPT_REVIEW,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=8000,
)

REFACTOR_TEMPLATE = PromptTemplate(
    query_type=QueryType.REFACTOR,
    system_prompt=SYSTEM_PROMPT_REFACTOR,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=8000,
)

DEBUG_TEMPLATE = PromptTemplate(
    query_type=QueryType.DEBUG,
    system_prompt=SYSTEM_PROMPT_DEBUG,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=10000,
)

DOCUMENT_TEMPLATE = PromptTemplate(
    query_type=QueryType.DOCUMENT,
    system_prompt=SYSTEM_PROMPT_DOCUMENT,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=6000,
)

TEST_TEMPLATE = PromptTemplate(
    query_type=QueryType.TEST,
    system_prompt=SYSTEM_PROMPT_TEST,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=True,
    max_context_tokens=8000,
)

GENERAL_TEMPLATE = PromptTemplate(
    query_type=QueryType.GENERAL,
    system_prompt=SYSTEM_PROMPT_GENERAL,
    user_template=USER_TEMPLATE_WITH_CONTEXT,
    requires_context=False,
    max_context_tokens=8000,
)


# Template registry
_TEMPLATES: dict[QueryType, PromptTemplate] = {
    QueryType.EXPLAIN: EXPLAIN_TEMPLATE,
    QueryType.TRACE: TRACE_TEMPLATE,
    QueryType.FIND: FIND_TEMPLATE,
    QueryType.GENERATE: GENERATE_TEMPLATE,
    QueryType.REVIEW: REVIEW_TEMPLATE,
    QueryType.REFACTOR: REFACTOR_TEMPLATE,
    QueryType.DEBUG: DEBUG_TEMPLATE,
    QueryType.DOCUMENT: DOCUMENT_TEMPLATE,
    QueryType.TEST: TEST_TEMPLATE,
    QueryType.GENERAL: GENERAL_TEMPLATE,
}


def get_prompt_template(query_type: QueryType) -> PromptTemplate:
    """Get the prompt template for a query type.

    Args:
        query_type: Type of query.

    Returns:
        Corresponding prompt template.

    Raises:
        ValueError: If query type is not supported.
    """
    template = _TEMPLATES.get(query_type)
    if template is None:
        raise ValueError(f"Unsupported query type: {query_type}")
    return template


def classify_query(question: str) -> QueryType:
    """Classify a question into a query type.

    Uses keyword matching to determine the most appropriate query type.

    Args:
        question: The user's question.

    Returns:
        Classified query type.
    """
    question_lower = question.lower()

    # Check for specific query types based on keywords
    if any(kw in question_lower for kw in ["explain", "how does", "what does", "understand"]):
        return QueryType.EXPLAIN

    if any(kw in question_lower for kw in ["trace", "flow", "path", "call chain", "follows"]):
        return QueryType.TRACE

    if any(kw in question_lower for kw in ["find", "search", "where", "locate", "which"]):
        return QueryType.FIND

    if any(kw in question_lower for kw in ["generate", "create", "write", "implement", "add"]):
        return QueryType.GENERATE

    if any(kw in question_lower for kw in ["review", "check", "issues", "problems"]):
        return QueryType.REVIEW

    if any(kw in question_lower for kw in ["refactor", "improve", "clean", "simplify"]):
        return QueryType.REFACTOR

    if any(kw in question_lower for kw in ["debug", "fix", "error", "bug", "broken"]):
        return QueryType.DEBUG

    if any(kw in question_lower for kw in ["document", "docstring", "comment"]):
        return QueryType.DOCUMENT

    if any(kw in question_lower for kw in ["test", "unittest", "pytest"]):
        return QueryType.TEST

    # Default to general for unclassified queries
    return QueryType.GENERAL
