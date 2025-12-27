"""LLM module for Kodo.

This module provides LLM integration for code understanding,
including prompt templates, context building, and Claude API client.
"""

from core.llm.client import ClaudeClient, LLMClient, LLMClientError, LLMResponse
from core.llm.context import CodeContext, ContextBuilder, ContextConfig
from core.llm.prompts import PromptTemplate, QueryType, get_prompt_template

__all__ = [
    # Client
    "ClaudeClient",
    "LLMClient",
    "LLMClientError",
    "LLMResponse",
    # Context
    "CodeContext",
    "ContextBuilder",
    "ContextConfig",
    # Prompts
    "PromptTemplate",
    "QueryType",
    "get_prompt_template",
]
