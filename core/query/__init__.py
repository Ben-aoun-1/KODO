"""Query module for Kodo.

This module provides the query engine that routes natural language questions
to appropriate handlers and orchestrates context building and LLM calls.
"""

from core.query.engine import QueryEngine, QueryEngineConfig
from core.query.handlers import (
    BaseHandler,
    DebugHandler,
    ExplainHandler,
    FindHandler,
    GeneralHandler,
    GenerateHandler,
    TraceHandler,
)
from core.query.models import (
    QueryRequest,
    QueryResponse,
    QueryResult,
    QueryStatus,
    SourceReference,
)
from core.query.router import QueryRouter, RouteResult

__all__ = [
    # Engine
    "QueryEngine",
    "QueryEngineConfig",
    # Handlers
    "BaseHandler",
    "DebugHandler",
    "ExplainHandler",
    "FindHandler",
    "GeneralHandler",
    "GenerateHandler",
    "TraceHandler",
    # Models
    "QueryRequest",
    "QueryResponse",
    "QueryResult",
    "QueryStatus",
    "SourceReference",
    # Router
    "QueryRouter",
    "RouteResult",
]
