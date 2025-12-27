"""Query router for classifying and routing queries.

This module provides query classification and routing logic.
"""

import re

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.llm.prompts import QueryType

logger = structlog.get_logger(__name__)


class RouteResult(BaseModel):
    """Result of query routing.

    Attributes:
        query_type: Classified query type.
        confidence: Confidence in the classification.
        extracted_entities: Entities extracted from the query.
        suggested_context: Suggested context types to include.
    """

    model_config = ConfigDict(frozen=True)

    query_type: QueryType = Field(..., description="Classified query type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Classification confidence")
    extracted_entities: list[str] = Field(
        default_factory=list, description="Extracted entity names"
    )
    suggested_context: list[str] = Field(
        default_factory=list, description="Suggested context types"
    )


class QueryRouter:
    """Routes queries to appropriate handlers based on classification.

    This class analyzes natural language queries to determine their type
    and extract relevant information for context building.
    """

    # Keywords for each query type (ordered by priority)
    _QUERY_PATTERNS: dict[QueryType, list[str]] = {
        QueryType.EXPLAIN: [
            r"\bexplain\b",
            r"\bhow does\b",
            r"\bwhat does\b",
            r"\bwhat is\b",
            r"\bunderstand\b",
            r"\bdescribe\b",
            r"\bwalk.*through\b",
            r"\btell me about\b",
        ],
        QueryType.TRACE: [
            r"\btrace\b",
            r"\bflow\b",
            r"\bpath\b",
            r"\bcall chain\b",
            r"\bfollows?\b",
            r"\bexecution\b",
            r"\bwhere.*called\b",
            r"\bcalls?\b.*\bfrom\b",
            r"\bdata.*flow\b",
            r"\bcontrol.*flow\b",
        ],
        QueryType.FIND: [
            r"\bfind\b",
            r"\bsearch\b",
            r"\bwhere\b",
            r"\blocate\b",
            r"\bwhich\b",
            r"\blist\b.*\ball\b",
            r"\bshow.*all\b",
            r"\blook.*for\b",
        ],
        QueryType.GENERATE: [
            r"\bgenerate\b",
            r"\bcreate\b",
            r"\bwrite\b",
            r"\bimplement\b",
            r"\badd\b",
            r"\bbuild\b",
            r"\bmake\b.*\bfunction\b",
            r"\bmake\b.*\bclass\b",
        ],
        QueryType.REVIEW: [
            r"\breview\b",
            r"\bcheck\b",
            r"\bissues?\b",
            r"\bproblems?\b",
            r"\bquality\b",
            r"\bcode\s*review\b",
        ],
        QueryType.REFACTOR: [
            r"\brefactor\b",
            r"\bimprove\b",
            r"\bclean\b",
            r"\bsimplify\b",
            r"\boptimize\b",
            r"\brestructure\b",
        ],
        QueryType.DEBUG: [
            r"\bdebug\b",
            r"\bfix\b",
            r"\berror\b",
            r"\bbug\b",
            r"\bbroken\b",
            r"\bfailing\b",
            r"\bcrash\b",
            r"\bexception\b",
            r"\bnot working\b",
        ],
        QueryType.DOCUMENT: [
            r"\bdocument\b",
            r"\bdocstring\b",
            r"\bcomment\b",
            r"\bdocumentation\b",
            r"\bapi\s*doc\b",
        ],
        QueryType.TEST: [
            r"\btest\b",
            r"\bunittest\b",
            r"\bpytest\b",
            r"\btest\s*case\b",
            r"\bunit\s*test\b",
        ],
    }

    # Entity extraction patterns
    _ENTITY_PATTERNS = [
        r"`([^`]+)`",  # Backtick-quoted names
        r"(?:function|method|class|module)\s+['\"]?(\w+)['\"]?",  # Named entities
        r"(?:the\s+)?(\w+)\s+(?:function|method|class|module)",  # "the X function"
        r"(\w+)\(\)",  # Function calls
        r"(\w+)\.(\w+)",  # Module.attribute
    ]

    # Context suggestions based on query type
    _CONTEXT_SUGGESTIONS: dict[QueryType, list[str]] = {
        QueryType.EXPLAIN: ["definition", "docstring", "callers", "callees"],
        QueryType.TRACE: ["callees", "callers", "data_flow"],
        QueryType.FIND: ["semantic_search", "graph_search"],
        QueryType.GENERATE: ["similar_code", "imports", "types"],
        QueryType.REVIEW: ["definition", "callers", "tests"],
        QueryType.REFACTOR: ["definition", "callers", "callees", "similar_code"],
        QueryType.DEBUG: ["definition", "callers", "callees", "stack_trace"],
        QueryType.DOCUMENT: ["definition", "usage_examples"],
        QueryType.TEST: ["definition", "existing_tests", "edge_cases"],
        QueryType.GENERAL: ["semantic_search"],
    }

    def __init__(self) -> None:
        """Initialize the query router."""
        self._logger = logger.bind(component="query_router")
        # Compile regex patterns
        self._compiled_patterns: dict[QueryType, list[re.Pattern[str]]] = {
            qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qtype, patterns in self._QUERY_PATTERNS.items()
        }
        self._entity_patterns = [re.compile(p, re.IGNORECASE) for p in self._ENTITY_PATTERNS]

    def route(self, question: str) -> RouteResult:
        """Route a query to determine its type and extract information.

        Args:
            question: The natural language question.

        Returns:
            Routing result with type, confidence, and extracted info.
        """
        # Classify the query type
        query_type, confidence = self._classify(question)

        # Extract entities mentioned in the query
        entities = self._extract_entities(question)

        # Get suggested context types
        suggestions = self._CONTEXT_SUGGESTIONS.get(query_type, ["semantic_search"])

        self._logger.debug(
            "query_routed",
            question_preview=question[:50],
            query_type=query_type.value,
            confidence=confidence,
            entities=entities,
        )

        return RouteResult(
            query_type=query_type,
            confidence=confidence,
            extracted_entities=entities,
            suggested_context=list(suggestions),
        )

    def _classify(self, question: str) -> tuple[QueryType, float]:
        """Classify the query type.

        Args:
            question: The question to classify.

        Returns:
            Tuple of (query_type, confidence).
        """
        scores: dict[QueryType, float] = {}

        for query_type, patterns in self._compiled_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern.search(question):
                    score += 1.0

            if score > 0:
                # Normalize by number of patterns (max 1.0)
                scores[query_type] = min(score / 2.0, 1.0)

        if not scores:
            return QueryType.GENERAL, 0.5

        # Get the highest scoring type
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        # Adjust confidence based on score distinctiveness
        if len(scores) > 1:
            sorted_scores = sorted(scores.values(), reverse=True)
            if sorted_scores[0] > sorted_scores[1] * 1.5:
                confidence = min(best_score + 0.1, 1.0)
            else:
                confidence = best_score * 0.9
        else:
            confidence = best_score

        return best_type, confidence

    def _extract_entities(self, question: str) -> list[str]:
        """Extract entity names from the question.

        Args:
            question: The question to extract from.

        Returns:
            List of extracted entity names.
        """
        entities: set[str] = set()

        for pattern in self._entity_patterns:
            matches = pattern.findall(question)
            for match in matches:
                if isinstance(match, tuple):
                    # Multi-group match (e.g., module.attribute)
                    entities.update(m for m in match if m and len(m) > 1)
                elif match and len(match) > 1:
                    entities.add(match)

        # Filter out common words
        common_words = {
            "the",
            "this",
            "that",
            "and",
            "for",
            "with",
            "from",
            "how",
            "what",
            "where",
            "when",
            "why",
            "does",
            "function",
            "method",
            "class",
            "module",
            "file",
            "code",
        }
        entities = {e for e in entities if e.lower() not in common_words}

        return list(entities)

    def get_handler_name(self, query_type: QueryType) -> str:
        """Get the handler name for a query type.

        Args:
            query_type: The query type.

        Returns:
            Handler class name.
        """
        handlers = {
            QueryType.EXPLAIN: "ExplainHandler",
            QueryType.TRACE: "TraceHandler",
            QueryType.FIND: "FindHandler",
            QueryType.GENERATE: "GenerateHandler",
            QueryType.REVIEW: "ReviewHandler",
            QueryType.REFACTOR: "RefactorHandler",
            QueryType.DEBUG: "DebugHandler",
            QueryType.DOCUMENT: "DocumentHandler",
            QueryType.TEST: "TestHandler",
            QueryType.GENERAL: "GeneralHandler",
        }
        return handlers.get(query_type, "GeneralHandler")
