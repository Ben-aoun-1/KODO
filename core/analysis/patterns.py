"""Pattern detection for common code patterns and anti-patterns.

This module provides tools for detecting design patterns, anti-patterns,
and code smells in Python code.
"""

import re
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.analysis.models import (
    AnalysisCategory,
    AnalysisSeverity,
    Finding,
)

logger = structlog.get_logger(__name__)


class PatternType(str, Enum):
    """Types of code patterns."""

    # Design patterns
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    DECORATOR = "decorator"
    STRATEGY = "strategy"
    ADAPTER = "adapter"
    FACADE = "facade"

    # Anti-patterns / Code smells
    GOD_CLASS = "god_class"
    LONG_METHOD = "long_method"
    LONG_PARAMETER_LIST = "long_parameter_list"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    MAGIC_NUMBERS = "magic_numbers"
    GLOBAL_STATE = "global_state"
    TIGHT_COUPLING = "tight_coupling"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"


class PatternCategory(str, Enum):
    """Categories of patterns."""

    DESIGN_PATTERN = "design_pattern"
    ANTI_PATTERN = "anti_pattern"
    CODE_SMELL = "code_smell"
    SECURITY = "security"
    PERFORMANCE = "performance"


class CodePattern(BaseModel):
    """Definition of a code pattern to detect.

    Attributes:
        id: Unique pattern identifier.
        name: Human-readable pattern name.
        pattern_type: Type of pattern.
        category: Pattern category.
        description: Description of the pattern.
        severity: Severity if this is a problem pattern.
        regex_patterns: Regex patterns to match.
        structural_rules: Structural rules for AST matching.
        examples: Example code snippets.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Pattern ID")
    name: str = Field(..., description="Pattern name")
    pattern_type: PatternType = Field(..., description="Pattern type")
    category: PatternCategory = Field(..., description="Category")
    description: str = Field(default="", description="Description")
    severity: AnalysisSeverity = Field(
        default=AnalysisSeverity.INFO,
        description="Severity for problem patterns",
    )
    regex_patterns: list[str] = Field(default_factory=list, description="Regex patterns")
    structural_rules: dict[str, Any] = Field(
        default_factory=dict,
        description="Structural matching rules",
    )
    examples: list[str] = Field(default_factory=list, description="Example code")


class PatternMatch(BaseModel):
    """A match of a pattern in code.

    Attributes:
        pattern_id: ID of the matched pattern.
        pattern_name: Name of the matched pattern.
        pattern_type: Type of pattern.
        category: Pattern category.
        file_path: Path to the file.
        start_line: Starting line number.
        end_line: Ending line number.
        entity_name: Name of the matched entity.
        entity_type: Type of the entity.
        confidence: Confidence score (0-1).
        matched_code: Snippet of matched code.
        context: Additional context.
    """

    model_config = ConfigDict(frozen=True)

    pattern_id: str = Field(..., description="Pattern ID")
    pattern_name: str = Field(..., description="Pattern name")
    pattern_type: PatternType = Field(..., description="Pattern type")
    category: PatternCategory = Field(..., description="Category")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(default=0, ge=0, description="Start line")
    end_line: int = Field(default=0, ge=0, description="End line")
    entity_name: str | None = Field(None, description="Entity name")
    entity_type: str | None = Field(None, description="Entity type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence")
    matched_code: str | None = Field(None, description="Matched code snippet")
    context: dict[str, Any] = Field(default_factory=dict, description="Context")


class PatternDetector:
    """Detector for code patterns and anti-patterns.

    This class provides methods for detecting various patterns in Python code
    using both regex matching and structural analysis.
    """

    # Default patterns to detect
    DEFAULT_PATTERNS: list[CodePattern] = [
        CodePattern(
            id="singleton",
            name="Singleton Pattern",
            pattern_type=PatternType.SINGLETON,
            category=PatternCategory.DESIGN_PATTERN,
            description="Class with private constructor and instance method",
            regex_patterns=[
                r"_instance\s*=\s*None",
                r"def\s+get_instance\s*\(",
                r"def\s+getInstance\s*\(",
            ],
        ),
        CodePattern(
            id="factory",
            name="Factory Pattern",
            pattern_type=PatternType.FACTORY,
            category=PatternCategory.DESIGN_PATTERN,
            description="Factory method or class for object creation",
            regex_patterns=[
                r"def\s+create_\w+\s*\(",
                r"class\s+\w*Factory\b",
                r"def\s+factory\s*\(",
            ],
        ),
        CodePattern(
            id="god_class",
            name="God Class",
            pattern_type=PatternType.GOD_CLASS,
            category=PatternCategory.CODE_SMELL,
            description="Class with too many responsibilities",
            severity=AnalysisSeverity.WARNING,
            structural_rules={
                "min_methods": 15,
                "min_lines": 500,
            },
        ),
        CodePattern(
            id="long_method",
            name="Long Method",
            pattern_type=PatternType.LONG_METHOD,
            category=PatternCategory.CODE_SMELL,
            description="Method that is too long",
            severity=AnalysisSeverity.WARNING,
            structural_rules={
                "max_lines": 50,
            },
        ),
        CodePattern(
            id="long_parameter_list",
            name="Long Parameter List",
            pattern_type=PatternType.LONG_PARAMETER_LIST,
            category=PatternCategory.CODE_SMELL,
            description="Function with too many parameters",
            severity=AnalysisSeverity.INFO,
            structural_rules={
                "max_params": 5,
            },
        ),
        CodePattern(
            id="magic_numbers",
            name="Magic Numbers",
            pattern_type=PatternType.MAGIC_NUMBERS,
            category=PatternCategory.CODE_SMELL,
            description="Unexplained numeric literals in code",
            severity=AnalysisSeverity.INFO,
            regex_patterns=[
                r"(?<![a-zA-Z0-9_])[2-9]\d{2,}(?![a-zA-Z0-9_])",  # Large numbers
                r"\[\s*\d+\s*:\s*\d+\s*\]",  # Slice with magic numbers
            ],
        ),
        CodePattern(
            id="global_state",
            name="Global State",
            pattern_type=PatternType.GLOBAL_STATE,
            category=PatternCategory.ANTI_PATTERN,
            description="Use of global variables",
            severity=AnalysisSeverity.WARNING,
            regex_patterns=[
                r"^\s*global\s+\w+",
                r"^[A-Z_][A-Z0-9_]*\s*=\s*(?!.*frozen)",  # Module-level mutable state
            ],
        ),
        CodePattern(
            id="bare_except",
            name="Bare Except",
            pattern_type=PatternType.GLOBAL_STATE,  # Reusing for simplicity
            category=PatternCategory.ANTI_PATTERN,
            description="Catching all exceptions without specificity",
            severity=AnalysisSeverity.WARNING,
            regex_patterns=[
                r"except\s*:",
                r"except\s+Exception\s*:",
            ],
        ),
    ]

    def __init__(self, patterns: list[CodePattern] | None = None) -> None:
        """Initialize the pattern detector.

        Args:
            patterns: List of patterns to detect. Uses defaults if not provided.
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._logger = logger.bind(component="pattern_detector")

    def detect_patterns(
        self,
        source_code: str,
        file_path: str,
        entities: list[dict[str, Any]] | None = None,
    ) -> list[PatternMatch]:
        """Detect patterns in source code.

        Args:
            source_code: Source code to analyze.
            file_path: Path to the file.
            entities: Parsed code entities with metadata.

        Returns:
            List of pattern matches found.
        """
        matches: list[PatternMatch] = []

        # Detect regex-based patterns
        for pattern in self.patterns:
            if pattern.regex_patterns:
                regex_matches = self._detect_regex_patterns(pattern, source_code, file_path)
                matches.extend(regex_matches)

        # Detect structural patterns using entities
        if entities:
            for pattern in self.patterns:
                if pattern.structural_rules:
                    structural_matches = self._detect_structural_patterns(
                        pattern, entities, file_path
                    )
                    matches.extend(structural_matches)

        return matches

    def _detect_regex_patterns(
        self,
        pattern: CodePattern,
        source_code: str,
        file_path: str,
    ) -> list[PatternMatch]:
        """Detect patterns using regex matching.

        Args:
            pattern: Pattern to detect.
            source_code: Source code to search.
            file_path: Path to the file.

        Returns:
            List of matches found.
        """
        matches: list[PatternMatch] = []
        lines = source_code.split("\n")

        for regex in pattern.regex_patterns:
            try:
                compiled = re.compile(regex, re.MULTILINE)

                for i, line in enumerate(lines):
                    if compiled.search(line):
                        matches.append(
                            PatternMatch(
                                pattern_id=pattern.id,
                                pattern_name=pattern.name,
                                pattern_type=pattern.pattern_type,
                                category=pattern.category,
                                file_path=file_path,
                                start_line=i + 1,
                                end_line=i + 1,
                                matched_code=line.strip()[:100],
                                confidence=0.8,
                            )
                        )
            except re.error as e:
                self._logger.warning(
                    "invalid_regex_pattern",
                    pattern_id=pattern.id,
                    regex=regex,
                    error=str(e),
                )

        return matches

    def _detect_structural_patterns(
        self,
        pattern: CodePattern,
        entities: list[dict[str, Any]],
        file_path: str,
    ) -> list[PatternMatch]:
        """Detect patterns using structural analysis.

        Args:
            pattern: Pattern to detect.
            entities: Parsed code entities.
            file_path: Path to the file.

        Returns:
            List of matches found.
        """
        matches: list[PatternMatch] = []
        rules = pattern.structural_rules

        for entity in entities:
            entity_type = entity.get("type", "")
            entity_name = entity.get("name", "")

            # Check God Class pattern
            if pattern.pattern_type == PatternType.GOD_CLASS and entity_type == "class":
                method_count = len(entity.get("methods", []))
                line_count = entity.get("end_line", 0) - entity.get("start_line", 0)

                if method_count >= rules.get("min_methods", 15) or line_count >= rules.get(
                    "min_lines", 500
                ):
                    matches.append(
                        PatternMatch(
                            pattern_id=pattern.id,
                            pattern_name=pattern.name,
                            pattern_type=pattern.pattern_type,
                            category=pattern.category,
                            file_path=file_path,
                            start_line=entity.get("start_line", 0),
                            end_line=entity.get("end_line", 0),
                            entity_name=entity_name,
                            entity_type=entity_type,
                            confidence=0.9,
                            context={
                                "method_count": method_count,
                                "line_count": line_count,
                            },
                        )
                    )

            # Check Long Method pattern
            if pattern.pattern_type == PatternType.LONG_METHOD and entity_type in (
                "function",
                "method",
            ):
                line_count = entity.get("end_line", 0) - entity.get("start_line", 0)
                max_lines = rules.get("max_lines", 50)

                if line_count > max_lines:
                    matches.append(
                        PatternMatch(
                            pattern_id=pattern.id,
                            pattern_name=pattern.name,
                            pattern_type=pattern.pattern_type,
                            category=pattern.category,
                            file_path=file_path,
                            start_line=entity.get("start_line", 0),
                            end_line=entity.get("end_line", 0),
                            entity_name=entity_name,
                            entity_type=entity_type,
                            confidence=0.9,
                            context={
                                "line_count": line_count,
                                "max_allowed": max_lines,
                            },
                        )
                    )

            # Check Long Parameter List pattern
            if pattern.pattern_type == PatternType.LONG_PARAMETER_LIST and entity_type in (
                "function",
                "method",
            ):
                param_count = len(entity.get("parameters", []))
                max_params = rules.get("max_params", 5)

                if param_count > max_params:
                    matches.append(
                        PatternMatch(
                            pattern_id=pattern.id,
                            pattern_name=pattern.name,
                            pattern_type=pattern.pattern_type,
                            category=pattern.category,
                            file_path=file_path,
                            start_line=entity.get("start_line", 0),
                            end_line=entity.get("end_line", 0),
                            entity_name=entity_name,
                            entity_type=entity_type,
                            confidence=0.9,
                            context={
                                "param_count": param_count,
                                "max_allowed": max_params,
                            },
                        )
                    )

        return matches

    def detect_singleton(
        self,
        source_code: str,
        file_path: str,
    ) -> list[PatternMatch]:
        """Detect singleton pattern implementations.

        Args:
            source_code: Source code to analyze.
            file_path: Path to the file.

        Returns:
            List of singleton pattern matches.
        """
        matches: list[PatternMatch] = []

        # Look for common singleton indicators
        singleton_indicators = [
            r"_instance\s*[:=]",
            r"__new__\s*\(",
            r"@classmethod\s*\n\s*def\s+(?:get_)?instance",
            r"cls\._instance",
        ]

        lines = source_code.split("\n")
        matched_lines: set[int] = set()

        for indicator in singleton_indicators:
            for i, line in enumerate(lines):
                if re.search(indicator, line) and i not in matched_lines:
                    matched_lines.add(i)

        if len(matched_lines) >= 2:
            min_line = min(matched_lines)
            max_line = max(matched_lines)
            matches.append(
                PatternMatch(
                    pattern_id="singleton",
                    pattern_name="Singleton Pattern",
                    pattern_type=PatternType.SINGLETON,
                    category=PatternCategory.DESIGN_PATTERN,
                    file_path=file_path,
                    start_line=min_line + 1,
                    end_line=max_line + 1,
                    confidence=0.85,
                    context={"indicator_count": len(matched_lines)},
                )
            )

        return matches

    def detect_factory(
        self,
        source_code: str,
        file_path: str,
    ) -> list[PatternMatch]:
        """Detect factory pattern implementations.

        Args:
            source_code: Source code to analyze.
            file_path: Path to the file.

        Returns:
            List of factory pattern matches.
        """
        matches: list[PatternMatch] = []

        # Look for factory indicators
        factory_patterns = [
            (r"class\s+(\w*Factory\w*)\s*[:\(]", "Factory class"),
            (r"def\s+(create_\w+)\s*\(", "Factory method"),
            (r"def\s+(make_\w+)\s*\(", "Factory method"),
            (r"def\s+(build_\w+)\s*\(", "Builder method"),
        ]

        lines = source_code.split("\n")

        for pattern, description in factory_patterns:
            for i, line in enumerate(lines):
                match = re.search(pattern, line)
                if match:
                    matches.append(
                        PatternMatch(
                            pattern_id="factory",
                            pattern_name="Factory Pattern",
                            pattern_type=PatternType.FACTORY,
                            category=PatternCategory.DESIGN_PATTERN,
                            file_path=file_path,
                            start_line=i + 1,
                            end_line=i + 1,
                            entity_name=match.group(1),
                            confidence=0.9,
                            context={"description": description},
                        )
                    )

        return matches

    def detect_decorator_pattern(
        self,
        source_code: str,
        file_path: str,
    ) -> list[PatternMatch]:
        """Detect decorator pattern (not Python decorators).

        Args:
            source_code: Source code to analyze.
            file_path: Path to the file.

        Returns:
            List of decorator pattern matches.
        """
        matches: list[PatternMatch] = []

        # Look for decorator pattern indicators
        # - Class with a component attribute
        # - Method that delegates to component

        decorator_indicators = [
            r"self\._component",
            r"self\.component",
            r"self\._wrapped",
            r"self\.wrapped",
        ]

        lines = source_code.split("\n")
        found_indicators: list[int] = []

        for indicator in decorator_indicators:
            for i, line in enumerate(lines):
                if re.search(indicator, line):
                    found_indicators.append(i)

        if len(found_indicators) >= 2:
            matches.append(
                PatternMatch(
                    pattern_id="decorator",
                    pattern_name="Decorator Pattern",
                    pattern_type=PatternType.DECORATOR,
                    category=PatternCategory.DESIGN_PATTERN,
                    file_path=file_path,
                    start_line=min(found_indicators) + 1,
                    end_line=max(found_indicators) + 1,
                    confidence=0.7,
                )
            )

        return matches

    def generate_findings(
        self,
        matches: list[PatternMatch],
    ) -> list[Finding]:
        """Generate findings from pattern matches.

        Only generates findings for anti-patterns and code smells.

        Args:
            matches: List of pattern matches.

        Returns:
            List of findings for problem patterns.
        """
        findings: list[Finding] = []

        for match in matches:
            # Only create findings for problems, not design patterns
            if match.category in (
                PatternCategory.ANTI_PATTERN,
                PatternCategory.CODE_SMELL,
            ):
                # Find the corresponding pattern for severity
                pattern = next(
                    (p for p in self.patterns if p.id == match.pattern_id),
                    None,
                )
                severity = pattern.severity if pattern else AnalysisSeverity.INFO

                suggestions = {
                    PatternType.GOD_CLASS: "Consider splitting into smaller, focused classes",
                    PatternType.LONG_METHOD: "Extract functionality into smaller methods",
                    PatternType.LONG_PARAMETER_LIST: "Use a configuration object or builder",
                    PatternType.MAGIC_NUMBERS: "Replace with named constants",
                    PatternType.GLOBAL_STATE: "Encapsulate state in classes",
                }

                findings.append(
                    Finding(
                        id=f"{match.file_path}:{match.start_line}:{match.pattern_id}",
                        category=AnalysisCategory.PATTERN,
                        severity=severity,
                        message=f"Detected: {match.pattern_name}",
                        file_path=match.file_path,
                        start_line=match.start_line,
                        end_line=match.end_line,
                        entity_name=match.entity_name,
                        entity_type=match.entity_type,
                        suggestion=suggestions.get(match.pattern_type),
                        metadata={
                            "pattern_type": match.pattern_type.value,
                            "confidence": match.confidence,
                            **match.context,
                        },
                    )
                )

        return findings
