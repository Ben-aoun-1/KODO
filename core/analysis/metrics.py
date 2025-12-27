"""Code metrics calculation.

This module provides functionality for calculating various code metrics
including cyclomatic complexity, cognitive complexity, coupling, and cohesion.
"""

import re

import structlog
from pydantic import BaseModel, ConfigDict, Field

from core.analysis.models import (
    AnalysisCategory,
    AnalysisConfig,
    AnalysisSeverity,
    EntityMetrics,
    FileMetrics,
    Finding,
)

logger = structlog.get_logger(__name__)


class ComplexityMetrics(BaseModel):
    """Complexity metrics for a code entity.

    Attributes:
        cyclomatic: Cyclomatic complexity (decision points + 1).
        cognitive: Cognitive complexity (mental effort to understand).
        nesting_depth: Maximum nesting depth.
        lines_of_code: Lines of code (excluding blanks/comments).
    """

    model_config = ConfigDict(frozen=True)

    cyclomatic: int = Field(default=1, ge=1, description="Cyclomatic complexity")
    cognitive: int = Field(default=0, ge=0, description="Cognitive complexity")
    nesting_depth: int = Field(default=0, ge=0, description="Max nesting depth")
    lines_of_code: int = Field(default=0, ge=0, description="Lines of code")


class CouplingMetrics(BaseModel):
    """Coupling metrics for a code entity.

    Attributes:
        afferent: Number of entities that depend on this one.
        efferent: Number of entities this one depends on.
        instability: Ratio of efferent to total (0=stable, 1=unstable).
        abstractness: Ratio of abstract to concrete elements.
    """

    model_config = ConfigDict(frozen=True)

    afferent: int = Field(default=0, ge=0, description="Afferent coupling")
    efferent: int = Field(default=0, ge=0, description="Efferent coupling")
    instability: float = Field(default=0.0, ge=0.0, le=1.0, description="Instability")
    abstractness: float = Field(default=0.0, ge=0.0, le=1.0, description="Abstractness")

    @property
    def distance_from_main_sequence(self) -> float:
        """Calculate distance from the main sequence (A + I = 1)."""
        return abs(self.abstractness + self.instability - 1.0)


class CodeMetrics(BaseModel):
    """Complete metrics for a code entity or file.

    Attributes:
        complexity: Complexity metrics.
        coupling: Coupling metrics.
        cohesion: Cohesion score (0-1).
        maintainability_index: Maintainability score (0-100).
    """

    model_config = ConfigDict(frozen=True)

    complexity: ComplexityMetrics = Field(
        default_factory=ComplexityMetrics,
        description="Complexity metrics",
    )
    coupling: CouplingMetrics = Field(
        default_factory=CouplingMetrics,
        description="Coupling metrics",
    )
    cohesion: float = Field(default=1.0, ge=0.0, le=1.0, description="Cohesion score")
    maintainability_index: float = Field(
        default=100.0, ge=0.0, le=100.0, description="Maintainability index"
    )


class MetricsCalculator:
    """Calculator for code metrics.

    This class provides methods for calculating various software metrics
    from source code and parsed AST data.
    """

    # Python control flow keywords that increase cyclomatic complexity
    PYTHON_DECISION_KEYWORDS = {
        "if",
        "elif",
        "for",
        "while",
        "except",
        "with",
        "assert",
        "and",
        "or",
    }

    # Python nesting keywords
    PYTHON_NESTING_KEYWORDS = {
        "if",
        "elif",
        "else",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "def",
        "class",
        "async",
    }

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize the metrics calculator.

        Args:
            config: Analysis configuration.
        """
        self.config = config or AnalysisConfig()
        self._logger = logger.bind(component="metrics_calculator")

    def calculate_cyclomatic_complexity(self, source_code: str) -> int:
        """Calculate cyclomatic complexity from source code.

        Cyclomatic complexity = E - N + 2P, where:
        - E = number of edges
        - N = number of nodes
        - P = number of connected components

        For a single function, this simplifies to: decisions + 1

        Args:
            source_code: The source code to analyze.

        Returns:
            Cyclomatic complexity score (minimum 1).
        """
        complexity = 1  # Base complexity

        # Count decision points
        for keyword in self.PYTHON_DECISION_KEYWORDS:
            # Match whole words only
            pattern = rf"\b{keyword}\b"
            complexity += len(re.findall(pattern, source_code))

        # Note: Ternary expressions (x if cond else y) are already counted
        # via the 'if' keyword, so no additional adjustment needed

        # Count list/dict/set comprehension conditions
        comp_pattern = r"\bfor\b.*\bif\b"
        complexity += len(re.findall(comp_pattern, source_code))

        return max(1, complexity)

    def calculate_cognitive_complexity(self, source_code: str) -> int:
        """Calculate cognitive complexity from source code.

        Cognitive complexity measures the mental effort required to
        understand code, penalizing:
        - Nesting depth
        - Breaks in linear flow
        - Recursion

        Args:
            source_code: The source code to analyze.

        Returns:
            Cognitive complexity score.
        """
        complexity = 0
        nesting_level = 0
        lines = source_code.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Detect nesting increase
            for keyword in self.PYTHON_NESTING_KEYWORDS:
                if re.match(rf"^{keyword}\b", stripped):
                    # Add 1 for the structure, plus nesting penalty
                    complexity += 1 + nesting_level

                    if keyword in {"if", "for", "while", "try", "with"}:
                        nesting_level += 1
                    break

            # Boolean operators add complexity
            complexity += stripped.count(" and ")
            complexity += stripped.count(" or ")

            # Recursion detection (simple heuristic)
            if "self." in stripped and "(" in stripped:
                # Could be recursive call
                pass

        return complexity

    def calculate_nesting_depth(self, source_code: str) -> int:
        """Calculate maximum nesting depth.

        Args:
            source_code: The source code to analyze.

        Returns:
            Maximum nesting depth.
        """
        max_depth = 0
        current_depth = 0
        lines = source_code.split("\n")
        indent_stack: list[int] = [0]

        for line in lines:
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())

            # Track indentation changes
            while indent_stack and indent < indent_stack[-1]:
                indent_stack.pop()
                current_depth = max(0, current_depth - 1)

            # Check if this line starts a new block
            stripped = line.strip()
            for keyword in self.PYTHON_NESTING_KEYWORDS:
                if re.match(rf"^{keyword}\b", stripped) and stripped.endswith(":"):
                    current_depth += 1
                    indent_stack.append(indent + 4)  # Assume 4-space indent
                    break

            max_depth = max(max_depth, current_depth)

        return max_depth

    def calculate_lines_of_code(self, source_code: str) -> tuple[int, int, int]:
        """Calculate lines of code metrics.

        Args:
            source_code: The source code to analyze.

        Returns:
            Tuple of (code_lines, comment_lines, blank_lines).
        """
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        in_multiline_string = False

        lines = source_code.split("\n")

        for line in lines:
            stripped = line.strip()

            if not stripped:
                blank_lines += 1
                continue

            # Handle multiline strings (docstrings)
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                count = stripped.count(quote)
                if count == 1:
                    in_multiline_string = not in_multiline_string
                comment_lines += 1
                continue

            if in_multiline_string:
                comment_lines += 1
                continue

            # Single-line comments
            if stripped.startswith("#"):
                comment_lines += 1
                continue

            # Code line (may have inline comment)
            code_lines += 1

        return code_lines, comment_lines, blank_lines

    def calculate_complexity_metrics(self, source_code: str) -> ComplexityMetrics:
        """Calculate all complexity metrics for source code.

        Args:
            source_code: The source code to analyze.

        Returns:
            ComplexityMetrics object.
        """
        code_lines, _, _ = self.calculate_lines_of_code(source_code)

        return ComplexityMetrics(
            cyclomatic=self.calculate_cyclomatic_complexity(source_code),
            cognitive=self.calculate_cognitive_complexity(source_code),
            nesting_depth=self.calculate_nesting_depth(source_code),
            lines_of_code=code_lines,
        )

    def calculate_coupling(
        self,
        dependencies: list[str],
        dependents: list[str],
        is_abstract: bool = False,
    ) -> CouplingMetrics:
        """Calculate coupling metrics.

        Args:
            dependencies: List of entities this one depends on.
            dependents: List of entities that depend on this one.
            is_abstract: Whether this is an abstract entity.

        Returns:
            CouplingMetrics object.
        """
        afferent = len(dependents)
        efferent = len(dependencies)
        total = afferent + efferent

        instability = efferent / total if total > 0 else 0.0
        abstractness = 1.0 if is_abstract else 0.0

        return CouplingMetrics(
            afferent=afferent,
            efferent=efferent,
            instability=instability,
            abstractness=abstractness,
        )

    def calculate_cohesion(
        self,
        methods: list[str],
        shared_attributes: dict[str, list[str]],
    ) -> float:
        """Calculate cohesion score for a class.

        Uses LCOM (Lack of Cohesion of Methods) variant:
        Higher score = more cohesive (methods share attributes).

        Args:
            methods: List of method names.
            shared_attributes: Dict mapping method names to used attributes.

        Returns:
            Cohesion score (0-1, higher is better).
        """
        if len(methods) <= 1:
            return 1.0

        # Count pairs of methods that share attributes
        sharing_pairs = 0
        total_pairs = 0

        for i, m1 in enumerate(methods):
            attrs1 = set(shared_attributes.get(m1, []))
            for m2 in methods[i + 1 :]:
                attrs2 = set(shared_attributes.get(m2, []))
                total_pairs += 1
                if attrs1 & attrs2:  # Intersection
                    sharing_pairs += 1

        if total_pairs == 0:
            return 1.0

        return sharing_pairs / total_pairs

    def calculate_maintainability_index(
        self,
        cyclomatic: int,
        lines_of_code: int,
        comment_ratio: float,
    ) -> float:
        """Calculate maintainability index.

        Based on the Microsoft Visual Studio formula:
        MI = MAX(0, (171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(L)) * 100 / 171)

        Simplified version using:
        - V (Halstead Volume) approximated by LOC
        - G (Cyclomatic Complexity)
        - L (Lines of Code)

        Args:
            cyclomatic: Cyclomatic complexity.
            lines_of_code: Lines of code.
            comment_ratio: Ratio of comments to code.

        Returns:
            Maintainability index (0-100).
        """
        import math

        if lines_of_code <= 0:
            return 100.0

        # Simplified MI calculation
        volume = math.log(max(1, lines_of_code))
        complexity_factor = cyclomatic * 0.23
        loc_factor = math.log(max(1, lines_of_code)) * 16.2

        mi = 171 - 5.2 * volume - complexity_factor - loc_factor

        # Bonus for comments
        mi += comment_ratio * 10

        # Normalize to 0-100
        mi = (mi * 100) / 171

        return max(0.0, min(100.0, mi))

    def calculate_entity_metrics(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str,
        file_path: str,
        source_code: str,
        start_line: int = 0,
        end_line: int = 0,
        dependencies: list[str] | None = None,
        dependents: list[str] | None = None,
        parameters: int = 0,
    ) -> EntityMetrics:
        """Calculate all metrics for a code entity.

        Args:
            entity_id: Unique entity identifier.
            entity_name: Name of the entity.
            entity_type: Type (function, class, method).
            file_path: Path to the containing file.
            source_code: Source code of the entity.
            start_line: Starting line number.
            end_line: Ending line number.
            dependencies: List of dependency names.
            dependents: List of dependent names.
            parameters: Number of function parameters.

        Returns:
            EntityMetrics object.
        """
        complexity = self.calculate_complexity_metrics(source_code)
        coupling = self.calculate_coupling(dependencies or [], dependents or [])

        return EntityMetrics(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            lines_of_code=complexity.lines_of_code,
            cyclomatic_complexity=complexity.cyclomatic,
            cognitive_complexity=complexity.cognitive,
            coupling_score=coupling.instability,
            cohesion_score=1.0,  # Calculated separately for classes
            dependency_count=coupling.efferent,
            dependent_count=coupling.afferent,
            parameters=parameters,
            nesting_depth=complexity.nesting_depth,
        )

    def calculate_file_metrics(
        self,
        file_path: str,
        source_code: str,
        language: str = "python",
        entities: list[EntityMetrics] | None = None,
    ) -> FileMetrics:
        """Calculate metrics for a file.

        Args:
            file_path: Path to the file.
            source_code: Full source code of the file.
            language: Programming language.
            entities: Pre-calculated entity metrics.

        Returns:
            FileMetrics object.
        """
        code_lines, comment_lines, blank_lines = self.calculate_lines_of_code(source_code)

        # Calculate import count
        import_pattern = r"^(?:from\s+\S+\s+)?import\s+"
        import_count = len(re.findall(import_pattern, source_code, re.MULTILINE))

        # Entity statistics
        entity_count = len(entities) if entities else 0
        avg_complexity = 0.0
        max_complexity = 0

        if entities:
            complexities = [e.cyclomatic_complexity for e in entities]
            avg_complexity = sum(complexities) / len(complexities)
            max_complexity = max(complexities)

        return FileMetrics(
            file_path=file_path,
            language=language,
            lines_of_code=code_lines,
            lines_of_comments=comment_lines,
            blank_lines=blank_lines,
            entity_count=entity_count,
            avg_complexity=avg_complexity,
            max_complexity=max_complexity,
            import_count=import_count,
            export_count=0,  # Would need AST analysis
        )

    def generate_findings(
        self,
        entity_metrics: EntityMetrics,
    ) -> list[Finding]:
        """Generate findings based on metrics thresholds.

        Args:
            entity_metrics: Metrics for the entity to check.

        Returns:
            List of findings for threshold violations.
        """
        findings: list[Finding] = []

        # Check cyclomatic complexity
        if entity_metrics.cyclomatic_complexity > self.config.max_complexity:
            findings.append(
                Finding(
                    id=f"{entity_metrics.entity_id}:complexity",
                    category=AnalysisCategory.COMPLEXITY,
                    severity=AnalysisSeverity.WARNING
                    if entity_metrics.cyclomatic_complexity <= self.config.max_complexity * 1.5
                    else AnalysisSeverity.ERROR,
                    message=f"High cyclomatic complexity: {entity_metrics.cyclomatic_complexity} "
                    f"(threshold: {self.config.max_complexity})",
                    file_path=entity_metrics.file_path,
                    start_line=entity_metrics.start_line,
                    end_line=entity_metrics.end_line,
                    entity_name=entity_metrics.entity_name,
                    entity_type=entity_metrics.entity_type,
                    suggestion="Consider breaking this into smaller functions.",
                    metadata={
                        "actual": entity_metrics.cyclomatic_complexity,
                        "threshold": self.config.max_complexity,
                    },
                )
            )

        # Check coupling
        if entity_metrics.coupling_score > self.config.max_coupling:
            findings.append(
                Finding(
                    id=f"{entity_metrics.entity_id}:coupling",
                    category=AnalysisCategory.COUPLING,
                    severity=AnalysisSeverity.WARNING,
                    message=f"High coupling score: {entity_metrics.coupling_score:.2f} "
                    f"(threshold: {self.config.max_coupling})",
                    file_path=entity_metrics.file_path,
                    start_line=entity_metrics.start_line,
                    end_line=entity_metrics.end_line,
                    entity_name=entity_metrics.entity_name,
                    entity_type=entity_metrics.entity_type,
                    suggestion="Consider reducing dependencies through abstraction.",
                    metadata={
                        "actual": entity_metrics.coupling_score,
                        "threshold": self.config.max_coupling,
                    },
                )
            )

        # Check cohesion
        if entity_metrics.cohesion_score < self.config.min_cohesion:
            findings.append(
                Finding(
                    id=f"{entity_metrics.entity_id}:cohesion",
                    category=AnalysisCategory.COHESION,
                    severity=AnalysisSeverity.INFO,
                    message=f"Low cohesion score: {entity_metrics.cohesion_score:.2f} "
                    f"(threshold: {self.config.min_cohesion})",
                    file_path=entity_metrics.file_path,
                    start_line=entity_metrics.start_line,
                    end_line=entity_metrics.end_line,
                    entity_name=entity_metrics.entity_name,
                    entity_type=entity_metrics.entity_type,
                    suggestion="Consider splitting this class into more focused classes.",
                    metadata={
                        "actual": entity_metrics.cohesion_score,
                        "threshold": self.config.min_cohesion,
                    },
                )
            )

        # Check nesting depth
        if entity_metrics.nesting_depth > 4:
            findings.append(
                Finding(
                    id=f"{entity_metrics.entity_id}:nesting",
                    category=AnalysisCategory.COMPLEXITY,
                    severity=AnalysisSeverity.WARNING
                    if entity_metrics.nesting_depth <= 6
                    else AnalysisSeverity.ERROR,
                    message=f"Deep nesting: {entity_metrics.nesting_depth} levels",
                    file_path=entity_metrics.file_path,
                    start_line=entity_metrics.start_line,
                    end_line=entity_metrics.end_line,
                    entity_name=entity_metrics.entity_name,
                    entity_type=entity_metrics.entity_type,
                    suggestion="Consider extracting nested logic into separate functions.",
                    metadata={"nesting_depth": entity_metrics.nesting_depth},
                )
            )

        # Check parameter count
        if entity_metrics.parameters > 5:
            findings.append(
                Finding(
                    id=f"{entity_metrics.entity_id}:parameters",
                    category=AnalysisCategory.COMPLEXITY,
                    severity=AnalysisSeverity.INFO,
                    message=f"Many parameters: {entity_metrics.parameters}",
                    file_path=entity_metrics.file_path,
                    start_line=entity_metrics.start_line,
                    end_line=entity_metrics.end_line,
                    entity_name=entity_metrics.entity_name,
                    entity_type=entity_metrics.entity_type,
                    suggestion="Consider using a configuration object or builder pattern.",
                    metadata={"parameter_count": entity_metrics.parameters},
                )
            )

        return findings
