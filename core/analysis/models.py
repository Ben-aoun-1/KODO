"""Data models for code analysis.

This module defines the Pydantic models used throughout the analysis module
for configuration, results, and intermediate data structures.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisSeverity(str, Enum):
    """Severity levels for analysis findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnalysisCategory(str, Enum):
    """Categories of analysis findings."""

    COMPLEXITY = "complexity"
    COUPLING = "coupling"
    COHESION = "cohesion"
    DATAFLOW = "dataflow"
    SECURITY = "security"
    PATTERN = "pattern"
    STYLE = "style"
    PERFORMANCE = "performance"


class AnalysisConfig(BaseModel):
    """Configuration for code analysis.

    Attributes:
        max_complexity: Maximum cyclomatic complexity threshold.
        max_coupling: Maximum coupling score threshold.
        min_cohesion: Minimum cohesion score threshold.
        include_metrics: Whether to include metrics in results.
        include_dataflow: Whether to include data flow analysis.
        include_patterns: Whether to include pattern detection.
        max_depth: Maximum depth for traversal operations.
    """

    model_config = ConfigDict(frozen=True)

    max_complexity: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum cyclomatic complexity threshold",
    )
    max_coupling: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum coupling score threshold",
    )
    min_cohesion: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cohesion score threshold",
    )
    include_metrics: bool = Field(
        default=True,
        description="Include metrics in analysis",
    )
    include_dataflow: bool = Field(
        default=True,
        description="Include data flow analysis",
    )
    include_patterns: bool = Field(
        default=True,
        description="Include pattern detection",
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum traversal depth",
    )


class Finding(BaseModel):
    """A single analysis finding.

    Attributes:
        id: Unique identifier for the finding.
        category: Category of the finding.
        severity: Severity level.
        message: Human-readable description.
        file_path: Path to the affected file.
        start_line: Starting line number.
        end_line: Ending line number.
        entity_name: Name of the affected entity.
        entity_type: Type of the affected entity.
        suggestion: Suggested fix or improvement.
        metadata: Additional finding-specific data.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Finding ID")
    category: AnalysisCategory = Field(..., description="Finding category")
    severity: AnalysisSeverity = Field(..., description="Severity level")
    message: str = Field(..., description="Finding description")
    file_path: str = Field(..., description="Affected file path")
    start_line: int = Field(default=0, ge=0, description="Start line")
    end_line: int = Field(default=0, ge=0, description="End line")
    entity_name: str | None = Field(None, description="Affected entity name")
    entity_type: str | None = Field(None, description="Affected entity type")
    suggestion: str | None = Field(None, description="Suggested fix")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra data")


class AnalysisSummary(BaseModel):
    """Summary statistics for an analysis run.

    Attributes:
        total_files: Number of files analyzed.
        total_entities: Number of code entities analyzed.
        total_findings: Total number of findings.
        findings_by_severity: Count of findings by severity.
        findings_by_category: Count of findings by category.
        avg_complexity: Average cyclomatic complexity.
        avg_coupling: Average coupling score.
        avg_cohesion: Average cohesion score.
    """

    model_config = ConfigDict(frozen=True)

    total_files: int = Field(default=0, ge=0, description="Files analyzed")
    total_entities: int = Field(default=0, ge=0, description="Entities analyzed")
    total_findings: int = Field(default=0, ge=0, description="Total findings")
    findings_by_severity: dict[str, int] = Field(
        default_factory=dict, description="Findings by severity"
    )
    findings_by_category: dict[str, int] = Field(
        default_factory=dict, description="Findings by category"
    )
    avg_complexity: float = Field(default=0.0, ge=0.0, description="Avg complexity")
    avg_coupling: float = Field(default=0.0, ge=0.0, le=1.0, description="Avg coupling")
    avg_cohesion: float = Field(default=0.0, ge=0.0, le=1.0, description="Avg cohesion")


class AnalysisResult(BaseModel):
    """Complete result of a code analysis run.

    Attributes:
        repo_id: Repository identifier.
        analysis_id: Unique analysis run ID.
        timestamp: When the analysis was performed.
        config: Configuration used for analysis.
        summary: Summary statistics.
        findings: List of all findings.
        duration_ms: Analysis duration in milliseconds.
        success: Whether the analysis completed successfully.
        error: Error message if analysis failed.
    """

    model_config = ConfigDict(frozen=True)

    repo_id: str = Field(..., description="Repository ID")
    analysis_id: str = Field(..., description="Analysis run ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Analysis timestamp",
    )
    config: AnalysisConfig = Field(
        default_factory=AnalysisConfig,
        description="Analysis configuration",
    )
    summary: AnalysisSummary = Field(
        default_factory=AnalysisSummary,
        description="Analysis summary",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Analysis findings",
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Duration in ms")
    success: bool = Field(default=True, description="Success status")
    error: str | None = Field(None, description="Error message if failed")

    def get_critical_findings(self) -> list[Finding]:
        """Get all critical severity findings."""
        return [f for f in self.findings if f.severity == AnalysisSeverity.CRITICAL]

    def get_findings_by_file(self, file_path: str) -> list[Finding]:
        """Get all findings for a specific file."""
        return [f for f in self.findings if f.file_path == file_path]

    def get_findings_by_category(self, category: AnalysisCategory) -> list[Finding]:
        """Get all findings for a specific category."""
        return [f for f in self.findings if f.category == category]


class EntityMetrics(BaseModel):
    """Metrics for a single code entity.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Name of the entity.
        entity_type: Type (function, class, method, etc.).
        file_path: Path to the containing file.
        start_line: Starting line number.
        end_line: Ending line number.
        lines_of_code: Lines of code.
        cyclomatic_complexity: Cyclomatic complexity score.
        cognitive_complexity: Cognitive complexity score.
        coupling_score: Coupling to other entities.
        cohesion_score: Internal cohesion.
        dependency_count: Number of dependencies.
        dependent_count: Number of dependents.
        parameters: Number of parameters (for functions).
        nesting_depth: Maximum nesting depth.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(default=0, ge=0, description="Start line")
    end_line: int = Field(default=0, ge=0, description="End line")
    lines_of_code: int = Field(default=0, ge=0, description="LOC")
    cyclomatic_complexity: int = Field(default=1, ge=1, description="Cyclomatic complexity")
    cognitive_complexity: int = Field(default=0, ge=0, description="Cognitive complexity")
    coupling_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Coupling")
    cohesion_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Cohesion")
    dependency_count: int = Field(default=0, ge=0, description="Dependencies")
    dependent_count: int = Field(default=0, ge=0, description="Dependents")
    parameters: int = Field(default=0, ge=0, description="Parameter count")
    nesting_depth: int = Field(default=0, ge=0, description="Max nesting depth")


class FileMetrics(BaseModel):
    """Aggregated metrics for a file.

    Attributes:
        file_path: Path to the file.
        language: Programming language.
        lines_of_code: Total lines of code.
        lines_of_comments: Lines of comments.
        blank_lines: Number of blank lines.
        entity_count: Number of entities in file.
        avg_complexity: Average complexity of entities.
        max_complexity: Maximum complexity in file.
        import_count: Number of imports.
        export_count: Number of exports/public items.
    """

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(..., description="File path")
    language: str = Field(default="unknown", description="Language")
    lines_of_code: int = Field(default=0, ge=0, description="LOC")
    lines_of_comments: int = Field(default=0, ge=0, description="Comment lines")
    blank_lines: int = Field(default=0, ge=0, description="Blank lines")
    entity_count: int = Field(default=0, ge=0, description="Entity count")
    avg_complexity: float = Field(default=0.0, ge=0.0, description="Avg complexity")
    max_complexity: int = Field(default=0, ge=0, description="Max complexity")
    import_count: int = Field(default=0, ge=0, description="Import count")
    export_count: int = Field(default=0, ge=0, description="Export count")
