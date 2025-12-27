"""Impact analysis for code changes.

This module provides tools for analyzing the potential impact of code changes,
identifying affected entities, and assessing risk levels.
"""

from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class ImpactLevel(str, Enum):
    """Impact severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChangeType(str, Enum):
    """Types of code changes."""

    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"


class AffectedEntity(BaseModel):
    """An entity affected by a code change.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Name of the entity.
        entity_type: Type (function, class, method, etc.).
        file_path: Path to the containing file.
        start_line: Starting line number.
        end_line: Ending line number.
        impact_level: Severity of impact.
        impact_reason: Why this entity is affected.
        distance: Graph distance from the changed entity.
        is_direct: Whether this is a direct dependency.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(default=0, ge=0, description="Start line")
    end_line: int = Field(default=0, ge=0, description="End line")
    impact_level: ImpactLevel = Field(default=ImpactLevel.LOW, description="Impact level")
    impact_reason: str = Field(default="", description="Impact reason")
    distance: int = Field(default=0, ge=0, description="Distance from change")
    is_direct: bool = Field(default=False, description="Direct dependency")


class PropagationPath(BaseModel):
    """A path of impact propagation through the codebase.

    Attributes:
        entities: List of entity IDs in the propagation chain.
        impact_type: Type of impact propagation.
        description: Human-readable description.
    """

    model_config = ConfigDict(frozen=True)

    entities: list[str] = Field(default_factory=list, description="Entity chain")
    impact_type: str = Field(default="call", description="Propagation type")
    description: str = Field(default="", description="Description")


class ImpactReport(BaseModel):
    """Complete impact analysis report.

    Attributes:
        changed_entity_id: ID of the entity being changed.
        changed_entity_name: Name of the entity being changed.
        change_type: Type of change being made.
        overall_impact: Overall impact level.
        affected_entities: List of affected entities.
        affected_files: Set of affected file paths.
        propagation_paths: Paths of impact propagation.
        risk_score: Numerical risk score (0-100).
        recommendations: Suggested actions.
        requires_review: Whether human review is recommended.
    """

    model_config = ConfigDict(frozen=True)

    changed_entity_id: str = Field(..., description="Changed entity ID")
    changed_entity_name: str = Field(..., description="Changed entity name")
    change_type: ChangeType = Field(default=ChangeType.MODIFY, description="Change type")
    overall_impact: ImpactLevel = Field(default=ImpactLevel.LOW, description="Overall impact")
    affected_entities: list[AffectedEntity] = Field(
        default_factory=list, description="Affected entities"
    )
    affected_files: list[str] = Field(default_factory=list, description="Affected files")
    propagation_paths: list[PropagationPath] = Field(
        default_factory=list, description="Propagation paths"
    )
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Risk score")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    requires_review: bool = Field(default=False, description="Requires review")

    def get_direct_dependents(self) -> list[AffectedEntity]:
        """Get entities that directly depend on the changed entity."""
        return [e for e in self.affected_entities if e.is_direct]

    def get_entities_by_impact(self, level: ImpactLevel) -> list[AffectedEntity]:
        """Get entities with a specific impact level."""
        return [e for e in self.affected_entities if e.impact_level == level]


class ImpactAnalyzer:
    """Analyzer for code change impact.

    This class provides methods for analyzing the potential impact of
    code changes by traversing the dependency graph.
    """

    def __init__(self, graph_store: Any = None) -> None:
        """Initialize the analyzer.

        Args:
            graph_store: GraphStore instance for querying dependencies.
        """
        self._graph_store = graph_store
        self._logger = logger.bind(component="impact_analyzer")

    async def analyze_change(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str,
        file_path: str,
        change_type: ChangeType = ChangeType.MODIFY,
        max_depth: int = 5,
        repo_id: str | None = None,
    ) -> ImpactReport:
        """Analyze the impact of changing an entity.

        Args:
            entity_id: ID of the entity being changed.
            entity_name: Name of the entity.
            entity_type: Type of the entity.
            file_path: Path to the file containing the entity.
            change_type: Type of change being made.
            max_depth: Maximum depth to traverse.
            repo_id: Repository identifier.

        Returns:
            ImpactReport with analysis results.
        """
        self._logger.info(
            "analyzing_change_impact",
            entity_id=entity_id,
            entity_name=entity_name,
            change_type=change_type.value,
        )

        affected: list[AffectedEntity] = []
        affected_files: set[str] = {file_path}
        paths: list[PropagationPath] = []

        # Get direct dependents
        direct_dependents = await self._get_dependents(entity_id, entity_name, repo_id, max_depth=1)

        for dep in direct_dependents:
            affected.append(
                AffectedEntity(
                    entity_id=dep["entity_id"],
                    entity_name=dep["entity_name"],
                    entity_type=dep["entity_type"],
                    file_path=dep["file_path"],
                    start_line=dep.get("start_line", 0),
                    end_line=dep.get("end_line", 0),
                    impact_level=self._calculate_impact_level(change_type, 1),
                    impact_reason=f"Directly calls {entity_name}",
                    distance=1,
                    is_direct=True,
                )
            )
            affected_files.add(dep["file_path"])

        # Get transitive dependents
        if max_depth > 1:
            transitive = await self._get_transitive_dependents(
                entity_id, entity_name, repo_id, max_depth
            )
            for dep in transitive:
                if dep["entity_id"] not in {a.entity_id for a in affected}:
                    affected.append(
                        AffectedEntity(
                            entity_id=dep["entity_id"],
                            entity_name=dep["entity_name"],
                            entity_type=dep["entity_type"],
                            file_path=dep["file_path"],
                            start_line=dep.get("start_line", 0),
                            end_line=dep.get("end_line", 0),
                            impact_level=self._calculate_impact_level(change_type, dep["distance"]),
                            impact_reason=f"Transitively depends on {entity_name}",
                            distance=dep["distance"],
                            is_direct=False,
                        )
                    )
                    affected_files.add(dep["file_path"])

                    # Add propagation path
                    if "path" in dep:
                        paths.append(
                            PropagationPath(
                                entities=dep["path"],
                                impact_type="call",
                                description=f"Call chain to {entity_name}",
                            )
                        )

        # Calculate overall impact and risk
        overall_impact = self._calculate_overall_impact(affected, change_type)
        risk_score = self._calculate_risk_score(affected, change_type)
        recommendations = self._generate_recommendations(affected, change_type, risk_score)

        return ImpactReport(
            changed_entity_id=entity_id,
            changed_entity_name=entity_name,
            change_type=change_type,
            overall_impact=overall_impact,
            affected_entities=affected,
            affected_files=sorted(affected_files),
            propagation_paths=paths,
            risk_score=risk_score,
            recommendations=recommendations,
            requires_review=risk_score > 50
            or overall_impact
            in (
                ImpactLevel.HIGH,
                ImpactLevel.CRITICAL,
            ),
        )

    async def _get_dependents(
        self,
        entity_id: str,
        entity_name: str,
        repo_id: str | None,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Get entities that depend on the given entity.

        Args:
            entity_id: Entity identifier.
            entity_name: Entity name.
            repo_id: Repository identifier.
            max_depth: Maximum traversal depth.

        Returns:
            List of dependent entity info dicts.
        """
        if self._graph_store is None:
            return []

        try:
            # Query graph for callers
            callers: list[dict[str, Any]] = await self._graph_store.find_callers(
                repo_id=repo_id,
                function_name=entity_name,
                max_depth=max_depth,
            )
            return callers
        except Exception as e:
            self._logger.warning("failed_to_get_dependents", error=str(e))
            return []

    async def _get_transitive_dependents(
        self,
        entity_id: str,
        entity_name: str,
        repo_id: str | None,
        max_depth: int,
    ) -> list[dict[str, Any]]:
        """Get transitive dependents through the call graph.

        Args:
            entity_id: Entity identifier.
            entity_name: Entity name.
            repo_id: Repository identifier.
            max_depth: Maximum traversal depth.

        Returns:
            List of transitive dependent entity info dicts.
        """
        if self._graph_store is None:
            return []

        transitive: list[dict[str, Any]] = []
        visited: set[str] = {entity_id}
        queue: list[tuple[str, str, int, list[str]]] = [(entity_id, entity_name, 0, [entity_name])]

        while queue:
            current_id, current_name, depth, path = queue.pop(0)

            if depth >= max_depth:
                continue

            dependents = await self._get_dependents(current_id, current_name, repo_id, max_depth=1)

            for dep in dependents:
                if dep["entity_id"] not in visited:
                    visited.add(dep["entity_id"])
                    new_path = path + [dep["entity_name"]]
                    dep["distance"] = depth + 1
                    dep["path"] = new_path
                    transitive.append(dep)

                    queue.append((dep["entity_id"], dep["entity_name"], depth + 1, new_path))

        return transitive

    def _calculate_impact_level(
        self,
        change_type: ChangeType,
        distance: int,
    ) -> ImpactLevel:
        """Calculate impact level based on change type and distance.

        Args:
            change_type: Type of change.
            distance: Graph distance from the changed entity.

        Returns:
            Impact level.
        """
        # Base impact by change type
        if change_type == ChangeType.DELETE:
            base_impact = 4  # Critical
        elif change_type == ChangeType.RENAME:
            base_impact = 3  # High
        elif change_type == ChangeType.MODIFY or change_type == ChangeType.MOVE:
            base_impact = 2  # Medium
        else:  # ADD
            base_impact = 1  # Low

        # Reduce impact by distance
        adjusted = max(0, base_impact - (distance - 1))

        if adjusted >= 4:
            return ImpactLevel.CRITICAL
        elif adjusted >= 3:
            return ImpactLevel.HIGH
        elif adjusted >= 2:
            return ImpactLevel.MEDIUM
        elif adjusted >= 1:
            return ImpactLevel.LOW
        else:
            return ImpactLevel.NONE

    def _calculate_overall_impact(
        self,
        affected: list[AffectedEntity],
        change_type: ChangeType,
    ) -> ImpactLevel:
        """Calculate overall impact level for the change.

        Args:
            affected: List of affected entities.
            change_type: Type of change.

        Returns:
            Overall impact level.
        """
        if not affected:
            return ImpactLevel.NONE

        # Check for critical impacts
        critical_count = sum(1 for a in affected if a.impact_level == ImpactLevel.CRITICAL)
        high_count = sum(1 for a in affected if a.impact_level == ImpactLevel.HIGH)
        direct_count = sum(1 for a in affected if a.is_direct)

        if critical_count > 0 or (high_count > 5) or (direct_count > 10):
            return ImpactLevel.CRITICAL
        elif high_count > 0 or (direct_count > 5):
            return ImpactLevel.HIGH
        elif direct_count > 2 or len(affected) > 10:
            return ImpactLevel.MEDIUM
        else:
            return ImpactLevel.LOW

    def _calculate_risk_score(
        self,
        affected: list[AffectedEntity],
        change_type: ChangeType,
    ) -> float:
        """Calculate numerical risk score.

        Args:
            affected: List of affected entities.
            change_type: Type of change.

        Returns:
            Risk score (0-100).
        """
        if not affected:
            return 0.0

        # Base score from change type
        type_scores = {
            ChangeType.DELETE: 40,
            ChangeType.RENAME: 30,
            ChangeType.MODIFY: 20,
            ChangeType.MOVE: 20,
            ChangeType.ADD: 10,
        }
        score = type_scores.get(change_type, 20)

        # Add score based on affected entities
        direct_count = sum(1 for a in affected if a.is_direct)
        score += min(30, direct_count * 5)

        # Add score based on impact levels
        for entity in affected:
            if entity.impact_level == ImpactLevel.CRITICAL:
                score += 10
            elif entity.impact_level == ImpactLevel.HIGH:
                score += 5
            elif entity.impact_level == ImpactLevel.MEDIUM:
                score += 2

        # Add score for number of affected files
        affected_files = len({a.file_path for a in affected})
        score += min(20, affected_files * 2)

        return min(100.0, float(score))

    def _generate_recommendations(
        self,
        affected: list[AffectedEntity],
        change_type: ChangeType,
        risk_score: float,
    ) -> list[str]:
        """Generate recommendations based on impact analysis.

        Args:
            affected: List of affected entities.
            change_type: Type of change.
            risk_score: Calculated risk score.

        Returns:
            List of recommendation strings.
        """
        recommendations: list[str] = []

        # High risk recommendations
        if risk_score > 70:
            recommendations.append("Consider breaking this change into smaller increments")
            recommendations.append("Ensure comprehensive test coverage before deploying")

        # Direct dependent recommendations
        direct_count = sum(1 for a in affected if a.is_direct)
        if direct_count > 5:
            recommendations.append(f"Update {direct_count} direct callers that may be affected")

        # Change type specific recommendations
        if change_type == ChangeType.DELETE:
            recommendations.append("Verify all callers have been updated or removed")
            recommendations.append("Consider deprecation period before full removal")
        elif change_type == ChangeType.RENAME:
            recommendations.append("Update all import statements and references")
            recommendations.append("Consider adding a deprecation alias")
        elif change_type == ChangeType.MODIFY:
            if risk_score > 30:
                recommendations.append("Review changes for backward compatibility")

        # File-based recommendations
        affected_files = len({a.file_path for a in affected})
        if affected_files > 3:
            recommendations.append(f"Changes may affect {affected_files} files")

        # Test recommendations
        if risk_score > 50:
            recommendations.append("Run integration tests for all affected modules")

        return recommendations

    async def analyze_function_change(
        self,
        function_name: str,
        file_path: str,
        change_type: ChangeType = ChangeType.MODIFY,
        repo_id: str | None = None,
    ) -> ImpactReport:
        """Convenience method to analyze impact of changing a function.

        Args:
            function_name: Name of the function.
            file_path: Path to the containing file.
            change_type: Type of change.
            repo_id: Repository identifier.

        Returns:
            ImpactReport with analysis results.
        """
        entity_id = f"{file_path}:{function_name}"
        return await self.analyze_change(
            entity_id=entity_id,
            entity_name=function_name,
            entity_type="function",
            file_path=file_path,
            change_type=change_type,
            repo_id=repo_id,
        )

    async def analyze_class_change(
        self,
        class_name: str,
        file_path: str,
        change_type: ChangeType = ChangeType.MODIFY,
        repo_id: str | None = None,
    ) -> ImpactReport:
        """Convenience method to analyze impact of changing a class.

        Args:
            class_name: Name of the class.
            file_path: Path to the containing file.
            change_type: Type of change.
            repo_id: Repository identifier.

        Returns:
            ImpactReport with analysis results.
        """
        entity_id = f"{file_path}:{class_name}"
        return await self.analyze_change(
            entity_id=entity_id,
            entity_name=class_name,
            entity_type="class",
            file_path=file_path,
            change_type=change_type,
            repo_id=repo_id,
        )

    async def analyze_file_change(
        self,
        file_path: str,
        change_type: ChangeType = ChangeType.MODIFY,
        repo_id: str | None = None,
    ) -> ImpactReport:
        """Analyze impact of changing an entire file.

        Args:
            file_path: Path to the file.
            change_type: Type of change.
            repo_id: Repository identifier.

        Returns:
            ImpactReport with analysis results.
        """
        entity_id = file_path
        return await self.analyze_change(
            entity_id=entity_id,
            entity_name=file_path,
            entity_type="file",
            file_path=file_path,
            change_type=change_type,
            repo_id=repo_id,
        )
