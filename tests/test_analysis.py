"""Tests for the analysis module.

This module contains comprehensive tests for:
- Code metrics calculation
- Data flow analysis
- Impact analysis
- Pattern detection
"""

from datetime import UTC, datetime

import pytest

from core.analysis.dataflow import (
    DataFlowAnalyzer,
    DataFlowEdge,
    DataFlowGraph,
    DataFlowNode,
    DataFlowNodeType,
)
from core.analysis.impact import (
    AffectedEntity,
    ChangeType,
    ImpactAnalyzer,
    ImpactLevel,
    ImpactReport,
    PropagationPath,
)
from core.analysis.metrics import (
    CodeMetrics,
    ComplexityMetrics,
    CouplingMetrics,
    MetricsCalculator,
)
from core.analysis.models import (
    AnalysisCategory,
    AnalysisConfig,
    AnalysisResult,
    AnalysisSeverity,
    AnalysisSummary,
    EntityMetrics,
    FileMetrics,
    Finding,
)
from core.analysis.patterns import (
    CodePattern,
    PatternCategory,
    PatternDetector,
    PatternMatch,
    PatternType,
)

# =============================================================================
# Analysis Models Tests
# =============================================================================


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = AnalysisConfig()
        assert config.max_complexity == 10
        assert config.max_coupling == 0.7
        assert config.min_cohesion == 0.3
        assert config.include_metrics is True
        assert config.include_dataflow is True
        assert config.max_depth == 10

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = AnalysisConfig(
            max_complexity=15,
            max_coupling=0.5,
            min_cohesion=0.5,
            include_patterns=False,
        )
        assert config.max_complexity == 15
        assert config.max_coupling == 0.5
        assert config.include_patterns is False


class TestFinding:
    """Tests for Finding model."""

    def test_create_finding(self):
        """Test creating a finding."""
        finding = Finding(
            id="test:1:complexity",
            category=AnalysisCategory.COMPLEXITY,
            severity=AnalysisSeverity.WARNING,
            message="High complexity",
            file_path="test.py",
            start_line=10,
            end_line=50,
            entity_name="process_data",
            entity_type="function",
            suggestion="Break into smaller functions",
        )
        assert finding.id == "test:1:complexity"
        assert finding.category == AnalysisCategory.COMPLEXITY
        assert finding.severity == AnalysisSeverity.WARNING
        assert finding.entity_name == "process_data"


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_create_result(self):
        """Test creating an analysis result."""
        result = AnalysisResult(
            repo_id="test-repo",
            analysis_id="analysis-123",
            success=True,
        )
        assert result.repo_id == "test-repo"
        assert result.success is True
        assert len(result.findings) == 0

    def test_result_with_findings(self):
        """Test result with findings."""
        findings = [
            Finding(
                id="f1",
                category=AnalysisCategory.COMPLEXITY,
                severity=AnalysisSeverity.CRITICAL,
                message="Critical issue",
                file_path="test.py",
            ),
            Finding(
                id="f2",
                category=AnalysisCategory.COUPLING,
                severity=AnalysisSeverity.WARNING,
                message="Warning",
                file_path="test.py",
            ),
        ]
        result = AnalysisResult(
            repo_id="test-repo",
            analysis_id="a1",
            findings=findings,
        )
        assert len(result.findings) == 2
        critical = result.get_critical_findings()
        assert len(critical) == 1
        assert critical[0].id == "f1"

    def test_result_get_findings_by_file(self):
        """Test filtering findings by file."""
        findings = [
            Finding(
                id="f1",
                category=AnalysisCategory.COMPLEXITY,
                severity=AnalysisSeverity.INFO,
                message="m1",
                file_path="a.py",
            ),
            Finding(
                id="f2",
                category=AnalysisCategory.COMPLEXITY,
                severity=AnalysisSeverity.INFO,
                message="m2",
                file_path="b.py",
            ),
            Finding(
                id="f3",
                category=AnalysisCategory.COMPLEXITY,
                severity=AnalysisSeverity.INFO,
                message="m3",
                file_path="a.py",
            ),
        ]
        result = AnalysisResult(repo_id="r", analysis_id="a", findings=findings)
        a_findings = result.get_findings_by_file("a.py")
        assert len(a_findings) == 2


class TestEntityMetrics:
    """Tests for EntityMetrics model."""

    def test_create_entity_metrics(self):
        """Test creating entity metrics."""
        metrics = EntityMetrics(
            entity_id="test.py:func:10",
            entity_name="func",
            entity_type="function",
            file_path="test.py",
            start_line=10,
            end_line=30,
            lines_of_code=20,
            cyclomatic_complexity=5,
            cognitive_complexity=8,
            parameters=3,
        )
        assert metrics.entity_name == "func"
        assert metrics.cyclomatic_complexity == 5
        assert metrics.parameters == 3


# =============================================================================
# Metrics Calculator Tests
# =============================================================================


class TestComplexityMetrics:
    """Tests for ComplexityMetrics model."""

    def test_default_metrics(self):
        """Test default complexity metrics."""
        metrics = ComplexityMetrics()
        assert metrics.cyclomatic == 1
        assert metrics.cognitive == 0
        assert metrics.nesting_depth == 0


class TestCouplingMetrics:
    """Tests for CouplingMetrics model."""

    def test_coupling_instability(self):
        """Test coupling instability calculation."""
        # High efferent = unstable
        metrics = CouplingMetrics(afferent=2, efferent=8, instability=0.8)
        assert metrics.instability == 0.8

    def test_distance_from_main_sequence(self):
        """Test distance from main sequence."""
        # Perfect balance: A + I = 1
        metrics = CouplingMetrics(instability=0.5, abstractness=0.5)
        assert metrics.distance_from_main_sequence == pytest.approx(0.0)

        # Off balance
        metrics = CouplingMetrics(instability=0.8, abstractness=0.8)
        assert metrics.distance_from_main_sequence == pytest.approx(0.6)


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a metrics calculator."""
        return MetricsCalculator()

    def test_simple_function_complexity(self, calculator):
        """Test complexity of a simple function."""
        code = """
def greet(name):
    return f"Hello, {name}!"
"""
        complexity = calculator.calculate_cyclomatic_complexity(code)
        assert complexity == 1  # No branches

    def test_function_with_if(self, calculator):
        """Test complexity with if statement."""
        code = """
def check_age(age):
    if age >= 18:
        return "Adult"
    else:
        return "Minor"
"""
        complexity = calculator.calculate_cyclomatic_complexity(code)
        assert complexity >= 2  # At least one decision

    def test_function_with_loops(self, calculator):
        """Test complexity with loops."""
        code = """
def process_items(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item)
    return result
"""
        complexity = calculator.calculate_cyclomatic_complexity(code)
        assert complexity >= 3  # for + if

    def test_cognitive_complexity(self, calculator):
        """Test cognitive complexity calculation."""
        code = """
def complex_function(data):
    if data:
        for item in data:
            if item > 0:
                for sub in item:
                    if sub:
                        process(sub)
"""
        cognitive = calculator.calculate_cognitive_complexity(code)
        assert cognitive > 5  # Nested structures

    def test_nesting_depth(self, calculator):
        """Test nesting depth calculation."""
        code = """
def nested():
    if True:
        for i in range(10):
            if i > 5:
                while True:
                    break
"""
        depth = calculator.calculate_nesting_depth(code)
        assert depth >= 3

    def test_lines_of_code(self, calculator):
        """Test lines of code calculation."""
        code = '''
def example():
    """This is a docstring."""
    # This is a comment
    x = 1
    y = 2

    return x + y
'''
        code_lines, comment_lines, blank_lines = calculator.calculate_lines_of_code(code)
        assert code_lines >= 3
        assert blank_lines >= 1

    def test_calculate_entity_metrics(self, calculator):
        """Test full entity metrics calculation."""
        code = """
def process(data, config):
    if not data:
        return None
    result = []
    for item in data:
        if item.valid:
            result.append(item)
    return result
"""
        metrics = calculator.calculate_entity_metrics(
            entity_id="test:process:1",
            entity_name="process",
            entity_type="function",
            file_path="test.py",
            source_code=code,
            start_line=1,
            end_line=10,
            parameters=2,
        )
        assert metrics.entity_name == "process"
        assert metrics.cyclomatic_complexity >= 3
        assert metrics.parameters == 2

    def test_calculate_file_metrics(self, calculator):
        """Test file metrics calculation."""
        code = """
import os
from typing import List

def func1():
    pass

def func2():
    pass

class MyClass:
    pass
"""
        metrics = calculator.calculate_file_metrics(
            file_path="test.py",
            source_code=code,
            language="python",
        )
        assert metrics.file_path == "test.py"
        assert metrics.import_count >= 1

    def test_generate_findings_complexity(self, calculator):
        """Test generating findings for complexity violations."""
        entity = EntityMetrics(
            entity_id="test:func:1",
            entity_name="complex_func",
            entity_type="function",
            file_path="test.py",
            cyclomatic_complexity=15,  # Above default threshold
        )
        findings = calculator.generate_findings(entity)
        assert len(findings) >= 1
        assert any(f.category == AnalysisCategory.COMPLEXITY for f in findings)


# =============================================================================
# Data Flow Tests
# =============================================================================


class TestDataFlowNode:
    """Tests for DataFlowNode model."""

    def test_create_node(self):
        """Test creating a data flow node."""
        node = DataFlowNode(
            id="node_1",
            name="x",
            node_type=DataFlowNodeType.ASSIGNMENT,
            file_path="test.py",
            line=10,
            scope="main",
        )
        assert node.name == "x"
        assert node.node_type == DataFlowNodeType.ASSIGNMENT


class TestDataFlowGraph:
    """Tests for DataFlowGraph."""

    def test_create_graph(self):
        """Test creating a data flow graph."""
        graph = DataFlowGraph(repo_id="test-repo", file_path="test.py")
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_nodes_and_edges(self):
        """Test adding nodes and edges."""
        graph = DataFlowGraph(repo_id="test-repo")

        node1 = DataFlowNode(
            id="n1",
            name="x",
            node_type=DataFlowNodeType.PARAMETER,
            file_path="test.py",
        )
        node2 = DataFlowNode(
            id="n2",
            name="y",
            node_type=DataFlowNodeType.ASSIGNMENT,
            file_path="test.py",
        )

        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(DataFlowEdge(source_id="n1", target_id="n2"))

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.get_node("n1") == node1

    def test_get_definitions_and_uses(self):
        """Test getting definitions and uses."""
        graph = DataFlowGraph(repo_id="test-repo")

        graph.add_node(
            DataFlowNode(
                id="d1", name="x", node_type=DataFlowNodeType.DEFINITION, file_path="test.py"
            )
        )
        graph.add_node(
            DataFlowNode(id="u1", name="x", node_type=DataFlowNodeType.USE, file_path="test.py")
        )
        graph.add_node(
            DataFlowNode(id="u2", name="x", node_type=DataFlowNodeType.USE, file_path="test.py")
        )

        defs = graph.get_definitions("x")
        uses = graph.get_uses("x")

        assert len(defs) == 1
        assert len(uses) == 2

    def test_trace_data_origin(self):
        """Test tracing data to origin."""
        graph = DataFlowGraph(repo_id="test-repo")

        # x -> y -> z
        graph.add_node(
            DataFlowNode(
                id="x", name="x", node_type=DataFlowNodeType.PARAMETER, file_path="test.py"
            )
        )
        graph.add_node(
            DataFlowNode(
                id="y", name="y", node_type=DataFlowNodeType.ASSIGNMENT, file_path="test.py"
            )
        )
        graph.add_node(
            DataFlowNode(
                id="z", name="z", node_type=DataFlowNodeType.ASSIGNMENT, file_path="test.py"
            )
        )

        graph.add_edge(DataFlowEdge(source_id="x", target_id="y"))
        graph.add_edge(DataFlowEdge(source_id="y", target_id="z"))

        origins = graph.trace_data_origin("z")
        assert len(origins) == 1
        assert origins[0].id == "x"


class TestDataFlowAnalyzer:
    """Tests for DataFlowAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a data flow analyzer."""
        return DataFlowAnalyzer(repo_id="test-repo")

    def test_analyze_simple_function(self, analyzer):
        """Test analyzing a simple function."""
        code = """
def add(x, y):
    result = x + y
    return result
"""
        graph = analyzer.analyze_function(
            function_name="add",
            source_code=code,
            file_path="test.py",
            parameters=[{"name": "x"}, {"name": "y"}],
        )

        assert len(graph.nodes) > 0
        # Should have parameter nodes
        params = [n for n in graph.nodes if n.node_type == DataFlowNodeType.PARAMETER]
        assert len(params) == 2

    def test_find_tainted_data(self, analyzer):
        """Test taint tracking."""
        graph = DataFlowGraph(repo_id="test-repo")

        # Create a flow: user_input -> processed -> result
        graph.add_node(
            DataFlowNode(
                id="input",
                name="user_input",
                node_type=DataFlowNodeType.PARAMETER,
                file_path="test.py",
            )
        )
        graph.add_node(
            DataFlowNode(
                id="processed",
                name="processed",
                node_type=DataFlowNodeType.ASSIGNMENT,
                file_path="test.py",
            )
        )
        graph.add_node(
            DataFlowNode(
                id="result", name="result", node_type=DataFlowNodeType.RETURN, file_path="test.py"
            )
        )

        graph.add_edge(DataFlowEdge(source_id="input", target_id="processed"))
        graph.add_edge(DataFlowEdge(source_id="processed", target_id="result"))

        tainted = analyzer.find_tainted_data(graph, ["user_input"])
        assert len(tainted) == 3  # All nodes tainted


# =============================================================================
# Impact Analysis Tests
# =============================================================================


class TestAffectedEntity:
    """Tests for AffectedEntity model."""

    def test_create_affected_entity(self):
        """Test creating an affected entity."""
        entity = AffectedEntity(
            entity_id="test:func:1",
            entity_name="func",
            entity_type="function",
            file_path="test.py",
            impact_level=ImpactLevel.HIGH,
            impact_reason="Directly calls modified function",
            distance=1,
            is_direct=True,
        )
        assert entity.impact_level == ImpactLevel.HIGH
        assert entity.is_direct is True


class TestImpactReport:
    """Tests for ImpactReport model."""

    def test_create_report(self):
        """Test creating an impact report."""
        report = ImpactReport(
            changed_entity_id="test:func:1",
            changed_entity_name="func",
            change_type=ChangeType.MODIFY,
            overall_impact=ImpactLevel.MEDIUM,
            risk_score=45.0,
        )
        assert report.change_type == ChangeType.MODIFY
        assert report.risk_score == 45.0

    def test_report_with_affected_entities(self):
        """Test report with affected entities."""
        affected = [
            AffectedEntity(
                entity_id="e1",
                entity_name="caller1",
                entity_type="function",
                file_path="a.py",
                is_direct=True,
            ),
            AffectedEntity(
                entity_id="e2",
                entity_name="caller2",
                entity_type="function",
                file_path="b.py",
                is_direct=False,
            ),
        ]
        report = ImpactReport(
            changed_entity_id="x",
            changed_entity_name="x",
            affected_entities=affected,
        )

        direct = report.get_direct_dependents()
        assert len(direct) == 1
        assert direct[0].entity_name == "caller1"


class TestImpactAnalyzer:
    """Tests for ImpactAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create an impact analyzer."""
        return ImpactAnalyzer(graph_store=None)

    @pytest.mark.asyncio
    async def test_analyze_change_no_graph(self, analyzer):
        """Test analyzing change without graph store."""
        report = await analyzer.analyze_change(
            entity_id="test:func:1",
            entity_name="process_data",
            entity_type="function",
            file_path="test.py",
            change_type=ChangeType.MODIFY,
        )

        assert report.changed_entity_name == "process_data"
        assert report.change_type == ChangeType.MODIFY
        # Without graph store, no affected entities
        assert len(report.affected_entities) == 0

    def test_calculate_impact_level(self, analyzer):
        """Test impact level calculation."""
        # Delete has high base impact
        level = analyzer._calculate_impact_level(ChangeType.DELETE, distance=1)
        assert level == ImpactLevel.CRITICAL

        # Modify has medium base impact
        level = analyzer._calculate_impact_level(ChangeType.MODIFY, distance=1)
        assert level == ImpactLevel.MEDIUM

        # Distance reduces impact
        level = analyzer._calculate_impact_level(ChangeType.DELETE, distance=3)
        assert level != ImpactLevel.CRITICAL  # Impact decreases with distance

    def test_calculate_risk_score(self, analyzer):
        """Test risk score calculation."""
        affected = [
            AffectedEntity(
                entity_id="e1",
                entity_name="f1",
                entity_type="function",
                file_path="a.py",
                is_direct=True,
                impact_level=ImpactLevel.HIGH,
            ),
        ]

        score = analyzer._calculate_risk_score(affected, ChangeType.DELETE)
        assert score > 0
        assert score <= 100

    def test_generate_recommendations(self, analyzer):
        """Test recommendation generation."""
        affected = [
            AffectedEntity(
                entity_id=f"e{i}",
                entity_name=f"f{i}",
                entity_type="function",
                file_path=f"{chr(97+i)}.py",
                is_direct=True,
            )
            for i in range(6)
        ]

        recommendations = analyzer._generate_recommendations(affected, ChangeType.DELETE, 75.0)

        assert len(recommendations) > 0
        assert any("direct" in r.lower() for r in recommendations)


# =============================================================================
# Pattern Detection Tests
# =============================================================================


class TestPatternMatch:
    """Tests for PatternMatch model."""

    def test_create_match(self):
        """Test creating a pattern match."""
        match = PatternMatch(
            pattern_id="singleton",
            pattern_name="Singleton Pattern",
            pattern_type=PatternType.SINGLETON,
            category=PatternCategory.DESIGN_PATTERN,
            file_path="test.py",
            start_line=10,
            confidence=0.9,
        )
        assert match.pattern_type == PatternType.SINGLETON
        assert match.confidence == 0.9


class TestPatternDetector:
    """Tests for PatternDetector."""

    @pytest.fixture
    def detector(self):
        """Create a pattern detector."""
        return PatternDetector()

    def test_detect_singleton_pattern(self, detector):
        """Test detecting singleton pattern."""
        code = """
class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
"""
        matches = detector.detect_singleton(code, "database.py")
        assert len(matches) >= 1
        assert matches[0].pattern_type == PatternType.SINGLETON

    def test_detect_factory_pattern(self, detector):
        """Test detecting factory pattern."""
        code = """
class ShapeFactory:
    @staticmethod
    def create_shape(shape_type):
        if shape_type == "circle":
            return Circle()
        elif shape_type == "square":
            return Square()
        return None
"""
        matches = detector.detect_factory(code, "factory.py")
        assert len(matches) >= 1
        factory_matches = [m for m in matches if m.pattern_type == PatternType.FACTORY]
        assert len(factory_matches) >= 1

    def test_detect_long_method(self, detector):
        """Test detecting long method anti-pattern."""
        entities = [
            {
                "name": "very_long_function",
                "type": "function",
                "start_line": 1,
                "end_line": 100,  # 100 lines
            }
        ]

        matches = detector.detect_patterns("", "test.py", entities)
        long_method_matches = [m for m in matches if m.pattern_type == PatternType.LONG_METHOD]
        assert len(long_method_matches) >= 1

    def test_detect_long_parameter_list(self, detector):
        """Test detecting long parameter list."""
        entities = [
            {
                "name": "complex_function",
                "type": "function",
                "start_line": 1,
                "end_line": 10,
                "parameters": [
                    {"name": "a"},
                    {"name": "b"},
                    {"name": "c"},
                    {"name": "d"},
                    {"name": "e"},
                    {"name": "f"},
                    {"name": "g"},
                ],
            }
        ]

        matches = detector.detect_patterns("", "test.py", entities)
        param_matches = [m for m in matches if m.pattern_type == PatternType.LONG_PARAMETER_LIST]
        assert len(param_matches) >= 1

    def test_detect_magic_numbers(self, detector):
        """Test detecting magic numbers."""
        code = """
def calculate_price(quantity):
    if quantity > 2500:
        return quantity * 0.85
    items = data[5:20]
    return quantity * 0.95
"""
        matches = detector.detect_patterns(code, "pricing.py")
        magic_matches = [m for m in matches if m.pattern_type == PatternType.MAGIC_NUMBERS]
        assert len(magic_matches) >= 1  # Should detect 2500 or slice [5:20]

    def test_detect_global_state(self, detector):
        """Test detecting global state."""
        code = """
_cache = {}
ITEMS = []

def add_to_cache(key, value):
    global _cache
    _cache[key] = value
"""
        matches = detector.detect_patterns(code, "state.py")
        global_matches = [m for m in matches if m.pattern_type == PatternType.GLOBAL_STATE]
        assert len(global_matches) >= 1

    def test_generate_findings(self, detector):
        """Test generating findings from matches."""
        matches = [
            PatternMatch(
                pattern_id="god_class",
                pattern_name="God Class",
                pattern_type=PatternType.GOD_CLASS,
                category=PatternCategory.CODE_SMELL,
                file_path="large.py",
                start_line=1,
                end_line=500,
                entity_name="MegaClass",
            ),
        ]

        findings = detector.generate_findings(matches)
        assert len(findings) == 1
        assert findings[0].category == AnalysisCategory.PATTERN
        assert findings[0].entity_name == "MegaClass"

    def test_no_findings_for_design_patterns(self, detector):
        """Test that design patterns don't generate findings."""
        matches = [
            PatternMatch(
                pattern_id="singleton",
                pattern_name="Singleton Pattern",
                pattern_type=PatternType.SINGLETON,
                category=PatternCategory.DESIGN_PATTERN,
                file_path="singleton.py",
                start_line=1,
            ),
        ]

        findings = detector.generate_findings(matches)
        assert len(findings) == 0  # Design patterns are not problems


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnalysisIntegration:
    """Integration tests for analysis module."""

    def test_full_metrics_analysis(self):
        """Test full metrics analysis workflow."""
        code = """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}

    def process(self, data, options, validate=True, transform=True, filter_empty=True, max_items=None):
        results = []
        for item in data:
            if validate:
                if not self._validate(item):
                    continue
            if transform:
                item = self._transform(item)
            if filter_empty and not item:
                continue
            results.append(item)
            if max_items and len(results) >= max_items:
                break
        return results

    def _validate(self, item):
        return item is not None

    def _transform(self, item):
        return str(item).upper()
"""
        calculator = MetricsCalculator()
        metrics = calculator.calculate_complexity_metrics(code)

        assert metrics.cyclomatic >= 5
        assert metrics.lines_of_code > 10

    def test_pattern_and_metrics_combined(self):
        """Test combining pattern detection with metrics."""
        code = """
class GodClass:
    def __init__(self):
        self.data = []
        self.config = {}
        self.cache = {}
        self.state = None
"""
        # Add many more methods to make it a "god class"
        for i in range(20):
            code += f"""
    def method_{i}(self):
        pass
"""

        detector = PatternDetector()
        entities = [
            {
                "name": "GodClass",
                "type": "class",
                "start_line": 1,
                "end_line": 100,
                "methods": [{"name": f"method_{i}"} for i in range(20)],
            }
        ]

        matches = detector.detect_patterns(code, "god.py", entities)
        god_class_matches = [m for m in matches if m.pattern_type == PatternType.GOD_CLASS]
        assert len(god_class_matches) >= 1
