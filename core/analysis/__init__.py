"""Code analysis module for Kodo.

This module provides static analysis capabilities including:
- Code metrics (complexity, coupling, cohesion)
- Data flow tracking
- Impact analysis
- Pattern detection
"""

from .dataflow import DataFlowAnalyzer, DataFlowGraph, DataFlowNode
from .impact import ImpactAnalyzer, ImpactLevel, ImpactReport
from .metrics import CodeMetrics, ComplexityMetrics, MetricsCalculator
from .models import AnalysisConfig, AnalysisResult
from .patterns import CodePattern, PatternDetector, PatternMatch

__all__ = [
    # Models
    "AnalysisResult",
    "AnalysisConfig",
    # Metrics
    "CodeMetrics",
    "MetricsCalculator",
    "ComplexityMetrics",
    # Data flow
    "DataFlowAnalyzer",
    "DataFlowGraph",
    "DataFlowNode",
    # Impact
    "ImpactAnalyzer",
    "ImpactReport",
    "ImpactLevel",
    # Patterns
    "PatternDetector",
    "CodePattern",
    "PatternMatch",
]
