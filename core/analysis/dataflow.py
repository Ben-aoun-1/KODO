"""Data flow analysis for tracking variable and data transformations.

This module provides tools for analyzing how data flows through code,
tracking variable definitions, uses, and transformations.
"""

from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class DataFlowNodeType(str, Enum):
    """Types of data flow nodes."""

    DEFINITION = "definition"
    USE = "use"
    ASSIGNMENT = "assignment"
    PARAMETER = "parameter"
    RETURN = "return"
    CALL_ARG = "call_arg"
    ATTRIBUTE = "attribute"
    IMPORT = "import"


class DataFlowNode(BaseModel):
    """A node in the data flow graph.

    Attributes:
        id: Unique node identifier.
        name: Variable or value name.
        node_type: Type of data flow node.
        file_path: Path to the containing file.
        line: Line number.
        column: Column number.
        scope: Scope of the variable (function name, class, etc.).
        data_type: Inferred or annotated type.
        value: Literal value if known.
        is_mutable: Whether the value can be modified.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Node ID")
    name: str = Field(..., description="Variable name")
    node_type: DataFlowNodeType = Field(..., description="Node type")
    file_path: str = Field(..., description="File path")
    line: int = Field(default=0, ge=0, description="Line number")
    column: int = Field(default=0, ge=0, description="Column number")
    scope: str = Field(default="<module>", description="Variable scope")
    data_type: str | None = Field(None, description="Data type")
    value: str | None = Field(None, description="Literal value")
    is_mutable: bool = Field(default=True, description="Is mutable")


class DataFlowEdge(BaseModel):
    """An edge in the data flow graph.

    Attributes:
        source_id: Source node ID.
        target_id: Target node ID.
        edge_type: Type of data flow relationship.
        transformation: Description of data transformation.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    edge_type: str = Field(default="flow", description="Edge type")
    transformation: str | None = Field(None, description="Transformation applied")


class DataFlowGraph(BaseModel):
    """A graph representing data flow through code.

    Attributes:
        repo_id: Repository identifier.
        file_path: Path to the analyzed file.
        nodes: List of data flow nodes.
        edges: List of data flow edges.
    """

    model_config = ConfigDict(frozen=False)

    repo_id: str = Field(..., description="Repository ID")
    file_path: str = Field(default="", description="File path")
    nodes: list[DataFlowNode] = Field(default_factory=list, description="Nodes")
    edges: list[DataFlowEdge] = Field(default_factory=list, description="Edges")

    def add_node(self, node: DataFlowNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: DataFlowEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> DataFlowNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_definitions(self, variable_name: str) -> list[DataFlowNode]:
        """Get all definition nodes for a variable."""
        return [
            n
            for n in self.nodes
            if n.name == variable_name
            and n.node_type in (DataFlowNodeType.DEFINITION, DataFlowNodeType.ASSIGNMENT)
        ]

    def get_uses(self, variable_name: str) -> list[DataFlowNode]:
        """Get all use nodes for a variable."""
        return [
            n for n in self.nodes if n.name == variable_name and n.node_type == DataFlowNodeType.USE
        ]

    def get_incoming_edges(self, node_id: str) -> list[DataFlowEdge]:
        """Get all edges flowing into a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[DataFlowEdge]:
        """Get all edges flowing out of a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def trace_data_origin(self, node_id: str) -> list[DataFlowNode]:
        """Trace data back to its origin(s).

        Args:
            node_id: ID of the node to trace from.

        Returns:
            List of origin nodes.
        """
        origins: list[DataFlowNode] = []
        visited: set[str] = set()

        def traverse(current_id: str) -> None:
            if current_id in visited:
                return
            visited.add(current_id)

            node = self.get_node(current_id)
            if not node:
                return

            incoming = self.get_incoming_edges(current_id)
            if not incoming:
                # This is an origin
                origins.append(node)
            else:
                for edge in incoming:
                    traverse(edge.source_id)

        traverse(node_id)
        return origins

    def trace_data_destination(self, node_id: str) -> list[DataFlowNode]:
        """Trace data forward to its destination(s).

        Args:
            node_id: ID of the node to trace from.

        Returns:
            List of destination nodes.
        """
        destinations: list[DataFlowNode] = []
        visited: set[str] = set()

        def traverse(current_id: str) -> None:
            if current_id in visited:
                return
            visited.add(current_id)

            node = self.get_node(current_id)
            if not node:
                return

            outgoing = self.get_outgoing_edges(current_id)
            if not outgoing:
                # This is a destination
                destinations.append(node)
            else:
                for edge in outgoing:
                    traverse(edge.target_id)

        traverse(node_id)
        return destinations


class DataFlowAnalyzer:
    """Analyzer for data flow through code.

    This class provides methods for building data flow graphs from
    source code and analyzing data dependencies.
    """

    def __init__(self, repo_id: str) -> None:
        """Initialize the analyzer.

        Args:
            repo_id: Repository identifier.
        """
        self.repo_id = repo_id
        self._logger = logger.bind(component="dataflow_analyzer", repo_id=repo_id)
        self._node_counter = 0

    def _generate_node_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def analyze_function(
        self,
        function_name: str,
        source_code: str,
        file_path: str,
        start_line: int = 0,
        parameters: list[dict[str, Any]] | None = None,
    ) -> DataFlowGraph:
        """Analyze data flow within a function.

        Args:
            function_name: Name of the function.
            source_code: Source code of the function.
            file_path: Path to the containing file.
            start_line: Starting line number.
            parameters: List of parameter info dicts.

        Returns:
            DataFlowGraph for the function.
        """
        graph = DataFlowGraph(repo_id=self.repo_id, file_path=file_path)

        # Add parameter nodes
        if parameters:
            for param in parameters:
                node = DataFlowNode(
                    id=self._generate_node_id("param"),
                    name=param.get("name", ""),
                    node_type=DataFlowNodeType.PARAMETER,
                    file_path=file_path,
                    line=start_line,
                    scope=function_name,
                    data_type=param.get("type_annotation"),
                )
                graph.add_node(node)

        # Parse assignments and uses from source
        self._analyze_assignments(graph, source_code, file_path, function_name, start_line)
        self._analyze_returns(graph, source_code, file_path, function_name, start_line)

        return graph

    def _analyze_assignments(
        self,
        graph: DataFlowGraph,
        source_code: str,
        file_path: str,
        scope: str,
        base_line: int,
    ) -> None:
        """Analyze variable assignments in source code.

        Args:
            graph: DataFlowGraph to add nodes to.
            source_code: Source code to analyze.
            file_path: Path to the file.
            scope: Current scope.
            base_line: Base line number offset.
        """
        import re

        lines = source_code.split("\n")

        for i, line in enumerate(lines):
            line_num = base_line + i

            # Match simple assignments: variable = value
            assignment_pattern = r"^(\s*)(\w+)\s*=\s*(.+)$"
            match = re.match(assignment_pattern, line)

            if match:
                var_name = match.group(2)
                value = match.group(3).strip()

                # Skip function definitions and class definitions
                if value.startswith(("def ", "class ", "lambda")):
                    continue

                # Create assignment node
                assign_node = DataFlowNode(
                    id=self._generate_node_id("assign"),
                    name=var_name,
                    node_type=DataFlowNodeType.ASSIGNMENT,
                    file_path=file_path,
                    line=line_num,
                    scope=scope,
                    value=value[:50] if len(value) < 50 else f"{value[:47]}...",
                )
                graph.add_node(assign_node)

                # Find variables used in the RHS
                var_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
                used_vars = set(re.findall(var_pattern, value))

                # Filter out Python keywords and function calls
                keywords = {
                    "True",
                    "False",
                    "None",
                    "and",
                    "or",
                    "not",
                    "if",
                    "else",
                    "for",
                    "in",
                    "is",
                }
                used_vars -= keywords

                # Create edges from used variables to this assignment
                for var in used_vars:
                    if var != var_name:  # Don't create self-loop
                        # Find the most recent definition of this variable
                        definitions = graph.get_definitions(var)
                        if definitions:
                            latest_def = definitions[-1]
                            graph.add_edge(
                                DataFlowEdge(
                                    source_id=latest_def.id,
                                    target_id=assign_node.id,
                                    edge_type="use",
                                )
                            )

    def _analyze_returns(
        self,
        graph: DataFlowGraph,
        source_code: str,
        file_path: str,
        scope: str,
        base_line: int,
    ) -> None:
        """Analyze return statements in source code.

        Args:
            graph: DataFlowGraph to add nodes to.
            source_code: Source code to analyze.
            file_path: Path to the file.
            scope: Current scope.
            base_line: Base line number offset.
        """
        import re

        lines = source_code.split("\n")

        for i, line in enumerate(lines):
            line_num = base_line + i

            # Match return statements
            return_pattern = r"^\s*return\s+(.+)?$"
            match = re.match(return_pattern, line)

            if match:
                return_value = match.group(1) if match.group(1) else ""

                return_node = DataFlowNode(
                    id=self._generate_node_id("return"),
                    name="<return>",
                    node_type=DataFlowNodeType.RETURN,
                    file_path=file_path,
                    line=line_num,
                    scope=scope,
                    value=return_value[:50] if return_value else None,
                )
                graph.add_node(return_node)

                # Find variables used in return
                if return_value:
                    var_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
                    used_vars = set(re.findall(var_pattern, return_value))

                    for var in used_vars:
                        definitions = graph.get_definitions(var)
                        if definitions:
                            latest_def = definitions[-1]
                            graph.add_edge(
                                DataFlowEdge(
                                    source_id=latest_def.id,
                                    target_id=return_node.id,
                                    edge_type="return_value",
                                )
                            )

    def find_tainted_data(
        self,
        graph: DataFlowGraph,
        taint_sources: list[str],
    ) -> list[DataFlowNode]:
        """Find all nodes that could be affected by tainted sources.

        This is useful for security analysis to track untrusted input.

        Args:
            graph: DataFlowGraph to analyze.
            taint_sources: Names of tainted source variables.

        Returns:
            List of potentially tainted nodes.
        """
        tainted: list[DataFlowNode] = []
        tainted_ids: set[str] = set()

        # Find initial tainted nodes
        for source_name in taint_sources:
            for node in graph.nodes:
                if node.name == source_name:
                    tainted.append(node)
                    tainted_ids.add(node.id)

        # Propagate taint
        changed = True
        while changed:
            changed = False
            for edge in graph.edges:
                if edge.source_id in tainted_ids and edge.target_id not in tainted_ids:
                    target = graph.get_node(edge.target_id)
                    if target:
                        tainted.append(target)
                        tainted_ids.add(edge.target_id)
                        changed = True

        return tainted

    def find_unused_assignments(self, graph: DataFlowGraph) -> list[DataFlowNode]:
        """Find assignments whose values are never used.

        Args:
            graph: DataFlowGraph to analyze.

        Returns:
            List of unused assignment nodes.
        """
        unused: list[DataFlowNode] = []

        for node in graph.nodes:
            if node.node_type in (
                DataFlowNodeType.ASSIGNMENT,
                DataFlowNodeType.DEFINITION,
            ):
                outgoing = graph.get_outgoing_edges(node.id)
                if not outgoing:
                    # Check if there's a later definition that shadows this
                    is_shadowed = False
                    for other in graph.nodes:
                        if (
                            other.id != node.id
                            and other.name == node.name
                            and other.line > node.line
                            and other.node_type
                            in (
                                DataFlowNodeType.ASSIGNMENT,
                                DataFlowNodeType.DEFINITION,
                            )
                        ):
                            is_shadowed = True
                            break

                    if not is_shadowed:
                        unused.append(node)

        return unused

    def analyze_file(
        self,
        file_path: str,
        source_code: str,
        entities: list[dict[str, Any]] | None = None,
    ) -> DataFlowGraph:
        """Analyze data flow for an entire file.

        Args:
            file_path: Path to the file.
            source_code: Full source code.
            entities: List of parsed entities with source info.

        Returns:
            Combined DataFlowGraph for the file.
        """
        graph = DataFlowGraph(repo_id=self.repo_id, file_path=file_path)

        # Analyze module-level code
        self._analyze_assignments(graph, source_code, file_path, "<module>", 0)

        # Analyze each function/method if entities provided
        if entities:
            for entity in entities:
                if entity.get("type") in ("function", "method"):
                    func_graph = self.analyze_function(
                        function_name=entity.get("name", ""),
                        source_code=entity.get("source_code", ""),
                        file_path=file_path,
                        start_line=entity.get("start_line", 0),
                        parameters=entity.get("parameters"),
                    )
                    # Merge into main graph
                    for node in func_graph.nodes:
                        graph.add_node(node)
                    for edge in func_graph.edges:
                        graph.add_edge(edge)

        return graph
