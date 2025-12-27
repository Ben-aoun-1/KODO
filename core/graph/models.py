"""Pydantic models for Neo4j graph nodes and relationships.

This module defines the data models used to represent entities stored in the
Neo4j graph database. These models map to the Neo4j schema defined in CLAUDE.md.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Enumeration of supported relationship types in the graph.

    These correspond to the Neo4j relationship types defined in the schema.
    """

    CONTAINS = "CONTAINS"  # Repository -> File
    DEFINES = "DEFINES"  # File -> Module|Class|Function
    HAS_METHOD = "HAS_METHOD"  # Class -> Method
    INHERITS = "INHERITS"  # Class -> Class
    CALLS = "CALLS"  # Function|Method -> Function|Method
    USES = "USES"  # Function|Method -> Variable
    IMPORTS = "IMPORTS"  # File -> Module|File
    RETURNS = "RETURNS"  # Function|Method -> Type
    ACCEPTS = "ACCEPTS"  # Function|Method -> Parameter type
    TYPE_OF = "TYPE_OF"  # Variable -> Type


class NodeType(str, Enum):
    """Enumeration of supported node types in the graph."""

    REPOSITORY = "Repository"
    FILE = "File"
    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    VARIABLE = "Variable"


class BaseNode(BaseModel):
    """Base model for all graph nodes.

    Attributes:
        id: Unique identifier for the node.
        node_type: The type of node (label in Neo4j).
    """

    id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Node type (Neo4j label)")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"

    def to_properties(self) -> dict[str, Any]:
        """Convert node to Neo4j properties dictionary.

        Returns:
            Dictionary of properties suitable for Neo4j storage.
        """
        props = self.model_dump(exclude={"node_type"})
        # Convert datetime objects to ISO format strings
        for key, value in props.items():
            if isinstance(value, datetime):
                props[key] = value.isoformat()
        return props


class RepositoryNode(BaseNode):
    """Model for Repository nodes.

    Represents a code repository in the graph.

    Attributes:
        id: Unique repository identifier.
        name: Repository name.
        url: Repository URL (GitHub, GitLab, etc.).
        default_branch: Default branch name (main, master, etc.).
        last_indexed: Timestamp of last indexing operation.
    """

    node_type: NodeType = Field(default=NodeType.REPOSITORY, description="Node type")
    name: str = Field(..., description="Repository name")
    url: str | None = Field(None, description="Repository URL")
    default_branch: str = Field(default="main", description="Default branch name")
    last_indexed: datetime | None = Field(None, description="Last indexing timestamp")


class FileNode(BaseNode):
    """Model for File nodes.

    Represents a source file in the repository.

    Attributes:
        id: Unique file identifier (typically the path).
        path: Relative path within the repository.
        language: Programming language of the file.
        hash: Content hash for change detection.
        size: File size in bytes.
        repo_id: ID of the containing repository.
    """

    node_type: NodeType = Field(default=NodeType.FILE, description="Node type")
    path: str = Field(..., description="Relative file path")
    language: str = Field(..., description="Programming language")
    hash: str | None = Field(None, description="Content hash for change detection")
    size: int | None = Field(None, ge=0, description="File size in bytes")
    repo_id: str = Field(..., description="ID of containing repository")


class ModuleNode(BaseNode):
    """Model for Module nodes.

    Represents a Python module or similar language construct.

    Attributes:
        id: Unique module identifier.
        name: Module name.
        file_path: Path to the file containing this module.
        docstring: Module-level docstring if present.
        repo_id: ID of the containing repository.
    """

    node_type: NodeType = Field(default=NodeType.MODULE, description="Node type")
    name: str = Field(..., description="Module name")
    file_path: str = Field(..., description="Path to containing file")
    docstring: str | None = Field(None, description="Module docstring")
    repo_id: str = Field(..., description="ID of containing repository")


class ClassNode(BaseNode):
    """Model for Class nodes.

    Represents a class definition in the codebase.

    Attributes:
        id: Unique class identifier.
        name: Class name.
        file_path: Path to the file containing this class.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        docstring: Class docstring if present.
        bases: List of base class names.
        decorators: List of decorator names.
        repo_id: ID of the containing repository.
    """

    node_type: NodeType = Field(default=NodeType.CLASS, description="Node type")
    name: str = Field(..., description="Class name")
    file_path: str = Field(..., description="Path to containing file")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    docstring: str | None = Field(None, description="Class docstring")
    bases: list[str] = Field(default_factory=list, description="Base class names")
    decorators: list[str] = Field(default_factory=list, description="Decorator names")
    repo_id: str = Field(..., description="ID of containing repository")


class FunctionNode(BaseNode):
    """Model for Function nodes.

    Represents a standalone function definition.

    Attributes:
        id: Unique function identifier.
        name: Function name.
        file_path: Path to the file containing this function.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        is_async: Whether the function is async.
        docstring: Function docstring if present.
        parameters: List of parameter names.
        return_type: Return type annotation if present.
        decorators: List of decorator names.
        repo_id: ID of the containing repository.
    """

    node_type: NodeType = Field(default=NodeType.FUNCTION, description="Node type")
    name: str = Field(..., description="Function name")
    file_path: str = Field(..., description="Path to containing file")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    is_async: bool = Field(default=False, description="Whether function is async")
    docstring: str | None = Field(None, description="Function docstring")
    parameters: list[str] = Field(default_factory=list, description="Parameter names")
    return_type: str | None = Field(None, description="Return type annotation")
    decorators: list[str] = Field(default_factory=list, description="Decorator names")
    repo_id: str = Field(..., description="ID of containing repository")


class MethodNode(BaseNode):
    """Model for Method nodes.

    Represents a method within a class.

    Attributes:
        id: Unique method identifier.
        name: Method name.
        file_path: Path to the file containing this method.
        start_line: Starting line number (1-indexed).
        end_line: Ending line number (1-indexed).
        is_async: Whether the method is async.
        docstring: Method docstring if present.
        parameters: List of parameter names.
        return_type: Return type annotation if present.
        decorators: List of decorator names.
        is_classmethod: Whether decorated with @classmethod.
        is_staticmethod: Whether decorated with @staticmethod.
        is_property: Whether decorated with @property.
        class_id: ID of the containing class.
        repo_id: ID of the containing repository.
    """

    node_type: NodeType = Field(default=NodeType.METHOD, description="Node type")
    name: str = Field(..., description="Method name")
    file_path: str = Field(..., description="Path to containing file")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    is_async: bool = Field(default=False, description="Whether method is async")
    docstring: str | None = Field(None, description="Method docstring")
    parameters: list[str] = Field(default_factory=list, description="Parameter names")
    return_type: str | None = Field(None, description="Return type annotation")
    decorators: list[str] = Field(default_factory=list, description="Decorator names")
    is_classmethod: bool = Field(default=False, description="Whether @classmethod")
    is_staticmethod: bool = Field(default=False, description="Whether @staticmethod")
    is_property: bool = Field(default=False, description="Whether @property")
    class_id: str | None = Field(None, description="ID of containing class")
    repo_id: str = Field(..., description="ID of containing repository")


class VariableNode(BaseNode):
    """Model for Variable nodes.

    Represents a variable definition (module-level or class attribute).

    Attributes:
        id: Unique variable identifier.
        name: Variable name.
        file_path: Path to the file containing this variable.
        line: Line number where variable is defined.
        type_annotation: Type annotation if present.
        is_constant: Whether this appears to be a constant.
        repo_id: ID of the containing repository.
    """

    node_type: NodeType = Field(default=NodeType.VARIABLE, description="Node type")
    name: str = Field(..., description="Variable name")
    file_path: str = Field(..., description="Path to containing file")
    line: int = Field(..., ge=1, description="Line number of definition")
    type_annotation: str | None = Field(None, description="Type annotation")
    is_constant: bool = Field(default=False, description="Whether appears constant")
    repo_id: str = Field(..., description="ID of containing repository")


class GraphRelationship(BaseModel):
    """Model for relationships between nodes.

    Attributes:
        source_id: ID of the source node.
        target_id: ID of the target node.
        relationship_type: Type of the relationship.
        properties: Optional properties on the relationship.
    """

    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship_type: RelationshipType = Field(..., description="Relationship type")
    properties: dict[str, Any] = Field(default_factory=dict, description="Relationship properties")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class GraphPath(BaseModel):
    """Model for a path in the graph.

    Represents a sequence of nodes and relationships forming a path.

    Attributes:
        nodes: List of node IDs in the path.
        relationships: List of relationship types between consecutive nodes.
        length: Number of relationships in the path.
    """

    nodes: list[str] = Field(..., description="Node IDs in order")
    relationships: list[RelationshipType] = Field(
        ..., description="Relationship types between nodes"
    )
    length: int = Field(..., ge=0, description="Path length")


class ImpactResult(BaseModel):
    """Model for impact analysis results.

    Represents the potential impact of changing a code entity.

    Attributes:
        source_id: ID of the modified entity.
        affected_nodes: List of affected node IDs.
        affected_files: Set of affected file paths.
        depth: Maximum depth of impact.
        total_affected: Total number of affected entities.
    """

    source_id: str = Field(..., description="ID of modified entity")
    affected_nodes: list[str] = Field(..., description="Affected node IDs")
    affected_files: list[str] = Field(..., description="Affected file paths")
    depth: int = Field(..., ge=0, description="Maximum impact depth")
    total_affected: int = Field(..., ge=0, description="Total affected count")


# Type alias for any node type
GraphNode = (
    RepositoryNode | FileNode | ModuleNode | ClassNode | FunctionNode | MethodNode | VariableNode
)
