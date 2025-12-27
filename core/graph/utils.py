"""Utility functions for graph operations.

This module provides helper functions for converting parser entities to graph
nodes, sanitizing Cypher parameters, and parsing Neo4j results.
"""

import re
from collections.abc import Sequence
from typing import Any

from core.parser.models import (
    ClassEntity,
    CodeEntity,
    EntityType,
    FunctionEntity,
    MethodEntity,
    ModuleEntity,
    VariableEntity,
)

from .models import (
    BaseNode,
    ClassNode,
    FileNode,
    FunctionNode,
    GraphNode,
    GraphRelationship,
    MethodNode,
    ModuleNode,
    NodeType,
    RelationshipType,
    RepositoryNode,
    VariableNode,
)


def generate_node_id(
    file_path: str,
    name: str,
    start_line: int | None = None,
    entity_type: str | None = None,
) -> str:
    """Generate a unique node ID.

    Creates a unique identifier in the format: {file_path}:{name}:{start_line}
    or {file_path}:{name} if start_line is not provided.

    Args:
        file_path: Path to the containing file.
        name: Name of the entity.
        start_line: Starting line number (optional).
        entity_type: Type of entity for disambiguation (optional).

    Returns:
        A unique identifier string.
    """
    if start_line is not None:
        return f"{file_path}:{name}:{start_line}"
    elif entity_type:
        return f"{file_path}:{entity_type}:{name}"
    return f"{file_path}:{name}"


def sanitize_cypher_string(value: str) -> str:
    """Sanitize a string for safe use in Cypher queries.

    Escapes special characters that could cause injection issues.
    Note: Always prefer parameterized queries over string sanitization.

    Args:
        value: The string to sanitize.

    Returns:
        Sanitized string safe for Cypher.
    """
    if not isinstance(value, str):
        return str(value)

    # Escape backslashes first, then single quotes
    value = value.replace("\\", "\\\\")
    value = value.replace("'", "\\'")
    value = value.replace('"', '\\"')
    return value


def sanitize_identifier(value: str) -> str:
    """Sanitize a string for use as a Cypher identifier.

    Removes or replaces characters that are invalid in Neo4j identifiers.

    Args:
        value: The identifier to sanitize.

    Returns:
        Sanitized identifier string.
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def entity_to_graph_node(entity: CodeEntity, repo_id: str) -> GraphNode | None:
    """Convert a parser entity to a graph node.

    Maps parser model types to their corresponding graph node types.

    Args:
        entity: The parsed code entity.
        repo_id: The repository ID.

    Returns:
        A graph node model, or None if conversion is not supported.
    """
    if isinstance(entity, FunctionEntity):
        return function_entity_to_node(entity, repo_id)
    elif isinstance(entity, MethodEntity):
        return method_entity_to_node(entity, repo_id)
    elif isinstance(entity, ClassEntity):
        return class_entity_to_node(entity, repo_id)
    elif isinstance(entity, VariableEntity):
        return variable_entity_to_node(entity, repo_id)
    elif isinstance(entity, ModuleEntity):
        return module_entity_to_node(entity, repo_id)
    return None


def function_entity_to_node(entity: FunctionEntity, repo_id: str) -> FunctionNode:
    """Convert a FunctionEntity to a FunctionNode.

    Args:
        entity: The parsed function entity.
        repo_id: The repository ID.

    Returns:
        A FunctionNode for graph storage.
    """
    return FunctionNode(
        id=entity.id,
        name=entity.name,
        file_path=entity.file_path,
        start_line=entity.start_line,
        end_line=entity.end_line,
        is_async=entity.is_async,
        docstring=entity.docstring,
        parameters=[p.name for p in entity.parameters],
        return_type=entity.return_type,
        decorators=entity.decorators,
        repo_id=repo_id,
    )


def method_entity_to_node(
    entity: MethodEntity, repo_id: str, class_id: str | None = None
) -> MethodNode:
    """Convert a MethodEntity to a MethodNode.

    Args:
        entity: The parsed method entity.
        repo_id: The repository ID.
        class_id: The ID of the containing class.

    Returns:
        A MethodNode for graph storage.
    """
    return MethodNode(
        id=entity.id,
        name=entity.name,
        file_path=entity.file_path,
        start_line=entity.start_line,
        end_line=entity.end_line,
        is_async=entity.is_async,
        docstring=entity.docstring,
        parameters=[p.name for p in entity.parameters],
        return_type=entity.return_type,
        decorators=entity.decorators,
        is_classmethod=entity.is_classmethod,
        is_staticmethod=entity.is_staticmethod,
        is_property=entity.is_property,
        class_id=class_id or entity.parent_id,
        repo_id=repo_id,
    )


def class_entity_to_node(entity: ClassEntity, repo_id: str) -> ClassNode:
    """Convert a ClassEntity to a ClassNode.

    Args:
        entity: The parsed class entity.
        repo_id: The repository ID.

    Returns:
        A ClassNode for graph storage.
    """
    return ClassNode(
        id=entity.id,
        name=entity.name,
        file_path=entity.file_path,
        start_line=entity.start_line,
        end_line=entity.end_line,
        docstring=entity.docstring,
        bases=entity.bases,
        decorators=entity.decorators,
        repo_id=repo_id,
    )


def variable_entity_to_node(entity: VariableEntity, repo_id: str) -> VariableNode:
    """Convert a VariableEntity to a VariableNode.

    Args:
        entity: The parsed variable entity.
        repo_id: The repository ID.

    Returns:
        A VariableNode for graph storage.
    """
    return VariableNode(
        id=entity.id,
        name=entity.name,
        file_path=entity.file_path,
        line=entity.start_line,
        type_annotation=entity.type_annotation,
        is_constant=entity.is_constant,
        repo_id=repo_id,
    )


def module_entity_to_node(entity: ModuleEntity, repo_id: str) -> ModuleNode:
    """Convert a ModuleEntity to a ModuleNode.

    Args:
        entity: The parsed module entity.
        repo_id: The repository ID.

    Returns:
        A ModuleNode for graph storage.
    """
    return ModuleNode(
        id=entity.id,
        name=entity.name,
        file_path=entity.file_path,
        docstring=entity.docstring,
        repo_id=repo_id,
    )


def parse_neo4j_node(record: dict[str, Any], node_key: str = "n") -> dict[str, Any]:
    """Parse a Neo4j node record into a dictionary.

    Extracts node properties and labels from a Neo4j result record.

    Args:
        record: The Neo4j result record.
        node_key: The key for the node in the record.

    Returns:
        Dictionary with node properties and labels.
    """
    if node_key not in record:
        return {}

    node = record[node_key]
    if node is None:
        return {}

    # Handle neo4j Node type
    if hasattr(node, "items"):
        properties = dict(node.items())
    elif isinstance(node, dict):
        properties = node
    else:
        return {}

    # Add labels if available
    if "labels" in record:
        properties["_labels"] = record["labels"]
    elif hasattr(node, "labels"):
        properties["_labels"] = list(node.labels)

    return properties


def record_to_node(record: dict[str, Any], node_key: str = "n") -> GraphNode | None:
    """Convert a Neo4j record to a typed graph node.

    Determines the node type from labels and creates the appropriate model.

    Args:
        record: The Neo4j result record.
        node_key: The key for the node in the record.

    Returns:
        A typed graph node, or None if conversion fails.
    """
    props = parse_neo4j_node(record, node_key)
    if not props:
        return None

    labels = props.pop("_labels", [])
    if not labels:
        return None

    # Determine node type from labels
    label = labels[0] if labels else None

    try:
        if label == "Repository":
            return RepositoryNode(**props)
        elif label == "File":
            return FileNode(**props)
        elif label == "Module":
            return ModuleNode(**props)
        elif label == "Class":
            return ClassNode(**props)
        elif label == "Function":
            return FunctionNode(**props)
        elif label == "Method":
            return MethodNode(**props)
        elif label == "Variable":
            return VariableNode(**props)
    except Exception:
        return None

    return None


def node_to_cypher_properties(node: BaseNode) -> dict[str, Any]:
    """Convert a graph node to Cypher-safe properties.

    Prepares node properties for use in Cypher queries.

    Args:
        node: The graph node to convert.

    Returns:
        Dictionary of Cypher-safe properties.
    """
    return node.to_properties()


def extract_calls_from_entity(
    entity: FunctionEntity | MethodEntity,
    repo_id: str,
    file_entities: dict[str, CodeEntity],
) -> list[GraphRelationship]:
    """Extract CALLS relationships from a function or method entity.

    Resolves call references to actual entity IDs where possible.

    Args:
        entity: The function or method entity.
        repo_id: The repository ID.
        file_entities: Dictionary of all entities in the file, keyed by name.

    Returns:
        List of GraphRelationship objects representing call relationships.
    """
    relationships = []

    for call_name in entity.calls:
        # Try to resolve the call to an entity in the same file
        if call_name in file_entities:
            target = file_entities[call_name]
            relationships.append(
                GraphRelationship(
                    source_id=entity.id,
                    target_id=target.id,
                    relationship_type=RelationshipType.CALLS,
                )
            )

    return relationships


def extract_relationships_from_module(
    module: ModuleEntity,
    repo_id: str,
) -> list[GraphRelationship]:
    """Extract all relationships from a module entity.

    Creates DEFINES, HAS_METHOD, and other relationships from a parsed module.

    Args:
        module: The parsed module entity.
        repo_id: The repository ID.

    Returns:
        List of all GraphRelationship objects for the module.
    """
    relationships = []
    file_id = f"{module.file_path}:file"

    # File DEFINES functions
    for func in module.functions:
        relationships.append(
            GraphRelationship(
                source_id=file_id,
                target_id=func.id,
                relationship_type=RelationshipType.DEFINES,
            )
        )

    # File DEFINES classes
    for cls in module.classes:
        relationships.append(
            GraphRelationship(
                source_id=file_id,
                target_id=cls.id,
                relationship_type=RelationshipType.DEFINES,
            )
        )

        # Class HAS_METHOD methods
        for method in cls.methods:
            relationships.append(
                GraphRelationship(
                    source_id=cls.id,
                    target_id=method.id,
                    relationship_type=RelationshipType.HAS_METHOD,
                )
            )

    # File DEFINES variables
    for var in module.variables:
        relationships.append(
            GraphRelationship(
                source_id=file_id,
                target_id=var.id,
                relationship_type=RelationshipType.DEFINES,
            )
        )

    return relationships


def batch_nodes(nodes: Sequence[GraphNode], batch_size: int = 500) -> list[Sequence[GraphNode]]:
    """Split a list of nodes into batches for efficient bulk operations.

    Args:
        nodes: List of nodes to batch.
        batch_size: Maximum nodes per batch.

    Returns:
        List of node batches.
    """
    return [nodes[i : i + batch_size] for i in range(0, len(nodes), batch_size)]


def batch_relationships(
    relationships: list[GraphRelationship], batch_size: int = 500
) -> list[list[GraphRelationship]]:
    """Split a list of relationships into batches for efficient bulk operations.

    Args:
        relationships: List of relationships to batch.
        batch_size: Maximum relationships per batch.

    Returns:
        List of relationship batches.
    """
    return [relationships[i : i + batch_size] for i in range(0, len(relationships), batch_size)]


def node_type_to_label(node_type: NodeType) -> str:
    """Convert a NodeType enum to a Neo4j label string.

    Args:
        node_type: The node type enum.

    Returns:
        The corresponding Neo4j label string.
    """
    return node_type.value


def entity_type_to_node_type(entity_type: EntityType) -> NodeType | None:
    """Convert a parser EntityType to a graph NodeType.

    Args:
        entity_type: The parser entity type.

    Returns:
        The corresponding graph node type, or None if not supported.
    """
    mapping = {
        EntityType.FUNCTION: NodeType.FUNCTION,
        EntityType.CLASS: NodeType.CLASS,
        EntityType.METHOD: NodeType.METHOD,
        EntityType.VARIABLE: NodeType.VARIABLE,
        EntityType.MODULE: NodeType.MODULE,
    }
    return mapping.get(entity_type)
