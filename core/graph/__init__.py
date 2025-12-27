"""Graph module for Neo4j operations.

This module provides all the components needed for interacting with the
Neo4j graph database, including connection management, data models,
query templates, and the main GraphStore class.

Example usage:
    ```python
    from core.graph import GraphConnection, GraphStore, RepositoryNode

    async with GraphConnection() as conn:
        store = GraphStore(conn)

        # Create a repository
        repo = RepositoryNode(
            id="my-repo",
            name="My Repository",
            url="https://github.com/user/my-repo",
        )
        await store.create_repository(repo)

        # Find callers of a function
        callers = await store.find_callers("path/to/file.py:my_function:10")
    ```
"""

from .connection import GraphConnection, GraphConnectionError
from .models import (
    BaseNode,
    ClassNode,
    FileNode,
    FunctionNode,
    GraphNode,
    GraphPath,
    GraphRelationship,
    ImpactResult,
    MethodNode,
    ModuleNode,
    NodeType,
    RelationshipType,
    RepositoryNode,
    VariableNode,
)
from .queries import QUERIES, CypherQueries
from .store import GraphStore, GraphStoreError
from .utils import (
    batch_nodes,
    batch_relationships,
    class_entity_to_node,
    entity_to_graph_node,
    entity_type_to_node_type,
    extract_relationships_from_module,
    function_entity_to_node,
    generate_node_id,
    method_entity_to_node,
    module_entity_to_node,
    node_to_cypher_properties,
    node_type_to_label,
    parse_neo4j_node,
    record_to_node,
    sanitize_cypher_string,
    sanitize_identifier,
    variable_entity_to_node,
)

__all__ = [
    # Connection
    "GraphConnection",
    "GraphConnectionError",
    # Store
    "GraphStore",
    "GraphStoreError",
    # Models - Node Types
    "BaseNode",
    "RepositoryNode",
    "FileNode",
    "ModuleNode",
    "ClassNode",
    "FunctionNode",
    "MethodNode",
    "VariableNode",
    "GraphNode",
    # Models - Relationships
    "GraphRelationship",
    "RelationshipType",
    # Models - Other
    "NodeType",
    "GraphPath",
    "ImpactResult",
    # Queries
    "QUERIES",
    "CypherQueries",
    # Utils
    "generate_node_id",
    "sanitize_cypher_string",
    "sanitize_identifier",
    "entity_to_graph_node",
    "function_entity_to_node",
    "method_entity_to_node",
    "class_entity_to_node",
    "variable_entity_to_node",
    "module_entity_to_node",
    "parse_neo4j_node",
    "record_to_node",
    "node_to_cypher_properties",
    "extract_relationships_from_module",
    "batch_nodes",
    "batch_relationships",
    "node_type_to_label",
    "entity_type_to_node_type",
]
