"""Main GraphStore class for Neo4j operations.

This module provides the GraphStore class which handles all CRUD operations
for code entities in the Neo4j graph database. It supports batch operations,
transactions, and various graph traversals.
"""

from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
from neo4j.exceptions import Neo4jError

from .connection import GraphConnection
from .models import (
    ClassNode,
    FileNode,
    FunctionNode,
    GraphNode,
    GraphPath,
    GraphRelationship,
    ImpactResult,
    MethodNode,
    ModuleNode,
    RelationshipType,
    RepositoryNode,
    VariableNode,
)
from .queries import QUERIES
from .utils import (
    batch_nodes,
    batch_relationships,
    node_to_cypher_properties,
    parse_neo4j_node,
    record_to_node,
)


class GraphStoreError(Exception):
    """Exception raised for graph store operation errors."""

    pass


class GraphStore:
    """Main class for Neo4j graph operations.

    Provides CRUD operations, batch inserts, relationship management,
    and various graph traversal methods for code analysis.

    Attributes:
        connection: The Neo4j connection instance.
    """

    def __init__(self, connection: GraphConnection) -> None:
        """Initialize the graph store.

        Args:
            connection: An established GraphConnection instance.
        """
        self.connection = connection

    # ==========================================================================
    # Repository Operations
    # ==========================================================================

    async def create_repository(self, repo: RepositoryNode) -> RepositoryNode:
        """Create or update a repository node.

        Args:
            repo: The repository node to create.

        Returns:
            The created repository node.

        Raises:
            GraphStoreError: If creation fails.
        """
        try:
            params = node_to_cypher_properties(repo)
            await self.connection.execute_write(QUERIES.CREATE_REPOSITORY, params)
            logger.debug(f"Created repository: {repo.id}")
            return repo
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create repository: {e}") from e

    async def get_repository(self, repo_id: str) -> RepositoryNode | None:
        """Get a repository by ID.

        Args:
            repo_id: The repository ID.

        Returns:
            The repository node, or None if not found.
        """
        try:
            result = await self.connection.execute_read(QUERIES.GET_REPOSITORY, {"id": repo_id})
            if result and result[0].get("r"):
                props = parse_neo4j_node(result[0], "r")
                return RepositoryNode(**props) if props else None
            return None
        except Neo4jError as e:
            logger.warning(f"Failed to get repository {repo_id}: {e}")
            return None

    async def update_repository_indexed(
        self, repo_id: str, indexed_at: datetime | None = None
    ) -> None:
        """Update the last_indexed timestamp of a repository.

        Args:
            repo_id: The repository ID.
            indexed_at: The timestamp (defaults to now).
        """
        timestamp = indexed_at or datetime.utcnow()
        query = """
            MATCH (r:Repository {id: $id})
            SET r.last_indexed = $last_indexed
        """
        await self.connection.execute_write(
            query, {"id": repo_id, "last_indexed": timestamp.isoformat()}
        )

    # ==========================================================================
    # File Operations
    # ==========================================================================

    async def create_file(self, file: FileNode) -> FileNode:
        """Create or update a file node.

        Args:
            file: The file node to create.

        Returns:
            The created file node.
        """
        try:
            params = node_to_cypher_properties(file)
            await self.connection.execute_write(QUERIES.CREATE_FILE, params)
            logger.debug(f"Created file: {file.path}")
            return file
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create file: {e}") from e

    async def get_file(self, file_id: str) -> FileNode | None:
        """Get a file by ID.

        Args:
            file_id: The file ID.

        Returns:
            The file node, or None if not found.
        """
        try:
            result = await self.connection.execute_read(QUERIES.GET_FILE, {"id": file_id})
            if result and result[0].get("f"):
                props = parse_neo4j_node(result[0], "f")
                return FileNode(**props) if props else None
            return None
        except Neo4jError:
            return None

    async def get_files_in_repository(self, repo_id: str) -> list[FileNode]:
        """Get all files in a repository.

        Args:
            repo_id: The repository ID.

        Returns:
            List of file nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.GET_FILES_IN_REPO, {"repo_id": repo_id}
            )
            files = []
            for record in result:
                if record.get("f"):
                    props = parse_neo4j_node(record, "f")
                    if props:
                        files.append(FileNode(**props))
            return files
        except Neo4jError as e:
            logger.warning(f"Failed to get files for repo {repo_id}: {e}")
            return []

    async def batch_create_files(self, files: list[FileNode]) -> int:
        """Create multiple file nodes in batch.

        Args:
            files: List of file nodes to create.

        Returns:
            Number of files created.
        """
        if not files:
            return 0

        total_created = 0
        for batch in batch_nodes(files):
            file_props = [node_to_cypher_properties(f) for f in batch]
            try:
                result = await self.connection.execute_write(
                    QUERIES.BATCH_CREATE_FILES, {"files": file_props}
                )
                if result:
                    total_created += result[0].get("created", 0)
            except Neo4jError as e:
                logger.warning(f"Batch file creation error: {e}")

        logger.info(f"Batch created {total_created} files")
        return total_created

    # ==========================================================================
    # Function Operations
    # ==========================================================================

    async def create_function(self, func: FunctionNode) -> FunctionNode:
        """Create or update a function node.

        Args:
            func: The function node to create.

        Returns:
            The created function node.
        """
        try:
            params = node_to_cypher_properties(func)
            await self.connection.execute_write(QUERIES.CREATE_FUNCTION, params)
            logger.debug(f"Created function: {func.name}")
            return func
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create function: {e}") from e

    async def get_function(self, function_id: str) -> FunctionNode | None:
        """Get a function by ID.

        Args:
            function_id: The function ID.

        Returns:
            The function node, or None if not found.
        """
        try:
            result = await self.connection.execute_read(QUERIES.GET_FUNCTION, {"id": function_id})
            if result and result[0].get("f"):
                props = parse_neo4j_node(result[0], "f")
                return FunctionNode(**props) if props else None
            return None
        except Neo4jError:
            return None

    async def find_function_by_name(self, name: str, repo_id: str) -> list[FunctionNode]:
        """Find functions by name in a repository.

        Args:
            name: The function name.
            repo_id: The repository ID.

        Returns:
            List of matching function nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.GET_FUNCTION_BY_NAME, {"name": name, "repo_id": repo_id}
            )
            functions = []
            for record in result:
                if record.get("f"):
                    props = parse_neo4j_node(record, "f")
                    if props:
                        functions.append(FunctionNode(**props))
            return functions
        except Neo4jError:
            return []

    async def batch_create_functions(self, functions: list[FunctionNode]) -> int:
        """Create multiple function nodes in batch.

        Args:
            functions: List of function nodes to create.

        Returns:
            Number of functions created.
        """
        if not functions:
            return 0

        total_created = 0
        for batch in batch_nodes(functions):
            func_props = [node_to_cypher_properties(f) for f in batch]
            try:
                result = await self.connection.execute_write(
                    QUERIES.BATCH_CREATE_FUNCTIONS, {"functions": func_props}
                )
                if result:
                    total_created += result[0].get("created", 0)
            except Neo4jError as e:
                logger.warning(f"Batch function creation error: {e}")

        logger.info(f"Batch created {total_created} functions")
        return total_created

    # ==========================================================================
    # Class Operations
    # ==========================================================================

    async def create_class(self, cls: ClassNode) -> ClassNode:
        """Create or update a class node.

        Args:
            cls: The class node to create.

        Returns:
            The created class node.
        """
        try:
            params = node_to_cypher_properties(cls)
            await self.connection.execute_write(QUERIES.CREATE_CLASS, params)
            logger.debug(f"Created class: {cls.name}")
            return cls
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create class: {e}") from e

    async def get_class(self, class_id: str) -> ClassNode | None:
        """Get a class by ID.

        Args:
            class_id: The class ID.

        Returns:
            The class node, or None if not found.
        """
        try:
            result = await self.connection.execute_read(QUERIES.GET_CLASS, {"id": class_id})
            if result and result[0].get("c"):
                props = parse_neo4j_node(result[0], "c")
                return ClassNode(**props) if props else None
            return None
        except Neo4jError:
            return None

    async def find_class_by_name(self, name: str, repo_id: str) -> list[ClassNode]:
        """Find classes by name in a repository.

        Args:
            name: The class name.
            repo_id: The repository ID.

        Returns:
            List of matching class nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.GET_CLASS_BY_NAME, {"name": name, "repo_id": repo_id}
            )
            classes = []
            for record in result:
                if record.get("c"):
                    props = parse_neo4j_node(record, "c")
                    if props:
                        classes.append(ClassNode(**props))
            return classes
        except Neo4jError:
            return []

    async def batch_create_classes(self, classes: list[ClassNode]) -> int:
        """Create multiple class nodes in batch.

        Args:
            classes: List of class nodes to create.

        Returns:
            Number of classes created.
        """
        if not classes:
            return 0

        total_created = 0
        for batch in batch_nodes(classes):
            class_props = [node_to_cypher_properties(c) for c in batch]
            try:
                result = await self.connection.execute_write(
                    QUERIES.BATCH_CREATE_CLASSES, {"classes": class_props}
                )
                if result:
                    total_created += result[0].get("created", 0)
            except Neo4jError as e:
                logger.warning(f"Batch class creation error: {e}")

        logger.info(f"Batch created {total_created} classes")
        return total_created

    # ==========================================================================
    # Method Operations
    # ==========================================================================

    async def create_method(self, method: MethodNode) -> MethodNode:
        """Create or update a method node.

        Args:
            method: The method node to create.

        Returns:
            The created method node.
        """
        try:
            params = node_to_cypher_properties(method)
            await self.connection.execute_write(QUERIES.CREATE_METHOD, params)
            logger.debug(f"Created method: {method.name}")
            return method
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create method: {e}") from e

    async def get_method(self, method_id: str) -> MethodNode | None:
        """Get a method by ID.

        Args:
            method_id: The method ID.

        Returns:
            The method node, or None if not found.
        """
        try:
            result = await self.connection.execute_read(QUERIES.GET_METHOD, {"id": method_id})
            if result and result[0].get("m"):
                props = parse_neo4j_node(result[0], "m")
                return MethodNode(**props) if props else None
            return None
        except Neo4jError:
            return None

    async def get_methods_of_class(self, class_id: str) -> list[MethodNode]:
        """Get all methods of a class.

        Args:
            class_id: The class ID.

        Returns:
            List of method nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.GET_METHODS_OF_CLASS, {"class_id": class_id}
            )
            methods = []
            for record in result:
                if record.get("m"):
                    props = parse_neo4j_node(record, "m")
                    if props:
                        methods.append(MethodNode(**props))
            return methods
        except Neo4jError:
            return []

    async def batch_create_methods(self, methods: list[MethodNode]) -> int:
        """Create multiple method nodes in batch.

        Args:
            methods: List of method nodes to create.

        Returns:
            Number of methods created.
        """
        if not methods:
            return 0

        total_created = 0
        for batch in batch_nodes(methods):
            method_props = [node_to_cypher_properties(m) for m in batch]
            try:
                result = await self.connection.execute_write(
                    QUERIES.BATCH_CREATE_METHODS, {"methods": method_props}
                )
                if result:
                    total_created += result[0].get("created", 0)
            except Neo4jError as e:
                logger.warning(f"Batch method creation error: {e}")

        logger.info(f"Batch created {total_created} methods")
        return total_created

    # ==========================================================================
    # Variable Operations
    # ==========================================================================

    async def create_variable(self, var: VariableNode) -> VariableNode:
        """Create or update a variable node.

        Args:
            var: The variable node to create.

        Returns:
            The created variable node.
        """
        try:
            params = node_to_cypher_properties(var)
            await self.connection.execute_write(QUERIES.CREATE_VARIABLE, params)
            logger.debug(f"Created variable: {var.name}")
            return var
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create variable: {e}") from e

    async def batch_create_variables(self, variables: list[VariableNode]) -> int:
        """Create multiple variable nodes in batch.

        Args:
            variables: List of variable nodes to create.

        Returns:
            Number of variables created.
        """
        if not variables:
            return 0

        total_created = 0
        for batch in batch_nodes(variables):
            var_props = [node_to_cypher_properties(v) for v in batch]
            try:
                result = await self.connection.execute_write(
                    QUERIES.BATCH_CREATE_VARIABLES, {"variables": var_props}
                )
                if result:
                    total_created += result[0].get("created", 0)
            except Neo4jError as e:
                logger.warning(f"Batch variable creation error: {e}")

        logger.info(f"Batch created {total_created} variables")
        return total_created

    # ==========================================================================
    # Module Operations
    # ==========================================================================

    async def create_module(self, module: ModuleNode) -> ModuleNode:
        """Create or update a module node.

        Args:
            module: The module node to create.

        Returns:
            The created module node.
        """
        try:
            params = node_to_cypher_properties(module)
            await self.connection.execute_write(QUERIES.CREATE_MODULE, params)
            logger.debug(f"Created module: {module.name}")
            return module
        except Neo4jError as e:
            raise GraphStoreError(f"Failed to create module: {e}") from e

    # ==========================================================================
    # Relationship Operations
    # ==========================================================================

    async def create_relationship(self, relationship: GraphRelationship) -> bool:
        """Create a relationship between two nodes.

        Args:
            relationship: The relationship to create.

        Returns:
            True if created successfully.
        """
        query_map = {
            RelationshipType.CONTAINS: QUERIES.CREATE_CONTAINS,
            RelationshipType.DEFINES: QUERIES.CREATE_DEFINES,
            RelationshipType.HAS_METHOD: QUERIES.CREATE_HAS_METHOD,
            RelationshipType.INHERITS: QUERIES.CREATE_INHERITS,
            RelationshipType.CALLS: QUERIES.CREATE_CALLS,
            RelationshipType.USES: QUERIES.CREATE_USES,
            RelationshipType.IMPORTS: QUERIES.CREATE_IMPORTS,
        }

        query = query_map.get(relationship.relationship_type)
        if not query:
            logger.warning(f"Unsupported relationship type: {relationship.relationship_type}")
            return False

        try:
            await self.connection.execute_write(
                query,
                {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                },
            )
            return True
        except Neo4jError as e:
            logger.warning(f"Failed to create relationship: {e}")
            return False

    async def batch_create_relationships(self, relationships: list[GraphRelationship]) -> int:
        """Create multiple relationships in batch.

        Args:
            relationships: List of relationships to create.

        Returns:
            Number of relationships created.
        """
        if not relationships:
            return 0

        created = 0
        for batch in batch_relationships(relationships):
            for rel in batch:
                if await self.create_relationship(rel):
                    created += 1

        logger.info(f"Created {created} relationships")
        return created

    async def batch_create_call_relationships(self, calls: list[tuple[str, str]]) -> int:
        """Create multiple CALLS relationships in batch.

        More efficient than batch_create_relationships for CALLS only.

        Args:
            calls: List of (caller_id, callee_id) tuples.

        Returns:
            Number of relationships created.
        """
        if not calls:
            return 0

        total_created = 0
        batch_size = 500

        for i in range(0, len(calls), batch_size):
            batch = calls[i : i + batch_size]
            call_props = [{"caller_id": c[0], "callee_id": c[1]} for c in batch]
            try:
                result = await self.connection.execute_write(
                    QUERIES.BATCH_CREATE_CALLS, {"calls": call_props}
                )
                if result:
                    total_created += result[0].get("created", 0)
            except Neo4jError as e:
                logger.warning(f"Batch call relationship creation error: {e}")

        logger.info(f"Batch created {total_created} call relationships")
        return total_created

    # ==========================================================================
    # Caller/Callee Analysis
    # ==========================================================================

    async def find_callers(self, target_id: str) -> list[GraphNode]:
        """Find all functions/methods that call a given function/method.

        Args:
            target_id: The ID of the function/method being called.

        Returns:
            List of caller nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.FIND_CALLERS, {"target_id": target_id}
            )
            callers = []
            for record in result:
                node = record_to_node(record, "caller")
                if node:
                    callers.append(node)
            return callers
        except Neo4jError as e:
            logger.warning(f"Failed to find callers for {target_id}: {e}")
            return []

    async def find_callees(self, source_id: str) -> list[GraphNode]:
        """Find all functions/methods called by a given function/method.

        Args:
            source_id: The ID of the calling function/method.

        Returns:
            List of callee nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.FIND_CALLEES, {"source_id": source_id}
            )
            callees = []
            for record in result:
                node = record_to_node(record, "callee")
                if node:
                    callees.append(node)
            return callees
        except Neo4jError as e:
            logger.warning(f"Failed to find callees for {source_id}: {e}")
            return []

    async def find_callers_recursive(
        self, target_id: str, max_depth: int = 5
    ) -> list[tuple[GraphNode, int]]:
        """Find all callers recursively up to a maximum depth.

        Args:
            target_id: The ID of the target function/method.
            max_depth: Maximum depth to traverse.

        Returns:
            List of (caller_node, depth) tuples.
        """
        query = QUERIES.FIND_CALLERS_RECURSIVE.format(depth=max_depth)
        try:
            result = await self.connection.execute_read(query, {"target_id": target_id})
            callers = []
            for record in result:
                node = record_to_node(record, "caller")
                if node:
                    depth = record.get("depth", 1)
                    callers.append((node, depth))
            return callers
        except Neo4jError as e:
            logger.warning(f"Failed to find recursive callers for {target_id}: {e}")
            return []

    # ==========================================================================
    # Call Path Analysis
    # ==========================================================================

    async def find_call_path(self, source_id: str, target_id: str) -> GraphPath | None:
        """Find the shortest call path between two functions/methods.

        Args:
            source_id: The starting function/method ID.
            target_id: The ending function/method ID.

        Returns:
            GraphPath if a path exists, None otherwise.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.FIND_CALL_PATH,
                {"source_id": source_id, "target_id": target_id},
            )
            if not result:
                return None

            record = result[0]
            node_ids = record.get("node_ids", [])
            path_length = record.get("path_length", 0)

            return GraphPath(
                nodes=node_ids,
                relationships=[RelationshipType.CALLS] * path_length,
                length=path_length,
            )
        except Neo4jError as e:
            logger.warning(f"Failed to find call path: {e}")
            return None

    async def find_all_call_paths(
        self, source_id: str, target_id: str, max_depth: int = 5, limit: int = 10
    ) -> list[GraphPath]:
        """Find all call paths between two functions/methods.

        Args:
            source_id: The starting function/method ID.
            target_id: The ending function/method ID.
            max_depth: Maximum path length to consider.
            limit: Maximum number of paths to return.

        Returns:
            List of GraphPath objects.
        """
        query = QUERIES.FIND_ALL_CALL_PATHS.format(max_depth=max_depth)
        try:
            result = await self.connection.execute_read(
                query,
                {"source_id": source_id, "target_id": target_id, "limit": limit},
            )
            paths = []
            for record in result:
                node_ids = record.get("node_ids", [])
                path_length = record.get("path_length", 0)
                paths.append(
                    GraphPath(
                        nodes=node_ids,
                        relationships=[RelationshipType.CALLS] * path_length,
                        length=path_length,
                    )
                )
            return paths
        except Neo4jError as e:
            logger.warning(f"Failed to find all call paths: {e}")
            return []

    # ==========================================================================
    # Impact Analysis
    # ==========================================================================

    async def analyze_impact(self, source_id: str, max_depth: int = 5) -> ImpactResult:
        """Analyze the impact of modifying a function/method.

        Finds all entities that could be affected by a change.

        Args:
            source_id: The ID of the function/method being modified.
            max_depth: Maximum depth to traverse for impact.

        Returns:
            ImpactResult with affected entities.
        """
        query = QUERIES.IMPACT_ANALYSIS_WITH_FILES.format(depth=max_depth)
        try:
            result = await self.connection.execute_read(query, {"source_id": source_id})

            affected_nodes = []
            affected_files = set()
            max_distance = 0

            for record in result:
                node_id = record.get("id")
                file_path = record.get("file_path")
                distance = record.get("distance", 1)

                if node_id:
                    affected_nodes.append(node_id)
                if file_path:
                    affected_files.add(file_path)
                max_distance = max(max_distance, distance)

            return ImpactResult(
                source_id=source_id,
                affected_nodes=affected_nodes,
                affected_files=list(affected_files),
                depth=max_distance,
                total_affected=len(affected_nodes),
            )
        except Neo4jError as e:
            logger.warning(f"Failed to analyze impact for {source_id}: {e}")
            return ImpactResult(
                source_id=source_id,
                affected_nodes=[],
                affected_files=[],
                depth=0,
                total_affected=0,
            )

    # ==========================================================================
    # Function Context
    # ==========================================================================

    async def get_function_context(self, function_id: str) -> dict[str, Any]:
        """Get the full context of a function/method.

        Includes callers, callees, used variables, and containing entities.

        Args:
            function_id: The function/method ID.

        Returns:
            Dictionary with context information.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.FUNCTION_FULL_CONTEXT, {"function_id": function_id}
            )
            if not result:
                return {}

            record = result[0]
            return {
                "function": record.get("function"),
                "callers": record.get("callers", []),
                "callees": record.get("callees", []),
                "variables": record.get("variables", []),
                "containing_file": record.get("containing_file"),
                "containing_class": record.get("containing_class"),
            }
        except Neo4jError as e:
            logger.warning(f"Failed to get function context for {function_id}: {e}")
            return {}

    # ==========================================================================
    # File Dependencies
    # ==========================================================================

    async def get_file_dependencies(self, file_id: str) -> list[GraphNode]:
        """Get all modules/files that a file imports.

        Args:
            file_id: The file ID.

        Returns:
            List of imported module/file nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.FILE_DEPENDENCIES, {"file_id": file_id}
            )
            deps = []
            for record in result:
                node = record_to_node(record, "dep")
                if node:
                    deps.append(node)
            return deps
        except Neo4jError:
            return []

    async def get_file_dependents(self, file_id: str) -> list[FileNode]:
        """Get all files that import a given file.

        Args:
            file_id: The file ID.

        Returns:
            List of dependent file nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.FILE_DEPENDENTS, {"file_id": file_id}
            )
            deps = []
            for record in result:
                if record.get("dependent"):
                    props = parse_neo4j_node(record, "dependent")
                    if props:
                        deps.append(FileNode(**props))
            return deps
        except Neo4jError:
            return []

    # ==========================================================================
    # Class Hierarchy
    # ==========================================================================

    async def get_class_parents(self, class_id: str) -> list[ClassNode]:
        """Get direct parent classes.

        Args:
            class_id: The class ID.

        Returns:
            List of parent class nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.CLASS_PARENTS, {"class_id": class_id}
            )
            parents = []
            for record in result:
                if record.get("parent"):
                    props = parse_neo4j_node(record, "parent")
                    if props:
                        parents.append(ClassNode(**props))
            return parents
        except Neo4jError:
            return []

    async def get_class_ancestors(
        self, class_id: str, max_depth: int = 10
    ) -> list[tuple[ClassNode, int]]:
        """Get all ancestor classes recursively.

        Args:
            class_id: The class ID.
            max_depth: Maximum inheritance depth.

        Returns:
            List of (ancestor, distance) tuples.
        """
        query = QUERIES.CLASS_ANCESTORS.format(depth=max_depth)
        try:
            result = await self.connection.execute_read(query, {"class_id": class_id})
            ancestors = []
            for record in result:
                if record.get("ancestor"):
                    props = parse_neo4j_node(record, "ancestor")
                    if props:
                        distance = record.get("distance", 1)
                        ancestors.append((ClassNode(**props), distance))
            return ancestors
        except Neo4jError:
            return []

    async def get_class_children(self, class_id: str) -> list[ClassNode]:
        """Get direct child classes.

        Args:
            class_id: The class ID.

        Returns:
            List of child class nodes.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.CLASS_CHILDREN, {"class_id": class_id}
            )
            children = []
            for record in result:
                if record.get("child"):
                    props = parse_neo4j_node(record, "child")
                    if props:
                        children.append(ClassNode(**props))
            return children
        except Neo4jError:
            return []

    async def get_class_descendants(
        self, class_id: str, max_depth: int = 10
    ) -> list[tuple[ClassNode, int]]:
        """Get all descendant classes recursively.

        Args:
            class_id: The class ID.
            max_depth: Maximum inheritance depth.

        Returns:
            List of (descendant, distance) tuples.
        """
        query = QUERIES.CLASS_DESCENDANTS.format(depth=max_depth)
        try:
            result = await self.connection.execute_read(query, {"class_id": class_id})
            descendants = []
            for record in result:
                if record.get("descendant"):
                    props = parse_neo4j_node(record, "descendant")
                    if props:
                        distance = record.get("distance", 1)
                        descendants.append((ClassNode(**props), distance))
            return descendants
        except Neo4jError:
            return []

    # ==========================================================================
    # Delete Operations
    # ==========================================================================

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its relationships.

        Args:
            node_id: The node ID to delete.

        Returns:
            True if deleted successfully.
        """
        try:
            await self.connection.execute_write(QUERIES.DELETE_NODE, {"id": node_id})
            logger.debug(f"Deleted node: {node_id}")
            return True
        except Neo4jError as e:
            logger.warning(f"Failed to delete node {node_id}: {e}")
            return False

    async def delete_repository(self, repo_id: str) -> bool:
        """Delete a repository and all related entities.

        Cascades deletion to all files, functions, classes, etc.

        Args:
            repo_id: The repository ID.

        Returns:
            True if deleted successfully.
        """
        try:
            await self.connection.execute_write(
                QUERIES.DELETE_REPOSITORY_CASCADE, {"repo_id": repo_id}
            )
            logger.info(f"Deleted repository and all contents: {repo_id}")
            return True
        except Neo4jError as e:
            logger.warning(f"Failed to delete repository {repo_id}: {e}")
            return False

    async def delete_file(self, file_id: str) -> bool:
        """Delete a file and all its defined entities.

        Args:
            file_id: The file ID.

        Returns:
            True if deleted successfully.
        """
        try:
            await self.connection.execute_write(QUERIES.DELETE_FILE_CASCADE, {"file_id": file_id})
            logger.debug(f"Deleted file and contents: {file_id}")
            return True
        except Neo4jError as e:
            logger.warning(f"Failed to delete file {file_id}: {e}")
            return False

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_repository_statistics(self, repo_id: str) -> dict[str, Any]:
        """Get statistics about a repository.

        Args:
            repo_id: The repository ID.

        Returns:
            Dictionary with node counts by type.
        """
        try:
            result = await self.connection.execute_read(
                QUERIES.REPO_STATISTICS, {"repo_id": repo_id}
            )
            if result:
                record = result[0]
                return {
                    "file_count": record.get("file_count", 0),
                    "function_count": record.get("function_count", 0),
                    "class_count": record.get("class_count", 0),
                    "method_count": record.get("method_count", 0),
                }
            return {}
        except Neo4jError:
            return {}

    # ==========================================================================
    # Generic Node Operations
    # ==========================================================================

    async def get_node_by_id(self, node_id: str) -> GraphNode | None:
        """Get any node by ID.

        Args:
            node_id: The node ID.

        Returns:
            The node, or None if not found.
        """
        try:
            result = await self.connection.execute_read(QUERIES.GET_NODE_BY_ID, {"id": node_id})
            if result:
                return record_to_node(result[0], "n")
            return None
        except Neo4jError:
            return None

    # ==========================================================================
    # Transaction Support
    # ==========================================================================

    async def execute_in_transaction(
        self,
        operations: list[tuple[str, dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        """Execute multiple operations in a single transaction.

        All operations succeed or all fail together.

        Args:
            operations: List of (query, parameters) tuples.

        Returns:
            List of results for each operation.

        Raises:
            GraphStoreError: If transaction fails.
        """
        async with self.connection.session() as session:

            async def _run_tx(tx: Any) -> list[list[dict[str, Any]]]:
                results = []
                for query, params in operations:
                    result = await tx.run(query, params)
                    data = await result.data()
                    results.append(data)
                return results

            try:
                return await session.execute_write(_run_tx)
            except Neo4jError as e:
                raise GraphStoreError(f"Transaction failed: {e}") from e
