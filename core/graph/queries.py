"""Cypher query templates for graph operations.

This module contains predefined Cypher queries for common graph operations
including traversals, impact analysis, and relationship queries.
All queries use parameterized values for security and performance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CypherQueries:
    """Collection of Cypher query templates.

    All queries use parameterized values (prefixed with $) for safety.
    Never concatenate user input directly into queries.
    """

    # ==========================================================================
    # Node Creation Queries
    # ==========================================================================

    CREATE_REPOSITORY = """
        MERGE (r:Repository {id: $id})
        SET r.name = $name,
            r.url = $url,
            r.default_branch = $default_branch,
            r.last_indexed = $last_indexed
        RETURN r
    """

    CREATE_FILE = """
        MERGE (f:File {id: $id})
        SET f.path = $path,
            f.language = $language,
            f.hash = $hash,
            f.size = $size,
            f.repo_id = $repo_id
        RETURN f
    """

    CREATE_MODULE = """
        MERGE (m:Module {id: $id})
        SET m.name = $name,
            m.file_path = $file_path,
            m.docstring = $docstring,
            m.repo_id = $repo_id
        RETURN m
    """

    CREATE_CLASS = """
        MERGE (c:Class {id: $id})
        SET c.name = $name,
            c.file_path = $file_path,
            c.start_line = $start_line,
            c.end_line = $end_line,
            c.docstring = $docstring,
            c.bases = $bases,
            c.decorators = $decorators,
            c.repo_id = $repo_id
        RETURN c
    """

    CREATE_FUNCTION = """
        MERGE (f:Function {id: $id})
        SET f.name = $name,
            f.file_path = $file_path,
            f.start_line = $start_line,
            f.end_line = $end_line,
            f.is_async = $is_async,
            f.docstring = $docstring,
            f.parameters = $parameters,
            f.return_type = $return_type,
            f.decorators = $decorators,
            f.repo_id = $repo_id
        RETURN f
    """

    CREATE_METHOD = """
        MERGE (m:Method {id: $id})
        SET m.name = $name,
            m.file_path = $file_path,
            m.start_line = $start_line,
            m.end_line = $end_line,
            m.is_async = $is_async,
            m.docstring = $docstring,
            m.parameters = $parameters,
            m.return_type = $return_type,
            m.decorators = $decorators,
            m.is_classmethod = $is_classmethod,
            m.is_staticmethod = $is_staticmethod,
            m.is_property = $is_property,
            m.class_id = $class_id,
            m.repo_id = $repo_id
        RETURN m
    """

    CREATE_VARIABLE = """
        MERGE (v:Variable {id: $id})
        SET v.name = $name,
            v.file_path = $file_path,
            v.line = $line,
            v.type_annotation = $type_annotation,
            v.is_constant = $is_constant,
            v.repo_id = $repo_id
        RETURN v
    """

    # ==========================================================================
    # Relationship Creation Queries
    # ==========================================================================

    CREATE_CONTAINS = """
        MATCH (a {id: $source_id}), (b {id: $target_id})
        MERGE (a)-[r:CONTAINS]->(b)
        RETURN r
    """

    CREATE_DEFINES = """
        MATCH (a {id: $source_id}), (b {id: $target_id})
        MERGE (a)-[r:DEFINES]->(b)
        RETURN r
    """

    CREATE_HAS_METHOD = """
        MATCH (a:Class {id: $source_id}), (b:Method {id: $target_id})
        MERGE (a)-[r:HAS_METHOD]->(b)
        RETURN r
    """

    CREATE_INHERITS = """
        MATCH (a:Class {id: $source_id}), (b:Class {id: $target_id})
        MERGE (a)-[r:INHERITS]->(b)
        RETURN r
    """

    CREATE_CALLS = """
        MATCH (a {id: $source_id}), (b {id: $target_id})
        WHERE (a:Function OR a:Method) AND (b:Function OR b:Method)
        MERGE (a)-[r:CALLS]->(b)
        RETURN r
    """

    CREATE_USES = """
        MATCH (a {id: $source_id}), (b:Variable {id: $target_id})
        WHERE (a:Function OR a:Method)
        MERGE (a)-[r:USES]->(b)
        RETURN r
    """

    CREATE_IMPORTS = """
        MATCH (a:File {id: $source_id}), (b {id: $target_id})
        WHERE (b:Module OR b:File)
        MERGE (a)-[r:IMPORTS]->(b)
        RETURN r
    """

    # ==========================================================================
    # Node Retrieval Queries
    # ==========================================================================

    GET_NODE_BY_ID = """
        MATCH (n {id: $id})
        RETURN n, labels(n) as labels
    """

    GET_REPOSITORY = """
        MATCH (r:Repository {id: $id})
        RETURN r
    """

    GET_FILE = """
        MATCH (f:File {id: $id})
        RETURN f
    """

    GET_FILES_IN_REPO = """
        MATCH (f:File {repo_id: $repo_id})
        RETURN f
        ORDER BY f.path
    """

    GET_FUNCTION = """
        MATCH (f:Function {id: $id})
        RETURN f
    """

    GET_FUNCTION_BY_NAME = """
        MATCH (f:Function {name: $name, repo_id: $repo_id})
        RETURN f
    """

    GET_CLASS = """
        MATCH (c:Class {id: $id})
        RETURN c
    """

    GET_CLASS_BY_NAME = """
        MATCH (c:Class {name: $name, repo_id: $repo_id})
        RETURN c
    """

    GET_METHOD = """
        MATCH (m:Method {id: $id})
        RETURN m
    """

    GET_METHODS_OF_CLASS = """
        MATCH (c:Class {id: $class_id})-[:HAS_METHOD]->(m:Method)
        RETURN m
        ORDER BY m.start_line
    """

    # ==========================================================================
    # Caller/Callee Queries
    # ==========================================================================

    FIND_CALLERS = """
        MATCH (caller)-[:CALLS]->(target {id: $target_id})
        WHERE (caller:Function OR caller:Method)
        RETURN caller
        ORDER BY caller.file_path, caller.start_line
    """

    FIND_CALLEES = """
        MATCH (source {{id: $source_id}})-[:CALLS]->(callee)
        WHERE (callee:Function OR callee:Method)
        RETURN callee
        ORDER BY callee.file_path, callee.start_line
    """

    FIND_CALLERS_RECURSIVE = """
        MATCH path = (caller)-[:CALLS*1..{depth}]->(target {{id: $target_id}})
        WHERE (caller:Function OR caller:Method)
        RETURN DISTINCT caller, length(path) as depth
        ORDER BY depth, caller.file_path
    """

    FIND_CALLEES_RECURSIVE = """
        MATCH path = (source {{id: $source_id}})-[:CALLS*1..{depth}]->(callee)
        WHERE (callee:Function OR callee:Method)
        RETURN DISTINCT callee, length(path) as depth
        ORDER BY depth, callee.file_path
    """

    # ==========================================================================
    # Call Path Queries
    # ==========================================================================

    FIND_CALL_PATH = """
        MATCH path = shortestPath(
            (source {{id: $source_id}})-[:CALLS*1..10]->(target {id: $target_id})
        )
        RETURN path,
               [node IN nodes(path) | node.id] as node_ids,
               length(path) as path_length
    """

    FIND_ALL_CALL_PATHS = """
        MATCH path = (source {{id: $source_id}})-[:CALLS*1..{max_depth}]->(target {id: $target_id})
        RETURN path,
               [node IN nodes(path) | node.id] as node_ids,
               length(path) as path_length
        ORDER BY path_length
        LIMIT $limit
    """

    # ==========================================================================
    # Impact Analysis Queries
    # ==========================================================================

    IMPACT_ANALYSIS = """
        MATCH path = (source {{id: $source_id}})<-[:CALLS*1..{depth}]-(affected)
        WHERE (affected:Function OR affected:Method)
        WITH affected, min(length(path)) as distance
        RETURN affected.id as id,
               affected.name as name,
               affected.file_path as file_path,
               distance
        ORDER BY distance, file_path
    """

    IMPACT_ANALYSIS_WITH_FILES = """
        MATCH path = (source {{id: $source_id}})<-[:CALLS*1..{depth}]-(affected)
        WHERE (affected:Function OR affected:Method)
        WITH affected, min(length(path)) as distance
        RETURN affected.id as id,
               affected.name as name,
               affected.file_path as file_path,
               distance,
               labels(affected) as labels
        ORDER BY distance, file_path
    """

    # ==========================================================================
    # Function Context Queries
    # ==========================================================================

    FUNCTION_CONTEXT = """
        MATCH (f {id: $function_id})
        WHERE (f:Function OR f:Method)
        OPTIONAL MATCH (f)-[:CALLS]->(callee)
        OPTIONAL MATCH (caller)-[:CALLS]->(f)
        OPTIONAL MATCH (f)-[:USES]->(var:Variable)
        RETURN f as function,
               collect(DISTINCT callee) as callees,
               collect(DISTINCT caller) as callers,
               collect(DISTINCT var) as variables
    """

    FUNCTION_FULL_CONTEXT = """
        MATCH (f {id: $function_id})
        WHERE (f:Function OR f:Method)
        OPTIONAL MATCH (f)-[:CALLS]->(callee)
        OPTIONAL MATCH (caller)-[:CALLS]->(f)
        OPTIONAL MATCH (f)-[:USES]->(var:Variable)
        OPTIONAL MATCH (file:File)-[:DEFINES]->(f)
        OPTIONAL MATCH (class:Class)-[:HAS_METHOD]->(f)
        RETURN f as function,
               collect(DISTINCT callee) as callees,
               collect(DISTINCT caller) as callers,
               collect(DISTINCT var) as variables,
               file as containing_file,
               class as containing_class
    """

    # ==========================================================================
    # File Dependency Queries
    # ==========================================================================

    FILE_DEPENDENCIES = """
        MATCH (f:File {id: $file_id})-[:IMPORTS]->(dep)
        WHERE (dep:Module OR dep:File)
        RETURN dep
        ORDER BY dep.path
    """

    FILE_DEPENDENTS = """
        MATCH (f:File {id: $file_id})<-[:IMPORTS]-(dependent:File)
        RETURN dependent
        ORDER BY dependent.path
    """

    FILE_DEPENDENCY_TREE = """
        MATCH path = (f:File {id: $file_id})-[:IMPORTS*1..{depth}]->(dep)
        WHERE (dep:Module OR dep:File)
        RETURN DISTINCT dep,
               length(path) as distance
        ORDER BY distance
    """

    # ==========================================================================
    # Class Hierarchy Queries
    # ==========================================================================

    CLASS_PARENTS = """
        MATCH (c:Class {id: $class_id})-[:INHERITS]->(parent:Class)
        RETURN parent
    """

    CLASS_ANCESTORS = """
        MATCH path = (c:Class {id: $class_id})-[:INHERITS*1..{depth}]->(ancestor:Class)
        RETURN DISTINCT ancestor,
               length(path) as distance
        ORDER BY distance
    """

    CLASS_CHILDREN = """
        MATCH (c:Class {id: $class_id})<-[:INHERITS]-(child:Class)
        RETURN child
    """

    CLASS_DESCENDANTS = """
        MATCH path = (c:Class {id: $class_id})<-[:INHERITS*1..{depth}]-(descendant:Class)
        RETURN DISTINCT descendant,
               length(path) as distance
        ORDER BY distance
    """

    CLASS_HIERARCHY = """
        MATCH (c:Class {id: $class_id})
        OPTIONAL MATCH ancestor_path = (c)-[:INHERITS*1..10]->(ancestor:Class)
        OPTIONAL MATCH descendant_path = (c)<-[:INHERITS*1..10]-(descendant:Class)
        RETURN c as class,
               collect(DISTINCT ancestor) as ancestors,
               collect(DISTINCT descendant) as descendants
    """

    # ==========================================================================
    # Delete Queries
    # ==========================================================================

    DELETE_NODE = """
        MATCH (n {id: $id})
        DETACH DELETE n
    """

    DELETE_REPOSITORY_CASCADE = """
        MATCH (r:Repository {id: $repo_id})
        OPTIONAL MATCH (r)-[*]->(n)
        DETACH DELETE r, n
    """

    DELETE_FILE_CASCADE = """
        MATCH (f:File {id: $file_id})
        OPTIONAL MATCH (f)-[:DEFINES]->(n)
        OPTIONAL MATCH (n)-[:HAS_METHOD]->(m:Method)
        DETACH DELETE f, n, m
    """

    DELETE_RELATIONSHIPS = """
        MATCH (n {id: $id})-[r]-()
        DELETE r
    """

    # ==========================================================================
    # Batch Queries
    # ==========================================================================

    BATCH_CREATE_NODES = """
        UNWIND $nodes as node
        CALL apoc.merge.node([node.label], {id: node.id}, node.properties) YIELD node as n
        RETURN count(n) as created
    """

    BATCH_CREATE_RELATIONSHIPS = """
        UNWIND $relationships as rel
        MATCH (a {id: rel.source_id}), (b {id: rel.target_id})
        CALL apoc.merge.relationship(a, rel.type, {}, {}, b) YIELD rel as r
        RETURN count(r) as created
    """

    # Simple batch queries without APOC
    BATCH_CREATE_FUNCTIONS = """
        UNWIND $functions as f
        MERGE (func:Function {id: f.id})
        SET func.name = f.name,
            func.file_path = f.file_path,
            func.start_line = f.start_line,
            func.end_line = f.end_line,
            func.is_async = f.is_async,
            func.docstring = f.docstring,
            func.parameters = f.parameters,
            func.return_type = f.return_type,
            func.decorators = f.decorators,
            func.repo_id = f.repo_id
        RETURN count(func) as created
    """

    BATCH_CREATE_CLASSES = """
        UNWIND $classes as c
        MERGE (cls:Class {id: c.id})
        SET cls.name = c.name,
            cls.file_path = c.file_path,
            cls.start_line = c.start_line,
            cls.end_line = c.end_line,
            cls.docstring = c.docstring,
            cls.bases = c.bases,
            cls.decorators = c.decorators,
            cls.repo_id = c.repo_id
        RETURN count(cls) as created
    """

    BATCH_CREATE_METHODS = """
        UNWIND $methods as m
        MERGE (method:Method {id: m.id})
        SET method.name = m.name,
            method.file_path = m.file_path,
            method.start_line = m.start_line,
            method.end_line = m.end_line,
            method.is_async = m.is_async,
            method.docstring = m.docstring,
            method.parameters = m.parameters,
            method.return_type = m.return_type,
            method.decorators = m.decorators,
            method.is_classmethod = m.is_classmethod,
            method.is_staticmethod = m.is_staticmethod,
            method.is_property = m.is_property,
            method.class_id = m.class_id,
            method.repo_id = m.repo_id
        RETURN count(method) as created
    """

    BATCH_CREATE_VARIABLES = """
        UNWIND $variables as v
        MERGE (var:Variable {id: v.id})
        SET var.name = v.name,
            var.file_path = v.file_path,
            var.line = v.line,
            var.type_annotation = v.type_annotation,
            var.is_constant = v.is_constant,
            var.repo_id = v.repo_id
        RETURN count(var) as created
    """

    BATCH_CREATE_CALLS = """
        UNWIND $calls as call
        MATCH (caller {id: call.caller_id})
        MATCH (callee {id: call.callee_id})
        WHERE (caller:Function OR caller:Method) AND (callee:Function OR callee:Method)
        MERGE (caller)-[r:CALLS]->(callee)
        RETURN count(r) as created
    """

    BATCH_CREATE_FILES = """
        UNWIND $files as f
        MERGE (file:File {id: f.id})
        SET file.path = f.path,
            file.language = f.language,
            file.hash = f.hash,
            file.size = f.size,
            file.repo_id = f.repo_id
        RETURN count(file) as created
    """

    # ==========================================================================
    # Statistics Queries
    # ==========================================================================

    REPO_STATISTICS = """
        MATCH (r:Repository {id: $repo_id})
        OPTIONAL MATCH (f:File {repo_id: $repo_id})
        OPTIONAL MATCH (func:Function {repo_id: $repo_id})
        OPTIONAL MATCH (cls:Class {repo_id: $repo_id})
        OPTIONAL MATCH (m:Method {repo_id: $repo_id})
        RETURN r,
               count(DISTINCT f) as file_count,
               count(DISTINCT func) as function_count,
               count(DISTINCT cls) as class_count,
               count(DISTINCT m) as method_count
    """

    COUNT_BY_TYPE = """
        MATCH (n {repo_id: $repo_id})
        WITH labels(n) as node_labels
        UNWIND node_labels as label
        RETURN label, count(*) as count
        ORDER BY count DESC
    """


# Singleton instance for easy import
QUERIES = CypherQueries()
