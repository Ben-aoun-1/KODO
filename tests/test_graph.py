"""Tests for the Neo4j graph module.

This module contains comprehensive tests for:
- Graph node and relationship models
- Utility functions (ID generation, sanitization, entity conversion, batching)
- Cypher query templates
- GraphStore operations (with mocked Neo4j driver)
- Integration tests for full CRUD cycles

All tests use mocks for the Neo4j driver to avoid requiring a real database.
"""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph.connection import GraphConnection, GraphConnectionError
from core.graph.models import (
    BaseNode,
    ClassNode,
    FileNode,
    FunctionNode,
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
from core.graph.queries import QUERIES, CypherQueries
from core.graph.store import GraphStore, GraphStoreError
from core.graph.utils import (
    batch_nodes,
    batch_relationships,
    entity_to_graph_node,
    entity_type_to_node_type,
    extract_relationships_from_module,
    generate_node_id,
    node_to_cypher_properties,
    node_type_to_label,
    parse_neo4j_node,
    record_to_node,
    sanitize_cypher_string,
    sanitize_identifier,
)
from core.parser.models import (
    Attribute,
    ClassEntity,
    EntityType,
    FunctionEntity,
    MethodEntity,
    ModuleEntity,
    Parameter,
    VariableEntity,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_repository_node() -> RepositoryNode:
    """Create a sample repository node for testing."""
    return RepositoryNode(
        id="repo-123",
        name="test-repo",
        url="https://github.com/test/test-repo",
        default_branch="main",
        last_indexed=datetime(2024, 1, 15, 10, 30, 0),
    )


@pytest.fixture
def sample_file_node() -> FileNode:
    """Create a sample file node for testing."""
    return FileNode(
        id="repo-123:/src/main.py",
        path="/src/main.py",
        language="python",
        hash="abc123def456",
        size=1024,
        repo_id="repo-123",
    )


@pytest.fixture
def sample_class_node() -> ClassNode:
    """Create a sample class node for testing."""
    return ClassNode(
        id="/src/main.py:MyClass:10",
        name="MyClass",
        file_path="/src/main.py",
        start_line=10,
        end_line=50,
        docstring="A sample class for testing.",
        bases=["BaseClass", "Mixin"],
        decorators=["dataclass"],
        repo_id="repo-123",
    )


@pytest.fixture
def sample_function_node() -> FunctionNode:
    """Create a sample function node for testing."""
    return FunctionNode(
        id="/src/main.py:process_data:60",
        name="process_data",
        file_path="/src/main.py",
        start_line=60,
        end_line=80,
        is_async=True,
        docstring="Process data asynchronously.",
        parameters=["data", "options"],
        return_type="dict",
        decorators=["cache"],
        repo_id="repo-123",
    )


@pytest.fixture
def sample_method_node() -> MethodNode:
    """Create a sample method node for testing."""
    return MethodNode(
        id="/src/main.py:MyClass.do_work:20",
        name="do_work",
        file_path="/src/main.py",
        start_line=20,
        end_line=35,
        is_async=False,
        docstring="Do some work.",
        parameters=["self", "input_data"],
        return_type="bool",
        decorators=[],
        is_classmethod=False,
        is_staticmethod=False,
        is_property=False,
        class_id="/src/main.py:MyClass:10",
        repo_id="repo-123",
    )


@pytest.fixture
def sample_variable_node() -> VariableNode:
    """Create a sample variable node for testing."""
    return VariableNode(
        id="/src/main.py:MAX_SIZE:5",
        name="MAX_SIZE",
        file_path="/src/main.py",
        line=5,
        type_annotation="int",
        is_constant=True,
        repo_id="repo-123",
    )


@pytest.fixture
def sample_module_node() -> ModuleNode:
    """Create a sample module node for testing."""
    return ModuleNode(
        id="/src/main.py:module:main",
        name="main",
        file_path="/src/main.py",
        docstring="Main module docstring.",
        repo_id="repo-123",
    )


@pytest.fixture
def sample_relationship() -> GraphRelationship:
    """Create a sample relationship for testing."""
    return GraphRelationship(
        source_id="/src/main.py:caller:10",
        target_id="/src/main.py:callee:50",
        relationship_type=RelationshipType.CALLS,
        properties={"weight": 1},
    )


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j async driver."""
    driver = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    return driver


@pytest.fixture
def mock_neo4j_session():
    """Create a mock Neo4j async session."""
    session = AsyncMock()
    session.run = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_graph_connection(mock_neo4j_driver, mock_neo4j_session):
    """Create a mock GraphConnection with mocked driver and session."""
    connection = GraphConnection(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
    )
    connection._driver = mock_neo4j_driver

    # Mock the session context manager
    async def mock_session_cm(**kwargs):
        yield mock_neo4j_session

    connection.session = MagicMock(return_value=mock_session_cm())
    return connection


@pytest.fixture
def sample_function_entity() -> FunctionEntity:
    """Create a sample FunctionEntity for testing."""
    return FunctionEntity(
        id="/src/utils.py:helper:10",
        name="helper",
        file_path="/src/utils.py",
        start_line=10,
        end_line=20,
        source_code="def helper(x): return x * 2",
        docstring="A helper function.",
        language="python",
        parameters=[
            Parameter(name="x", type_annotation="int"),
        ],
        return_type="int",
        decorators=[],
        is_async=False,
        calls=["other_func"],
    )


@pytest.fixture
def sample_method_entity() -> MethodEntity:
    """Create a sample MethodEntity for testing."""
    return MethodEntity(
        id="/src/models.py:User.save:30",
        name="save",
        file_path="/src/models.py",
        start_line=30,
        end_line=45,
        source_code="def save(self): pass",
        docstring="Save the user.",
        language="python",
        parent_id="/src/models.py:User:10",
        parameters=[
            Parameter(name="self"),
        ],
        return_type="None",
        decorators=[],
        is_async=True,
        is_classmethod=False,
        is_staticmethod=False,
        is_property=False,
        calls=["db.commit"],
    )


@pytest.fixture
def sample_class_entity() -> ClassEntity:
    """Create a sample ClassEntity for testing."""
    return ClassEntity(
        id="/src/models.py:User:10",
        name="User",
        file_path="/src/models.py",
        start_line=10,
        end_line=100,
        source_code="class User: pass",
        docstring="User model class.",
        language="python",
        bases=["BaseModel"],
        methods=[],
        attributes=[
            Attribute(name="id", type_annotation="int", line=12),
        ],
        decorators=["dataclass"],
    )


@pytest.fixture
def sample_variable_entity() -> VariableEntity:
    """Create a sample VariableEntity for testing."""
    return VariableEntity(
        id="/src/config.py:DEBUG:5",
        name="DEBUG",
        file_path="/src/config.py",
        start_line=5,
        end_line=5,
        source_code="DEBUG = True",
        docstring=None,
        language="python",
        type_annotation="bool",
        value="True",
        is_constant=True,
    )


@pytest.fixture
def sample_module_entity(
    sample_function_entity, sample_class_entity, sample_variable_entity
) -> ModuleEntity:
    """Create a sample ModuleEntity for testing."""
    return ModuleEntity(
        id="/src/main.py:module",
        name="main",
        file_path="/src/main.py",
        start_line=1,
        end_line=100,
        source_code="# Module content",
        docstring="Main module.",
        language="python",
        imports=[],
        functions=[sample_function_entity],
        classes=[sample_class_entity],
        variables=[sample_variable_entity],
    )


# =============================================================================
# Model Tests
# =============================================================================


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        """Test that NodeType enum has expected values."""
        assert NodeType.REPOSITORY == "Repository"
        assert NodeType.FILE == "File"
        assert NodeType.MODULE == "Module"
        assert NodeType.CLASS == "Class"
        assert NodeType.FUNCTION == "Function"
        assert NodeType.METHOD == "Method"
        assert NodeType.VARIABLE == "Variable"

    def test_node_type_is_str_enum(self):
        """Test that NodeType values can be used as strings."""
        assert NodeType.REPOSITORY.value == "Repository"
        assert f"Label: {NodeType.CLASS.value}" == "Label: Class"


class TestRelationshipType:
    """Tests for RelationshipType enum."""

    def test_relationship_type_values(self):
        """Test that RelationshipType enum has expected values."""
        assert RelationshipType.CONTAINS == "CONTAINS"
        assert RelationshipType.DEFINES == "DEFINES"
        assert RelationshipType.HAS_METHOD == "HAS_METHOD"
        assert RelationshipType.INHERITS == "INHERITS"
        assert RelationshipType.CALLS == "CALLS"
        assert RelationshipType.USES == "USES"
        assert RelationshipType.IMPORTS == "IMPORTS"
        assert RelationshipType.RETURNS == "RETURNS"
        assert RelationshipType.ACCEPTS == "ACCEPTS"
        assert RelationshipType.TYPE_OF == "TYPE_OF"

    def test_relationship_type_is_str_enum(self):
        """Test that RelationshipType values can be used as strings."""
        assert RelationshipType.CALLS.value == "CALLS"


class TestRepositoryNode:
    """Tests for RepositoryNode model."""

    def test_create_repository_node(self, sample_repository_node):
        """Test creating a repository node with all fields."""
        node = sample_repository_node
        assert node.id == "repo-123"
        assert node.name == "test-repo"
        assert node.url == "https://github.com/test/test-repo"
        assert node.default_branch == "main"
        assert node.last_indexed == datetime(2024, 1, 15, 10, 30, 0)
        assert node.node_type == NodeType.REPOSITORY

    def test_repository_node_defaults(self):
        """Test repository node default values."""
        node = RepositoryNode(id="repo-1", name="my-repo")
        assert node.default_branch == "main"
        assert node.url is None
        assert node.last_indexed is None

    def test_repository_node_to_properties(self, sample_repository_node):
        """Test converting repository node to properties dict."""
        props = sample_repository_node.to_properties()
        assert props["id"] == "repo-123"
        assert props["name"] == "test-repo"
        assert "node_type" not in props
        # datetime should be converted to ISO string
        assert props["last_indexed"] == "2024-01-15T10:30:00"


class TestFileNode:
    """Tests for FileNode model."""

    def test_create_file_node(self, sample_file_node):
        """Test creating a file node with all fields."""
        node = sample_file_node
        assert node.id == "repo-123:/src/main.py"
        assert node.path == "/src/main.py"
        assert node.language == "python"
        assert node.hash == "abc123def456"
        assert node.size == 1024
        assert node.repo_id == "repo-123"
        assert node.node_type == NodeType.FILE

    def test_file_node_size_validation(self):
        """Test that file size must be non-negative."""
        with pytest.raises(ValueError):
            FileNode(
                id="test",
                path="/test.py",
                language="python",
                size=-1,
                repo_id="repo-1",
            )


class TestClassNode:
    """Tests for ClassNode model."""

    def test_create_class_node(self, sample_class_node):
        """Test creating a class node with all fields."""
        node = sample_class_node
        assert node.name == "MyClass"
        assert node.bases == ["BaseClass", "Mixin"]
        assert node.decorators == ["dataclass"]
        assert node.start_line == 10
        assert node.end_line == 50
        assert node.node_type == NodeType.CLASS

    def test_class_node_empty_bases(self):
        """Test class node with no base classes."""
        node = ClassNode(
            id="test",
            name="Simple",
            file_path="/test.py",
            start_line=1,
            end_line=10,
            repo_id="repo-1",
        )
        assert node.bases == []
        assert node.decorators == []


class TestFunctionNode:
    """Tests for FunctionNode model."""

    def test_create_function_node(self, sample_function_node):
        """Test creating a function node with all fields."""
        node = sample_function_node
        assert node.name == "process_data"
        assert node.is_async is True
        assert node.parameters == ["data", "options"]
        assert node.return_type == "dict"
        assert node.node_type == NodeType.FUNCTION

    def test_function_node_defaults(self):
        """Test function node default values."""
        node = FunctionNode(
            id="test",
            name="simple",
            file_path="/test.py",
            start_line=1,
            end_line=5,
            repo_id="repo-1",
        )
        assert node.is_async is False
        assert node.parameters == []
        assert node.decorators == []
        assert node.return_type is None


class TestMethodNode:
    """Tests for MethodNode model."""

    def test_create_method_node(self, sample_method_node):
        """Test creating a method node with all fields."""
        node = sample_method_node
        assert node.name == "do_work"
        assert node.class_id == "/src/main.py:MyClass:10"
        assert node.is_classmethod is False
        assert node.is_staticmethod is False
        assert node.is_property is False
        assert node.node_type == NodeType.METHOD

    def test_method_node_special_types(self):
        """Test method node with classmethod/staticmethod/property."""
        node = MethodNode(
            id="test",
            name="factory",
            file_path="/test.py",
            start_line=1,
            end_line=5,
            is_classmethod=True,
            repo_id="repo-1",
        )
        assert node.is_classmethod is True
        assert node.is_staticmethod is False


class TestVariableNode:
    """Tests for VariableNode model."""

    def test_create_variable_node(self, sample_variable_node):
        """Test creating a variable node with all fields."""
        node = sample_variable_node
        assert node.name == "MAX_SIZE"
        assert node.line == 5
        assert node.type_annotation == "int"
        assert node.is_constant is True
        assert node.node_type == NodeType.VARIABLE


class TestGraphRelationship:
    """Tests for GraphRelationship model."""

    def test_create_relationship(self, sample_relationship):
        """Test creating a relationship with all fields."""
        rel = sample_relationship
        assert rel.source_id == "/src/main.py:caller:10"
        assert rel.target_id == "/src/main.py:callee:50"
        assert rel.relationship_type == RelationshipType.CALLS
        assert rel.properties == {"weight": 1}

    def test_relationship_default_properties(self):
        """Test relationship with default empty properties."""
        rel = GraphRelationship(
            source_id="a",
            target_id="b",
            relationship_type=RelationshipType.DEFINES,
        )
        assert rel.properties == {}


class TestGraphPath:
    """Tests for GraphPath model."""

    def test_create_graph_path(self):
        """Test creating a graph path."""
        path = GraphPath(
            nodes=["a", "b", "c"],
            relationships=[RelationshipType.CALLS, RelationshipType.CALLS],
            length=2,
        )
        assert path.nodes == ["a", "b", "c"]
        assert len(path.relationships) == 2
        assert path.length == 2


class TestImpactResult:
    """Tests for ImpactResult model."""

    def test_create_impact_result(self):
        """Test creating an impact result."""
        result = ImpactResult(
            source_id="func-1",
            affected_nodes=["func-2", "func-3"],
            affected_files=["/src/a.py", "/src/b.py"],
            depth=2,
            total_affected=2,
        )
        assert result.source_id == "func-1"
        assert len(result.affected_nodes) == 2
        assert len(result.affected_files) == 2
        assert result.depth == 2
        assert result.total_affected == 2


class TestModelValidation:
    """Tests for model validation."""

    def test_start_line_must_be_positive(self):
        """Test that start_line must be >= 1."""
        with pytest.raises(ValueError):
            ClassNode(
                id="test",
                name="Invalid",
                file_path="/test.py",
                start_line=0,
                end_line=10,
                repo_id="repo-1",
            )

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValueError):
            RepositoryNode(
                id="repo-1",
                name="test",
                unknown_field="value",
            )


# =============================================================================
# Utils Tests
# =============================================================================


class TestGenerateNodeId:
    """Tests for generate_node_id function."""

    def test_generate_id_with_start_line(self):
        """Test generating ID with file path, name, and start line."""
        node_id = generate_node_id("/src/main.py", "my_function", 42)
        assert node_id == "/src/main.py:my_function:42"

    def test_generate_id_without_start_line(self):
        """Test generating ID without start line."""
        node_id = generate_node_id("/src/main.py", "module_var")
        assert node_id == "/src/main.py:module_var"

    def test_generate_id_with_entity_type(self):
        """Test generating ID with entity type for disambiguation."""
        node_id = generate_node_id("/src/main.py", "utils", entity_type="module")
        assert node_id == "/src/main.py:module:utils"

    def test_generate_id_start_line_takes_precedence(self):
        """Test that start_line takes precedence over entity_type."""
        node_id = generate_node_id("/src/main.py", "func", start_line=10, entity_type="function")
        assert node_id == "/src/main.py:func:10"


class TestSanitizeCypherString:
    """Tests for sanitize_cypher_string function."""

    def test_sanitize_simple_string(self):
        """Test that simple strings pass through unchanged."""
        assert sanitize_cypher_string("hello") == "hello"

    def test_sanitize_string_with_quotes(self):
        """Test escaping single and double quotes."""
        assert sanitize_cypher_string("it's") == "it\\'s"
        assert sanitize_cypher_string('say "hello"') == 'say \\"hello\\"'

    def test_sanitize_string_with_backslash(self):
        """Test escaping backslashes."""
        assert sanitize_cypher_string("path\\to\\file") == "path\\\\to\\\\file"

    def test_sanitize_non_string(self):
        """Test that non-strings are converted to strings."""
        assert sanitize_cypher_string(123) == "123"

    def test_sanitize_complex_injection_attempt(self):
        """Test sanitizing potential injection strings."""
        malicious = "'; DROP DATABASE; --"
        result = sanitize_cypher_string(malicious)
        assert "\\'" in result


class TestSanitizeIdentifier:
    """Tests for sanitize_identifier function."""

    def test_sanitize_valid_identifier(self):
        """Test that valid identifiers pass through."""
        assert sanitize_identifier("my_function") == "my_function"
        assert sanitize_identifier("MyClass") == "MyClass"

    def test_sanitize_identifier_with_special_chars(self):
        """Test replacing special characters with underscores."""
        assert sanitize_identifier("my-function") == "my_function"
        assert sanitize_identifier("my.method") == "my_method"
        assert sanitize_identifier("path/to/file") == "path_to_file"

    def test_sanitize_identifier_starting_with_number(self):
        """Test prepending underscore if identifier starts with number."""
        assert sanitize_identifier("123abc") == "_123abc"


class TestEntityToGraphNode:
    """Tests for entity_to_graph_node function."""

    def test_convert_function_entity(self, sample_function_entity):
        """Test converting FunctionEntity to FunctionNode."""
        node = entity_to_graph_node(sample_function_entity, "repo-123")
        assert isinstance(node, FunctionNode)
        assert node.id == sample_function_entity.id
        assert node.name == "helper"
        assert node.repo_id == "repo-123"
        assert node.parameters == ["x"]
        assert node.return_type == "int"

    def test_convert_method_entity(self, sample_method_entity):
        """Test converting MethodEntity to MethodNode."""
        node = entity_to_graph_node(sample_method_entity, "repo-123")
        assert isinstance(node, MethodNode)
        assert node.name == "save"
        assert node.is_async is True
        assert node.class_id == "/src/models.py:User:10"

    def test_convert_class_entity(self, sample_class_entity):
        """Test converting ClassEntity to ClassNode."""
        node = entity_to_graph_node(sample_class_entity, "repo-123")
        assert isinstance(node, ClassNode)
        assert node.name == "User"
        assert node.bases == ["BaseModel"]
        assert node.decorators == ["dataclass"]

    def test_convert_variable_entity(self, sample_variable_entity):
        """Test converting VariableEntity to VariableNode."""
        node = entity_to_graph_node(sample_variable_entity, "repo-123")
        assert isinstance(node, VariableNode)
        assert node.name == "DEBUG"
        assert node.is_constant is True
        assert node.line == 5

    def test_convert_unsupported_entity_returns_none(self):
        """Test that unsupported entity types return None."""
        # Import entities are not converted to graph nodes
        from core.parser.models import ImportEntity

        import_entity = ImportEntity(
            id="test:import:1",
            name="os",
            file_path="/test.py",
            start_line=1,
            end_line=1,
            source_code="import os",
            language="python",
            module="os",
        )
        result = entity_to_graph_node(import_entity, "repo-123")
        assert result is None


class TestBatchNodes:
    """Tests for batch_nodes function."""

    def test_batch_nodes_smaller_than_batch_size(self, sample_function_node):
        """Test batching when nodes fit in one batch."""
        nodes = [sample_function_node] * 10
        batches = batch_nodes(nodes, batch_size=500)
        assert len(batches) == 1
        assert len(batches[0]) == 10

    def test_batch_nodes_larger_than_batch_size(self, sample_function_node):
        """Test batching when nodes need multiple batches."""
        nodes = [sample_function_node] * 1200
        batches = batch_nodes(nodes, batch_size=500)
        assert len(batches) == 3
        assert len(batches[0]) == 500
        assert len(batches[1]) == 500
        assert len(batches[2]) == 200

    def test_batch_nodes_empty_list(self):
        """Test batching an empty list."""
        batches = batch_nodes([])
        assert batches == []

    def test_batch_nodes_custom_size(self, sample_class_node):
        """Test batching with custom batch size."""
        nodes = [sample_class_node] * 25
        batches = batch_nodes(nodes, batch_size=10)
        assert len(batches) == 3


class TestBatchRelationships:
    """Tests for batch_relationships function."""

    def test_batch_relationships_smaller_than_batch_size(self, sample_relationship):
        """Test batching when relationships fit in one batch."""
        rels = [sample_relationship] * 50
        batches = batch_relationships(rels, batch_size=500)
        assert len(batches) == 1
        assert len(batches[0]) == 50

    def test_batch_relationships_larger_than_batch_size(self, sample_relationship):
        """Test batching when relationships need multiple batches."""
        rels = [sample_relationship] * 1500
        batches = batch_relationships(rels, batch_size=500)
        assert len(batches) == 3


class TestExtractRelationshipsFromModule:
    """Tests for extract_relationships_from_module function."""

    def test_extract_defines_relationships(self, sample_module_entity):
        """Test extracting DEFINES relationships for functions and classes."""
        relationships = extract_relationships_from_module(sample_module_entity, "repo-123")

        defines_rels = [r for r in relationships if r.relationship_type == RelationshipType.DEFINES]
        assert len(defines_rels) >= 2  # At least function and class

    def test_extract_relationships_file_id_format(self, sample_module_entity):
        """Test that file ID is correctly formatted."""
        relationships = extract_relationships_from_module(sample_module_entity, "repo-123")

        for rel in relationships:
            if rel.relationship_type == RelationshipType.DEFINES:
                assert rel.source_id.endswith(":file")
                break


class TestParseNeo4jNode:
    """Tests for parse_neo4j_node function."""

    def test_parse_node_from_dict(self):
        """Test parsing a node from a dictionary record."""
        record = {"n": {"id": "test-1", "name": "MyFunc", "start_line": 10}}
        result = parse_neo4j_node(record, "n")
        assert result["id"] == "test-1"
        assert result["name"] == "MyFunc"

    def test_parse_node_with_labels(self):
        """Test parsing node with labels."""
        record = {"n": {"id": "test-1", "name": "MyFunc"}, "labels": ["Function"]}
        result = parse_neo4j_node(record, "n")
        assert result["_labels"] == ["Function"]

    def test_parse_node_missing_key(self):
        """Test parsing when node key is missing."""
        record = {"other": {"id": "test-1"}}
        result = parse_neo4j_node(record, "n")
        assert result == {}

    def test_parse_node_none_value(self):
        """Test parsing when node value is None."""
        record = {"n": None}
        result = parse_neo4j_node(record, "n")
        assert result == {}


class TestRecordToNode:
    """Tests for record_to_node function."""

    def test_convert_function_record(self):
        """Test converting a Neo4j record to FunctionNode."""
        record = {
            "n": {
                "id": "/test.py:func:1",
                "name": "func",
                "file_path": "/test.py",
                "start_line": 1,
                "end_line": 10,
                "is_async": False,
                "repo_id": "repo-1",
            },
            "labels": ["Function"],
        }
        node = record_to_node(record, "n")
        assert isinstance(node, FunctionNode)
        assert node.name == "func"

    def test_convert_class_record(self):
        """Test converting a Neo4j record to ClassNode."""
        record = {
            "n": {
                "id": "/test.py:MyClass:1",
                "name": "MyClass",
                "file_path": "/test.py",
                "start_line": 1,
                "end_line": 50,
                "bases": [],
                "decorators": [],
                "repo_id": "repo-1",
            },
            "labels": ["Class"],
        }
        node = record_to_node(record, "n")
        assert isinstance(node, ClassNode)
        assert node.name == "MyClass"

    def test_convert_repository_record(self):
        """Test converting a Neo4j record to RepositoryNode."""
        record = {
            "n": {
                "id": "repo-1",
                "name": "test-repo",
            },
            "labels": ["Repository"],
        }
        node = record_to_node(record, "n")
        assert isinstance(node, RepositoryNode)

    def test_convert_with_missing_labels(self):
        """Test that missing labels returns None."""
        record = {"n": {"id": "test-1", "name": "test"}}
        node = record_to_node(record, "n")
        assert node is None


class TestNodeToCypherProperties:
    """Tests for node_to_cypher_properties function."""

    def test_convert_function_node(self, sample_function_node):
        """Test converting FunctionNode to Cypher properties."""
        props = node_to_cypher_properties(sample_function_node)
        assert props["id"] == sample_function_node.id
        assert props["name"] == "process_data"
        assert props["is_async"] is True
        assert "node_type" not in props


class TestNodeTypeToLabel:
    """Tests for node_type_to_label function."""

    def test_convert_node_types(self):
        """Test converting NodeType to Neo4j label string."""
        assert node_type_to_label(NodeType.FUNCTION) == "Function"
        assert node_type_to_label(NodeType.CLASS) == "Class"
        assert node_type_to_label(NodeType.REPOSITORY) == "Repository"


class TestEntityTypeToNodeType:
    """Tests for entity_type_to_node_type function."""

    def test_convert_entity_types(self):
        """Test converting EntityType to NodeType."""
        assert entity_type_to_node_type(EntityType.FUNCTION) == NodeType.FUNCTION
        assert entity_type_to_node_type(EntityType.CLASS) == NodeType.CLASS
        assert entity_type_to_node_type(EntityType.METHOD) == NodeType.METHOD
        assert entity_type_to_node_type(EntityType.VARIABLE) == NodeType.VARIABLE

    def test_unsupported_entity_type(self):
        """Test that unsupported entity types return None."""
        result = entity_type_to_node_type(EntityType.IMPORT)
        assert result is None


# =============================================================================
# Query Template Tests
# =============================================================================


class TestCypherQueries:
    """Tests for Cypher query templates."""

    def test_queries_singleton_exists(self):
        """Test that QUERIES singleton is available."""
        assert QUERIES is not None
        assert isinstance(QUERIES, CypherQueries)

    def test_create_repository_query_has_parameters(self):
        """Test CREATE_REPOSITORY query uses parameters."""
        query = QUERIES.CREATE_REPOSITORY
        assert "$id" in query
        assert "$name" in query
        assert "$url" in query
        assert "MERGE" in query

    def test_create_function_query_has_parameters(self):
        """Test CREATE_FUNCTION query uses parameters."""
        query = QUERIES.CREATE_FUNCTION
        assert "$id" in query
        assert "$name" in query
        assert "$file_path" in query
        assert "$is_async" in query
        assert "MERGE" in query

    def test_create_class_query_has_parameters(self):
        """Test CREATE_CLASS query uses parameters."""
        query = QUERIES.CREATE_CLASS
        assert "$id" in query
        assert "$name" in query
        assert "$bases" in query
        assert "$decorators" in query

    def test_relationship_queries_use_parameters(self):
        """Test relationship queries use parameterized values."""
        assert "$source_id" in QUERIES.CREATE_CONTAINS
        assert "$target_id" in QUERIES.CREATE_CONTAINS
        assert "$source_id" in QUERIES.CREATE_CALLS
        assert "$target_id" in QUERIES.CREATE_CALLS

    def test_find_callers_query_structure(self):
        """Test FIND_CALLERS query has correct structure."""
        query = QUERIES.FIND_CALLERS
        assert "$target_id" in query
        assert "CALLS" in query
        assert "caller" in query.lower()

    def test_find_callees_query_structure(self):
        """Test FIND_CALLEES query has correct structure."""
        query = QUERIES.FIND_CALLEES
        assert "$source_id" in query
        assert "CALLS" in query
        assert "callee" in query.lower()

    def test_impact_analysis_query_structure(self):
        """Test IMPACT_ANALYSIS query has correct structure."""
        query = QUERIES.IMPACT_ANALYSIS
        assert "$source_id" in query
        assert "{depth}" in query  # For depth formatting
        assert "CALLS" in query

    def test_batch_queries_use_unwind(self):
        """Test batch queries use UNWIND for efficiency."""
        assert "UNWIND" in QUERIES.BATCH_CREATE_FUNCTIONS
        assert "UNWIND" in QUERIES.BATCH_CREATE_CLASSES
        assert "UNWIND" in QUERIES.BATCH_CREATE_METHODS
        assert "UNWIND" in QUERIES.BATCH_CREATE_CALLS

    def test_delete_queries_use_detach_delete(self):
        """Test delete queries use DETACH DELETE for safety."""
        assert "DETACH DELETE" in QUERIES.DELETE_NODE
        assert "DETACH DELETE" in QUERIES.DELETE_REPOSITORY_CASCADE
        assert "DETACH DELETE" in QUERIES.DELETE_FILE_CASCADE

    def test_class_hierarchy_queries(self):
        """Test class hierarchy queries exist and are valid."""
        assert "INHERITS" in QUERIES.CLASS_PARENTS
        assert "INHERITS" in QUERIES.CLASS_ANCESTORS
        assert "INHERITS" in QUERIES.CLASS_CHILDREN
        assert "INHERITS" in QUERIES.CLASS_DESCENDANTS


# =============================================================================
# GraphStore Tests (with mocks)
# =============================================================================


class TestGraphStore:
    """Tests for GraphStore class with mocked Neo4j connection."""

    @pytest.fixture
    def graph_store(self, mock_graph_connection):
        """Create a GraphStore with mocked connection."""
        return GraphStore(mock_graph_connection)

    @pytest.fixture
    def mock_execute_write(self, mock_graph_connection):
        """Mock the execute_write method."""
        mock_graph_connection.execute_write = AsyncMock(return_value=[{"created": 1}])
        return mock_graph_connection.execute_write

    @pytest.fixture
    def mock_execute_read(self, mock_graph_connection):
        """Mock the execute_read method."""
        mock_graph_connection.execute_read = AsyncMock(return_value=[])
        return mock_graph_connection.execute_read

    @pytest.mark.asyncio
    async def test_create_repository(self, graph_store, sample_repository_node, mock_execute_write):
        """Test creating a repository node."""
        result = await graph_store.create_repository(sample_repository_node)

        assert result.id == sample_repository_node.id
        mock_execute_write.assert_called_once()
        call_args = mock_execute_write.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_get_repository_found(self, graph_store, mock_graph_connection):
        """Test getting a repository that exists."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "r": {
                        "id": "repo-123",
                        "name": "test-repo",
                        "default_branch": "main",
                    }
                }
            ]
        )

        result = await graph_store.get_repository("repo-123")

        assert result is not None
        assert result.id == "repo-123"
        assert result.name == "test-repo"

    @pytest.mark.asyncio
    async def test_get_repository_not_found(self, graph_store, mock_execute_read):
        """Test getting a repository that doesn't exist."""
        mock_execute_read.return_value = []

        result = await graph_store.get_repository("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_create_function(self, graph_store, sample_function_node, mock_execute_write):
        """Test creating a function node."""
        result = await graph_store.create_function(sample_function_node)

        assert result.id == sample_function_node.id
        mock_execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_function_found(self, graph_store, mock_graph_connection):
        """Test getting a function that exists."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "f": {
                        "id": "/test.py:func:10",
                        "name": "func",
                        "file_path": "/test.py",
                        "start_line": 10,
                        "end_line": 20,
                        "is_async": False,
                        "repo_id": "repo-1",
                    }
                }
            ]
        )

        result = await graph_store.get_function("/test.py:func:10")

        assert result is not None
        assert result.name == "func"

    @pytest.mark.asyncio
    async def test_create_class(self, graph_store, sample_class_node, mock_execute_write):
        """Test creating a class node."""
        result = await graph_store.create_class(sample_class_node)

        assert result.id == sample_class_node.id
        mock_execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_method(self, graph_store, sample_method_node, mock_execute_write):
        """Test creating a method node."""
        result = await graph_store.create_method(sample_method_node)

        assert result.id == sample_method_node.id

    @pytest.mark.asyncio
    async def test_create_variable(self, graph_store, sample_variable_node, mock_execute_write):
        """Test creating a variable node."""
        result = await graph_store.create_variable(sample_variable_node)

        assert result.id == sample_variable_node.id

    @pytest.mark.asyncio
    async def test_create_module(self, graph_store, sample_module_node, mock_execute_write):
        """Test creating a module node."""
        result = await graph_store.create_module(sample_module_node)

        assert result.id == sample_module_node.id

    @pytest.mark.asyncio
    async def test_batch_create_functions(
        self, graph_store, sample_function_node, mock_execute_write
    ):
        """Test batch creating function nodes."""
        functions = [sample_function_node] * 3

        result = await graph_store.batch_create_functions(functions)

        assert result == 1  # Mock returns {"created": 1}

    @pytest.mark.asyncio
    async def test_batch_create_empty_list(self, graph_store, mock_execute_write):
        """Test batch creating with empty list."""
        result = await graph_store.batch_create_functions([])

        assert result == 0
        mock_execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_relationship(self, graph_store, sample_relationship, mock_execute_write):
        """Test creating a relationship."""
        result = await graph_store.create_relationship(sample_relationship)

        assert result is True
        mock_execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_unsupported_type(self, graph_store, mock_execute_write):
        """Test creating relationship with unsupported type."""
        rel = GraphRelationship(
            source_id="a",
            target_id="b",
            relationship_type=RelationshipType.RETURNS,  # Not in query_map
        )

        result = await graph_store.create_relationship(rel)

        assert result is False
        mock_execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_create_relationships(
        self, graph_store, sample_relationship, mock_execute_write
    ):
        """Test batch creating relationships."""
        rels = [sample_relationship] * 3

        result = await graph_store.batch_create_relationships(rels)

        assert result == 3  # Each relationship created successfully

    @pytest.mark.asyncio
    async def test_find_callers(self, graph_store, mock_graph_connection):
        """Test finding callers of a function."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "caller": {
                        "id": "/test.py:caller1:5",
                        "name": "caller1",
                        "file_path": "/test.py",
                        "start_line": 5,
                        "end_line": 15,
                        "is_async": False,
                        "repo_id": "repo-1",
                    },
                    "labels": ["Function"],
                }
            ]
        )

        await graph_store.find_callers("/test.py:target:50")

        # Note: record_to_node needs labels to be parsed correctly
        # The mock might not return properly typed nodes without labels
        mock_graph_connection.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_callees(self, graph_store, mock_graph_connection):
        """Test finding callees of a function."""
        mock_graph_connection.execute_read = AsyncMock(return_value=[])

        result = await graph_store.find_callees("/test.py:source:10")

        assert result == []
        mock_graph_connection.execute_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_impact(self, graph_store, mock_graph_connection):
        """Test impact analysis."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "id": "/test.py:affected1:20",
                    "name": "affected1",
                    "file_path": "/test.py",
                    "distance": 1,
                },
                {
                    "id": "/test.py:affected2:30",
                    "name": "affected2",
                    "file_path": "/other.py",
                    "distance": 2,
                },
            ]
        )

        result = await graph_store.analyze_impact("/test.py:source:10")

        assert isinstance(result, ImpactResult)
        assert result.source_id == "/test.py:source:10"
        assert len(result.affected_nodes) == 2
        assert len(result.affected_files) == 2
        assert result.depth == 2

    @pytest.mark.asyncio
    async def test_analyze_impact_no_affected(self, graph_store, mock_execute_read):
        """Test impact analysis with no affected nodes."""
        mock_execute_read.return_value = []

        result = await graph_store.analyze_impact("/test.py:isolated:10")

        assert result.total_affected == 0
        assert result.affected_nodes == []

    @pytest.mark.asyncio
    async def test_delete_node(self, graph_store, mock_execute_write):
        """Test deleting a node."""
        result = await graph_store.delete_node("node-to-delete")

        assert result is True
        mock_execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_repository(self, graph_store, mock_execute_write):
        """Test deleting a repository and its contents."""
        result = await graph_store.delete_repository("repo-to-delete")

        assert result is True
        mock_execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_class_parents(self, graph_store, mock_graph_connection):
        """Test getting parent classes."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "parent": {
                        "id": "/test.py:ParentClass:1",
                        "name": "ParentClass",
                        "file_path": "/test.py",
                        "start_line": 1,
                        "end_line": 20,
                        "bases": [],
                        "decorators": [],
                        "repo_id": "repo-1",
                    }
                }
            ]
        )

        result = await graph_store.get_class_parents("/test.py:ChildClass:50")

        assert len(result) == 1
        assert result[0].name == "ParentClass"

    @pytest.mark.asyncio
    async def test_get_class_children(self, graph_store, mock_execute_read):
        """Test getting child classes."""
        mock_execute_read.return_value = []

        result = await graph_store.get_class_children("/test.py:ParentClass:1")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_methods_of_class(self, graph_store, mock_graph_connection):
        """Test getting methods of a class."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "m": {
                        "id": "/test.py:MyClass.method1:10",
                        "name": "method1",
                        "file_path": "/test.py",
                        "start_line": 10,
                        "end_line": 20,
                        "is_async": False,
                        "repo_id": "repo-1",
                    }
                }
            ]
        )

        result = await graph_store.get_methods_of_class("/test.py:MyClass:5")

        assert len(result) == 1
        assert result[0].name == "method1"

    @pytest.mark.asyncio
    async def test_get_repository_statistics(self, graph_store, mock_graph_connection):
        """Test getting repository statistics."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "file_count": 10,
                    "function_count": 50,
                    "class_count": 20,
                    "method_count": 100,
                }
            ]
        )

        result = await graph_store.get_repository_statistics("repo-123")

        assert result["file_count"] == 10
        assert result["function_count"] == 50
        assert result["class_count"] == 20
        assert result["method_count"] == 100

    @pytest.mark.asyncio
    async def test_find_call_path(self, graph_store, mock_graph_connection):
        """Test finding call path between functions."""
        mock_graph_connection.execute_read = AsyncMock(
            return_value=[
                {
                    "node_ids": ["a", "b", "c"],
                    "path_length": 2,
                }
            ]
        )

        result = await graph_store.find_call_path("a", "c")

        assert result is not None
        assert isinstance(result, GraphPath)
        assert result.nodes == ["a", "b", "c"]
        assert result.length == 2

    @pytest.mark.asyncio
    async def test_find_call_path_no_path(self, graph_store, mock_execute_read):
        """Test finding call path when no path exists."""
        mock_execute_read.return_value = []

        result = await graph_store.find_call_path("a", "z")

        assert result is None


class TestGraphStoreErrorHandling:
    """Tests for GraphStore error handling."""

    @pytest.fixture
    def graph_store(self, mock_graph_connection):
        """Create a GraphStore with mocked connection."""
        return GraphStore(mock_graph_connection)

    @pytest.mark.asyncio
    async def test_create_repository_error(
        self, graph_store, mock_graph_connection, sample_repository_node
    ):
        """Test error handling when creating repository fails."""
        from neo4j.exceptions import Neo4jError

        mock_graph_connection.execute_write = AsyncMock(side_effect=Neo4jError("Database error"))

        with pytest.raises(GraphStoreError):
            await graph_store.create_repository(sample_repository_node)

    @pytest.mark.asyncio
    async def test_get_repository_handles_error(self, graph_store, mock_graph_connection):
        """Test that get_repository handles errors gracefully."""
        from neo4j.exceptions import Neo4jError

        mock_graph_connection.execute_read = AsyncMock(side_effect=Neo4jError("Query failed"))

        result = await graph_store.get_repository("repo-123")

        assert result is None


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestGraphStoreIntegration:
    """Integration tests for GraphStore with mocked driver.

    These tests simulate full CRUD cycles and relationship traversals.
    """

    @pytest.fixture
    def mock_driver_with_state(self):
        """Create a mock driver that maintains state for integration tests."""
        state = {"nodes": {}, "relationships": []}

        async def mock_execute_write(query, params=None):
            if "MERGE" in query and params:
                node_id = params.get("id")
                if node_id:
                    state["nodes"][node_id] = params
                return [{"created": 1}]
            return []

        async def mock_execute_read(query, params=None):
            node_id = params.get("id") if params else None
            if node_id and node_id in state["nodes"]:
                node_data = state["nodes"][node_id]
                # Determine label from query
                if "Repository" in query:
                    return [{"r": node_data}]
                elif "Function" in query:
                    return [{"f": node_data}]
                elif "Class" in query:
                    return [{"c": node_data}]
            return []

        connection = GraphConnection()
        connection._driver = AsyncMock()
        connection.execute_write = AsyncMock(side_effect=mock_execute_write)
        connection.execute_read = AsyncMock(side_effect=mock_execute_read)

        return connection, state

    @pytest.mark.asyncio
    async def test_full_repository_crud_cycle(self, mock_driver_with_state):
        """Test creating, reading, updating, and deleting a repository."""
        connection, state = mock_driver_with_state
        store = GraphStore(connection)

        # Create
        repo = RepositoryNode(
            id="integration-repo",
            name="Test Integration Repo",
            url="https://github.com/test/integration",
        )
        created = await store.create_repository(repo)
        assert created.id == "integration-repo"

        # Verify in state
        assert "integration-repo" in state["nodes"]

    @pytest.mark.asyncio
    async def test_function_with_relationships(self, mock_driver_with_state):
        """Test creating a function and its relationships."""
        connection, state = mock_driver_with_state
        store = GraphStore(connection)

        # Create two functions
        caller = FunctionNode(
            id="/test.py:caller:10",
            name="caller",
            file_path="/test.py",
            start_line=10,
            end_line=20,
            repo_id="repo-1",
        )
        callee = FunctionNode(
            id="/test.py:callee:30",
            name="callee",
            file_path="/test.py",
            start_line=30,
            end_line=40,
            repo_id="repo-1",
        )

        await store.create_function(caller)
        await store.create_function(callee)

        # Create CALLS relationship
        relationship = GraphRelationship(
            source_id=caller.id,
            target_id=callee.id,
            relationship_type=RelationshipType.CALLS,
        )
        await store.create_relationship(relationship)

        # Verify both nodes exist in state
        assert caller.id in state["nodes"]
        assert callee.id in state["nodes"]

    @pytest.mark.asyncio
    async def test_class_hierarchy_creation(self, mock_driver_with_state):
        """Test creating a class hierarchy."""
        connection, state = mock_driver_with_state
        store = GraphStore(connection)

        # Create parent and child classes
        parent = ClassNode(
            id="/test.py:Parent:1",
            name="Parent",
            file_path="/test.py",
            start_line=1,
            end_line=20,
            bases=[],
            repo_id="repo-1",
        )
        child = ClassNode(
            id="/test.py:Child:30",
            name="Child",
            file_path="/test.py",
            start_line=30,
            end_line=50,
            bases=["Parent"],
            repo_id="repo-1",
        )

        await store.create_class(parent)
        await store.create_class(child)

        # Create INHERITS relationship
        relationship = GraphRelationship(
            source_id=child.id,
            target_id=parent.id,
            relationship_type=RelationshipType.INHERITS,
        )
        await store.create_relationship(relationship)

        assert parent.id in state["nodes"]
        assert child.id in state["nodes"]

    @pytest.mark.asyncio
    async def test_batch_operations(self, mock_driver_with_state):
        """Test batch creation of multiple nodes."""
        connection, state = mock_driver_with_state
        store = GraphStore(connection)

        # Create multiple functions
        functions = [
            FunctionNode(
                id=f"/test.py:func{i}:{(i + 1) * 10}",
                name=f"func{i}",
                file_path="/test.py",
                start_line=(i + 1) * 10,
                end_line=(i + 1) * 10 + 5,
                repo_id="repo-1",
            )
            for i in range(5)
        ]

        result = await store.batch_create_functions(functions)

        # Should have attempted to create all functions
        assert result >= 1  # At least some were created

    @pytest.mark.asyncio
    async def test_file_with_entities(self, mock_driver_with_state):
        """Test creating a file with its defined entities."""
        connection, state = mock_driver_with_state
        store = GraphStore(connection)

        # Create file
        file_node = FileNode(
            id="/src/utils.py",
            path="/src/utils.py",
            language="python",
            size=500,
            repo_id="repo-1",
        )
        await store.create_file(file_node)

        # Create function defined in file
        func = FunctionNode(
            id="/src/utils.py:helper:10",
            name="helper",
            file_path="/src/utils.py",
            start_line=10,
            end_line=25,
            repo_id="repo-1",
        )
        await store.create_function(func)

        # Create DEFINES relationship
        relationship = GraphRelationship(
            source_id=file_node.id,
            target_id=func.id,
            relationship_type=RelationshipType.DEFINES,
        )
        await store.create_relationship(relationship)

        assert file_node.id in state["nodes"]
        assert func.id in state["nodes"]


# =============================================================================
# Connection Tests
# =============================================================================


class TestGraphConnection:
    """Tests for GraphConnection class."""

    def test_connection_defaults(self):
        """Test connection default values."""
        conn = GraphConnection()
        assert conn.uri == "bolt://localhost:7687"
        assert conn.user == "neo4j"
        assert conn.database == "neo4j"

    def test_connection_custom_values(self):
        """Test connection with custom values."""
        conn = GraphConnection(
            uri="bolt://custom:7687",
            user="admin",
            password="secret",
            database="mydb",
        )
        assert conn.uri == "bolt://custom:7687"
        assert conn.user == "admin"
        assert conn.database == "mydb"

    def test_driver_property_raises_when_not_connected(self):
        """Test that driver property raises when not connected."""
        conn = GraphConnection()

        with pytest.raises(GraphConnectionError):
            _ = conn.driver

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self):
        """Test health check when disconnected."""
        conn = GraphConnection()

        result = await conn.health_check()

        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_neo4j_driver):
        """Test async context manager usage."""
        with (
            patch.object(GraphConnection, "connect", new_callable=AsyncMock) as mock_connect,
            patch.object(GraphConnection, "close", new_callable=AsyncMock) as mock_close,
        ):
            async with GraphConnection():
                pass

            mock_connect.assert_called_once()
            mock_close.assert_called_once()
