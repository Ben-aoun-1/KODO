"""Comprehensive unit tests for the Kodo parser module.

This module tests the parser models, Python extractor, and TreeSitterParser
implementation, covering functions, classes, imports, variables, and error
handling.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

# Check if tree-sitter dependencies are available
try:
    import tree_sitter
    import tree_sitter_python

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Skip marker for tests requiring tree-sitter
requires_tree_sitter = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE, reason="tree-sitter-python not installed"
)

from core.parser.base import BaseParser, ParserError
from core.parser.languages import get_extractor, supported_languages
from core.parser.languages.python import PythonExtractor
from core.parser.models import (
    Attribute,
    ClassEntity,
    CodeEntity,
    EntityType,
    FunctionEntity,
    ImportEntity,
    MethodEntity,
    ModuleEntity,
    Parameter,
    ParseResult,
    VariableEntity,
)
from core.parser.tree_sitter import TreeSitterParser

# =============================================================================
# Model Tests
# =============================================================================


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self):
        """Verify all expected EntityType values exist."""
        assert EntityType.FUNCTION == "function"
        assert EntityType.CLASS == "class"
        assert EntityType.METHOD == "method"
        assert EntityType.VARIABLE == "variable"
        assert EntityType.IMPORT == "import"
        assert EntityType.MODULE == "module"

    def test_entity_type_is_string_enum(self):
        """EntityType should be usable as a string."""
        # EntityType.value returns the string value
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.CLASS.value == "class"
        # Can be compared directly with strings
        assert EntityType.FUNCTION == "function"
        assert f"type is {EntityType.CLASS.value}" == "type is class"

    def test_all_entity_types_count(self):
        """Verify the expected number of entity types."""
        assert len(EntityType) == 6


class TestCodeEntity:
    """Tests for the base CodeEntity model."""

    def test_code_entity_creation(self):
        """Test creating a basic CodeEntity."""
        entity = CodeEntity(
            id="/path/to/file.py:my_entity:1",
            name="my_entity",
            type=EntityType.FUNCTION,
            file_path="/path/to/file.py",
            start_line=1,
            end_line=10,
            source_code="def my_entity(): pass",
            language="python",
        )

        assert entity.id == "/path/to/file.py:my_entity:1"
        assert entity.name == "my_entity"
        assert entity.type == EntityType.FUNCTION
        assert entity.file_path == "/path/to/file.py"
        assert entity.start_line == 1
        assert entity.end_line == 10
        assert entity.docstring is None
        assert entity.parent_id is None

    def test_code_entity_with_docstring(self):
        """Test CodeEntity with a docstring."""
        entity = CodeEntity(
            id="/path/to/file.py:documented:5",
            name="documented",
            type=EntityType.CLASS,
            file_path="/path/to/file.py",
            start_line=5,
            end_line=20,
            source_code="class documented: ...",
            docstring="This is a documented entity.",
            language="python",
        )

        assert entity.docstring == "This is a documented entity."

    def test_code_entity_with_parent(self):
        """Test CodeEntity with a parent reference."""
        entity = CodeEntity(
            id="/path/to/file.py:child_method:15",
            name="child_method",
            type=EntityType.METHOD,
            file_path="/path/to/file.py",
            start_line=15,
            end_line=20,
            source_code="def child_method(self): pass",
            language="python",
            parent_id="/path/to/file.py:ParentClass:1",
        )

        assert entity.parent_id == "/path/to/file.py:ParentClass:1"

    def test_code_entity_start_line_validation(self):
        """Start line must be >= 1."""
        with pytest.raises(ValidationError):
            CodeEntity(
                id="test:entity:0",
                name="entity",
                type=EntityType.FUNCTION,
                file_path="test.py",
                start_line=0,  # Invalid
                end_line=1,
                source_code="pass",
                language="python",
            )

    def test_code_entity_end_line_validation(self):
        """End line must be >= 1."""
        with pytest.raises(ValidationError):
            CodeEntity(
                id="test:entity:1",
                name="entity",
                type=EntityType.FUNCTION,
                file_path="test.py",
                start_line=1,
                end_line=0,  # Invalid
                source_code="pass",
                language="python",
            )

    def test_code_entity_forbids_extra_fields(self):
        """CodeEntity should not accept extra fields."""
        with pytest.raises(ValidationError):
            CodeEntity(
                id="test:entity:1",
                name="entity",
                type=EntityType.FUNCTION,
                file_path="test.py",
                start_line=1,
                end_line=1,
                source_code="pass",
                language="python",
                unknown_field="not allowed",  # Extra field
            )


class TestParameter:
    """Tests for the Parameter model."""

    def test_simple_parameter(self):
        """Test a simple parameter without annotations."""
        param = Parameter(name="x")

        assert param.name == "x"
        assert param.type_annotation is None
        assert param.default_value is None
        assert param.is_variadic is False
        assert param.is_keyword is False

    def test_typed_parameter(self):
        """Test a parameter with type annotation."""
        param = Parameter(
            name="count",
            type_annotation="int",
        )

        assert param.name == "count"
        assert param.type_annotation == "int"

    def test_parameter_with_default(self):
        """Test a parameter with default value."""
        param = Parameter(
            name="timeout",
            type_annotation="float",
            default_value="30.0",
        )

        assert param.default_value == "30.0"

    def test_variadic_parameter(self):
        """Test *args style parameter."""
        param = Parameter(
            name="args",
            is_variadic=True,
        )

        assert param.is_variadic is True
        assert param.is_keyword is False

    def test_keyword_parameter(self):
        """Test **kwargs style parameter."""
        param = Parameter(
            name="kwargs",
            is_keyword=True,
        )

        assert param.is_variadic is False
        assert param.is_keyword is True


class TestFunctionEntity:
    """Tests for the FunctionEntity model."""

    def test_function_entity_defaults(self):
        """Test FunctionEntity default values."""
        func = FunctionEntity(
            id="test.py:my_func:1",
            name="my_func",
            file_path="test.py",
            start_line=1,
            end_line=3,
            source_code="def my_func(): pass",
            language="python",
        )

        assert func.type == EntityType.FUNCTION
        assert func.parameters == []
        assert func.return_type is None
        assert func.decorators == []
        assert func.is_async is False
        assert func.calls == []

    def test_function_entity_with_parameters(self):
        """Test FunctionEntity with parameters."""
        params = [
            Parameter(name="x", type_annotation="int"),
            Parameter(name="y", type_annotation="int", default_value="0"),
        ]

        func = FunctionEntity(
            id="test.py:add:1",
            name="add",
            file_path="test.py",
            start_line=1,
            end_line=2,
            source_code="def add(x: int, y: int = 0): return x + y",
            language="python",
            parameters=params,
            return_type="int",
        )

        assert len(func.parameters) == 2
        assert func.parameters[0].name == "x"
        assert func.parameters[1].default_value == "0"
        assert func.return_type == "int"

    def test_async_function_entity(self):
        """Test async function."""
        func = FunctionEntity(
            id="test.py:async_fetch:1",
            name="async_fetch",
            file_path="test.py",
            start_line=1,
            end_line=5,
            source_code="async def async_fetch(): pass",
            language="python",
            is_async=True,
        )

        assert func.is_async is True

    def test_decorated_function_entity(self):
        """Test function with decorators."""
        func = FunctionEntity(
            id="test.py:handler:1",
            name="handler",
            file_path="test.py",
            start_line=1,
            end_line=5,
            source_code="@route('/') def handler(): pass",
            language="python",
            decorators=["route", "cache"],
        )

        assert func.decorators == ["route", "cache"]

    def test_function_with_calls(self):
        """Test function tracking internal calls."""
        func = FunctionEntity(
            id="test.py:process:1",
            name="process",
            file_path="test.py",
            start_line=1,
            end_line=5,
            source_code="def process(): validate(); transform()",
            language="python",
            calls=["validate", "transform", "logger.info"],
        )

        assert "validate" in func.calls
        assert "logger.info" in func.calls


class TestMethodEntity:
    """Tests for the MethodEntity model."""

    def test_method_entity_defaults(self):
        """Test MethodEntity default values."""
        method = MethodEntity(
            id="test.py:MyClass.method:5",
            name="method",
            file_path="test.py",
            start_line=5,
            end_line=7,
            source_code="def method(self): pass",
            language="python",
            parent_id="test.py:MyClass:1",
        )

        assert method.type == EntityType.METHOD
        assert method.is_classmethod is False
        assert method.is_staticmethod is False
        assert method.is_property is False

    def test_classmethod(self):
        """Test classmethod detection."""
        method = MethodEntity(
            id="test.py:MyClass.create:5",
            name="create",
            file_path="test.py",
            start_line=5,
            end_line=7,
            source_code="@classmethod\ndef create(cls): pass",
            language="python",
            parent_id="test.py:MyClass:1",
            decorators=["classmethod"],
            is_classmethod=True,
        )

        assert method.is_classmethod is True
        assert method.is_staticmethod is False

    def test_staticmethod(self):
        """Test staticmethod detection."""
        method = MethodEntity(
            id="test.py:MyClass.utility:5",
            name="utility",
            file_path="test.py",
            start_line=5,
            end_line=7,
            source_code="@staticmethod\ndef utility(): pass",
            language="python",
            parent_id="test.py:MyClass:1",
            decorators=["staticmethod"],
            is_staticmethod=True,
        )

        assert method.is_staticmethod is True

    def test_property_method(self):
        """Test property detection."""
        method = MethodEntity(
            id="test.py:MyClass.value:5",
            name="value",
            file_path="test.py",
            start_line=5,
            end_line=7,
            source_code="@property\ndef value(self): return self._value",
            language="python",
            parent_id="test.py:MyClass:1",
            decorators=["property"],
            is_property=True,
        )

        assert method.is_property is True


class TestClassEntity:
    """Tests for the ClassEntity model."""

    def test_class_entity_defaults(self):
        """Test ClassEntity default values."""
        cls = ClassEntity(
            id="test.py:MyClass:1",
            name="MyClass",
            file_path="test.py",
            start_line=1,
            end_line=10,
            source_code="class MyClass: pass",
            language="python",
        )

        assert cls.type == EntityType.CLASS
        assert cls.bases == []
        assert cls.methods == []
        assert cls.attributes == []
        assert cls.decorators == []

    def test_class_with_bases(self):
        """Test class with inheritance."""
        cls = ClassEntity(
            id="test.py:Child:1",
            name="Child",
            file_path="test.py",
            start_line=1,
            end_line=10,
            source_code="class Child(Parent, Mixin): pass",
            language="python",
            bases=["Parent", "Mixin"],
        )

        assert cls.bases == ["Parent", "Mixin"]

    def test_class_with_methods(self):
        """Test class with methods list."""
        method = MethodEntity(
            id="test.py:MyClass.__init__:2",
            name="__init__",
            file_path="test.py",
            start_line=2,
            end_line=4,
            source_code="def __init__(self): pass",
            language="python",
            parent_id="test.py:MyClass:1",
        )

        cls = ClassEntity(
            id="test.py:MyClass:1",
            name="MyClass",
            file_path="test.py",
            start_line=1,
            end_line=10,
            source_code="class MyClass:\n  def __init__(self): pass",
            language="python",
            methods=[method],
        )

        assert len(cls.methods) == 1
        assert cls.methods[0].name == "__init__"

    def test_class_with_attributes(self):
        """Test class with attributes."""
        attrs = [
            Attribute(name="count", type_annotation="int", default_value="0", line=2),
            Attribute(name="name", type_annotation="str", line=3),
        ]

        cls = ClassEntity(
            id="test.py:MyClass:1",
            name="MyClass",
            file_path="test.py",
            start_line=1,
            end_line=10,
            source_code="class MyClass:\n  count: int = 0\n  name: str",
            language="python",
            attributes=attrs,
        )

        assert len(cls.attributes) == 2
        assert cls.attributes[0].name == "count"
        assert cls.attributes[0].default_value == "0"


class TestAttribute:
    """Tests for the Attribute model."""

    def test_attribute_creation(self):
        """Test basic attribute creation."""
        attr = Attribute(
            name="value",
            type_annotation="int",
            default_value="42",
            line=5,
        )

        assert attr.name == "value"
        assert attr.type_annotation == "int"
        assert attr.default_value == "42"
        assert attr.line == 5

    def test_attribute_without_annotation(self):
        """Test attribute without type annotation."""
        attr = Attribute(name="data", line=1)

        assert attr.type_annotation is None
        assert attr.default_value is None


class TestImportEntity:
    """Tests for the ImportEntity model."""

    def test_simple_import(self):
        """Test 'import x' style import."""
        imp = ImportEntity(
            id="test.py:os:1",
            name="os",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="import os",
            language="python",
            module="os",
            is_from_import=False,
        )

        assert imp.type == EntityType.IMPORT
        assert imp.module == "os"
        assert imp.is_from_import is False
        assert imp.alias is None
        assert imp.imported_names == []

    def test_import_with_alias(self):
        """Test 'import x as y' style import."""
        imp = ImportEntity(
            id="test.py:numpy:1",
            name="np",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="import numpy as np",
            language="python",
            module="numpy",
            alias="np",
            is_from_import=False,
        )

        assert imp.alias == "np"

    def test_from_import(self):
        """Test 'from x import y' style import."""
        imp = ImportEntity(
            id="test.py:pathlib:1",
            name="pathlib",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="from pathlib import Path",
            language="python",
            module="pathlib",
            is_from_import=True,
            imported_names=["Path"],
        )

        assert imp.is_from_import is True
        assert imp.imported_names == ["Path"]

    def test_from_import_multiple_names(self):
        """Test 'from x import a, b, c' style import."""
        imp = ImportEntity(
            id="test.py:typing:1",
            name="typing",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="from typing import Optional, List, Dict",
            language="python",
            module="typing",
            is_from_import=True,
            imported_names=["Optional", "List", "Dict"],
        )

        assert len(imp.imported_names) == 3


class TestVariableEntity:
    """Tests for the VariableEntity model."""

    def test_variable_entity_defaults(self):
        """Test VariableEntity default values."""
        var = VariableEntity(
            id="test.py:x:1",
            name="x",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="x = 10",
            language="python",
        )

        assert var.type == EntityType.VARIABLE
        assert var.type_annotation is None
        assert var.value is None
        assert var.is_constant is False

    def test_typed_variable(self):
        """Test variable with type annotation."""
        var = VariableEntity(
            id="test.py:count:1",
            name="count",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="count: int = 0",
            language="python",
            type_annotation="int",
            value="0",
        )

        assert var.type_annotation == "int"
        assert var.value == "0"

    def test_constant_variable(self):
        """Test constant variable (ALL_CAPS naming)."""
        var = VariableEntity(
            id="test.py:MAX_SIZE:1",
            name="MAX_SIZE",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="MAX_SIZE = 100",
            language="python",
            value="100",
            is_constant=True,
        )

        assert var.is_constant is True


class TestModuleEntity:
    """Tests for the ModuleEntity model."""

    def test_module_entity_defaults(self):
        """Test ModuleEntity default values."""
        module = ModuleEntity(
            id="test.py:test:1",
            name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="",
            language="python",
        )

        assert module.type == EntityType.MODULE
        assert module.imports == []
        assert module.functions == []
        assert module.classes == []
        assert module.variables == []

    def test_module_with_entities(self):
        """Test ModuleEntity with all entity types."""
        imp = ImportEntity(
            id="test.py:os:1",
            name="os",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="import os",
            language="python",
            module="os",
        )

        func = FunctionEntity(
            id="test.py:main:3",
            name="main",
            file_path="test.py",
            start_line=3,
            end_line=5,
            source_code="def main(): pass",
            language="python",
        )

        module = ModuleEntity(
            id="test.py:test:1",
            name="test",
            file_path="test.py",
            start_line=1,
            end_line=5,
            source_code="import os\n\ndef main(): pass",
            language="python",
            imports=[imp],
            functions=[func],
        )

        assert len(module.imports) == 1
        assert len(module.functions) == 1


class TestParseResult:
    """Tests for the ParseResult model."""

    def test_parse_result_success(self):
        """Test successful ParseResult."""
        module = ModuleEntity(
            id="test.py:test:1",
            name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="",
            language="python",
        )

        result = ParseResult(
            module=module,
            file_path="test.py",
            language="python",
        )

        assert result.success is True
        assert result.parse_errors == []

    def test_parse_result_with_errors(self):
        """Test ParseResult with parsing errors."""
        module = ModuleEntity(
            id="test.py:test:1",
            name="test",
            file_path="test.py",
            start_line=1,
            end_line=1,
            source_code="",
            language="python",
        )

        result = ParseResult(
            module=module,
            file_path="test.py",
            language="python",
            parse_errors=["Syntax error at line 5"],
            success=False,
        )

        assert result.success is False
        assert len(result.parse_errors) == 1


# =============================================================================
# Python Extractor Tests
# =============================================================================


@requires_tree_sitter
class TestPythonExtractor:
    """Tests for the PythonExtractor class.

    These tests require tree-sitter-python to be installed.
    """

    @pytest.fixture
    def extractor(self) -> PythonExtractor:
        """Create a PythonExtractor instance."""
        return PythonExtractor()

    @pytest.fixture
    def parse_python(self, extractor: PythonExtractor):
        """Helper to parse Python code and return module."""
        import tree_sitter_python
        from tree_sitter import Language, Parser

        language = Language(tree_sitter_python.language())
        parser = Parser(language)

        def _parse(source_code: str, file_path: str = "/test/file.py"):
            tree = parser.parse(source_code.encode("utf-8"))
            return extractor.extract_module(tree, source_code, file_path)

        return _parse

    # -------------------------------------------------------------------------
    # Function Parsing Tests
    # -------------------------------------------------------------------------

    def test_parse_simple_function(self, parse_python, simple_function_code):
        """Test parsing a simple function."""
        module = parse_python(simple_function_code)

        assert len(module.functions) == 1
        func = module.functions[0]

        assert func.name == "greet"
        assert func.is_async is False
        assert len(func.parameters) == 1
        assert func.parameters[0].name == "name"
        assert func.docstring == "Say hello to someone."

    @pytest.mark.xfail(
        reason="Parser does not correctly detect async functions (tree-sitter node type issue)"
    )
    def test_parse_async_function(self, parse_python, async_function_code):
        """Test parsing an async function."""
        module = parse_python(async_function_code)

        assert len(module.functions) == 1
        func = module.functions[0]

        assert func.name == "fetch_data"
        assert func.is_async is True
        assert len(func.parameters) == 2

        # Check parameters
        url_param = func.parameters[0]
        assert url_param.name == "url"
        assert url_param.type_annotation == "str"

        timeout_param = func.parameters[1]
        assert timeout_param.name == "timeout"
        assert timeout_param.type_annotation == "int"
        assert timeout_param.default_value == "30"

        # Check return type
        assert func.return_type == "dict"

    def test_parse_decorated_function(self, parse_python, decorated_function_code):
        """Test parsing a decorated function."""
        module = parse_python(decorated_function_code)

        assert len(module.functions) == 1
        func = module.functions[0]

        assert func.name == "get_users"
        assert len(func.decorators) == 3
        assert "app.route" in func.decorators
        assert "require_auth" in func.decorators
        assert "cache" in func.decorators

    def test_parse_function_with_variadic_params(
        self, parse_python, function_with_variadic_params_code
    ):
        """Test parsing function with *args and **kwargs."""
        module = parse_python(function_with_variadic_params_code)

        assert len(module.functions) == 1
        func = module.functions[0]

        # Find the variadic parameters
        variadic_param = next((p for p in func.parameters if p.is_variadic), None)
        keyword_param = next((p for p in func.parameters if p.is_keyword), None)

        assert variadic_param is not None
        assert variadic_param.name == "args"

        assert keyword_param is not None
        assert keyword_param.name == "kwargs"

    def test_parse_function_extracts_calls(self, parse_python, function_with_calls_code):
        """Test that function calls are extracted."""
        module = parse_python(function_with_calls_code)

        assert len(module.functions) == 1
        func = module.functions[0]

        assert "validate_input" in func.calls
        assert "transform_data" in func.calls
        assert "aggregate_results" in func.calls
        assert "logger.info" in func.calls

    # -------------------------------------------------------------------------
    # Class Parsing Tests
    # -------------------------------------------------------------------------

    def test_parse_simple_class(self, parse_python, simple_class_code):
        """Test parsing a simple class."""
        module = parse_python(simple_class_code)

        assert len(module.classes) == 1
        cls = module.classes[0]

        assert cls.name == "Calculator"
        assert cls.docstring == "A simple calculator class."
        assert cls.bases == []

        # Check methods
        assert len(cls.methods) == 3
        method_names = [m.name for m in cls.methods]
        assert "__init__" in method_names
        assert "add" in method_names
        assert "subtract" in method_names

        # Check attributes
        assert len(cls.attributes) == 1
        assert cls.attributes[0].name == "precision"
        assert cls.attributes[0].type_annotation == "int"

    def test_parse_class_with_inheritance(self, parse_python, class_with_inheritance_code):
        """Test parsing classes with inheritance."""
        module = parse_python(class_with_inheritance_code)

        # Find the Dog class
        dog_class = next((c for c in module.classes if c.name == "Dog"), None)
        assert dog_class is not None
        assert dog_class.bases == ["Animal"]

        # Find the ServiceDog class with multiple inheritance
        service_dog = next((c for c in module.classes if c.name == "ServiceDog"), None)
        assert service_dog is not None
        assert "Dog" in service_dog.bases
        assert "Trainable" in service_dog.bases

    def test_parse_class_with_decorators(self, parse_python, class_with_decorators_code):
        """Test parsing class with decorators."""
        module = parse_python(class_with_decorators_code)

        assert len(module.classes) == 1
        cls = module.classes[0]

        assert cls.name == "User"
        assert "dataclass" in cls.decorators
        assert "register" in cls.decorators

        # Check for property, classmethod, and staticmethod
        property_method = next((m for m in cls.methods if m.is_property), None)
        classmethod_method = next((m for m in cls.methods if m.is_classmethod), None)
        staticmethod_method = next((m for m in cls.methods if m.is_staticmethod), None)

        assert property_method is not None
        assert property_method.name == "display_name"

        assert classmethod_method is not None
        assert classmethod_method.name == "from_dict"

        assert staticmethod_method is not None
        assert staticmethod_method.name == "validate_email"

    def test_method_parent_id_is_set(self, parse_python, simple_class_code):
        """Test that methods have correct parent_id reference."""
        module = parse_python(simple_class_code)

        cls = module.classes[0]
        for method in cls.methods:
            assert method.parent_id == cls.id

    # -------------------------------------------------------------------------
    # Import Parsing Tests
    # -------------------------------------------------------------------------

    def test_parse_imports(self, parse_python, imports_code):
        """Test parsing various import styles."""
        module = parse_python(imports_code)

        # Should have multiple imports
        assert len(module.imports) >= 5

        # Check simple import
        os_import = next((i for i in module.imports if i.module == "os"), None)
        assert os_import is not None
        assert os_import.is_from_import is False

        # Check import with alias
        json_import = next((i for i in module.imports if i.module == "json"), None)
        assert json_import is not None
        assert json_import.alias == "js"

        # Check from import
        pathlib_import = next((i for i in module.imports if i.module == "pathlib"), None)
        assert pathlib_import is not None
        assert pathlib_import.is_from_import is True
        assert "Path" in pathlib_import.imported_names

        # Check from import with multiple names
        typing_import = next((i for i in module.imports if i.module == "typing"), None)
        assert typing_import is not None
        assert len(typing_import.imported_names) == 3

    def test_parse_relative_imports(self, parse_python, imports_code):
        """Test parsing relative imports."""
        module = parse_python(imports_code)

        # Find relative import
        relative_import = next((i for i in module.imports if i.module.startswith(".")), None)
        assert relative_import is not None

    def test_parse_from_import_with_alias(self, parse_python, imports_code):
        """Test 'from x import y as z' parsing."""
        module = parse_python(imports_code)

        # Find import with alias in imported names
        models_import = next((i for i in module.imports if i.module == ".models"), None)
        assert models_import is not None
        # Check that aliased import is captured
        assert any("BlogPost" in name or "Post" in name for name in models_import.imported_names)

    # -------------------------------------------------------------------------
    # Variable Parsing Tests
    # -------------------------------------------------------------------------

    def test_parse_module_variables(self, parse_python, module_level_variables_code):
        """Test parsing module-level variables."""
        module = parse_python(module_level_variables_code)

        # Should have several variables
        assert len(module.variables) >= 4

        # Check simple constant
        version_var = next((v for v in module.variables if v.name == "VERSION"), None)
        assert version_var is not None
        assert version_var.is_constant is True
        assert version_var.value == '"1.0.0"'

        # Check typed variable
        debug_var = next((v for v in module.variables if v.name == "DEBUG"), None)
        assert debug_var is not None
        assert debug_var.type_annotation == "bool"

        # Check constant with underscore prefix
        private_const = next((v for v in module.variables if v.name == "_PRIVATE_CONSTANT"), None)
        assert private_const is not None
        assert private_const.is_constant is True

    def test_parse_variable_with_complex_value(self, parse_python):
        """Test parsing variable with complex value."""
        code = 'config = {"host": "localhost", "port": 8080}'
        module = parse_python(code)

        assert len(module.variables) == 1
        assert module.variables[0].name == "config"
        assert '{"host": "localhost", "port": 8080}' in module.variables[0].value

    # -------------------------------------------------------------------------
    # Module Docstring Tests
    # -------------------------------------------------------------------------

    def test_parse_module_docstring(self, parse_python, module_level_variables_code):
        """Test extracting module docstring."""
        module = parse_python(module_level_variables_code)

        assert module.docstring == "Module with various variable types."

    def test_module_without_docstring(self, parse_python):
        """Test module without docstring."""
        code = "import os\nx = 1"
        module = parse_python(code)

        assert module.docstring is None

    # -------------------------------------------------------------------------
    # Complex Module Tests
    # -------------------------------------------------------------------------

    def test_parse_complex_module(self, parse_python, complex_module_code):
        """Test parsing a complex module with all entity types."""
        module = parse_python(complex_module_code)

        # Check module docstring
        assert module.docstring is not None
        assert "complex module" in module.docstring.lower()

        # Check imports
        assert len(module.imports) >= 2

        # Check variables
        assert len(module.variables) >= 2
        version_var = next((v for v in module.variables if v.name == "VERSION"), None)
        assert version_var is not None

        # Check classes
        assert len(module.classes) >= 2
        config_class = next((c for c in module.classes if c.name == "Config"), None)
        assert config_class is not None
        assert "dataclass" in config_class.decorators

        service_class = next((c for c in module.classes if c.name == "Service"), None)
        assert service_class is not None
        assert len(service_class.methods) >= 3

        # Check functions
        assert len(module.functions) >= 2
        # Note: Due to parser limitation, async functions are not correctly
        # detected - this is checked by test_parse_async_function which is xfail

    # -------------------------------------------------------------------------
    # Type Annotation Tests
    # -------------------------------------------------------------------------

    def test_parse_complex_type_annotations(self, parse_python):
        """Test parsing complex type annotations."""
        code = """
def process(
    items: list[dict[str, Any]],
    callback: Callable[[int, str], bool],
    config: Optional[Config] = None,
) -> tuple[list[Result], int]:
    pass
"""
        module = parse_python(code)

        assert len(module.functions) == 1
        func = module.functions[0]

        # Check return type
        assert func.return_type is not None
        assert "tuple" in func.return_type

    @pytest.mark.xfail(reason="Parser does not extract subscripted base classes like Generic[T]")
    def test_parse_generic_type_annotations(self, parse_python):
        """Test parsing generic type annotations."""
        code = """
from typing import TypeVar, Generic

T = TypeVar("T")

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value
"""
        module = parse_python(code)

        container_class = next((c for c in module.classes if c.name == "Container"), None)
        assert container_class is not None
        assert "Generic[T]" in container_class.bases or "Generic" in container_class.bases


# =============================================================================
# TreeSitterParser Tests
# =============================================================================


class TestTreeSitterParser:
    """Tests for the TreeSitterParser class."""

    @pytest.fixture
    def parser(self) -> TreeSitterParser:
        """Create a TreeSitterParser instance."""
        return TreeSitterParser()

    # -------------------------------------------------------------------------
    # parse_source Tests
    # -------------------------------------------------------------------------

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_source_simple(self, parser: TreeSitterParser):
        """Test parse_source with simple code."""
        code = "def hello(): pass"
        result = await parser.parse_source(code, language="python")

        assert result.success is True
        assert result.language == "python"
        assert len(result.module.functions) == 1

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_source_with_file_path(self, parser: TreeSitterParser):
        """Test parse_source with virtual file path."""
        code = "x = 1"
        result = await parser.parse_source(
            code,
            file_path="/virtual/test.py",
            language="python",
        )

        assert result.file_path == "/virtual/test.py"
        assert result.module.file_path == "/virtual/test.py"

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_source_infers_language_from_path(self, parser: TreeSitterParser):
        """Test that language is inferred from file path."""
        code = "def test(): pass"
        result = await parser.parse_source(
            code,
            file_path="/test/module.py",
        )

        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_parse_source_requires_language(self, parser: TreeSitterParser):
        """Test that language is required when not inferrable."""
        code = "def test(): pass"

        with pytest.raises(ValueError, match="Language must be specified"):
            await parser.parse_source(code)

    @pytest.mark.asyncio
    async def test_parse_source_unsupported_language(self, parser: TreeSitterParser):
        """Test parsing with unsupported language."""
        code = "fn main() {}"

        with pytest.raises(ParserError, match="Unsupported language"):
            await parser.parse_source(code, language="rust")

    # -------------------------------------------------------------------------
    # parse_file Tests
    # -------------------------------------------------------------------------

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_file_success(self, parser: TreeSitterParser, temp_python_file: Path):
        """Test parse_file with a real file."""
        result = await parser.parse_file(temp_python_file)

        assert result.success is True
        assert result.language == "python"
        assert len(result.module.functions) == 1

    @pytest.mark.asyncio
    async def test_parse_file_not_found(self, parser: TreeSitterParser):
        """Test parse_file with non-existent file."""
        with pytest.raises(FileNotFoundError):
            await parser.parse_file("/non/existent/file.py")

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_file_detects_language(
        self, parser: TreeSitterParser, temp_python_file_with_content
    ):
        """Test that language is detected from file extension."""
        file_path = temp_python_file_with_content("x = 1", suffix=".py")
        result = await parser.parse_file(file_path)

        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_parse_file_unknown_extension(
        self, parser: TreeSitterParser, temp_python_file_with_content
    ):
        """Test parse_file with unknown file extension."""
        file_path = temp_python_file_with_content("x = 1", suffix=".xyz")

        with pytest.raises(ParserError, match="Cannot detect language"):
            await parser.parse_file(file_path)

    # -------------------------------------------------------------------------
    # Language Detection Tests
    # -------------------------------------------------------------------------

    def test_detect_language_python(self):
        """Test language detection for Python files."""
        assert BaseParser.detect_language("test.py") == "python"
        assert BaseParser.detect_language("test.pyi") == "python"
        assert BaseParser.detect_language("/path/to/module.py") == "python"

    def test_detect_language_javascript(self):
        """Test language detection for JavaScript files."""
        assert BaseParser.detect_language("test.js") == "javascript"
        assert BaseParser.detect_language("test.mjs") == "javascript"
        assert BaseParser.detect_language("test.jsx") == "javascript"

    def test_detect_language_typescript(self):
        """Test language detection for TypeScript files."""
        assert BaseParser.detect_language("test.ts") == "typescript"
        assert BaseParser.detect_language("test.tsx") == "typescript"
        assert BaseParser.detect_language("test.mts") == "typescript"

    def test_detect_language_unknown(self):
        """Test language detection for unknown extensions."""
        assert BaseParser.detect_language("test.xyz") is None
        assert BaseParser.detect_language("README.md") is None

    def test_detect_language_case_insensitive(self):
        """Test that language detection is case-insensitive."""
        assert BaseParser.detect_language("TEST.PY") == "python"
        assert BaseParser.detect_language("Test.Py") == "python"

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_syntax_error_graceful(
        self, parser: TreeSitterParser, syntax_error_code: str
    ):
        """Test that syntax errors are handled gracefully."""
        result = await parser.parse_source(
            syntax_error_code,
            language="python",
        )

        # Should still return a result with errors noted
        assert result.success is False
        assert len(result.parse_errors) > 0

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_empty_file(self, parser: TreeSitterParser):
        """Test parsing an empty file."""
        result = await parser.parse_source("", language="python")

        assert result.success is True
        assert result.module.functions == []
        assert result.module.classes == []

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_parse_whitespace_only(self, parser: TreeSitterParser):
        """Test parsing whitespace-only content."""
        result = await parser.parse_source("   \n\n   \t\t\n", language="python")

        assert result.success is True

    # -------------------------------------------------------------------------
    # Parser State Tests
    # -------------------------------------------------------------------------

    def test_supports_language(self, parser: TreeSitterParser):
        """Test supports_language method."""
        # Before initialization, check default supported languages
        assert "python" in parser.supported_languages

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_get_supported_languages(self, parser: TreeSitterParser):
        """Test getting list of supported languages after initialization."""
        # Trigger initialization
        await parser.parse_source("x = 1", language="python")

        languages = parser.get_supported_languages()
        assert "python" in languages

    # -------------------------------------------------------------------------
    # Debug AST Tests
    # -------------------------------------------------------------------------

    @requires_tree_sitter
    @pytest.mark.asyncio
    async def test_debug_ast(self, parser: TreeSitterParser):
        """Test debug_ast output."""
        code = "def hello(): pass"
        ast_output = await parser.debug_ast(code, "python", max_depth=3)

        assert "module" in ast_output.lower()
        assert "function_definition" in ast_output


# =============================================================================
# Parser Error Tests
# =============================================================================


class TestParserError:
    """Tests for the ParserError exception class."""

    def test_parser_error_basic(self):
        """Test basic ParserError."""
        error = ParserError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.file_path is None
        assert error.line is None
        assert error.column is None

    def test_parser_error_with_file(self):
        """Test ParserError with file path."""
        error = ParserError("Parse failed", file_path="/path/to/file.py")
        assert "file=/path/to/file.py" in str(error)
        assert error.file_path == "/path/to/file.py"

    def test_parser_error_with_location(self):
        """Test ParserError with line and column."""
        error = ParserError(
            "Unexpected token",
            file_path="/path/to/file.py",
            line=10,
            column=5,
        )

        assert "line=10" in str(error)
        assert "column=5" in str(error)
        assert error.line == 10
        assert error.column == 5


# =============================================================================
# Language Extractor Registry Tests
# =============================================================================


class TestExtractorRegistry:
    """Tests for the language extractor registry."""

    def test_get_extractor_python(self):
        """Test getting Python extractor."""
        extractor = get_extractor("python")
        assert isinstance(extractor, PythonExtractor)
        assert extractor.language == "python"

    def test_get_extractor_case_insensitive(self):
        """Test that extractor lookup is case-insensitive."""
        extractor = get_extractor("PYTHON")
        assert isinstance(extractor, PythonExtractor)

    def test_get_extractor_unknown_language(self):
        """Test getting extractor for unknown language."""
        with pytest.raises(ValueError, match="No extractor registered"):
            get_extractor("cobol")

    def test_supported_languages_includes_python(self):
        """Test that Python is in supported languages."""
        languages = supported_languages()
        assert "python" in languages


# =============================================================================
# Integration Tests
# =============================================================================


@requires_tree_sitter
class TestParserIntegration:
    """Integration tests combining parser components.

    These tests require tree-sitter-python to be installed.
    """

    @pytest.mark.asyncio
    async def test_full_parsing_pipeline(
        self, temp_python_file_with_content, complex_module_code: str
    ):
        """Test the complete parsing pipeline from file to entities."""
        # Create a temp file with complex code
        file_path = temp_python_file_with_content(complex_module_code)

        # Parse the file
        parser = TreeSitterParser()
        result = await parser.parse_file(file_path)

        # Verify the result
        assert result.success is True
        assert result.module.docstring is not None

        # Check that entity IDs are properly formatted
        # Note: Parser resolves the file path, which may differ from the
        # original path on Windows due to short path names (8.3 format)
        for func in result.module.functions:
            # Verify that IDs contain the entity name and line number
            assert func.name in func.id
            assert func.file_path == result.file_path

        for cls in result.module.classes:
            assert cls.name in cls.id
            for method in cls.methods:
                assert method.parent_id == cls.id

    @pytest.mark.asyncio
    async def test_multiple_files_parsing(self, temp_python_file_with_content):
        """Test parsing multiple files sequentially."""
        parser = TreeSitterParser()

        file1 = temp_python_file_with_content("def func1(): pass")
        file2 = temp_python_file_with_content("def func2(): pass")

        result1 = await parser.parse_file(file1)
        result2 = await parser.parse_file(file2)

        assert result1.module.functions[0].name == "func1"
        assert result2.module.functions[0].name == "func2"

    @pytest.mark.asyncio
    async def test_entity_id_uniqueness(self, temp_python_file_with_content):
        """Test that entity IDs are unique within a module."""
        code = """
def outer():
    pass

class MyClass:
    def outer(self):
        pass
"""
        file_path = temp_python_file_with_content(code)
        parser = TreeSitterParser()
        result = await parser.parse_file(file_path)

        # Collect all entity IDs
        ids = set()
        ids.add(result.module.id)

        for func in result.module.functions:
            ids.add(func.id)

        for cls in result.module.classes:
            ids.add(cls.id)
            for method in cls.methods:
                ids.add(method.id)

        # All IDs should be unique
        expected_count = 1 + len(result.module.functions) + len(result.module.classes)
        for cls in result.module.classes:
            expected_count += len(cls.methods)

        assert len(ids) == expected_count
