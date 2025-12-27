"""Python language extractor using tree-sitter.

This module provides the Python-specific entity extractor that traverses
tree-sitter ASTs to extract functions, classes, methods, imports, and
variables from Python source code.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import structlog

logger = structlog.get_logger(__name__)

from ..models import (
    Attribute,
    ClassEntity,
    EntityType,
    FunctionEntity,
    ImportEntity,
    MethodEntity,
    ModuleEntity,
    Parameter,
    VariableEntity,
)
from .base import BaseExtractor

if TYPE_CHECKING:
    from tree_sitter import Node, Tree


class PythonExtractor(BaseExtractor):
    """Python-specific entity extractor using tree-sitter.

    Extracts all code entities from Python source files including:
    - Functions (sync and async, with decorators)
    - Classes (with inheritance, methods, attributes)
    - Methods (instance, class, static, properties)
    - Imports (import and from...import styles)
    - Variables (module-level assignments)
    - Module docstrings

    Attributes:
        language: The language identifier ('python').
    """

    language: str = "python"

    def extract_module(
        self,
        tree: "Tree",
        source_code: str,
        file_path: str,
    ) -> ModuleEntity:
        """Extract a complete module entity from the Python AST.

        Args:
            tree: The tree-sitter parse tree.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            A ModuleEntity containing all extracted entities.
        """
        root_node = tree.root_node
        lines = source_code.splitlines()
        line_count = len(lines) if lines else 1

        # Extract module name from file path
        path = Path(file_path)
        module_name = path.stem

        # Extract module docstring
        docstring = self._extract_module_docstring(root_node, source_code)

        # Extract all top-level entities
        imports = self._extract_all_imports(root_node, source_code, file_path)
        functions = self._extract_top_level_functions(root_node, source_code, file_path)
        classes = self._extract_all_classes(root_node, source_code, file_path)
        variables = self._extract_module_variables(root_node, source_code, file_path)

        module_id = self.generate_entity_id(file_path, module_name, 1)

        return ModuleEntity(
            id=module_id,
            name=module_name,
            type=EntityType.MODULE,
            file_path=file_path,
            start_line=1,
            end_line=line_count,
            source_code=source_code,
            docstring=docstring,
            language=self.language,
            parent_id=None,
            imports=imports,
            functions=functions,
            classes=classes,
            variables=variables,
        )

    def extract_functions(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[FunctionEntity]:
        """Extract function definitions from a node.

        Args:
            node: The tree-sitter node to extract from.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of FunctionEntity objects found in the node.
        """
        return self._extract_top_level_functions(node, source_code, file_path)

    def extract_classes(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[ClassEntity]:
        """Extract class definitions from a node.

        Args:
            node: The tree-sitter node to extract from.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of ClassEntity objects found in the node.
        """
        return self._extract_all_classes(node, source_code, file_path)

    def extract_imports(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[ImportEntity]:
        """Extract import statements from a node.

        Args:
            node: The tree-sitter node to extract from.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of ImportEntity objects found in the node.
        """
        return self._extract_all_imports(node, source_code, file_path)

    def extract_variables(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[VariableEntity]:
        """Extract variable definitions from a node.

        Args:
            node: The tree-sitter node to extract from.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of VariableEntity objects found in the node.
        """
        return self._extract_module_variables(node, source_code, file_path)

    # -------------------------------------------------------------------------
    # Module Docstring Extraction
    # -------------------------------------------------------------------------

    def _extract_module_docstring(
        self,
        root_node: "Node",
        source_code: str,
    ) -> str | None:
        """Extract the module-level docstring if present.

        Args:
            root_node: The root node of the AST.
            source_code: The original source code.

        Returns:
            The docstring text if found, None otherwise.
        """
        for child in root_node.children:
            # Skip comments and whitespace
            if child.type in ("comment", "newline"):
                continue

            # Check for expression_statement containing a string
            if child.type == "expression_statement":
                for expr_child in child.children:
                    if expr_child.type == "string":
                        return self._parse_string_literal(expr_child, source_code)
                break
            else:
                # First non-comment statement is not a docstring
                break

        return None

    def _parse_string_literal(self, node: "Node", source_code: str) -> str:
        """Parse a string literal node and return its content.

        Handles both single and triple-quoted strings.

        Args:
            node: The string node.
            source_code: The original source code.

        Returns:
            The string content without quotes.
        """
        text = self.get_node_text(node, source_code)

        # Handle triple-quoted strings
        for quote in ['"""', "'''"]:
            if text.startswith(quote) and text.endswith(quote):
                return text[3:-3].strip()

        # Handle single-quoted strings
        for quote in ['"', "'"]:
            if text.startswith(quote) and text.endswith(quote):
                return text[1:-1].strip()

        # Handle f-strings and other prefixed strings
        for prefix in ["f", "r", "b", "fr", "rf", "br", "rb"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix) :]
                break

        for quote in ['"""', "'''", '"', "'"]:
            if text.startswith(quote) and text.endswith(quote):
                quote_len = len(quote)
                return text[quote_len:-quote_len].strip()

        return text.strip()

    # -------------------------------------------------------------------------
    # Import Extraction
    # -------------------------------------------------------------------------

    def _extract_all_imports(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[ImportEntity]:
        """Extract all import statements from the AST.

        Args:
            node: The node to extract from (usually root).
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of ImportEntity objects.
        """
        imports: list[ImportEntity] = []

        for child in node.children:
            if child.type == "import_statement":
                imports.extend(self._extract_import_statement(child, source_code, file_path))
            elif child.type == "import_from_statement":
                imports.extend(self._extract_from_import_statement(child, source_code, file_path))

        return imports

    def _extract_import_statement(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[ImportEntity]:
        """Extract 'import x' style imports.

        Args:
            node: The import_statement node.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of ImportEntity objects (one per imported name).
        """
        imports: list[ImportEntity] = []
        start_line, end_line = self.get_node_line_range(node)
        import_source = self.get_node_text(node, source_code)

        for child in node.children:
            if child.type == "dotted_name":
                module_name = self.get_node_text(child, source_code)
                entity_id = self.generate_entity_id(file_path, module_name, start_line)

                imports.append(
                    ImportEntity(
                        id=entity_id,
                        name=module_name,
                        type=EntityType.IMPORT,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        source_code=import_source,
                        docstring=None,
                        language=self.language,
                        parent_id=None,
                        module=module_name,
                        alias=None,
                        is_from_import=False,
                        imported_names=[],
                    )
                )
            elif child.type == "aliased_import":
                module_name = None
                alias = None
                for aliased_child in child.children:
                    if aliased_child.type == "dotted_name":
                        module_name = self.get_node_text(aliased_child, source_code)
                    elif aliased_child.type == "identifier":
                        alias = self.get_node_text(aliased_child, source_code)

                if module_name:
                    entity_id = self.generate_entity_id(file_path, module_name, start_line)
                    imports.append(
                        ImportEntity(
                            id=entity_id,
                            name=alias or module_name,
                            type=EntityType.IMPORT,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            source_code=import_source,
                            docstring=None,
                            language=self.language,
                            parent_id=None,
                            module=module_name,
                            alias=alias,
                            is_from_import=False,
                            imported_names=[],
                        )
                    )

        return imports

    def _extract_from_import_statement(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[ImportEntity]:
        """Extract 'from x import y' style imports.

        Args:
            node: The import_from_statement node.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of ImportEntity objects.
        """
        start_line, end_line = self.get_node_line_range(node)
        import_source = self.get_node_text(node, source_code)

        # Find the module name and imported names
        module_name = ""
        imported_names: list[str] = []
        alias: str | None = None
        found_import_keyword = False

        for child in node.children:
            if child.type == "from":
                continue
            elif child.type == "import":
                found_import_keyword = True
                continue
            elif child.type == "dotted_name":
                name_text = self.get_node_text(child, source_code)
                if not found_import_keyword:
                    # This is the module name (before 'import' keyword)
                    module_name = name_text
                else:
                    # This is an imported name (after 'import' keyword)
                    imported_names.append(name_text)
            elif child.type == "relative_import":
                # Handle relative imports like 'from . import x' or 'from ..foo import bar'
                dots = ""
                name = ""
                for rel_child in child.children:
                    if rel_child.type == "import_prefix":
                        dots = self.get_node_text(rel_child, source_code)
                    elif rel_child.type == "dotted_name":
                        name = self.get_node_text(rel_child, source_code)
                module_name = dots + name
            elif child.type == "aliased_import":
                name = None
                name_alias = None
                for aliased_child in child.children:
                    if aliased_child.type in ("identifier", "dotted_name"):
                        if name is None:
                            name = self.get_node_text(aliased_child, source_code)
                        else:
                            name_alias = self.get_node_text(aliased_child, source_code)
                if name:
                    if name_alias:
                        imported_names.append(f"{name} as {name_alias}")
                    else:
                        imported_names.append(name)
            elif child.type == "wildcard_import":
                imported_names.append("*")

        # Create a single import entity for the from import
        entity_id = self.generate_entity_id(file_path, module_name, start_line)

        return [
            ImportEntity(
                id=entity_id,
                name=module_name,
                type=EntityType.IMPORT,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                source_code=import_source,
                docstring=None,
                language=self.language,
                parent_id=None,
                module=module_name,
                alias=alias,
                is_from_import=True,
                imported_names=imported_names,
            )
        ]

    # -------------------------------------------------------------------------
    # Function Extraction
    # -------------------------------------------------------------------------

    def _extract_top_level_functions(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[FunctionEntity]:
        """Extract top-level function definitions.

        Args:
            node: The node to extract from (usually root).
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of FunctionEntity objects for top-level functions.
        """
        functions: list[FunctionEntity] = []

        for child in node.children:
            if child.type in ("function_definition", "async_function_definition"):
                func = self._extract_function(child, source_code, file_path, decorators=[])
                if func:
                    functions.append(func)
            elif child.type == "decorated_definition":
                func = self._extract_decorated_function(child, source_code, file_path)
                if func:
                    functions.append(func)

        return functions

    def _extract_decorated_function(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> FunctionEntity | None:
        """Extract a decorated function definition.

        Args:
            node: The decorated_definition node.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            FunctionEntity if found, None otherwise.
        """
        decorators: list[str] = []

        for child in node.children:
            if child.type == "decorator":
                decorator_text = self._extract_decorator_name(child, source_code)
                if decorator_text:
                    decorators.append(decorator_text)
            elif child.type in ("function_definition", "async_function_definition"):
                return self._extract_function(child, source_code, file_path, decorators, node)

        return None

    def _extract_function(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
        decorators: list[str],
        outer_node: Optional["Node"] = None,
    ) -> FunctionEntity | None:
        """Extract a single function definition.

        Args:
            node: The function_definition or async_function_definition node.
            source_code: The original source code.
            file_path: Path to the source file.
            decorators: List of decorator names.
            outer_node: The decorated_definition node if applicable.

        Returns:
            FunctionEntity if extraction succeeds, None otherwise.
        """
        is_async = node.type == "async_function_definition"

        # Find the function name
        name = None
        parameters: list[Parameter] = []
        return_type: str | None = None
        body_node: Node | None = None

        for child in node.children:
            if child.type == "identifier" and name is None or child.type == "name":
                name = self.get_node_text(child, source_code)
            elif child.type == "parameters":
                parameters = self._extract_parameters(child, source_code)
            elif child.type == "type":
                return_type = self.get_node_text(child, source_code)
            elif child.type == "block":
                body_node = child

        if not name:
            return None

        # Get line range from the outer node if decorated, otherwise from the function node
        source_node = outer_node if outer_node else node
        start_line, end_line = self.get_node_line_range(source_node)
        func_source = self.get_node_text(source_node, source_code)

        # Extract docstring from function body
        docstring = None
        if body_node:
            docstring = self._extract_body_docstring(body_node, source_code)

        # Extract function calls from body
        calls: list[str] = []
        if body_node:
            calls = self._extract_calls(body_node, source_code)

        entity_id = self.generate_entity_id(file_path, name, start_line)

        return FunctionEntity(
            id=entity_id,
            name=name,
            type=EntityType.FUNCTION,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=func_source,
            docstring=docstring,
            language=self.language,
            parent_id=None,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            calls=calls,
        )

    def _extract_parameters(
        self,
        node: "Node",
        source_code: str,
    ) -> list[Parameter]:
        """Extract function parameters from a parameters node.

        Args:
            node: The parameters node.
            source_code: The original source code.

        Returns:
            List of Parameter objects.
        """
        parameters: list[Parameter] = []

        for child in node.children:
            if child.type == "identifier":
                # Simple parameter without annotation
                param_name = self.get_node_text(child, source_code)
                parameters.append(
                    Parameter(
                        name=param_name,
                        type_annotation=None,
                        default_value=None,
                        is_variadic=False,
                        is_keyword=False,
                    )
                )
            elif child.type == "typed_parameter":
                param = self._extract_typed_parameter(child, source_code)
                if param:
                    parameters.append(param)
            elif child.type == "default_parameter":
                param = self._extract_default_parameter(child, source_code)
                if param:
                    parameters.append(param)
            elif child.type == "typed_default_parameter":
                param = self._extract_typed_default_parameter(child, source_code)
                if param:
                    parameters.append(param)
            elif child.type == "list_splat_pattern":
                # *args
                for splat_child in child.children:
                    if splat_child.type == "identifier":
                        param_name = self.get_node_text(splat_child, source_code)
                        parameters.append(
                            Parameter(
                                name=param_name,
                                type_annotation=None,
                                default_value=None,
                                is_variadic=True,
                                is_keyword=False,
                            )
                        )
            elif child.type == "dictionary_splat_pattern":
                # **kwargs
                for splat_child in child.children:
                    if splat_child.type == "identifier":
                        param_name = self.get_node_text(splat_child, source_code)
                        parameters.append(
                            Parameter(
                                name=param_name,
                                type_annotation=None,
                                default_value=None,
                                is_variadic=False,
                                is_keyword=True,
                            )
                        )

        return parameters

    def _extract_typed_parameter(
        self,
        node: "Node",
        source_code: str,
    ) -> Parameter | None:
        """Extract a typed parameter (name: type).

        Args:
            node: The typed_parameter node.
            source_code: The original source code.

        Returns:
            Parameter object if extraction succeeds.
        """
        param_name = None
        type_annotation = None

        for child in node.children:
            if child.type == "identifier":
                param_name = self.get_node_text(child, source_code)
            elif child.type == "type":
                type_annotation = self.get_node_text(child, source_code)

        if param_name:
            return Parameter(
                name=param_name,
                type_annotation=type_annotation,
                default_value=None,
                is_variadic=False,
                is_keyword=False,
            )
        return None

    def _extract_default_parameter(
        self,
        node: "Node",
        source_code: str,
    ) -> Parameter | None:
        """Extract a parameter with default value (name=value).

        Args:
            node: The default_parameter node.
            source_code: The original source code.

        Returns:
            Parameter object if extraction succeeds.
        """
        param_name = None
        default_value = None

        children = list(node.children)
        for _i, child in enumerate(children):
            if child.type == "identifier" and param_name is None:
                param_name = self.get_node_text(child, source_code)
            elif param_name is not None and child.type != "=":
                default_value = self.get_node_text(child, source_code)
                break

        if param_name:
            return Parameter(
                name=param_name,
                type_annotation=None,
                default_value=default_value,
                is_variadic=False,
                is_keyword=False,
            )
        return None

    def _extract_typed_default_parameter(
        self,
        node: "Node",
        source_code: str,
    ) -> Parameter | None:
        """Extract a typed parameter with default (name: type = value).

        Args:
            node: The typed_default_parameter node.
            source_code: The original source code.

        Returns:
            Parameter object if extraction succeeds.
        """
        param_name = None
        type_annotation = None
        default_value = None

        children = list(node.children)
        found_equals = False
        for child in children:
            if child.type == "identifier" and param_name is None:
                param_name = self.get_node_text(child, source_code)
            elif child.type == "type":
                type_annotation = self.get_node_text(child, source_code)
            elif child.type == "=":
                found_equals = True
            elif found_equals and default_value is None:
                default_value = self.get_node_text(child, source_code)

        if param_name:
            return Parameter(
                name=param_name,
                type_annotation=type_annotation,
                default_value=default_value,
                is_variadic=False,
                is_keyword=False,
            )
        return None

    def _extract_decorator_name(
        self,
        node: "Node",
        source_code: str,
    ) -> str | None:
        """Extract the decorator name from a decorator node.

        Args:
            node: The decorator node.
            source_code: The original source code.

        Returns:
            The decorator name/expression as a string.
        """
        for child in node.children:
            if child.type in ("identifier", "dotted_name", "attribute"):
                return self.get_node_text(child, source_code)
            elif child.type == "call":
                # Decorator with arguments like @decorator(args)
                for call_child in child.children:
                    if call_child.type in ("identifier", "attribute"):
                        return self.get_node_text(call_child, source_code)
        return None

    def _extract_body_docstring(
        self,
        body_node: "Node",
        source_code: str,
    ) -> str | None:
        """Extract docstring from a function/method/class body.

        Args:
            body_node: The block node containing the body.
            source_code: The original source code.

        Returns:
            The docstring text if found, None otherwise.
        """
        for child in body_node.children:
            if child.type in ("comment", "newline"):
                continue
            if child.type == "expression_statement":
                for expr_child in child.children:
                    if expr_child.type == "string":
                        return self._parse_string_literal(expr_child, source_code)
                break
            else:
                break
        return None

    def _extract_calls(
        self,
        node: "Node",
        source_code: str,
    ) -> list[str]:
        """Extract function/method calls from a node.

        Args:
            node: The node to search for calls.
            source_code: The original source code.

        Returns:
            List of called function/method names.
        """
        calls: list[str] = []
        self._collect_calls_recursive(node, source_code, calls)
        return list(dict.fromkeys(calls))  # Remove duplicates while preserving order

    def _collect_calls_recursive(
        self,
        node: "Node",
        source_code: str,
        calls: list[str],
    ) -> None:
        """Recursively collect function/method calls.

        Args:
            node: The current node.
            source_code: The original source code.
            calls: List to append found calls to.
        """
        if node.type == "call":
            call_name = self._extract_call_name(node, source_code)
            if call_name:
                calls.append(call_name)

        for child in node.children:
            self._collect_calls_recursive(child, source_code, calls)

    def _extract_call_name(
        self,
        node: "Node",
        source_code: str,
    ) -> str | None:
        """Extract the name of a function/method being called.

        Args:
            node: The call node.
            source_code: The original source code.

        Returns:
            The call name (e.g., 'func', 'obj.method', 'module.func').
        """
        for child in node.children:
            if child.type == "identifier" or child.type == "attribute":
                return self.get_node_text(child, source_code)
        return None

    # -------------------------------------------------------------------------
    # Class Extraction
    # -------------------------------------------------------------------------

    def _extract_all_classes(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[ClassEntity]:
        """Extract all class definitions.

        Args:
            node: The node to extract from (usually root).
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of ClassEntity objects.
        """
        classes: list[ClassEntity] = []

        for child in node.children:
            if child.type == "class_definition":
                cls = self._extract_class(child, source_code, file_path, decorators=[])
                if cls:
                    classes.append(cls)
            elif child.type == "decorated_definition":
                cls = self._extract_decorated_class(child, source_code, file_path)
                if cls:
                    classes.append(cls)

        return classes

    def _extract_decorated_class(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> ClassEntity | None:
        """Extract a decorated class definition.

        Args:
            node: The decorated_definition node.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            ClassEntity if found, None otherwise.
        """
        decorators: list[str] = []

        for child in node.children:
            if child.type == "decorator":
                decorator_text = self._extract_decorator_name(child, source_code)
                if decorator_text:
                    decorators.append(decorator_text)
            elif child.type == "class_definition":
                return self._extract_class(child, source_code, file_path, decorators, node)

        return None

    def _extract_class(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
        decorators: list[str],
        outer_node: Optional["Node"] = None,
    ) -> ClassEntity | None:
        """Extract a single class definition.

        Args:
            node: The class_definition node.
            source_code: The original source code.
            file_path: Path to the source file.
            decorators: List of decorator names.
            outer_node: The decorated_definition node if applicable.

        Returns:
            ClassEntity if extraction succeeds, None otherwise.
        """
        name = None
        bases: list[str] = []
        body_node: Node | None = None

        for child in node.children:
            if child.type == "identifier" and name is None:
                name = self.get_node_text(child, source_code)
            elif child.type == "argument_list":
                # Extract base classes
                bases = self._extract_class_bases(child, source_code)
            elif child.type == "block":
                body_node = child

        if not name:
            return None

        # Get line range
        source_node = outer_node if outer_node else node
        start_line, end_line = self.get_node_line_range(source_node)
        class_source = self.get_node_text(source_node, source_code)

        # Extract docstring
        docstring = None
        if body_node:
            docstring = self._extract_body_docstring(body_node, source_code)

        entity_id = self.generate_entity_id(file_path, name, start_line)

        # Extract methods and attributes from class body
        methods: list[MethodEntity] = []
        attributes: list[Attribute] = []

        if body_node:
            methods = self._extract_class_methods(body_node, source_code, file_path, entity_id)
            attributes = self._extract_class_attributes(body_node, source_code)

        return ClassEntity(
            id=entity_id,
            name=name,
            type=EntityType.CLASS,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=class_source,
            docstring=docstring,
            language=self.language,
            parent_id=None,
            bases=bases,
            methods=methods,
            attributes=attributes,
            decorators=decorators,
        )

    def _extract_class_bases(
        self,
        node: "Node",
        source_code: str,
    ) -> list[str]:
        """Extract base class names from argument_list.

        Args:
            node: The argument_list node.
            source_code: The original source code.

        Returns:
            List of base class names.
        """
        bases: list[str] = []

        for child in node.children:
            if child.type in ("identifier", "attribute"):
                bases.append(self.get_node_text(child, source_code))
            elif child.type == "keyword_argument":
                # Handle metaclass= and other keyword arguments in bases
                pass  # Skip keyword arguments for base class list

        return bases

    def _extract_class_methods(
        self,
        body_node: "Node",
        source_code: str,
        file_path: str,
        parent_id: str,
    ) -> list[MethodEntity]:
        """Extract method definitions from a class body.

        Args:
            body_node: The block node containing the class body.
            source_code: The original source code.
            file_path: Path to the source file.
            parent_id: ID of the parent class.

        Returns:
            List of MethodEntity objects.
        """
        methods: list[MethodEntity] = []

        for child in body_node.children:
            if child.type in ("function_definition", "async_function_definition"):
                method = self._extract_method(child, source_code, file_path, parent_id, [])
                if method:
                    methods.append(method)
            elif child.type == "decorated_definition":
                method = self._extract_decorated_method(child, source_code, file_path, parent_id)
                if method:
                    methods.append(method)

        return methods

    def _extract_decorated_method(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
        parent_id: str,
    ) -> MethodEntity | None:
        """Extract a decorated method definition.

        Args:
            node: The decorated_definition node.
            source_code: The original source code.
            file_path: Path to the source file.
            parent_id: ID of the parent class.

        Returns:
            MethodEntity if found, None otherwise.
        """
        decorators: list[str] = []

        for child in node.children:
            if child.type == "decorator":
                decorator_text = self._extract_decorator_name(child, source_code)
                if decorator_text:
                    decorators.append(decorator_text)
            elif child.type in ("function_definition", "async_function_definition"):
                return self._extract_method(
                    child, source_code, file_path, parent_id, decorators, node
                )

        return None

    def _extract_method(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
        parent_id: str,
        decorators: list[str],
        outer_node: Optional["Node"] = None,
    ) -> MethodEntity | None:
        """Extract a single method definition.

        Args:
            node: The function_definition or async_function_definition node.
            source_code: The original source code.
            file_path: Path to the source file.
            parent_id: ID of the parent class.
            decorators: List of decorator names.
            outer_node: The decorated_definition node if applicable.

        Returns:
            MethodEntity if extraction succeeds, None otherwise.
        """
        is_async = node.type == "async_function_definition"

        name = None
        parameters: list[Parameter] = []
        return_type: str | None = None
        body_node: Node | None = None

        for child in node.children:
            if child.type == "identifier" and name is None or child.type == "name":
                name = self.get_node_text(child, source_code)
            elif child.type == "parameters":
                parameters = self._extract_parameters(child, source_code)
            elif child.type == "type":
                return_type = self.get_node_text(child, source_code)
            elif child.type == "block":
                body_node = child

        if not name:
            return None

        # Determine method type from decorators
        decorator_lower = [d.lower() for d in decorators]
        is_classmethod = "classmethod" in decorator_lower
        is_staticmethod = "staticmethod" in decorator_lower
        is_property = "property" in decorator_lower or any(
            d.endswith(".setter") or d.endswith(".getter") or d.endswith(".deleter")
            for d in decorator_lower
        )

        # Get line range
        source_node = outer_node if outer_node else node
        start_line, end_line = self.get_node_line_range(source_node)
        method_source = self.get_node_text(source_node, source_code)

        # Extract docstring
        docstring = None
        if body_node:
            docstring = self._extract_body_docstring(body_node, source_code)

        # Extract method calls
        calls: list[str] = []
        if body_node:
            calls = self._extract_calls(body_node, source_code)

        entity_id = self.generate_entity_id(file_path, name, start_line)

        return MethodEntity(
            id=entity_id,
            name=name,
            type=EntityType.METHOD,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=method_source,
            docstring=docstring,
            language=self.language,
            parent_id=parent_id,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            is_async=is_async,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_property=is_property,
            calls=calls,
        )

    def _extract_class_attributes(
        self,
        body_node: "Node",
        source_code: str,
    ) -> list[Attribute]:
        """Extract class-level attributes from a class body.

        Args:
            body_node: The block node containing the class body.
            source_code: The original source code.

        Returns:
            List of Attribute objects.
        """
        attributes: list[Attribute] = []

        for child in body_node.children:
            if child.type == "expression_statement":
                # Check for assignment inside expression_statement
                for expr_child in child.children:
                    if expr_child.type == "assignment":
                        attr = self._extract_attribute_from_assignment(expr_child, source_code)
                        if attr:
                            attributes.append(attr)
                    elif expr_child.type == "augmented_assignment":
                        # Handle += etc.
                        pass
            elif child.type == "annotated_assignment":
                attr = self._extract_attribute_from_annotated(child, source_code)
                if attr:
                    attributes.append(attr)

        return attributes

    def _extract_attribute_from_assignment(
        self,
        node: "Node",
        source_code: str,
    ) -> Attribute | None:
        """Extract an attribute from an assignment node.

        Handles both simple assignments (x = 1) and annotated assignments (x: int = 1).

        Args:
            node: The assignment node.
            source_code: The original source code.

        Returns:
            Attribute if extraction succeeds, None otherwise.
        """
        name = None
        type_annotation = None
        value = None
        line = node.start_point[0] + 1

        children = list(node.children)
        found_equals = False
        for child in children:
            if child.type == "identifier" and name is None:
                name = self.get_node_text(child, source_code)
            elif child.type == "type":
                type_annotation = self.get_node_text(child, source_code)
            elif child.type == "=":
                found_equals = True
            elif found_equals and value is None:
                value = self.get_node_text(child, source_code)

        if name:
            return Attribute(
                name=name,
                type_annotation=type_annotation,
                default_value=value,
                line=line,
            )
        return None

    def _extract_attribute_from_annotated(
        self,
        node: "Node",
        source_code: str,
    ) -> Attribute | None:
        """Extract an attribute from an annotated_assignment node.

        Args:
            node: The annotated_assignment node.
            source_code: The original source code.

        Returns:
            Attribute if extraction succeeds, None otherwise.
        """
        name = None
        type_annotation = None
        value = None
        line = node.start_point[0] + 1

        children = list(node.children)
        found_equals = False
        for child in children:
            if child.type == "identifier" and name is None:
                name = self.get_node_text(child, source_code)
            elif child.type == "type":
                type_annotation = self.get_node_text(child, source_code)
            elif child.type == "=":
                found_equals = True
            elif found_equals and value is None:
                value = self.get_node_text(child, source_code)

        if name:
            return Attribute(
                name=name,
                type_annotation=type_annotation,
                default_value=value,
                line=line,
            )
        return None

    # -------------------------------------------------------------------------
    # Variable Extraction
    # -------------------------------------------------------------------------

    def _extract_module_variables(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> list[VariableEntity]:
        """Extract module-level variable assignments.

        Args:
            node: The node to extract from (usually root).
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            List of VariableEntity objects.
        """
        variables: list[VariableEntity] = []

        for child in node.children:
            if child.type == "expression_statement":
                for expr_child in child.children:
                    if expr_child.type == "assignment":
                        var = self._extract_variable_from_assignment(
                            expr_child, source_code, file_path
                        )
                        if var:
                            variables.append(var)
            elif child.type == "annotated_assignment":
                var = self._extract_variable_from_annotated(child, source_code, file_path)
                if var:
                    variables.append(var)

        return variables

    def _extract_variable_from_assignment(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> VariableEntity | None:
        """Extract a variable from an assignment node.

        Handles both simple assignments (x = 1) and annotated assignments (x: int = 1).

        Args:
            node: The assignment node.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            VariableEntity if extraction succeeds, None otherwise.
        """
        name = None
        type_annotation = None
        value = None

        children = list(node.children)
        found_equals = False
        for child in children:
            if child.type == "identifier" and name is None:
                name = self.get_node_text(child, source_code)
            elif child.type == "pattern_list":
                # Tuple unpacking - skip for now
                return None
            elif child.type == "type":
                type_annotation = self.get_node_text(child, source_code)
            elif child.type == "=":
                found_equals = True
            elif found_equals and value is None:
                value = self.get_node_text(child, source_code)

        if not name:
            return None

        start_line, end_line = self.get_node_line_range(node)
        var_source = self.get_node_text(node, source_code)
        is_constant = name.isupper() or (name.startswith("_") and name[1:].isupper())
        entity_id = self.generate_entity_id(file_path, name, start_line)

        return VariableEntity(
            id=entity_id,
            name=name,
            type=EntityType.VARIABLE,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=var_source,
            docstring=None,
            language=self.language,
            parent_id=None,
            type_annotation=type_annotation,
            value=value,
            is_constant=is_constant,
        )

    def _extract_variable_from_annotated(
        self,
        node: "Node",
        source_code: str,
        file_path: str,
    ) -> VariableEntity | None:
        """Extract a variable from an annotated_assignment node.

        Args:
            node: The annotated_assignment node.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            VariableEntity if extraction succeeds, None otherwise.
        """
        name = None
        type_annotation = None
        value = None

        children = list(node.children)
        found_equals = False
        for child in children:
            if child.type == "identifier" and name is None:
                name = self.get_node_text(child, source_code)
            elif child.type == "type":
                type_annotation = self.get_node_text(child, source_code)
            elif child.type == "=":
                found_equals = True
            elif found_equals and value is None:
                value = self.get_node_text(child, source_code)

        if not name:
            return None

        start_line, end_line = self.get_node_line_range(node)
        var_source = self.get_node_text(node, source_code)
        is_constant = name.isupper() or (name.startswith("_") and name[1:].isupper())
        entity_id = self.generate_entity_id(file_path, name, start_line)

        return VariableEntity(
            id=entity_id,
            name=name,
            type=EntityType.VARIABLE,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            source_code=var_source,
            docstring=None,
            language=self.language,
            parent_id=None,
            type_annotation=type_annotation,
            value=value,
            is_constant=is_constant,
        )
