"""Pydantic models for code entities.

This module defines the data models used to represent parsed code entities
such as functions, classes, methods, variables, and imports. All models
follow the CodeEntity base specification from CLAUDE.md.
"""

from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Enumeration of supported code entity types."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"


class CodeEntity(BaseModel):
    """Base model for all parsed code entities.

    This is the foundational model that all specific entity types extend.
    It contains the common attributes shared across all code constructs.

    Attributes:
        id: Unique identifier in format {file_path}:{name}:{start_line}.
        name: The name of the entity (function name, class name, etc.).
        type: The type of entity (function, class, method, etc.).
        file_path: Absolute path to the file containing this entity.
        start_line: Line number where the entity begins (1-indexed).
        end_line: Line number where the entity ends (1-indexed).
        source_code: The complete source code of the entity.
        docstring: The docstring if present, None otherwise.
        language: Programming language (python, javascript, typescript, etc.).
        parent_id: ID of the parent entity if nested, None for top-level.
    """

    id: str = Field(..., description="Unique identifier: {file_path}:{name}:{start_line}")
    name: str = Field(..., description="Name of the entity")
    type: EntityType = Field(..., description="Type of the code entity")
    file_path: str = Field(..., description="Absolute path to the source file")
    start_line: int = Field(..., ge=1, description="Starting line number (1-indexed)")
    end_line: int = Field(..., ge=1, description="Ending line number (1-indexed)")
    source_code: str = Field(..., description="Complete source code of the entity")
    docstring: str | None = Field(None, description="Docstring if present")
    language: str = Field(..., description="Programming language identifier")
    parent_id: str | None = Field(None, description="ID of parent entity if nested")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class Parameter(BaseModel):
    """Represents a function or method parameter.

    Attributes:
        name: Parameter name.
        type_annotation: Type annotation if present.
        default_value: Default value as string if present.
        is_variadic: True if *args style parameter.
        is_keyword: True if **kwargs style parameter.
    """

    name: str = Field(..., description="Parameter name")
    type_annotation: str | None = Field(None, description="Type annotation")
    default_value: str | None = Field(None, description="Default value as string")
    is_variadic: bool = Field(False, description="True for *args style")
    is_keyword: bool = Field(False, description="True for **kwargs style")


class FunctionEntity(CodeEntity):
    """Model for function definitions.

    Extends CodeEntity with function-specific attributes like parameters,
    return type, decorators, and call information.

    Attributes:
        parameters: List of function parameters.
        return_type: Return type annotation if present.
        decorators: List of decorator names applied to the function.
        is_async: Whether the function is async.
        calls: List of function/method names called within this function.
    """

    type: EntityType = Field(default=EntityType.FUNCTION, description="Entity type")
    parameters: list[Parameter] = Field(default_factory=list, description="Function parameters")
    return_type: str | None = Field(None, description="Return type annotation")
    decorators: list[str] = Field(default_factory=list, description="Applied decorators")
    is_async: bool = Field(False, description="Whether function is async")
    calls: list[str] = Field(default_factory=list, description="Functions/methods called within")


class MethodEntity(CodeEntity):
    """Model for method definitions within classes.

    Similar to FunctionEntity but represents methods bound to a class.
    The parent_id should reference the containing class.

    Attributes:
        parameters: List of method parameters (including self/cls).
        return_type: Return type annotation if present.
        decorators: List of decorator names applied to the method.
        is_async: Whether the method is async.
        is_classmethod: Whether decorated with @classmethod.
        is_staticmethod: Whether decorated with @staticmethod.
        is_property: Whether decorated with @property.
        calls: List of function/method names called within this method.
    """

    type: EntityType = Field(default=EntityType.METHOD, description="Entity type")
    parameters: list[Parameter] = Field(default_factory=list, description="Method parameters")
    return_type: str | None = Field(None, description="Return type annotation")
    decorators: list[str] = Field(default_factory=list, description="Applied decorators")
    is_async: bool = Field(False, description="Whether method is async")
    is_classmethod: bool = Field(False, description="True if @classmethod")
    is_staticmethod: bool = Field(False, description="True if @staticmethod")
    is_property: bool = Field(False, description="True if @property")
    calls: list[str] = Field(default_factory=list, description="Functions/methods called within")


class Attribute(BaseModel):
    """Represents a class attribute.

    Attributes:
        name: Attribute name.
        type_annotation: Type annotation if present.
        default_value: Default value as string if present.
        line: Line number where attribute is defined.
    """

    name: str = Field(..., description="Attribute name")
    type_annotation: str | None = Field(None, description="Type annotation")
    default_value: str | None = Field(None, description="Default value as string")
    line: int = Field(..., ge=1, description="Line number of definition")


class ClassEntity(CodeEntity):
    """Model for class definitions.

    Extends CodeEntity with class-specific attributes like base classes,
    methods, and class attributes.

    Attributes:
        bases: List of base class names this class inherits from.
        methods: List of method entities defined in this class.
        attributes: List of class attributes.
        decorators: List of decorator names applied to the class.
    """

    type: EntityType = Field(default=EntityType.CLASS, description="Entity type")
    bases: list[str] = Field(default_factory=list, description="Base class names")
    methods: list[MethodEntity] = Field(default_factory=list, description="Methods in this class")
    attributes: list[Attribute] = Field(default_factory=list, description="Class attributes")
    decorators: list[str] = Field(default_factory=list, description="Applied decorators")


class ImportEntity(CodeEntity):
    """Model for import statements.

    Represents both simple imports (import x) and from imports (from x import y).

    Attributes:
        module: The module being imported from.
        alias: Alias if using 'as' syntax.
        is_from_import: True for 'from x import y' style imports.
        imported_names: List of names imported (for from imports).
    """

    type: EntityType = Field(default=EntityType.IMPORT, description="Entity type")
    module: str = Field(..., description="Module being imported")
    alias: str | None = Field(None, description="Import alias if using 'as'")
    is_from_import: bool = Field(False, description="True for 'from x import y'")
    imported_names: list[str] = Field(
        default_factory=list, description="Names imported (for from imports)"
    )


class VariableEntity(CodeEntity):
    """Model for variable definitions.

    Represents module-level and class-level variable assignments.

    Attributes:
        type_annotation: Type annotation if present.
        value: The assigned value as a string representation.
        is_constant: Whether this appears to be a constant (ALL_CAPS naming).
    """

    type: EntityType = Field(default=EntityType.VARIABLE, description="Entity type")
    type_annotation: str | None = Field(None, description="Type annotation")
    value: str | None = Field(None, description="Assigned value as string")
    is_constant: bool = Field(False, description="True if appears to be constant")


class ModuleEntity(CodeEntity):
    """Model for module definitions.

    Represents an entire file/module with its contents.

    Attributes:
        imports: List of import entities in this module.
        functions: List of top-level functions.
        classes: List of class definitions.
        variables: List of module-level variables.
    """

    type: EntityType = Field(default=EntityType.MODULE, description="Entity type")
    imports: list[ImportEntity] = Field(default_factory=list, description="Module imports")
    functions: list[FunctionEntity] = Field(default_factory=list, description="Top-level functions")
    classes: list[ClassEntity] = Field(default_factory=list, description="Class definitions")
    variables: list[VariableEntity] = Field(
        default_factory=list, description="Module-level variables"
    )


class ParseResult(BaseModel):
    """Result of parsing a source file.

    Contains the parsed module entity along with metadata about the parsing.

    Attributes:
        module: The parsed module entity.
        file_path: Path to the parsed file.
        language: Detected or specified language.
        parse_errors: List of any parsing errors encountered.
        success: Whether parsing completed successfully.
    """

    module: ModuleEntity = Field(..., description="Parsed module entity")
    file_path: str = Field(..., description="Path to the parsed file")
    language: str = Field(..., description="Programming language")
    parse_errors: list[str] = Field(default_factory=list, description="Parsing errors encountered")
    success: bool = Field(True, description="Whether parsing succeeded")
