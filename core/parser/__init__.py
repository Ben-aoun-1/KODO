"""Parser module for source code parsing and entity extraction.

This module provides tree-sitter based parsing for multiple programming
languages, extracting code entities like functions, classes, methods,
imports, and variables into structured Pydantic models.

Example:
    >>> from core.parser import TreeSitterParser, ParseResult
    >>> parser = TreeSitterParser()
    >>> result = await parser.parse_file("example.py")
    >>> print(result.module.functions)
"""

from .base import BaseParser, ParserError
from .models import (
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
from .tree_sitter import TreeSitterParser

__all__ = [
    # Base classes
    "BaseParser",
    "ParserError",
    # Parser implementations
    "TreeSitterParser",
    # Entity types
    "EntityType",
    # Entity models
    "CodeEntity",
    "FunctionEntity",
    "MethodEntity",
    "ClassEntity",
    "ImportEntity",
    "VariableEntity",
    "ModuleEntity",
    # Supporting models
    "Parameter",
    "Attribute",
    "ParseResult",
]
