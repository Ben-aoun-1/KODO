"""Abstract base class for language-specific entity extractors.

This module defines the interface that all language extractors must implement.
Each language extractor is responsible for traversing a tree-sitter AST and
extracting code entities specific to that language.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

from ..models import (
    ClassEntity,
    FunctionEntity,
    ImportEntity,
    ModuleEntity,
    VariableEntity,
)


class BaseExtractor(ABC):
    """Abstract base class for language-specific entity extractors.

    Each language extractor knows how to traverse a tree-sitter AST for its
    specific language and extract all code entities (functions, classes,
    methods, imports, variables).

    Attributes:
        language: The language identifier this extractor handles.
    """

    language: str = ""

    @abstractmethod
    def extract_module(
        self,
        tree: "Tree",
        source_code: str,
        file_path: str,
    ) -> ModuleEntity:
        """Extract a complete module entity from the AST.

        This is the main entry point for extraction. It should traverse the
        entire AST and extract all top-level entities.

        Args:
            tree: The tree-sitter parse tree.
            source_code: The original source code.
            file_path: Path to the source file.

        Returns:
            A ModuleEntity containing all extracted entities.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    def get_node_text(self, node: "Node", source_code: str) -> str:
        """Extract the text content of a node from source code.

        Args:
            node: The tree-sitter node.
            source_code: The original source code.

        Returns:
            The text content of the node.
        """
        return source_code[node.start_byte : node.end_byte]

    def get_node_line_range(self, node: "Node") -> tuple[int, int]:
        """Get the line range of a node (1-indexed).

        Args:
            node: The tree-sitter node.

        Returns:
            Tuple of (start_line, end_line), 1-indexed.
        """
        # tree-sitter uses 0-indexed lines, we want 1-indexed
        return (node.start_point[0] + 1, node.end_point[0] + 1)

    def generate_entity_id(
        self,
        file_path: str,
        name: str,
        start_line: int,
    ) -> str:
        """Generate a unique entity ID.

        Args:
            file_path: Path to the source file.
            name: Name of the entity.
            start_line: Starting line number.

        Returns:
            Entity ID in format {file_path}:{name}:{start_line}.
        """
        return f"{file_path}:{name}:{start_line}"
