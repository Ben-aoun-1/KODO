"""Tree-sitter based parser implementation.

This module provides the main parser implementation using tree-sitter for
AST parsing. It supports multiple languages through language-specific
extractors in the languages/ subdirectory.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

logger = structlog.get_logger(__name__)

from .base import BaseParser, ParserError
from .languages import get_extractor
from .languages.base import BaseExtractor
from .models import (
    EntityType,
    ModuleEntity,
    ParseResult,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser, Tree


class TreeSitterParser(BaseParser):
    """Tree-sitter based source code parser.

    Uses tree-sitter grammars to parse source code into ASTs, then extracts
    code entities using language-specific extractors.

    Attributes:
        supported_languages: Languages this parser can handle.
    """

    supported_languages: set[str] = {"python", "javascript", "typescript"}

    def __init__(self) -> None:
        """Initialize the TreeSitterParser.

        Sets up tree-sitter and loads available language grammars.
        """
        self._initialized = False
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        self._extractors: dict[str, BaseExtractor] = {}
        logger.debug("TreeSitterParser created (lazy initialization)")

    async def _ensure_initialized(self) -> None:
        """Lazily initialize tree-sitter parsers.

        This is called before any parsing operation to ensure the tree-sitter
        library and language grammars are loaded.
        """
        if self._initialized:
            return

        try:
            self._init_python_parser()
        except ImportError as e:
            logger.warning(f"Failed to initialize Python parser: {e}")

        try:
            self._init_javascript_parser()
        except ImportError as e:
            logger.warning(f"Failed to initialize JavaScript parser: {e}")

        try:
            self._init_typescript_parser()
        except ImportError as e:
            logger.warning(f"Failed to initialize TypeScript parser: {e}")

        # Update supported languages based on what was actually loaded
        self.supported_languages = set(self._parsers.keys())

        self._initialized = True
        logger.info(
            f"TreeSitterParser initialized with languages: {', '.join(self.supported_languages)}"
        )

    def _init_python_parser(self) -> None:
        """Initialize the Python tree-sitter parser."""
        import tree_sitter_python
        from tree_sitter import Language, Parser

        language = Language(tree_sitter_python.language())
        parser = Parser(language)

        self._languages["python"] = language
        self._parsers["python"] = parser
        logger.debug("Python tree-sitter parser initialized")

    def _init_javascript_parser(self) -> None:
        """Initialize the JavaScript tree-sitter parser."""
        import tree_sitter_javascript
        from tree_sitter import Language, Parser

        language = Language(tree_sitter_javascript.language())
        parser = Parser(language)

        self._languages["javascript"] = language
        self._parsers["javascript"] = parser
        logger.debug("JavaScript tree-sitter parser initialized")

    def _init_typescript_parser(self) -> None:
        """Initialize the TypeScript tree-sitter parser."""
        import tree_sitter_typescript
        from tree_sitter import Language, Parser

        # TypeScript module provides both TypeScript and TSX
        ts_language = Language(tree_sitter_typescript.language_typescript())
        ts_parser = Parser(ts_language)

        self._languages["typescript"] = ts_language
        self._parsers["typescript"] = ts_parser

        # Also register TSX
        tsx_language = Language(tree_sitter_typescript.language_tsx())
        tsx_parser = Parser(tsx_language)
        self._languages["tsx"] = tsx_language
        self._parsers["tsx"] = tsx_parser

        logger.debug("TypeScript tree-sitter parser initialized")

    def _get_parser(self, language: str) -> "Parser":
        """Get the tree-sitter parser for a language.

        Args:
            language: The programming language.

        Returns:
            The tree-sitter Parser instance.

        Raises:
            ParserError: If no parser is available for the language.
        """
        if language not in self._parsers:
            raise ParserError(f"No parser available for language: {language}")
        return self._parsers[language]

    def _get_extractor(self, language: str) -> BaseExtractor:
        """Get the entity extractor for a language.

        Uses cached extractors for efficiency.

        Args:
            language: The programming language.

        Returns:
            The language extractor instance.

        Raises:
            ParserError: If no extractor is available for the language.
        """
        if language not in self._extractors:
            try:
                extractor = get_extractor(language)
                self._extractors[language] = extractor
            except ValueError as e:
                raise ParserError(str(e))
        return self._extractors[language]

    def _parse_to_tree(self, source_code: str, language: str) -> "Tree":
        """Parse source code into a tree-sitter AST.

        Args:
            source_code: The source code to parse.
            language: The programming language.

        Returns:
            The tree-sitter Tree (AST).

        Raises:
            ParserError: If parsing fails.
        """
        parser = self._get_parser(language)

        # tree-sitter requires bytes
        source_bytes = source_code.encode("utf-8")

        tree = parser.parse(source_bytes)

        if tree is None:
            raise ParserError(f"tree-sitter failed to parse {language} source")

        return tree

    async def parse_file(
        self,
        file_path: Path | str,
        *,
        encoding: str = "utf-8",
    ) -> ParseResult:
        """Parse a source code file using tree-sitter.

        Args:
            file_path: Path to the source file to parse.
            encoding: Character encoding of the file. Defaults to utf-8.

        Returns:
            ParseResult containing the parsed module and any errors.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file cannot be decoded.
            ParserError: If a fatal parsing error occurs.
        """
        await self._ensure_initialized()

        path = Path(file_path) if isinstance(file_path, str) else file_path
        abs_path = str(path.resolve())

        if not path.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")

        if not path.is_file():
            raise ParserError(f"Path is not a file: {abs_path}", file_path=abs_path)

        # Detect language from file extension
        language = self.detect_language(path)
        if not language:
            raise ParserError(
                f"Cannot detect language for file: {path.name}",
                file_path=abs_path,
            )

        if not self.supports_language(language):
            raise ParserError(
                f"Unsupported language: {language}",
                file_path=abs_path,
            )

        # Read file contents
        try:
            source_code = path.read_text(encoding=encoding)
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to decode {abs_path} with {encoding}: {e}")
            raise

        logger.debug(f"Parsing file: {abs_path} (language={language})")

        return await self.parse_source(
            source_code,
            file_path=abs_path,
            language=language,
        )

    async def parse_source(
        self,
        source_code: str,
        *,
        file_path: str | None = None,
        language: str | None = None,
    ) -> ParseResult:
        """Parse source code from a string using tree-sitter.

        Args:
            source_code: The source code to parse.
            file_path: Optional virtual file path for entity IDs.
            language: Language identifier.

        Returns:
            ParseResult containing the parsed module and any errors.

        Raises:
            ValueError: If language cannot be determined.
            ParserError: If a fatal parsing error occurs.
        """
        await self._ensure_initialized()

        # Determine file path for entity IDs
        effective_path = file_path or "<string>"

        # Determine language
        if not language and file_path:
            language = self.detect_language(file_path)

        if not language:
            raise ValueError("Language must be specified or inferrable from file_path")

        if not self.supports_language(language):
            raise ParserError(
                f"Unsupported language: {language}",
                file_path=effective_path,
            )

        parse_errors: list[str] = []

        try:
            # Parse source code into AST
            tree = self._parse_to_tree(source_code, language)

            # Check for syntax errors in the tree
            if tree.root_node.has_error:
                error_nodes = self._find_error_nodes(tree.root_node)
                for error_node in error_nodes:
                    line = error_node.start_point[0] + 1
                    col = error_node.start_point[1] + 1
                    parse_errors.append(f"Syntax error at line {line}, column {col}")
                logger.warning(
                    f"Parse tree contains {len(error_nodes)} error(s) in {effective_path}"
                )

            # Get the appropriate extractor and extract entities
            extractor = self._get_extractor(language)
            module = extractor.extract_module(tree, source_code, effective_path)

            logger.debug(
                f"Parsed {effective_path}: "
                f"{len(module.functions)} functions, "
                f"{len(module.classes)} classes, "
                f"{len(module.imports)} imports, "
                f"{len(module.variables)} variables"
            )

            return ParseResult(
                module=module,
                file_path=effective_path,
                language=language,
                parse_errors=parse_errors,
                success=len(parse_errors) == 0,
            )

        except ParserError:
            # Re-raise parser errors as-is
            raise

        except Exception as e:
            error_msg = f"Failed to parse source: {e}"
            logger.error(error_msg)
            parse_errors.append(error_msg)

            # Return empty module with error
            module = self._create_fallback_module(
                source_code=source_code,
                file_path=effective_path,
                language=language,
            )

            return ParseResult(
                module=module,
                file_path=effective_path,
                language=language,
                parse_errors=parse_errors,
                success=False,
            )

    def _find_error_nodes(self, node: "Node") -> list["Node"]:
        """Find all error nodes in the parse tree.

        Args:
            node: The root node to search from.

        Returns:
            List of nodes representing parsing errors.
        """
        errors: list[Node] = []

        if node.type == "ERROR" or node.is_missing:
            errors.append(node)

        for child in node.children:
            errors.extend(self._find_error_nodes(child))

        return errors

    def _create_fallback_module(
        self,
        source_code: str,
        file_path: str,
        language: str,
    ) -> ModuleEntity:
        """Create a fallback module entity when parsing fails.

        This provides minimal information when the full parse fails,
        allowing graceful degradation.

        Args:
            source_code: The source code.
            file_path: Path to the source file.
            language: Programming language.

        Returns:
            A ModuleEntity with minimal information.
        """
        lines = source_code.splitlines()
        line_count = len(lines) if lines else 1

        # Extract module name from file path
        path = Path(file_path)
        module_name = path.stem

        return ModuleEntity(
            id=f"{file_path}:{module_name}:1",
            name=module_name,
            type=EntityType.MODULE,
            file_path=file_path,
            start_line=1,
            end_line=line_count,
            source_code=source_code,
            docstring=None,
            language=language,
            parent_id=None,
            imports=[],
            functions=[],
            classes=[],
            variables=[],
        )

    async def get_language_extractor(self, language: str) -> BaseExtractor:
        """Get the language-specific entity extractor.

        Args:
            language: The programming language.

        Returns:
            The language extractor instance.

        Raises:
            ParserError: If no extractor is available for the language.
        """
        await self._ensure_initialized()
        return self._get_extractor(language)

    def get_supported_languages(self) -> list[str]:
        """Get list of languages with available parsers.

        Returns:
            List of language identifiers with working parsers.
        """
        return list(self._parsers.keys())

    async def debug_ast(
        self,
        source_code: str,
        language: str,
        max_depth: int = 10,
    ) -> str:
        """Generate a debug representation of the AST.

        Useful for understanding the tree structure when developing
        language extractors.

        Args:
            source_code: The source code to parse.
            language: The programming language.
            max_depth: Maximum depth to traverse.

        Returns:
            String representation of the AST.
        """
        await self._ensure_initialized()

        tree = self._parse_to_tree(source_code, language)
        return self._format_node(tree.root_node, source_code, 0, max_depth)

    def _format_node(
        self,
        node: "Node",
        source_code: str,
        depth: int,
        max_depth: int,
    ) -> str:
        """Format a node and its children for debugging.

        Args:
            node: The node to format.
            source_code: The original source code.
            depth: Current depth.
            max_depth: Maximum depth.

        Returns:
            Formatted string representation.
        """
        if depth >= max_depth:
            return "  " * depth + "...\n"

        indent = "  " * depth
        node_text = source_code[node.start_byte : node.end_byte]

        # Truncate long text
        if len(node_text) > 50:
            node_text = node_text[:50] + "..."

        # Escape newlines
        node_text = node_text.replace("\n", "\\n")

        result = (
            f"{indent}{node.type} [{node.start_point[0]+1}:{node.start_point[1]}]"
            f" = {repr(node_text)}\n"
        )

        for child in node.children:
            result += self._format_node(child, source_code, depth + 1, max_depth)

        return result
