"""Abstract base parser interface.

This module defines the abstract base class that all language parsers must
implement. It provides a consistent interface for parsing source code files
and extracting code entities.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .models import ParseResult


class BaseParser(ABC):
    """Abstract base class for source code parsers.

    All language-specific parsers must inherit from this class and implement
    the abstract methods for parsing files and source code strings.

    Attributes:
        supported_languages: Set of language identifiers this parser supports.
    """

    supported_languages: set[str] = set()

    @abstractmethod
    async def parse_file(
        self,
        file_path: Path | str,
        *,
        encoding: str = "utf-8",
    ) -> ParseResult:
        """Parse a source code file and extract all code entities.

        Reads the file from disk and parses its contents, extracting all
        recognized code entities (functions, classes, imports, etc.).

        Args:
            file_path: Path to the source file to parse.
            encoding: Character encoding of the file. Defaults to utf-8.

        Returns:
            ParseResult containing the parsed module and any errors.

        Raises:
            FileNotFoundError: If the file does not exist.
            UnicodeDecodeError: If the file cannot be decoded with the
                specified encoding.
            ParserError: If a fatal parsing error occurs.
        """
        ...

    @abstractmethod
    async def parse_source(
        self,
        source_code: str,
        *,
        file_path: str | None = None,
        language: str | None = None,
    ) -> ParseResult:
        """Parse source code from a string.

        Parses the provided source code string and extracts all recognized
        code entities. Useful for parsing code that is not yet saved to disk
        or for testing purposes.

        Args:
            source_code: The source code to parse.
            file_path: Optional virtual file path for entity IDs. If not
                provided, a placeholder will be used.
            language: Language identifier. If not provided, must be
                inferrable from file_path or parser configuration.

        Returns:
            ParseResult containing the parsed module and any errors.

        Raises:
            ValueError: If language cannot be determined.
            ParserError: If a fatal parsing error occurs.
        """
        ...

    def supports_language(self, language: str) -> bool:
        """Check if this parser supports the given language.

        Args:
            language: Language identifier to check (e.g., 'python', 'javascript').

        Returns:
            True if the language is supported, False otherwise.
        """
        return language.lower() in self.supported_languages

    @staticmethod
    def detect_language(file_path: Path | str) -> str | None:
        """Detect the programming language from a file path.

        Uses file extension to determine the likely programming language.

        Args:
            file_path: Path to the file.

        Returns:
            Language identifier if detected, None otherwise.
        """
        extension_map: dict[str, str] = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mts": "typescript",
            ".cts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".swift": "swift",
            ".scala": "scala",
            ".lua": "lua",
            ".r": "r",
            ".R": "r",
        }

        path = Path(file_path) if isinstance(file_path, str) else file_path
        suffix = path.suffix.lower()

        return extension_map.get(suffix)


class ParserError(Exception):
    """Exception raised for parser errors.

    Attributes:
        message: Explanation of the error.
        file_path: Path to the file being parsed when error occurred.
        line: Line number where error occurred, if known.
        column: Column number where error occurred, if known.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        """Initialize the ParserError.

        Args:
            message: Explanation of the error.
            file_path: Path to the file being parsed.
            line: Line number where error occurred.
            column: Column number where error occurred.
        """
        self.message = message
        self.file_path = file_path
        self.line = line
        self.column = column

        # Build detailed error message
        details = []
        if file_path:
            details.append(f"file={file_path}")
        if line is not None:
            details.append(f"line={line}")
        if column is not None:
            details.append(f"column={column}")

        full_message = f"{message} ({', '.join(details)})" if details else message

        super().__init__(full_message)
