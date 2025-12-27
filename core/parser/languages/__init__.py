"""Language-specific entity extractors.

This module contains extractors for different programming languages.
Each extractor knows how to traverse a tree-sitter AST for its language
and extract code entities.

Supported languages:
    - Python (python.py)
    - JavaScript (javascript.py) [TODO]
    - TypeScript (typescript.py) [TODO]
"""

from .base import BaseExtractor
from .python import PythonExtractor

# Registry of language extractors
_extractors: dict[str, type[BaseExtractor]] = {
    "python": PythonExtractor,
}


def register_extractor(language: str, extractor_class: type[BaseExtractor]) -> None:
    """Register a language extractor.

    Args:
        language: Language identifier (e.g., 'python', 'javascript').
        extractor_class: The extractor class to register.
    """
    _extractors[language.lower()] = extractor_class


def get_extractor(language: str) -> BaseExtractor:
    """Get an extractor instance for the given language.

    Args:
        language: Language identifier.

    Returns:
        An instance of the appropriate extractor.

    Raises:
        ValueError: If no extractor is registered for the language.
    """
    language = language.lower()
    if language not in _extractors:
        raise ValueError(f"No extractor registered for language: {language}")
    return _extractors[language]()


def supported_languages() -> list[str]:
    """Get list of languages with registered extractors.

    Returns:
        List of supported language identifiers.
    """
    return list(_extractors.keys())


__all__ = [
    "BaseExtractor",
    "PythonExtractor",
    "register_extractor",
    "get_extractor",
    "supported_languages",
]
