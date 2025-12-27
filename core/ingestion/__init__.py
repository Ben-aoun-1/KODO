"""Ingestion module for repository parsing and indexing.

This module provides functionality for ingesting source code repositories,
including Git operations, file discovery, incremental updates via diff
analysis, and the main ingestion pipeline that orchestrates parsing.

Example:
    >>> from core.ingestion import IngestionPipeline, IngestionConfig
    >>> config = IngestionConfig(languages=["python", "typescript"])
    >>> pipeline = IngestionPipeline(config)
    >>> result = await pipeline.ingest_full("/path/to/repo")
    >>> print(f"Processed {result.files_processed} files")

    >>> # Incremental update
    >>> result = await pipeline.ingest_incremental("/path/to/repo", "abc123")
"""

from .diff import DiffAnalyzer, DiffError
from .discovery import FileDiscovery
from .models import (
    FileChange,
    FileChangeType,
    FileInfo,
    IngestionConfig,
    IngestionError,
    IngestionResult,
    RepositoryInfo,
)
from .pipeline import IngestionPipeline, ProgressCallback
from .repo import RepositoryError, RepositoryManager

__all__ = [
    # Main pipeline
    "IngestionPipeline",
    "ProgressCallback",
    # Models
    "RepositoryInfo",
    "FileInfo",
    "FileChange",
    "FileChangeType",
    "IngestionResult",
    "IngestionError",
    "IngestionConfig",
    # Git operations
    "RepositoryManager",
    "RepositoryError",
    # File discovery
    "FileDiscovery",
    # Diff analysis
    "DiffAnalyzer",
    "DiffError",
]
