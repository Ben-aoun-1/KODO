"""Pydantic models for the ingestion module.

This module defines the data models used for repository ingestion, including
repository metadata, file information, ingestion configuration, and results.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FileChangeType(str, Enum):
    """Type of file change detected in a diff."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class RepositoryInfo(BaseModel):
    """Information about a repository being ingested.

    Attributes:
        id: Unique identifier for the repository.
        name: Repository name (typically from URL or directory name).
        url: Remote URL if cloned, None for local repositories.
        local_path: Absolute path to the repository on disk.
        default_branch: Default branch name (e.g., 'main', 'master').
        last_indexed: Timestamp of last successful indexing, None if never indexed.
        last_commit: SHA of the last indexed commit.
    """

    id: str = Field(..., description="Unique repository identifier")
    name: str = Field(..., description="Repository name")
    url: str | None = Field(None, description="Remote URL if cloned")
    local_path: str = Field(..., description="Absolute path on disk")
    default_branch: str = Field("main", description="Default branch name")
    last_indexed: datetime | None = Field(None, description="Last indexing timestamp")
    last_commit: str | None = Field(None, description="SHA of last indexed commit")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class FileInfo(BaseModel):
    """Information about a source file.

    Attributes:
        path: Relative path from repository root.
        language: Detected programming language.
        size: File size in bytes.
        hash: Content hash (SHA-256) for change detection.
        last_modified: File modification timestamp.
    """

    path: str = Field(..., description="Relative path from repo root")
    language: str | None = Field(None, description="Detected programming language")
    size: int = Field(..., ge=0, description="File size in bytes")
    hash: str = Field(..., description="Content hash (SHA-256)")
    last_modified: datetime = Field(..., description="Last modification timestamp")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class FileChange(BaseModel):
    """Represents a changed file in a diff.

    Attributes:
        path: Relative path to the file.
        change_type: Type of change (added, modified, deleted, renamed).
        old_path: Previous path if renamed, None otherwise.
        language: Detected programming language.
    """

    path: str = Field(..., description="Relative path to the file")
    change_type: FileChangeType = Field(..., description="Type of change")
    old_path: str | None = Field(None, description="Previous path if renamed")
    language: str | None = Field(None, description="Detected programming language")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class IngestionError(BaseModel):
    """Details about an error during ingestion.

    Attributes:
        file_path: Path to the file that caused the error.
        error_type: Type/class of the error.
        message: Human-readable error message.
        line: Line number if applicable.
    """

    file_path: str = Field(..., description="Path to the problematic file")
    error_type: str = Field(..., description="Error type/class name")
    message: str = Field(..., description="Error message")
    line: int | None = Field(None, description="Line number if applicable")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"


class IngestionResult(BaseModel):
    """Result of a repository ingestion operation.

    Attributes:
        repo_id: Repository identifier.
        commit_sha: SHA of the commit that was indexed.
        files_processed: Number of files successfully processed.
        entities_extracted: Total number of code entities extracted.
        errors: List of errors encountered during ingestion.
        duration_ms: Total ingestion time in milliseconds.
        is_incremental: Whether this was an incremental update.
    """

    repo_id: str = Field(..., description="Repository identifier")
    commit_sha: str = Field(..., description="Indexed commit SHA")
    files_processed: int = Field(..., ge=0, description="Files successfully processed")
    entities_extracted: int = Field(..., ge=0, description="Total entities extracted")
    errors: list[IngestionError] = Field(default_factory=list, description="Errors encountered")
    duration_ms: int = Field(..., ge=0, description="Ingestion duration in ms")
    is_incremental: bool = Field(False, description="Whether incremental update")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of file processing.

        Returns:
            Percentage of files processed without errors (0.0 to 100.0).
        """
        total = self.files_processed + len(self.errors)
        if total == 0:
            return 100.0
        return (self.files_processed / total) * 100.0


class IngestionConfig(BaseModel):
    """Configuration for repository ingestion.

    Controls which files are included/excluded, size limits, and supported
    languages during the ingestion process.

    Attributes:
        include_patterns: Glob patterns for files to include (default all).
        exclude_patterns: Glob patterns for files to exclude.
        max_file_size_kb: Maximum file size to process in kilobytes.
        languages: List of languages to process, None for all supported.
        respect_gitignore: Whether to respect .gitignore patterns.
        batch_size: Number of files to process in parallel batches.
    """

    include_patterns: list[str] = Field(
        default_factory=lambda: ["**/*"],
        description="Glob patterns for files to include",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/.git/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/build/**",
            "**/.tox/**",
            "**/.pytest_cache/**",
            "**/.mypy_cache/**",
            "**/coverage/**",
            "**/*.min.js",
            "**/*.min.css",
            "**/*.map",
        ],
        description="Glob patterns for files to exclude",
    )
    max_file_size_kb: int = Field(500, ge=1, description="Maximum file size in kilobytes")
    languages: list[str] | None = Field(
        None, description="Languages to process, None for all supported"
    )
    respect_gitignore: bool = Field(True, description="Whether to respect .gitignore")
    batch_size: int = Field(50, ge=1, description="Batch size for parallel processing")

    class Config:
        """Pydantic model configuration."""

        frozen = False
        extra = "forbid"

    def should_include_language(self, language: str | None) -> bool:
        """Check if a language should be included based on config.

        Args:
            language: The language to check, or None if unknown.

        Returns:
            True if the language should be processed.
        """
        if language is None:
            return False
        if self.languages is None:
            return True
        return language.lower() in [lang.lower() for lang in self.languages]
