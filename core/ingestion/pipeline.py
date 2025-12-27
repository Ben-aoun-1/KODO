"""Main ingestion pipeline orchestrator.

This module provides the IngestionPipeline class which orchestrates the full
repository ingestion process, including file discovery, parsing, and
incremental updates. It coordinates the parser, file discovery, and diff
analyzer components.
"""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from core.parser import CodeEntity, ParseResult, TreeSitterParser

from .diff import DiffAnalyzer
from .discovery import FileDiscovery
from .models import (
    FileInfo,
    IngestionConfig,
    IngestionError,
    IngestionResult,
    RepositoryInfo,
)
from .repo import RepositoryManager

logger = structlog.get_logger(__name__)


# Type for progress callback
ProgressCallback = Callable[[int, int, str], None]


class IngestionPipeline:
    """Orchestrates repository ingestion and parsing.

    Coordinates file discovery, parsing, and incremental updates to ingest
    source code from repositories. Supports both full and incremental
    ingestion modes with progress callbacks.

    Attributes:
        config: Ingestion configuration.
        parser: Tree-sitter parser instance.
        discovery: File discovery instance.
        diff_analyzer: Diff analyzer instance.
        repo_manager: Repository manager instance.
    """

    def __init__(
        self,
        config: IngestionConfig | None = None,
        parser: TreeSitterParser | None = None,
    ) -> None:
        """Initialize the IngestionPipeline.

        Args:
            config: Ingestion configuration. Uses defaults if not provided.
            parser: Parser instance. Creates new TreeSitterParser if not provided.
        """
        self.config = config or IngestionConfig()
        self.parser = parser or TreeSitterParser()
        self.discovery = FileDiscovery(self.config)
        self.diff_analyzer = DiffAnalyzer()
        self.repo_manager = RepositoryManager()

        logger.debug("IngestionPipeline initialized", config=self.config.model_dump())

    async def ingest_full(
        self,
        repo_path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Perform full ingestion of a repository.

        Discovers all source files and parses them, extracting code entities.
        This is typically used for initial repository indexing.

        Args:
            repo_path: Path to the repository root.
            progress_callback: Optional callback for progress updates.
                Called with (current, total, current_file) on each file.

        Returns:
            IngestionResult with statistics about the ingestion.
        """
        start_time = time.time()
        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        path_str = str(root.resolve())

        logger.info(f"Starting full ingestion of {path_str}")

        # Get repository info
        repo_info = await self.repo_manager.open_local(path_str)

        # Discover all files
        files = await self.discovery.discover_all(path_str)
        total_files = len(files)

        logger.info(f"Discovered {total_files} files to process")

        # Process files in batches
        entities_extracted = 0
        files_processed = 0
        errors: list[IngestionError] = []

        for batch_start in range(0, total_files, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, total_files)
            batch = files[batch_start:batch_end]

            # Process batch concurrently
            tasks = [self._process_file(root, file_info) for file_info in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                file_info = batch[i]
                current_index = batch_start + i + 1

                # Report progress
                if progress_callback:
                    progress_callback(current_index, total_files, file_info.path)

                if isinstance(result, BaseException):
                    errors.append(
                        IngestionError(
                            file_path=file_info.path,
                            error_type=type(result).__name__,
                            message=str(result),
                            line=None,
                        )
                    )
                    logger.warning(f"Failed to process {file_info.path}: {result}")
                elif result is not None:
                    files_processed += 1
                    entities_extracted += self._count_entities(result)

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Full ingestion complete",
            files_processed=files_processed,
            entities_extracted=entities_extracted,
            errors=len(errors),
            duration_ms=duration_ms,
        )

        return IngestionResult(
            repo_id=repo_info.id,
            commit_sha=repo_info.last_commit or "",
            files_processed=files_processed,
            entities_extracted=entities_extracted,
            errors=errors,
            duration_ms=duration_ms,
            is_incremental=False,
        )

    async def ingest_incremental(
        self,
        repo_path: str | Path,
        since_commit: str,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestionResult:
        """Perform incremental ingestion since a specific commit.

        Only processes files that have changed since the given commit,
        making it much faster than full ingestion for updates.

        Args:
            repo_path: Path to the repository root.
            since_commit: Commit SHA to use as baseline.
            progress_callback: Optional callback for progress updates.

        Returns:
            IngestionResult with statistics about the ingestion.
        """
        start_time = time.time()
        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        path_str = str(root.resolve())

        logger.info(f"Starting incremental ingestion from {since_commit[:8]}")

        # Get repository info
        repo_info = await self.repo_manager.open_local(path_str)
        current_commit = repo_info.last_commit or ""

        # Check if there are any changes
        has_changes = await self.diff_analyzer.has_changes(path_str, since_commit, current_commit)

        if not has_changes:
            logger.info("No changes detected since last commit")
            return IngestionResult(
                repo_id=repo_info.id,
                commit_sha=current_commit,
                files_processed=0,
                entities_extracted=0,
                errors=[],
                duration_ms=int((time.time() - start_time) * 1000),
                is_incremental=True,
            )

        # Get changes
        changes = await self.diff_analyzer.get_changes_since_commit(
            path_str, since_commit, current_commit
        )

        # Filter by language if configured
        if self.config.languages:
            changes = await self.diff_analyzer.filter_by_language(changes, self.config.languages)

        # Categorize changes
        files_to_process, files_to_remove = await self.diff_analyzer.get_files_to_reindex(changes)

        total_files = len(files_to_process)
        logger.info(f"Processing {total_files} changed files, removing {len(files_to_remove)}")

        # Process changed files
        entities_extracted = 0
        files_processed = 0
        errors: list[IngestionError] = []

        for i, file_path in enumerate(files_to_process):
            if progress_callback:
                progress_callback(i + 1, total_files, file_path)

            # Get file info and process
            file_info = await self.discovery.get_file_info(path_str, file_path)

            if file_info is None:
                # File may have been deleted or filtered out
                continue

            try:
                result = await self._process_file(root, file_info)
                if result is not None:
                    files_processed += 1
                    entities_extracted += self._count_entities(result)
            except Exception as e:
                errors.append(
                    IngestionError(
                        file_path=file_path,
                        error_type=type(e).__name__,
                        message=str(e),
                        line=None,
                    )
                )
                logger.warning(f"Failed to process {file_path}: {e}")

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Incremental ingestion complete",
            files_processed=files_processed,
            entities_extracted=entities_extracted,
            files_removed=len(files_to_remove),
            errors=len(errors),
            duration_ms=duration_ms,
        )

        return IngestionResult(
            repo_id=repo_info.id,
            commit_sha=current_commit,
            files_processed=files_processed,
            entities_extracted=entities_extracted,
            errors=errors,
            duration_ms=duration_ms,
            is_incremental=True,
        )

    async def ingest_file(
        self,
        file_path: str | Path,
    ) -> list[CodeEntity]:
        """Parse a single file and extract code entities.

        Useful for processing individual files outside of full repository
        ingestion, such as for live editing support.

        Args:
            file_path: Absolute path to the file.

        Returns:
            List of CodeEntity objects extracted from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ParserError: If parsing fails.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        abs_path = str(path.resolve())

        logger.debug(f"Ingesting single file: {abs_path}")

        # Parse the file
        result = await self.parser.parse_file(abs_path)

        # Extract all entities from the module
        entities = self._extract_all_entities(result)

        logger.debug(f"Extracted {len(entities)} entities from {abs_path}")

        return entities

    async def ingest_files(
        self,
        file_paths: list[str | Path],
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[CodeEntity], list[IngestionError]]:
        """Parse multiple files and extract code entities.

        Args:
            file_paths: List of absolute paths to files.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple of (all extracted entities, errors encountered).
        """
        all_entities: list[CodeEntity] = []
        errors: list[IngestionError] = []
        total = len(file_paths)

        for i, file_path in enumerate(file_paths):
            path_str = str(file_path)

            if progress_callback:
                progress_callback(i + 1, total, path_str)

            try:
                entities = await self.ingest_file(file_path)
                all_entities.extend(entities)
            except Exception as e:
                errors.append(
                    IngestionError(
                        file_path=path_str,
                        error_type=type(e).__name__,
                        message=str(e),
                        line=None,
                    )
                )
                logger.warning(f"Failed to ingest {path_str}: {e}")

        return all_entities, errors

    async def _process_file(
        self,
        repo_root: Path,
        file_info: FileInfo,
    ) -> ParseResult | None:
        """Process a single file from the repository.

        Args:
            repo_root: Path to the repository root.
            file_info: FileInfo for the file to process.

        Returns:
            ParseResult if successful, None if file should be skipped.
        """
        full_path = repo_root / file_info.path

        if not full_path.exists():
            logger.warning(f"File no longer exists: {file_info.path}")
            return None

        try:
            result = await self.parser.parse_file(full_path)
            return result
        except Exception:
            # Re-raise to be handled by caller
            raise

    def _count_entities(self, result: ParseResult) -> int:
        """Count total entities in a parse result.

        Args:
            result: ParseResult to count entities from.

        Returns:
            Total number of entities.
        """
        module = result.module
        count = 1  # Count the module itself

        count += len(module.imports)
        count += len(module.functions)
        count += len(module.variables)

        for cls in module.classes:
            count += 1  # The class itself
            count += len(cls.methods)
            count += len(cls.attributes)

        return count

    def _extract_all_entities(self, result: ParseResult) -> list[CodeEntity]:
        """Extract all entities from a parse result as a flat list.

        Args:
            result: ParseResult to extract from.

        Returns:
            List of all CodeEntity objects.
        """
        entities: list[CodeEntity] = []
        module = result.module

        # Add module
        entities.append(module)

        # Add imports
        entities.extend(module.imports)

        # Add functions
        entities.extend(module.functions)

        # Add variables
        entities.extend(module.variables)

        # Add classes and their methods
        for cls in module.classes:
            entities.append(cls)
            entities.extend(cls.methods)

        return entities

    async def get_repository_info(self, repo_path: str | Path) -> RepositoryInfo:
        """Get information about a repository.

        Args:
            repo_path: Path to the repository.

        Returns:
            RepositoryInfo for the repository.
        """
        path_str = str(Path(repo_path).resolve())
        return await self.repo_manager.open_local(path_str)

    async def estimate_ingestion_size(
        self,
        repo_path: str | Path,
    ) -> dict[str, int]:
        """Estimate the size of a full ingestion.

        Args:
            repo_path: Path to the repository.

        Returns:
            Dictionary with file counts by language and total size.
        """
        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        path_str = str(root.resolve())

        files = await self.discovery.discover_all(path_str)

        by_language: dict[str, int] = {}
        total_size = 0

        for file_info in files:
            lang = file_info.language or "unknown"
            by_language[lang] = by_language.get(lang, 0) + 1
            total_size += file_info.size

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            **by_language,
        }

    async def validate_repository(self, repo_path: str | Path) -> list[str]:
        """Validate that a repository can be ingested.

        Checks for common issues that would prevent successful ingestion.

        Args:
            repo_path: Path to the repository.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []
        root = Path(repo_path) if isinstance(repo_path, str) else repo_path

        if not root.exists():
            errors.append(f"Path does not exist: {root}")
            return errors

        if not root.is_dir():
            errors.append(f"Path is not a directory: {root}")
            return errors

        # Check if it's a git repository
        is_git = await self.repo_manager.is_git_repo(root)
        if not is_git:
            errors.append("Path is not a Git repository")

        # Check for parseable files
        files = await self.discovery.discover_all(root)
        if not files:
            errors.append("No parseable source files found")

        return errors
