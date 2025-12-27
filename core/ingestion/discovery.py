"""File discovery for the ingestion module.

This module provides functionality to discover source files in a repository,
respecting .gitignore patterns and applying include/exclude filters based
on glob patterns, file size, and language.
"""

import asyncio
import fnmatch
import hashlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import structlog

from core.parser import BaseParser

from .models import FileInfo, IngestionConfig

logger = structlog.get_logger(__name__)


class FileDiscovery:
    """Discovers source files in a repository.

    Walks the repository directory structure, identifies source code files,
    and applies filtering based on configuration settings including gitignore
    patterns, glob patterns, file size limits, and language filters.

    Attributes:
        config: Ingestion configuration controlling discovery behavior.
    """

    def __init__(self, config: IngestionConfig | None = None) -> None:
        """Initialize the FileDiscovery.

        Args:
            config: Ingestion configuration. Uses defaults if not provided.
        """
        self.config = config or IngestionConfig()
        self._gitignore_patterns: list[str] = []
        logger.debug("FileDiscovery initialized", config=self.config.model_dump())

    async def discover(
        self,
        repo_path: str | Path,
    ) -> AsyncGenerator[FileInfo, None]:
        """Discover all source files in a repository.

        Walks the repository directory and yields FileInfo for each discovered
        source file that passes all filters.

        Args:
            repo_path: Path to the repository root.

        Yields:
            FileInfo for each discovered source file.
        """
        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        root = root.resolve()

        if not root.exists():
            logger.error(f"Repository path does not exist: {root}")
            return

        if not root.is_dir():
            logger.error(f"Repository path is not a directory: {root}")
            return

        # Load gitignore patterns if configured
        if self.config.respect_gitignore:
            await self._load_gitignore(root)

        logger.info(f"Starting file discovery in {root}")
        file_count = 0

        # Walk directory tree
        for file_path in await self._walk_directory(root):
            relative_path = file_path.relative_to(root)
            relative_str = str(relative_path).replace("\\", "/")

            # Apply filters
            if not self._should_include(relative_str):
                continue

            # Check file size
            try:
                stat = file_path.stat()
                size_kb = stat.st_size / 1024

                if size_kb > self.config.max_file_size_kb:
                    logger.debug(f"Skipping large file: {relative_str} ({size_kb:.1f}KB)")
                    continue

                # Detect language
                language = BaseParser.detect_language(file_path)

                # Filter by language if configured
                if not self.config.should_include_language(language):
                    continue

                # Calculate file hash
                file_hash = await self._calculate_hash(file_path)

                # Get modification time
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

                file_count += 1
                yield FileInfo(
                    path=relative_str,
                    language=language,
                    size=stat.st_size,
                    hash=file_hash,
                    last_modified=mtime,
                )

            except OSError as e:
                logger.warning(f"Failed to process file {relative_str}: {e}")
                continue

        logger.info(f"File discovery complete: {file_count} files found")

    async def discover_all(self, repo_path: str | Path) -> list[FileInfo]:
        """Discover all source files and return as a list.

        Convenience method that collects all discovered files into a list.

        Args:
            repo_path: Path to the repository root.

        Returns:
            List of FileInfo for all discovered files.
        """
        files: list[FileInfo] = []
        async for file_info in self.discover(repo_path):
            files.append(file_info)
        return files

    async def _walk_directory(self, root: Path) -> list[Path]:
        """Walk directory tree and collect all files.

        Runs in a thread pool to avoid blocking the event loop.

        Args:
            root: Root directory to walk.

        Returns:
            List of all file paths.
        """

        def _do_walk() -> list[Path]:
            files: list[Path] = []
            for path in root.rglob("*"):
                if path.is_file():
                    files.append(path)
            return files

        return await asyncio.get_event_loop().run_in_executor(None, _do_walk)

    async def _load_gitignore(self, repo_path: Path) -> None:
        """Load .gitignore patterns from repository.

        Args:
            repo_path: Path to the repository root.
        """
        gitignore_path = repo_path / ".gitignore"

        if not gitignore_path.exists():
            self._gitignore_patterns = []
            return

        def _read_gitignore() -> list[str]:
            patterns: list[str] = []
            try:
                content = gitignore_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        patterns.append(line)
            except OSError as e:
                logger.warning(f"Failed to read .gitignore: {e}")
            return patterns

        self._gitignore_patterns = await asyncio.get_event_loop().run_in_executor(
            None, _read_gitignore
        )
        logger.debug(f"Loaded {len(self._gitignore_patterns)} gitignore patterns")

    def _should_include(self, relative_path: str) -> bool:
        """Check if a file should be included based on all filters.

        Args:
            relative_path: Path relative to repository root.

        Returns:
            True if file should be included, False otherwise.
        """
        # Check gitignore patterns
        if self.config.respect_gitignore and self._matches_gitignore(relative_path):
            return False

        # Check exclude patterns
        if self._matches_patterns(relative_path, self.config.exclude_patterns):
            return False

        # Check include patterns
        return self._matches_patterns(relative_path, self.config.include_patterns)

    def _matches_gitignore(self, relative_path: str) -> bool:
        """Check if path matches any gitignore pattern.

        Implements simplified gitignore matching logic.

        Args:
            relative_path: Path relative to repository root.

        Returns:
            True if path matches a gitignore pattern.
        """
        # Normalize path separators
        path = relative_path.replace("\\", "/")
        path_parts = path.split("/")

        for pattern in self._gitignore_patterns:
            # Handle negation patterns (we don't fully support them yet)
            if pattern.startswith("!"):
                continue

            # Handle directory patterns (ending with /)
            if pattern.endswith("/"):
                pattern = pattern[:-1]
                # Check if any path component matches
                if any(fnmatch.fnmatch(part, pattern) for part in path_parts):
                    return True
                continue

            # Handle patterns with path separators
            if "/" in pattern:
                if pattern.startswith("/"):
                    # Anchored pattern (relative to repo root)
                    if fnmatch.fnmatch(path, pattern[1:]):
                        return True
                else:
                    # Pattern can match at any level
                    if fnmatch.fnmatch(path, pattern):
                        return True
                    if fnmatch.fnmatch(path, f"**/{pattern}"):
                        return True
            else:
                # Pattern matches filename or directory name
                filename = path_parts[-1]
                if fnmatch.fnmatch(filename, pattern):
                    return True
                # Also check if pattern matches any directory in path
                if any(fnmatch.fnmatch(part, pattern) for part in path_parts[:-1]):
                    return True

        return False

    def _matches_patterns(self, relative_path: str, patterns: list[str]) -> bool:
        """Check if path matches any of the glob patterns.

        Args:
            relative_path: Path relative to repository root.
            patterns: List of glob patterns to check.

        Returns:
            True if path matches at least one pattern.
        """
        path = relative_path.replace("\\", "/")

        for pattern in patterns:
            # Handle ** for recursive matching
            if "**" in pattern:
                # Convert ** to a regex-like pattern
                regex_pattern = pattern.replace("**", "*")
                if fnmatch.fnmatch(path, regex_pattern):
                    return True

                # Also try matching with the pattern as-is for simpler cases
                parts = pattern.split("**/")
                if len(parts) == 2 and parts[0] == "":
                    # Pattern like "**/foo" or "**/*.py"
                    suffix_pattern = parts[1]
                    if fnmatch.fnmatch(path, suffix_pattern):
                        return True
                    if "/" in path:
                        filename = path.split("/")[-1]
                        if fnmatch.fnmatch(filename, suffix_pattern):
                            return True
                        # Check full path against suffix
                        if fnmatch.fnmatch(path, f"*/{suffix_pattern}"):
                            return True
            else:
                if fnmatch.fnmatch(path, pattern):
                    return True

        return False

    async def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content.

        Args:
            file_path: Path to the file.

        Returns:
            Hex-encoded SHA-256 hash.
        """

        def _do_hash() -> str:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        return await asyncio.get_event_loop().run_in_executor(None, _do_hash)

    async def get_file_info(
        self,
        repo_path: str | Path,
        file_path: str,
    ) -> FileInfo | None:
        """Get FileInfo for a specific file.

        Args:
            repo_path: Path to the repository root.
            file_path: Relative path to the file.

        Returns:
            FileInfo if file exists and passes filters, None otherwise.
        """
        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        root = root.resolve()

        full_path = root / file_path
        if not full_path.exists() or not full_path.is_file():
            return None

        relative_str = file_path.replace("\\", "/")

        # Load gitignore if needed
        if self.config.respect_gitignore and not self._gitignore_patterns:
            await self._load_gitignore(root)

        # Apply filters
        if not self._should_include(relative_str):
            return None

        try:
            stat = full_path.stat()
            size_kb = stat.st_size / 1024

            if size_kb > self.config.max_file_size_kb:
                return None

            language = BaseParser.detect_language(full_path)
            if not self.config.should_include_language(language):
                return None

            file_hash = await self._calculate_hash(full_path)
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

            return FileInfo(
                path=relative_str,
                language=language,
                size=stat.st_size,
                hash=file_hash,
                last_modified=mtime,
            )
        except OSError as e:
            logger.warning(f"Failed to get file info for {file_path}: {e}")
            return None

    async def filter_by_extensions(
        self,
        files: list[FileInfo],
        extensions: list[str],
    ) -> list[FileInfo]:
        """Filter files by extension.

        Args:
            files: List of FileInfo to filter.
            extensions: List of extensions to include (with or without dot).

        Returns:
            Filtered list of FileInfo.
        """
        # Normalize extensions
        normalized_exts = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

        return [f for f in files if any(f.path.endswith(ext) for ext in normalized_exts)]
