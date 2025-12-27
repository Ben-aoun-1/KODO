"""Git diff parsing for incremental updates.

This module provides functionality to detect and categorize file changes
between commits, enabling incremental repository indexing by only processing
files that have changed since the last indexed commit.
"""

import asyncio
from pathlib import Path

import structlog

from core.parser import BaseParser

from .models import FileChange, FileChangeType

logger = structlog.get_logger(__name__)


class DiffError(Exception):
    """Exception raised for diff operation errors.

    Attributes:
        message: Explanation of the error.
        repo_path: Path to the repository, if applicable.
    """

    def __init__(self, message: str, repo_path: str | None = None) -> None:
        """Initialize the DiffError.

        Args:
            message: Explanation of the error.
            repo_path: Path to the repository.
        """
        self.message = message
        self.repo_path = repo_path

        full_message = f"{message} (repo={repo_path})" if repo_path else message
        super().__init__(full_message)


class DiffAnalyzer:
    """Analyzes Git diffs to detect file changes.

    Provides methods to compare commits and identify which files have been
    added, modified, deleted, or renamed between two points in Git history.
    """

    def __init__(self) -> None:
        """Initialize the DiffAnalyzer."""
        logger.debug("DiffAnalyzer initialized")

    async def get_changes_since_commit(
        self,
        repo_path: str | Path,
        since_commit: str,
        until_commit: str | None = None,
    ) -> list[FileChange]:
        """Get all file changes between two commits.

        Args:
            repo_path: Path to the repository.
            since_commit: The base commit SHA (exclusive).
            until_commit: The target commit SHA (inclusive), None for HEAD.

        Returns:
            List of FileChange objects describing all changes.

        Raises:
            DiffError: If diff operation fails.
        """
        from git import InvalidGitRepositoryError, NoSuchPathError, Repo

        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        path_str = str(root.resolve())

        logger.info(
            f"Getting changes from {since_commit[:8]} to {until_commit[:8] if until_commit else 'HEAD'}"
        )

        def _get_diff() -> list[FileChange]:
            try:
                repo = Repo(path_str)
            except NoSuchPathError:
                raise DiffError(f"Path does not exist: {path_str}", repo_path=path_str)
            except InvalidGitRepositoryError:
                raise DiffError(f"Not a valid Git repository: {path_str}", repo_path=path_str)

            try:
                base_commit = repo.commit(since_commit)
            except Exception:
                raise DiffError(f"Commit not found: {since_commit}", repo_path=path_str)

            try:
                target_commit = repo.commit(until_commit) if until_commit else repo.head.commit
            except Exception:
                raise DiffError(f"Commit not found: {until_commit}", repo_path=path_str)

            # Get diff between commits
            diff_index = base_commit.diff(target_commit)

            changes: list[FileChange] = []

            # Process added files (in target but not in base)
            for diff_item in diff_index.iter_change_type("A"):
                file_path = diff_item.b_path
                language = BaseParser.detect_language(file_path)
                changes.append(
                    FileChange(
                        path=file_path,
                        change_type=FileChangeType.ADDED,
                        old_path=None,
                        language=language,
                    )
                )

            # Process modified files
            for diff_item in diff_index.iter_change_type("M"):
                file_path = diff_item.b_path
                language = BaseParser.detect_language(file_path)
                changes.append(
                    FileChange(
                        path=file_path,
                        change_type=FileChangeType.MODIFIED,
                        old_path=None,
                        language=language,
                    )
                )

            # Process deleted files
            for diff_item in diff_index.iter_change_type("D"):
                file_path = diff_item.a_path
                language = BaseParser.detect_language(file_path)
                changes.append(
                    FileChange(
                        path=file_path,
                        change_type=FileChangeType.DELETED,
                        old_path=None,
                        language=language,
                    )
                )

            # Process renamed files
            for diff_item in diff_index.iter_change_type("R"):
                new_path = diff_item.b_path
                old_path = diff_item.a_path
                language = BaseParser.detect_language(new_path)
                changes.append(
                    FileChange(
                        path=new_path,
                        change_type=FileChangeType.RENAMED,
                        old_path=old_path,
                        language=language,
                    )
                )

            return changes

        try:
            changes = await asyncio.get_event_loop().run_in_executor(None, _get_diff)
        except DiffError:
            raise
        except Exception as e:
            raise DiffError(f"Failed to get diff: {e}", repo_path=path_str)

        logger.info(
            f"Found {len(changes)} changed files",
            added=len([c for c in changes if c.change_type == FileChangeType.ADDED]),
            modified=len([c for c in changes if c.change_type == FileChangeType.MODIFIED]),
            deleted=len([c for c in changes if c.change_type == FileChangeType.DELETED]),
            renamed=len([c for c in changes if c.change_type == FileChangeType.RENAMED]),
        )

        return changes

    async def get_uncommitted_changes(
        self,
        repo_path: str | Path,
    ) -> list[FileChange]:
        """Get uncommitted changes in the working directory.

        Includes both staged and unstaged changes.

        Args:
            repo_path: Path to the repository.

        Returns:
            List of FileChange objects for uncommitted changes.

        Raises:
            DiffError: If operation fails.
        """
        from git import InvalidGitRepositoryError, NoSuchPathError, Repo

        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        path_str = str(root.resolve())

        logger.debug("Getting uncommitted changes")

        def _get_uncommitted() -> list[FileChange]:
            try:
                repo = Repo(path_str)
            except NoSuchPathError:
                raise DiffError(f"Path does not exist: {path_str}", repo_path=path_str)
            except InvalidGitRepositoryError:
                raise DiffError(f"Not a valid Git repository: {path_str}", repo_path=path_str)

            changes: list[FileChange] = []

            # Get staged changes (index vs HEAD)
            staged_diff = repo.index.diff(repo.head.commit)
            for diff_item in staged_diff:
                change_type = self._diff_change_type(diff_item.change_type)
                file_path = diff_item.b_path or diff_item.a_path
                language = BaseParser.detect_language(file_path)
                changes.append(
                    FileChange(
                        path=file_path,
                        change_type=change_type,
                        old_path=diff_item.a_path
                        if change_type == FileChangeType.RENAMED
                        else None,
                        language=language,
                    )
                )

            # Get unstaged changes (working dir vs index)
            unstaged_diff = repo.index.diff(None)
            for diff_item in unstaged_diff:
                change_type = self._diff_change_type(diff_item.change_type)
                file_path = diff_item.b_path or diff_item.a_path
                language = BaseParser.detect_language(file_path)
                # Avoid duplicates (prefer staged status)
                if not any(c.path == file_path for c in changes):
                    changes.append(
                        FileChange(
                            path=file_path,
                            change_type=change_type,
                            old_path=diff_item.a_path
                            if change_type == FileChangeType.RENAMED
                            else None,
                            language=language,
                        )
                    )

            # Get untracked files
            for file_path in repo.untracked_files:
                language = BaseParser.detect_language(file_path)
                changes.append(
                    FileChange(
                        path=file_path,
                        change_type=FileChangeType.ADDED,
                        old_path=None,
                        language=language,
                    )
                )

            return changes

        try:
            changes = await asyncio.get_event_loop().run_in_executor(None, _get_uncommitted)
        except DiffError:
            raise
        except Exception as e:
            raise DiffError(f"Failed to get uncommitted changes: {e}", repo_path=path_str)

        logger.debug(f"Found {len(changes)} uncommitted changes")
        return changes

    def _diff_change_type(self, git_change_type: str) -> FileChangeType:
        """Convert Git change type to FileChangeType.

        Args:
            git_change_type: Git change type character (A, M, D, R).

        Returns:
            Corresponding FileChangeType.
        """
        mapping = {
            "A": FileChangeType.ADDED,
            "M": FileChangeType.MODIFIED,
            "D": FileChangeType.DELETED,
            "R": FileChangeType.RENAMED,
        }
        return mapping.get(git_change_type, FileChangeType.MODIFIED)

    async def filter_by_language(
        self,
        changes: list[FileChange],
        languages: list[str],
    ) -> list[FileChange]:
        """Filter changes to only include specific languages.

        Args:
            changes: List of file changes.
            languages: List of language identifiers to include.

        Returns:
            Filtered list of changes.
        """
        normalized_langs = [lang.lower() for lang in languages]
        return [c for c in changes if c.language and c.language.lower() in normalized_langs]

    async def get_files_to_reindex(
        self,
        changes: list[FileChange],
    ) -> tuple[list[str], list[str]]:
        """Categorize changes into files to add/update and files to remove.

        Args:
            changes: List of file changes.

        Returns:
            Tuple of (files_to_process, files_to_remove).
        """
        to_process: list[str] = []
        to_remove: list[str] = []

        for change in changes:
            if change.change_type == FileChangeType.DELETED:
                to_remove.append(change.path)
            elif change.change_type == FileChangeType.RENAMED:
                if change.old_path:
                    to_remove.append(change.old_path)
                to_process.append(change.path)
            else:
                # ADDED or MODIFIED
                to_process.append(change.path)

        return to_process, to_remove

    async def get_changed_files_in_directory(
        self,
        repo_path: str | Path,
        directory: str,
        since_commit: str,
        until_commit: str | None = None,
    ) -> list[FileChange]:
        """Get changes only within a specific directory.

        Args:
            repo_path: Path to the repository.
            directory: Directory to filter by (relative path).
            since_commit: Base commit SHA.
            until_commit: Target commit SHA, None for HEAD.

        Returns:
            List of changes within the specified directory.
        """
        all_changes = await self.get_changes_since_commit(repo_path, since_commit, until_commit)

        # Normalize directory path
        dir_path = directory.replace("\\", "/")
        if not dir_path.endswith("/"):
            dir_path = dir_path + "/"

        return [
            c
            for c in all_changes
            if c.path.startswith(dir_path) or (c.old_path and c.old_path.startswith(dir_path))
        ]

    async def has_changes(
        self,
        repo_path: str | Path,
        since_commit: str,
        until_commit: str | None = None,
    ) -> bool:
        """Check if there are any changes between commits.

        More efficient than getting all changes when you only need to know
        if changes exist.

        Args:
            repo_path: Path to the repository.
            since_commit: Base commit SHA.
            until_commit: Target commit SHA, None for HEAD.

        Returns:
            True if there are changes, False otherwise.
        """
        from git import InvalidGitRepositoryError, NoSuchPathError, Repo

        root = Path(repo_path) if isinstance(repo_path, str) else repo_path
        path_str = str(root.resolve())

        def _check_changes() -> bool:
            try:
                repo = Repo(path_str)
            except (NoSuchPathError, InvalidGitRepositoryError):
                return False

            try:
                base_commit = repo.commit(since_commit)
                target_commit = repo.commit(until_commit) if until_commit else repo.head.commit
            except Exception:
                return False

            # Quick check - compare tree hashes
            base_hash: str = base_commit.tree.hexsha
            target_hash: str = target_commit.tree.hexsha
            return base_hash != target_hash

        try:
            result: bool = await asyncio.get_event_loop().run_in_executor(None, _check_changes)
            return result
        except Exception:
            return False
