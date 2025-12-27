"""Git repository operations for the ingestion module.

This module provides Git-related functionality including cloning repositories,
pulling updates, getting commit information, and reading file contents at
specific commits. Uses GitPython for Git operations.
"""

import asyncio
import contextlib
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import structlog

from .models import RepositoryInfo

if TYPE_CHECKING:
    from git import Repo

logger = structlog.get_logger(__name__)


class RepositoryError(Exception):
    """Exception raised for repository operation errors.

    Attributes:
        message: Explanation of the error.
        repo_path: Path to the repository, if applicable.
    """

    def __init__(self, message: str, repo_path: str | None = None) -> None:
        """Initialize the RepositoryError.

        Args:
            message: Explanation of the error.
            repo_path: Path to the repository.
        """
        self.message = message
        self.repo_path = repo_path

        full_message = f"{message} (repo={repo_path})" if repo_path else message
        super().__init__(full_message)


class RepositoryManager:
    """Manages Git repository operations.

    Provides async methods for cloning, pulling, and inspecting Git repositories.
    All blocking Git operations are executed in a thread pool to maintain
    async compatibility.

    Attributes:
        clone_timeout: Timeout in seconds for clone operations.
        pull_timeout: Timeout in seconds for pull operations.
    """

    def __init__(
        self,
        clone_timeout: int = 300,
        pull_timeout: int = 120,
    ) -> None:
        """Initialize the RepositoryManager.

        Args:
            clone_timeout: Timeout in seconds for clone operations.
            pull_timeout: Timeout in seconds for pull operations.
        """
        self.clone_timeout = clone_timeout
        self.pull_timeout = pull_timeout
        logger.debug("RepositoryManager initialized")

    def _get_repo(self, path: str | Path) -> "Repo":
        """Get a GitPython Repo object for a path.

        Args:
            path: Path to the repository.

        Returns:
            GitPython Repo instance.

        Raises:
            RepositoryError: If path is not a valid Git repository.
        """
        from git import InvalidGitRepositoryError, NoSuchPathError, Repo

        path_str = str(path)

        try:
            return Repo(path_str)
        except NoSuchPathError:
            raise RepositoryError(f"Path does not exist: {path_str}", repo_path=path_str)
        except InvalidGitRepositoryError:
            raise RepositoryError(f"Not a valid Git repository: {path_str}", repo_path=path_str)

    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from a URL.

        Args:
            url: Repository URL.

        Returns:
            Repository name extracted from URL.
        """
        # Handle both HTTPS and SSH URLs
        parsed = urlparse(url)

        if parsed.path:
            # Remove .git suffix and get the last path component
            path = parsed.path.rstrip("/")
            if path.endswith(".git"):
                path = path[:-4]
            return path.split("/")[-1]

        return "unknown"

    def _generate_repo_id(self, name: str, url: str | None, path: str) -> str:
        """Generate a unique repository ID.

        Args:
            name: Repository name.
            url: Repository URL if available.
            path: Local path to repository.

        Returns:
            Unique identifier for the repository.
        """
        # Use URL if available, otherwise use path
        identifier = url or path
        hash_input = f"{name}:{identifier}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def clone(
        self,
        url: str,
        destination: str | Path,
        branch: str | None = None,
        depth: int | None = None,
    ) -> RepositoryInfo:
        """Clone a repository from a URL.

        Args:
            url: Remote repository URL.
            destination: Local path where repository will be cloned.
            branch: Specific branch to clone, None for default.
            depth: Clone depth for shallow clone, None for full clone.

        Returns:
            RepositoryInfo for the cloned repository.

        Raises:
            RepositoryError: If clone fails.
        """
        from git import Repo

        dest_path = Path(destination) if isinstance(destination, str) else destination
        dest_str = str(dest_path.resolve())

        logger.info(f"Cloning repository from {url} to {dest_str}")

        def _do_clone() -> "Repo":
            kwargs: dict[str, Any] = {"url": url, "to_path": dest_str}

            if branch:
                kwargs["branch"] = branch

            if depth:
                kwargs["depth"] = depth

            return Repo.clone_from(**kwargs)

        try:
            repo = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _do_clone),
                timeout=self.clone_timeout,
            )
        except TimeoutError:
            raise RepositoryError(
                f"Clone operation timed out after {self.clone_timeout}s",
                repo_path=dest_str,
            )
        except Exception as e:
            raise RepositoryError(f"Failed to clone repository: {e}", repo_path=dest_str)

        # Get repository info
        name = self._extract_repo_name(url)
        default_branch = self._get_default_branch(repo)
        commit_sha = repo.head.commit.hexsha

        repo_id = self._generate_repo_id(name, url, dest_str)

        logger.info(
            f"Successfully cloned {name} at commit {commit_sha[:8]}",
            repo_id=repo_id,
        )

        return RepositoryInfo(
            id=repo_id,
            name=name,
            url=url,
            local_path=dest_str,
            default_branch=default_branch,
            last_indexed=None,
            last_commit=commit_sha,
        )

    async def open_local(self, path: str | Path) -> RepositoryInfo:
        """Open an existing local repository.

        Args:
            path: Path to the local repository.

        Returns:
            RepositoryInfo for the repository.

        Raises:
            RepositoryError: If path is not a valid Git repository.
        """
        repo_path = Path(path) if isinstance(path, str) else path
        path_str = str(repo_path.resolve())

        logger.debug(f"Opening local repository at {path_str}")

        def _do_open() -> tuple["Repo", str, str, str | None]:
            repo = self._get_repo(path_str)
            default_branch = self._get_default_branch(repo)
            commit_sha = repo.head.commit.hexsha

            # Try to get remote URL
            url = None
            if repo.remotes:
                with contextlib.suppress(AttributeError):
                    url = repo.remotes.origin.url

            return repo, default_branch, commit_sha, url

        try:
            _, default_branch, commit_sha, url = await asyncio.get_event_loop().run_in_executor(
                None, _do_open
            )
        except RepositoryError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to open repository: {e}", repo_path=path_str)

        name = repo_path.name
        repo_id = self._generate_repo_id(name, url, path_str)

        logger.info(
            f"Opened repository {name} at commit {commit_sha[:8]}",
            repo_id=repo_id,
        )

        return RepositoryInfo(
            id=repo_id,
            name=name,
            url=url,
            local_path=path_str,
            default_branch=default_branch,
            last_indexed=None,
            last_commit=commit_sha,
        )

    async def pull(self, path: str | Path, remote: str = "origin") -> str:
        """Pull latest changes from remote.

        Args:
            path: Path to the local repository.
            remote: Remote name to pull from.

        Returns:
            New HEAD commit SHA after pull.

        Raises:
            RepositoryError: If pull fails.
        """
        repo_path = Path(path) if isinstance(path, str) else path
        path_str = str(repo_path.resolve())

        logger.info(f"Pulling latest changes for {path_str}")

        def _do_pull() -> str:
            repo = self._get_repo(path_str)

            try:
                remote_obj = repo.remote(remote)
            except ValueError:
                raise RepositoryError(f"Remote '{remote}' not found", repo_path=path_str)

            remote_obj.pull()
            hexsha: str = repo.head.commit.hexsha
            return hexsha

        try:
            commit_sha = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, _do_pull),
                timeout=self.pull_timeout,
            )
        except TimeoutError:
            raise RepositoryError(
                f"Pull operation timed out after {self.pull_timeout}s",
                repo_path=path_str,
            )
        except RepositoryError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to pull: {e}", repo_path=path_str)

        logger.info(f"Pull complete, now at commit {commit_sha[:8]}")
        return commit_sha

    async def get_current_commit(self, path: str | Path) -> str:
        """Get the current HEAD commit SHA.

        Args:
            path: Path to the repository.

        Returns:
            Current HEAD commit SHA.

        Raises:
            RepositoryError: If repository cannot be accessed.
        """
        repo_path = Path(path) if isinstance(path, str) else path
        path_str = str(repo_path.resolve())

        def _get_commit() -> str:
            repo = self._get_repo(path_str)
            hexsha: str = repo.head.commit.hexsha
            return hexsha

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _get_commit)
        except RepositoryError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to get current commit: {e}", repo_path=path_str)

    async def get_commit_timestamp(
        self, path: str | Path, commit_sha: str | None = None
    ) -> datetime:
        """Get the timestamp of a commit.

        Args:
            path: Path to the repository.
            commit_sha: Commit SHA, None for HEAD.

        Returns:
            Commit timestamp as datetime.

        Raises:
            RepositoryError: If commit cannot be found.
        """
        repo_path = Path(path) if isinstance(path, str) else path
        path_str = str(repo_path.resolve())

        def _get_timestamp() -> datetime:
            repo = self._get_repo(path_str)

            if commit_sha:
                try:
                    commit = repo.commit(commit_sha)
                except Exception:
                    raise RepositoryError(f"Commit not found: {commit_sha}", repo_path=path_str)
            else:
                commit = repo.head.commit

            return datetime.fromtimestamp(commit.committed_date, tz=UTC)

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _get_timestamp)
        except RepositoryError:
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to get commit timestamp: {e}", repo_path=path_str)

    async def get_file_content(
        self,
        path: str | Path,
        file_path: str,
        commit_sha: str | None = None,
    ) -> str:
        """Get file content at a specific commit.

        Args:
            path: Path to the repository.
            file_path: Relative path to the file within the repository.
            commit_sha: Commit SHA, None for current working tree.

        Returns:
            File content as string.

        Raises:
            RepositoryError: If file cannot be read.
            FileNotFoundError: If file does not exist.
        """
        repo_path = Path(path) if isinstance(path, str) else path
        path_str = str(repo_path.resolve())

        def _get_content() -> str:
            if commit_sha:
                # Read from specific commit
                repo = self._get_repo(path_str)
                try:
                    commit = repo.commit(commit_sha)
                    blob = commit.tree / file_path
                    content: str = blob.data_stream.read().decode("utf-8")
                    return content
                except KeyError:
                    raise FileNotFoundError(f"File not found at commit {commit_sha}: {file_path}")
                except Exception as e:
                    raise RepositoryError(
                        f"Failed to read file from commit: {e}", repo_path=path_str
                    )
            else:
                # Read from working tree
                full_path = repo_path / file_path
                if not full_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                return full_path.read_text(encoding="utf-8")

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _get_content)
        except (FileNotFoundError, RepositoryError):
            raise
        except Exception as e:
            raise RepositoryError(f"Failed to read file content: {e}", repo_path=path_str)

    async def file_exists_at_commit(
        self,
        path: str | Path,
        file_path: str,
        commit_sha: str,
    ) -> bool:
        """Check if a file exists at a specific commit.

        Args:
            path: Path to the repository.
            file_path: Relative path to the file.
            commit_sha: Commit SHA to check.

        Returns:
            True if file exists at commit, False otherwise.
        """
        repo_path = Path(path) if isinstance(path, str) else path
        path_str = str(repo_path.resolve())

        def _check_exists() -> bool:
            repo = self._get_repo(path_str)
            try:
                commit = repo.commit(commit_sha)
                _ = commit.tree / file_path
                return True
            except KeyError:
                return False

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _check_exists)
        except Exception:
            return False

    def _get_default_branch(self, repo: "Repo") -> str:
        """Get the default branch name for a repository.

        Args:
            repo: GitPython Repo instance.

        Returns:
            Default branch name.
        """
        # Try to get from HEAD reference
        try:
            if repo.head.is_detached:
                # If detached HEAD, try common branch names
                for branch_name in ["main", "master", "develop"]:
                    if branch_name in [b.name for b in repo.branches]:
                        return branch_name
                # Fall back to first branch or "main"
                if repo.branches:
                    first_branch_name: str = repo.branches[0].name
                    return first_branch_name
                return "main"

            active_branch_name: str = repo.active_branch.name
            return active_branch_name
        except Exception:
            return "main"

    async def is_git_repo(self, path: str | Path) -> bool:
        """Check if a path is a Git repository.

        Args:
            path: Path to check.

        Returns:
            True if path is a Git repository, False otherwise.
        """
        repo_path = Path(path) if isinstance(path, str) else path

        def _check_repo() -> bool:
            from git import InvalidGitRepositoryError, NoSuchPathError, Repo

            try:
                Repo(str(repo_path))
                return True
            except (InvalidGitRepositoryError, NoSuchPathError):
                return False

        return await asyncio.get_event_loop().run_in_executor(None, _check_repo)
