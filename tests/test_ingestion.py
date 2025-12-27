"""Tests for the ingestion module.

This module contains comprehensive tests for:
- Pydantic models (RepositoryInfo, FileInfo, IngestionConfig, IngestionResult)
- RepositoryManager (git operations with mocking)
- FileDiscovery (file discovery with filtering)
- DiffAnalyzer (git diff operations)
- IngestionPipeline (full ingestion workflow)
"""

import tempfile
from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from core.ingestion.models import (
    FileChange,
    FileChangeType,
    FileInfo,
    IngestionConfig,
    IngestionError,
    IngestionResult,
    RepositoryInfo,
)


def _tree_sitter_available() -> bool:
    """Check if tree-sitter Python parser is available."""
    try:
        import tree_sitter_python

        return True
    except ImportError:
        return False


# =============================================================================
# Model Tests
# =============================================================================


class TestRepositoryInfo:
    """Tests for RepositoryInfo model."""

    def test_create_repository_info_minimal(self):
        """Test creating RepositoryInfo with required fields only."""
        repo = RepositoryInfo(
            id="abc123",
            name="test-repo",
            local_path="/path/to/repo",
        )
        assert repo.id == "abc123"
        assert repo.name == "test-repo"
        assert repo.local_path == "/path/to/repo"
        assert repo.url is None
        assert repo.default_branch == "main"
        assert repo.last_indexed is None
        assert repo.last_commit is None

    def test_create_repository_info_full(self):
        """Test creating RepositoryInfo with all fields."""
        now = datetime.now(tz=UTC)
        repo = RepositoryInfo(
            id="abc123",
            name="test-repo",
            url="https://github.com/test/repo.git",
            local_path="/path/to/repo",
            default_branch="develop",
            last_indexed=now,
            last_commit="deadbeef123",
        )
        assert repo.url == "https://github.com/test/repo.git"
        assert repo.default_branch == "develop"
        assert repo.last_indexed == now
        assert repo.last_commit == "deadbeef123"

    def test_repository_info_forbids_extra_fields(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            RepositoryInfo(
                id="abc123",
                name="test-repo",
                local_path="/path/to/repo",
                extra_field="not allowed",
            )


class TestFileInfo:
    """Tests for FileInfo model."""

    def test_create_file_info(self):
        """Test creating FileInfo with all fields."""
        now = datetime.now(tz=UTC)
        file_info = FileInfo(
            path="src/main.py",
            language="python",
            size=1024,
            hash="abc123hash",
            last_modified=now,
        )
        assert file_info.path == "src/main.py"
        assert file_info.language == "python"
        assert file_info.size == 1024
        assert file_info.hash == "abc123hash"
        assert file_info.last_modified == now

    def test_file_info_size_must_be_non_negative(self):
        """Test that size must be >= 0."""
        with pytest.raises(ValidationError):
            FileInfo(
                path="test.py",
                size=-1,
                hash="abc",
                last_modified=datetime.now(tz=UTC),
            )

    def test_file_info_language_optional(self):
        """Test that language can be None."""
        file_info = FileInfo(
            path="unknown.xyz",
            language=None,
            size=100,
            hash="abc",
            last_modified=datetime.now(tz=UTC),
        )
        assert file_info.language is None


class TestFileChangeType:
    """Tests for FileChangeType enum."""

    def test_enum_values(self):
        """Test that all expected values exist."""
        assert FileChangeType.ADDED == "added"
        assert FileChangeType.MODIFIED == "modified"
        assert FileChangeType.DELETED == "deleted"
        assert FileChangeType.RENAMED == "renamed"

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert FileChangeType("added") == FileChangeType.ADDED
        assert FileChangeType("modified") == FileChangeType.MODIFIED


class TestFileChange:
    """Tests for FileChange model."""

    def test_create_file_change_added(self):
        """Test creating a FileChange for an added file."""
        change = FileChange(
            path="new_file.py",
            change_type=FileChangeType.ADDED,
            language="python",
        )
        assert change.path == "new_file.py"
        assert change.change_type == FileChangeType.ADDED
        assert change.old_path is None
        assert change.language == "python"

    def test_create_file_change_renamed(self):
        """Test creating a FileChange for a renamed file."""
        change = FileChange(
            path="new_name.py",
            change_type=FileChangeType.RENAMED,
            old_path="old_name.py",
            language="python",
        )
        assert change.path == "new_name.py"
        assert change.old_path == "old_name.py"
        assert change.change_type == FileChangeType.RENAMED


class TestIngestionConfig:
    """Tests for IngestionConfig model."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = IngestionConfig()
        assert config.include_patterns == ["**/*"]
        assert "**/node_modules/**" in config.exclude_patterns
        assert "**/.git/**" in config.exclude_patterns
        assert config.max_file_size_kb == 500
        assert config.languages is None
        assert config.respect_gitignore is True
        assert config.batch_size == 50

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = IngestionConfig(
            include_patterns=["**/*.py"],
            exclude_patterns=["**/test_*.py"],
            max_file_size_kb=100,
            languages=["python", "javascript"],
            respect_gitignore=False,
            batch_size=20,
        )
        assert config.include_patterns == ["**/*.py"]
        assert config.exclude_patterns == ["**/test_*.py"]
        assert config.max_file_size_kb == 100
        assert config.languages == ["python", "javascript"]
        assert config.respect_gitignore is False
        assert config.batch_size == 20

    def test_should_include_language_no_filter(self):
        """Test language inclusion when no filter is set."""
        config = IngestionConfig(languages=None)
        assert config.should_include_language("python") is True
        assert config.should_include_language("javascript") is True

    def test_should_include_language_with_filter(self):
        """Test language inclusion with filter set."""
        config = IngestionConfig(languages=["python", "JavaScript"])
        assert config.should_include_language("python") is True
        assert config.should_include_language("Python") is True  # Case insensitive
        assert config.should_include_language("javascript") is True
        assert config.should_include_language("typescript") is False

    def test_should_include_language_none(self):
        """Test that None language is not included."""
        config = IngestionConfig(languages=None)
        assert config.should_include_language(None) is False


class TestIngestionError:
    """Tests for IngestionError model."""

    def test_create_ingestion_error(self):
        """Test creating an IngestionError."""
        error = IngestionError(
            file_path="broken.py",
            error_type="SyntaxError",
            message="Invalid syntax on line 10",
            line=10,
        )
        assert error.file_path == "broken.py"
        assert error.error_type == "SyntaxError"
        assert error.message == "Invalid syntax on line 10"
        assert error.line == 10

    def test_create_ingestion_error_no_line(self):
        """Test creating an IngestionError without line number."""
        error = IngestionError(
            file_path="missing.py",
            error_type="FileNotFoundError",
            message="File not found",
        )
        assert error.line is None


class TestIngestionResult:
    """Tests for IngestionResult model."""

    def test_create_ingestion_result(self):
        """Test creating an IngestionResult."""
        result = IngestionResult(
            repo_id="abc123",
            commit_sha="deadbeef",
            files_processed=100,
            entities_extracted=500,
            errors=[],
            duration_ms=5000,
            is_incremental=False,
        )
        assert result.repo_id == "abc123"
        assert result.commit_sha == "deadbeef"
        assert result.files_processed == 100
        assert result.entities_extracted == 500
        assert result.errors == []
        assert result.duration_ms == 5000
        assert result.is_incremental is False

    def test_success_rate_no_errors(self):
        """Test success rate calculation with no errors."""
        result = IngestionResult(
            repo_id="abc123",
            commit_sha="deadbeef",
            files_processed=100,
            entities_extracted=500,
            errors=[],
            duration_ms=5000,
        )
        assert result.success_rate == 100.0

    def test_success_rate_with_errors(self):
        """Test success rate calculation with errors."""
        errors = [
            IngestionError(
                file_path="error1.py",
                error_type="Error",
                message="Error 1",
            ),
            IngestionError(
                file_path="error2.py",
                error_type="Error",
                message="Error 2",
            ),
        ]
        result = IngestionResult(
            repo_id="abc123",
            commit_sha="deadbeef",
            files_processed=8,
            entities_extracted=40,
            errors=errors,
            duration_ms=5000,
        )
        # 8 processed, 2 errors = 10 total, 80% success
        assert result.success_rate == 80.0

    def test_success_rate_no_files(self):
        """Test success rate when no files processed."""
        result = IngestionResult(
            repo_id="abc123",
            commit_sha="deadbeef",
            files_processed=0,
            entities_extracted=0,
            errors=[],
            duration_ms=100,
        )
        assert result.success_rate == 100.0


# =============================================================================
# RepositoryManager Tests
# =============================================================================


class TestRepositoryManager:
    """Tests for RepositoryManager class."""

    @pytest.fixture
    def repo_manager(self):
        """Create a RepositoryManager instance."""
        from core.ingestion.repo import RepositoryManager

        return RepositoryManager()

    def test_extract_repo_name_https(self, repo_manager):
        """Test extracting repo name from HTTPS URL."""
        name = repo_manager._extract_repo_name("https://github.com/owner/repo-name.git")
        assert name == "repo-name"

    def test_extract_repo_name_https_no_git_suffix(self, repo_manager):
        """Test extracting repo name from URL without .git suffix."""
        name = repo_manager._extract_repo_name("https://github.com/owner/repo-name")
        assert name == "repo-name"

    def test_extract_repo_name_ssh(self, repo_manager):
        """Test extracting repo name from SSH URL."""
        name = repo_manager._extract_repo_name("git@github.com:owner/repo-name.git")
        assert name == "repo-name"

    def test_generate_repo_id(self, repo_manager):
        """Test generating unique repo ID."""
        id1 = repo_manager._generate_repo_id("repo", "https://github.com/a/repo.git", "/path")
        id2 = repo_manager._generate_repo_id("repo", "https://github.com/b/repo.git", "/path")
        id3 = repo_manager._generate_repo_id("repo", "https://github.com/a/repo.git", "/path")

        assert len(id1) == 16
        assert id1 != id2  # Different URLs should produce different IDs
        assert id1 == id3  # Same inputs should produce same ID

    @pytest.mark.asyncio
    async def test_is_git_repo_true(self, repo_manager, temp_git_repo):
        """Test checking if path is a git repo (true case)."""
        is_repo = await repo_manager.is_git_repo(temp_git_repo)
        assert is_repo is True

    @pytest.mark.asyncio
    async def test_is_git_repo_false(self, repo_manager):
        """Test checking if path is a git repo (false case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            is_repo = await repo_manager.is_git_repo(tmpdir)
            assert is_repo is False

    @pytest.mark.asyncio
    async def test_open_local_valid_repo(self, repo_manager, temp_git_repo):
        """Test opening a valid local repository."""
        repo_info = await repo_manager.open_local(temp_git_repo)

        assert repo_info.name == "test_repo"
        assert repo_info.local_path == str(temp_git_repo.resolve())
        assert repo_info.last_commit is not None
        assert len(repo_info.last_commit) == 40  # Git SHA length
        assert repo_info.default_branch in ["main", "master"]

    @pytest.mark.asyncio
    async def test_open_local_invalid_path(self, repo_manager):
        """Test opening a non-existent path."""
        from core.ingestion.repo import RepositoryError

        with pytest.raises(RepositoryError):
            await repo_manager.open_local("/nonexistent/path")

    @pytest.mark.asyncio
    async def test_open_local_not_a_repo(self, repo_manager):
        """Test opening a directory that is not a git repo."""
        from core.ingestion.repo import RepositoryError

        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(RepositoryError):
            await repo_manager.open_local(tmpdir)

    @pytest.mark.asyncio
    async def test_get_current_commit(self, repo_manager, temp_git_repo):
        """Test getting current commit SHA."""
        commit_sha = await repo_manager.get_current_commit(temp_git_repo)
        assert len(commit_sha) == 40
        assert all(c in "0123456789abcdef" for c in commit_sha)

    @pytest.mark.asyncio
    async def test_get_file_content(self, repo_manager, temp_git_repo):
        """Test reading file content from working tree."""
        content = await repo_manager.get_file_content(
            temp_git_repo,
            "src/calculator.py",
        )
        assert 'VERSION = "1.0.0"' in content
        assert "class Calculator:" in content

    @pytest.mark.asyncio
    async def test_get_file_content_not_found(self, repo_manager, temp_git_repo):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            await repo_manager.get_file_content(
                temp_git_repo,
                "nonexistent.py",
            )

    @pytest.mark.asyncio
    async def test_clone_with_mock(self, repo_manager):
        """Test cloning with mocked git operations."""

        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "abc123def456789" + "0" * 25  # Full 40-char SHA
        mock_repo.active_branch.name = "main"
        mock_repo.branches = [MagicMock(name="main")]
        mock_repo.remotes = []
        mock_repo.head.is_detached = False

        # Patch git.Repo since it's imported inside the method
        with patch("git.Repo") as MockRepo:
            MockRepo.clone_from.return_value = mock_repo

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cloned"
                repo_info = await repo_manager.clone(
                    "https://github.com/test/repo.git",
                    dest,
                )

                assert repo_info.name == "repo"
                assert repo_info.url == "https://github.com/test/repo.git"
                MockRepo.clone_from.assert_called_once()


# =============================================================================
# FileDiscovery Tests
# =============================================================================


class TestFileDiscovery:
    """Tests for FileDiscovery class."""

    @pytest.fixture
    def discovery(self):
        """Create a FileDiscovery instance."""
        from core.ingestion.discovery import FileDiscovery

        return FileDiscovery()

    @pytest.fixture
    def discovery_with_config(self):
        """Create a FileDiscovery instance with custom config."""
        from core.ingestion.discovery import FileDiscovery

        config = IngestionConfig(
            max_file_size_kb=100,
            languages=["python"],
        )
        return FileDiscovery(config)

    @pytest.mark.asyncio
    async def test_discover_files_in_temp_directory(self, discovery, sample_python_files):
        """Test discovering files in a temporary directory."""
        root, _ = sample_python_files
        files = await discovery.discover_all(root)

        # Should find Python files (filtering out large file and txt file)
        paths = [f.path for f in files]
        assert any("module1.py" in p for p in paths)
        assert any("module2.py" in p for p in paths)

    @pytest.mark.asyncio
    async def test_discover_respects_gitignore(self, discovery, temp_git_repo):
        """Test that discovery respects .gitignore patterns."""
        files = await discovery.discover_all(temp_git_repo)
        paths = [f.path for f in files]

        # __pycache__ files should be excluded
        assert not any("__pycache__" in p for p in paths)
        assert not any(".pyc" in p for p in paths)

    @pytest.mark.asyncio
    async def test_discover_max_file_size(self, discovery_with_config, sample_python_files):
        """Test that large files are excluded."""
        root, _ = sample_python_files
        files = await discovery_with_config.discover_all(root)
        paths = [f.path for f in files]

        # The large file (~600KB) should be excluded (limit is 100KB)
        assert not any("large_file.py" in p for p in paths)

    @pytest.mark.asyncio
    async def test_discover_language_filter(self, sample_python_files):
        """Test filtering by language."""
        from core.ingestion.discovery import FileDiscovery

        config = IngestionConfig(languages=["python"])
        discovery = FileDiscovery(config)

        root, _ = sample_python_files
        files = await discovery.discover_all(root)

        # All files should be Python
        for f in files:
            assert f.language == "python"

    @pytest.mark.asyncio
    async def test_discover_calculates_hash(self, discovery, sample_python_files):
        """Test that file hash is calculated."""
        root, _ = sample_python_files
        files = await discovery.discover_all(root)

        for f in files:
            assert f.hash is not None
            assert len(f.hash) == 64  # SHA-256 hex length

    @pytest.mark.asyncio
    async def test_get_file_info(self, discovery, sample_python_files):
        """Test getting info for a specific file."""
        root, _ = sample_python_files
        file_info = await discovery.get_file_info(root, "module1.py")

        assert file_info is not None
        assert file_info.path == "module1.py"
        assert file_info.language == "python"
        assert file_info.size > 0

    @pytest.mark.asyncio
    async def test_get_file_info_nonexistent(self, discovery, sample_python_files):
        """Test getting info for non-existent file."""
        root, _ = sample_python_files
        file_info = await discovery.get_file_info(root, "nonexistent.py")
        assert file_info is None

    @pytest.mark.asyncio
    async def test_filter_by_extensions(self, discovery, sample_python_files):
        """Test filtering files by extension."""
        root, _ = sample_python_files
        files = await discovery.discover_all(root)

        # Filter to only .py files
        py_files = await discovery.filter_by_extensions(files, [".py"])
        for f in py_files:
            assert f.path.endswith(".py")

    def test_should_include_gitignore_pattern(self, discovery):
        """Test gitignore pattern matching."""
        discovery._gitignore_patterns = ["*.pyc", "__pycache__/", "build/"]

        # These should be excluded
        assert discovery._matches_gitignore("test.pyc") is True
        assert discovery._matches_gitignore("__pycache__/cache.py") is True

        # These should not be excluded
        assert discovery._matches_gitignore("test.py") is False

    def test_matches_patterns_glob(self, discovery):
        """Test glob pattern matching."""
        patterns = ["**/*.py", "src/*"]

        assert discovery._matches_patterns("test.py", patterns) is True
        assert discovery._matches_patterns("src/module.js", patterns) is True
        assert discovery._matches_patterns("lib/module.js", patterns) is False


# =============================================================================
# DiffAnalyzer Tests
# =============================================================================


class TestDiffAnalyzer:
    """Tests for DiffAnalyzer class."""

    @pytest.fixture
    def diff_analyzer(self):
        """Create a DiffAnalyzer instance."""
        from core.ingestion.diff import DiffAnalyzer

        return DiffAnalyzer()

    @pytest.mark.asyncio
    async def test_get_changes_between_commits(self, diff_analyzer, temp_git_repo_with_history):
        """Test getting changes between two commits."""
        repo_path, first_commit, second_commit = temp_git_repo_with_history

        changes = await diff_analyzer.get_changes_since_commit(
            repo_path, first_commit, second_commit
        )

        # Should have added, modified, and deleted files
        change_types = {c.change_type for c in changes}
        assert FileChangeType.ADDED in change_types
        assert FileChangeType.MODIFIED in change_types
        assert FileChangeType.DELETED in change_types

        # Verify specific changes
        paths = {c.path for c in changes}
        assert any("new_module.py" in p for p in paths)  # Added
        assert any("calculator.py" in p for p in paths)  # Modified
        assert any("helper.js" in p for p in paths)  # Deleted

    @pytest.mark.asyncio
    async def test_filter_by_language(self, diff_analyzer):
        """Test filtering changes by language."""
        changes = [
            FileChange(path="main.py", change_type=FileChangeType.MODIFIED, language="python"),
            FileChange(path="app.js", change_type=FileChangeType.MODIFIED, language="javascript"),
            FileChange(path="style.css", change_type=FileChangeType.MODIFIED, language="css"),
        ]

        filtered = await diff_analyzer.filter_by_language(changes, ["python", "javascript"])

        assert len(filtered) == 2
        languages = {c.language for c in filtered}
        assert "python" in languages
        assert "javascript" in languages
        assert "css" not in languages

    @pytest.mark.asyncio
    async def test_get_files_to_reindex(self, diff_analyzer):
        """Test categorizing changes into add/update vs remove."""
        changes = [
            FileChange(path="new.py", change_type=FileChangeType.ADDED, language="python"),
            FileChange(path="updated.py", change_type=FileChangeType.MODIFIED, language="python"),
            FileChange(path="old.py", change_type=FileChangeType.DELETED, language="python"),
            FileChange(
                path="renamed.py",
                change_type=FileChangeType.RENAMED,
                old_path="original.py",
                language="python",
            ),
        ]

        to_process, to_remove = await diff_analyzer.get_files_to_reindex(changes)

        assert "new.py" in to_process
        assert "updated.py" in to_process
        assert "renamed.py" in to_process
        assert "old.py" in to_remove
        assert "original.py" in to_remove  # Old path from rename

    @pytest.mark.asyncio
    async def test_has_changes_true(self, diff_analyzer, temp_git_repo_with_history):
        """Test checking if changes exist (true case)."""
        repo_path, first_commit, second_commit = temp_git_repo_with_history

        has_changes = await diff_analyzer.has_changes(repo_path, first_commit, second_commit)
        assert has_changes is True

    @pytest.mark.asyncio
    async def test_has_changes_false(self, diff_analyzer, temp_git_repo):
        """Test checking if changes exist (false case - same commit)."""
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
            text=True,
        )
        commit_sha = result.stdout.strip()

        has_changes = await diff_analyzer.has_changes(temp_git_repo, commit_sha, commit_sha)
        assert has_changes is False

    def test_diff_change_type_mapping(self, diff_analyzer):
        """Test Git change type to FileChangeType mapping."""
        assert diff_analyzer._diff_change_type("A") == FileChangeType.ADDED
        assert diff_analyzer._diff_change_type("M") == FileChangeType.MODIFIED
        assert diff_analyzer._diff_change_type("D") == FileChangeType.DELETED
        assert diff_analyzer._diff_change_type("R") == FileChangeType.RENAMED
        assert (
            diff_analyzer._diff_change_type("X") == FileChangeType.MODIFIED
        )  # Unknown defaults to modified


# =============================================================================
# IngestionPipeline Tests
# =============================================================================


class TestIngestionPipeline:
    """Tests for IngestionPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create an IngestionPipeline instance."""
        from core.ingestion.pipeline import IngestionPipeline

        return IngestionPipeline()

    @pytest.fixture
    def pipeline_with_config(self):
        """Create a pipeline with custom config."""
        from core.ingestion.pipeline import IngestionPipeline

        config = IngestionConfig(
            languages=["python"],
            batch_size=5,
        )
        return IngestionPipeline(config=config)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not _tree_sitter_available(),
        reason="tree-sitter Python parser not available",
    )
    async def test_ingest_single_file(self, pipeline, temp_python_file):
        """Test ingesting a single file."""
        entities = await pipeline.ingest_file(temp_python_file)

        # Should have at least the module entity
        assert len(entities) >= 1

        # Check that we got a module entity
        module_entities = [e for e in entities if e.type.value == "module"]
        assert len(module_entities) == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not _tree_sitter_available(),
        reason="tree-sitter Python parser not available",
    )
    async def test_ingest_multiple_files(self, pipeline, sample_python_files):
        """Test ingesting multiple files."""
        root, files = sample_python_files
        py_files = [f for f in files if f.suffix == ".py" and "large" not in str(f)]

        entities, errors = await pipeline.ingest_files(py_files)

        # Should have extracted entities
        assert len(entities) > 0
        # Some errors may occur for malformed files but small test files should be fine
        # Just verify we got some results

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not _tree_sitter_available(),
        reason="tree-sitter Python parser not available",
    )
    async def test_ingest_full_repository(self, pipeline_with_config, temp_git_repo):
        """Test full repository ingestion."""
        result = await pipeline_with_config.ingest_full(temp_git_repo)

        assert result.repo_id is not None
        assert result.commit_sha is not None
        assert result.files_processed > 0
        assert result.entities_extracted > 0
        assert result.is_incremental is False
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_ingest_with_progress_callback(self, pipeline_with_config, temp_git_repo):
        """Test ingestion with progress callback."""
        progress_calls = []

        def progress_callback(current: int, total: int, file_path: str):
            progress_calls.append((current, total, file_path))

        await pipeline_with_config.ingest_full(temp_git_repo, progress_callback=progress_callback)

        # Progress callback should have been called
        assert len(progress_calls) > 0

        # Last call should have current == total
        if progress_calls:
            last_call = progress_calls[-1]
            assert last_call[0] == last_call[1]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not _tree_sitter_available(),
        reason="tree-sitter Python parser not available",
    )
    async def test_ingest_incremental(self, pipeline_with_config, temp_git_repo_with_history):
        """Test incremental ingestion."""
        repo_path, first_commit, second_commit = temp_git_repo_with_history

        result = await pipeline_with_config.ingest_incremental(repo_path, since_commit=first_commit)

        assert result.is_incremental is True
        assert result.commit_sha == second_commit
        # Should have processed fewer files than full ingestion
        assert result.files_processed >= 1  # At least the modified and new files

    @pytest.mark.asyncio
    async def test_ingest_incremental_no_changes(self, pipeline_with_config, temp_git_repo):
        """Test incremental ingestion when no changes exist."""
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=temp_git_repo,
            capture_output=True,
            check=True,
            text=True,
        )
        commit_sha = result.stdout.strip()

        result = await pipeline_with_config.ingest_incremental(
            temp_git_repo, since_commit=commit_sha
        )

        assert result.is_incremental is True
        assert result.files_processed == 0
        assert result.entities_extracted == 0

    @pytest.mark.asyncio
    async def test_get_repository_info(self, pipeline, temp_git_repo):
        """Test getting repository info via pipeline."""
        repo_info = await pipeline.get_repository_info(temp_git_repo)

        assert repo_info.name == "test_repo"
        assert repo_info.local_path == str(temp_git_repo.resolve())

    @pytest.mark.asyncio
    async def test_estimate_ingestion_size(self, pipeline_with_config, temp_git_repo):
        """Test estimating ingestion size."""
        estimate = await pipeline_with_config.estimate_ingestion_size(temp_git_repo)

        assert "total_files" in estimate
        assert "total_size_bytes" in estimate
        assert estimate["total_files"] > 0
        assert estimate["total_size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_validate_repository_valid(self, pipeline, temp_git_repo):
        """Test validating a valid repository."""
        errors = await pipeline.validate_repository(temp_git_repo)
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_repository_not_git(self, pipeline):
        """Test validating a non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            errors = await pipeline.validate_repository(tmpdir)
            assert any("not a Git repository" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_repository_nonexistent(self, pipeline):
        """Test validating a nonexistent path."""
        errors = await pipeline.validate_repository("/nonexistent/path")
        assert any("does not exist" in e for e in errors)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not _tree_sitter_available(),
        reason="tree-sitter Python parser not available",
    )
    async def test_error_handling_in_batch(self, pipeline_with_config, temp_git_repo):
        """Test that errors in one file don't stop the pipeline."""
        # Create a file with syntax errors
        bad_file = temp_git_repo / "src" / "bad_syntax.py"
        bad_file.write_text("def broken(\n    return 'oops")

        # Should still complete and report errors
        result = await pipeline_with_config.ingest_full(temp_git_repo)

        # Pipeline should complete
        assert result.files_processed > 0
        # May or may not have errors depending on parser behavior


# =============================================================================
# Fixtures from conftest.py (duplicated for standalone testing)
# =============================================================================


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository with sample files."""
    import gc
    import shutil
    import subprocess
    import sys
    import time

    tmpdir = tempfile.mkdtemp()
    repo_path = Path(tmpdir) / "test_repo"
    repo_path.mkdir()

    subprocess.run(["git", "init"], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    gitignore = repo_path / ".gitignore"
    gitignore.write_text("*.pyc\n__pycache__/\n.env\nnode_modules/\n")

    src_dir = repo_path / "src"
    src_dir.mkdir()

    python_file = src_dir / "calculator.py"
    python_file.write_text(
        '"""A simple calculator module."""\n\nVERSION = "1.0.0"\n\n\n'
        "class Calculator:\n"
        '    """A basic calculator class."""\n\n'
        "    def __init__(self, initial: float = 0.0):\n"
        '        """Initialize with an optional starting value."""\n'
        "        self.value = initial\n\n"
        "    def add(self, x: float) -> float:\n"
        '        """Add a number to the current value."""\n'
        "        self.value += x\n"
        "        return self.value\n\n"
        "    def subtract(self, x: float) -> float:\n"
        '        """Subtract a number from the current value."""\n'
        "        self.value -= x\n"
        "        return self.value\n\n\n"
        "def create_calculator(initial: float = 0.0) -> Calculator:\n"
        '    """Factory function to create a Calculator."""\n'
        "    return Calculator(initial)\n"
    )

    utils_file = src_dir / "utils.py"
    utils_file.write_text(
        '"""Utility functions."""\n\n'
        "from typing import List\n\n\n"
        "def sum_list(numbers: List[int]) -> int:\n"
        '    """Sum all numbers in a list."""\n'
        "    return sum(numbers)\n\n\n"
        "def multiply_list(numbers: List[int]) -> int:\n"
        '    """Multiply all numbers in a list."""\n'
        "    result = 1\n"
        "    for n in numbers:\n"
        "        result *= n\n"
        "    return result\n"
    )

    js_file = src_dir / "helper.js"
    js_file.write_text(
        "// Helper functions\n\n"
        "function greet(name) {\n"
        "    return `Hello, ${name}!`;\n"
        "}\n\n"
        "const DEFAULT_TIMEOUT = 5000;\n\n"
        "module.exports = { greet, DEFAULT_TIMEOUT };\n"
    )

    ignored_dir = repo_path / "__pycache__"
    ignored_dir.mkdir()
    cache_file = ignored_dir / "calculator.cpython-311.pyc"
    cache_file.write_bytes(b"\x00\x00\x00\x00")

    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    yield repo_path

    # Cleanup: Force garbage collection to release git handles on Windows
    gc.collect()
    time.sleep(0.1)  # Brief delay for Windows to release file handles

    # Try to remove the directory, with retries on Windows
    for attempt in range(3):
        try:
            shutil.rmtree(tmpdir, ignore_errors=False)
            break
        except PermissionError:
            if attempt < 2:
                gc.collect()
                time.sleep(0.5)
            else:
                # Last resort: ignore errors
                shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_git_repo_with_history(temp_git_repo):
    """Create a git repo with multiple commits for diff testing."""
    import subprocess

    repo_path = temp_git_repo

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        check=True,
        text=True,
    )
    first_commit = result.stdout.strip()

    src_dir = repo_path / "src"

    calc_file = src_dir / "calculator.py"
    content = calc_file.read_text()
    calc_file.write_text(content.replace('VERSION = "1.0.0"', 'VERSION = "1.1.0"'))

    new_file = src_dir / "new_module.py"
    new_file.write_text(
        '"""A new module added in second commit."""\n\n'
        "def new_function():\n"
        '    """A brand new function."""\n'
        "    pass\n"
    )

    js_file = src_dir / "helper.js"
    js_file.unlink()

    subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Second commit with changes"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        check=True,
        text=True,
    )
    second_commit = result.stdout.strip()

    yield repo_path, first_commit, second_commit


@pytest.fixture
def sample_python_files():
    """Create a temporary directory with multiple Python files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        files = []

        module1 = root / "module1.py"
        module1.write_text(
            '"""Module 1 docstring."""\n\ndef func1():\n    """Function 1."""\n    pass\n'
        )
        files.append(module1)

        pkg_dir = root / "package"
        pkg_dir.mkdir()

        init_file = pkg_dir / "__init__.py"
        init_file.write_text('"""Package init."""\n')
        files.append(init_file)

        module2 = pkg_dir / "module2.py"
        module2.write_text(
            '"""Module 2 docstring."""\n\n'
            "class MyClass:\n"
            '    """A sample class."""\n\n'
            "    def method1(self):\n"
            '        """Method 1."""\n'
            "        pass\n"
        )
        files.append(module2)

        nested_dir = root / "deep" / "nested" / "path"
        nested_dir.mkdir(parents=True)

        nested_file = nested_dir / "deep_module.py"
        nested_file.write_text('"""Deeply nested module."""\n\nCONSTANT = 42\n')
        files.append(nested_file)

        large_file = root / "large_file.py"
        large_file.write_text("# " + "x" * 600 * 1024)
        files.append(large_file)

        text_file = root / "readme.txt"
        text_file.write_text("This is a readme file.\n")
        files.append(text_file)

        yield root, files


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file with simple function code."""
    code = '''def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        f.flush()
        temp_path = Path(f.name)

    yield temp_path

    temp_path.unlink(missing_ok=True)
