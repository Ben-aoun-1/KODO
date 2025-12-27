"""Tests for GitHub integration module."""

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock

import pytest

from integrations.github.client import GitHubClient, GitHubClientConfig
from integrations.github.models import (
    DiffHunk,
    FileDiff,
    FileStatus,
    GitHubBranch,
    GitHubComment,
    GitHubCommit,
    GitHubFile,
    GitHubPullRequest,
    GitHubRepository,
    GitHubReview,
    GitHubUser,
    PullRequestState,
    ReviewState,
    WebhookEvent,
    WebhookPayload,
)
from integrations.github.pr_review import (
    DiffParser,
    PRReviewer,
    ReviewComment,
    ReviewResult,
    ReviewSeverity,
)
from integrations.github.webhooks import (
    InstallationEventHandler,
    PullRequestEventHandler,
    PushEventHandler,
    WebhookProcessor,
    WebhookResult,
)

# =============================================================================
# Model Tests
# =============================================================================


class TestGitHubModels:
    """Tests for GitHub Pydantic models."""

    def test_github_user(self):
        """Test GitHubUser model."""
        user = GitHubUser(
            id=12345,
            login="testuser",
            name="Test User",
            email="test@example.com",
        )
        assert user.id == 12345
        assert user.login == "testuser"
        assert user.name == "Test User"

    def test_github_repository(self):
        """Test GitHubRepository model."""
        owner = GitHubUser(id=1, login="owner")
        repo = GitHubRepository(
            id=67890,
            name="test-repo",
            full_name="owner/test-repo",
            owner=owner,
            private=False,
            html_url="https://github.com/owner/test-repo",
            default_branch="main",
        )
        assert repo.id == 67890
        assert repo.full_name == "owner/test-repo"
        assert repo.default_branch == "main"

    def test_github_pull_request(self):
        """Test GitHubPullRequest model."""
        user = GitHubUser(id=1, login="author")
        head = GitHubBranch(ref="feature", sha="abc123")
        base = GitHubBranch(ref="main", sha="def456")

        pr = GitHubPullRequest(
            id=111,
            number=42,
            title="Add feature",
            body="This PR adds a new feature",
            state=PullRequestState.OPEN,
            user=user,
            html_url="https://github.com/owner/repo/pull/42",
            head=head,
            base=base,
        )
        assert pr.number == 42
        assert pr.state == PullRequestState.OPEN
        assert pr.head.ref == "feature"
        assert pr.base.ref == "main"

    def test_github_file(self):
        """Test GitHubFile model."""
        file = GitHubFile(
            filename="src/main.py",
            status=FileStatus.MODIFIED,
            additions=10,
            deletions=5,
            changes=15,
            patch="@@ -1,5 +1,10 @@\n+added line",
        )
        assert file.filename == "src/main.py"
        assert file.status == FileStatus.MODIFIED
        assert file.additions == 10

    def test_github_commit(self):
        """Test GitHubCommit model."""
        commit = GitHubCommit(
            sha="abc123def456",
            message="Fix bug in parser",
        )
        assert commit.sha == "abc123def456"
        assert "parser" in commit.message

    def test_github_review(self):
        """Test GitHubReview model."""
        user = GitHubUser(id=1, login="reviewer")
        review = GitHubReview(
            id=999,
            user=user,
            body="LGTM",
            state=ReviewState.APPROVED,
        )
        assert review.state == ReviewState.APPROVED

    def test_webhook_event_enum(self):
        """Test WebhookEvent enum values."""
        assert WebhookEvent.PUSH.value == "push"
        assert WebhookEvent.PULL_REQUEST.value == "pull_request"
        assert WebhookEvent.INSTALLATION.value == "installation"

    def test_file_status_enum(self):
        """Test FileStatus enum values."""
        assert FileStatus.ADDED.value == "added"
        assert FileStatus.MODIFIED.value == "modified"
        assert FileStatus.REMOVED.value == "removed"
        assert FileStatus.RENAMED.value == "renamed"

    def test_diff_hunk(self):
        """Test DiffHunk model."""
        hunk = DiffHunk(
            old_start=1,
            old_count=5,
            new_start=1,
            new_count=7,
            header="def function():",
            lines=[" context", "+added", "-removed"],
        )
        assert hunk.old_start == 1
        assert hunk.new_count == 7
        assert len(hunk.lines) == 3

    def test_file_diff(self):
        """Test FileDiff model."""
        hunk = DiffHunk(
            old_start=1,
            old_count=1,
            new_start=1,
            new_count=2,
            header="",
            lines=["+added"],
        )
        diff = FileDiff(
            filename="test.py",
            status=FileStatus.MODIFIED,
            hunks=[hunk],
            additions=1,
            deletions=0,
        )
        assert diff.filename == "test.py"
        assert len(diff.hunks) == 1


# =============================================================================
# Client Tests
# =============================================================================


class TestGitHubClientConfig:
    """Tests for GitHubClientConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GitHubClientConfig()
        assert config.base_url == "https://api.github.com"
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_app_auth_config(self):
        """Test GitHub App configuration."""
        config = GitHubClientConfig(
            app_id=12345,
            private_key="test-private-key-pem-content",
            installation_id=67890,
        )
        assert config.app_id == 12345
        assert config.installation_id == 67890

    def test_token_auth_config(self):
        """Test token authentication configuration."""
        config = GitHubClientConfig(access_token="ghp_xxxx")
        assert config.access_token == "ghp_xxxx"


class TestGitHubClient:
    """Tests for GitHubClient."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        config = GitHubClientConfig(access_token="test-token")
        return GitHubClient(config)

    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.config.access_token == "test-token"
        assert client._http_client is None

    def test_verify_webhook_signature_valid(self, client):
        """Test valid webhook signature verification."""
        client.config = GitHubClientConfig(
            access_token="test-token",
            webhook_secret="secret123",
        )
        payload = b'{"action": "opened"}'
        expected = hmac.new(b"secret123", payload, hashlib.sha256).hexdigest()
        signature = f"sha256={expected}"

        assert client.verify_webhook_signature(payload, signature) is True

    def test_verify_webhook_signature_invalid(self, client):
        """Test invalid webhook signature."""
        client.config = GitHubClientConfig(
            access_token="test-token",
            webhook_secret="secret123",
        )
        payload = b'{"action": "opened"}'

        assert client.verify_webhook_signature(payload, "sha256=invalid") is False

    def test_verify_webhook_no_secret(self, client):
        """Test webhook verification without secret configured."""
        result = client.verify_webhook_signature(b"payload", "sha256=abc")
        assert result is False

    def test_parse_user(self, client):
        """Test user parsing."""
        data = {
            "id": 123,
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
        }
        user = client._parse_user(data)
        assert user is not None
        assert user.id == 123
        assert user.login == "testuser"

    def test_parse_user_none(self, client):
        """Test parsing None user."""
        assert client._parse_user(None) is None

    def test_parse_file(self, client):
        """Test file parsing."""
        data = {
            "filename": "src/test.py",
            "status": "modified",
            "additions": 5,
            "deletions": 2,
            "changes": 7,
            "patch": "@@ -1 +1,5 @@\n+new line",
        }
        file = client._parse_file(data)
        assert file.filename == "src/test.py"
        assert file.status == FileStatus.MODIFIED
        assert file.additions == 5

    def test_parse_commit(self, client):
        """Test commit parsing."""
        data = {
            "sha": "abc123",
            "commit": {"message": "Fix bug"},
            "files": [],
            "parents": [{"sha": "parent1"}],
        }
        commit = client._parse_commit(data)
        assert commit.sha == "abc123"
        assert commit.message == "Fix bug"
        assert "parent1" in commit.parents


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhookHandlers:
    """Tests for webhook handlers."""

    @pytest.mark.asyncio
    async def test_push_handler_can_handle(self):
        """Test push handler event matching."""
        handler = PushEventHandler()
        assert await handler.can_handle(WebhookEvent.PUSH, None) is True
        assert await handler.can_handle(WebhookEvent.PULL_REQUEST, None) is False

    @pytest.mark.asyncio
    async def test_push_handler_handle_no_repo(self):
        """Test push handler with missing repository."""
        handler = PushEventHandler()
        payload = WebhookPayload()
        client = MagicMock()

        result = await handler.handle(payload, client)
        assert result.success is False
        assert "No repository" in result.message

    @pytest.mark.asyncio
    async def test_push_handler_handle_success(self):
        """Test successful push handling."""
        callback = AsyncMock()
        handler = PushEventHandler(indexer_callback=callback)

        owner = GitHubUser(id=1, login="owner")
        repo = GitHubRepository(
            id=123,
            name="test",
            full_name="owner/test",
            owner=owner,
            html_url="https://github.com/owner/test",
        )
        commit = GitHubCommit(
            sha="abc123",
            message="Update",
            files=[GitHubFile(filename="test.py", status=FileStatus.MODIFIED)],
        )
        payload = WebhookPayload(
            repository=repo,
            ref="refs/heads/main",
            commits=[commit],
        )
        client = MagicMock()

        result = await handler.handle(payload, client)
        assert result.success is True
        assert "test.py" in result.data["changed_files"]
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_pr_handler_can_handle(self):
        """Test PR handler event matching."""
        handler = PullRequestEventHandler()
        assert await handler.can_handle(WebhookEvent.PULL_REQUEST, "opened") is True
        assert await handler.can_handle(WebhookEvent.PULL_REQUEST, "synchronize") is True
        assert await handler.can_handle(WebhookEvent.PULL_REQUEST, "closed") is False
        assert await handler.can_handle(WebhookEvent.PUSH, "opened") is False

    @pytest.mark.asyncio
    async def test_pr_handler_handle_success(self):
        """Test successful PR handling."""
        callback = AsyncMock()
        handler = PullRequestEventHandler(reviewer_callback=callback)

        owner = GitHubUser(id=1, login="owner")
        repo = GitHubRepository(
            id=123,
            name="test",
            full_name="owner/test",
            owner=owner,
            html_url="https://github.com/owner/test",
        )
        user = GitHubUser(id=2, login="author")
        head = GitHubBranch(ref="feature", sha="abc")
        base = GitHubBranch(ref="main", sha="def")
        pr = GitHubPullRequest(
            id=456,
            number=42,
            title="Feature",
            state=PullRequestState.OPEN,
            user=user,
            html_url="https://github.com/owner/test/pull/42",
            head=head,
            base=base,
        )
        payload = WebhookPayload(
            action="opened",
            repository=repo,
            pull_request=pr,
        )
        client = MagicMock()

        result = await handler.handle(payload, client)
        assert result.success is True
        assert result.data["pr_number"] == 42
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_installation_handler_can_handle(self):
        """Test installation handler event matching."""
        handler = InstallationEventHandler()
        assert await handler.can_handle(WebhookEvent.INSTALLATION, "created") is True
        assert await handler.can_handle(WebhookEvent.INSTALLATION_REPOSITORIES, "added") is True
        assert await handler.can_handle(WebhookEvent.PUSH, None) is False


class TestWebhookProcessor:
    """Tests for WebhookProcessor."""

    def test_register_handler(self):
        """Test handler registration."""
        processor = WebhookProcessor()
        handler = PushEventHandler()
        processor.register_handler(handler)
        assert handler in processor._handlers

    def test_register_defaults(self):
        """Test registering default handlers."""
        processor = WebhookProcessor()
        processor.register_defaults()
        assert len(processor._handlers) == 3

    def test_parse_payload_push(self):
        """Test parsing push event payload."""
        processor = WebhookProcessor()
        raw = {
            "ref": "refs/heads/main",
            "before": "aaa",
            "after": "bbb",
            "repository": {
                "id": 123,
                "name": "test",
                "full_name": "owner/test",
                "owner": {"id": 1, "login": "owner"},
                "html_url": "https://github.com/owner/test",
            },
            "sender": {"id": 1, "login": "owner"},
        }

        event, payload = processor.parse_payload("push", raw)
        assert event == WebhookEvent.PUSH
        assert payload.ref == "refs/heads/main"
        assert payload.repository is not None
        assert payload.repository.name == "test"

    def test_parse_payload_pull_request(self):
        """Test parsing PR event payload."""
        processor = WebhookProcessor()
        raw = {
            "action": "opened",
            "repository": {
                "id": 123,
                "name": "test",
                "full_name": "owner/test",
                "owner": {"id": 1, "login": "owner"},
                "html_url": "https://github.com/owner/test",
            },
            "pull_request": {
                "id": 456,
                "number": 42,
                "title": "Test PR",
                "state": "open",
                "user": {"id": 2, "login": "author"},
                "html_url": "https://github.com/owner/test/pull/42",
                "head": {"ref": "feature", "sha": "abc"},
                "base": {"ref": "main", "sha": "def"},
            },
        }

        event, payload = processor.parse_payload("pull_request", raw)
        assert event == WebhookEvent.PULL_REQUEST
        assert payload.action == "opened"
        assert payload.pull_request is not None
        assert payload.pull_request.number == 42

    @pytest.mark.asyncio
    async def test_process_with_handler(self):
        """Test processing with matching handler."""
        processor = WebhookProcessor()
        processor.register_defaults()

        raw = {
            "ref": "refs/heads/main",
            "repository": {
                "id": 123,
                "name": "test",
                "full_name": "owner/test",
                "owner": {"id": 1, "login": "owner"},
                "html_url": "https://github.com/owner/test",
            },
            "commits": [],
        }
        client = MagicMock()

        result = await processor.process("push", raw, client)
        assert result.success is True
        assert result.event_type == WebhookEvent.PUSH


# =============================================================================
# Diff Parser Tests
# =============================================================================


class TestDiffParser:
    """Tests for DiffParser."""

    @pytest.fixture
    def parser(self):
        """Create a diff parser."""
        return DiffParser()

    def test_parse_simple_patch(self, parser):
        """Test parsing a simple patch."""
        patch = """@@ -1,3 +1,4 @@
 line1
+added line
 line2
 line3"""
        diff = parser.parse_patch(patch, "test.py")
        assert diff.filename == "test.py"
        assert len(diff.hunks) == 1
        assert diff.hunks[0].old_start == 1
        assert diff.hunks[0].new_start == 1
        assert diff.additions == 1

    def test_parse_multiple_hunks(self, parser):
        """Test parsing patch with multiple hunks."""
        patch = """@@ -1,3 +1,4 @@
 line1
+added
 line2
@@ -10,2 +11,3 @@
 line10
+another
 line11"""
        diff = parser.parse_patch(patch, "test.py")
        assert len(diff.hunks) == 2
        assert diff.hunks[0].old_start == 1
        assert diff.hunks[1].old_start == 10

    def test_parse_empty_patch(self, parser):
        """Test parsing empty patch."""
        diff = parser.parse_patch("", "test.py")
        assert len(diff.hunks) == 0
        assert diff.additions == 0
        assert diff.deletions == 0

    def test_parse_deletion(self, parser):
        """Test parsing deletions."""
        patch = """@@ -1,3 +1,2 @@
 line1
-removed line
 line3"""
        diff = parser.parse_patch(patch, "test.py")
        assert diff.deletions == 1
        assert diff.additions == 0

    def test_get_new_line_number(self, parser):
        """Test getting new line numbers."""
        hunk = DiffHunk(
            old_start=1,
            old_count=3,
            new_start=1,
            new_count=4,
            header="",
            lines=[" context", "+added", " more context"],
        )

        # Context line
        assert parser.get_new_line_number(hunk, 0) == 1
        # Added line
        assert parser.get_new_line_number(hunk, 1) == 2
        # Context after added
        assert parser.get_new_line_number(hunk, 2) == 3

    def test_get_new_line_number_deleted(self, parser):
        """Test line number for deleted line."""
        hunk = DiffHunk(
            old_start=1,
            old_count=3,
            new_start=1,
            new_count=2,
            header="",
            lines=[" context", "-deleted"],
        )

        assert parser.get_new_line_number(hunk, 1) is None


# =============================================================================
# PR Reviewer Tests
# =============================================================================


class TestReviewModels:
    """Tests for review models."""

    def test_review_comment(self):
        """Test ReviewComment model."""
        comment = ReviewComment(
            file_path="test.py",
            line=42,
            body="Consider using a constant here",
            severity=ReviewSeverity.SUGGESTION,
        )
        assert comment.file_path == "test.py"
        assert comment.line == 42
        assert comment.severity == ReviewSeverity.SUGGESTION

    def test_review_result(self):
        """Test ReviewResult model."""
        result = ReviewResult(
            pr_number=42,
            repo_full_name="owner/repo",
            summary="All good",
            comments=[],
            approval_status="APPROVE",
        )
        assert result.pr_number == 42
        assert result.approval_status == "APPROVE"


class TestPRReviewer:
    """Tests for PRReviewer."""

    @pytest.fixture
    def reviewer(self):
        """Create a PR reviewer."""
        return PRReviewer()

    def test_detect_language(self, reviewer):
        """Test language detection."""
        assert reviewer._detect_language("test.py") == "python"
        assert reviewer._detect_language("test.js") == "javascript"
        assert reviewer._detect_language("test.ts") == "typescript"
        assert reviewer._detect_language("test.go") == "go"
        assert reviewer._detect_language("test.txt") is None

    def test_analyze_line_eval(self, reviewer):
        """Test detecting eval() usage."""
        comments = reviewer._analyze_line("+result = eval(user_input)", 10, "test.py")
        assert len(comments) == 1
        assert "security" in comments[0].body.lower()
        assert comments[0].severity == ReviewSeverity.ERROR

    def test_analyze_line_todo(self, reviewer):
        """Test detecting TODO comments."""
        comments = reviewer._analyze_line("+# TODO: fix this later", 5, "test.py")
        assert len(comments) == 1
        assert "todo" in comments[0].body.lower()
        assert comments[0].severity == ReviewSeverity.INFO

    def test_analyze_line_print(self, reviewer):
        """Test detecting debug prints."""
        comments = reviewer._analyze_line("+print('debug info')", 15, "test.py")
        assert len(comments) == 1
        assert "debug" in comments[0].body.lower()

    def test_analyze_line_bare_except(self, reviewer):
        """Test detecting bare except."""
        comments = reviewer._analyze_line("+except:", 20, "test.py")
        assert len(comments) == 1
        assert "except" in comments[0].body.lower()

    def test_analyze_line_wildcard_import(self, reviewer):
        """Test detecting wildcard imports."""
        comments = reviewer._analyze_line("+from os import *", 1, "test.py")
        assert len(comments) == 1
        assert "wildcard" in comments[0].body.lower()

    def test_analyze_line_deleted_line(self, reviewer):
        """Test skipping deleted lines."""
        comments = reviewer._analyze_line("-eval(x)", 10, "test.py")
        assert len(comments) == 0

    def test_analyze_line_no_issues(self, reviewer):
        """Test line with no issues."""
        comments = reviewer._analyze_line("+x = 5 + 3", 10, "test.py")
        assert len(comments) == 0

    def test_analyze_diff(self, reviewer):
        """Test analyzing a file diff."""
        file = GitHubFile(
            filename="test.py",
            status=FileStatus.MODIFIED,
            patch="""@@ -1,3 +1,5 @@
 def func():
+    print('debug')
+    eval(x)
     return 1""",
        )

        diff, comments = reviewer.analyze_diff(file)
        assert len(diff.hunks) == 1
        # Should find print and eval issues
        assert len(comments) >= 2

    def test_generate_summary_no_issues(self, reviewer):
        """Test summary generation with no issues."""
        user = GitHubUser(id=1, login="author")
        head = GitHubBranch(ref="feature", sha="abc")
        base = GitHubBranch(ref="main", sha="def")
        pr = GitHubPullRequest(
            id=1,
            number=42,
            title="Test",
            state=PullRequestState.OPEN,
            user=user,
            html_url="https://github.com/owner/repo/pull/42",
            head=head,
            base=base,
        )

        summary = reviewer._generate_summary(
            pr=pr,
            files_analyzed=5,
            total_additions=100,
            total_deletions=50,
            comments=[],
        )

        assert "5" in summary  # Files analyzed
        assert "+100" in summary
        assert "-50" in summary
        assert "Good job" in summary

    def test_generate_summary_with_issues(self, reviewer):
        """Test summary generation with issues."""
        user = GitHubUser(id=1, login="author")
        head = GitHubBranch(ref="feature", sha="abc")
        base = GitHubBranch(ref="main", sha="def")
        pr = GitHubPullRequest(
            id=1,
            number=42,
            title="Test",
            state=PullRequestState.OPEN,
            user=user,
            html_url="https://github.com/owner/repo/pull/42",
            head=head,
            base=base,
        )

        comments = [
            ReviewComment(
                file_path="test.py",
                line=1,
                body="Error",
                severity=ReviewSeverity.ERROR,
            ),
            ReviewComment(
                file_path="test.py",
                line=2,
                body="Warning",
                severity=ReviewSeverity.WARNING,
            ),
        ]

        summary = reviewer._generate_summary(
            pr=pr,
            files_analyzed=3,
            total_additions=20,
            total_deletions=10,
            comments=comments,
        )

        assert "1 error" in summary
        assert "1 warning" in summary

    @pytest.mark.asyncio
    async def test_review_pr(self, reviewer):
        """Test full PR review."""
        client = AsyncMock()

        user = GitHubUser(id=1, login="author")
        head = GitHubBranch(ref="feature", sha="abc123")
        base = GitHubBranch(ref="main", sha="def456")
        pr = GitHubPullRequest(
            id=1,
            number=42,
            title="Add feature",
            state=PullRequestState.OPEN,
            user=user,
            html_url="https://github.com/owner/repo/pull/42",
            head=head,
            base=base,
        )

        files = [
            GitHubFile(
                filename="main.py",
                status=FileStatus.MODIFIED,
                additions=5,
                deletions=2,
                patch="""@@ -1,3 +1,5 @@
 def main():
+    print('debug')
     return 0""",
            )
        ]

        client.get_pull_request.return_value = pr
        client.get_pull_request_files.return_value = files

        result = await reviewer.review_pr(client, "owner", "repo", 42)

        assert result.pr_number == 42
        assert result.files_analyzed == 1
        assert len(result.comments) >= 1  # At least the print statement


# =============================================================================
# Integration Tests
# =============================================================================


class TestGitHubIntegration:
    """Integration tests for GitHub module."""

    def test_full_webhook_flow(self):
        """Test complete webhook processing flow."""
        processor = WebhookProcessor()

        callback_called = False

        async def mock_callback(**kwargs):
            nonlocal callback_called
            callback_called = True

        processor.register_defaults(indexer_callback=mock_callback)

        # Create a push event payload
        raw_payload = {
            "ref": "refs/heads/main",
            "before": "aaa",
            "after": "bbb",
            "repository": {
                "id": 123,
                "name": "test-repo",
                "full_name": "owner/test-repo",
                "owner": {"id": 1, "login": "owner"},
                "html_url": "https://github.com/owner/test-repo",
                "private": False,
            },
            "sender": {"id": 1, "login": "owner"},
            "commits": [
                {
                    "id": "commit1",
                    "message": "Update file",
                    "added": ["new.py"],
                    "modified": ["existing.py"],
                    "removed": [],
                }
            ],
        }

        event, payload = processor.parse_payload("push", raw_payload)

        assert event == WebhookEvent.PUSH
        assert payload.repository is not None
        assert payload.ref == "refs/heads/main"
        assert payload.commits is not None
        assert len(payload.commits) == 1

    def test_review_severity_levels(self):
        """Test review comment severity levels."""
        assert ReviewSeverity.ERROR.value == "error"
        assert ReviewSeverity.WARNING.value == "warning"
        assert ReviewSeverity.SUGGESTION.value == "suggestion"
        assert ReviewSeverity.INFO.value == "info"

    def test_file_status_mapping(self):
        """Test file status mapping."""
        reviewer = PRReviewer()

        # Test various file types
        file = GitHubFile(
            filename="test.py",
            status=FileStatus.ADDED,
            patch="+new content",
        )
        diff, _ = reviewer.analyze_diff(file)
        assert diff.status == FileStatus.MODIFIED  # Parser defaults to MODIFIED
