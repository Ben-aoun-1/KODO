"""GitHub API client for Kodo.

This module provides an async client for interacting with the GitHub API,
including authentication via GitHub App or personal access token.
"""

import hashlib
import hmac
import time
from typing import Any

import httpx
import jwt
import structlog
from pydantic import BaseModel, ConfigDict, Field

from .models import (
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
)

logger = structlog.get_logger(__name__)


class GitHubClientConfig(BaseModel):
    """Configuration for GitHub client."""

    model_config = ConfigDict(frozen=True)

    # GitHub App authentication
    app_id: int | None = Field(None, description="GitHub App ID")
    private_key: str | None = Field(None, description="GitHub App private key (PEM)")
    installation_id: int | None = Field(None, description="Installation ID")

    # Personal access token authentication
    access_token: str | None = Field(None, description="Personal access token")

    # API settings
    base_url: str = Field(default="https://api.github.com", description="API base URL")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Webhook settings
    webhook_secret: str | None = Field(None, description="Webhook secret for validation")


class GitHubClient:
    """Async GitHub API client.

    Supports authentication via GitHub App or personal access token.
    Provides methods for repository, PR, and webhook operations.
    """

    def __init__(self, config: GitHubClientConfig) -> None:
        """Initialize the GitHub client.

        Args:
            config: Client configuration.
        """
        self.config = config
        self._logger = logger.bind(component="github_client")
        self._http_client: httpx.AsyncClient | None = None
        self._installation_token: str | None = None
        self._token_expires_at: float = 0

    async def __aenter__(self) -> "GitHubClient":
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={"Accept": "application/vnd.github+json"},
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _generate_jwt(self) -> str:
        """Generate a JWT for GitHub App authentication.

        Returns:
            JWT token string.
        """
        if not self.config.app_id or not self.config.private_key:
            raise ValueError("GitHub App credentials not configured")

        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60 seconds ago
            "exp": now + 600,  # Expires in 10 minutes
            "iss": self.config.app_id,
        }

        token: str = jwt.encode(payload, self.config.private_key, algorithm="RS256")
        return token

    async def _get_installation_token(self) -> str:
        """Get or refresh installation access token.

        Returns:
            Installation access token.
        """
        # Return cached token if still valid
        if self._installation_token and time.time() < self._token_expires_at - 60:
            return self._installation_token

        if not self.config.installation_id:
            raise ValueError("Installation ID not configured")

        client = await self._ensure_client()
        jwt_token = self._generate_jwt()

        response = await client.post(
            f"/app/installations/{self.config.installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {jwt_token}"},
        )
        response.raise_for_status()

        data = response.json()
        token: str = data["token"]
        self._installation_token = token
        # Token expires in 1 hour, cache expiry time
        self._token_expires_at = time.time() + 3600

        self._logger.debug("obtained_installation_token")
        return token

    async def _get_auth_header(self) -> dict[str, str]:
        """Get authorization header for API requests.

        Returns:
            Dict with Authorization header.
        """
        if self.config.access_token:
            return {"Authorization": f"Bearer {self.config.access_token}"}
        elif self.config.app_id:
            token = await self._get_installation_token()
            return {"Authorization": f"Bearer {token}"}
        else:
            return {}

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Make an authenticated API request.

        Args:
            method: HTTP method.
            path: API path.
            params: Query parameters.
            json_data: JSON body data.

        Returns:
            Response JSON data.
        """
        client = await self._ensure_client()
        headers = await self._get_auth_header()

        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json_data,
                    headers=headers,
                )

                if response.status_code == 401 and self.config.app_id:
                    # Token expired, refresh and retry
                    self._installation_token = None
                    headers = await self._get_auth_header()
                    continue

                response.raise_for_status()
                result: dict[str, Any] | list[Any] = response.json()
                return result

            except httpx.HTTPStatusError as e:
                if attempt < self.config.max_retries - 1 and e.response.status_code >= 500:
                    self._logger.warning(
                        "request_failed_retrying",
                        attempt=attempt + 1,
                        status=e.response.status_code,
                    )
                    continue
                raise

        return {}

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook payload signature.

        Args:
            payload: Raw request body.
            signature: X-Hub-Signature-256 header value.

        Returns:
            True if signature is valid.
        """
        if not self.config.webhook_secret:
            self._logger.warning("webhook_secret_not_configured")
            return False

        expected = hmac.new(
            self.config.webhook_secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        expected_signature = f"sha256={expected}"
        return hmac.compare_digest(expected_signature, signature)

    # Repository operations

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information.

        Args:
            owner: Repository owner.
            repo: Repository name.

        Returns:
            GitHubRepository object.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}")
        return self._parse_repository(data)  # type: ignore[arg-type]

    async def list_repositories(self) -> list[GitHubRepository]:
        """List repositories accessible to the authenticated user/installation.

        Returns:
            List of repositories.
        """
        if self.config.installation_id:
            data = await self._request("GET", "/installation/repositories")
            repos = data.get("repositories", []) if isinstance(data, dict) else []
        else:
            data = await self._request("GET", "/user/repos")
            repos = data if isinstance(data, list) else []

        return [self._parse_repository(r) for r in repos]

    # Pull request operations

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Get pull request details.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            GitHubPullRequest object.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
        return self._parse_pull_request(data)  # type: ignore[arg-type]

    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 30,
    ) -> list[GitHubPullRequest]:
        """List pull requests.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: PR state filter (open, closed, all).
            per_page: Results per page.

        Returns:
            List of pull requests.
        """
        data = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls",
            params={"state": state, "per_page": per_page},
        )
        return [self._parse_pull_request(pr) for pr in data]  # type: ignore[arg-type]

    async def get_pull_request_files(
        self, owner: str, repo: str, pr_number: int
    ) -> list[GitHubFile]:
        """Get files changed in a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            List of changed files.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/files")
        return [self._parse_file(f) for f in data]  # type: ignore[arg-type]

    async def get_pull_request_commits(
        self, owner: str, repo: str, pr_number: int
    ) -> list[GitHubCommit]:
        """Get commits in a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            List of commits.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/commits")
        return [self._parse_commit(c) for c in data]  # type: ignore[arg-type]

    # Review operations

    async def create_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
        comments: list[dict[str, Any]] | None = None,
    ) -> GitHubReview:
        """Create a pull request review.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            body: Review body.
            event: Review event (APPROVE, REQUEST_CHANGES, COMMENT).
            comments: List of review comments.

        Returns:
            Created review.
        """
        json_data: dict[str, Any] = {"body": body, "event": event}
        if comments:
            json_data["comments"] = comments

        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
            json_data=json_data,
        )
        return self._parse_review(data)  # type: ignore[arg-type]

    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        commit_id: str,
        path: str,
        line: int,
        side: str = "RIGHT",
    ) -> GitHubComment:
        """Create a review comment on a specific line.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            body: Comment body.
            commit_id: Commit SHA.
            path: File path.
            line: Line number.
            side: Side of diff (LEFT, RIGHT).

        Returns:
            Created comment.
        """
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            json_data={
                "body": body,
                "commit_id": commit_id,
                "path": path,
                "line": line,
                "side": side,
            },
        )
        return self._parse_comment(data)  # type: ignore[arg-type]

    async def list_reviews(self, owner: str, repo: str, pr_number: int) -> list[GitHubReview]:
        """List reviews on a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            List of reviews.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews")
        return [self._parse_review(r) for r in data]  # type: ignore[arg-type]

    # Commit operations

    async def get_commit(self, owner: str, repo: str, sha: str) -> GitHubCommit:
        """Get commit details.

        Args:
            owner: Repository owner.
            repo: Repository name.
            sha: Commit SHA.

        Returns:
            GitHubCommit object.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
        return self._parse_commit(data)  # type: ignore[arg-type]

    async def compare_commits(self, owner: str, repo: str, base: str, head: str) -> dict[str, Any]:
        """Compare two commits.

        Args:
            owner: Repository owner.
            repo: Repository name.
            base: Base commit/branch.
            head: Head commit/branch.

        Returns:
            Comparison data including files changed.
        """
        data = await self._request("GET", f"/repos/{owner}/{repo}/compare/{base}...{head}")
        return data  # type: ignore[return-value]

    # File content operations

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str | None = None
    ) -> str:
        """Get file content from repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path.
            ref: Git reference (branch, tag, sha).

        Returns:
            File content as string.
        """
        import base64

        params = {"ref": ref} if ref else None
        data = await self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)

        if isinstance(data, dict) and data.get("encoding") == "base64":
            content = data.get("content", "")
            return base64.b64decode(content).decode("utf-8")

        return ""

    # Comment operations

    async def create_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> GitHubComment:
        """Create a comment on an issue or PR.

        Args:
            owner: Repository owner.
            repo: Repository name.
            issue_number: Issue/PR number.
            body: Comment body.

        Returns:
            Created comment.
        """
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            json_data={"body": body},
        )
        return self._parse_comment(data)  # type: ignore[arg-type]

    # Parsing helpers

    def _parse_user(self, data: dict[str, Any] | None) -> GitHubUser | None:
        """Parse user data."""
        if not data:
            return None
        return GitHubUser(
            id=data["id"],
            login=data["login"],
            name=data.get("name"),
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
            html_url=data.get("html_url"),
        )

    def _parse_repository(self, data: dict[str, Any]) -> GitHubRepository:
        """Parse repository data."""
        owner = self._parse_user(data.get("owner"))
        if not owner:
            owner = GitHubUser(id=0, login="unknown")

        return GitHubRepository(
            id=data["id"],
            name=data["name"],
            full_name=data["full_name"],
            owner=owner,
            private=data.get("private", False),
            html_url=data["html_url"],
            clone_url=data.get("clone_url"),
            default_branch=data.get("default_branch", "main"),
            language=data.get("language"),
            description=data.get("description"),
        )

    def _parse_branch(self, data: dict[str, Any] | None) -> GitHubBranch | None:
        """Parse branch reference data."""
        if not data:
            return None
        return GitHubBranch(
            ref=data.get("ref", ""),
            sha=data.get("sha", ""),
            label=data.get("label"),
            user=self._parse_user(data.get("user")),
            repo=self._parse_repository(data["repo"]) if data.get("repo") else None,
        )

    def _parse_file(self, data: dict[str, Any]) -> GitHubFile:
        """Parse file data."""
        return GitHubFile(
            filename=data["filename"],
            status=FileStatus(data.get("status", "modified")),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changes=data.get("changes", 0),
            patch=data.get("patch"),
            blob_url=data.get("blob_url"),
            raw_url=data.get("raw_url"),
            contents_url=data.get("contents_url"),
            previous_filename=data.get("previous_filename"),
        )

    def _parse_commit(self, data: dict[str, Any]) -> GitHubCommit:
        """Parse commit data."""
        commit_data = data.get("commit", {})
        author_data = data.get("author") or commit_data.get("author", {})

        return GitHubCommit(
            sha=data["sha"],
            message=commit_data.get("message", data.get("message", "")),
            author=self._parse_user(author_data)
            if isinstance(author_data, dict) and author_data.get("id")
            else None,
            committer=self._parse_user(data.get("committer")),
            html_url=data.get("html_url"),
            files=[self._parse_file(f) for f in data.get("files", [])],
            parents=[p["sha"] for p in data.get("parents", [])],
        )

    def _parse_pull_request(self, data: dict[str, Any]) -> GitHubPullRequest:
        """Parse pull request data."""
        user = self._parse_user(data.get("user"))
        if not user:
            user = GitHubUser(id=0, login="unknown")

        head = self._parse_branch(data.get("head"))
        base = self._parse_branch(data.get("base"))

        if not head:
            head = GitHubBranch(ref="", sha="")
        if not base:
            base = GitHubBranch(ref="", sha="")

        state_str = data.get("state", "open")
        if data.get("merged"):
            state = PullRequestState.MERGED
        elif state_str == "closed":
            state = PullRequestState.CLOSED
        else:
            state = PullRequestState.OPEN

        return GitHubPullRequest(
            id=data["id"],
            number=data["number"],
            title=data["title"],
            body=data.get("body"),
            state=state,
            user=user,
            html_url=data["html_url"],
            diff_url=data.get("diff_url"),
            patch_url=data.get("patch_url"),
            head=head,
            base=base,
            merged=data.get("merged", False),
            mergeable=data.get("mergeable"),
            merged_by=self._parse_user(data.get("merged_by")),
            commits=data.get("commits", 0),
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changed_files=data.get("changed_files", 0),
            labels=[label["name"] for label in data.get("labels", [])],
        )

    def _parse_comment(self, data: dict[str, Any]) -> GitHubComment:
        """Parse comment data."""
        user = self._parse_user(data.get("user"))
        if not user:
            user = GitHubUser(id=0, login="unknown")

        return GitHubComment(
            id=data["id"],
            body=data.get("body", ""),
            user=user,
            html_url=data.get("html_url"),
            path=data.get("path"),
            line=data.get("line"),
            side=data.get("side"),
            commit_id=data.get("commit_id"),
            in_reply_to_id=data.get("in_reply_to_id"),
        )

    def _parse_review(self, data: dict[str, Any]) -> GitHubReview:
        """Parse review data."""
        user = self._parse_user(data.get("user"))
        if not user:
            user = GitHubUser(id=0, login="unknown")

        return GitHubReview(
            id=data["id"],
            user=user,
            body=data.get("body"),
            state=ReviewState(data.get("state", "COMMENTED")),
            html_url=data.get("html_url"),
            commit_id=data.get("commit_id"),
        )
