"""GitHub webhook handlers for Kodo.

This module provides handlers for processing GitHub webhook events
including push events, pull request events, and review events.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from .client import GitHubClient
from .models import (
    FileStatus,
    GitHubCommit,
    GitHubFile,
    GitHubPullRequest,
    GitHubRepository,
    GitHubUser,
    WebhookEvent,
    WebhookPayload,
)

logger = structlog.get_logger(__name__)


class WebhookResult(BaseModel):
    """Result of processing a webhook."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(..., description="Whether processing succeeded")
    event_type: WebhookEvent = Field(..., description="Event type processed")
    action: str | None = Field(None, description="Event action")
    message: str = Field(default="", description="Result message")
    data: dict[str, Any] = Field(default_factory=dict, description="Result data")


class WebhookHandler(ABC):
    """Abstract base class for webhook handlers."""

    @abstractmethod
    async def can_handle(self, event_type: WebhookEvent, action: str | None) -> bool:
        """Check if this handler can process the event.

        Args:
            event_type: GitHub event type.
            action: Event action (e.g., 'opened', 'closed').

        Returns:
            True if handler can process this event.
        """

    @abstractmethod
    async def handle(
        self,
        payload: WebhookPayload,
        client: GitHubClient,
    ) -> WebhookResult:
        """Handle the webhook event.

        Args:
            payload: Parsed webhook payload.
            client: GitHub API client.

        Returns:
            WebhookResult with processing outcome.
        """


class PushEventHandler(WebhookHandler):
    """Handler for push events.

    Triggers repository re-indexing when code is pushed.
    """

    def __init__(self, indexer_callback: Any | None = None) -> None:
        """Initialize handler.

        Args:
            indexer_callback: Async callback to trigger indexing.
        """
        self._indexer_callback = indexer_callback
        self._logger = logger.bind(handler="push")

    async def can_handle(self, event_type: WebhookEvent, action: str | None) -> bool:
        """Check if this is a push event."""
        return event_type == WebhookEvent.PUSH

    async def handle(
        self,
        payload: WebhookPayload,
        client: GitHubClient,
    ) -> WebhookResult:
        """Handle push event.

        Args:
            payload: Push event payload.
            client: GitHub API client.

        Returns:
            WebhookResult.
        """
        repo = payload.repository
        if not repo:
            return WebhookResult(
                success=False,
                event_type=WebhookEvent.PUSH,
                message="No repository in payload",
            )

        ref = payload.ref or ""
        before = payload.before or ""
        after = payload.after or ""
        commits = payload.commits or []

        self._logger.info(
            "push_received",
            repo=repo.full_name,
            ref=ref,
            commits=len(commits),
        )

        # Get changed files from commits
        changed_files: set[str] = set()
        for commit in commits:
            for file in commit.files:
                changed_files.add(file.filename)

        # Trigger indexing if callback provided
        if self._indexer_callback:
            try:
                await self._indexer_callback(
                    repo_id=str(repo.id),
                    repo_name=repo.full_name,
                    ref=ref,
                    changed_files=list(changed_files),
                )
            except Exception as e:
                self._logger.error("indexer_callback_failed", error=str(e))
                return WebhookResult(
                    success=False,
                    event_type=WebhookEvent.PUSH,
                    message=f"Indexing failed: {e}",
                )

        return WebhookResult(
            success=True,
            event_type=WebhookEvent.PUSH,
            message=f"Processed push to {ref} with {len(commits)} commits",
            data={
                "repo": repo.full_name,
                "ref": ref,
                "before": before,
                "after": after,
                "changed_files": list(changed_files),
            },
        )


class PullRequestEventHandler(WebhookHandler):
    """Handler for pull request events.

    Triggers PR analysis and review when PRs are opened or updated.
    """

    def __init__(self, reviewer_callback: Any | None = None) -> None:
        """Initialize handler.

        Args:
            reviewer_callback: Async callback to trigger PR review.
        """
        self._reviewer_callback = reviewer_callback
        self._logger = logger.bind(handler="pull_request")

    async def can_handle(self, event_type: WebhookEvent, action: str | None) -> bool:
        """Check if this is a relevant PR event."""
        if event_type != WebhookEvent.PULL_REQUEST:
            return False
        # Handle opened, synchronize (push to PR), and reopened
        return action in ("opened", "synchronize", "reopened")

    async def handle(
        self,
        payload: WebhookPayload,
        client: GitHubClient,
    ) -> WebhookResult:
        """Handle pull request event.

        Args:
            payload: PR event payload.
            client: GitHub API client.

        Returns:
            WebhookResult.
        """
        pr = payload.pull_request
        repo = payload.repository
        action = payload.action

        if not pr or not repo:
            return WebhookResult(
                success=False,
                event_type=WebhookEvent.PULL_REQUEST,
                action=action,
                message="No PR or repository in payload",
            )

        self._logger.info(
            "pr_event_received",
            repo=repo.full_name,
            pr_number=pr.number,
            action=action,
        )

        # Trigger review if callback provided
        if self._reviewer_callback:
            try:
                await self._reviewer_callback(
                    repo_owner=repo.full_name.split("/")[0],
                    repo_name=repo.name,
                    pr_number=pr.number,
                )
            except Exception as e:
                self._logger.error("reviewer_callback_failed", error=str(e))
                return WebhookResult(
                    success=False,
                    event_type=WebhookEvent.PULL_REQUEST,
                    action=action,
                    message=f"Review failed: {e}",
                )

        return WebhookResult(
            success=True,
            event_type=WebhookEvent.PULL_REQUEST,
            action=action,
            message=f"Processed PR #{pr.number} action: {action}",
            data={
                "repo": repo.full_name,
                "pr_number": pr.number,
                "pr_title": pr.title,
                "action": action,
            },
        )


class InstallationEventHandler(WebhookHandler):
    """Handler for GitHub App installation events."""

    def __init__(self, install_callback: Any | None = None) -> None:
        """Initialize handler.

        Args:
            install_callback: Async callback for installation events.
        """
        self._install_callback = install_callback
        self._logger = logger.bind(handler="installation")

    async def can_handle(self, event_type: WebhookEvent, action: str | None) -> bool:
        """Check if this is an installation event."""
        return event_type in (
            WebhookEvent.INSTALLATION,
            WebhookEvent.INSTALLATION_REPOSITORIES,
        )

    async def handle(
        self,
        payload: WebhookPayload,
        client: GitHubClient,
    ) -> WebhookResult:
        """Handle installation event.

        Args:
            payload: Installation event payload.
            client: GitHub API client.

        Returns:
            WebhookResult.
        """
        action = payload.action
        installation = payload.installation

        if not installation:
            return WebhookResult(
                success=False,
                event_type=WebhookEvent.INSTALLATION,
                action=action,
                message="No installation in payload",
            )

        installation_id = installation.get("id")
        account = installation.get("account", {})

        self._logger.info(
            "installation_event",
            installation_id=installation_id,
            account=account.get("login"),
            action=action,
        )

        if self._install_callback:
            try:
                await self._install_callback(
                    installation_id=installation_id,
                    account_login=account.get("login"),
                    action=action,
                )
            except Exception as e:
                self._logger.error("install_callback_failed", error=str(e))

        return WebhookResult(
            success=True,
            event_type=WebhookEvent.INSTALLATION,
            action=action,
            message=f"Processed installation {action}",
            data={
                "installation_id": installation_id,
                "account": account.get("login"),
            },
        )


class WebhookProcessor:
    """Processes GitHub webhooks by routing to appropriate handlers."""

    def __init__(self) -> None:
        """Initialize the processor."""
        self._handlers: list[WebhookHandler] = []
        self._logger = logger.bind(component="webhook_processor")

    def register_handler(self, handler: WebhookHandler) -> None:
        """Register a webhook handler.

        Args:
            handler: Handler to register.
        """
        self._handlers.append(handler)

    def register_defaults(
        self,
        indexer_callback: Any | None = None,
        reviewer_callback: Any | None = None,
        install_callback: Any | None = None,
    ) -> None:
        """Register default handlers.

        Args:
            indexer_callback: Callback for push events.
            reviewer_callback: Callback for PR events.
            install_callback: Callback for installation events.
        """
        self.register_handler(PushEventHandler(indexer_callback))
        self.register_handler(PullRequestEventHandler(reviewer_callback))
        self.register_handler(InstallationEventHandler(install_callback))

    def parse_payload(
        self,
        event_type: str,
        raw_payload: dict[str, Any],
    ) -> tuple[WebhookEvent, WebhookPayload]:
        """Parse raw webhook payload into models.

        Args:
            event_type: X-GitHub-Event header value.
            raw_payload: Raw JSON payload.

        Returns:
            Tuple of (WebhookEvent, WebhookPayload).
        """
        try:
            event = WebhookEvent(event_type)
        except ValueError:
            # Unknown event type, default to push
            event = WebhookEvent.PUSH

        # Parse sender
        sender = None
        if raw_payload.get("sender"):
            sender_data = raw_payload["sender"]
            sender = GitHubUser(
                id=sender_data["id"],
                login=sender_data["login"],
                avatar_url=sender_data.get("avatar_url"),
            )

        # Parse repository
        repository = None
        if raw_payload.get("repository"):
            repo_data = raw_payload["repository"]
            owner_data = repo_data.get("owner", {})
            owner = GitHubUser(
                id=owner_data.get("id", 0),
                login=owner_data.get("login", "unknown"),
            )
            repository = GitHubRepository(
                id=repo_data["id"],
                name=repo_data["name"],
                full_name=repo_data["full_name"],
                owner=owner,
                private=repo_data.get("private", False),
                html_url=repo_data["html_url"],
                default_branch=repo_data.get("default_branch", "main"),
            )

        # Parse pull request if present
        pull_request = None
        if raw_payload.get("pull_request"):
            pr_data = raw_payload["pull_request"]
            pr_user = GitHubUser(
                id=pr_data["user"]["id"],
                login=pr_data["user"]["login"],
            )
            from .models import GitHubBranch, PullRequestState

            head = GitHubBranch(
                ref=pr_data["head"]["ref"],
                sha=pr_data["head"]["sha"],
            )
            base = GitHubBranch(
                ref=pr_data["base"]["ref"],
                sha=pr_data["base"]["sha"],
            )

            state_str = pr_data.get("state", "open")
            if pr_data.get("merged"):
                state = PullRequestState.MERGED
            elif state_str == "closed":
                state = PullRequestState.CLOSED
            else:
                state = PullRequestState.OPEN

            pull_request = GitHubPullRequest(
                id=pr_data["id"],
                number=pr_data["number"],
                title=pr_data["title"],
                body=pr_data.get("body"),
                state=state,
                user=pr_user,
                html_url=pr_data["html_url"],
                head=head,
                base=base,
            )

        # Parse commits for push events
        commits = None
        if raw_payload.get("commits"):
            commits = []
            for commit_data in raw_payload["commits"]:
                files = []
                for added in commit_data.get("added", []):
                    files.append(GitHubFile(filename=added, status=FileStatus.ADDED))
                for modified in commit_data.get("modified", []):
                    files.append(GitHubFile(filename=modified, status=FileStatus.MODIFIED))
                for removed in commit_data.get("removed", []):
                    files.append(GitHubFile(filename=removed, status=FileStatus.REMOVED))

                commits.append(
                    GitHubCommit(
                        sha=commit_data["id"],
                        message=commit_data["message"],
                        files=files,
                    )
                )

        payload = WebhookPayload(
            action=raw_payload.get("action"),
            sender=sender,
            repository=repository,
            installation=raw_payload.get("installation"),
            pull_request=pull_request,
            commits=commits,
            ref=raw_payload.get("ref"),
            before=raw_payload.get("before"),
            after=raw_payload.get("after"),
        )

        return event, payload

    async def process(
        self,
        event_type: str,
        raw_payload: dict[str, Any],
        client: GitHubClient,
    ) -> WebhookResult:
        """Process a webhook event.

        Args:
            event_type: X-GitHub-Event header value.
            raw_payload: Raw JSON payload.
            client: GitHub API client.

        Returns:
            WebhookResult from the handler.
        """
        event, payload = self.parse_payload(event_type, raw_payload)
        action = payload.action

        self._logger.info(
            "processing_webhook",
            event_type=event.value,
            action=action,
        )

        # Find and execute handler
        for handler in self._handlers:
            if await handler.can_handle(event, action):
                try:
                    return await handler.handle(payload, client)
                except Exception as e:
                    self._logger.error(
                        "handler_error",
                        event_type=event.value,
                        error=str(e),
                    )
                    return WebhookResult(
                        success=False,
                        event_type=event,
                        action=action,
                        message=f"Handler error: {e}",
                    )

        # No handler found
        self._logger.debug(
            "no_handler_for_event",
            event_type=event.value,
            action=action,
        )
        return WebhookResult(
            success=True,
            event_type=event,
            action=action,
            message="No handler registered for this event",
        )
