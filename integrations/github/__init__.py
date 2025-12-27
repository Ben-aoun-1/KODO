"""GitHub integration for Kodo.

This module provides GitHub App integration including:
- GitHub API client for repository operations
- Webhook handlers for push and PR events
- PR diff analysis and review comments
- Issue integration
"""

from .client import GitHubClient, GitHubClientConfig
from .models import (
    GitHubComment,
    GitHubCommit,
    GitHubFile,
    GitHubPullRequest,
    GitHubRepository,
    GitHubReview,
    WebhookEvent,
    WebhookPayload,
)
from .pr_review import PRReviewer, ReviewComment, ReviewResult
from .webhooks import WebhookHandler, WebhookProcessor

__all__ = [
    # Client
    "GitHubClient",
    "GitHubClientConfig",
    # Models
    "GitHubRepository",
    "GitHubPullRequest",
    "GitHubCommit",
    "GitHubFile",
    "GitHubReview",
    "GitHubComment",
    "WebhookEvent",
    "WebhookPayload",
    # PR Review
    "PRReviewer",
    "ReviewResult",
    "ReviewComment",
    # Webhooks
    "WebhookHandler",
    "WebhookProcessor",
]
