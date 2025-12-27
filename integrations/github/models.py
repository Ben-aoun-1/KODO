"""Pydantic models for GitHub integration.

This module defines data models for GitHub entities including
repositories, pull requests, commits, files, and webhooks.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WebhookEvent(str, Enum):
    """GitHub webhook event types."""

    PUSH = "push"
    PULL_REQUEST = "pull_request"
    PULL_REQUEST_REVIEW = "pull_request_review"
    PULL_REQUEST_REVIEW_COMMENT = "pull_request_review_comment"
    ISSUE_COMMENT = "issue_comment"
    ISSUES = "issues"
    CREATE = "create"
    DELETE = "delete"
    INSTALLATION = "installation"
    INSTALLATION_REPOSITORIES = "installation_repositories"


class PullRequestState(str, Enum):
    """Pull request states."""

    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class ReviewState(str, Enum):
    """Pull request review states."""

    PENDING = "PENDING"
    COMMENTED = "COMMENTED"
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    DISMISSED = "DISMISSED"


class FileStatus(str, Enum):
    """File change status in a commit or PR."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"
    COPIED = "copied"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


class GitHubUser(BaseModel):
    """GitHub user model."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="User ID")
    login: str = Field(..., description="Username")
    name: str | None = Field(None, description="Display name")
    email: str | None = Field(None, description="Email address")
    avatar_url: str | None = Field(None, description="Avatar URL")
    html_url: str | None = Field(None, description="Profile URL")


class GitHubRepository(BaseModel):
    """GitHub repository model."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="Repository ID")
    name: str = Field(..., description="Repository name")
    full_name: str = Field(..., description="Full name (owner/repo)")
    owner: GitHubUser = Field(..., description="Repository owner")
    private: bool = Field(default=False, description="Is private repository")
    html_url: str = Field(..., description="Repository URL")
    clone_url: str | None = Field(None, description="Clone URL")
    default_branch: str = Field(default="main", description="Default branch")
    language: str | None = Field(None, description="Primary language")
    description: str | None = Field(None, description="Repository description")
    created_at: datetime | None = Field(None, description="Creation time")
    updated_at: datetime | None = Field(None, description="Last update time")
    pushed_at: datetime | None = Field(None, description="Last push time")


class GitHubBranch(BaseModel):
    """GitHub branch reference."""

    model_config = ConfigDict(frozen=True)

    ref: str = Field(..., description="Branch reference")
    sha: str = Field(..., description="Commit SHA")
    label: str | None = Field(None, description="Branch label")
    user: GitHubUser | None = Field(None, description="Branch owner")
    repo: GitHubRepository | None = Field(None, description="Repository")


class GitHubFile(BaseModel):
    """GitHub file in a commit or PR."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(..., description="File path")
    status: FileStatus = Field(..., description="Change status")
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")
    changes: int = Field(default=0, description="Total changes")
    patch: str | None = Field(None, description="Diff patch")
    blob_url: str | None = Field(None, description="Blob URL")
    raw_url: str | None = Field(None, description="Raw content URL")
    contents_url: str | None = Field(None, description="Contents API URL")
    previous_filename: str | None = Field(None, description="Previous name if renamed")


class GitHubCommit(BaseModel):
    """GitHub commit model."""

    model_config = ConfigDict(frozen=True)

    sha: str = Field(..., description="Commit SHA")
    message: str = Field(..., description="Commit message")
    author: GitHubUser | None = Field(None, description="Commit author")
    committer: GitHubUser | None = Field(None, description="Committer")
    html_url: str | None = Field(None, description="Commit URL")
    timestamp: datetime | None = Field(None, description="Commit timestamp")
    files: list[GitHubFile] = Field(default_factory=list, description="Changed files")
    parents: list[str] = Field(default_factory=list, description="Parent commit SHAs")


class GitHubPullRequest(BaseModel):
    """GitHub pull request model."""

    model_config = ConfigDict(frozen=False)

    id: int = Field(..., description="PR ID")
    number: int = Field(..., description="PR number")
    title: str = Field(..., description="PR title")
    body: str | None = Field(None, description="PR description")
    state: PullRequestState = Field(..., description="PR state")
    user: GitHubUser = Field(..., description="PR author")
    html_url: str = Field(..., description="PR URL")
    diff_url: str | None = Field(None, description="Diff URL")
    patch_url: str | None = Field(None, description="Patch URL")
    head: GitHubBranch = Field(..., description="Head branch")
    base: GitHubBranch = Field(..., description="Base branch")
    merged: bool = Field(default=False, description="Is merged")
    mergeable: bool | None = Field(None, description="Is mergeable")
    merged_by: GitHubUser | None = Field(None, description="Merged by user")
    merged_at: datetime | None = Field(None, description="Merge time")
    created_at: datetime | None = Field(None, description="Creation time")
    updated_at: datetime | None = Field(None, description="Last update time")
    closed_at: datetime | None = Field(None, description="Close time")
    commits: int = Field(default=0, description="Number of commits")
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")
    changed_files: int = Field(default=0, description="Files changed")
    files: list[GitHubFile] = Field(default_factory=list, description="Changed files")
    labels: list[str] = Field(default_factory=list, description="Labels")
    reviewers: list[GitHubUser] = Field(default_factory=list, description="Requested reviewers")


class GitHubComment(BaseModel):
    """GitHub comment model (issue or PR comment)."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="Comment ID")
    body: str = Field(..., description="Comment body")
    user: GitHubUser = Field(..., description="Comment author")
    html_url: str | None = Field(None, description="Comment URL")
    created_at: datetime | None = Field(None, description="Creation time")
    updated_at: datetime | None = Field(None, description="Last update time")
    # For PR review comments
    path: str | None = Field(None, description="File path for review comments")
    line: int | None = Field(None, description="Line number")
    side: str | None = Field(None, description="Side of diff (LEFT/RIGHT)")
    commit_id: str | None = Field(None, description="Associated commit SHA")
    in_reply_to_id: int | None = Field(None, description="Parent comment ID")


class GitHubReview(BaseModel):
    """GitHub pull request review model."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="Review ID")
    user: GitHubUser = Field(..., description="Reviewer")
    body: str | None = Field(None, description="Review body")
    state: ReviewState = Field(..., description="Review state")
    html_url: str | None = Field(None, description="Review URL")
    commit_id: str | None = Field(None, description="Reviewed commit SHA")
    submitted_at: datetime | None = Field(None, description="Submission time")
    comments: list[GitHubComment] = Field(default_factory=list, description="Review comments")


class WebhookPayload(BaseModel):
    """GitHub webhook payload."""

    model_config = ConfigDict(frozen=False, extra="allow")

    action: str | None = Field(None, description="Webhook action")
    sender: GitHubUser | None = Field(None, description="Event sender")
    repository: GitHubRepository | None = Field(None, description="Repository")
    installation: dict[str, Any] | None = Field(None, description="GitHub App installation")

    # Event-specific fields
    pull_request: GitHubPullRequest | None = Field(None, description="PR for PR events")
    review: GitHubReview | None = Field(None, description="Review for review events")
    comment: GitHubComment | None = Field(None, description="Comment for comment events")
    commits: list[GitHubCommit] | None = Field(None, description="Commits for push events")
    ref: str | None = Field(None, description="Git ref for push events")
    before: str | None = Field(None, description="Before SHA for push events")
    after: str | None = Field(None, description="After SHA for push events")
    head_commit: GitHubCommit | None = Field(None, description="Head commit for push")


class GitHubInstallation(BaseModel):
    """GitHub App installation model."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(..., description="Installation ID")
    account: GitHubUser = Field(..., description="Installation account")
    app_id: int = Field(..., description="GitHub App ID")
    target_type: str = Field(..., description="Target type (User/Organization)")
    permissions: dict[str, str] = Field(default_factory=dict, description="Permissions")
    events: list[str] = Field(default_factory=list, description="Subscribed events")
    created_at: datetime | None = Field(None, description="Creation time")
    updated_at: datetime | None = Field(None, description="Last update time")
    repositories: list[GitHubRepository] = Field(
        default_factory=list, description="Accessible repos"
    )


class DiffHunk(BaseModel):
    """A hunk from a diff patch."""

    model_config = ConfigDict(frozen=True)

    old_start: int = Field(..., description="Old file start line")
    old_count: int = Field(..., description="Lines in old file")
    new_start: int = Field(..., description="New file start line")
    new_count: int = Field(..., description="Lines in new file")
    header: str = Field(..., description="Hunk header")
    lines: list[str] = Field(default_factory=list, description="Diff lines")


class FileDiff(BaseModel):
    """Parsed diff for a single file."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(..., description="File path")
    status: FileStatus = Field(..., description="Change status")
    hunks: list[DiffHunk] = Field(default_factory=list, description="Diff hunks")
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")
    language: str | None = Field(None, description="File language")
    old_filename: str | None = Field(None, description="Previous name if renamed")
