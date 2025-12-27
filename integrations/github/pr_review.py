"""Pull request review functionality for Kodo.

This module provides PR diff analysis and automated review generation
using code context from the knowledge graph and LLM-powered insights.
"""

import re
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from .client import GitHubClient
from .models import DiffHunk, FileDiff, FileStatus, GitHubFile, GitHubPullRequest

logger = structlog.get_logger(__name__)


class ReviewSeverity(str, Enum):
    """Severity levels for review comments."""

    INFO = "info"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    ERROR = "error"


class ReviewComment(BaseModel):
    """A review comment for a specific line in a file."""

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(..., description="File path")
    line: int = Field(..., description="Line number in new file")
    body: str = Field(..., description="Comment body")
    severity: ReviewSeverity = Field(
        default=ReviewSeverity.SUGGESTION, description="Comment severity"
    )
    suggestion: str | None = Field(None, description="Suggested code change")
    start_line: int | None = Field(None, description="Start line for multi-line comments")


class ReviewResult(BaseModel):
    """Result of PR review analysis."""

    model_config = ConfigDict(frozen=False)

    pr_number: int = Field(..., description="PR number")
    repo_full_name: str = Field(..., description="Repository full name")
    summary: str = Field(default="", description="Overall review summary")
    comments: list[ReviewComment] = Field(default_factory=list, description="Review comments")
    approval_status: str = Field(
        default="COMMENT", description="APPROVE, REQUEST_CHANGES, or COMMENT"
    )
    files_analyzed: int = Field(default=0, description="Number of files analyzed")
    issues_found: int = Field(default=0, description="Number of issues found")
    suggestions_made: int = Field(default=0, description="Number of suggestions made")


class DiffParser:
    """Parser for GitHub diff patches."""

    HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")

    def parse_patch(self, patch: str, filename: str) -> FileDiff:
        """Parse a diff patch into structured hunks.

        Args:
            patch: Git diff patch string.
            filename: File name.

        Returns:
            FileDiff with parsed hunks.
        """
        if not patch:
            return FileDiff(filename=filename, status=FileStatus.MODIFIED, hunks=[])

        hunks: list[DiffHunk] = []
        current_hunk: DiffHunk | None = None
        lines: list[str] = []

        for line in patch.split("\n"):
            match = self.HUNK_HEADER_PATTERN.match(line)
            if match:
                # Save previous hunk
                if current_hunk:
                    hunks.append(
                        DiffHunk(
                            old_start=current_hunk.old_start,
                            old_count=current_hunk.old_count,
                            new_start=current_hunk.new_start,
                            new_count=current_hunk.new_count,
                            header=current_hunk.header,
                            lines=lines,
                        )
                    )
                    lines = []

                # Start new hunk
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) else 1
                header = match.group(5).strip()

                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=header,
                    lines=[],
                )
            elif current_hunk:
                lines.append(line)

        # Save last hunk
        if current_hunk:
            hunks.append(
                DiffHunk(
                    old_start=current_hunk.old_start,
                    old_count=current_hunk.old_count,
                    new_start=current_hunk.new_start,
                    new_count=current_hunk.new_count,
                    header=current_hunk.header,
                    lines=lines,
                )
            )

        # Count additions and deletions
        additions = sum(1 for h in hunks for line in h.lines if line.startswith("+"))
        deletions = sum(1 for h in hunks for line in h.lines if line.startswith("-"))

        return FileDiff(
            filename=filename,
            status=FileStatus.MODIFIED,
            hunks=hunks,
            additions=additions,
            deletions=deletions,
        )

    def get_new_line_number(self, hunk: DiffHunk, line_index: int) -> int | None:
        """Get the new file line number for a diff line.

        Args:
            hunk: Diff hunk.
            line_index: Index within hunk lines.

        Returns:
            Line number in new file, or None if deleted line.
        """
        new_line = hunk.new_start

        for i, line in enumerate(hunk.lines):
            if i == line_index:
                if line.startswith("-"):
                    return None  # Deleted line
                return new_line

            if line.startswith("+"):
                new_line += 1
            elif line.startswith("-"):
                pass  # Don't increment for deletions
            else:
                new_line += 1  # Context line

        return None


class PRReviewer:
    """Automated PR reviewer using code analysis and LLM insights."""

    # Common code issues to detect
    ISSUE_PATTERNS = [
        # Security issues
        (r"\beval\s*\(", "security", "Avoid using eval() - security risk"),
        (r"\bexec\s*\(", "security", "Avoid using exec() - security risk"),
        (
            r"password\s*=\s*['\"][^'\"]+['\"]",
            "security",
            "Hardcoded password detected",
        ),
        (
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            "security",
            "Hardcoded API key detected",
        ),
        (r"\.execute\([^)]*%", "security", "Possible SQL injection vulnerability"),
        # Code quality
        (r"#\s*TODO", "todo", "TODO comment - consider addressing"),
        (r"#\s*FIXME", "todo", "FIXME comment - needs attention"),
        (r"#\s*HACK", "todo", "HACK comment - technical debt"),
        (r"print\s*\(", "debug", "Debug print statement left in code"),
        (r"console\.log\s*\(", "debug", "Debug console.log left in code"),
        (r"debugger\s*;?", "debug", "Debugger statement left in code"),
        # Best practices
        (r"except\s*:", "exception", "Bare except clause - catch specific exceptions"),
        (r"pass\s*$", "empty", "Empty pass block - consider implementation or removal"),
        (
            r"from .+ import \*",
            "import",
            "Wildcard import - import specific names instead",
        ),
    ]

    # Language detection
    LANGUAGE_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
    }

    def __init__(
        self,
        llm_client: Any | None = None,
        context_builder: Any | None = None,
    ) -> None:
        """Initialize the PR reviewer.

        Args:
            llm_client: LLM client for generating insights.
            context_builder: Context builder for code context.
        """
        self._llm_client = llm_client
        self._context_builder = context_builder
        self._diff_parser = DiffParser()
        self._logger = logger.bind(component="pr_reviewer")

    def _detect_language(self, filename: str) -> str | None:
        """Detect programming language from filename."""
        for ext, lang in self.LANGUAGE_EXTENSIONS.items():
            if filename.endswith(ext):
                return lang
        return None

    def _analyze_line(self, line: str, line_number: int, filename: str) -> list[ReviewComment]:
        """Analyze a single line for issues.

        Args:
            line: Line content.
            line_number: Line number in new file.
            filename: File path.

        Returns:
            List of review comments.
        """
        comments: list[ReviewComment] = []

        # Skip deleted lines (start with -)
        if line.startswith("-"):
            return comments

        # Get actual content (remove + prefix for added lines)
        content = line[1:] if line.startswith("+") else line

        for pattern, category, message in self.ISSUE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                severity = ReviewSeverity.WARNING
                if category == "security":
                    severity = ReviewSeverity.ERROR
                elif category in ("todo", "debug"):
                    severity = ReviewSeverity.INFO

                comments.append(
                    ReviewComment(
                        file_path=filename,
                        line=line_number,
                        body=f"**{category.upper()}**: {message}",
                        severity=severity,
                    )
                )

        return comments

    def analyze_diff(self, file: GitHubFile) -> tuple[FileDiff, list[ReviewComment]]:
        """Analyze a file diff for issues.

        Args:
            file: GitHub file with patch.

        Returns:
            Tuple of (FileDiff, list of comments).
        """
        comments: list[ReviewComment] = []

        # Parse the diff
        diff = self._diff_parser.parse_patch(file.patch or "", file.filename)

        # Analyze each hunk
        for hunk in diff.hunks:
            for i, line in enumerate(hunk.lines):
                # Only analyze added or modified lines
                if not line.startswith("+"):
                    continue

                line_number = self._diff_parser.get_new_line_number(hunk, i)
                if line_number:
                    line_comments = self._analyze_line(line, line_number, file.filename)
                    comments.extend(line_comments)

        return diff, comments

    async def review_pr(
        self,
        client: GitHubClient,
        owner: str,
        repo: str,
        pr_number: int,
        include_llm_review: bool = False,
    ) -> ReviewResult:
        """Review a pull request.

        Args:
            client: GitHub API client.
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            include_llm_review: Whether to include LLM-generated insights.

        Returns:
            ReviewResult with analysis and comments.
        """
        self._logger.info(
            "reviewing_pr",
            owner=owner,
            repo=repo,
            pr_number=pr_number,
        )

        # Get PR details and files
        pr = await client.get_pull_request(owner, repo, pr_number)
        files = await client.get_pull_request_files(owner, repo, pr_number)

        all_comments: list[ReviewComment] = []
        files_analyzed = 0
        total_additions = 0
        total_deletions = 0

        for file in files:
            # Skip binary files and large files
            if not file.patch:
                continue

            language = self._detect_language(file.filename)
            if not language:
                continue  # Skip unknown file types

            files_analyzed += 1
            diff, comments = self.analyze_diff(file)
            all_comments.extend(comments)
            total_additions += diff.additions
            total_deletions += diff.deletions

        # Include LLM review if requested and available
        if include_llm_review and self._llm_client:
            llm_comments = await self._generate_llm_review(pr, files)
            all_comments.extend(llm_comments)

        # Determine approval status based on findings
        error_count = sum(1 for c in all_comments if c.severity == ReviewSeverity.ERROR)
        warning_count = sum(1 for c in all_comments if c.severity == ReviewSeverity.WARNING)

        approval_status = "REQUEST_CHANGES" if error_count > 0 or warning_count > 3 else "COMMENT"

        # Generate summary
        summary = self._generate_summary(
            pr=pr,
            files_analyzed=files_analyzed,
            total_additions=total_additions,
            total_deletions=total_deletions,
            comments=all_comments,
        )

        return ReviewResult(
            pr_number=pr_number,
            repo_full_name=f"{owner}/{repo}",
            summary=summary,
            comments=all_comments,
            approval_status=approval_status,
            files_analyzed=files_analyzed,
            issues_found=error_count + warning_count,
            suggestions_made=sum(
                1 for c in all_comments if c.severity == ReviewSeverity.SUGGESTION
            ),
        )

    def _generate_summary(
        self,
        pr: GitHubPullRequest,
        files_analyzed: int,
        total_additions: int,
        total_deletions: int,
        comments: list[ReviewComment],
    ) -> str:
        """Generate a review summary.

        Args:
            pr: Pull request.
            files_analyzed: Number of files analyzed.
            total_additions: Total lines added.
            total_deletions: Total lines deleted.
            comments: Review comments.

        Returns:
            Markdown summary string.
        """
        error_count = sum(1 for c in comments if c.severity == ReviewSeverity.ERROR)
        warning_count = sum(1 for c in comments if c.severity == ReviewSeverity.WARNING)
        info_count = sum(1 for c in comments if c.severity == ReviewSeverity.INFO)

        summary_parts = [
            "## Kodo Review Summary",
            "",
            f"**Files analyzed:** {files_analyzed}",
            f"**Changes:** +{total_additions} / -{total_deletions}",
            "",
        ]

        if error_count > 0:
            summary_parts.append(f"- {error_count} error(s) found")
        if warning_count > 0:
            summary_parts.append(f"- {warning_count} warning(s) found")
        if info_count > 0:
            summary_parts.append(f"- {info_count} informational note(s)")

        if not comments:
            summary_parts.append("No issues found. Good job!")

        return "\n".join(summary_parts)

    async def _generate_llm_review(
        self,
        pr: GitHubPullRequest,
        files: list[GitHubFile],
    ) -> list[ReviewComment]:
        """Generate LLM-powered review comments.

        Args:
            pr: Pull request.
            files: Changed files.

        Returns:
            List of LLM-generated review comments.
        """
        if not self._llm_client:
            return []

        # This would be implemented with actual LLM calls
        # For now, return empty list
        self._logger.debug("llm_review_not_implemented")
        return []

    async def submit_review(
        self,
        client: GitHubClient,
        owner: str,
        repo: str,
        review: ReviewResult,
    ) -> None:
        """Submit a review to GitHub.

        Args:
            client: GitHub API client.
            owner: Repository owner.
            repo: Repository name.
            review: Review result to submit.
        """
        self._logger.info(
            "submitting_review",
            pr_number=review.pr_number,
            comments=len(review.comments),
            status=review.approval_status,
        )

        # Format comments for GitHub API
        github_comments: list[dict[str, Any]] = []
        for comment in review.comments:
            github_comment: dict[str, Any] = {
                "path": comment.file_path,
                "line": comment.line,
                "body": comment.body,
            }
            if comment.start_line:
                github_comment["start_line"] = comment.start_line
            github_comments.append(github_comment)

        # Submit the review
        await client.create_review(
            owner=owner,
            repo=repo,
            pr_number=review.pr_number,
            body=review.summary,
            event=review.approval_status,
            comments=github_comments if github_comments else None,
        )

        self._logger.info(
            "review_submitted",
            pr_number=review.pr_number,
        )
