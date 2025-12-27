"""Repository management endpoints for the Kodo API.

This module provides endpoints for managing code repositories,
including adding, listing, syncing, and removing repositories.
"""

from datetime import datetime
from enum import Enum

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from api.dependencies import GraphStoreDep, IngestionPipelineDep
from core.graph.models import RepositoryNode
from core.ingestion.models import IngestionError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/repos", tags=["Repositories"])


class RepositoryStatus(str, Enum):
    """Status of a repository in the system."""

    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class CreateRepositoryRequest(BaseModel):
    """Request model for adding a new repository.

    Attributes:
        url: Git repository URL (HTTPS or SSH).
        name: Optional display name (derived from URL if not provided).
        branch: Branch to index (defaults to default branch).
    """

    url: HttpUrl = Field(..., description="Git repository URL")
    name: str | None = Field(None, description="Optional display name")
    branch: str | None = Field(None, description="Branch to index")


class RepositoryResponse(BaseModel):
    """Response model for repository details.

    Attributes:
        id: Unique repository identifier.
        name: Repository display name.
        url: Repository URL.
        default_branch: Default branch name.
        status: Current repository status.
        last_indexed: Timestamp of last indexing.
        file_count: Number of indexed files.
        function_count: Number of indexed functions.
        class_count: Number of indexed classes.
    """

    id: str = Field(..., description="Repository ID")
    name: str = Field(..., description="Repository name")
    url: str | None = Field(None, description="Repository URL")
    default_branch: str = Field(..., description="Default branch")
    status: RepositoryStatus = Field(..., description="Repository status")
    last_indexed: datetime | None = Field(None, description="Last indexing timestamp")
    file_count: int = Field(default=0, description="Number of indexed files")
    function_count: int = Field(default=0, description="Number of indexed functions")
    class_count: int = Field(default=0, description="Number of indexed classes")


class RepositoryListResponse(BaseModel):
    """Response model for listing repositories.

    Attributes:
        repositories: List of repository details.
        total: Total number of repositories.
    """

    repositories: list[RepositoryResponse] = Field(..., description="List of repositories")
    total: int = Field(..., description="Total count")


class SyncResponse(BaseModel):
    """Response model for repository sync operation.

    Attributes:
        repo_id: Repository identifier.
        status: Sync status.
        message: Status message.
        files_processed: Number of files processed.
        entities_extracted: Number of entities extracted.
        errors: List of errors encountered.
    """

    repo_id: str = Field(..., description="Repository ID")
    status: str = Field(..., description="Sync status")
    message: str = Field(..., description="Status message")
    files_processed: int = Field(default=0, description="Files processed")
    entities_extracted: int = Field(default=0, description="Entities extracted")
    errors: list[IngestionError] = Field(default_factory=list, description="Errors")


class DeleteResponse(BaseModel):
    """Response model for repository deletion.

    Attributes:
        repo_id: Deleted repository ID.
        deleted: Whether deletion was successful.
        message: Status message.
    """

    repo_id: str = Field(..., description="Repository ID")
    deleted: bool = Field(..., description="Deletion success")
    message: str = Field(..., description="Status message")


@router.post(
    "",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add repository",
    description="Add a new repository to be indexed.",
)
async def add_repository(
    request: CreateRepositoryRequest,
    graph_store: GraphStoreDep,
    ingestion_pipeline: IngestionPipelineDep,
    background_tasks: BackgroundTasks,
) -> RepositoryResponse:
    """Add a new repository for indexing.

    Clones the repository (if remote) and begins the indexing process
    in the background.

    Args:
        request: Repository creation request.
        graph_store: Graph store for database operations.
        ingestion_pipeline: Pipeline for repository indexing.
        background_tasks: FastAPI background task runner.

    Returns:
        RepositoryResponse with initial repository details.

    Raises:
        HTTPException: If repository already exists or URL is invalid.
    """
    # Generate repository ID from URL
    url_str = str(request.url)
    repo_name = request.name or url_str.rstrip("/").split("/")[-1].replace(".git", "")
    repo_id = f"repo:{repo_name}"

    # Check if repository already exists
    existing = await graph_store.get_repository(repo_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{repo_name}' already exists",
        )

    # Create repository node
    repo_node = RepositoryNode(
        id=repo_id,
        name=repo_name,
        url=url_str,
        default_branch=request.branch or "main",
        last_indexed=None,
    )

    try:
        await graph_store.create_repository(repo_node)
    except Exception as e:
        logger.error("Failed to create repository", repo_id=repo_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create repository",
        ) from e

    # TODO: Add background task to clone and index repository
    # background_tasks.add_task(index_repository, repo_id, url_str)

    logger.info("Repository created", repo_id=repo_id, name=repo_name)

    return RepositoryResponse(
        id=repo_id,
        name=repo_name,
        url=url_str,
        default_branch=request.branch or "main",
        status=RepositoryStatus.PENDING,
        last_indexed=None,
        file_count=0,
        function_count=0,
        class_count=0,
    )


@router.get(
    "",
    response_model=RepositoryListResponse,
    summary="List repositories",
    description="List all indexed repositories.",
)
async def list_repositories(
    graph_store: GraphStoreDep,
) -> RepositoryListResponse:
    """List all repositories in the system.

    Args:
        graph_store: Graph store for database operations.

    Returns:
        RepositoryListResponse with list of all repositories.
    """
    # Query all repositories from graph
    # For now, we return an empty list as we need to implement
    # a proper query for listing all repositories
    repositories: list[RepositoryResponse] = []

    # TODO: Implement proper repository listing from Neo4j
    # This requires a new query in the GraphStore

    return RepositoryListResponse(
        repositories=repositories,
        total=len(repositories),
    )


@router.get(
    "/{repo_id}",
    response_model=RepositoryResponse,
    summary="Get repository",
    description="Get details of a specific repository.",
)
async def get_repository(
    repo_id: str,
    graph_store: GraphStoreDep,
) -> RepositoryResponse:
    """Get details of a specific repository.

    Args:
        repo_id: Repository identifier.
        graph_store: Graph store for database operations.

    Returns:
        RepositoryResponse with repository details.

    Raises:
        HTTPException: If repository not found.
    """
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Get repository statistics
    stats = await graph_store.get_repository_statistics(repo_id)

    return RepositoryResponse(
        id=repo.id,
        name=repo.name,
        url=repo.url,
        default_branch=repo.default_branch,
        status=RepositoryStatus.READY if repo.last_indexed else RepositoryStatus.PENDING,
        last_indexed=repo.last_indexed,
        file_count=stats.get("file_count", 0),
        function_count=stats.get("function_count", 0),
        class_count=stats.get("class_count", 0),
    )


@router.post(
    "/{repo_id}/sync",
    response_model=SyncResponse,
    summary="Sync repository",
    description="Trigger re-indexing of a repository.",
)
async def sync_repository(
    repo_id: str,
    graph_store: GraphStoreDep,
    ingestion_pipeline: IngestionPipelineDep,
) -> SyncResponse:
    """Trigger re-indexing of a repository.

    Performs incremental indexing based on changes since last sync.
    For initial indexing, performs full ingestion.

    Args:
        repo_id: Repository identifier.
        graph_store: Graph store for database operations.
        ingestion_pipeline: Pipeline for repository indexing.

    Returns:
        SyncResponse with indexing results.

    Raises:
        HTTPException: If repository not found.
    """
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # TODO: Implement actual re-indexing
    # For now, return a placeholder response

    logger.info("Sync requested", repo_id=repo_id)

    return SyncResponse(
        repo_id=repo_id,
        status="pending",
        message="Sync operation queued. Check status for updates.",
        files_processed=0,
        entities_extracted=0,
        errors=[],
    )


@router.delete(
    "/{repo_id}",
    response_model=DeleteResponse,
    summary="Delete repository",
    description="Remove a repository and all its indexed data.",
)
async def delete_repository(
    repo_id: str,
    graph_store: GraphStoreDep,
) -> DeleteResponse:
    """Delete a repository and all its indexed data.

    This operation is irreversible. All files, functions, classes,
    and relationships will be removed.

    Args:
        repo_id: Repository identifier.
        graph_store: Graph store for database operations.

    Returns:
        DeleteResponse indicating success or failure.

    Raises:
        HTTPException: If repository not found.
    """
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    try:
        success = await graph_store.delete_repository(repo_id)
        if success:
            logger.info("Repository deleted", repo_id=repo_id)
            return DeleteResponse(
                repo_id=repo_id,
                deleted=True,
                message=f"Repository '{repo_id}' and all data deleted successfully",
            )
        else:
            return DeleteResponse(
                repo_id=repo_id,
                deleted=False,
                message="Failed to delete repository",
            )
    except Exception as e:
        logger.error("Failed to delete repository", repo_id=repo_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete repository",
        ) from e
