"""Code query endpoints for the Kodo API.

This module provides endpoints for querying code entities, relationships,
and performing graph traversals like caller/callee analysis and impact analysis.
"""

from enum import Enum

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.dependencies import GraphStoreDep
from core.graph.models import (
    ClassNode,
    FileNode,
    FunctionNode,
    GraphNode,
    ImpactResult,
    MethodNode,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/repos/{repo_id}", tags=["Code Query"])


class EntityType(str, Enum):
    """Types of code entities."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    FILE = "file"
    VARIABLE = "variable"


class FileResponse(BaseModel):
    """Response model for file information.

    Attributes:
        id: File identifier.
        path: Relative file path.
        language: Programming language.
        size: File size in bytes.
    """

    id: str = Field(..., description="File ID")
    path: str = Field(..., description="Relative file path")
    language: str = Field(..., description="Programming language")
    size: int | None = Field(None, description="File size in bytes")


class FileListResponse(BaseModel):
    """Response model for listing files.

    Attributes:
        files: List of files.
        total: Total count.
    """

    files: list[FileResponse] = Field(..., description="List of files")
    total: int = Field(..., description="Total count")


class FunctionResponse(BaseModel):
    """Response model for function information.

    Attributes:
        id: Function identifier.
        name: Function name.
        file_path: Path to containing file.
        start_line: Starting line number.
        end_line: Ending line number.
        is_async: Whether function is async.
        docstring: Function docstring.
        parameters: List of parameter names.
        return_type: Return type annotation.
    """

    id: str = Field(..., description="Function ID")
    name: str = Field(..., description="Function name")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    is_async: bool = Field(default=False, description="Is async")
    docstring: str | None = Field(None, description="Docstring")
    parameters: list[str] = Field(default_factory=list, description="Parameters")
    return_type: str | None = Field(None, description="Return type")


class FunctionListResponse(BaseModel):
    """Response model for listing functions.

    Attributes:
        functions: List of functions.
        total: Total count.
    """

    functions: list[FunctionResponse] = Field(..., description="List of functions")
    total: int = Field(..., description="Total count")


class ClassResponse(BaseModel):
    """Response model for class information.

    Attributes:
        id: Class identifier.
        name: Class name.
        file_path: Path to containing file.
        start_line: Starting line number.
        end_line: Ending line number.
        docstring: Class docstring.
        bases: List of base class names.
        method_count: Number of methods.
    """

    id: str = Field(..., description="Class ID")
    name: str = Field(..., description="Class name")
    file_path: str = Field(..., description="File path")
    start_line: int = Field(..., description="Start line")
    end_line: int = Field(..., description="End line")
    docstring: str | None = Field(None, description="Docstring")
    bases: list[str] = Field(default_factory=list, description="Base classes")
    method_count: int = Field(default=0, description="Method count")


class ClassListResponse(BaseModel):
    """Response model for listing classes.

    Attributes:
        classes: List of classes.
        total: Total count.
    """

    classes: list[ClassResponse] = Field(..., description="List of classes")
    total: int = Field(..., description="Total count")


class CallerCalleeResponse(BaseModel):
    """Response model for caller/callee information.

    Attributes:
        entity_id: The queried entity ID.
        entity_name: The queried entity name.
        results: List of caller or callee entities.
        total: Total count.
    """

    entity_id: str = Field(..., description="Queried entity ID")
    entity_name: str = Field(..., description="Queried entity name")
    results: list[FunctionResponse] = Field(..., description="Caller/callee list")
    total: int = Field(..., description="Total count")


class ImpactAnalysisResponse(BaseModel):
    """Response model for impact analysis.

    Attributes:
        source_id: The modified entity ID.
        source_name: The modified entity name.
        affected_entities: List of affected entity IDs.
        affected_files: List of affected file paths.
        impact_depth: Maximum depth of impact.
        total_affected: Total number of affected entities.
    """

    source_id: str = Field(..., description="Modified entity ID")
    source_name: str = Field(..., description="Modified entity name")
    affected_entities: list[str] = Field(..., description="Affected entity IDs")
    affected_files: list[str] = Field(..., description="Affected file paths")
    impact_depth: int = Field(..., description="Maximum impact depth")
    total_affected: int = Field(..., description="Total affected count")


def _file_node_to_response(node: FileNode) -> FileResponse:
    """Convert FileNode to FileResponse."""
    return FileResponse(
        id=node.id,
        path=node.path,
        language=node.language,
        size=node.size,
    )


def _function_node_to_response(node: FunctionNode | MethodNode) -> FunctionResponse:
    """Convert FunctionNode or MethodNode to FunctionResponse."""
    return FunctionResponse(
        id=node.id,
        name=node.name,
        file_path=node.file_path,
        start_line=node.start_line,
        end_line=node.end_line,
        is_async=node.is_async,
        docstring=node.docstring,
        parameters=node.parameters,
        return_type=node.return_type,
    )


def _class_node_to_response(node: ClassNode, method_count: int = 0) -> ClassResponse:
    """Convert ClassNode to ClassResponse."""
    return ClassResponse(
        id=node.id,
        name=node.name,
        file_path=node.file_path,
        start_line=node.start_line,
        end_line=node.end_line,
        docstring=node.docstring,
        bases=node.bases,
        method_count=method_count,
    )


def _graph_node_to_function_response(node: GraphNode) -> FunctionResponse | None:
    """Convert any GraphNode to FunctionResponse if applicable."""
    if isinstance(node, FunctionNode | MethodNode):
        return _function_node_to_response(node)
    return None


@router.get(
    "/files",
    response_model=FileListResponse,
    summary="List files",
    description="List all files in a repository.",
)
async def list_files(
    repo_id: str,
    graph_store: GraphStoreDep,
    language: str | None = Query(None, description="Filter by language"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
) -> FileListResponse:
    """List all files in a repository.

    Args:
        repo_id: Repository identifier.
        graph_store: Graph store for database operations.
        language: Optional language filter.
        limit: Maximum number of results.
        offset: Offset for pagination.

    Returns:
        FileListResponse with list of files.

    Raises:
        HTTPException: If repository not found.
    """
    # Verify repository exists
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Get files from repository
    files = await graph_store.get_files_in_repository(repo_id)

    # Apply language filter if provided
    if language:
        files = [f for f in files if f.language.lower() == language.lower()]

    # Apply pagination
    total = len(files)
    files = files[offset : offset + limit]

    return FileListResponse(
        files=[_file_node_to_response(f) for f in files],
        total=total,
    )


@router.get(
    "/functions",
    response_model=FunctionListResponse,
    summary="List functions",
    description="List all functions in a repository.",
)
async def list_functions(
    repo_id: str,
    graph_store: GraphStoreDep,
    name: str | None = Query(None, description="Filter by function name"),
    file_path: str | None = Query(None, description="Filter by file path"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
) -> FunctionListResponse:
    """List all functions in a repository.

    Args:
        repo_id: Repository identifier.
        graph_store: Graph store for database operations.
        name: Optional name filter (partial match).
        file_path: Optional file path filter.
        limit: Maximum number of results.
        offset: Offset for pagination.

    Returns:
        FunctionListResponse with list of functions.

    Raises:
        HTTPException: If repository not found.
    """
    # Verify repository exists
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Get functions matching criteria
    functions: list[FunctionNode] = []
    if name:
        functions = await graph_store.find_function_by_name(name, repo_id)
    else:
        # TODO: Implement get_all_functions_in_repository
        # For now, return empty list if no name filter
        pass

    # Apply file path filter if provided
    if file_path:
        functions = [f for f in functions if file_path in f.file_path]

    # Apply pagination
    total = len(functions)
    functions = functions[offset : offset + limit]

    return FunctionListResponse(
        functions=[_function_node_to_response(f) for f in functions],
        total=total,
    )


@router.get(
    "/classes",
    response_model=ClassListResponse,
    summary="List classes",
    description="List all classes in a repository.",
)
async def list_classes(
    repo_id: str,
    graph_store: GraphStoreDep,
    name: str | None = Query(None, description="Filter by class name"),
    file_path: str | None = Query(None, description="Filter by file path"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
) -> ClassListResponse:
    """List all classes in a repository.

    Args:
        repo_id: Repository identifier.
        graph_store: Graph store for database operations.
        name: Optional name filter (partial match).
        file_path: Optional file path filter.
        limit: Maximum number of results.
        offset: Offset for pagination.

    Returns:
        ClassListResponse with list of classes.

    Raises:
        HTTPException: If repository not found.
    """
    # Verify repository exists
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Get classes matching criteria
    classes: list[ClassNode] = []
    if name:
        classes = await graph_store.find_class_by_name(name, repo_id)
    else:
        # TODO: Implement get_all_classes_in_repository
        # For now, return empty list if no name filter
        pass

    # Apply file path filter if provided
    if file_path:
        classes = [c for c in classes if file_path in c.file_path]

    # Apply pagination
    total = len(classes)
    classes = classes[offset : offset + limit]

    # Get method counts for each class
    class_responses = []
    for cls in classes:
        methods = await graph_store.get_methods_of_class(cls.id)
        class_responses.append(_class_node_to_response(cls, len(methods)))

    return ClassListResponse(
        classes=class_responses,
        total=total,
    )


@router.get(
    "/graph/callers/{function_name}",
    response_model=CallerCalleeResponse,
    summary="Find callers",
    description="Find all functions that call a given function.",
)
async def find_callers(
    repo_id: str,
    function_name: str,
    graph_store: GraphStoreDep,
    max_depth: int = Query(1, ge=1, le=10, description="Maximum depth for recursive search"),
) -> CallerCalleeResponse:
    """Find all functions that call a given function.

    Args:
        repo_id: Repository identifier.
        function_name: Name of the function to find callers for.
        graph_store: Graph store for database operations.
        max_depth: Maximum depth for recursive caller search.

    Returns:
        CallerCalleeResponse with list of callers.

    Raises:
        HTTPException: If repository or function not found.
    """
    # Verify repository exists
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Find the function
    functions = await graph_store.find_function_by_name(function_name, repo_id)
    if not functions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Function '{function_name}' not found in repository",
        )

    # Use the first matching function
    target_func = functions[0]

    # Find callers
    if max_depth > 1:
        callers_with_depth = await graph_store.find_callers_recursive(target_func.id, max_depth)
        caller_nodes = [node for node, _ in callers_with_depth]
    else:
        caller_nodes = await graph_store.find_callers(target_func.id)

    # Convert to response format
    caller_responses = []
    for node in caller_nodes:
        response = _graph_node_to_function_response(node)
        if response:
            caller_responses.append(response)

    return CallerCalleeResponse(
        entity_id=target_func.id,
        entity_name=target_func.name,
        results=caller_responses,
        total=len(caller_responses),
    )


@router.get(
    "/graph/callees/{function_name}",
    response_model=CallerCalleeResponse,
    summary="Find callees",
    description="Find all functions called by a given function.",
)
async def find_callees(
    repo_id: str,
    function_name: str,
    graph_store: GraphStoreDep,
) -> CallerCalleeResponse:
    """Find all functions called by a given function.

    Args:
        repo_id: Repository identifier.
        function_name: Name of the function to find callees for.
        graph_store: Graph store for database operations.

    Returns:
        CallerCalleeResponse with list of callees.

    Raises:
        HTTPException: If repository or function not found.
    """
    # Verify repository exists
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Find the function
    functions = await graph_store.find_function_by_name(function_name, repo_id)
    if not functions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Function '{function_name}' not found in repository",
        )

    # Use the first matching function
    source_func = functions[0]

    # Find callees
    callee_nodes = await graph_store.find_callees(source_func.id)

    # Convert to response format
    callee_responses = []
    for node in callee_nodes:
        response = _graph_node_to_function_response(node)
        if response:
            callee_responses.append(response)

    return CallerCalleeResponse(
        entity_id=source_func.id,
        entity_name=source_func.name,
        results=callee_responses,
        total=len(callee_responses),
    )


@router.get(
    "/graph/impact/{entity_name}",
    response_model=ImpactAnalysisResponse,
    summary="Impact analysis",
    description="Analyze the impact of modifying a code entity.",
)
async def analyze_impact(
    repo_id: str,
    entity_name: str,
    graph_store: GraphStoreDep,
    entity_type: EntityType = Query(EntityType.FUNCTION, description="Type of entity to analyze"),
    max_depth: int = Query(5, ge=1, le=10, description="Maximum depth for impact analysis"),
) -> ImpactAnalysisResponse:
    """Analyze the potential impact of modifying a code entity.

    Finds all code that could be affected by changes to the specified entity.

    Args:
        repo_id: Repository identifier.
        entity_name: Name of the entity to analyze.
        graph_store: Graph store for database operations.
        entity_type: Type of entity (function, class, etc.).
        max_depth: Maximum depth for impact traversal.

    Returns:
        ImpactAnalysisResponse with affected entities and files.

    Raises:
        HTTPException: If repository or entity not found.
    """
    # Verify repository exists
    repo = await graph_store.get_repository(repo_id)
    if not repo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' not found",
        )

    # Find the entity based on type
    entity_id: str | None = None
    if entity_type == EntityType.FUNCTION:
        functions = await graph_store.find_function_by_name(entity_name, repo_id)
        if functions:
            entity_id = functions[0].id
    elif entity_type == EntityType.CLASS:
        classes = await graph_store.find_class_by_name(entity_name, repo_id)
        if classes:
            entity_id = classes[0].id
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impact analysis not supported for entity type: {entity_type}",
        )

    if not entity_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{entity_type.value.capitalize()} '{entity_name}' not found",
        )

    # Perform impact analysis
    impact: ImpactResult = await graph_store.analyze_impact(entity_id, max_depth)

    return ImpactAnalysisResponse(
        source_id=impact.source_id,
        source_name=entity_name,
        affected_entities=impact.affected_nodes,
        affected_files=impact.affected_files,
        impact_depth=impact.depth,
        total_affected=impact.total_affected,
    )
