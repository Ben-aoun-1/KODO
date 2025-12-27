"""Health check endpoints for the Kodo API.

This module provides endpoints for monitoring application health,
readiness, and liveness. Used by orchestration systems like Kubernetes.
"""

from enum import Enum

import structlog
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from api.dependencies import GraphConnectionDep, SettingsDep

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthResponse(BaseModel):
    """Response model for health check endpoints.

    Attributes:
        status: Overall health status.
        message: Optional status message.
    """

    status: HealthStatus = Field(..., description="Health status")
    message: str | None = Field(None, description="Optional status message")


class ReadinessResponse(BaseModel):
    """Response model for readiness check.

    Attributes:
        status: Overall readiness status.
        checks: Individual service check results.
    """

    status: HealthStatus = Field(..., description="Overall readiness status")
    checks: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Individual service check results",
    )


class LivenessResponse(BaseModel):
    """Response model for liveness check.

    Attributes:
        status: Liveness status.
        uptime_seconds: Time since application started.
    """

    status: HealthStatus = Field(..., description="Liveness status")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status of the API.",
)
async def health_check() -> HealthResponse:
    """Basic health check endpoint.

    Returns a simple healthy status indicating the API is running.
    This endpoint does not check external dependencies.

    Returns:
        HealthResponse with healthy status.
    """
    return HealthResponse(
        status=HealthStatus.HEALTHY,
        message="Kodo API is running",
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Checks if the application is ready to handle requests.",
    responses={
        status.HTTP_200_OK: {"description": "Application is ready"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Application is not ready"},
    },
)
async def readiness_check(
    settings: SettingsDep,
    graph_connection: GraphConnectionDep,
) -> ReadinessResponse:
    """Readiness check endpoint.

    Verifies that all required external services (databases, etc.)
    are accessible and the application can handle requests.

    Args:
        settings: Application settings.
        graph_connection: Neo4j connection for health check.

    Returns:
        ReadinessResponse with check results for each service.
    """
    checks: dict[str, dict[str, str]] = {}
    overall_healthy = True

    # Check Neo4j connection
    try:
        neo4j_health = await graph_connection.health_check()
        if neo4j_health.get("status") == "healthy":
            checks["neo4j"] = {
                "status": "healthy",
                "uri": settings.neo4j_uri,
            }
        else:
            checks["neo4j"] = {
                "status": "unhealthy",
                "message": neo4j_health.get("message", "Unknown error"),
            }
            overall_healthy = False
    except Exception as e:
        logger.warning("Neo4j health check failed", error=str(e))
        checks["neo4j"] = {
            "status": "unhealthy",
            "message": str(e),
        }
        overall_healthy = False

    # Future: Add Qdrant health check
    # Future: Add PostgreSQL health check if configured

    return ReadinessResponse(
        status=HealthStatus.HEALTHY if overall_healthy else HealthStatus.UNHEALTHY,
        checks=checks,
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness check",
    description="Checks if the application process is alive.",
)
async def liveness_check() -> LivenessResponse:
    """Liveness check endpoint.

    Returns a simple status indicating the application process is alive.
    Used by orchestration systems to determine if the process should
    be restarted.

    Returns:
        LivenessResponse with alive status.
    """
    return LivenessResponse(status=HealthStatus.HEALTHY)
