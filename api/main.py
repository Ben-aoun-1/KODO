"""FastAPI application for the Kodo API.

This module creates and configures the main FastAPI application,
including routers, middleware, CORS settings, and lifecycle events.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .dependencies import init_dependencies, shutdown_dependencies
from .middleware import RequestLoggingMiddleware, TimingMiddleware
from .routers import health_router, query_router, repos_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events for the application.
    Initializes database connections on startup and cleans up on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup is complete.
    """
    # Startup
    settings = get_settings()
    logger.info(
        "Starting Kodo API",
        version=settings.app_version,
        debug=settings.debug,
    )

    try:
        await init_dependencies(settings)
        logger.info("Dependencies initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize dependencies", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down Kodo API")
    await shutdown_dependencies()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Kodo API",
        description=(
            "Code-aware AI assistant API that understands codebases as "
            "interconnected systems. Provides semantic search, code analysis, "
            "and natural language queries over code repositories."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)

    # Include routers
    # Health routes are at root level (/health)
    app.include_router(health_router)

    # API routes are prefixed with /api/v1
    app.include_router(repos_router, prefix=settings.api_prefix)
    app.include_router(query_router, prefix=settings.api_prefix)

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Root endpoint returning API information."""
        return JSONResponse(
            content={
                "name": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "health": "/health",
            }
        )

    return app


# Create the application instance
app = create_app()
