"""Custom middleware for the Kodo API.

This module provides middleware components for request logging,
response timing, and other cross-cutting concerns.
"""

import time
import uuid
from collections.abc import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs all incoming requests and outgoing responses.

    Logs request method, path, status code, and timing information
    for observability and debugging purposes.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Process the request and log details.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response.
        """
        # Generate a unique request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log incoming request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            client_host=request.client.host if request.client else None,
        )

        # Process the request
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as e:
            # Log unhandled exceptions
            logger.exception(
                "Request failed with unhandled exception",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
            )
            raise

        # Calculate response time
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log completed request
        log_level = "info" if response.status_code < 400 else "warning"
        getattr(logger, log_level)(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add request ID to response headers for tracing
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds response timing headers.

    Adds X-Response-Time header to all responses indicating
    how long the request took to process.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Process the request and add timing header.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response with timing header.
        """
        start_time = time.perf_counter()

        response = await call_next(request)

        # Calculate and add response time header
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response
