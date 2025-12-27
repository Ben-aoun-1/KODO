"""Kodo - API module for REST endpoints.

This module provides the FastAPI application and all related components
for the Kodo code analysis API.
"""

from .config import Settings, get_settings
from .dependencies import (
    get_graph_connection,
    get_graph_store,
    get_ingestion_pipeline,
)
from .main import app, create_app

__all__ = [
    "app",
    "create_app",
    "get_graph_connection",
    "get_graph_store",
    "get_ingestion_pipeline",
    "get_settings",
    "Settings",
]
