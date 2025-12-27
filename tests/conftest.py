"""Pytest configuration and shared fixtures for parser tests.

This module provides common fixtures used across parser unit tests,
including sample Python code snippets and parser instances.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Sample Python Code Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_function_code() -> str:
    """Simple function without decorators or type hints."""
    return '''def greet(name):
    """Say hello to someone."""
    return f"Hello, {name}!"
'''


@pytest.fixture
def async_function_code() -> str:
    """Async function with type hints."""
    return '''async def fetch_data(url: str, timeout: int = 30) -> dict:
    """Fetch data from a URL.

    Args:
        url: The URL to fetch from.
        timeout: Request timeout in seconds.

    Returns:
        The response data as a dictionary.
    """
    response = await http_client.get(url, timeout=timeout)
    return response.json()
'''


@pytest.fixture
def decorated_function_code() -> str:
    """Function with multiple decorators."""
    return '''@app.route("/api/users")
@require_auth
@cache(ttl=300)
def get_users(limit: int = 10, offset: int = 0) -> list[dict]:
    """Get a list of users."""
    return db.query(User).limit(limit).offset(offset).all()
'''


@pytest.fixture
def function_with_variadic_params_code() -> str:
    """Function with *args and **kwargs."""
    return '''def flexible_function(required: str, *args, default: int = 0, **kwargs) -> None:
    """A function with various parameter types."""
    print(required, args, default, kwargs)
'''


@pytest.fixture
def simple_class_code() -> str:
    """Simple class without inheritance."""
    return '''class Calculator:
    """A simple calculator class."""

    precision: int = 2

    def __init__(self, initial_value: float = 0.0):
        """Initialize the calculator."""
        self.value = initial_value

    def add(self, x: float) -> float:
        """Add a number to the current value."""
        self.value += x
        return self.value

    def subtract(self, x: float) -> float:
        """Subtract a number from the current value."""
        self.value -= x
        return self.value
'''


@pytest.fixture
def class_with_inheritance_code() -> str:
    """Class with single and multiple inheritance."""
    return '''class Animal:
    """Base animal class."""

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        raise NotImplementedError


class Dog(Animal):
    """A dog that can bark."""

    def speak(self) -> str:
        return "Woof!"


class ServiceDog(Dog, Trainable):
    """A service dog with training capabilities."""

    trained: bool = False

    def perform_task(self, task: str) -> bool:
        """Perform a trained task."""
        return self.trained and task in self.known_tasks
'''


@pytest.fixture
def class_with_decorators_code() -> str:
    """Class with class and method decorators."""
    return '''@dataclass
@register
class User:
    """A user model."""

    id: int
    name: str
    email: str

    @property
    def display_name(self) -> str:
        """Get the display name."""
        return self.name.title()

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create a user from a dictionary."""
        return cls(**data)

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate an email address."""
        return "@" in email
'''


@pytest.fixture
def imports_code() -> str:
    """Various import statement styles."""
    return """import os
import sys
import json as js
from pathlib import Path
from typing import Optional, List, Dict
from collections.abc import Mapping, Sequence
from ..utils import helper
from . import config
from .models import User, Post as BlogPost
"""


@pytest.fixture
def module_level_variables_code() -> str:
    """Module-level variable definitions."""
    return '''"""Module with various variable types."""

VERSION = "1.0.0"
DEBUG: bool = True
MAX_CONNECTIONS: int = 100
_PRIVATE_CONSTANT = "secret"

config: dict = {
    "host": "localhost",
    "port": 8080,
}

database_url: str = "postgresql://localhost/db"
'''


@pytest.fixture
def complex_module_code() -> str:
    """A complete module with all entity types."""
    return '''"""A complex module demonstrating various Python constructs."""

from typing import Optional, List
from dataclasses import dataclass

VERSION = "2.0.0"
DEFAULT_TIMEOUT: int = 30


@dataclass
class Config:
    """Application configuration."""

    host: str = "localhost"
    port: int = 8080
    debug: bool = False


class Service:
    """A service class with various method types."""

    _instances: List["Service"] = []

    def __init__(self, config: Config):
        """Initialize the service."""
        self.config = config
        self._running = False

    async def start(self) -> None:
        """Start the service."""
        self._running = True
        await self._initialize()

    async def _initialize(self) -> None:
        """Internal initialization."""
        pass

    @classmethod
    def get_instances(cls) -> List["Service"]:
        """Get all service instances."""
        return cls._instances.copy()


def create_service(host: str = "localhost", port: int = 8080) -> Service:
    """Factory function to create a service."""
    config = Config(host=host, port=port)
    return Service(config)


async def shutdown_all() -> None:
    """Shutdown all running services."""
    for service in Service.get_instances():
        await service.stop()
'''


@pytest.fixture
def syntax_error_code() -> str:
    """Code with syntax errors for error handling tests."""
    return '''def broken_function(
    """This function has a syntax error."""
    return "oops
'''


@pytest.fixture
def function_with_calls_code() -> str:
    """Function that calls other functions."""
    return '''def process_data(data: list) -> dict:
    """Process data through multiple steps."""
    validated = validate_input(data)
    transformed = transform_data(validated)
    result = aggregate_results(transformed)
    logger.info("Processing complete")
    return result
'''


# ---------------------------------------------------------------------------
# Parser Instance Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tree_sitter_parser():
    """Create a TreeSitterParser instance."""
    from core.parser.tree_sitter import TreeSitterParser

    return TreeSitterParser()


@pytest.fixture
def python_extractor():
    """Create a PythonExtractor instance."""
    from core.parser.languages.python import PythonExtractor

    return PythonExtractor()


# ---------------------------------------------------------------------------
# Temporary File Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_python_file(simple_function_code: str) -> Generator[Path, None, None]:
    """Create a temporary Python file with simple function code."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(simple_function_code)
        f.flush()
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_python_file_with_content():
    """Factory fixture to create temp files with custom content."""
    created_files: list[Path] = []

    def _create_file(content: str, suffix: str = ".py") -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(content)
            f.flush()
            temp_path = Path(f.name)
            created_files.append(temp_path)
            return temp_path

    yield _create_file

    # Cleanup all created files
    for path in created_files:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Async Test Support
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop_policy():
    """Provide an event loop policy for async tests."""
    import asyncio

    return asyncio.DefaultEventLoopPolicy()
