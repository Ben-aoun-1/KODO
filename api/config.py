"""Settings management for the Kodo API.

This module provides centralized configuration management using pydantic-settings.
All configuration is loaded from environment variables with sensible defaults.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        app_name: Name of the application.
        app_version: Application version.
        debug: Enable debug mode.
        log_level: Logging level.

        neo4j_uri: Neo4j connection URI.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.

        qdrant_url: Qdrant server URL.
        qdrant_api_key: Optional Qdrant API key.

        postgres_url: PostgreSQL connection URL.

        anthropic_api_key: Anthropic API key for Claude.
        voyage_api_key: Voyage AI API key for embeddings.
        openai_api_key: OpenAI API key (alternative for embeddings).

        github_app_id: GitHub App ID for integrations.
        github_private_key: GitHub App private key.

        cors_origins: Allowed CORS origins.
        api_prefix: API route prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="Kodo API", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Neo4j settings
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")

    # Qdrant settings
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional Qdrant API key",
    )

    # PostgreSQL settings
    postgres_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL",
    )

    # LLM API keys
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude",
    )
    voyage_api_key: str | None = Field(
        default=None,
        description="Voyage AI API key for embeddings",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (alternative for embeddings)",
    )

    # GitHub integration
    github_app_id: str | None = Field(
        default=None,
        description="GitHub App ID",
    )
    github_private_key: str | None = Field(
        default=None,
        description="GitHub App private key",
    )

    # CORS settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )

    # API settings
    api_prefix: str = Field(default="/api/v1", description="API route prefix")

    # Query engine settings
    max_context_tokens: int = Field(
        default=8000,
        description="Maximum tokens for code context",
    )
    max_response_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM response",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Uses lru_cache to ensure settings are loaded once and reused.

    Returns:
        The application settings instance.
    """
    return Settings()
