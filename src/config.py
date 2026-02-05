"""Configuration module for the Capstone application."""

from pathlib import Path
from urllib.parse import unquote, urlparse
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


def _parse_pg_url(url: str) -> dict[str, str | int]:
    """Parse a PostgreSQL URL into host, port, user, password, database."""
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise ValueError("Database URL must use postgresql:// or postgres://")
    port = parsed.port or 5432
    path = (parsed.path or "/").lstrip("/")
    database = path.split("?")[0] or "postgres"
    return {
        "host": parsed.hostname or "localhost",
        "port": port,
        "user": unquote(parsed.username) if parsed.username else "",
        "password": unquote(parsed.password) if parsed.password else "",
        "database": database,
    }


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Optional single connection string (e.g. Supabase); when set, overrides PG_* for connections
    DATABASE_URL: str | None = Field(
        default=None,
        description="Full PostgreSQL connection URL (e.g. Supabase). If set, overrides PG_* for connections.",
    )

    PG_HOST: str = Field(default="localhost", description="PostgreSQL host address")
    PG_PORT: int = Field(default=5432, description="PostgreSQL port number")
    PG_USER: str = Field(default="", description="PostgreSQL username")
    PG_PASSWORD: str = Field(default="", description="PostgreSQL user password")
    PG_DATABASE: str = Field(default="postgres", description="PostgreSQL database name")

    SUPABASE_URL: str | None = Field(default=None, description="Supabase project URL (e.g., https://<project-ref>.supabase.co)")
    SUPABASE_KEY: str | None = Field(default=None, description="Supabase API key")
    SUPABASE_BUCKET_NAME: str = Field(default="capstone-data-documents", description="Supabase storage bucket")

    # Vector Store Configuration
    VECTOR_TABLE_NAME: str = Field(
        default="documents", description="Name of the table to store document vectors"
    )

    EMBED_DIM: int = Field(
        default=768,
        description="Dimension of the embedding vectors (auto-detected from model)",
    )

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama API"
    )

    CHAT_MODEL: str = Field(
        default="gemma3:4b", description="Name of the chat model to use"
    )

    EMBEDDING_MODEL: str = Field(
        default="embeddinggemma", description="Name of the embedding model to use"
    )

    DATA_FOLDER: Path = BASE_DIR / "data"

    @property
    def database_url(self) -> str:
        """PostgreSQL connection URL (from APP_DATABASE_URL or built from PG_*)."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DATABASE}"

    @property
    def effective_pg_host(self) -> str:
        """Host for PGVectorStore (from DATABASE_URL if set, else PG_HOST)."""
        if self.DATABASE_URL:
            return _parse_pg_url(self.DATABASE_URL)["host"]
        return self.PG_HOST

    @property
    def effective_pg_port(self) -> int:
        """Port for PGVectorStore (from DATABASE_URL if set, else PG_PORT)."""
        if self.DATABASE_URL:
            return int(_parse_pg_url(self.DATABASE_URL)["port"])
        return self.PG_PORT

    @property
    def effective_pg_user(self) -> str:
        """User for PGVectorStore (from DATABASE_URL if set, else PG_USER)."""
        if self.DATABASE_URL:
            return _parse_pg_url(self.DATABASE_URL)["user"]
        return self.PG_USER

    @property
    def effective_pg_password(self) -> str:
        """Password for PGVectorStore (from DATABASE_URL if set, else PG_PASSWORD)."""
        if self.DATABASE_URL:
            return _parse_pg_url(self.DATABASE_URL)["password"]
        return self.PG_PASSWORD

    @property
    def effective_pg_database(self) -> str:
        """Database name for PGVectorStore (from DATABASE_URL if set, else PG_DATABASE)."""
        if self.DATABASE_URL:
            return _parse_pg_url(self.DATABASE_URL)["database"]
        return self.PG_DATABASE

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DATA_FOLDER")
    def validate_directories(cls, v):
        """Ensure that DATA_FOLDER is a valid Path object."""
        if not isinstance(v, Path):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

settings = Settings()