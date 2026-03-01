"""
Tests for configuration module (src/config.py).

Validates PostgreSQL URL parsing and Settings property behaviour.
"""

import pytest
from src.config import _parse_pg_url


class TestParsePgUrl:
    def test_valid_full_url(self):
        result = _parse_pg_url(
            "postgresql://user:pass@host.example.com:6543/mydb"
        )
        assert result["host"] == "host.example.com"
        assert result["port"] == 6543
        assert result["user"] == "user"
        assert result["password"] == "pass"
        assert result["database"] == "mydb"

    def test_postgres_scheme(self):
        result = _parse_pg_url("postgres://u:p@localhost/db")
        assert result["host"] == "localhost"
        assert result["database"] == "db"

    def test_default_port(self):
        result = _parse_pg_url("postgresql://u:p@localhost/db")
        assert result["port"] == 5432

    def test_url_with_query_params(self):
        result = _parse_pg_url(
            "postgresql://u:p@host:5432/db?sslmode=require"
        )
        assert result["database"] == "db"

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="postgresql://"):
            _parse_pg_url("mysql://user:pass@host/db")

    def test_encoded_password(self):
        result = _parse_pg_url("postgresql://user:p%40ssword@host/db")
        assert result["password"] == "p@ssword"

    def test_missing_path_defaults_to_postgres(self):
        result = _parse_pg_url("postgresql://u:p@host:5432/")
        assert result["database"] == "postgres"


class TestSettingsProperties:
    """Test that Settings correctly resolves DATABASE_URL vs PG_* fields."""

    def test_database_url_from_pg_fields(self, monkeypatch):
        # Must unset DATABASE_URL so it falls through to PG_* fields.
        # Also override _env_file to prevent reading the .env file on disk.
        monkeypatch.delenv("APP_DATABASE_URL", raising=False)
        monkeypatch.setenv("APP_PG_HOST", "myhost")
        monkeypatch.setenv("APP_PG_PORT", "5433")
        monkeypatch.setenv("APP_PG_USER", "admin")
        monkeypatch.setenv("APP_PG_PASSWORD", "secret")
        monkeypatch.setenv("APP_PG_DATABASE", "capstone")

        from src.config import Settings
        s = Settings(_env_file=None)  # skip .env file
        assert s.database_url == "postgresql://admin:secret@myhost:5433/capstone"
        assert s.effective_pg_host == "myhost"
        assert s.effective_pg_port == 5433

    def test_database_url_override(self, monkeypatch):
        monkeypatch.setenv(
            "APP_DATABASE_URL",
            "postgresql://cloud_user:cloud_pass@cloud.host:6543/clouddb",
        )
        from src.config import Settings
        s = Settings(_env_file=None)  # skip .env file
        assert s.effective_pg_host == "cloud.host"
        assert s.effective_pg_port == 6543
        assert s.effective_pg_user == "cloud_user"
        assert s.effective_pg_database == "clouddb"
