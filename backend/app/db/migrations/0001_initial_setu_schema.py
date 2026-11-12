"""
Initial Alembic migration 0001: PostgreSQL 16 + PostGIS + pg_trgm schema and
append-only audit triggers (Section 6 & Section 21).
"""

from backend.app.db.pool import POSTGRES_APPEND_ONLY_SCHEMA_SQL


revision: str = "0001_initial_setu_schema"
down_revision: str | None = None


def get_upgrade_sql() -> str:
    """Return the idempotent PostgreSQL 16 + PostGIS + append-only trigger DDL."""
    return POSTGRES_APPEND_ONLY_SCHEMA_SQL
