"""
Migration 0002: SQL-backed synthetic demo fleet (`demo_vehicles` and `demo_assignments`)
with lifecycle timestamps, simulated coordinates, and active-assignment uniqueness guards.
"""

from typing import Any

from backend.app.db.pool import (
    DEMO_FLEET_SCHEMA_SQL,
    DEMO_VEHICLES_SEED,
    NeonPostgresStore,
)


revision: str = "0002_demo_fleet_and_lifecycle"
down_revision: str | None = "0001_initial_setu_schema"


def get_upgrade_sql() -> str:
    """Return the idempotent PostgreSQL DDL for demo_vehicles and demo_assignments."""
    return DEMO_FLEET_SCHEMA_SQL


def apply_migration(store: NeonPostgresStore | None = None) -> dict[str, Any]:
    """Idempotently create demo_vehicles + demo_assignments and seed the 9 demo vehicles."""
    pg = store or NeonPostgresStore()
    return pg.ensure_demo_fleet_schema_and_seed()


__all__ = [
    "DEMO_VEHICLES_SEED",
    "apply_migration",
    "down_revision",
    "get_upgrade_sql",
    "revision",
]
