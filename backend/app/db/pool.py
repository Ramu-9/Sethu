"""
PostgreSQL 16 + PostGIS + pg_trgm connection pool configuration,
atomic resource locking queries (FOR UPDATE SKIP LOCKED),
database-enforced append-only audit DDL, and non-destructive Neon PostgreSQL
persistence store (Sections 6, 8, 10.3, 12, 13, 14, 20, and 21.2).
"""

import json
import os
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from uuid import UUID, uuid4, uuid5

from backend.app.config import get_runtime_env_map


DATABASE_URL_ENV = "DATABASE_URL"
MIGRATION_DATABASE_URL_ENV = "MIGRATION_DATABASE_URL"

# Atomic nearest-available synthetic resource allocation query (Section 10.3)
SELECT_NEAREST_AVAILABLE_RESOURCE_FOR_UPDATE_SQL: str = """
SELECT
    resource_id,
    category,
    callsign,
    ST_Y(location_geom) AS latitude,
    ST_X(location_geom) AS longitude,
    ST_DistanceSphere(
        location_geom,
        ST_SetSRID(ST_MakePoint(:caller_lon, :caller_lat), 4326)
    ) AS distance_meters
FROM synthetic_resources
WHERE category = :required_category
  AND is_available = TRUE
ORDER BY location_geom <-> ST_SetSRID(ST_MakePoint(:caller_lon, :caller_lat), 4326) ASC
LIMIT 1
FOR UPDATE SKIP LOCKED;
"""

# Duplicate incident spatio-temporal detection query within 200m and 5 minutes (Section 14)
FIND_DUPLICATE_INCIDENT_WITHIN_200M_5MIN_SQL: str = """
SELECT
    incident_id,
    ST_DistanceSphere(
        location_geom,
        ST_SetSRID(ST_MakePoint(:caller_lon, :caller_lat), 4326)
    ) AS distance_meters
FROM incidents
WHERE incident_id != :current_incident_id
  AND category = :category
  AND category != 'unknown'
  AND location_confirmation_status = 'confirmed'
  AND created_at >= (:reference_time - INTERVAL '5 minutes')
  AND ST_DWithin(
        location_geom::geography,
        ST_SetSRID(ST_MakePoint(:caller_lon, :caller_lat), 4326)::geography,
        200.0
      )
ORDER BY created_at DESC
LIMIT 1;
"""

# Complete PostgreSQL + PostGIS + pg_trgm DDL enforcing Sections 6, 8, 10, 12, 13, 20, and 21.2
POSTGRES_APPEND_ONLY_SCHEMA_SQL: str = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Authoritative Sessions Table (Section 8)
CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    incident_id UUID NOT NULL UNIQUE,
    session_state VARCHAR(32) NOT NULL,
    incident_state VARCHAR(32) NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    last_event_sequence INTEGER NOT NULL DEFAULT 1,
    initial_language VARCHAR(8) NOT NULL,
    active_language VARCHAR(8) NOT NULL,
    consecutive_unknown_count INTEGER NOT NULL DEFAULT 0,
    consecutive_provider_blocks INTEGER NOT NULL DEFAULT 0,
    unknown_fallback_applied BOOLEAN NOT NULL DEFAULT FALSE,
    control_room_review_required BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Incidents Table with PostGIS Geometry (Sections 9, 11, 12, 14)
CREATE TABLE IF NOT EXISTS incidents (
    incident_id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(session_id),
    category VARCHAR(32) NOT NULL DEFAULT 'unknown',
    category_established_above_threshold BOOLEAN NOT NULL DEFAULT FALSE,
    location_geom GEOMETRY(Point, 4326),
    accuracy_m DOUBLE PRECISION,
    location_source VARCHAR(32),
    location_confirmation_status VARCHAR(64) NOT NULL DEFAULT 'unresolved',
    possible_duplicate_flag BOOLEAN NOT NULL DEFAULT FALSE,
    matched_duplicate_incident_id UUID,
    prank_review_status VARCHAR(64) NOT NULL DEFAULT 'not_reviewed',
    review_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incidents_location_gist
    ON incidents USING GIST (location_geom);

-- 3. Finalized Utterances Table (Rule 6 & Section 9.2)
CREATE TABLE IF NOT EXISTS utterances (
    utterance_id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(session_id),
    sequence_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    language VARCHAR(8),
    utterance_role VARCHAR(64),
    finalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sequence_number)
);

-- 4. Validated Provenanced Facts Table (Rule 3 & Section 9.2)
CREATE TABLE IF NOT EXISTS validated_facts (
    fact_id UUID PRIMARY KEY,
    incident_id UUID NOT NULL REFERENCES incidents(incident_id),
    field_name VARCHAR(128) NOT NULL,
    field_value JSONB NOT NULL,
    source_utterance_id UUID NOT NULL REFERENCES utterances(utterance_id),
    evidence_text TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Synthetic Emergency Resources Table (Section 10)
CREATE TABLE IF NOT EXISTS synthetic_resources (
    resource_id VARCHAR(64) PRIMARY KEY,
    category VARCHAR(64) NOT NULL,
    callsign VARCHAR(128) NOT NULL,
    location_geom GEOMETRY(Point, 4326) NOT NULL,
    is_available BOOLEAN NOT NULL DEFAULT TRUE,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_synthetic_resources_geom_gist
    ON synthetic_resources USING GIST (location_geom);

-- 6. Simulated Resource Assignments Table (Section 10.3 & 10.4)
CREATE TABLE IF NOT EXISTS simulated_assignments (
    assignment_id UUID PRIMARY KEY,
    incident_id UUID NOT NULL REFERENCES incidents(incident_id),
    idempotency_key VARCHAR(256) NOT NULL UNIQUE,
    resource_id VARCHAR(64) REFERENCES synthetic_resources(resource_id),
    resource_callsign VARCHAR(128),
    resource_category VARCHAR(64) NOT NULL,
    status VARCHAR(64) NOT NULL,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    simulated_eta_minutes INTEGER,
    distance_meters DOUBLE PRECISION,
    provenance JSONB NOT NULL,
    is_simulated BOOLEAN NOT NULL DEFAULT TRUE
);

-- 7. Synthetic Landmarks Table with pg_trgm Fuzzy Search Index (Section 12.2 & Gate 0.7b)
CREATE TABLE IF NOT EXISTS synthetic_landmarks (
    landmark_id VARCHAR(64) PRIMARY KEY,
    name_romanised TEXT NOT NULL,
    name_native TEXT NOT NULL,
    language VARCHAR(8) NOT NULL,
    city VARCHAR(128) NOT NULL,
    location_geom GEOMETRY(Point, 4326) NOT NULL,
    source_identifier VARCHAR(128) NOT NULL,
    source_licence TEXT NOT NULL,
    ingestion_date DATE NOT NULL,
    is_synthetic BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_synthetic_landmarks_romanised_trgm
    ON synthetic_landmarks USING GIN (name_romanised gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_synthetic_landmarks_native_trgm
    ON synthetic_landmarks USING GIN (name_native gin_trgm_ops);

-- 8. Synthetic Hospital Directory Table (Section 13 & Gate 0.7)
CREATE TABLE IF NOT EXISTS synthetic_hospitals (
    hospital_id VARCHAR(64) PRIMARY KEY,
    name TEXT NOT NULL,
    location_geom GEOMETRY(Point, 4326) NOT NULL,
    address TEXT NOT NULL,
    capabilities JSONB NOT NULL,
    source_identifier VARCHAR(128) NOT NULL,
    verification_date DATE NOT NULL,
    is_synthetic_directory BOOLEAN NOT NULL DEFAULT TRUE
);

-- 9. Structured Incident Handoffs Table (Section 20)
CREATE TABLE IF NOT EXISTS incident_handoffs (
    handoff_id UUID PRIMARY KEY,
    incident_id UUID NOT NULL REFERENCES incidents(incident_id),
    session_id UUID NOT NULL REFERENCES sessions(session_id),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transcript_retained BOOLEAN NOT NULL DEFAULT TRUE,
    handoff_payload JSONB NOT NULL
);

-- 10. Immutable Append-Only Audit Events Table & Triggers (Section 21.2)
CREATE TABLE IF NOT EXISTS audit_events (
    event_id UUID PRIMARY KEY,
    session_id UUID NOT NULL,
    incident_id UUID NOT NULL,
    event_type VARCHAR(128) NOT NULL,
    severity VARCHAR(16) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    outcome TEXT NOT NULL,
    error_code VARCHAR(64),
    reason_code VARCHAR(64),
    provenance_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE FUNCTION forbid_audit_events_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'Append-only violation: % on audit_events is strictly prohibited (Section 21.2)', TG_OP;
END;
$$;

DROP TRIGGER IF EXISTS trg_forbid_audit_update_delete ON audit_events;
CREATE TRIGGER trg_forbid_audit_update_delete
BEFORE UPDATE OR DELETE ON audit_events
FOR EACH ROW EXECUTE FUNCTION forbid_audit_events_mutation();

DROP TRIGGER IF EXISTS trg_forbid_audit_truncate ON audit_events;
CREATE TRIGGER trg_forbid_audit_truncate
BEFORE TRUNCATE ON audit_events
FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_events_mutation();
"""

DEMO_FLEET_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS demo_vehicles (
        vehicle_id VARCHAR(32) PRIMARY KEY,
        vehicle_code VARCHAR(32) NOT NULL UNIQUE,
        category VARCHAR(32) NOT NULL CHECK (category IN ('ambulance', 'fire_engine', 'police_patrol')),
        callsign VARCHAR(128) NOT NULL,
        status VARCHAR(32) NOT NULL DEFAULT 'AVAILABLE' CHECK (status IN ('AVAILABLE', 'ASSIGNED', 'EN_ROUTE', 'ARRIVED')),
        current_incident_id UUID,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        home_latitude DOUBLE PRECISION NOT NULL,
        home_longitude DOUBLE PRECISION NOT NULL,
        is_simulated BOOLEAN NOT NULL DEFAULT TRUE CHECK (is_simulated = TRUE),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS demo_assignments (
        assignment_id UUID PRIMARY KEY,
        incident_id UUID NOT NULL,
        session_id UUID,
        idempotency_key VARCHAR(128) NOT NULL UNIQUE,
        vehicle_id VARCHAR(32) REFERENCES demo_vehicles(vehicle_id),
        vehicle_code VARCHAR(32),
        resource_category VARCHAR(64) NOT NULL,
        assignment_status VARCHAR(32) NOT NULL CHECK (assignment_status IN ('ASSIGNED', 'EN_ROUTE', 'ARRIVED', 'UNAVAILABLE')),
        assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        en_route_at TIMESTAMPTZ,
        arrived_at TIMESTAMPTZ,
        released_at TIMESTAMPTZ,
        simulated_eta_minutes INTEGER,
        current_latitude DOUBLE PRECISION,
        current_longitude DOUBLE PRECISION,
        target_latitude DOUBLE PRECISION,
        target_longitude DOUBLE PRECISION,
        allocation_mode VARCHAR(16) NOT NULL DEFAULT 'DEMO',
        is_simulated BOOLEAN NOT NULL DEFAULT TRUE CHECK (is_simulated = TRUE),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_assignments_inc_veh_active
    ON demo_assignments (incident_id, vehicle_id)
    WHERE vehicle_id IS NOT NULL AND released_at IS NULL;
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_demo_assignments_vehicle_active
    ON demo_assignments (vehicle_id)
    WHERE vehicle_id IS NOT NULL AND released_at IS NULL;
    """,
)

DEMO_FLEET_SCHEMA_SQL: str = "\n\n".join(stmt.strip() for stmt in DEMO_FLEET_DDL_STATEMENTS)

DEMO_VEHICLES_SEED: tuple[tuple[str, str, str, str, float, float], ...] = (
    ("AMB-01", "AMB-01", "ambulance", "DEMO-Ambulance AMB-01 (Madhapur)", 17.4455, 78.3795),
    ("AMB-02", "AMB-02", "ambulance", "DEMO-Ambulance AMB-02 (Secunderabad)", 17.4350, 78.5000),
    ("AMB-03", "AMB-03", "ambulance", "DEMO-Ambulance AMB-03 (Banjara Hills)", 17.4195, 78.4480),
    ("FIR-01", "FIR-01", "fire_engine", "DEMO-Fire-Engine FIR-01 (Hitech City)", 17.4480, 78.3820),
    ("FIR-02", "FIR-02", "fire_engine", "DEMO-Fire-Engine FIR-02 (Jubilee Hills)", 17.4320, 78.4070),
    ("FIR-03", "FIR-03", "fire_engine", "DEMO-Fire-Engine FIR-03 (Charminar)", 17.3750, 78.4760),
    ("POL-01", "POL-01", "police_patrol", "DEMO-Police-Patrol POL-01 (Cyberabad)", 17.4440, 78.3780),
    ("POL-02", "POL-02", "police_patrol", "DEMO-Police-Patrol POL-02 (Begumpet)", 17.4290, 78.4610),
    ("POL-03", "POL-03", "police_patrol", "DEMO-Police-Patrol POL-03 (Abids)", 17.3920, 78.4890),
)


def get_database_url() -> str:
    """Return configured PostgreSQL asyncpg runtime connection string (least-privilege role)."""
    env = get_runtime_env_map()
    return env.get(
        DATABASE_URL_ENV,
        os.environ.get(
            DATABASE_URL_ENV,
            "postgresql+asyncpg://setu_runtime:setu_runtime@localhost:5432/setu",
        ),
    )


def get_migration_database_url() -> str:
    """Return configured PostgreSQL migration/owner connection string (Section 21.2)."""
    env = get_runtime_env_map()
    return env.get(
        MIGRATION_DATABASE_URL_ENV,
        os.environ.get(
            MIGRATION_DATABASE_URL_ENV,
            "postgresql+asyncpg://setu_owner:setu_owner_dev_password@localhost:5432/setu",
        ),
    )


def _normalize_pg_connection_string(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").strip()


class NeonPostgresStore:
    """
    Non-destructive PostgreSQL persistence and schema manager using the configured
    DATABASE_URL and MIGRATION_DATABASE_URL over TLS.
    """

    def __init__(
        self,
        *,
        runtime_url: str | None = None,
        migration_url: str | None = None,
    ) -> None:
        self._runtime_url = _normalize_pg_connection_string(runtime_url or get_database_url())
        self._migration_url = _normalize_pg_connection_string(
            migration_url or get_migration_database_url()
        )
        parsed = urllib.parse.urlsplit(self._runtime_url)
        self._host = parsed.hostname or ""
        self._is_remote_neon = bool(self._host and self._host != "localhost")

    @property
    def is_configured(self) -> bool:
        return self._is_remote_neon

    def execute_sql(
        self,
        query: str,
        params: list[Any] | None = None,
        *,
        use_migration_role: bool = False,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Execute a parameterized SQL statement against the configured PostgreSQL database."""
        conn_str = self._migration_url if use_migration_role else self._runtime_url
        parsed = urllib.parse.urlsplit(conn_str)
        host = parsed.hostname
        if not host or host == "localhost":
            raise RuntimeError("No remote PostgreSQL host configured in DATABASE_URL.")

        endpoint = f"https://{host}/sql"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({"query": query, "params": params or []}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Neon-Connection-String": conn_str,
                "User-Agent": "setu-postgres-client/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=max(1.0, timeout_seconds)) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"PostgreSQL query error ({exc.code}): {err_body}") from exc

    def ensure_schema_and_seed_data(self) -> dict[str, Any]:
        """
        Idempotently apply 0001_initial_setu_schema.py DDL (CREATE TABLE IF NOT EXISTS
        and append-only audit triggers) and insert default synthetic resources, landmarks,
        and hospitals using ON CONFLICT DO NOTHING without modifying or deleting existing data.
        """
        statements = [
            "CREATE EXTENSION IF NOT EXISTS postgis;",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id UUID PRIMARY KEY,
                incident_id UUID NOT NULL UNIQUE,
                session_state VARCHAR(32) NOT NULL,
                incident_state VARCHAR(32) NOT NULL,
                revision INTEGER NOT NULL DEFAULT 1,
                last_event_sequence INTEGER NOT NULL DEFAULT 1,
                initial_language VARCHAR(8) NOT NULL,
                active_language VARCHAR(8) NOT NULL,
                consecutive_unknown_count INTEGER NOT NULL DEFAULT 0,
                consecutive_provider_blocks INTEGER NOT NULL DEFAULT 0,
                unknown_fallback_applied BOOLEAN NOT NULL DEFAULT FALSE,
                control_room_review_required BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id UUID PRIMARY KEY,
                session_id UUID NOT NULL REFERENCES sessions(session_id),
                category VARCHAR(32) NOT NULL DEFAULT 'unknown',
                category_established_above_threshold BOOLEAN NOT NULL DEFAULT FALSE,
                location_geom GEOMETRY(Point, 4326),
                accuracy_m DOUBLE PRECISION,
                location_source VARCHAR(32),
                location_confirmation_status VARCHAR(64) NOT NULL DEFAULT 'unresolved',
                possible_duplicate_flag BOOLEAN NOT NULL DEFAULT FALSE,
                matched_duplicate_incident_id UUID,
                prank_review_status VARCHAR(64) NOT NULL DEFAULT 'not_reviewed',
                review_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_incidents_location_gist ON incidents USING GIST (location_geom);",
            """
            CREATE TABLE IF NOT EXISTS utterances (
                utterance_id UUID PRIMARY KEY,
                session_id UUID NOT NULL REFERENCES sessions(session_id),
                sequence_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                language VARCHAR(8),
                utterance_role VARCHAR(64),
                finalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (session_id, sequence_number)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS validated_facts (
                fact_id UUID PRIMARY KEY,
                incident_id UUID NOT NULL REFERENCES incidents(incident_id),
                field_name VARCHAR(128) NOT NULL,
                field_value JSONB NOT NULL,
                source_utterance_id UUID NOT NULL REFERENCES utterances(utterance_id),
                evidence_text TEXT NOT NULL,
                recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS synthetic_resources (
                resource_id VARCHAR(64) PRIMARY KEY,
                category VARCHAR(64) NOT NULL,
                callsign VARCHAR(128) NOT NULL,
                location_geom GEOMETRY(Point, 4326) NOT NULL,
                is_available BOOLEAN NOT NULL DEFAULT TRUE,
                is_simulated BOOLEAN NOT NULL DEFAULT TRUE
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_synthetic_resources_geom_gist ON synthetic_resources USING GIST (location_geom);",
            """
            CREATE TABLE IF NOT EXISTS simulated_assignments (
                assignment_id UUID PRIMARY KEY,
                incident_id UUID NOT NULL REFERENCES incidents(incident_id),
                idempotency_key VARCHAR(256) NOT NULL UNIQUE,
                resource_id VARCHAR(64) REFERENCES synthetic_resources(resource_id),
                resource_callsign VARCHAR(128),
                resource_category VARCHAR(64) NOT NULL,
                status VARCHAR(64) NOT NULL,
                assigned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                simulated_eta_minutes INTEGER,
                distance_meters DOUBLE PRECISION,
                provenance JSONB NOT NULL,
                is_simulated BOOLEAN NOT NULL DEFAULT TRUE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS synthetic_landmarks (
                landmark_id VARCHAR(64) PRIMARY KEY,
                name_romanised TEXT NOT NULL,
                name_native TEXT NOT NULL,
                language VARCHAR(8) NOT NULL,
                city VARCHAR(128) NOT NULL,
                location_geom GEOMETRY(Point, 4326) NOT NULL,
                source_identifier VARCHAR(128) NOT NULL,
                source_licence TEXT NOT NULL,
                ingestion_date DATE NOT NULL,
                is_synthetic BOOLEAN NOT NULL DEFAULT TRUE
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_synthetic_landmarks_romanised_trgm ON synthetic_landmarks USING GIN (name_romanised gin_trgm_ops);",
            "CREATE INDEX IF NOT EXISTS idx_synthetic_landmarks_native_trgm ON synthetic_landmarks USING GIN (name_native gin_trgm_ops);",
            """
            CREATE TABLE IF NOT EXISTS synthetic_hospitals (
                hospital_id VARCHAR(64) PRIMARY KEY,
                name TEXT NOT NULL,
                location_geom GEOMETRY(Point, 4326) NOT NULL,
                address TEXT NOT NULL,
                capabilities JSONB NOT NULL,
                source_identifier VARCHAR(128) NOT NULL,
                verification_date DATE NOT NULL,
                is_synthetic_directory BOOLEAN NOT NULL DEFAULT TRUE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS incident_handoffs (
                handoff_id UUID PRIMARY KEY,
                incident_id UUID NOT NULL REFERENCES incidents(incident_id),
                session_id UUID NOT NULL REFERENCES sessions(session_id),
                generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                transcript_retained BOOLEAN NOT NULL DEFAULT TRUE,
                handoff_payload JSONB NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id UUID PRIMARY KEY,
                session_id UUID NOT NULL,
                incident_id UUID NOT NULL,
                event_type VARCHAR(128) NOT NULL,
                severity VARCHAR(16) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                outcome TEXT NOT NULL,
                error_code VARCHAR(64),
                reason_code VARCHAR(64),
                provenance_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                details JSONB NOT NULL DEFAULT '{}'::jsonb
            );
            """,
            """
            CREATE OR REPLACE FUNCTION forbid_audit_events_mutation()
            RETURNS TRIGGER
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'Append-only violation: % on audit_events is strictly prohibited (Section 21.2)', TG_OP;
            END;
            $$;
            """,
            "DROP TRIGGER IF EXISTS trg_forbid_audit_update_delete ON audit_events;",
            """
            CREATE TRIGGER trg_forbid_audit_update_delete
            BEFORE UPDATE OR DELETE ON audit_events
            FOR EACH ROW EXECUTE FUNCTION forbid_audit_events_mutation();
            """,
            "DROP TRIGGER IF EXISTS trg_forbid_audit_truncate ON audit_events;",
            """
            CREATE TRIGGER trg_forbid_audit_truncate
            BEFORE TRUNCATE ON audit_events
            FOR EACH STATEMENT EXECUTE FUNCTION forbid_audit_events_mutation();
            """,
        ]

        for stmt in statements:
            self.execute_sql(stmt, use_migration_role=True)

        # Non-destructively seed synthetic resources (ON CONFLICT DO NOTHING)
        resources = [
            ("SIM-AMB-HYD-101", "ambulance", "SIM-Ambulance-Hitech-101", 78.3795, 17.4455),
            ("SIM-AMB-HYD-102", "ambulance", "SIM-Ambulance-Secunderabad-102", 78.5000, 17.4350),
            ("SIM-FIRE-HYD-201", "fire_engine", "SIM-FireEngine-Madhapur-201", 78.3820, 17.4480),
            ("SIM-FRESC-HYD-301", "fire_and_rescue", "SIM-FireRescue-Jubilee-301", 78.4070, 17.4320),
            ("SIM-POL-HYD-401", "police_patrol", "SIM-PolicePatrol-Cyberabad-401", 78.3780, 17.4440),
        ]
        for res_id, cat, callsign, lon, lat in resources:
            self.execute_sql(
                """
                INSERT INTO synthetic_resources (resource_id, category, callsign, location_geom, is_available, is_simulated)
                VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($4::double precision, $5::double precision), 4326), TRUE, TRUE)
                ON CONFLICT (resource_id) DO NOTHING;
                """,
                [res_id, cat, callsign, lon, lat],
            )

        # Non-destructively seed synthetic hospital directory records (ON CONFLICT DO NOTHING)
        hospitals = [
            (
                "HOSP-HYD-01",
                "Hitech Metro Care Hospital (Synthetic Directory)",
                78.3810,
                17.4460,
                "Hitech City Main Rd, Madhapur, Hyderabad",
                json.dumps(["emergency_department", "trauma", "burn_care", "cardiac"]),
                "setu-synthetic-hospital-directory-v1",
                "2026-11-01",
            ),
            (
                "HOSP-HYD-02",
                "NIMS Emergency & Trauma Centre (Synthetic Directory)",
                78.4522,
                17.4225,
                "Punjagutta, Hyderabad",
                json.dumps(["emergency_department", "trauma", "neurosurgery"]),
                "setu-synthetic-hospital-directory-v1",
                "2026-11-01",
            ),
            (
                "HOSP-HYD-03",
                "Secunderabad General Emergency Facility (Synthetic Directory)",
                78.5020,
                17.4400,
                "Station Rd, Secunderabad",
                json.dumps(["emergency_department", "general_medicine"]),
                "setu-synthetic-hospital-directory-v1",
                "2026-11-01",
            ),
        ]
        for h_id, name, lon, lat, addr, caps_json, src_id, ver_date in hospitals:
            self.execute_sql(
                """
                INSERT INTO synthetic_hospitals (
                    hospital_id, name, location_geom, address, capabilities,
                    source_identifier, verification_date, is_synthetic_directory
                )
                VALUES (
                    $1, $2, ST_SetSRID(ST_MakePoint($3::double precision, $4::double precision), 4326),
                    $5, $6::jsonb, $7, $8::date, TRUE
                )
                ON CONFLICT (hospital_id) DO NOTHING;
                """,
                [h_id, name, lon, lat, addr, caps_json, src_id, ver_date],
            )

        # Keep demo fleet tables & 9 synthetic demo vehicles initialized idempotently
        self.ensure_demo_fleet_schema_and_seed()

        tables_res = self.execute_sql(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
        )
        return {
            "tables": [r["table_name"] for r in tables_res.get("rows", [])],
        }

    def ensure_demo_fleet_schema_and_seed(self) -> dict[str, Any]:
        """
        Idempotently apply 0002_demo_fleet_and_lifecycle DDL and seed the 9 demo vehicles
        (AMB-01..03, FIR-01..03, POL-01..03) without altering existing production or incident data.
        """
        if not self._is_remote_neon:
            return {"seeded": 0, "tables": []}

        for stmt in DEMO_FLEET_DDL_STATEMENTS:
            self.execute_sql(stmt.strip(), use_migration_role=True)

        for v_id, v_code, category, callsign, lat, lon in DEMO_VEHICLES_SEED:
            self.execute_sql(
                """
                INSERT INTO demo_vehicles (
                    vehicle_id, vehicle_code, category, callsign, status,
                    latitude, longitude, home_latitude, home_longitude, is_simulated
                )
                VALUES ($1, $2, $3, $4, 'AVAILABLE', $5::double precision, $6::double precision, $5::double precision, $6::double precision, TRUE)
                ON CONFLICT (vehicle_id) DO NOTHING;
                """,
                [v_id, v_code, category, callsign, lat, lon],
                use_migration_role=True,
            )

        self.release_stale_demo_assignments()
        rows = self.list_demo_vehicles()
        return {"seeded": len(rows), "vehicles": rows}

    def release_stale_demo_assignments(self, max_idle_minutes: int = 10) -> None:
        """
        Release active demo assignments belonging to completed or abandoned sessions so a
        restarted process does not hit unique active-vehicle conflicts against dead sessions.
        """
        if not self._is_remote_neon:
            return
        self.execute_sql(
            """
            UPDATE demo_assignments da
            SET released_at = NOW(),
                updated_at = NOW()
            WHERE da.released_at IS NULL
              AND (
                  EXISTS (
                      SELECT 1 FROM sessions s
                      WHERE s.session_id = da.session_id
                        AND (
                            s.session_state = 'completed'
                            OR s.updated_at < NOW() - ($1::int * INTERVAL '1 minute')
                        )
                  )
                  OR (
                      NOT EXISTS (
                          SELECT 1 FROM sessions s
                          WHERE s.session_id = da.session_id
                      )
                      AND da.updated_at < NOW() - ($1::int * INTERVAL '1 minute')
                  )
              );
            """,
            [max(1, int(max_idle_minutes))],
        )
        self.execute_sql(
            """
            UPDATE demo_vehicles dv
            SET status = 'AVAILABLE',
                current_incident_id = NULL,
                latitude = dv.home_latitude,
                longitude = dv.home_longitude,
                updated_at = NOW()
            WHERE dv.status <> 'AVAILABLE'
              AND NOT EXISTS (
                  SELECT 1 FROM demo_assignments da
                  WHERE da.vehicle_id = dv.vehicle_id
                    AND da.released_at IS NULL
              );
            """
        )

    def list_demo_vehicles(self) -> list[dict[str, Any]]:
        """Return ordered synthetic demo vehicles from SQL."""
        if not self._is_remote_neon:
            return []
        res = self.execute_sql(
            """
            SELECT vehicle_id, vehicle_code, category, callsign, status,
                   current_incident_id, latitude, longitude, home_latitude,
                   home_longitude, is_simulated, updated_at
            FROM demo_vehicles
            ORDER BY vehicle_code ASC;
            """
        )
        return list(res.get("rows", []))

    def list_demo_assignments(self, incident_id: str | None = None) -> list[dict[str, Any]]:
        """Return synthetic demo assignments from SQL (optionally filtered by incident_id)."""
        if not self._is_remote_neon:
            return []
        if incident_id:
            res = self.execute_sql(
                """
                SELECT assignment_id, incident_id, session_id, idempotency_key,
                       vehicle_id, vehicle_code, resource_category, assignment_status,
                       assigned_at, en_route_at, arrived_at, released_at,
                       simulated_eta_minutes, current_latitude, current_longitude,
                       target_latitude, target_longitude, allocation_mode, is_simulated
                FROM demo_assignments
                WHERE incident_id = $1::uuid
                ORDER BY assigned_at ASC;
                """,
                [incident_id],
            )
        else:
            res = self.execute_sql(
                """
                SELECT assignment_id, incident_id, session_id, idempotency_key,
                       vehicle_id, vehicle_code, resource_category, assignment_status,
                       assigned_at, en_route_at, arrived_at, released_at,
                       simulated_eta_minutes, current_latitude, current_longitude,
                       target_latitude, target_longitude, allocation_mode, is_simulated
                FROM demo_assignments
                ORDER BY assigned_at DESC
                LIMIT 100;
                """
            )
        return list(res.get("rows", []))

    def release_demo_incident_vehicles(self, incident_id: str) -> None:
        """
        Mark active demo assignments for a closed session/incident as released and restore
        their synthetic vehicles to AVAILABLE at depot coordinates (preserving assignment history).
        """
        if not self._is_remote_neon:
            return
        self.execute_sql(
            """
            UPDATE demo_assignments
            SET released_at = NOW(), updated_at = NOW()
            WHERE incident_id = $1::uuid AND released_at IS NULL;
            """,
            [incident_id],
        )
        self.execute_sql(
            """
            UPDATE demo_vehicles
            SET status = 'AVAILABLE',
                current_incident_id = NULL,
                latitude = home_latitude,
                longitude = home_longitude,
                updated_at = NOW()
            WHERE current_incident_id = $1::uuid;
            """,
            [incident_id],
        )

    def persist_demo_fleet_state(self, snapshot: dict[str, Any]) -> None:
        """Synchronize demo assignments and demo_vehicles lifecycle/coordinates to SQL."""
        if not self._is_remote_neon:
            return
        incident_id = str(snapshot["incident_id"])
        session_id = str(snapshot["session_id"])
        is_completed = str(snapshot.get("session_state", "")).lower() == "completed"

        for asgn in snapshot.get("assignments") or []:
            if str(asgn.get("allocation_mode", "DEMO")) != "DEMO" and not (
                asgn.get("vehicle_code") or str(asgn.get("resource_id") or "").startswith(("AMB-", "FIR-", "POL-"))
            ):
                continue
            vehicle_id = asgn.get("resource_id")
            vehicle_code = asgn.get("vehicle_code") or vehicle_id
            lifecycle = str(asgn.get("lifecycle_status") or "ASSIGNED")
            if (
                str(asgn.get("status") or "") == "pending_location"
                or not vehicle_id
                or lifecycle not in ("ASSIGNED", "EN_ROUTE", "ARRIVED", "UNAVAILABLE")
            ):
                continue
            self.execute_sql(
                """
                INSERT INTO demo_assignments (
                    assignment_id, incident_id, session_id, idempotency_key,
                    vehicle_id, vehicle_code, resource_category, assignment_status,
                    assigned_at, en_route_at, arrived_at, released_at,
                    simulated_eta_minutes, current_latitude, current_longitude,
                    target_latitude, target_longitude, allocation_mode, is_simulated, updated_at
                )
                VALUES (
                    $1::uuid, $2::uuid, $3::uuid, $4,
                    $5, $6, $7, $8,
                    COALESCE($9::timestamptz, NOW()), $10::timestamptz, $11::timestamptz,
                    CASE WHEN $12::boolean THEN NOW() ELSE NULL END,
                    $13::int, $14::double precision, $15::double precision,
                    $16::double precision, $17::double precision, 'DEMO', TRUE, NOW()
                )
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    vehicle_id = EXCLUDED.vehicle_id,
                    vehicle_code = EXCLUDED.vehicle_code,
                    assignment_status = EXCLUDED.assignment_status,
                    en_route_at = COALESCE(EXCLUDED.en_route_at, demo_assignments.en_route_at),
                    arrived_at = COALESCE(EXCLUDED.arrived_at, demo_assignments.arrived_at),
                    released_at = COALESCE(EXCLUDED.released_at, demo_assignments.released_at),
                    simulated_eta_minutes = EXCLUDED.simulated_eta_minutes,
                    current_latitude = EXCLUDED.current_latitude,
                    current_longitude = EXCLUDED.current_longitude,
                    target_latitude = EXCLUDED.target_latitude,
                    target_longitude = EXCLUDED.target_longitude,
                    updated_at = NOW();
                """,
                [
                    str(asgn["assignment_id"]),
                    incident_id,
                    session_id,
                    str(asgn["idempotency_key"]),
                    vehicle_id,
                    vehicle_code,
                    str(asgn["resource_category"]),
                    lifecycle,
                    asgn.get("assigned_at"),
                    asgn.get("en_route_at"),
                    asgn.get("arrived_at"),
                    is_completed,
                    int(asgn["simulated_eta_minutes"]) if asgn.get("simulated_eta_minutes") is not None else None,
                    float(asgn["current_latitude"]) if asgn.get("current_latitude") is not None else None,
                    float(asgn["current_longitude"]) if asgn.get("current_longitude") is not None else None,
                    float(asgn["target_latitude"]) if asgn.get("target_latitude") is not None else None,
                    float(asgn["target_longitude"]) if asgn.get("target_longitude") is not None else None,
                ],
            )
            if vehicle_id and lifecycle in ("ASSIGNED", "EN_ROUTE", "ARRIVED"):
                if is_completed:
                    self.execute_sql(
                        """
                        UPDATE demo_vehicles
                        SET status = 'AVAILABLE',
                            current_incident_id = NULL,
                            latitude = home_latitude,
                            longitude = home_longitude,
                            updated_at = NOW()
                        WHERE vehicle_id = $1 AND current_incident_id = $2::uuid;
                        """,
                        [vehicle_id, incident_id],
                    )
                else:
                    self.execute_sql(
                        """
                        UPDATE demo_vehicles
                        SET status = $2,
                            current_incident_id = $3::uuid,
                            latitude = COALESCE($4::double precision, latitude),
                            longitude = COALESCE($5::double precision, longitude),
                            updated_at = NOW()
                        WHERE vehicle_id = $1;
                        """,
                        [
                            vehicle_id,
                            lifecycle,
                            incident_id,
                            float(asgn["current_latitude"]) if asgn.get("current_latitude") is not None else None,
                            float(asgn["current_longitude"]) if asgn.get("current_longitude") is not None else None,
                        ],
                    )
        if is_completed:
            self.release_demo_incident_vehicles(incident_id)

    def persist_session_snapshot(
        self,
        snapshot: dict[str, Any],
        audit_events: list[Any],
    ) -> bool:
        """
        Persist session, incident, finalized utterances, validated facts, resource assignments,
        handoff record, and append-only audit events to PostgreSQL without mutating existing audit rows.
        Enforces monotonic session revision guards so out-of-order async tasks cannot overwrite newer state.
        Returns True if current revision was applied, or False if skipped as stale/unconfigured.
        """
        if not self._is_remote_neon:
            return False

        session_id = str(snapshot["session_id"])
        incident_id = str(snapshot["incident_id"])

        # 1. Revision-guarded upsert on session (only update mutable columns when revision >= existing)
        session_res = self.execute_sql(
            """
            INSERT INTO sessions (
                session_id, incident_id, session_state, incident_state, revision,
                last_event_sequence, initial_language, active_language,
                consecutive_unknown_count, consecutive_provider_blocks,
                unknown_fallback_applied, control_room_review_required, updated_at
            )
            VALUES ($1::uuid, $2::uuid, $3, $4, $5::int, $6::int, $7, $8, $9::int, $10::int, $11::boolean, $12::boolean, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                session_state = EXCLUDED.session_state,
                incident_state = EXCLUDED.incident_state,
                revision = EXCLUDED.revision,
                last_event_sequence = EXCLUDED.last_event_sequence,
                active_language = EXCLUDED.active_language,
                consecutive_unknown_count = EXCLUDED.consecutive_unknown_count,
                consecutive_provider_blocks = EXCLUDED.consecutive_provider_blocks,
                unknown_fallback_applied = EXCLUDED.unknown_fallback_applied,
                control_room_review_required = EXCLUDED.control_room_review_required,
                updated_at = NOW()
            WHERE EXCLUDED.revision >= sessions.revision
            RETURNING revision;
            """,
            [
                session_id,
                incident_id,
                str(snapshot["session_state"]),
                str(snapshot["incident_state"]),
                int(snapshot["revision"]),
                int(snapshot["last_event_sequence"]),
                str(snapshot["initial_language"]),
                str(snapshot["active_language"]),
                int(snapshot["consecutive_unknown_count"]),
                int(snapshot["consecutive_provider_blocks"]),
                bool(snapshot["unknown_fallback_applied"]),
                bool(snapshot["control_room_review_required"]),
            ],
        )
        is_latest_revision = bool(session_res.get("rows"))

        # Always insert idempotent append-only utterances first (parent rows for facts)
        for utt in snapshot.get("utterances") or []:
            self.execute_sql(
                """
                INSERT INTO utterances (
                    utterance_id, session_id, sequence_number, text, language, utterance_role
                )
                VALUES ($1::uuid, $2::uuid, $3::int, $4, $5, $6)
                ON CONFLICT (utterance_id) DO NOTHING;
                """,
                [
                    str(utt["utterance_id"]),
                    session_id,
                    int(utt["sequence_number"]),
                    str(utt["text"]),
                    str(utt.get("language") or "en"),
                    str(utt.get("utterance_role") or "new_description"),
                ],
            )

        # Always insert append-only audit events (ON CONFLICT DO NOTHING — never updates or deletes!)
        for evt in audit_events:
            self.execute_sql(
                """
                INSERT INTO audit_events (
                    event_id, session_id, incident_id, event_type, severity,
                    outcome, error_code, reason_code, provenance_refs, details
                )
                VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb)
                ON CONFLICT (event_id) DO NOTHING;
                """,
                [
                    str(evt.event_id),
                    str(evt.session_id),
                    str(evt.incident_id),
                    str(evt.event_type),
                    str(evt.severity.value if hasattr(evt.severity, "value") else evt.severity),
                    str(evt.outcome),
                    evt.error_code,
                    evt.reason_code,
                    json.dumps(list(evt.provenance_refs)),
                    json.dumps(evt.details, default=str),
                ],
            )

        if not is_latest_revision:
            return False

        # 2. Upsert incident
        loc = snapshot.get("location")
        reason_codes_json = json.dumps(snapshot.get("review_reason_codes") or [])
        if loc and loc.get("latitude") is not None and loc.get("longitude") is not None:
            self.execute_sql(
                """
                INSERT INTO incidents (
                    incident_id, session_id, category, category_established_above_threshold,
                    location_geom, accuracy_m, location_source, location_confirmation_status,
                    review_reason_codes, updated_at
                )
                VALUES (
                    $1::uuid, $2::uuid, $3, $4::boolean,
                    ST_SetSRID(ST_MakePoint($5::double precision, $6::double precision), 4326),
                    $7::double precision, $8, $9, $10::jsonb, NOW()
                )
                ON CONFLICT (incident_id) DO UPDATE SET
                    category = EXCLUDED.category,
                    category_established_above_threshold = EXCLUDED.category_established_above_threshold,
                    location_geom = EXCLUDED.location_geom,
                    accuracy_m = EXCLUDED.accuracy_m,
                    location_source = EXCLUDED.location_source,
                    location_confirmation_status = EXCLUDED.location_confirmation_status,
                    review_reason_codes = EXCLUDED.review_reason_codes,
                    updated_at = NOW();
                """,
                [
                    incident_id,
                    session_id,
                    str(snapshot["category"]),
                    bool(snapshot["category_established_above_threshold"]),
                    float(loc["longitude"]),
                    float(loc["latitude"]),
                    float(loc["accuracy_m"]) if loc.get("accuracy_m") is not None else None,
                    str(loc.get("source_type") or ""),
                    str(loc.get("confirmation_status") or "unresolved"),
                    reason_codes_json,
                ],
            )
        else:
            self.execute_sql(
                """
                INSERT INTO incidents (
                    incident_id, session_id, category, category_established_above_threshold,
                    review_reason_codes, updated_at
                )
                VALUES ($1::uuid, $2::uuid, $3, $4::boolean, $5::jsonb, NOW())
                ON CONFLICT (incident_id) DO UPDATE SET
                    category = EXCLUDED.category,
                    category_established_above_threshold = EXCLUDED.category_established_above_threshold,
                    review_reason_codes = EXCLUDED.review_reason_codes,
                    updated_at = NOW();
                """,
                [
                    incident_id,
                    session_id,
                    str(snapshot["category"]),
                    bool(snapshot["category_established_above_threshold"]),
                    reason_codes_json,
                ],
            )

        # 3-7 mutable state updates complete only when latest revision check succeeded
        # (utterances & audit_events already persisted above)

        # 4. Insert / update validated facts (keyed deterministically per incident + field_name)
        for fact in snapshot.get("validated_facts") or []:
            det_fact_id = str(uuid5(UUID(incident_id), str(fact["field_name"])))
            self.execute_sql(
                """
                INSERT INTO validated_facts (
                    fact_id, incident_id, field_name, field_value, source_utterance_id, evidence_text
                )
                VALUES ($1::uuid, $2::uuid, $3, $4::jsonb, $5::uuid, $6)
                ON CONFLICT (fact_id) DO UPDATE SET
                    field_value = EXCLUDED.field_value,
                    source_utterance_id = EXCLUDED.source_utterance_id,
                    evidence_text = EXCLUDED.evidence_text;
                """,
                [
                    det_fact_id,
                    incident_id,
                    str(fact["field_name"]),
                    json.dumps(fact["value"]),
                    str(fact["source_utterance_id"]),
                    str(fact["evidence_text"]),
                ],
            )

        # 5. Upsert simulated resource assignments (legacy non-demo fleet only;
        # DEMO fleet rows use demo_assignments + demo_vehicles via step 5b)
        for asgn in snapshot.get("assignments") or []:
            is_demo_asgn = (
                str(asgn.get("allocation_mode") or "") == "DEMO"
                or bool(asgn.get("vehicle_code"))
                or str(asgn.get("resource_id") or "").startswith(("AMB-", "FIR-", "POL-"))
            )
            if is_demo_asgn:
                continue
            self.execute_sql(
                """
                INSERT INTO simulated_assignments (
                    assignment_id, incident_id, idempotency_key, resource_id,
                    resource_callsign, resource_category, status,
                    simulated_eta_minutes, distance_meters, provenance, is_simulated
                )
                VALUES (
                    $1::uuid, $2::uuid, $3, $4, $5, $6, $7,
                    $8::int, $9::double precision, $10::jsonb, TRUE
                )
                ON CONFLICT (idempotency_key) DO UPDATE SET
                    resource_id = EXCLUDED.resource_id,
                    resource_callsign = EXCLUDED.resource_callsign,
                    status = EXCLUDED.status,
                    simulated_eta_minutes = EXCLUDED.simulated_eta_minutes,
                    distance_meters = EXCLUDED.distance_meters;
                """,
                [
                    str(asgn["assignment_id"]),
                    incident_id,
                    str(asgn["idempotency_key"]),
                    asgn.get("resource_id"),
                    asgn.get("resource_callsign"),
                    str(asgn["resource_category"]),
                    str(asgn["status"]),
                    int(asgn["simulated_eta_minutes"])
                    if asgn.get("simulated_eta_minutes") is not None
                    else None,
                    float(asgn["distance_meters"])
                    if asgn.get("distance_meters") is not None
                    else None,
                    json.dumps({"source": asgn.get("provenance_source", "")}),
                ],
            )

        # 5b. Sync demo_assignments and demo_vehicles lifecycle/coordinates cleanly
        self.persist_demo_fleet_state(snapshot)

        # 7. Upsert structured handoff snapshot cleanly per session
        if snapshot.get("handoff"):
            det_handoff_id = str(uuid5(UUID(session_id), "latest_handoff"))
            self.execute_sql(
                """
                INSERT INTO incident_handoffs (
                    handoff_id, incident_id, session_id, transcript_retained, handoff_payload
                )
                VALUES ($1::uuid, $2::uuid, $3::uuid, TRUE, $4::jsonb)
                ON CONFLICT (handoff_id) DO UPDATE SET
                    incident_id = EXCLUDED.incident_id,
                    generated_at = NOW(),
                    handoff_payload = EXCLUDED.handoff_payload;
                """,
                [
                    det_handoff_id,
                    incident_id,
                    session_id,
                    json.dumps(snapshot["handoff"], default=str),
                ],
            )
        return True

    def get_latest_handoff_by_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch latest persisted structured handoff payload from Postgres by session_id."""
        if not self._is_remote_neon:
            return None
        res = self.execute_sql(
            """
            SELECT handoff_payload
            FROM incident_handoffs
            WHERE session_id = $1::uuid
            ORDER BY generated_at DESC
            LIMIT 1;
            """,
            [session_id],
        )
        rows = list(res.get("rows", []))
        if not rows:
            return None
        payload = rows[0].get("handoff_payload")
        if isinstance(payload, str):
            return json.loads(payload)
        return payload if isinstance(payload, dict) else None
