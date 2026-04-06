import os
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extensions


class PostgreSQLExecutor:
    """
    Manages one Postgres connection per optimization episode.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn: Optional[psycopg2.extensions.connection] = None

        self.hints_enabled: bool = False
        self.available_extensions: List[str] = []

        self._query_timeout_ms = int(
            os.getenv("QUERY_TIMEOUT_MS", "30000")
        )

    def _get_conn(self) -> psycopg2.extensions.connection:
        """Helper to satisfy type checker and ensure connection exists."""
        if self.conn is None:
            raise RuntimeError("Database connection not established. Call connect() first.")
        return self.conn

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open a connection to the user's Postgres database.
        Immediately probes for extensions.
        """
        try:
            self.conn = psycopg2.connect(
                self.db_url,
                connect_timeout=10,
                options=f"-c statement_timeout={self._query_timeout_ms}",
            )
            self.conn.autocommit = True
        except psycopg2.OperationalError as e:
            raise ConnectionError(
                f"Could not connect to database. "
                f"Check that your db_url is correct and the database is reachable.\n"
                f"Original error: {e}"
            )

        # Discover all loaded extensions dynamically
        self.available_extensions = self._fetch_all_extensions()
        self.hints_enabled = "pg_hint_plan" in self.available_extensions

        if self.hints_enabled:
            print("[db] pg_hint_plan detected — Tier 2 hint actions enabled")
        else:
            print(
                "[db] pg_hint_plan NOT detected — Tier 2 hint actions disabled. "
                "Structural rewrites (Tier 1) still available. "
                "To enable hints: https://github.com/ossc-db/pg_hint_plan"
            )

    def close(self) -> None:
        """Close the connection. Called at episode end."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None

    def _fetch_all_extensions(self) -> List[str]:
        """Query pg_extension to find all installed and loaded extensions."""
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("SELECT extname FROM pg_extension")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"[db] Extension discovery failed: {e} — assuming none installed")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # EXPLAIN ANALYZE
    # ─────────────────────────────────────────────────────────────────────────

    def get_explain_plan(self, sql: str) -> Dict:
        try:
            cursor = self._get_conn().cursor()
            cursor.execute(f"EXPLAIN (FORMAT JSON, ANALYZE) {sql}")
            result = cursor.fetchone()
            if result and result[0]:
                return result[0][0]   # unwrap [{ "Plan": {...} }]
            return {}
        except Exception as e:
            print(f"[db] EXPLAIN ANALYZE failed: {e}")
            return {}

    def measure_execution_time(self, sql: str) -> float:
        plan = self.get_explain_plan(sql)
        return plan.get("Execution Time", 999999.0)

    # ─────────────────────────────────────────────────────────────────────────
    # CORRECTNESS VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def verify_correctness(self, original_sql: str, rewritten_sql: str) -> bool:
        try:
            cursor = self._get_conn().cursor()
            
            cursor.execute(f"SELECT md5(array_agg(t.*)::text) FROM ({original_sql}) t")
            row1 = cursor.fetchone()
            hash_original = row1[0] if row1 else None

            cursor.execute(f"SELECT md5(array_agg(t.*)::text) FROM ({rewritten_sql}) t")
            row2 = cursor.fetchone()
            hash_rewritten = row2[0] if row2 else None

            return hash_original == hash_rewritten
        except Exception as e:
            print(f"[db] Correctness check failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # SCHEMA DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def get_available_indexes(self) -> Dict[str, List[str]]:
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("""
                SELECT
                    t.relname  AS table_name,
                    i.relname  AS index_name
                FROM
                    pg_class     t
                    JOIN pg_index  ix ON t.oid = ix.indrelid
                    JOIN pg_class  i  ON i.oid = ix.indexrelid
                    JOIN pg_namespace n ON t.relnamespace = n.oid
                WHERE
                    t.relkind = 'r'
                    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                    AND NOT ix.indisprimary
                ORDER BY
                    t.relname, i.relname
            """)
            indexes: Dict[str, List[str]] = {}
            for table_name, index_name in cursor.fetchall():
                if table_name not in indexes:
                    indexes[table_name] = []
                indexes[table_name].append(index_name)
            return indexes
        except Exception as e:
            print(f"[db] Index discovery failed: {e}")
            return {}

    def get_column_names(self, table_name: str) -> List[str]:
        try:
            cursor = self._get_conn().cursor()
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,)
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"[db] Column discovery failed for {table_name}: {e}")
            return []

    def get_table_stats(self, table_name: str) -> Dict:
        try:
            cursor = self._get_conn().cursor()
            cursor.execute(
                """
                SELECT
                    reltuples::bigint AS estimated_rows,
                    relpages          AS pages
                FROM pg_class
                WHERE relname = %s
                """,
                (table_name,)
            )
            row = cursor.fetchone()
            if row:
                return {"estimated_rows": row[0], "pages": row[1]}
            return {}
        except Exception as e:
            print(f"[db] Table stats failed for {table_name}: {e}")
            return {}