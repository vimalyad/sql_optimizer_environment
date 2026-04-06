# ─────────────────────────────────────────────────────────────────────────────
# server/sql_optimizer_environment.py
#
# Core RL environment.
#
# Episode flow:
#   1. User calls reset(query, db_url)
#   2. Environment connects to user's database
#   3. Environment probes for pg_hint_plan → sets hints_enabled flag
#   4. Environment measures baseline execution time via EXPLAIN ANALYZE
#   5. Environment discovers indexes and schema from pg_catalog
#   6. Agent receives first observation with legal_actions
#   7. Agent calls step(action) repeatedly
#   8. Each step rewrites the SQL, measures new time, returns reward
#   9. Episode ends when agent submits or max steps reached
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
from typing import Any, Dict, List, Optional

import sqlglot
import sqlglot.expressions as exp

from openenv.core.env_server import Environment
from sql_optimizer_env.models import (
    ACTION_REGISTRY,
    SQLAction,
    SQLObservation,
    SQLState,
    get_action_name,
)
from sql_optimizer_env.server.db import PostgreSQLExecutor


class SQLOptimizerEnvironment(Environment):
    """
    RL environment that teaches an agent to rewrite slow SQL queries.

    The environment has no knowledge of any schema upfront.
    At reset() time the user provides:
      - query:  the slow SQL query they want optimized
      - db_url: connection string to their Postgres database

    The environment then discovers everything it needs automatically
    from pg_catalog and information_schema.
    """

    def __init__(self):
        self._max_steps = int(os.getenv("MAX_STEPS", "10"))

        # Active DB connection — replaced at every reset() call
        self._db: Optional[PostgreSQLExecutor] = None

        # Episode state — all reset at reset() time
        self._episode_id: str = ""
        self._original_query: str = ""
        self._current_query: str = ""
        self._baseline_time_ms: float = 0.0
        self._current_time_ms: float = 0.0
        self._rewrites_applied: List[str] = []
        self._available_extensions: List[str] = []
        self._step_count: int = 0
        self._total_reward: float = 0.0

        # Discovered at reset() time from pg_catalog
        # maps table_name → list of index names
        self._available_indexes: Dict[str, List[str]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # CORE INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, query: str, db_url: str) -> SQLObservation:
        """
        Start a new optimization episode.

        Args:
            query:  The slow SQL query to optimize.
                    Any valid SELECT statement against the user's database.
            db_url: Postgres connection string.
                    e.g. "postgresql://user:pass@host:5432/dbname"

        The environment will:
          1. Connect to the database and probe for available extensions
          2. Discover available indexes automatically from pg_catalog
          3. Run EXPLAIN ANALYZE to measure the baseline execution time
          4. Return the first observation with legal_actions populated

        Raises:
            ConnectionError: if the database cannot be reached
            ValueError: if the query is empty or not a SELECT
        """
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")
        if not query.upper().startswith("SELECT"):
            raise ValueError(
                "Only SELECT queries are supported. "
                "DML statements (INSERT, UPDATE, DELETE) cannot be optimized."
            )

        # Close any previous connection cleanly
        if self._db:
            self._db.close()

        # Connect — db.connect() probes for extensions internally
        self._db = PostgreSQLExecutor(db_url)
        self._db.connect()

        # Build the extensions list from what the DB reported
        # PostgreSQLExecutor.available_extensions is a List[str] of
        # extension names e.g. ["pg_hint_plan", "pg_stat_statements"]
        self._available_extensions = self._db.available_extensions

        # Reset episode state
        self._episode_id       = str(uuid.uuid4())
        self._original_query   = query
        self._current_query    = query
        self._rewrites_applied = []
        self._step_count       = 0
        self._total_reward     = 0.0

        # Discover indexes from pg_catalog
        self._available_indexes = self._db.get_available_indexes()

        # Measure baseline
        plan = self._db.get_explain_plan(self._original_query)
        self._baseline_time_ms = plan.get("Execution Time", 999999.0)
        self._current_time_ms  = self._baseline_time_ms

        return self._build_observation(plan=plan, reward=0.0, done=False)

    def step(self, action: SQLAction) -> SQLObservation:
        """
        Apply one rewrite action to the current query.

        The action must be chosen from the legal_actions list in the
        last observation. Choosing an action not in legal_actions is
        allowed but will return reward=-0.05 and no change.

        Returns the new observation including:
          - the rewritten SQL
          - the new observation_vector
          - updated legal_actions for the next step
          - the reward for this step
        """
        if self._db is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Validate action against registry
        try:
            action.validate()
        except ValueError as e:
            print(f"[env] Invalid action: {e}")
            plan = self._db.get_explain_plan(self._current_query)
            return self._build_observation(plan=plan, reward=-0.05, done=False)

        # Terminal action
        if action.action_id == 9:
            return self._end_episode()

        # Apply the rewrite
        try:
            rewritten = self._apply_action(self._current_query, action)
        except Exception as e:
            print(f"[env] Rewrite failed for action {action.action_id}: {e}")
            plan = self._db.get_explain_plan(self._current_query)
            return self._build_observation(plan=plan, reward=-0.05, done=False)

        # Verify correctness
        if not self._db.verify_correctness(self._original_query, rewritten):
            print(f"[env] Correctness check failed for action {action.action_id}")
            plan = self._db.get_explain_plan(self._current_query)
            return self._build_observation(plan=plan, reward=-1.0, done=False)

        # Measure new execution time
        new_plan = self._db.get_explain_plan(rewritten)
        new_time = new_plan.get("Execution Time", self._current_time_ms)

        # Reward: normalised improvement relative to baseline
        reward = (self._current_time_ms - new_time) / self._baseline_time_ms

        # Update state
        prev_time = self._current_time_ms
        self._current_query    = rewritten
        self._current_time_ms  = new_time
        self._total_reward    += reward
        self._rewrites_applied.append(get_action_name(action.action_id))

        print(
            f"[env] step={self._step_count} "
            f"action={get_action_name(action.action_id)} "
            f"time={prev_time:.1f}ms → {new_time:.1f}ms "
            f"reward={reward:+.3f}"
        )

        # Check termination
        if self._step_count >= self._max_steps:
            return self._end_episode()

        return self._build_observation(plan=new_plan, reward=reward, done=False)

    @property
    def state(self) -> SQLState:
        improvement = 0.0
        if self._baseline_time_ms > 0:
            improvement = (
                (self._baseline_time_ms - self._current_time_ms)
                / self._baseline_time_ms * 100
            )
        return SQLState(
            episode_id=self._episode_id,
            original_query=self._original_query,
            current_query=self._current_query,
            baseline_time_ms=self._baseline_time_ms,
            current_time_ms=self._current_time_ms,
            rewrites_applied=self._rewrites_applied,
            available_extensions=self._available_extensions,
            step_count=self._step_count,
            total_reward=self._total_reward,
            improvement_pct=round(improvement, 2),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LEGAL ACTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_legal_actions(
        self, sql: str, plan: Dict
    ) -> List[Dict[str, Any]]:
        """
        Compute which actions are valid for the current query state.

        Two passes:
          1. Filter ACTION_REGISTRY by available_extensions → candidate set
          2. For each candidate, inspect the SQL AST to check applicability

        This replaces the old hardcoded TIER1/TIER2 split.
        Adding a new action only requires an entry in ACTION_REGISTRY.
        """
        try:
            tree = sqlglot.parse_one(sql, dialect="postgres")
        except Exception:
            # Unparseable — only submit is safe
            return [{
                "action_id": 9,
                "name": "submit",
                "params": {},
                "description": "submit current query as final answer",
            }]

        signals     = self._extract_signals(plan)
        tables      = [t.name for t in tree.find_all(exp.Table)]
        joins       = list(tree.find_all(exp.Join))
        inner_joins = [
            j for j in joins
            if j.args.get("kind") in (None, "INNER", "")
        ]
        has_subquery = bool(list(tree.find_all(exp.Subquery)))
        has_cte      = bool(tree.find(exp.With))
        has_star     = bool(tree.find(exp.Star))
        where        = tree.find(exp.Where)
        used_tables  = {c.table for c in tree.find_all(exp.Column) if c.table}

        # Applicability checks — keyed by action_id
        # Returns a list of (params, description) pairs.
        # Empty list means action is not applicable to the current query.
        def applicable(action_id: int) -> List[Dict[str, Any]]:
            has_seq_scan = signals[2] == 1.0
            rows_removed = signals[4]

            if action_id == 1:  # add_index_hint
                if not (has_seq_scan and rows_removed > 1000):
                    return []
                return [
                    {
                        "params": {"table": table, "index": index},
                        "description": f"hint: use index {index} on {table}",
                    }
                    for table in tables
                    for index in self._available_indexes.get(table, [])
                ]

            if action_id == 2:  # add_join_order_hint
                if len(inner_joins) < 2:
                    return []
                return [{
                    "params": {"table_order": tables},
                    "description": f"hint: join order {' → '.join(tables)}",
                }]

            if action_id == 3:  # add_join_method_hint
                if len(inner_joins) < 1 or len(tables) < 2:
                    return []
                return [
                    {
                        "params": {
                            "table_a": tables[0],
                            "table_b": tables[1],
                            "method": method,
                        },
                        "description": f"hint: use {method} for {tables[0]} × {tables[1]}",
                    }
                    for method in ["HashJoin", "NestLoop", "MergeJoin"]
                ]

            if action_id == 4:  # push_predicate
                if not (where and joins):
                    return []
                for condition in where.find_all(exp.EQ):
                    col = condition.find(exp.Column)
                    if col and col.table:
                        return [{
                            "params": {"target_table": col.table},
                            "description": f"push predicate on {col.table} into JOIN ON",
                        }]
                return []

            if action_id == 5:  # replace_subquery_with_join
                if not has_subquery:
                    return []
                return [{"params": {}, "description": "replace IN (SELECT...) subquery with JOIN"}]

            if action_id == 6:  # remove_redundant_join
                candidates = []
                for join in joins:
                    t = join.find(exp.Table)
                    a = join.find(exp.TableAlias)
                    ref = (a.name if a else t.name) if t else None
                    if ref and ref not in used_tables:
                        candidates.append({
                            "params": {"table": ref},
                            "description": f"remove unused JOIN on {ref}",
                        })
                return candidates

            if action_id == 7:  # replace_select_star
                if not has_star:
                    return []
                return [{"params": {}, "description": "expand SELECT * to explicit column list"}]

            if action_id == 8:  # materialize_cte
                if not has_cte:
                    return []
                return [{"params": {}, "description": "add MATERIALIZED to CTE"}]

            if action_id == 9:  # submit — always available
                return [{"params": {}, "description": "submit current query as final answer"}]

            return []

        # Build legal_actions: extension filter first, then applicability
        legal = []
        for action_id, meta in ACTION_REGISTRY.items():
            # Skip if a required extension is not available
            if not all(ext in self._available_extensions for ext in meta["requires"]):
                continue
            # Check query-level applicability
            for match in applicable(action_id):
                legal.append({
                    "action_id": action_id,
                    "name": meta["name"],
                    "params": match["params"],
                    "description": match["description"],
                })

        return legal

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION FUNCTIONS — one per action_id
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_action(self, sql: str, action: SQLAction) -> str:
        dispatch = {
            1: self._add_index_hint,
            2: self._add_join_order_hint,
            3: self._add_join_method_hint,
            4: self._push_predicate,
            5: self._replace_subquery_with_join,
            6: self._remove_redundant_join,
            7: self._replace_select_star,
            8: self._materialize_cte,
        }
        fn = dispatch.get(action.action_id)
        if fn is None:
            return sql
        return fn(sql, **action.params)

    def _add_hint_comment(self, sql: str, new_hint: str) -> str:
        """Append to existing /*+ ... */ block or create a new one."""
        stripped = sql.strip()
        if stripped.startswith("/*+"):
            closing = stripped.index("*/")
            return stripped[:closing].rstrip() + f" {new_hint}" + stripped[closing:]
        return f"/*+ {new_hint} */\n{sql}"

    def _add_index_hint(self, sql: str, table: str, index: str, **_) -> str:
        return self._add_hint_comment(sql, f"IndexScan({table} {index})")

    def _add_join_order_hint(self, sql: str, table_order: List[str], **_) -> str:
        return self._add_hint_comment(sql, f"Leading({' '.join(table_order)})")

    def _add_join_method_hint(
        self, sql: str, table_a: str, table_b: str, method: str, **_
    ) -> str:
        return self._add_hint_comment(sql, f"{method}({table_a} {table_b})")

    def _push_predicate(self, sql: str, target_table: str, **_) -> str:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        where = tree.find(exp.Where)
        if not where:
            return sql

        to_push = []
        for cond in list(where.find_all(exp.EQ)):
            col = cond.find(exp.Column)
            if col and col.table == target_table:
                to_push.append(cond.copy())
                cond.pop()

        if not to_push:
            return sql

        for join in tree.find_all(exp.Join):
            t = join.find(exp.Table)
            if t and t.name == target_table:
                existing = join.args.get("on")
                for cond in to_push:
                    if existing:
                        join.set("on", exp.And(this=existing, expression=cond))
                        existing = join.args.get("on")
                    else:
                        join.set("on", cond)
                break

        return tree.sql(dialect="postgres")

    def _replace_subquery_with_join(self, sql: str, **_) -> str:
        tree = sqlglot.parse_one(sql, dialect="postgres")

        for in_expr in list(tree.find_all(exp.In)):
            subquery = in_expr.args.get("query")
            if not subquery:
                continue

            outer_col   = in_expr.this
            inner_table = subquery.find(exp.Table)
            inner_col   = subquery.find(exp.Column)
            if not all([outer_col, inner_table, inner_col]):
                continue

            alias = f"sq_{inner_table.name}"

            new_join = exp.Join(
                this=exp.Alias(
                    this=inner_table.copy(),
                    alias=exp.TableAlias(
                        this=exp.Identifier(this=alias)
                    ),
                ),
                on=exp.EQ(
                    this=outer_col.copy(),
                    expression=exp.Column(
                        this=exp.Identifier(this=inner_col.name),
                        table=exp.Identifier(this=alias),
                    ),
                ),
                kind="INNER",
            )

            inner_where = subquery.find(exp.Where)
            if inner_where:
                outer_where = tree.find(exp.Where)
                if outer_where:
                    outer_where.set(
                        "this",
                        exp.And(
                            this=outer_where.this,
                            expression=inner_where.this.copy()
                        )
                    )

            from_clause = tree.find(exp.From)
            if from_clause:
                from_clause.parent.append("joins", new_join)

            in_expr.pop()

        return tree.sql(dialect="postgres")

    def _remove_redundant_join(self, sql: str, table: str, **_) -> str:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        used = {c.table for c in tree.find_all(exp.Column) if c.table}

        for join in tree.find_all(exp.Join):
            t = join.find(exp.Table)
            a = join.find(exp.TableAlias)
            ref = (a.name if a else t.name) if t else None
            if ref == table and ref not in used:
                join.pop()
                break

        return tree.sql(dialect="postgres")

    def _replace_select_star(self, sql: str, **_) -> str:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        if not tree.find(exp.Star):
            return sql

        tables = [t.name for t in tree.find_all(exp.Table)]
        explicit = []

        for table in tables:
            columns = self._db.get_column_names(table)
            for col_name in columns:
                explicit.append(
                    exp.Column(
                        this=exp.Identifier(this=col_name),
                        table=exp.Identifier(this=table),
                    )
                )

        if explicit:
            tree.find(exp.Select).set("expressions", explicit)

        return tree.sql(dialect="postgres")

    def _materialize_cte(self, sql: str, cte_name: str = "", **_) -> str:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        for cte in tree.find_all(exp.CTE):
            if not cte_name or cte.alias == cte_name:
                cte.set("materialized", True)
                break
        return tree.sql(dialect="postgres")

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL EXTRACTION — EXPLAIN JSON → flat feature vector
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_signals(self, plan: Dict) -> List[float]:
        """
        Walk the EXPLAIN ANALYZE JSON tree and extract a flat feature vector.
        Position is fixed — agent sees only numbers, not names.
        """
        signals = {
            "execution_time_ms":       plan.get("Execution Time", 0.0),
            "total_plan_cost":         0.0,
            "has_seq_scan":            0.0,
            "has_subquery":            0.0,
            "max_rows_removed":        0.0,
            "num_joins":               0.0,
            "has_redundant_join":      0.0,
            "has_cte":                 0.0,
            "has_select_star":         0.0,
            "estimated_vs_actual_gap": 0.0,
        }

        def walk(node: Dict) -> None:
            node_type = node.get("Node Type", "")
            signals["total_plan_cost"] = max(
                signals["total_plan_cost"], node.get("Total Cost", 0.0)
            )
            if node_type == "Seq Scan":
                signals["has_seq_scan"] = 1.0
                signals["max_rows_removed"] = max(
                    signals["max_rows_removed"],
                    float(node.get("Rows Removed by Filter", 0))
                )
                signals["estimated_vs_actual_gap"] = max(
                    signals["estimated_vs_actual_gap"],
                    float(abs(node.get("Plan Rows", 0) - node.get("Actual Rows", 0)))
                )
            if node_type in ("Subquery Scan", "InitPlan"):
                signals["has_subquery"] = 1.0
            if node_type in ("Hash Join", "Nested Loop", "Merge Join"):
                signals["num_joins"] += 1.0
            for child in node.get("Plans", []):
                walk(child)

        if "Plan" in plan:
            walk(plan["Plan"])

        return list(signals.values())

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(
        self, plan: Dict, reward: float, done: bool
    ) -> SQLObservation:
        signals = self._extract_signals(plan)
        legal   = self._compute_legal_actions(self._current_query, plan) if not done else []

        return SQLObservation(
            current_query=self._current_query,
            observation_vector=signals,
            legal_actions=legal,
            explain_plan=plan,
            done=done,
            reward=reward,
            metadata={
                "episode_id":       self._episode_id,
                "step":             self._step_count,
                "baseline_ms":      self._baseline_time_ms,
                "current_ms":       self._current_time_ms,
                "improvement_pct":  round(
                    (self._baseline_time_ms - self._current_time_ms)
                    / max(self._baseline_time_ms, 1) * 100, 2
                ),
                "rewrites_applied": self._rewrites_applied,
            }
        )

    def _end_episode(self) -> SQLObservation:
        plan       = self._db.get_explain_plan(self._current_query)
        final_time = plan.get("Execution Time", self._current_time_ms)
        reward     = (self._baseline_time_ms - final_time) / max(self._baseline_time_ms, 1)

        self._current_time_ms = final_time
        self._db.close()

        return SQLObservation(
            current_query=self._current_query,
            observation_vector=self._extract_signals(plan),
            legal_actions=[],
            explain_plan=plan,
            done=True,
            reward=reward,
            metadata={
                "episode_id":       self._episode_id,
                "step":             self._step_count,
                "baseline_ms":      self._baseline_time_ms,
                "final_ms":         final_time,
                "improvement_pct":  round(reward * 100, 2),
                "rewrites_applied": self._rewrites_applied,
            }
        )