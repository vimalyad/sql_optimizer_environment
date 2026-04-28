---
title: SQL Optimizer Environment
emoji: 🗃️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# 🗃️ SQL Optimizer Environment

> An [OpenEnv](https://github.com/meta-pytorch/openenv) reinforcement-learning environment that teaches agents to **rewrite slow SQL into fast SQL** — verified against a real PostgreSQL database with `EXPLAIN ANALYZE`.

---

## ✨ What it does

You hand the environment a slow query and a Postgres connection string. The agent then has up to **N rewrite steps** to make it faster. Every rewrite is **executed against a real database**, and the reward is the actual measured speedup — no simulation, no synthetic cost models.

```
slow query  ──►  agent picks rewrite  ──►  EXPLAIN ANALYZE  ──►  reward = % faster
                       ▲                                              │
                       └──────────── new observation ◄────────────────┘
```

The environment auto-discovers schema, indexes, and statistics from `pg_catalog` / `information_schema` — **no upfront schema definition required**. Point it at any Postgres DB and go.

---

## 🧠 The 9 actions

Actions split into **structural rewrites** (always available) and **hint-based rewrites** (require the `pg_hint_plan` extension). The environment hides hint actions automatically when the extension isn't installed.

> ⚠️ **Heads up:** Actions **1, 2, and 3** (`add_index_hint`, `add_join_order_hint`, `add_join_method_hint`) only work if the [`pg_hint_plan`](https://github.com/ossc-db/pg_hint_plan) extension is installed on your Postgres instance. Without it, these actions are stripped from `legal_actions` and only the structural rewrites (4–9) are available to the agent.

| ID | Action | Requires | What it does |
|----|--------|----------|--------------|
| 1 | `add_index_hint` | `pg_hint_plan` | Force a specific index via `/*+ IndexScan(...) */` |
| 2 | `add_join_order_hint` | `pg_hint_plan` | Force join order via `/*+ Leading(...) */` |
| 3 | `add_join_method_hint` | `pg_hint_plan` | Force HashJoin / NestLoop / MergeJoin |
| 4 | `push_predicate` | — | Move a `WHERE` filter into the `JOIN ON` clause |
| 5 | `replace_subquery_with_join` | — | Rewrite `IN (SELECT ...)` as a `JOIN` |
| 6 | `remove_redundant_join` | — | Drop a `JOIN` whose columns are never referenced |
| 7 | `replace_select_star` | — | Expand `SELECT *` to only the columns needed |
| 8 | `materialize_cte` | — | Add `MATERIALIZED` to a `WITH` clause |
| 9 | `submit` | — | End the episode and return the final query |

Adding a new action = one entry in `ACTION_REGISTRY` (`sql_optimizer/models.py`). Nothing else changes.

---

## 🎯 Observation, Action, State

Strict typed contracts via Pydantic:

```python
class SQLAction:           # what the agent sends
    action_id: int
    params: Dict[str, Any]

class SQLObservation:      # what the agent gets back
    current_query: str
    observation_vector: List[float]   # featurized plan stats
    legal_actions: List[Dict]         # filtered by available extensions
    explain_plan: Dict
    done: bool
    reward: float

class SQLState:            # full episode metadata (env.state())
    original_query: str
    current_query: str
    baseline_time_ms: float
    current_time_ms: float
    rewrites_applied: List[str]
    step_count: int
    total_reward: float
    improvement_pct: float
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│  Agent (your RL loop)                                │
│   └── SQLOptimizerEnv  ◄── client.py (typed wrapper) │
└─────────────────────┬────────────────────────────────┘
                      │ HTTP (FastAPI / OpenEnv core)
┌─────────────────────▼────────────────────────────────┐
│  Env Server  (sql_optimizer/server/app.py)           │
│   • parses & validates actions                       │
│   • applies rewrite to current query                 │
│   • runs EXPLAIN ANALYZE on Postgres                 │
│   • computes reward + builds next observation        │
└─────────────────────┬────────────────────────────────┘
                      │ psycopg2
┌─────────────────────▼────────────────────────────────┐
│  PostgreSQL ≥ 13  (+ pg_hint_plan, optional)         │
│   schema discovered live from pg_catalog             │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech stack

- **Python 3.13** · `pyproject.toml` + `uv` for dep mgmt
- **FastAPI** + **uvicorn** — HTTP env server
- **OpenEnv core** — `EnvClient` / `Action` / `Observation` / `State` base classes
- **Pydantic v2** — typed contracts, automatic action validation
- **psycopg2** — Postgres driver
- **PostgreSQL 13+** with optional **pg_hint_plan** extension
- **Docker / docker-compose** — one-command local stack (env + DB)
- **Hugging Face Spaces** — cloud deployment via `openenv push`

---

## 🚀 Quickstart

### Run locally with Docker

```bash
git clone https://github.com/AJ5831A/sql_optimizer_environment
cd sql_optimizer_environment
docker-compose up -d --build
```

This brings up:
- `sql_optimizer_db` — Postgres with `pg_hint_plan` and a sample schema preloaded
- `sql_optimizer_env` — the OpenEnv server on `http://localhost:8000`

### Use it from Python

```python
from sql_optimizer.client import SQLOptimizerEnv
from sql_optimizer.models import SQLAction

env = SQLOptimizerEnv(base_url="http://localhost:8000")

obs = env.reset(query="SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE region='EU')")

# Agent picks an action from obs.legal_actions
result = env.step(SQLAction(action_id=5, params={}))   # replace_subquery_with_join
print(result.reward, result.observation.current_query)

env.step(SQLAction(action_id=9, params={}))            # submit
print(env.state().improvement_pct, "% faster")
```

---

## ⚙️ Configuration

All knobs are environment variables (see `openenv.yaml`):

| Var | Default | Purpose |
|-----|---------|---------|
| `WORKERS` | `4` | uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | `100` | concurrent sessions per worker |
| `QUERY_TIMEOUT_MS` | `30000` | per-query execution cap |
| `MAX_STEPS` | `10` | max rewrites per episode |
| `DATABASE_URL` | — | Postgres connection string |

---

## 📦 Project layout

```
sql_optimizer_environment/
├── openenv.yaml              # env spec (actions, runtime, hardware)
├── Dockerfile                # HF Spaces / openenv build
├── docker-compose.yml        # local dev stack (env + db)
├── db.Dockerfile             # Postgres + pg_hint_plan + sample schema
├── client.py / models.py     # root re-exports for openenv push
├── sql_optimizer/
│   ├── client.py             # SQLOptimizerEnv (typed client)
│   ├── models.py             # ACTION_REGISTRY + dataclasses
│   ├── db.py                 # schema discovery, EXPLAIN ANALYZE runner
│   └── server/
│       ├── app.py            # FastAPI env server
│       └── Dockerfile
└── pyproject.toml
```

---

## ☁️ Deploy to Hugging Face Spaces

```bash
hf auth login
openenv push --repo-id <your-username>/sql-optimizer-environment
```

Live demo: **[huggingface.co/spaces/ILoveTemples/sql-optimizer-environment](https://huggingface.co/spaces/ILoveTemples/sql-optimizer-environment)**

---

## 📜 License

BSD-style — see `LICENSE`.
