from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from sql_optimizer.models import SQLAction, SQLObservation, SQLState


class SQLOptimizerEnv(EnvClient[SQLAction, SQLObservation, SQLState]):
    """Client for the SQL Optimizer Environment."""

    def _step_payload(self, action: SQLAction) -> Dict[str, Any]:
        """Convert the typed Action into a raw payload for the server."""
        return {
            "action_id": action.action_id,
            "params":    action.params,
            "metadata":  action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLObservation]:
        """Parse the server's raw step response into a StepResult."""
        obs = payload.get("observation", {})
        return StepResult(
            observation=SQLObservation(
                current_query=obs.get("current_query", ""),
                observation_vector=obs.get("observation_vector", []),
                legal_actions=obs.get("legal_actions", []),
                explain_plan=obs.get("explain_plan", {}),
                # hints_available=obs.get("hint", False)
                done=obs.get("done", False),
                reward=obs.get("reward", 0.0),
                metadata=obs.get("metadata", {}),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SQLState:
        """Parse the server's state response into a State object."""
        return SQLState(
            episode_id=payload.get("episode_id"),
            original_query=payload.get("original_query", ""),
            current_query=payload.get("current_query", ""),
            baseline_time_ms=payload.get("baseline_time_ms", 0.0),
            current_time_ms=payload.get("current_time_ms", 0.0),
            rewrites_applied=payload.get("rewrites_applied", []),
            available_extensions=payload.get("available_extensions", []),  # added
            step_count=payload.get("step_count", 0),
            total_reward=payload.get("total_reward", 0.0),
            improvement_pct=payload.get("improvement_pct", 0.0),
        )