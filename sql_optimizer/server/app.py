import os
from openenv.core.env_server import create_fastapi_app

# Import the environment class and your Pydantic models
from sql_optimizer.server.sql_optimizer_environment import SQLOptimizerEnvironment
from sql_optimizer.models import SQLAction, SQLObservation

# Wrap the environment in the standard OpenEnv FastAPI server.
# Notice we pass the CLASS itself (SQLOptimizerEnvironment) as the factory, 
# not an instance (SQLOptimizerEnvironment()).
app = create_fastapi_app(
    env=SQLOptimizerEnvironment,
    action_cls=SQLAction,
    observation_cls=SQLObservation
)

def main():
    import uvicorn
    # It is best practice to pass the app as an import string when using uvicorn
    uvicorn.run("sql_optimizer.server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()