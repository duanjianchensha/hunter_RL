"""追猎式多智能体 RL 环境包。"""

from hunt_env.config.loader import load_config
from hunt_env.env.hunt_parallel import HuntParallelEnv

__all__ = ["load_config", "HuntParallelEnv"]
