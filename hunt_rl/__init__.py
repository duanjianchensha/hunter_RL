"""PyTorch PPO 训练组件（追猎环境）。"""

from hunt_rl.device import get_train_device
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig

__all__ = ["MultiAgentPPOTrainer", "PPOConfig", "get_train_device"]
