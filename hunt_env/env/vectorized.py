"""大批量并行环境：同一 HuntBatchEngine，暴露 numpy 张量 API。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hunt_env.config.loader import load_config
from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.engine import HuntBatchEngine
from hunt_env.core.state import agent_names


@dataclass
class HuntVectorizedEnv:
    """
    向量化多环境，用于 RL 训练循环。

    - reset() -> obs (E, N, D)
    - step(actions) -> obs, rew, term, trunc 均为 (E, N, ...)
    """

    cfg: HuntEnvConfig
    num_envs: int

    def __post_init__(self) -> None:
        self.engine = HuntBatchEngine(self.cfg, num_envs=self.num_envs)
        self.agent_ids = agent_names(self.cfg.agents.n_hunters, self.cfg.agents.n_escapers)
        self.n_agents = len(self.agent_ids)
        self.obs_dim = self.engine.obs_dim()

    @classmethod
    def from_yaml(cls, path: str | None = None) -> HuntVectorizedEnv:
        cfg = load_config(path)
        return cls(cfg=cfg, num_envs=cfg.vectorization.num_envs)

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self.engine.reset(seed=seed)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """actions: (E, N, 2) float"""
        if actions.shape != (self.num_envs, self.n_agents, 2):
            raise ValueError(f"actions 形状应为 ({self.num_envs}, {self.n_agents}, 2)，收到 {actions.shape}")
        return self.engine.step(actions)
