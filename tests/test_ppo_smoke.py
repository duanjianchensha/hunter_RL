"""PPO 模块冒烟测试（需安装 torch，未装则跳过）。"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from hunt_env.config.loader import load_config
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig


def test_ppo_one_train_step_cpu():
    cfg = load_config(None)
    cfg.vectorization.num_envs = 1  # type: ignore[misc]
    env = HuntVectorizedEnv(cfg=cfg, num_envs=1)
    ppo = PPOConfig(update_epochs=1, num_minibatches=1)
    trainer = MultiAgentPPOTrainer(
        cfg,
        env,
        ppo_cfg=ppo,
        hidden_sizes=(32, 32),
        use_cuda_if_available=False,
    )
    obs = env.reset(seed=42)
    next_obs, logs = trainer.train_step(8, obs)
    assert next_obs.shape == obs.shape
    assert len(logs) == env.n_agents
    assert all("loss" in lg for lg in logs)
