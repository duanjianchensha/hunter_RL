"""逃脱者 RL + 规则猎人 冒烟。"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from hunt_env.config.loader import load_config
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig


def test_ppo_escaper_only_rule_hunter():
    cfg = load_config(None)
    cfg.vectorization.num_envs = 1  # type: ignore[misc]
    if cfg.agents.n_escapers < 1 or cfg.agents.n_hunters < 1:
        pytest.skip("需至少 1 猎人 1 逃脱者")
    env = HuntVectorizedEnv(cfg=cfg, num_envs=1)
    ppo = PPOConfig(update_epochs=1, num_minibatches=1)
    trainer = MultiAgentPPOTrainer(
        cfg,
        env,
        ppo_cfg=ppo,
        hidden_sizes=(32, 32),
        use_cuda_if_available=False,
        hunter_mode="rule",
    )
    assert "hunter" not in trainer.policies
    obs = env.reset(seed=0)
    next_obs, logs, metrics = trainer.train_step(6, obs)
    assert next_obs.shape == obs.shape
    assert len(logs) == env.cfg.agents.n_escapers
    assert all(lg["role"] == "escaper" for lg in logs)
    assert "hunter_rew_mean" in metrics
    assert "escaper_rew_mean" in metrics
