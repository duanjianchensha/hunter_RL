"""仿真引擎：捕获、观测维度、确定性。"""

from __future__ import annotations

import numpy as np

from hunt_env.core.engine import HuntBatchEngine
from hunt_env.core.state import total_obs_dim


def test_reset_obs_shape_and_deterministic(minimal_cfg) -> None:
    eng = HuntBatchEngine(minimal_cfg, num_envs=1)
    d = total_obs_dim(minimal_cfg)
    o1 = eng.reset(seed=42)
    eng2 = HuntBatchEngine(minimal_cfg, num_envs=1)
    o2 = eng2.reset(seed=42)
    assert o1.shape == (1, eng.n_agents, d)
    np.testing.assert_array_equal(o1, o2)


def test_capture_and_terminal(minimal_cfg) -> None:
    """手工放置：猎人贴近逃脱者，一步内应捕获并 termination。"""
    eng = HuntBatchEngine(minimal_cfg, num_envs=1)
    eng.reset(seed=0)
    # 强制状态：猎人 (5,5)，逃脱者 (5.5,5)，距离 0.5 < capture_radius 2.0
    eng.pos[0, 0, :] = [5.0, 5.0]
    eng.pos[0, 1, :] = [5.5, 5.0]
    eng.theta[...] = 0.0
    eng.speed[...] = 0.0
    eng.active[...] = True
    eng.step_count[...] = 0
    eng._min_hunter_dist_prev = None

    act = np.zeros((1, eng.n_agents, 2))
    obs, rew, term, trunc, info = eng.step(act)
    assert bool(info["all_caught"][0])
    assert term[0].any()
    assert not trunc[0].any()


def test_vectorized_batch_shapes(minimal_cfg) -> None:
    minimal_cfg.vectorization.num_envs = 3  # type: ignore[misc]
    eng = HuntBatchEngine(minimal_cfg, num_envs=3)
    obs = eng.reset(seed=7)
    assert obs.shape[0] == 3
    act = np.zeros((3, eng.n_agents, 2))
    obs2, rew, term, trunc, info = eng.step(act)
    assert rew.shape == (3, eng.n_agents)
    assert obs2.shape == obs.shape


def test_timeout_truncation(minimal_cfg) -> None:
    minimal_cfg.sim.max_episode_steps = 3
    eng = HuntBatchEngine(minimal_cfg, num_envs=1)
    eng.reset(seed=1)
    act = np.zeros((1, eng.n_agents, 2))
    for _ in range(2):
        _, _, term, trunc, _ = eng.step(act)
        assert not term.any()
        assert not trunc.any()
    _, _, term, trunc, info = eng.step(act)
    assert bool(info["timeout"][0])
    assert trunc[0].all()
    assert not term[0].any()
