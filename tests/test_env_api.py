"""PettingZoo ParallelEnv 与向量化 API。"""

from __future__ import annotations

import numpy as np

from hunt_env.env.hunt_parallel import HuntParallelEnv
from hunt_env.env.vectorized import HuntVectorizedEnv


def test_parallel_reset_step(minimal_cfg) -> None:
    env = HuntParallelEnv(cfg=minimal_cfg, render_mode=None)
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == set(env.possible_agents)
    actions = {a: np.zeros(2, dtype=np.float32) for a in env.possible_agents}
    obs2, rew, term, trunc, infos2 = env.step(actions)
    assert len(rew) == len(env.possible_agents)
    env.close()


def test_vectorized_from_config(minimal_cfg) -> None:
    minimal_cfg.vectorization.num_envs = 2  # type: ignore[misc]
    v = HuntVectorizedEnv(cfg=minimal_cfg, num_envs=2)
    obs = v.reset(seed=0)
    a = np.zeros((2, v.n_agents, 2))
    obs2, r, t, tr, _ = v.step(a)
    assert obs2.shape == obs.shape == (2, v.n_agents, v.obs_dim)
