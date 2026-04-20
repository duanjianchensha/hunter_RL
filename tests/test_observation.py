"""观测向量维度一致。"""

from __future__ import annotations

import numpy as np

from hunt_env.core.observation import build_observations_batch
from hunt_env.core.state import total_obs_dim


def test_obs_dim_matches_tensor(minimal_cfg) -> None:
    d = total_obs_dim(minimal_cfg)
    e, n = 1, minimal_cfg.agents.n_hunters + minimal_cfg.agents.n_escapers
    pos = np.zeros((e, n, 2))
    theta = np.zeros((e, n))
    speed = np.zeros((e, n))
    prev_a = np.zeros((e, n))
    prev_omega = np.zeros((e, n))
    active = np.ones((e, n), dtype=bool)
    step_count = np.zeros(e, dtype=np.int32)
    obs = build_observations_batch(pos, theta, speed, prev_a, prev_omega, active, step_count, minimal_cfg)
    assert obs.shape[-1] == d
