"""奖励函数。"""

from __future__ import annotations

import numpy as np

from hunt_env.core.rewards import compute_step_rewards, compute_terminal_rewards


def test_terminal_only_on_just_won(minimal_cfg) -> None:
    e = 1
    nh, ne = minimal_cfg.agents.n_hunters, minimal_cfg.agents.n_escapers
    all_caught = np.array([False])
    r0 = compute_terminal_rewards(minimal_cfg, all_caught)
    assert np.allclose(r0, 0.0)
    all_caught = np.array([True])
    r1 = compute_terminal_rewards(minimal_cfg, all_caught)
    assert r1[0, 0] == minimal_cfg.rewards.hunter_win
    assert r1[0, nh] < 0  # 逃脱者惩罚


def test_step_capture_reward(minimal_cfg) -> None:
    e, ne = 1, minimal_cfg.agents.n_escapers
    prev = np.ones((e, ne), dtype=bool)
    now = np.zeros((e, ne), dtype=bool)
    min_prev = np.ones((e, ne)) * 5.0
    min_now = np.ones((e, ne)) * 0.1
    rew, jc = compute_step_rewards(minimal_cfg, prev, now, None, min_now)
    assert jc[0, 0]
    assert rew[0, 0] >= minimal_cfg.rewards.hunter_capture
