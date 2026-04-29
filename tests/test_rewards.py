"""奖励函数。"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_approach_shaping_finite_when_dead_or_placeholder(minimal_cfg) -> None:
    """死槽或占位大距离下塑形仍为有限；步初步末无存活则不求差分。"""
    minimal_cfg.rewards.hunter_approach_shaping_scale = 0.1
    e, ne = 1, minimal_cfg.agents.n_escapers
    prev = np.zeros((e, ne), dtype=bool)
    now = np.zeros((e, ne), dtype=bool)
    dp = np.full((e, ne), 1.0e6)
    dn = np.full((e, ne), 1.0e6)
    rew, _ = compute_step_rewards(minimal_cfg, prev, now, dp, dn)
    assert np.all(np.isfinite(rew))


def test_flee_shaping_sign(minimal_cfg) -> None:
    """拉远 shaping：距离增大则逃脱者得正增量。"""
    minimal_cfg.rewards.escaper_flee_shaping_scale = 0.2
    e, ne = 1, minimal_cfg.agents.n_escapers
    nh = minimal_cfg.agents.n_hunters
    prev = np.ones((e, ne), dtype=bool)
    now = np.ones((e, ne), dtype=bool)
    dp = np.full((e, ne), 3.0)
    dn = np.full((e, ne), 5.0)
    rew, _ = compute_step_rewards(minimal_cfg, prev, now, dp, dn)
    assert rew[0, nh] == pytest.approx(0.2 * 2.0)
    dn2 = np.full((e, ne), 2.0)
    rew2, _ = compute_step_rewards(minimal_cfg, prev, now, dp, dn2)
    assert rew2[0, nh] == pytest.approx(0.2 * (-1.0))


def test_step_capture_reward(minimal_cfg) -> None:
    e, ne = 1, minimal_cfg.agents.n_escapers
    prev = np.ones((e, ne), dtype=bool)
    now = np.zeros((e, ne), dtype=bool)
    min_prev = np.ones((e, ne)) * 5.0
    min_now = np.ones((e, ne)) * 0.1
    rew, jc = compute_step_rewards(minimal_cfg, prev, now, None, min_now)
    assert jc[0, 0]
    assert rew[0, 0] >= minimal_cfg.rewards.hunter_capture
