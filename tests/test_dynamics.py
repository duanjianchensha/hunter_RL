"""动力学与边界。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import WorldConfig
from hunt_env.core.dynamics import clip_actions, step_unicycle_batch


def test_clip_actions_broadcast() -> None:
    a = np.array([[10.0, -10.0]])
    w = np.array([[5.0, -5.0]])
    max_a = np.array([3.0, 3.0])
    max_w = np.array([2.0, 2.0])
    ac, wc = clip_actions(a, w, max_a, max_w)
    np.testing.assert_array_almost_equal(ac, [[3.0, -3.0]])
    np.testing.assert_array_almost_equal(wc, [[2.0, -2.0]])


def test_boundary_no_escape() -> None:
    """矩形内积分：强加速度撞墙后位置不越界。"""
    world = WorldConfig(width=10.0, height=10.0, origin_x=0.0, origin_y=0.0)
    e, n = 1, 1
    pos = np.array([[[9.9, 5.0]]])
    theta = np.zeros((e, n))
    speed = np.array([[50.0]])
    active = np.ones((e, n), dtype=bool)
    a_cmd = np.zeros((e, n))
    w_cmd = np.zeros((e, n))
    max_speed = np.array([100.0])
    dt = 0.1
    for _ in range(20):
        pos, theta, speed = step_unicycle_batch(
            pos, theta, speed, active, a_cmd, w_cmd, max_speed, dt, world
        )
    assert pos[0, 0, 0] <= 10.0 + 1e-6
    assert pos[0, 0, 0] >= 0.0 - 1e-6
