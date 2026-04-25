"""
状态张量布局与观测维度（单位：位置与半径为世界坐标，角度为弧度）。

单智能体观测（Box）拼接顺序：
- 自身 8 维：x, y, vx, vy, prev_a, prev_omega, theta, prev_omega_dup
  其中 prev_omega 出现两次时合并为 7 维？计划要求：位置、速度、加速度代理、角度、角速度。
  采用：x,y,vx,vy, prev_a, prev_omega, theta（7 维），角速度即 prev_omega 与 prev_omega 重复无意义。
  故自身 7 维：pos(2)+vel(2)+last_control(2)+theta(1)，角速度用 last_control 的 omega 体现。
"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig


def self_obs_dim() -> int:
    """自身观测维度。"""
    return 7


def other_slot_dim() -> int:
    """单个 Top-K 槽位：rel_x, rel_y, rvx, rvy, theta_j, mask, other_is_escaper（1/0）。"""
    return 7


def total_obs_dim(cfg: HuntEnvConfig) -> int:
    """单智能体展平观测总维度。"""
    k = cfg.visibility.k_visible
    base = self_obs_dim() + k * other_slot_dim()
    extra = 0
    if cfg.observation.include_remaining_steps:
        extra += 1
    if cfg.observation.include_captured_count:
        extra += 1
    if cfg.observation.include_world_bounds:
        extra += 4
    return base + extra


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """映射到 [-pi, pi]。"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def agent_names(n_hunters: int, n_escapers: int) -> list[str]:
    """ParallelEnv 使用的智能体名列表顺序。"""
    names = [f"hunter_{i}" for i in range(n_hunters)]
    names += [f"escaper_{j}" for j in range(n_escapers)]
    return names
