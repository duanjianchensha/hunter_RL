"""
与 `core/observation.build_observations_batch` 一致的展平观测切片，供规则策略仅依赖向量解码（与 RL 同输入布局）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.state import total_obs_dim


@dataclass(frozen=True)
class DecodedObs:
    """单智能体一帧观测解码结果。"""

    self_vec: np.ndarray  # (7,) x,y,vx,vy,prev_a,prev_omega,theta
    slots: np.ndarray  # (k, 7) 每行 rel_x,rel_y,rvx,rvy,theta_j,mask,other_is_escaper
    remaining_frac: float | None
    caught_frac: float | None
    # 西、东、下、上边界到自车的距离（世界系），与 state 中 total_obs 顺序一致
    wall_dist: np.ndarray | None  # (4,)


def decode_observation(obs: np.ndarray, cfg: HuntEnvConfig) -> DecodedObs:
    """
    将一维展平 `obs` 按当前 `cfg` 切回字段；长度须等于 `hunt_env.core.state.total_obs_dim(cfg)`。
    """
    flat = np.asarray(obs, dtype=np.float64).ravel()
    expected = total_obs_dim(cfg)
    n = len(flat)
    if n != expected:
        raise ValueError(
            f"obs 维数 {n} 与当前配置不一致，期望 {expected}（= total_obs_dim(cfg)）；"
            f"请检查 include_remaining_steps / include_captured_count / include_world_bounds 与向量是否同源"
        )

    k = cfg.visibility.k_visible
    i = 0

    self_vec = flat[i : i + 7].copy()
    i += 7
    slots = flat[i : i + k * 7].reshape(k, 7).copy()
    i += k * 7

    rem: float | None = None
    if cfg.observation.include_remaining_steps:
        if i >= n:
            raise ValueError("obs 缺少 remaining_steps 1 维，与 include_remaining_steps 的配置不一致")
        rem = float(flat[i])
        i += 1
    caught: float | None = None
    if cfg.observation.include_captured_count:
        if i >= n:
            raise ValueError("obs 缺少 caught_frac 1 维，与 include_captured_count 的配置不一致")
        caught = float(flat[i])
        i += 1
    wall: np.ndarray | None = None
    if cfg.observation.include_world_bounds:
        if i + 4 > n:
            raise ValueError("obs 缺少世界边界 4 维，与 include_world_bounds 或 total_obs 布局不一致")
        wall = flat[i : i + 4].copy()
        i += 4

    if i != len(flat):
        raise ValueError(f"obs 与配置不匹配: 已解析 {i} 维, 余长 {len(flat) - i}")

    return DecodedObs(
        self_vec=self_vec,
        slots=slots,
        remaining_frac=rem,
        caught_frac=caught,
        wall_dist=wall,
    )


def rel_to_world_delta(rx: float, ry: float, theta_self: float, use_ego_frame: bool) -> tuple[float, float]:
    """
    将「自车指向他车」的相对位置从机体/世界量转为**世界系** (dx, dy) = 他车-自车（与无 ego 时 obs 中含义一致）。
    """
    if not use_ego_frame:
        return float(rx), float(ry)
    c, s = float(np.cos(theta_self)), float(np.sin(theta_self))
    wx = c * rx - s * ry
    wy = s * rx + c * ry
    return wx, wy
