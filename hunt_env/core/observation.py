"""从仿真张量构造定长观测向量。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.state import total_obs_dim, wrap_angle
from hunt_env.core.visibility import topk_visible_indices, visible_pair_mask


def build_observations_batch(
    pos: np.ndarray,
    theta: np.ndarray,
    speed: np.ndarray,
    prev_a: np.ndarray,
    prev_omega: np.ndarray,
    active: np.ndarray,
    step_count: np.ndarray,
    cfg: HuntEnvConfig,
) -> np.ndarray:
    """
    返回 obs: (E, N, D) 展平观测。
    """
    e, n, _ = pos.shape
    vx = speed * np.cos(theta)
    vy = speed * np.sin(theta)

    self_part = np.stack(
        [
            pos[..., 0],
            pos[..., 1],
            vx,
            vy,
            prev_a,
            prev_omega,
            wrap_angle(theta),
        ],
        axis=-1,
    )  # (E, N, 7)

    vis = visible_pair_mask(pos, theta, active, cfg)
    _, idx_top = topk_visible_indices(pos, active, vis, cfg.visibility.k_visible)
    k = cfg.visibility.k_visible
    nh = cfg.agents.n_hunters

    other = np.zeros((e, n, k, 7), dtype=np.float64)
    for ei in range(e):
        for i in range(n):
            for t in range(k):
                j = int(idx_top[ei, i, t])
                if j < 0:
                    continue
                dx = pos[ei, j, 0] - pos[ei, i, 0]
                dy = pos[ei, j, 1] - pos[ei, i, 1]
                vjx = speed[ei, j] * np.cos(theta[ei, j])
                vjy = speed[ei, j] * np.sin(theta[ei, j])
                if cfg.observation.use_ego_frame_for_others:
                    c = np.cos(theta[ei, i])
                    s = np.sin(theta[ei, i])
                    rx = c * dx + s * dy
                    ry = -s * dx + c * dy
                    rvx = c * vjx + s * vjy
                    rvy = -s * vjx + c * vjy
                else:
                    rx, ry, rvx, rvy = dx, dy, vjx, vjy
                other[ei, i, t, 0] = rx
                other[ei, i, t, 1] = ry
                other[ei, i, t, 2] = rvx
                other[ei, i, t, 3] = rvy
                other[ei, i, t, 4] = float(wrap_angle(np.asarray(theta[ei, j], dtype=np.float64)))
                other[ei, i, t, 5] = 1.0
                other[ei, i, t, 6] = 1.0 if j >= nh else 0.0

    other_flat = other.reshape(e, n, k * 7)

    parts = [self_part, other_flat]
    if cfg.observation.include_remaining_steps:
        rem = (cfg.sim.max_episode_steps - step_count[:, None]).astype(np.float64) / float(
            cfg.sim.max_episode_steps
        )
        rem = np.broadcast_to(rem, (e, n))
        parts.append(rem[..., None])
    if cfg.observation.include_captured_count:
        nh = cfg.agents.n_hunters
        ne = cfg.agents.n_escapers
        escapers = active[:, nh:]
        caught_frac = (ne - np.sum(escapers.astype(np.float64), axis=-1)) / max(float(ne), 1.0)
        caught_frac = np.broadcast_to(caught_frac[:, None], (e, n))
        parts.append(caught_frac[..., None])

    if cfg.observation.include_world_bounds:
        wcfg = cfg.world
        ox = wcfg.origin_x
        oy = wcfg.origin_y
        ww = wcfg.width
        hh = wcfg.height
        x = pos[..., 0]
        y = pos[..., 1]
        d_west = x - ox
        d_east = (ox + ww) - x
        d_south = y - oy
        d_north = (oy + hh) - y
        wall = np.stack([d_west, d_east, d_south, d_north], axis=-1)
        parts.append(wall)

    return np.concatenate(parts, axis=-1)


def assert_obs_dim(cfg: HuntEnvConfig) -> int:
    d = total_obs_dim(cfg)
    return d
