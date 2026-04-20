"""逐步奖励与终局奖励（系数来自配置）。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig


def compute_step_rewards(
    cfg: HuntEnvConfig,
    escaper_alive_prev: np.ndarray,
    escaper_alive_now: np.ndarray,
    min_hunter_dist_prev: np.ndarray | None,
    min_hunter_dist_now: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    向量化逐步奖励 (E, N) 与 just_caught (E, Ee)。

    escaper_alive_*: (E, Ee)
    min_hunter_dist_*: (E, Ee)，每个逃脱者到最近猎人的距离。
    """
    nh = cfg.agents.n_hunters
    ne = cfg.agents.n_escapers
    e_dim = escaper_alive_now.shape[0]
    rw = np.zeros((e_dim, nh + ne), dtype=np.float64)
    rc = cfg.rewards

    just_caught = escaper_alive_prev & (~escaper_alive_now)
    n_new = np.sum(just_caught, axis=-1)

    rw[:, :nh] += rc.hunter_capture * n_new[:, None]
    rw[:, nh:] += rc.escaper_caught_penalty * just_caught.astype(np.float64)

    if rc.hunter_approach_shaping_scale != 0.0 and min_hunter_dist_prev is not None:
        delta = min_hunter_dist_prev - min_hunter_dist_now
        sh = rc.hunter_approach_shaping_scale * delta * escaper_alive_now.astype(np.float64)
        s = np.sum(sh, axis=-1) / max(nh, 1)
        rw[:, :nh] += s[:, None]

    rw[:, :nh] += rc.hunter_step
    rw[:, nh:] += rc.escaper_step * escaper_alive_now.astype(np.float64)
    rw[:, nh:] += rc.escaper_survive * escaper_alive_now.astype(np.float64)

    return rw, just_caught


def compute_terminal_rewards(
    cfg: HuntEnvConfig,
    all_caught: np.ndarray,
) -> np.ndarray:
    """终局一次性奖励 (E, N)。"""
    e_dim = all_caught.shape[0]
    nh = cfg.agents.n_hunters
    rw = np.zeros((e_dim, nh + cfg.agents.n_escapers), dtype=np.float64)
    rc = cfg.rewards
    rw[:, :nh] += rc.hunter_win * all_caught[:, None]
    rw[:, nh:] += rc.escaper_all_caught_penalty * all_caught[:, None]
    return rw
