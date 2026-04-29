"""视野：独立、无队伍融合。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.visibility import visible_pair_mask


def _cfg_sector() -> HuntEnvConfig:
    return HuntEnvConfig.model_validate(
        {
            "sim": {"dt": 0.05, "max_episode_steps": 100, "seed": 0},
            "world": {"width": 20.0, "height": 20.0, "origin_x": 0.0, "origin_y": 0.0},
            "agents": {
                "n_hunters": 2,
                "n_escapers": 1,
                "hunter_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
                "escaper_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
                "spawn": {"mode": "uniform", "disk_radius_frac": 0.2},
            },
            "visibility": {"view_radius": 50.0, "use_sector_fov": True, "fov_deg": 90.0, "k_visible": 4},
            "capture": {"capture_radius": 1.0, "remove_captured": True},
        }
    )


def test_teammate_not_visible_if_far() -> None:
    """两名猎人相距很远时互不可见（同阵营亦不共享）。"""
    cfg = HuntEnvConfig.model_validate(
        {
            "sim": {"dt": 0.05, "max_episode_steps": 100, "seed": 0},
            "world": {"width": 100.0, "height": 100.0, "origin_x": 0.0, "origin_y": 0.0},
            "agents": {
                "n_hunters": 2,
                "n_escapers": 1,
                "hunter_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
                "escaper_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
                "spawn": {"mode": "uniform", "disk_radius_frac": 0.2},
            },
            "visibility": {"view_radius": 5.0, "use_sector_fov": False, "fov_deg": None, "k_visible": 4},
            "capture": {"capture_radius": 1.0, "remove_captured": True},
        }
    )
    pos = np.array([[[0.0, 0.0], [20.0, 0.0], [10.0, 0.0]]])  # h0, h1, e0
    theta = np.zeros((1, 3))
    active = np.ones((1, 3), dtype=bool)
    vis = visible_pair_mask(pos, theta, active, cfg)
    assert not vis[0, 0, 1]
    assert not vis[0, 1, 0]


def test_asymmetric_view_radius() -> None:
    """猎人/逃脱者可用不同距离圆：近距一侧可见另一侧不可见。"""
    cfg = HuntEnvConfig.model_validate(
        {
            "sim": {"dt": 0.05, "max_episode_steps": 100, "seed": 0},
            "world": {"width": 100.0, "height": 100.0, "origin_x": 0.0, "origin_y": 0.0},
            "agents": {
                "n_hunters": 1,
                "n_escapers": 1,
                "hunter_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
                "escaper_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
                "spawn": {"mode": "uniform", "disk_radius_frac": 0.2},
            },
            "visibility": {
                "view_radius": 99.0,
                "hunter_view_radius": 5.0,
                "escaper_view_radius": 10.0,
                "use_sector_fov": False,
                "fov_deg": None,
                "k_visible": 4,
            },
            "capture": {"capture_radius": 1.0, "remove_captured": True},
        }
    )
    pos = np.array([[[0.0, 0.0], [6.0, 0.0]]])
    theta = np.zeros((1, 2))
    active = np.ones((1, 2), dtype=bool)
    vis = visible_pair_mask(pos, theta, active, cfg)
    assert not vis[0, 0, 1]
    assert vis[0, 1, 0]


def test_sector_masks_behind() -> None:
    cfg = _cfg_sector()
    # 猎人 0 在原点朝右；猎人 1 在身后 (-5,0) 应在 90° FOV 外（±45° 前方）
    pos = np.array([[[0.0, 0.0], [-5.0, 0.0], [3.0, 0.0]]])
    theta = np.array([[0.0, 0.0, 0.0]])
    active = np.ones((1, 3), dtype=bool)
    vis = visible_pair_mask(pos, theta, active, cfg)
    assert not vis[0, 0, 1]
    assert vis[0, 0, 2]
