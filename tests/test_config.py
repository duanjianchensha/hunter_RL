"""配置加载与 Pydantic 校验。"""

from __future__ import annotations

import pytest

from hunt_env.config.loader import load_config
from hunt_env.config.schema import HuntEnvConfig


def test_load_default_yaml(tiny_cfg_path: str) -> None:
    cfg = load_config(tiny_cfg_path)
    assert cfg.agents.n_hunters >= 1
    assert cfg.world.width > 0


def test_merge_override() -> None:
    cfg = load_config(merge={"sim": {"max_episode_steps": 999}})
    assert cfg.sim.max_episode_steps == 999


def test_sector_requires_fov() -> None:
    raw = {
        "sim": {"dt": 0.05, "max_episode_steps": 100, "seed": None},
        "world": {"width": 10.0, "height": 10.0, "origin_x": 0.0, "origin_y": 0.0},
        "agents": {
            "n_hunters": 1,
            "n_escapers": 1,
            "hunter_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
            "escaper_limits": {"max_speed": 1.0, "max_accel": 1.0, "max_omega": 1.0},
            "spawn": {"mode": "uniform", "disk_radius_frac": 0.2},
        },
        "visibility": {"view_radius": 5.0, "use_sector_fov": True, "fov_deg": None, "k_visible": 4},
        "capture": {"capture_radius": 1.0, "remove_captured": True},
        "observation": {},
        "rewards": {},
        "vectorization": {"num_envs": 1},
        "render": {},
        "human_control": {},
    }
    with pytest.raises(Exception):
        HuntEnvConfig.model_validate(raw)


def test_model_validate_nested_defaults() -> None:
    d = {
        "sim": {"dt": 0.1, "max_episode_steps": 10, "seed": 0},
        "world": {"width": 5.0, "height": 5.0, "origin_x": 0.0, "origin_y": 0.0},
        "agents": {
            "n_hunters": 1,
            "n_escapers": 1,
            "hunter_limits": {"max_speed": 2.0, "max_accel": 2.0, "max_omega": 2.0},
            "escaper_limits": {"max_speed": 2.0, "max_accel": 2.0, "max_omega": 2.0},
            "spawn": {"mode": "uniform", "disk_radius_frac": 0.3},
        },
        "visibility": {"view_radius": 10.0, "use_sector_fov": False, "fov_deg": None, "k_visible": 3},
        "capture": {"capture_radius": 0.5, "remove_captured": True},
    }
    cfg = HuntEnvConfig.model_validate(d)
    assert cfg.observation.include_remaining_steps is True
