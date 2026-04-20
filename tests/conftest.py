"""pytest 公共 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest

from hunt_env.config.loader import load_config
from hunt_env.config.schema import HuntEnvConfig


@pytest.fixture
def tiny_cfg_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


@pytest.fixture
def tiny_cfg(tiny_cfg_path: str) -> HuntEnvConfig:
    return load_config(tiny_cfg_path)


@pytest.fixture
def minimal_cfg() -> HuntEnvConfig:
    """极小确定性场景：1 猎人 1 逃脱者，易测捕获与边界。"""
    return HuntEnvConfig.model_validate(
        {
            "sim": {"dt": 0.1, "max_episode_steps": 50, "seed": 123},
            "world": {"width": 10.0, "height": 10.0, "origin_x": 0.0, "origin_y": 0.0},
            "agents": {
                "n_hunters": 1,
                "n_escapers": 1,
                "hunter_limits": {"max_speed": 5.0, "max_accel": 5.0, "max_omega": 3.0},
                "escaper_limits": {"max_speed": 5.0, "max_accel": 5.0, "max_omega": 3.0},
                "spawn": {"mode": "uniform", "disk_radius_frac": 0.2},
            },
            "visibility": {"view_radius": 20.0, "use_sector_fov": False, "fov_deg": None, "k_visible": 2},
            "capture": {"capture_radius": 2.0, "remove_captured": True},
            "observation": {
                "use_ego_frame_for_others": False,
                "include_remaining_steps": True,
                "include_captured_count": True,
            },
            "rewards": {
                "hunter_step": 0.0,
                "hunter_capture": 1.0,
                "hunter_win": 10.0,
                "hunter_approach_shaping_scale": 0.0,
                "escaper_step": 0.0,
                "escaper_survive": 0.0,
                "escaper_caught_penalty": -1.0,
                "escaper_all_caught_penalty": -5.0,
            },
            "vectorization": {"num_envs": 1},
            "render": {
                "window_width": 320,
                "window_height": 320,
                "fps": 60,
                "draw_view_radius": False,
                "draw_sector": False,
            },
            "human_control": {
                "accel_step": 1.0,
                "omega_step": 1.0,
                "initial_control_agent": "hunter",
                "initial_agent_index": 0,
            },
        }
    )
