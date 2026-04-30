"""脚本 CLI 共享常量（与 `configs/default.yaml` 对齐；单边训练与可视化对称命名）。"""

from __future__ import annotations

from typing import Any

from hunt_env.config.loader import load_config
from hunt_env.config.schema import HuntEnvConfig

# 相对仓库根目录运行（python scripts/...）
DEFAULT_CONFIG_YAML = "configs/default.yaml"

# 可视化脚本对称默认值（公平对比：同种子、局数、默认录屏路径）
VIZ_SEED = 0
VIZ_MAX_EPISODES = 10
VIZ_MP4_HUNTER_POLICY = "runs/viz_hunter_policy.mp4"
VIZ_MP4_ESCAPER_POLICY = "runs/viz_escaper_policy.mp4"
VIZ_MP4_DUAL_POLICY = "runs/viz_dual_policy.mp4"
VIZ_MP4_RULE_BASELINE = "runs/viz_rule_baseline.mp4"


def env_cfg_for_viz(config_path: str | None, max_episode_steps: int | None) -> HuntEnvConfig:
    """
    可视化专用：物理世界、观测项、奖励系数等均以 YAML 为准（默认 default.yaml）；
    checkpoint 仅用于加载策略权重与（维数匹配时的）obs_rms。
    """
    cfg = load_config(config_path or DEFAULT_CONFIG_YAML)
    if max_episode_steps is None:
        return cfg
    return cfg.model_copy(update={"sim": cfg.sim.model_copy(update={"max_episode_steps": max_episode_steps})})


def train_env_merge(max_episode_steps: int | None, num_envs: int | None) -> dict[str, Any] | None:
    """构造 `load_config(..., merge=)`：仅包含调用方显式传入的非 None 项。"""
    merge: dict[str, Any] = {}
    if max_episode_steps is not None:
        merge["sim"] = {"max_episode_steps": max_episode_steps}
    if num_envs is not None:
        merge["vectorization"] = {"num_envs": num_envs}
    return merge if merge else None
