"""加载并校验 YAML 配置。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from hunt_env.config.schema import HuntEnvConfig

DEFAULT_CONFIG_ENV = "HUNT_ENV_CONFIG"


def load_config(
    path: str | Path | None = None,
    *,
    merge: dict[str, Any] | None = None,
) -> HuntEnvConfig:
    """
    从 YAML 加载配置。path 为 None 时尝试环境变量 HUNT_ENV_CONFIG，否则使用项目内 default.yaml。

    merge：深合并到 YAML 根 dict 后再做 Pydantic 校验（用于实验覆盖、单测）。勿使用 **kwargs 传嵌套键，否则键名会被误解析。
    """
    if path is None:
        env = os.environ.get(DEFAULT_CONFIG_ENV)
        if env:
            path = env
        else:
            path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if merge:
        _deep_merge(raw, merge)
    return HuntEnvConfig.model_validate(raw)


def _deep_merge(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
