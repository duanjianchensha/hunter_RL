"""PettingZoo Parallel API 包装（单并行世界，num_envs=1）。"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from hunt_env.config.loader import load_config
from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.engine import HuntBatchEngine
from hunt_env.core.state import agent_names


class HuntParallelEnv(ParallelEnv):
    """追猎环境：配置文件驱动，底层为 HuntBatchEngine(num_envs=1)。"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "hunt_parallel_v0",
    }

    def __init__(
        self,
        config_path: str | None = None,
        *,
        cfg: HuntEnvConfig | None = None,
        render_mode: str | None = None,
    ):
        if cfg is None:
            cfg = load_config(config_path)
        self._cfg = cfg
        self.render_mode = render_mode
        self._engine = HuntBatchEngine(cfg, num_envs=1)
        self.possible_agents = agent_names(cfg.agents.n_hunters, cfg.agents.n_escapers)
        self.agents = self.possible_agents.copy()
        self._obs_dim = self._engine.obs_dim()

        self.observation_spaces = self._build_obs_spaces()
        self.action_spaces = self._build_action_spaces()
        self._renderer = None
        # max_num_agents 由 PettingZoo ParallelEnv 基类 property 提供（len(possible_agents)）

    def _build_obs_spaces(self) -> dict[str, gym.Space]:
        d = self._obs_dim
        return {a: spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float64) for a in self.possible_agents}

    def _build_action_spaces(self) -> dict[str, gym.Space]:
        cfg = self._cfg
        nh = cfg.agents.n_hunters
        spaces_dict: dict[str, gym.Space] = {}
        for i, name in enumerate(self.possible_agents):
            if i < nh:
                ma = cfg.agents.hunter_limits.max_accel
                mo = cfg.agents.hunter_limits.max_omega
            else:
                ma = cfg.agents.escaper_limits.max_accel
                mo = cfg.agents.escaper_limits.max_omega
            spaces_dict[name] = spaces.Box(
                low=np.array([-ma, -mo], dtype=np.float32),
                high=np.array([ma, mo], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
        return spaces_dict

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        obs_b = self._engine.reset(seed=seed)
        if self._renderer is not None:
            self._renderer.clear_trajectories()
        self.agents = self.possible_agents.copy()
        obs = self._split_obs(obs_b)
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        act = self._dict_actions_to_batch(actions)
        obs_b, rew_b, term_b, trunc_b, info = self._engine.step(act)
        obs = self._split_obs(obs_b)
        rewards = {a: float(rew_b[0, i]) for i, a in enumerate(self.possible_agents)}
        terminations = {a: bool(term_b[0, i]) for i, a in enumerate(self.possible_agents)}
        truncations = {a: bool(trunc_b[0, i]) for i, a in enumerate(self.possible_agents)}
        infos = {
            a: {
                "just_caught": info["just_caught"][0],
                "all_caught": bool(info["all_caught"][0]),
                "timeout": bool(info["timeout"][0]),
            }
            for a in self.possible_agents
        }
        # episode 结束：清空 agents（PettingZoo 约定）
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def _dict_actions_to_batch(self, actions: dict[str, np.ndarray]) -> np.ndarray:
        nh = self._cfg.agents.n_hunters
        ne = self._cfg.agents.n_escapers
        out = np.zeros((1, nh + ne, 2), dtype=np.float64)
        for i, name in enumerate(self.possible_agents):
            out[0, i] = np.asarray(actions[name], dtype=np.float64)
        return out

    def _split_obs(self, obs_b: np.ndarray) -> dict[str, np.ndarray]:
        return {a: obs_b[0, i].copy() for i, a in enumerate(self.possible_agents)}

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        from hunt_env.render.pygame_backend import PygameHuntRenderer

        if self._renderer is None:
            self._renderer = PygameHuntRenderer(self._cfg)
        return_rgb = self.render_mode == "rgb_array"
        return self._renderer.render(self._engine, env_index=0, return_rgb=return_rgb)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
