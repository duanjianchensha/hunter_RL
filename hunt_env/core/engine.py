"""向量化仿真核心：reset/step、捕获、观测与奖励。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.dynamics import clip_actions, step_unicycle_batch
from hunt_env.core.observation import assert_obs_dim, build_observations_batch
from hunt_env.core.rewards import compute_step_rewards, compute_terminal_rewards
from hunt_env.core.state import agent_names
from hunt_env.core.visibility import visible_pair_mask


def _min_hunter_escaper_dist(pos: np.ndarray, nh: int) -> np.ndarray:
    """(E, Ee) 每个逃脱者到最近猎人的距离。"""
    hunt = pos[:, :nh, :]
    esc = pos[:, nh:, :]
    # (E, Ee, 1, 2) - (E, 1, nh, 2)
    diff = esc[:, :, None, :] - hunt[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    return np.min(d, axis=-1)


def _pairwise_min_dist(pos_row: np.ndarray) -> float:
    """单行 (N,2) 位置两两最小距离。"""
    n = pos_row.shape[0]
    if n < 2:
        return np.inf
    dmin = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos_row[i] - pos_row[j]))
            dmin = min(dmin, d)
    return dmin


def _spawn_positions(rng: np.random.Generator, cfg: HuntEnvConfig, e: int, n: int) -> np.ndarray:
    """(E, N, 2)，带两两最小间距，避免首帧即进入 capture_radius 导致立刻终局。"""
    wcfg = cfg.world
    ox, oy = wcfg.origin_x, wcfg.origin_y
    w, h = wcfg.width, wcfg.height
    mode = cfg.agents.spawn.mode
    sep = cfg.agents.spawn.min_pairwise_separation
    if sep is None:
        sep = max(cfg.capture.capture_radius * 1.5, 0.5)
    out = np.empty((e, n, 2), dtype=np.float64)
    max_tries = 4000

    for ei in range(e):
        for _ in range(max_tries):
            if mode == "uniform":
                row = np.stack(
                    [
                        rng.uniform(ox, ox + w, size=n),
                        rng.uniform(oy, oy + h, size=n),
                    ],
                    axis=-1,
                )
            else:
                cx = ox + w * 0.5
                cy = oy + h * 0.5
                rmax = min(w, h) * cfg.agents.spawn.disk_radius_frac
                t = rng.uniform(0.0, 1.0, size=n)
                ang = rng.uniform(-np.pi, np.pi, size=n)
                rad = np.sqrt(t) * rmax
                row = np.stack(
                    [cx + rad * np.cos(ang), cy + rad * np.sin(ang)],
                    axis=-1,
                )
                row[:, 0] = np.clip(row[:, 0], ox, ox + w)
                row[:, 1] = np.clip(row[:, 1], oy, oy + h)
            if _pairwise_min_dist(row) >= sep - 1e-9:
                out[ei] = row
                break
        else:
            # 极难满足时退化为无间距采样，避免死循环（小地图多智能体需调大场地或减小人数）
            if mode == "uniform":
                out[ei, :, 0] = rng.uniform(ox, ox + w, size=n)
                out[ei, :, 1] = rng.uniform(oy, oy + h, size=n)
            else:
                cx = ox + w * 0.5
                cy = oy + h * 0.5
                rmax = min(w, h) * cfg.agents.spawn.disk_radius_frac
                t = rng.uniform(0.0, 1.0, size=n)
                ang = rng.uniform(-np.pi, np.pi, size=n)
                rad = np.sqrt(t) * rmax
                out[ei, :, 0] = cx + rad * np.cos(ang)
                out[ei, :, 1] = cy + rad * np.sin(ang)
                out[ei, :, 0] = np.clip(out[ei, :, 0], ox, ox + w)
                out[ei, :, 1] = np.clip(out[ei, :, 1], oy, oy + h)
    return out


@dataclass
class HuntBatchEngine:
    """支持 n_envs 并行环境，同一套 HuntEnvConfig。"""

    cfg: HuntEnvConfig
    num_envs: int

    def __post_init__(self) -> None:
        self.nh = self.cfg.agents.n_hunters
        self.ne = self.cfg.agents.n_escapers
        self.n_agents = self.nh + self.ne
        self.agent_ids = agent_names(self.nh, self.ne)
        self._obs_dim = assert_obs_dim(self.cfg)

        self.rng = np.random.default_rng(self.cfg.sim.seed)

        self.pos = np.zeros((self.num_envs, self.n_agents, 2), dtype=np.float64)
        self.theta = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        self.speed = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        self.prev_a = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        self.prev_omega = np.zeros((self.num_envs, self.n_agents), dtype=np.float64)
        self.active = np.ones((self.num_envs, self.n_agents), dtype=bool)
        self.step_count = np.zeros(self.num_envs, dtype=np.int32)

        self._max_speed = np.empty(self.n_agents, dtype=np.float64)
        self._max_accel = np.empty(self.n_agents, dtype=np.float64)
        self._max_omega = np.empty(self.n_agents, dtype=np.float64)
        self._max_speed[: self.nh] = self.cfg.agents.hunter_limits.max_speed
        self._max_accel[: self.nh] = self.cfg.agents.hunter_limits.max_accel
        self._max_omega[: self.nh] = self.cfg.agents.hunter_limits.max_omega
        self._max_speed[self.nh :] = self.cfg.agents.escaper_limits.max_speed
        self._max_accel[self.nh :] = self.cfg.agents.escaper_limits.max_accel
        self._max_omega[self.nh :] = self.cfg.agents.escaper_limits.max_omega

        self._min_hunter_dist_prev: np.ndarray | None = None

    def obs_dim(self) -> int:
        return self._obs_dim

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.cfg.sim.seed is not None:
            self.rng = np.random.default_rng(self.cfg.sim.seed)

        e = self.num_envs
        self.pos[...] = _spawn_positions(self.rng, self.cfg, e, self.n_agents)
        self.theta[...] = self.rng.uniform(-np.pi, np.pi, size=(e, self.n_agents))
        self.speed[...] = 0.0
        self.prev_a[...] = 0.0
        self.prev_omega[...] = 0.0
        self.active[...] = True
        self.step_count[...] = 0
        self._min_hunter_dist_prev = _min_hunter_escaper_dist(self.pos, self.nh)
        return build_observations_batch(
            self.pos,
            self.theta,
            self.speed,
            self.prev_a,
            self.prev_omega,
            self.active,
            self.step_count,
            self.cfg,
        )

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        actions: (E, N, 2) 线加速度、角速度

        返回 obs (E,N,D), rew (E,N), term (E,N) bool, trunc (E,N) bool, info
        """
        cfg = self.cfg
        dt = cfg.sim.dt
        nh, ne = self.nh, self.ne
        e = self.num_envs

        escaper_alive_prev = self.active[:, nh:].copy()

        min_prev = self._min_hunter_dist_prev
        if min_prev is None:
            min_prev = _min_hunter_escaper_dist(self.pos, nh)
            min_prev = np.where(escaper_alive_prev, min_prev, np.inf)

        a_cmd = actions[..., 0]
        w_cmd = actions[..., 1]
        a_cmd, w_cmd = clip_actions(a_cmd, w_cmd, self._max_accel, self._max_omega)

        # 非活跃智能体不移动
        a_cmd = np.where(self.active, a_cmd, 0.0)
        w_cmd = np.where(self.active, w_cmd, 0.0)

        self.pos, self.theta, self.speed = step_unicycle_batch(
            self.pos,
            self.theta,
            self.speed,
            self.active,
            a_cmd,
            w_cmd,
            self._max_speed,
            dt,
            cfg.world,
        )
        self.prev_a = np.where(self.active, a_cmd, self.prev_a)
        self.prev_omega = np.where(self.active, w_cmd, self.prev_omega)

        min_now = _min_hunter_escaper_dist(self.pos, nh)
        min_now = np.where(escaper_alive_prev, min_now, np.inf)

        # 捕获：仅仍存活的逃脱者
        cap_r = cfg.capture.capture_radius
        caught_now = escaper_alive_prev & (min_now <= cap_r)
        if cfg.capture.remove_captured:
            self.active[:, nh:] &= ~caught_now
        else:
            # 不移除则只标记速度清零（仍占空间）— 简化仍用 active False
            self.active[:, nh:] &= ~caught_now

        escaper_alive_now = self.active[:, nh:]
        # 被捕获后速度清零（便于观测与渲染）
        self.speed[:, nh:] = np.where(self.active[:, nh:], self.speed[:, nh:], 0.0)

        rew, just_caught = compute_step_rewards(
            cfg,
            escaper_alive_prev,
            escaper_alive_now,
            min_prev if cfg.rewards.hunter_approach_shaping_scale != 0.0 else None,
            np.where(escaper_alive_prev, min_now, np.inf),
        )

        self._min_hunter_dist_prev = _min_hunter_escaper_dist(self.pos, nh)

        self.step_count += 1
        max_steps = cfg.sim.max_episode_steps
        all_caught = np.all(~escaper_alive_now, axis=-1)
        timeout = self.step_count >= max_steps

        # 终局奖励：仅在「本步首次达成全捕获」时发放，避免 all_caught 在后续步重复刷分
        had_alive_escaper = np.any(escaper_alive_prev, axis=-1)
        just_all_caught = all_caught & had_alive_escaper
        rew += compute_terminal_rewards(cfg, just_all_caught)

        # Gymnasium 语义：自然结束 termination；限时 truncation
        term = np.zeros((e, self.n_agents), dtype=bool)
        trunc = np.zeros((e, self.n_agents), dtype=bool)
        term[:, :] = all_caught[:, None]
        trunc[:, :] = (timeout & (~all_caught))[:, None]

        obs = build_observations_batch(
            self.pos,
            self.theta,
            self.speed,
            self.prev_a,
            self.prev_omega,
            self.active,
            self.step_count,
            cfg,
        )

        info = {
            "just_caught": just_caught,
            "all_caught": all_caught,
            "timeout": timeout,
            "visibility": visible_pair_mask(self.pos, self.theta, self.active, cfg),
        }
        return obs, rew, term, trunc, info
