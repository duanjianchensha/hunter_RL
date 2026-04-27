"""
多智能体 PPO：猎人共用一套 ActorCritic，逃脱者共用一套。
向量化环境 (E, N, D) 上采集 rollout；episode 结束时对整批环境 reset（与当前 HuntBatchEngine 一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_env.policies.rules import rule_action_escaper
from hunt_rl.actor_critic import ActorCritic, action_bounds_from_cfg
from hunt_rl.device import get_train_device
from hunt_rl.running_stats import RunningMeanStd, RunningRewardRMS


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    rewards: (T, E)，values: (T+1, E)，dones: (T, E)。
    """
    t_len, n_e = rewards.shape
    adv = np.zeros((t_len, n_e), dtype=np.float64)
    last_gae = np.zeros(n_e, dtype=np.float64)
    for t in reversed(range(t_len)):
        mask = 1.0 - dones[t].astype(np.float64)
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    ret = adv + values[:-1]
    return adv.astype(np.float32), ret.astype(np.float32)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 4
    normalize_advantage: bool = True
    # 在线统计：采集步用 raw 更新 RMS，前向用 normalize+clip；PPO 更新时仅 normalize（不重复 update）
    normalize_obs: bool = True
    normalize_reward: bool = True
    obs_clip: float = 10.0


class MultiAgentPPOTrainer:
    def __init__(
        self,
        cfg: HuntEnvConfig,
        vec_env: HuntVectorizedEnv,
        *,
        ppo_cfg: PPOConfig | None = None,
        hidden_sizes: tuple[int, ...] = (256, 256),
        device: torch.device | None = None,
        use_cuda_if_available: bool = True,
        escaper_mode: Literal["learn", "rule"] = "learn",
    ):
        self.cfg = cfg
        self.env = vec_env
        self.nh = cfg.agents.n_hunters
        self.ne = cfg.agents.n_escapers
        self.n_agents = vec_env.n_agents
        self.obs_dim = vec_env.obs_dim
        self.ppo = ppo_cfg or PPOConfig()
        self.device = device or get_train_device(use_cuda_if_available)
        if escaper_mode not in ("learn", "rule"):
            raise ValueError("escaper_mode 应为 'learn' 或 'rule'")
        if escaper_mode == "rule" and self.nh < 1:
            raise ValueError("escaper_mode=rule 时需至少 1 名猎人用于 RL")
        # 逃脱者用规则控制时，不建策略网，仅 PPO 训练猎人
        self._use_rule_escaper: bool = escaper_mode == "rule" and self.ne > 0

        self.policies: dict[str, ActorCritic] = {}
        self.optimizers: dict[str, optim.Optimizer] = {}
        if self.nh > 0:
            self.policies["hunter"] = ActorCritic(self.obs_dim, 2, hidden_sizes).to(self.device)
            self.optimizers["hunter"] = optim.Adam(self.policies["hunter"].parameters(), lr=self.ppo.lr)
        if self.ne > 0 and not self._use_rule_escaper:
            self.policies["escaper"] = ActorCritic(self.obs_dim, 2, hidden_sizes).to(self.device)
            self.optimizers["escaper"] = optim.Adam(self.policies["escaper"].parameters(), lr=self.ppo.lr)

        self._al_h = torch.tensor(action_bounds_from_cfg(cfg, "hunter")[0], device=self.device, dtype=torch.float32)
        self._ah_h = torch.tensor(action_bounds_from_cfg(cfg, "hunter")[1], device=self.device, dtype=torch.float32)
        self._al_e = torch.tensor(action_bounds_from_cfg(cfg, "escaper")[0], device=self.device, dtype=torch.float32)
        self._ah_e = torch.tensor(action_bounds_from_cfg(cfg, "escaper")[1], device=self.device, dtype=torch.float32)

        self._lo_np = np.stack(
            [action_bounds_from_cfg(cfg, "hunter" if i < self.nh else "escaper")[0] for i in range(self.n_agents)]
        )
        self._hi_np = np.stack(
            [action_bounds_from_cfg(cfg, "hunter" if i < self.nh else "escaper")[1] for i in range(self.n_agents)]
        )

        self._obs_rms: dict[str, RunningMeanStd] = {}
        self._rew_rms: dict[str, RunningRewardRMS] = {}
        if self.ppo.normalize_obs:
            if self.nh > 0:
                self._obs_rms["hunter"] = RunningMeanStd(self.obs_dim)
            if self.ne > 0 and not self._use_rule_escaper:
                self._obs_rms["escaper"] = RunningMeanStd(self.obs_dim)
        if self.ppo.normalize_reward:
            if self.nh > 0:
                self._rew_rms["hunter"] = RunningRewardRMS()
            if self.ne > 0 and not self._use_rule_escaper:
                self._rew_rms["escaper"] = RunningRewardRMS()

    @staticmethod
    def _obs_role(ni: int, nh: int) -> str:
        return "hunter" if ni < nh else "escaper"

    def _obs_for_policy_forward(self, raw: np.ndarray, agent_idx: int) -> np.ndarray:
        """
        采集时：用 raw 更新统计量，再返回 normalize+clip 后的观测给策略网络。
        raw: (E, d) 或 (T,E,d) 等，最后一维为 d。
        """
        r = np.asarray(raw, dtype=np.float32)
        if not self.ppo.normalize_obs:
            return r
        role = self._obs_role(agent_idx, self.nh)
        m = self._obs_rms[role]
        m.update(r.astype(np.float64, copy=False))
        o = m.normalize(r)
        c = self.ppo.obs_clip
        return np.clip(o, -c, c).astype(np.float32, copy=False)

    def _obs_for_value_bootstrap(self, raw: np.ndarray, agent_idx: int) -> np.ndarray:
        """
        步末用 o_{t+1} 估 V 时**不** update（下一步 act 时会对同一条 o 做 update+norm，避免 Welford 双计）。
        """
        return self._obs_for_ppo_replay(raw, agent_idx)

    def _obs_for_ppo_replay(self, raw: np.ndarray, agent_idx: int) -> np.ndarray:
        """PPO 小批量内：不再次 update，仅 normalize+clip。"""
        r = np.asarray(raw, dtype=np.float32)
        if not self.ppo.normalize_obs:
            return r
        role = self._obs_role(agent_idx, self.nh)
        o = self._obs_rms[role].normalize(r)
        c = self.ppo.obs_clip
        return np.clip(o, -c, c).astype(np.float32, copy=False)

    def _rew_for_gae(self, r_raw: np.ndarray, agent_idx: int) -> np.ndarray:
        """(T, E) 原奖励；用于 GAE 的归一化后奖励。"""
        r = np.asarray(r_raw, dtype=np.float32)
        if not self.ppo.normalize_reward:
            return r
        role = self._obs_role(agent_idx, self.nh)
        return self._rew_rms[role].normalize(r)

    def _bounds(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < self.nh:
            return self._al_h, self._ah_h
        return self._al_e, self._ah_e

    def _policy(self, idx: int) -> ActorCritic:
        return self.policies["hunter"] if idx < self.nh else self.policies["escaper"]

    def collect_rollout(self, num_steps: int, obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        e = self.env.num_envs
        n = self.n_agents
        d = self.obs_dim

        obs_buf = np.zeros((num_steps + 1, e, n, d), dtype=np.float32)
        act_buf = np.zeros((num_steps, e, n, 2), dtype=np.float32)  # 高斯采样原始动作（未裁剪），与 logp 一致
        logp_buf = np.zeros((num_steps, e, n), dtype=np.float32)
        rew_buf = np.zeros((num_steps, e, n), dtype=np.float32)
        # 与逐步时刻一致：每步先 `normalize(当前 rms)` 再 `update`，与 val 的 GAE 因果对齐
        rew_norm_buf = np.zeros((num_steps, e, n), dtype=np.float32)
        val_buf = np.zeros((num_steps + 1, e, n), dtype=np.float32)
        done_buf = np.zeros((num_steps, e), dtype=np.bool_)

        obs_buf[0] = obs
        for t in range(num_steps):
            actions = np.zeros((e, n, 2), dtype=np.float32)
            for ni in range(n):
                if self._use_rule_escaper and ni >= self.nh:
                    for ei in range(e):
                        raw_a = rule_action_escaper(obs_buf[t, ei, ni], self.cfg)
                        act_buf[t, ei, ni, 0] = float(raw_a[0])
                        act_buf[t, ei, ni, 1] = float(raw_a[1])
                    actions[:, ni, 0] = act_buf[t, :, ni, 0]
                    actions[:, ni, 1] = act_buf[t, :, ni, 1]
                    logp_buf[t, :, ni] = 0.0
                    val_buf[t, :, ni] = 0.0
                    continue

                pol = self._policy(ni)
                lo, hi = self._bounds(ni)
                o_in = self._obs_for_policy_forward(obs_buf[t, :, ni, :], ni)
                ot = torch.as_tensor(o_in, device=self.device)
                with torch.no_grad():
                    a_env, logp, val, raw = pol.act(ot, lo, hi)
                    act_buf[t, :, ni, :] = raw.detach().cpu().numpy()
                    logp_buf[t, :, ni] = logp.detach().cpu().numpy()
                    val_buf[t, :, ni] = val.detach().cpu().numpy()
                actions[:, ni, :] = a_env.detach().cpu().numpy()

            next_obs, rew, term, trunc, _ = self.env.step(actions)
            rew_buf[t] = rew
            rew_norm_buf[t] = np.asarray(rew, dtype=np.float32, copy=True)
            if self.ppo.normalize_reward and self.nh > 0:
                h = rew[:, : self.nh]
                rew_norm_buf[t, :, : self.nh] = self._rew_rms["hunter"].normalize(
                    h.astype(np.float32, copy=False)
                )
                self._rew_rms["hunter"].update(h.astype(np.float64, copy=False))
            if self.ppo.normalize_reward and self.ne > 0 and not self._use_rule_escaper:
                ep = rew[:, self.nh :]
                rew_norm_buf[t, :, self.nh :] = self._rew_rms["escaper"].normalize(
                    ep.astype(np.float32, copy=False)
                )
                self._rew_rms["escaper"].update(ep.astype(np.float64, copy=False))
            done_buf[t] = np.logical_or(term[:, 0], trunc[:, 0])
            # 单并行：终局后立即 reset，避免在吸收态上继续采样。
            # 多并行：仅当本步所有 env 均终局时整批 reset（否则保留 step 输出；GAE 用 done 掩码）。
            if self.env.num_envs == 1:
                if done_buf[t, 0]:
                    next_obs = self.env.reset(seed=None)
            elif np.all(done_buf[t]):
                next_obs = self.env.reset(seed=None)

            obs_buf[t + 1] = next_obs

            with torch.no_grad():
                for ni in range(n):
                    if self._use_rule_escaper and ni >= self.nh:
                        val_buf[t + 1, :, ni] = 0.0
                        continue
                    pol = self._policy(ni)
                    o2_in = self._obs_for_value_bootstrap(obs_buf[t + 1, :, ni, :], ni)
                    o2 = torch.as_tensor(o2_in, device=self.device)
                    _, _, v2 = pol(o2)
                    val_buf[t + 1, :, ni] = v2.cpu().numpy()

        return obs_buf[-1].copy(), {
            "obs": obs_buf,
            "actions": act_buf,
            "logp": logp_buf,
            "rew": rew_buf,
            "rew_norm": rew_norm_buf,
            "val": val_buf,
            "done": done_buf,
        }

    def ppo_update_agent(self, storage: dict[str, Any], agent_idx: int) -> dict[str, float]:
        pol = self._policy(agent_idx)
        opt = self.optimizers["hunter" if agent_idx < self.nh else "escaper"]
        lo, hi = self._bounds(agent_idx)

        obs = storage["obs"][:-1, :, agent_idx, :]
        obs = self._obs_for_ppo_replay(obs, agent_idx)
        act = storage["actions"][:, :, agent_idx, :]
        old_logp = storage["logp"][:, :, agent_idx]
        if self.ppo.normalize_reward and "rew_norm" in storage:
            rew = storage["rew_norm"][:, :, agent_idx]
        else:
            rew = self._rew_for_gae(storage["rew"][:, :, agent_idx], agent_idx)
        val = storage["val"][:, :, agent_idx]
        done = storage["done"]

        adv, ret = compute_gae(rew, val, done, self.ppo.gamma, self.ppo.gae_lambda)
        if self.ppo.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        t_len, n_e, dim = obs.shape
        flat_obs = obs.reshape(-1, dim)
        flat_act = act.reshape(-1, 2)
        flat_old_logp = old_logp.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_ret = ret.reshape(-1)

        batch_size = flat_obs.shape[0]
        mb = max(batch_size // self.ppo.num_minibatches, 1)
        idx = np.arange(batch_size)
        stats = {"loss": 0.0, "pg": 0.0, "v": 0.0, "ent": 0.0}
        n_upd = 0

        for _ in range(self.ppo.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb):
                end = min(start + mb, batch_size)
                b = idx[start:end]
                ob = torch.as_tensor(flat_obs[b], device=self.device)
                ac = torch.as_tensor(flat_act[b], device=self.device)
                old_lp = torch.as_tensor(flat_old_logp[b], device=self.device)
                ad = torch.as_tensor(flat_adv[b], device=self.device)
                rt = torch.as_tensor(flat_ret[b], device=self.device)

                new_lp, entropy, vpred = pol.evaluate(ob, ac, lo, hi)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1.0 - self.ppo.clip_coef, 1.0 + self.ppo.clip_coef) * ad
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * ((vpred - rt) ** 2).mean()
                ent_loss = -entropy.mean()
                loss = pg_loss + self.ppo.vf_coef * v_loss + self.ppo.ent_coef * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), self.ppo.max_grad_norm)
                opt.step()

                stats["loss"] += float(loss.detach().cpu())
                stats["pg"] += float(pg_loss.detach().cpu())
                stats["v"] += float(v_loss.detach().cpu())
                stats["ent"] += float((-ent_loss).detach().cpu())
                n_upd += 1

        for k in stats:
            stats[k] /= max(n_upd, 1)
        return stats

    def train_step(
        self, num_steps: int, obs: np.ndarray
    ) -> tuple[np.ndarray, list[dict[str, float]], dict[str, float]]:
        next_obs, storage = self.collect_rollout(num_steps, obs)
        metrics: dict[str, float] = {}
        if self.nh > 0:
            rh = storage["rew"][:, :, : self.nh]
            metrics["hunter_rew_mean"] = float(np.mean(rh))
            metrics["hunter_rew_sum"] = float(np.sum(rh))
        if self.ne > 0:
            re = storage["rew"][:, :, self.nh :]
            metrics["escaper_rew_mean"] = float(np.mean(re))

        logs: list[dict[str, float]] = []
        update_range = range(self.nh) if self._use_rule_escaper else range(self.n_agents)
        for ni in update_range:
            s = self.ppo_update_agent(storage, ni)
            s["agent_idx"] = float(ni)
            s["role"] = "hunter" if ni < self.nh else "escaper"
            logs.append(s)
        return next_obs, logs, metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "cfg": self.cfg.model_dump(),
            "state_dicts": {},
            "escaper_mode": "rule" if self._use_rule_escaper else "learn",
        }
        for k, pol in self.policies.items():
            payload["state_dicts"][k] = pol.state_dict()
        if self._obs_rms:
            payload["obs_rms"] = {k: v.get_state() for k, v in self._obs_rms.items()}
        if self._rew_rms:
            payload["rew_rms"] = {k: v.get_state() for k, v in self._rew_rms.items()}
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        cfg: HuntEnvConfig,
        vec_env: HuntVectorizedEnv,
        **kwargs: Any,
    ) -> MultiAgentPPOTrainer:
        path = Path(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        load_kwargs = {**kwargs}
        if "escaper_mode" not in load_kwargs and isinstance(payload, dict):
            em = payload.get("escaper_mode")
            if em in ("learn", "rule"):
                load_kwargs["escaper_mode"] = em
        trainer = cls(cfg, vec_env, **load_kwargs)
        for k, sd in payload["state_dicts"].items():
            if k in trainer.policies:
                trainer.policies[k].load_state_dict(sd)
        if "obs_rms" in payload and isinstance(payload["obs_rms"], dict):
            for k, st in payload["obs_rms"].items():
                if k in trainer._obs_rms:
                    trainer._obs_rms[k].set_state(st)
        if "rew_rms" in payload and isinstance(payload["rew_rms"], dict):
            for k, st in payload["rew_rms"].items():
                if k in trainer._rew_rms:
                    trainer._rew_rms[k].set_state(st)
        return trainer
