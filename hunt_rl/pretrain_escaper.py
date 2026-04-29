"""
逃脱者策略：猎人用规则示教并行采集，行为克隆规则动作（单位盒目标）+ GAE 回报训练价值头。
与 PPO 一致：escaper 的 RunningMeanStd、可选 RunningRewardRMS、ActorCritic 与动作仿射。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_env.policies.rules import rule_action_escaper, rule_action_hunter
from hunt_rl.actor_critic import ActorCritic, action_bounds_from_cfg
from hunt_rl.device import get_train_device
from hunt_rl.running_stats import RunningMeanStd, RunningRewardRMS
from hunt_rl.trainer import compute_gae


def _env_to_unit_np(a_env: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """a_env: (..., 2)；lo, hi: (2,)，广播。"""
    return (2.0 * (a_env - lo) / (hi - lo) - 1.0).astype(np.float32, copy=False)


@dataclass
class EscaperPretrainConfig:
    """BC + 价值联合训练超参（与 HunterPretrainConfig 字段对齐）。"""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    bc_coef: float = 1.0
    vf_coef: float = 0.5
    log_std_reg: float = 0.0
    max_grad_norm: float = 0.5
    update_epochs: int = 8
    num_minibatches: int = 4
    normalize_obs: bool = True
    normalize_reward: bool = True
    obs_clip: float = 10.0
    hidden_sizes: tuple[int, ...] = (256, 256)


def save_escaper_pretrain(
    path: str | Path,
    *,
    cfg: HuntEnvConfig,
    pol: ActorCritic,
    obs_rms: Optional[RunningMeanStd],
    rew_rms: RunningRewardRMS | None,
    meta: dict[str, Any] | None = None,
) -> None:
    """与 `MultiAgentPPOTrainer` 的 escaper 子结构兼容的 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cfg": cfg.model_dump(),
        "state_dicts": {"escaper": pol.state_dict()},
        "escaper_mode": "learn",
        "hunter_mode": "rule",
    }
    if obs_rms is not None:
        payload["obs_rms"] = {"escaper": obs_rms.get_state()}
    if rew_rms is not None:
        payload["rew_rms"] = {"escaper": rew_rms.get_state()}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


class EscaperRulePretrainer:
    """多逃脱者共网；猎人用规则。仅训练逃脱者 ActorCritic。"""

    def __init__(
        self,
        cfg: HuntEnvConfig,
        vec_env: HuntVectorizedEnv,
        *,
        pre_cfg: EscaperPretrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = vec_env
        self.nh = cfg.agents.n_hunters
        self.ne = cfg.agents.n_escapers
        self.n = vec_env.n_agents
        self.d = vec_env.obs_dim
        if self.ne < 1:
            raise ValueError("预训练逃脱者需要至少 1 名逃脱者")
        if self.nh < 1:
            raise ValueError("预训练逃脱者需要至少 1 名规则猎人")
        self.p = pre_cfg or EscaperPretrainConfig()
        self.device = device or get_train_device(prefer_cuda=False)
        self.pol = ActorCritic(self.d, 2, self.p.hidden_sizes).to(self.device)
        self.opt = optim.Adam(self.pol.parameters(), lr=self.p.lr)
        self.obs_rms = RunningMeanStd(self.d) if self.p.normalize_obs else None
        self.rew_rms: RunningRewardRMS | None = (
            RunningRewardRMS() if self.p.normalize_reward else None
        )
        al, ah = action_bounds_from_cfg(cfg, "escaper")
        self._lo_np = al.astype(np.float32)
        self._hi_np = ah.astype(np.float32)

    def _obs_forward(self, raw: np.ndarray) -> np.ndarray:
        r = np.asarray(raw, dtype=np.float32)
        if not self.p.normalize_obs or self.obs_rms is None:
            return r
        self.obs_rms.update(raw.astype(np.float64, copy=False))
        o = self.obs_rms.normalize(r)
        c = self.p.obs_clip
        return np.clip(o, -c, c).astype(np.float32, copy=False)

    def _obs_bootstrap(self, raw: np.ndarray) -> np.ndarray:
        r = np.asarray(raw, dtype=np.float32)
        if not self.p.normalize_obs or self.obs_rms is None:
            return r
        o = self.obs_rms.normalize(r)
        c = self.p.obs_clip
        return np.clip(o, -c, c).astype(np.float32, copy=False)

    def _obs_replay(self, raw: np.ndarray) -> np.ndarray:
        return self._obs_bootstrap(raw)

    def collect(
        self, num_steps: int, obs0: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        e = self.env.num_envs
        nh, ne, d, n = self.nh, self.ne, self.d, self.n
        t_len = num_steps
        obs_buf = np.zeros((t_len + 1, e, n, d), dtype=np.float32)
        a_n_buf = np.zeros((t_len, e, ne, 2), dtype=np.float32)
        rew_e = np.zeros((t_len, e, ne), dtype=np.float32)
        rew_n_e = np.zeros((t_len, e, ne), dtype=np.float32)
        val_buf = np.zeros((t_len + 1, e, ne), dtype=np.float32)
        done_buf = np.zeros((t_len, e), dtype=np.bool_)

        obs_buf[0] = obs0
        for t in range(t_len):
            actions = np.zeros((e, n, 2), dtype=np.float32)
            for ni in range(nh):
                for ei in range(e):
                    actions[ei, ni, :] = rule_action_hunter(obs_buf[t, ei, ni, :], self.cfg)
            for j in range(ne):
                ai = nh + j
                for ei in range(e):
                    actions[ei, ai, :] = rule_action_escaper(obs_buf[t, ei, ai, :], self.cfg)
            a_n_buf[t] = _env_to_unit_np(actions[:, nh:, :], self._lo_np, self._hi_np)
            np.clip(a_n_buf[t], -1.0, 1.0, out=a_n_buf[t])
            for j in range(ne):
                ai = nh + j
                o_in = self._obs_forward(obs_buf[t, :, ai, :])
                ot = torch.as_tensor(o_in, device=self.device)
                with torch.no_grad():
                    _, _, v = self.pol(ot)
                    val_buf[t, :, j] = v.cpu().numpy()
            next_obs, rew, term, trunc, _ = self.env.step(actions)
            raw_row = rew[:, nh:]
            rew_e[t] = raw_row
            if self.rew_rms is not None:
                rew_n_e[t] = self.rew_rms.normalize(np.asarray(raw_row, dtype=np.float32))
                self.rew_rms.update(np.asarray(raw_row, dtype=np.float64, copy=False))
            else:
                rew_n_e[t] = np.asarray(raw_row, dtype=np.float32, copy=False)
            done_buf[t] = np.logical_or(term[:, 0], trunc[:, 0])
            if e == 1 and done_buf[t, 0]:
                next_obs = self.env.reset(seed=None)
            elif e > 1 and np.all(done_buf[t]):
                next_obs = self.env.reset(seed=None)
            obs_buf[t + 1] = next_obs
            for j in range(ne):
                ai = nh + j
                o2 = self._obs_bootstrap(obs_buf[t + 1, :, ai, :])
                o2t = torch.as_tensor(o2, device=self.device)
                with torch.no_grad():
                    _, _, v2 = self.pol(o2t)
                    val_buf[t + 1, :, j] = v2.cpu().numpy()

        metrics = {
            "escaper_rew_mean": float(np.mean(rew_e)),
        }
        return obs_buf[-1].copy(), {
            "obs": obs_buf,
            "a_n": a_n_buf,
            "rew_e": rew_e,
            "rew_n_e": rew_n_e,
            "val": val_buf,
            "done": done_buf,
        }, metrics

    def train_on_batch(
        self, storage: dict[str, np.ndarray]
    ) -> dict[str, float]:
        obs = storage["obs"][:-1]
        _t_len, _n_e, _n_agents, d = obs.shape
        assert _n_agents == self.n
        a_n = storage["a_n"]
        val = storage["val"]
        done = storage["done"]

        if storage.get("rew_n_e") is not None:
            rew_n = storage["rew_n_e"]
        elif self.p.normalize_reward and self.rew_rms is not None:
            rew_n = self.rew_rms.normalize(storage["rew_e"].astype(np.float32, copy=False))
        else:
            rew_n = storage["rew_e"]

        obs_flat: list[np.ndarray] = []
        a_flat: list[np.ndarray] = []
        ret_flat: list[np.ndarray] = []
        nh = self.nh
        for j in range(self.ne):
            ai = nh + j
            o_raw = obs[:, :, ai, :]
            o_n = self._obs_replay(o_raw)
            o_b = o_n.reshape(-1, d)
            an_b = a_n[:, :, j, :].reshape(-1, 2)
            rw = rew_n[:, :, j]
            _adv, ret = compute_gae(
                rw, val[:, :, j], done, self.p.gamma, self.p.gae_lambda
            )
            r_b = ret.reshape(-1)
            obs_flat.append(o_b)
            a_flat.append(an_b)
            ret_flat.append(r_b)
        O = np.concatenate(obs_flat, axis=0)
        A = np.concatenate(a_flat, axis=0)
        R = np.concatenate(ret_flat, axis=0)
        assert O.shape[0] == A.shape[0] == R.shape[0]
        m = A.shape[0]
        mb = max(m // self.p.num_minibatches, 1)

        stats = {"bc": 0.0, "v": 0.0, "log_std_reg": 0.0, "loss": 0.0}
        n_u = 0
        for _ in range(self.p.update_epochs):
            idx = np.random.permutation(m)
            for s in range(0, m, mb):
                b = idx[s : s + mb]
                ob = torch.as_tensor(O[b], device=self.device)
                a_tgt = torch.as_tensor(A[b], device=self.device)
                rt = torch.as_tensor(R[b], device=self.device)
                mean, _std, vpred = self.pol(ob)
                loss_bc = (mean - a_tgt).pow(2).mean()
                loss_v = 0.5 * (vpred - rt).pow(2).mean()
                ls = self.pol.actor_log_std
                loss_std = (ls**2).mean() if self.p.log_std_reg > 0.0 else torch.zeros(1, device=ob.device)
                loss = (
                    self.p.bc_coef * loss_bc
                    + self.p.vf_coef * loss_v
                    + self.p.log_std_reg * loss_std
                )
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.pol.parameters(), self.p.max_grad_norm)
                self.opt.step()
                stats["bc"] += float(loss_bc.detach().cpu())
                stats["v"] += float(loss_v.detach().cpu())
                stats["log_std_reg"] += float(loss_std.detach().cpu())
                stats["loss"] += float(loss.detach().cpu())
                n_u += 1
        for k in stats:
            stats[k] /= max(n_u, 1)
        return stats

    def train_step(
        self, num_steps: int, obs: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
        next_obs, storage, m_collect = self.collect(num_steps, obs)
        tstats = self.train_on_batch(storage)
        return next_obs, tstats, m_collect
