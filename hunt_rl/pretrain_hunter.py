"""
猎人策略：在规则示教下并行采集，行为克隆高斯策略的均值到「单位盒」目标 + GAE 回报训练价值头。
与 PPO 一致：hunter 的 RunningMeanStd + 可选 RunningRewardRMS、ActorCritic 与动作仿射。
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
class HunterPretrainConfig:
    """BC + 价值联合训练超参。"""

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


def save_hunter_pretrain(
    path: str | Path,
    *,
    cfg: HuntEnvConfig,
    pol: ActorCritic,
    obs_rms: Optional[RunningMeanStd],
    rew_rms: RunningRewardRMS | None,
    meta: dict[str, Any] | None = None,
) -> None:
    """与 `MultiAgentPPOTrainer` 的 hunter 子结构兼容的 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "cfg": cfg.model_dump(),
        "state_dicts": {"hunter": pol.state_dict()},
        "escaper_mode": "learn",
    }
    if obs_rms is not None:
        payload["obs_rms"] = {"hunter": obs_rms.get_state()}
    if rew_rms is not None:
        payload["rew_rms"] = {"hunter": rew_rms.get_state()}
    if meta:
        payload["meta"] = meta
    torch.save(payload, path)


class HunterRulePretrainer:
    """多猎人共网；逃脱者用规则。仅训练猎人 ActorCritic。"""

    def __init__(
        self,
        cfg: HuntEnvConfig,
        vec_env: HuntVectorizedEnv,
        *,
        pre_cfg: HunterPretrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        self.env = vec_env
        self.nh = cfg.agents.n_hunters
        self.ne = cfg.agents.n_escapers
        self.n = vec_env.n_agents
        self.d = vec_env.obs_dim
        if self.nh < 1:
            raise ValueError("预训练猎人需要至少 1 名猎人")
        self.p = pre_cfg or HunterPretrainConfig()
        self.device = device or get_train_device(prefer_cuda=False)
        self.pol = ActorCritic(self.d, 2, self.p.hidden_sizes).to(self.device)
        self.opt = optim.Adam(self.pol.parameters(), lr=self.p.lr)
        self.obs_rms = RunningMeanStd(self.d) if self.p.normalize_obs else None
        self.rew_rms: RunningRewardRMS | None = (
            RunningRewardRMS() if self.p.normalize_reward else None
        )
        al, ah = action_bounds_from_cfg(cfg, "hunter")
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
        nh, d, n = self.nh, self.d, self.n
        t_len = num_steps
        obs_buf = np.zeros((t_len + 1, e, n, d), dtype=np.float32)
        a_n_buf = np.zeros((t_len, e, nh, 2), dtype=np.float32)
        rew_h = np.zeros((t_len, e, nh), dtype=np.float32)
        # 与采集时刻一致的逐步奖励：先 `normalize(当前 rms)` 再 `update`，供 GAE 与 val 的因果时刻对齐。
        # 勿在 `train_on_batch` 里用「整段 rollout 结束后」的 rms 去归一化整段 r，否则早期步会带上未来方差、TD 目标震荡。
        rew_n_h = np.zeros((t_len, e, nh), dtype=np.float32)
        val_buf = np.zeros((t_len + 1, e, nh), dtype=np.float32)
        done_buf = np.zeros((t_len, e), dtype=np.bool_)

        obs_buf[0] = obs0
        for t in range(t_len):
            actions = np.zeros((e, n, 2), dtype=np.float32)
            for ni in range(nh):
                for ei in range(e):
                    a_env = rule_action_hunter(obs_buf[t, ei, ni, :], self.cfg)
                    actions[ei, ni, :] = a_env
            a_n_buf[t] = _env_to_unit_np(actions[:, :nh, :], self._lo_np, self._hi_np)
            np.clip(a_n_buf[t], -1.0, 1.0, out=a_n_buf[t])
            for ni in range(nh, n):
                for ei in range(e):
                    actions[ei, ni, :] = rule_action_escaper(obs_buf[t, ei, ni, :], self.cfg)
            for ni in range(nh):
                o_in = self._obs_forward(obs_buf[t, :, ni, :])
                ot = torch.as_tensor(o_in, device=self.device)
                with torch.no_grad():
                    _, _, v = self.pol(ot)
                    val_buf[t, :, ni] = v.cpu().numpy()
            next_obs, rew, term, trunc, _ = self.env.step(actions)
            raw_row = rew[:, :nh]
            rew_h[t] = raw_row
            if self.rew_rms is not None:
                rew_n_h[t] = self.rew_rms.normalize(np.asarray(raw_row, dtype=np.float32))
                self.rew_rms.update(np.asarray(raw_row, dtype=np.float64, copy=False))
            else:
                rew_n_h[t] = np.asarray(raw_row, dtype=np.float32, copy=False)
            done_buf[t] = np.logical_or(term[:, 0], trunc[:, 0])
            if e == 1 and done_buf[t, 0]:
                next_obs = self.env.reset(seed=None)
            elif e > 1 and np.all(done_buf[t]):
                next_obs = self.env.reset(seed=None)
            obs_buf[t + 1] = next_obs
            for ni in range(nh):
                o2 = self._obs_bootstrap(obs_buf[t + 1, :, ni, :])
                o2t = torch.as_tensor(o2, device=self.device)
                with torch.no_grad():
                    _, _, v2 = self.pol(o2t)
                    val_buf[t + 1, :, ni] = v2.cpu().numpy()

        metrics = {
            "hunter_rew_mean": float(np.mean(rew_h)),
        }
        return obs_buf[-1].copy(), {
            "obs": obs_buf,
            "a_n": a_n_buf,
            "rew_h": rew_h,
            "rew_n_h": rew_n_h,
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
        rew_h = storage["rew_h"]
        val = storage["val"]
        done = storage["done"]

        if storage.get("rew_n_h") is not None:
            rew_n = storage["rew_n_h"]
        elif self.p.normalize_reward and self.rew_rms is not None:
            # 兼容无 rew_n_h 的旧 buffer：用当前 rms 一次归一整段（因果略差，易使 v 目标震荡）
            rew_n = self.rew_rms.normalize(rew_h.astype(np.float32, copy=False))
        else:
            rew_n = rew_h

        obs_flat: list[np.ndarray] = []
        a_flat: list[np.ndarray] = []
        ret_flat: list[np.ndarray] = []
        for ni in range(self.nh):
            o_raw = obs[:, :, ni, :]
            o_n = self._obs_replay(o_raw)
            o_b = o_n.reshape(-1, d)
            an_b = a_n[:, :, ni, :].reshape(-1, 2)
            rw = rew_n[:, :, ni]
            _adv, ret = compute_gae(
                rw, val[:, :, ni], done, self.p.gamma, self.p.gae_lambda
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
                # BC：令高斯均值逼近规则在单位盒上的等效目标（与 act 中 clip 前 raw 的常见中心一致）
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
