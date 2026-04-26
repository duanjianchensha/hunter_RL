"""高斯策略 + 状态价值（连续动作）。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def denorm_action_from_unit_box(
    a_unit: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """
    将 a_unit 各维自 [-1,1] 仿射到 [low, high]：a = low + 0.5 * (a_unit + 1) * (high - low)。
    """
    return action_low + 0.5 * (a_unit + 1.0) * (action_high - action_low)


class ActorCritic(nn.Module):
    """
    对角高斯策略 + 价值头。
    在**归一化动作空间** ℝ² 上采样，再 clip 到 [-1,1]，仿射到环境 [low,high]；log_prob 对未截断的 raw 用高斯密度（与常见 clip 高斯 PPO 一致）。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_sizes: tuple[int, ...] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        hs = [obs_dim, *hidden_sizes]
        self.backbone = _build_mlp(hs, activation)
        last = hidden_sizes[-1]
        self.actor_mean = nn.Linear(last, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 mean, std, value。"""
        h = self.backbone(obs)
        mean = self.actor_mean(h)
        std = torch.exp(self.actor_log_std.clamp(-20, 2)).expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def act(
        self,
        obs: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 a_env、logp、v、raw。raw ~ N(μ,σ)（对 logp/缓冲区）；a_n=clip(raw,-1,1) 后仿射为环境动作。
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        if deterministic:
            raw = mean
        else:
            raw = dist.rsample()
        logp = dist.log_prob(raw).sum(dim=-1)
        a_n = torch.clamp(raw, -1.0, 1.0)
        a_env = denorm_action_from_unit_box(a_n, action_low, action_high)
        return a_env, logp, value, raw

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        给定已存储的**未截断** raw 动作（高斯前 clip）与归一化后的 obs，计算 log_prob、entropy、v。
        action_low/high 保留签名兼容，估值头不依赖动作界。
        """
        del action_low, action_high
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        logp = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logp, entropy, value


def action_bounds_from_cfg(cfg, role: str) -> tuple[np.ndarray, np.ndarray]:
    """role: 'hunter' | 'escaper'"""
    if role == "hunter":
        lim = cfg.agents.hunter_limits
    else:
        lim = cfg.agents.escaper_limits
    low = np.array([-lim.max_accel, -lim.max_omega], dtype=np.float32)
    high = np.array([lim.max_accel, lim.max_omega], dtype=np.float32)
    return low, high
