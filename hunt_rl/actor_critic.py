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


class ActorCritic(nn.Module):
    """
    对角高斯策略 + 价值头。
    动作在环境 low/high 内裁剪；log_prob 在未裁剪的高斯上计算（常见近似）。
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作、log_prob、价值。
        action_low/high: 形状与 action 一致，可为 broadcast (action_dim,) 或 (batch, action_dim)。
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        if deterministic:
            raw = mean
        else:
            raw = dist.rsample()
        logp = dist.log_prob(raw).sum(dim=-1)
        a = torch.max(torch.min(raw, action_high), action_low)
        return a, logp, value

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """给定观测与已执行动作，计算 log_prob、entropy、value（用于 PPO 更新）。"""
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
