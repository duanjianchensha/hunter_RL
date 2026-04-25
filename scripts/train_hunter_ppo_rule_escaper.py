#!/usr/bin/env python3
"""
仅训练猎人，逃脱者使用规则策略（hunt_env.policies.rules）——用于在 ~10 分钟内
观察 loss 与奖励是否正常变化。

  pip install -e ".[rl]"
  python scripts/train_hunter_ppo_rule_escaper.py --time-sec 600
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hunt_env.config.loader import load_config
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.device import get_train_device
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig


def main() -> None:
    p = argparse.ArgumentParser(description="猎人 PPO + 规则逃脱者（短训验证）")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="基础 YAML 配置")
    p.add_argument(
        "--time-sec",
        type=float,
        default=600.0,
        help="墙上时钟预算（秒），默认 10 分钟",
    )
    p.add_argument(
        "--rollout-len",
        type=int,
        default=256,
        help="每轮 PPO 更新前采集的步数",
    )
    p.add_argument("--max-episode-steps", type=int, default=200, help="单局步数（覆盖配置，短局加快 reset）")
    p.add_argument("--num-envs", type=int, default=4, help="并行环境数，提高吞吐")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--save", type=str, default=None, help="结束时可保存 .pt 检查点")
    args = p.parse_args()

    if args.device == "auto":
        device = get_train_device(prefer_cuda=True)
    elif args.device == "cuda":
        device = get_train_device(prefer_cuda=True)
        if device.type != "cuda":
            raise RuntimeError("指定了 --device cuda 但当前不可用")
    else:
        device = get_train_device(prefer_cuda=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    except ImportError as e:
        raise SystemExit("需要安装 PyTorch：pip install -e \".[rl]\"") from e

    cfg = load_config(
        args.config,
        merge={"sim": {"max_episode_steps": args.max_episode_steps}, "vectorization": {"num_envs": args.num_envs}},
    )
    env = HuntVectorizedEnv(cfg=cfg, num_envs=cfg.vectorization.num_envs)

    ppo_cfg = PPOConfig(
        lr=args.lr,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )
    # 小网络，CPU 上几十秒内能跑多轮
    trainer = MultiAgentPPOTrainer(
        env.cfg,
        env,
        ppo_cfg=ppo_cfg,
        hidden_sizes=(128, 128),
        device=device,
        escaper_mode="rule",
    )

    obs = env.reset(seed=args.seed)
    t0 = time.perf_counter()
    n_upd = 0
    total_env_steps = 0

    print(
        f"开始：device={device}，时间预算={args.time_sec}s，"
        f"rollout={args.rollout_len}，num_envs={env.num_envs}，"
        f"escaper=rule，只更新猎人 PPO。",
    )

    while time.perf_counter() - t0 < args.time_sec:
        next_obs, logs, metrics = trainer.train_step(args.rollout_len, obs)
        obs = next_obs
        n_upd += 1
        inc = args.rollout_len * env.num_envs
        total_env_steps += inc
        elapsed = time.perf_counter() - t0
        hrew = metrics.get("hunter_rew_mean", float("nan"))
        lg0 = logs[0] if logs else {}
        line = (
            f"[{elapsed:6.1f}s] upd {n_upd:4d}  env_steps={total_env_steps:6d}  "
            f"h_rew_mean={hrew:8.4f}  loss={lg0.get('loss', 0.0):8.4f}  "
            f"pg={lg0.get('pg', 0.0):7.3f}  v={lg0.get('v', 0.0):7.3f}  ent~={lg0.get('ent', 0.0):6.2f}"
        )
        print(line)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.save)
        print(f"已保存: {args.save}")

    print("结束：若 h_rew_mean 与 loss 有波动（非全 NaN/全零），可认为管线正常。")


if __name__ == "__main__":
    main()
