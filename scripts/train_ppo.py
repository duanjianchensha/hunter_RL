#!/usr/bin/env python3
"""
PPO 训练入口：CPU 默认可跑；若安装 CUDA 版 PyTorch 且可见 GPU，自动用 GPU。

示例：
  pip install -e ".[rl]"
  python scripts/train_ppo.py --config configs/default.yaml --total-steps 50000
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hunt_env.cli_defaults import DEFAULT_CONFIG_YAML
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.device import get_train_device
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig


def main() -> None:
    p = argparse.ArgumentParser(description="追猎环境 PPO 训练（PyTorch）")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG_YAML, help="YAML 配置路径")
    p.add_argument("--total-steps", type=int, default=100_000, help="目标环境步数（约等于并行 env 数 × rollout 步数 × 更新次数）")
    p.add_argument("--rollout-len", type=int, default=2048, help="每次更新前采集的步数")
    p.add_argument("--num-envs", type=int, default=None, help="并行环境数（默认读配置 vectorization.num_envs）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save", type=str, default=None, help="checkpoint 保存路径（.pt）")
    p.add_argument("--save-every", type=int, default=50_000, help="每隔多少环境步保存一次")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--init-hunter",
        type=str,
        default=None,
        help="从预训练等 checkpoint 只加载 policies['hunter'] 与 obs/rew 统计（若存在键），不覆盖你指定的 --save 以外文件",
    )
    p.add_argument(
        "--init-escaper",
        type=str,
        default=None,
        help="从预训练等 checkpoint 只加载 policies['escaper'] 与 obs/rew 统计（若存在键）",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="若指定路径，将 stdout/stderr 同步写入该日志文件（UTF-8），控制台仍输出",
    )
    args = p.parse_args()

    from hunt_rl.train_log import tee_stdout_stderr

    with tee_stdout_stderr(args.log_file):
        _train_ppo_run(args)


def _train_ppo_run(args: Namespace) -> None:
    if args.device == "auto":
        device = get_train_device(prefer_cuda=True)
    elif args.device == "cuda":
        device = get_train_device(prefer_cuda=True)
        if device.type != "cuda":
            raise RuntimeError("指定了 --device cuda 但当前不可用（未装 CUDA 版 torch 或无 GPU）")
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

    env = HuntVectorizedEnv.from_yaml(args.config, num_envs=args.num_envs)
    ppo_cfg = PPOConfig(lr=args.lr)
    trainer = MultiAgentPPOTrainer(env.cfg, env, ppo_cfg=ppo_cfg, device=device)

    if args.init_hunter:
        path_h = Path(args.init_hunter)
        try:
            ck = torch.load(path_h, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path_h, map_location="cpu")
        sds = ck.get("state_dicts") or {}
        if "hunter" in sds and "hunter" in trainer.policies:
            trainer.policies["hunter"].load_state_dict(sds["hunter"])
        orms = ck.get("obs_rms")
        if isinstance(orms, dict) and "hunter" in orms and "hunter" in trainer._obs_rms:
            trainer._obs_rms["hunter"].set_state(orms["hunter"])
        rr = ck.get("rew_rms")
        if isinstance(rr, dict) and "hunter" in rr and "hunter" in trainer._rew_rms:
            trainer._rew_rms["hunter"].set_state(rr["hunter"])
        print(f"已从 {path_h} 热启动猎人（网络与 RMS，若存在）")

    if args.init_escaper:
        path_e = Path(args.init_escaper)
        try:
            ck_e = torch.load(path_e, map_location="cpu", weights_only=False)
        except TypeError:
            ck_e = torch.load(path_e, map_location="cpu")
        sds_e = ck_e.get("state_dicts") or {}
        if "escaper" in sds_e and "escaper" in trainer.policies:
            trainer.policies["escaper"].load_state_dict(sds_e["escaper"])
        orms_e = ck_e.get("obs_rms")
        if isinstance(orms_e, dict) and "escaper" in orms_e and "escaper" in trainer._obs_rms:
            trainer._obs_rms["escaper"].set_state(orms_e["escaper"])
        rr_e = ck_e.get("rew_rms")
        if isinstance(rr_e, dict) and "escaper" in rr_e and "escaper" in trainer._rew_rms:
            trainer._rew_rms["escaper"].set_state(rr_e["escaper"])
        print(f"已从 {path_e} 热启动逃脱者（网络与 RMS，若存在）")

    obs = env.reset(seed=args.seed)
    total = 0
    update_idx = 0
    print(f"device={device}, num_envs={env.num_envs}, rollout_len={args.rollout_len}")

    while total < args.total_steps:
        next_obs, logs, metrics = trainer.train_step(args.rollout_len, obs)
        obs = next_obs
        inc = args.rollout_len * env.num_envs
        total += inc
        update_idx += 1

        parts = [f"step {total}/{args.total_steps}"]
        if "hunter_rew_mean" in metrics:
            parts.append(f"h_rew_mean={metrics['hunter_rew_mean']:.4f}")
        for lg in logs:
            i = int(lg.get("agent_idx", 0))
            parts.append(f"ag{i} loss={lg['loss']:.4f} pg={lg['pg']:.4f} v={lg['v']:.4f}")
        print(" | ".join(parts))

        if args.save and total // args.save_every > (total - inc) // args.save_every:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            trainer.save(args.save)
            print(f"saved {args.save}")

    if args.save:
        trainer.save(args.save)
        print(f"final save {args.save}")


if __name__ == "__main__":
    main()
