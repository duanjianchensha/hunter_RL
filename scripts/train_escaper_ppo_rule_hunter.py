#!/usr/bin/env python3
"""
仅训练逃脱者，猎人用规则策略 `rule_action_hunter(obs, cfg)`（与 RL 同一观测）。

默认使用 `configs/default.yaml`（单局步数、并行 env 数以 YAML 为准，不设 CLI 覆盖即不改写）。

  pip install -e ".[rl]"
  python scripts/train_escaper_ppo_rule_hunter.py --save runs/escaper_rl.pt --time-sec 600

预训练热启（隐层 256×256）：

  python scripts/train_escaper_ppo_rule_hunter.py --init-escaper pretrained/escaper_rule/escaper.pt --save runs/escaper_rl.pt --time-sec 3600
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hunt_env.cli_defaults import DEFAULT_CONFIG_YAML, train_env_merge
from hunt_env.config.loader import load_config
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.device import get_train_device
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig


def main() -> None:
    p = argparse.ArgumentParser(description="逃脱者 PPO + 规则猎人")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG_YAML, help="YAML（默认 configs/default.yaml）")
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
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="覆盖 YAML 中单局最大步数；默认不指定则沿用配置文件（default 一般为 1000）",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="并行环境数；默认不指定则沿用 YAML 的 vectorization.num_envs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--save", type=str, default=None, help="结束时可保存 .pt 检查点")
    p.add_argument(
        "--init-escaper",
        type=str,
        default=None,
        help="从预训练/PPO checkpoint 加载 policies['escaper'] 与 obs/rew 统计（若存在）；隐层自动用 256×256 与预训练对齐",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="若指定路径，将 stdout/stderr 同步写入该日志文件（UTF-8）",
    )
    args = p.parse_args()

    from hunt_rl.train_log import tee_stdout_stderr

    with tee_stdout_stderr(args.log_file):
        _train_escaper_rule_hunter_run(args)


def _train_escaper_rule_hunter_run(args: Namespace) -> None:
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

    merge = train_env_merge(args.max_episode_steps, args.num_envs)
    cfg = load_config(args.config, merge=merge) if merge else load_config(args.config)
    env = HuntVectorizedEnv(cfg=cfg, num_envs=cfg.vectorization.num_envs)

    ppo_cfg = PPOConfig(
        lr=args.lr,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )
    hidden = (256, 256) if args.init_escaper else (128, 128)
    trainer = MultiAgentPPOTrainer(
        env.cfg,
        env,
        ppo_cfg=ppo_cfg,
        hidden_sizes=hidden,
        device=device,
        hunter_mode="rule",
    )

    if args.init_escaper:
        import torch

        path_e = Path(args.init_escaper)
        try:
            ck = torch.load(path_e, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path_e, map_location="cpu")
        sds = ck.get("state_dicts") or {}
        if "escaper" in sds and "escaper" in trainer.policies:
            trainer.policies["escaper"].load_state_dict(sds["escaper"])
        orms = ck.get("obs_rms")
        if isinstance(orms, dict) and "escaper" in orms and "escaper" in trainer._obs_rms:
            trainer._obs_rms["escaper"].set_state(orms["escaper"])
        rr = ck.get("rew_rms")
        if isinstance(rr, dict) and "escaper" in rr and "escaper" in trainer._rew_rms:
            trainer._rew_rms["escaper"].set_state(rr["escaper"])
        print(f"已从 {path_e} 热启动逃脱者（网络与 RMS，若存在）")

    obs = env.reset(seed=args.seed)
    t0 = time.perf_counter()
    n_upd = 0
    total_env_steps = 0

    print(
        f"开始：device={device}，时间预算={args.time_sec}s，"
        f"rollout={args.rollout_len}，num_envs={env.num_envs}，"
        f"sim.max_episode_steps={cfg.sim.max_episode_steps}，hidden={hidden}，"
        f"hunter=rule，只更新逃脱者 PPO"
        + (f"，init={args.init_escaper}" if args.init_escaper else "")
        + "。",
    )

    while time.perf_counter() - t0 < args.time_sec:
        next_obs, logs, metrics = trainer.train_step(args.rollout_len, obs)
        obs = next_obs
        n_upd += 1
        inc = args.rollout_len * env.num_envs
        total_env_steps += inc
        elapsed = time.perf_counter() - t0
        erew = metrics.get("escaper_rew_mean", float("nan"))
        lg0 = logs[0] if logs else {}
        line = (
            f"[{elapsed:6.1f}s] upd {n_upd:4d}  env_steps={total_env_steps:6d}  "
            f"e_rew_mean={erew:8.4f}  loss={lg0.get('loss', 0.0):8.4f}  "
            f"pg={lg0.get('pg', 0.0):7.3f}  v={lg0.get('v', 0.0):7.3f}  ent~={lg0.get('ent', 0.0):6.2f}"
        )
        print(line)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.save)
        print(f"已保存: {args.save}")

    print("结束：若 e_rew_mean 与 loss 有波动（非全 NaN/全零），可认为管线正常。")


if __name__ == "__main__":
    main()
