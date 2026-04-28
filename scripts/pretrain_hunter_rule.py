#!/usr/bin/env python3
"""
猎人预训练：并行环境中逃脱者、其余猎人槽位用规则，仅优化猎人共享 ActorCritic。

- 行为克隆：策略头输出之均值，监督为规则动作经仿射反变换后的单位盒坐标（与 PPO `act` 的 clip/仿射前语义一致）；
- 价值头：在规则 rollouts 上用 GAE 回报作回归。

权重与 hunter 的 obs_rms（及可选 rew_rms）存于单独目录，默认不覆盖；后续 PPO 可用
`--init-hunter` 加载起点做对比实验。

  pip install -e ".[rl]"
  python scripts/pretrain_hunter_rule.py --config configs/default.yaml --out-dir pretrained/hunter_rule
  python scripts/pretrain_hunter_rule.py ... --log-file pretrained/hunter_rule/train.log
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.device import get_train_device
from hunt_rl.pretrain_hunter import HunterPretrainConfig, HunterRulePretrainer, save_hunter_pretrain


def main() -> None:
    p = argparse.ArgumentParser(
        description="规则示教下并行预训练猎人 ActorCritic（BC + 价值 GAE）"
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--out-dir",
        type=str,
        default="pretrained/hunter_rule",
        help="只写该目录，不污染其它 checkpoint；内含 hunter.pt 与 run.json",
    )
    p.add_argument(
        "--total-env-steps",
        type=int,
        default=2_000_000,
        help="总环境步数约等于 num_envs × rollout_len × 更新轮数",
    )
    p.add_argument("--rollout-len", type=int, default=1024)
    p.add_argument("--num-envs", type=int, default=32, help="并行环境数，越大吞吐越高")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--update-epochs", type=int, default=8)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--bc-coef", type=float, default=1.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--log-std-reg", type=float, default=0.0, help="对 actor_log_std 的 L2，默认 0")
    p.add_argument(
        "--no-norm-obs", action="store_true", help="关闭观测 online RMS（与 PPO 默认不一致，慎用）"
    )
    p.add_argument(
        "--no-norm-rew", action="store_true", help="关闭 GAE 前奖励标量方差规约"
    )
    p.add_argument("--print-every", type=int, default=10, help="每多少次更新打印一行")
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="若指定路径，将 stdout/stderr 同步写入该日志文件（UTF-8）",
    )
    args = p.parse_args()

    from hunt_rl.train_log import tee_stdout_stderr

    with tee_stdout_stderr(args.log_file):
        _pretrain_hunter_rule_run(args)


def _pretrain_hunter_rule_run(args: Namespace) -> None:
    if args.device == "auto":
        device = get_train_device(prefer_cuda=True)
    elif args.device == "cuda":
        device = get_train_device(prefer_cuda=True)
        if device.type != "cuda":
            raise RuntimeError("请求 cuda 但当前不可用")
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
        raise SystemExit("需要 torch：pip install -e \".[rl]\"") from e

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.txt").write_text(
        "本目录为猎人规则预训练 checkpoint，与主训练保存路径分离；PPO 请用 "
        "python scripts/train_ppo.py --init-hunter 指向 hunter.pt\n",
        encoding="utf-8",
    )

    env = HuntVectorizedEnv.from_yaml(args.config, num_envs=args.num_envs)
    pre = HunterPretrainConfig(
        lr=args.lr,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        bc_coef=args.bc_coef,
        vf_coef=args.vf_coef,
        log_std_reg=args.log_std_reg,
        normalize_obs=not args.no_norm_obs,
        normalize_reward=not args.no_norm_rew,
    )
    trainer = HunterRulePretrainer(env.cfg, env, pre_cfg=pre, device=device)
    print(
        f"device={device}, num_envs={env.num_envs}, "
        f"rollout={args.rollout_len}, target_steps={args.total_env_steps}"
    )

    inc = args.rollout_len * env.num_envs
    total = 0
    upd = 0
    obs = env.reset(seed=args.seed)
    meta = {
        "script": "pretrain_hunter_rule.py",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "env_steps_per_update": inc,
    }

    while total < args.total_env_steps:
        next_obs, tstats, m_coll = trainer.train_step(args.rollout_len, obs)
        obs = next_obs
        total += inc
        upd += 1
        if upd % args.print_every == 0 or total >= args.total_env_steps:
            print(
                f"env_steps {total}/{args.total_env_steps} | h_rew_mean={m_coll['hunter_rew_mean']:.4f} | "
                f"bc={tstats['bc']:.4f} v={tstats['v']:.4f} loss={tstats['loss']:.4f}"
            )

    path_pt = out / "hunter.pt"
    meta_out = {
        **meta,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_env_steps": total,
    }
    if args.no_norm_obs:
        meta_out["note"] = "未作观测 RMS；PPO 若使用 normalize_obs 请勿混用此权重"
    save_hunter_pretrain(
        path_pt,
        cfg=env.cfg,
        pol=trainer.pol,
        obs_rms=trainer.obs_rms,
        rew_rms=trainer.rew_rms,
        meta=meta_out,
    )
    (out / "run.json").write_text(
        json.dumps(
            {**meta_out, "hunter_pt": str(path_pt.resolve())},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"已保存: {path_pt.resolve()}")


if __name__ == "__main__":
    main()
