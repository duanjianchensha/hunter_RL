#!/usr/bin/env python3
"""
猎人 + 逃脱者 **同时** PPO（`MultiAgentPPOTrainer` 默认双侧 learn）。

环境与并行数默认来自 YAML（一般为 `configs/default.yaml`）；可选用各自的规则预训练 checkpoint 热启动。

  pip install -e ".[rl]"
  python scripts/train_bilateral_ppo.py

从双方规则预训练继续 RL：

  python scripts/train_bilateral_ppo.py ^
    --init-hunter pretrained/hunter_rule/hunter.pt ^
    --init-escaper pretrained/escaper_rule/escaper.pt

说明：`scripts/train_ppo.py` 在 `n_hunters>=1` 且 `n_escapers>=1` 且未指定 rule 模式时行为与本脚本一致；
本脚本仅收紧默认值（默认 `--save`、`--total-steps`）并在启动前校验 YAML。
双边博弈默认 **5000 万**环境步量级；可用 `--total-steps` / `--save-every` 按需调整。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hunt_env.cli_defaults import DEFAULT_CONFIG_YAML
from hunt_env.config.loader import load_config

import train_ppo as tp


def main() -> None:
    p = tp.build_train_ppo_parser(description="双边（猎人+逃脱者）同时 PPO")
    # 默认长训：50M 步；save_every=5M 控制中间 ckpt 数量，避免过小间隔占满磁盘
    p.set_defaults(
        save="runs/bilateral_ppo.pt",
        total_steps=50_000_000,
        save_every=1_000_000,
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.agents.n_hunters < 1 or cfg.agents.n_escapers < 1:
        raise SystemExit(
            "双边训练要求 YAML 中 agents.n_hunters>=1 且 agents.n_escapers>=1；"
            f"当前 nh={cfg.agents.n_hunters}, ne={cfg.agents.n_escapers}"
        )

    num_envs_eff = args.num_envs if args.num_envs is not None else cfg.vectorization.num_envs
    steps_per_update = args.rollout_len * num_envs_eff
    est_updates = math.ceil(args.total_steps / steps_per_update) if steps_per_update > 0 else 0

    from hunt_rl.train_log import tee_stdout_stderr

    with tee_stdout_stderr(args.log_file):
        src = "CLI --num-envs" if args.num_envs is not None else "YAML vectorization.num_envs"
        print(
            f"双边 PPO：config={args.config}，nh={cfg.agents.n_hunters}，ne={cfg.agents.n_escapers}，"
            f"并行 env={num_envs_eff}（{src}）"
        )
        print(
            f"终止条件：累计环境步数 ≥ --total-steps（当前 {args.total_steps}）；"
            f"每轮约采集 {steps_per_update} 步，预计约 {est_updates} 次更新后出现 final save（属正常结束）。"
        )
        tp.run_train_ppo(args)


if __name__ == "__main__":
    main()
