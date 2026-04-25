"""
规则策略可视化：与 RL **同一 `obs` 向量** 输入；猎人/逃脱者用 `build_rule_actions_dict` 从观测算动作。

可见目标仅来自观测中 Top-K 槽位 + `other_is_escaper` 位；不访问引擎真值。
"""

from __future__ import annotations

import argparse
import sys

from hunt_env.config.loader import load_config
from hunt_env.env.hunt_parallel import HuntParallelEnv
from hunt_env.policies.rules import build_rule_actions_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="规则基线策略可视化")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    parser.add_argument("--seed", type=int, default=None, help="reset 随机种子")
    parser.add_argument("--max-episodes", type=int, default=3, help="自动进行的局数（每局结束会 reset）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = HuntParallelEnv(cfg=cfg, render_mode="human")

    import pygame

    pygame.init()
    clock = pygame.time.Clock()

    print("规则说明：与 RL 同观测维；槽位中可见的最近敌对方 → 追/逃；否则搜索或向心贴边。")
    print("按 ESC 或关闭窗口退出。")

    obs, infos = env.reset(seed=args.seed)
    session = 1
    print(f"--- 第 {session} 局 ---")
    env.render()

    running = True
    while running:
        if not env.agents:
            session += 1
            if session > args.max_episodes:
                break
            obs, infos = env.reset(seed=None)
            print(f"--- 第 {session} 局 ---")
            env.render()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        actions = build_rule_actions_dict(
            {a: obs[a] for a in obs},
            cfg=cfg,
            agent_names=env.possible_agents,
        )
        obs, rews, terms, truncs, infos = env.step(actions)
        env.render()

        if any(terms.values()) or any(truncs.values()):
            reason = "全捕获" if any(terms.values()) else "超时"
            print(f"局结束：{reason}，示例奖励 hunter_0={rews.get('hunter_0', 0):.2f}")

        clock.tick(cfg.render.fps)

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
