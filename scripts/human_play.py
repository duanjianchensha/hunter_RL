"""键盘试玩：一名智能体由人控制，其余动作为零。"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="追猎环境人类试玩")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置路径")
    args = parser.parse_args()

    from hunt_env.config.loader import load_config
    from hunt_env.env.hunt_parallel import HuntParallelEnv

    cfg = load_config(args.config)
    env = HuntParallelEnv(cfg=cfg, render_mode="human")
    hc = cfg.human_control
    nh = cfg.agents.n_hunters
    names = env.possible_agents
    if hc.initial_control_agent == "hunter":
        ctrl_idx = min(hc.initial_agent_index, nh - 1)
    else:
        ctrl_idx = nh + min(hc.initial_agent_index, cfg.agents.n_escapers - 1)
    ctrl_name = names[ctrl_idx]

    import pygame

    pygame.init()
    clock = pygame.time.Clock()
    print("控制说明：W/S 线加减速，A/D 角速度；Q 切换控制的智能体；ESC 退出")
    print(f"当前控制：{ctrl_name}")
    print("提示：请先点击游戏窗口再按 WASD（否则键盘可能读不到）。")

    running = True
    obs, infos = env.reset()
    # 先创建显示窗口，再进入主循环；否则 Windows 下首次 step 时 pygame.key.get_pressed 常全为假
    env.render()

    while running:
        if not env.agents:
            obs, infos = env.reset()
        pygame.event.pump()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                ctrl_idx = (ctrl_idx + 1) % len(names)
                ctrl_name = names[ctrl_idx]
                print(f"当前控制：{ctrl_name}")

        keys = pygame.key.get_pressed()
        a = 0.0
        w = 0.0
        if keys[pygame.K_w]:
            a += hc.accel_step
        if keys[pygame.K_s]:
            a -= hc.accel_step
        if keys[pygame.K_d]:
            w += hc.omega_step
        if keys[pygame.K_a]:
            w -= hc.omega_step

        actions = {n: np.zeros(2, dtype=np.float32) for n in names}
        actions[ctrl_name] = np.array([a, w], dtype=np.float32)

        obs, rews, terms, truncs, infos = env.step(actions)
        env.render()

        if any(terms.values()) or any(truncs.values()):
            print("局结束，下一帧将 reset")

        clock.tick(cfg.render.fps)

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
