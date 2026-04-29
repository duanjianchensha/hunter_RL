"""
规则策略可视化：`build_rule_actions_dict` 驱动，与 RL 同观测。

环境参数完全来自 `--config`（默认 configs/default.yaml）；
`seed 0`、`max-episodes 10`、默认录制 `runs/viz_rule_baseline.mp4`。关闭录屏：`--no-record-mp4`。
"""

from __future__ import annotations

import argparse
import sys

from hunt_env.cli_defaults import (
    DEFAULT_CONFIG_YAML,
    VIZ_MAX_EPISODES,
    VIZ_MP4_RULE_BASELINE,
    VIZ_SEED,
)
from hunt_env.config.loader import load_config
from hunt_env.env.hunt_parallel import HuntParallelEnv
from hunt_env.policies.rules import build_rule_actions_dict
from hunt_env.render.mp4 import Mp4Recorder


def main() -> None:
    parser = argparse.ArgumentParser(description="规则基线策略可视化")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_YAML, help="环境参数来源（默认项目 configs/default.yaml）")
    parser.add_argument("--seed", type=int, default=VIZ_SEED, help="首局 reset 种子")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=VIZ_MAX_EPISODES,
        help="跑满多少局后自动退出；≤0 不限",
    )
    parser.add_argument(
        "--record-mp4",
        type=str,
        default=VIZ_MP4_RULE_BASELINE,
        help="MP4 路径（默认 runs/viz_rule_baseline.mp4）",
    )
    parser.add_argument("--no-record-mp4", action="store_true", help="不写 MP4")
    parser.add_argument(
        "--record-max-frames",
        type=int,
        default=None,
        help="最多写入帧数；达到后停止写入但可继续交互",
    )
    args = parser.parse_args()

    record_mp4_out = None if args.no_record_mp4 else args.record_mp4

    cfg = load_config(args.config)
    env = HuntParallelEnv(cfg=cfg, render_mode="human")

    recorder: Mp4Recorder | None = None
    if record_mp4_out:
        recorder = Mp4Recorder(record_mp4_out, cfg.render.fps)
        print(f"录制 MP4 → {record_mp4_out}，fps={cfg.render.fps}")
    rec_cap = args.record_max_frames

    import pygame

    pygame.init()
    clock = pygame.time.Clock()

    def _render_viz() -> None:
        nonlocal recorder
        use_rgb = recorder is not None
        fr = env.render(rgb=use_rgb)
        if recorder is not None and fr is not None:
            recorder.append(fr)
            if rec_cap is not None and recorder.frame_count >= rec_cap:
                n = recorder.frame_count
                recorder.close()
                recorder = None
                print(f"已达 --record-max-frames={rec_cap}，已写入 {n} 帧，停止录屏")

    try:
        print("规则说明：与 RL 同观测维；槽位中可见的最近敌对方 → 追/逃；否则搜索或向心贴边。")
        _lim = f"{args.max_episodes} 局" if args.max_episodes > 0 else "不限局数"
        print(f"局数上限: {_lim}（--max-episodes）。按 ESC 或关闭窗口退出。")

        obs, infos = env.reset(seed=args.seed)
        session = 1
        print(f"--- 第 {session} 局 ---")
        _render_viz()

        running = True
        while running:
            if not env.agents:
                session += 1
                if args.max_episodes > 0 and session > args.max_episodes:
                    break
                obs, infos = env.reset(seed=None)
                print(f"--- 第 {session} 局 ---")
                _render_viz()

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
            _render_viz()

            if any(terms.values()) or any(truncs.values()):
                reason = "全捕获" if any(terms.values()) else "超时"
                print(f"局结束：{reason}，示例奖励 hunter_0={rews.get('hunter_0', 0):.2f}")

            clock.tick(cfg.render.fps)
    finally:
        if recorder is not None:
            n = recorder.frame_count
            path_mp4 = recorder.path
            recorder.close()
            print(f"已保存 MP4：{path_mp4}（{n} 帧）")
        env.close()
        pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
