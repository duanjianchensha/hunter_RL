"""
加载 checkpoint 中的猎人网络可视化；逃脱者用 `rule_action_escaper`。

默认：`configs/default.yaml` 驱动环境（非 checkpoint 内嵌 cfg）；`--seed 0`、`--max-episodes 10`、录制 `runs/viz_hunter_policy.mp4`；
不加 `--checkpoint` 无法运行（必填）。关闭录屏：`--no-record-mp4`。

  pip install -e ".[viz]"
  python scripts/viz_hunter_policy.py --checkpoint runs/hunter_rl.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from hunt_env.cli_defaults import (
    DEFAULT_CONFIG_YAML,
    VIZ_MAX_EPISODES,
    VIZ_MP4_HUNTER_POLICY,
    VIZ_SEED,
    env_cfg_for_viz,
)
from hunt_env.env.hunt_parallel import HuntParallelEnv
from hunt_env.policies.rules import rule_action_escaper
from hunt_env.render.mp4 import Mp4Recorder
from hunt_rl.actor_critic import ActorCritic, action_bounds_from_cfg
from hunt_rl.device import get_train_device
from hunt_rl.running_stats import RunningMeanStd


def _infer_backbone(
    state_dict: dict[str, torch.Tensor], prefix: str = "backbone"
) -> tuple[int, tuple[int, ...]]:
    """从 `ActorCritic.state_dict` 恢复 obs 维与隐藏层元组（Linear 在偶数下标）。"""
    obs_dim: int | None = None
    hiddens: list[int] = []
    i = 0
    while f"{prefix}.{i}.weight" in state_dict:
        w = state_dict[f"{prefix}.{i}.weight"]
        if obs_dim is None:
            obs_dim = int(w.shape[1])
        hiddens.append(int(w.shape[0]))
        i += 2
    if obs_dim is None:
        raise ValueError("state_dict 中未找到 backbone.*.weight，无法推断网络结构")
    return obs_dim, tuple(hiddens)


_CKPT_REJECT_SUFFIXES = frozenset({".log", ".txt", ".md", ".csv", ".json"})


def _load_checkpoint(path: Path) -> dict:
    suf = path.suffix.lower()
    if suf in _CKPT_REJECT_SUFFIXES:
        raise SystemExit(
            f"--checkpoint 指向文本文件 ({path.name})，无法加载权重。\n"
            "请使用训练保存的 .pt（例如 scripts/train_hunter_ppo_rule_escaper.py 的 --save 路径），"
            "不要使用 .log 日志。"
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        pass
    except Exception as e:
        _checkpoint_load_failed(path, e)
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        _checkpoint_load_failed(path, e)


def _checkpoint_load_failed(path: Path, err: BaseException) -> None:
    raise SystemExit(
        f"无法用 torch.load 解析权重文件：{path}\n"
        f"原因：{err}\n"
        "请确认路径为 PyTorch 保存的 .pt/.pth（含 state_dicts）。"
    ) from None


def main() -> None:
    p = argparse.ArgumentParser(
        description="猎人策略可视化：从 .pt 加载网络，与规则逃脱者同场"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="含 state_dicts['hunter'] 的 checkpoint（预训练或 PPO 存盘）",
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_YAML,
        help="环境与观测等均以该 YAML 为准（默认 configs/default.yaml）；与 checkpoint 内嵌 cfg 无关",
    )
    p.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="覆盖 YAML 中的单局最大步数；默认完全沿用 YAML",
    )
    p.add_argument("--seed", type=int, default=VIZ_SEED, help="首局 reset 种子；后续局随机")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=VIZ_MAX_EPISODES,
        help="自动退出前运行的局数；≤0 不限",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="对策略高斯采样（默认关闭：用确定性均值，便于和规则比行为）",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="前向设备（单环境可视化，CPU 通常足够）",
    )
    p.add_argument(
        "--obs-clip",
        type=float,
        default=10.0,
        help="与 HunterPretrainConfig.obs_clip 对齐；归一化 obs 的 clip 范围",
    )
    p.add_argument(
        "--record-mp4",
        type=str,
        default=VIZ_MP4_HUNTER_POLICY,
        help="MP4 输出路径（默认 runs/viz_hunter_policy.mp4）",
    )
    p.add_argument(
        "--no-record-mp4",
        action="store_true",
        help="只弹窗预览，不写 MP4",
    )
    p.add_argument(
        "--record-max-frames",
        type=int,
        default=None,
        help="最多写入帧数；达到后停止写入但可继续交互，直至退出",
    )
    args = p.parse_args()
    record_mp4_out = None if args.no_record_mp4 else args.record_mp4

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"找不到 checkpoint: {path}")

    if args.device == "auto":
        device = get_train_device(prefer_cuda=False)
    elif args.device == "cuda":
        device = get_train_device(prefer_cuda=True)
        if device.type != "cuda":
            raise RuntimeError("指定了 --device cuda 但当前不可用")
    else:
        device = get_train_device(prefer_cuda=False)

    ck = _load_checkpoint(path)
    sds = ck.get("state_dicts") or {}
    if "hunter" not in sds:
        raise SystemExit("checkpoint 中无 state_dicts['hunter']")
    sd_h = sds["hunter"]

    cfg = env_cfg_for_viz(args.config, args.max_episode_steps)
    yaml_disp = args.config
    print(f"环境参数来自 YAML：{yaml_disp}；sim.max_episode_steps={cfg.sim.max_episode_steps}")

    obs_dim_net, hidden = _infer_backbone(sd_h)
    env = HuntParallelEnv(cfg=cfg, render_mode="human")
    o_dim_env = env._obs_dim  # type: ignore[attr-defined]
    if o_dim_env != obs_dim_net:
        env.close()
        raise SystemExit(
            f"当前 YAML 观测维 {o_dim_env} 与 checkpoint 网络输入维 {obs_dim_net} 不一致。"
            "请使用与训练一致的 YAML（或更换 checkpoint）。"
        )

    pol = ActorCritic(obs_dim_net, 2, hidden).to(device)
    pol.load_state_dict(sd_h)
    pol.eval()

    # 与 PPO/预训练推理一致：不更新 running 统计；维数须与环境一致
    obs_rms: RunningMeanStd | None = None
    orm = ck.get("obs_rms")
    if isinstance(orm, dict) and "hunter" in orm:
        st = orm["hunter"]
        d = int(np.asarray(st["mean"]).size)
        if d != o_dim_env:
            print(f"警告：checkpoint obs_rms 维数 {d} 与 YAML 观测维 {o_dim_env} 不符，跳过 RMS，使用原始观测。")
        else:
            obs_rms = RunningMeanStd(d)
            obs_rms.set_state(st)

    al, ah = action_bounds_from_cfg(cfg, "hunter")
    lo = torch.as_tensor(al, dtype=torch.float32, device=device)
    hi = torch.as_tensor(ah, dtype=torch.float32, device=device)
    c_clip = float(args.obs_clip)

    def _prep_obs_vec(raw: np.ndarray) -> torch.Tensor:
        r = np.asarray(raw, dtype=np.float32)
        if obs_rms is None:
            t = r
        else:
            t = np.clip(obs_rms.normalize(r), -c_clip, c_clip).astype(np.float32, copy=False)
        return torch.as_tensor(t, device=device).unsqueeze(0)

    # 与预训练、train_ppo 的 escaper=规则 行为对齐：逃脱者用规则
    def build_actions(
        obs: dict[str, np.ndarray], agent_names: list[str]
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        nh = cfg.agents.n_hunters
        for i, name in enumerate(agent_names):
            o = obs[name]
            if i < nh:
                with torch.no_grad():
                    ot = _prep_obs_vec(o)
                    a_env, _, _, _ = pol.act(ot, lo, hi, deterministic=not args.stochastic)
                out[name] = a_env.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            else:
                out[name] = rule_action_escaper(o, cfg)
        return out

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
                print(f"已达 --record-max-frames={rec_cap}，已写入 {n} 帧，停止录屏（可继续操作窗口）")

    try:
        print(f"已加载: {path}  device={device}  确定性均值={'否' if args.stochastic else '是'}")
        _lim = f"{args.max_episodes} 局" if args.max_episodes > 0 else "不限局数"
        print(
            f"逃脱者: rule_action_escaper（与预训练/规则对照一致）。"
            f"局数上限: {_lim}（可用 --max-episodes 调整）。按 ESC 或关闭窗口退出。"
        )

        ob, infos = env.reset(seed=args.seed)
        session = 1
        print(f"--- 第 {session} 局 ---")
        _render_viz()

        running = True
        while running:
            if not env.agents:
                session += 1
                if args.max_episodes > 0 and session > args.max_episodes:
                    break
                ob, infos = env.reset(seed=None)
                print(f"--- 第 {session} 局 ---")
                _render_viz()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    running = False

            actions = build_actions({a: ob[a] for a in ob}, env.possible_agents)
            ob, rews, terms, truncs, infos = env.step(actions)
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
