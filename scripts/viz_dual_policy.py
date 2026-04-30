"""
从同一 checkpoint 加载猎人 + 逃脱者网络同场可视化（双边 RL / 双边预训练存盘）。

环境与观测维以 `--config`（默认 configs/default.yaml）为准；checkpoint 仅提供权重与 obs_rms。

  pip install -e ".[viz]"
  python scripts/viz_dual_policy.py --checkpoint runs/bilateral_ppo.pt
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
    VIZ_MP4_DUAL_POLICY,
    VIZ_SEED,
    env_cfg_for_viz,
)
from hunt_env.env.hunt_parallel import HuntParallelEnv
from hunt_env.render.mp4 import Mp4Recorder
from hunt_rl.actor_critic import ActorCritic, action_bounds_from_cfg
from hunt_rl.device import get_train_device
from hunt_rl.running_stats import RunningMeanStd


def _infer_backbone(
    state_dict: dict[str, torch.Tensor], prefix: str = "backbone"
) -> tuple[int, tuple[int, ...]]:
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
            f"--checkpoint 指向非权重文件 ({path.name})。请使用 train_ppo / train_bilateral_ppo 保存的 .pt。"
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
        f"无法用 torch.load 解析：{path}\n原因：{err}\n请确认含 state_dicts['hunter'] 与 state_dicts['escaper']。"
    ) from None


def main() -> None:
    p = argparse.ArgumentParser(description="双边策略同场可视化（checkpoint 须含 hunter + escaper）")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="含 state_dicts['hunter'] 与 state_dicts['escaper'] 的 checkpoint",
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_YAML,
        help="环境与观测以该 YAML 为准；与 checkpoint 内嵌 cfg 无关",
    )
    p.add_argument("--max-episode-steps", type=int, default=None, help="覆盖单局最大步数")
    p.add_argument("--seed", type=int, default=VIZ_SEED, help="首局 reset 种子")
    p.add_argument(
        "--max-episodes",
        type=int,
        default=VIZ_MAX_EPISODES,
        help="自动退出局数；≤0 不限",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="策略高斯采样（默认定性：用均值）",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--obs-clip", type=float, default=10.0, help="obs 归一化后 clip（与 PPO 默认一致）")
    p.add_argument(
        "--record-mp4",
        type=str,
        default=VIZ_MP4_DUAL_POLICY,
        help="默认 runs/viz_dual_policy.mp4",
    )
    p.add_argument("--no-record-mp4", action="store_true")
    p.add_argument("--record-max-frames", type=int, default=None)
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
    if "hunter" not in sds or "escaper" not in sds:
        raise SystemExit("checkpoint 须同时包含 state_dicts['hunter'] 与 state_dicts['escaper']（双边训练存盘）")

    sd_h, sd_e = sds["hunter"], sds["escaper"]
    o_h, hid_h = _infer_backbone(sd_h)
    o_e, hid_e = _infer_backbone(sd_e)
    if o_h != o_e:
        raise SystemExit(f"猎人与逃脱者观测维不一致：{o_h} vs {o_e}，无法同场可视化")

    cfg = env_cfg_for_viz(args.config, args.max_episode_steps)
    print(f"环境参数来自 YAML：{args.config}；sim.max_episode_steps={cfg.sim.max_episode_steps}")

    env = HuntParallelEnv(cfg=cfg, render_mode="human")
    o_dim_env = env._obs_dim  # type: ignore[attr-defined]
    if o_dim_env != o_h:
        env.close()
        raise SystemExit(
            f"YAML 观测维 {o_dim_env} 与 checkpoint 网络输入维 {o_h} 不一致；请换 YAML 或 checkpoint。"
        )

    pol_h = ActorCritic(o_h, 2, hid_h).to(device)
    pol_e = ActorCritic(o_e, 2, hid_e).to(device)
    pol_h.load_state_dict(sd_h)
    pol_e.load_state_dict(sd_e)
    pol_h.eval()
    pol_e.eval()

    obs_rms_h: RunningMeanStd | None = None
    obs_rms_e: RunningMeanStd | None = None
    orm = ck.get("obs_rms")
    if isinstance(orm, dict):
        if "hunter" in orm:
            st = orm["hunter"]
            d = int(np.asarray(st["mean"]).size)
            if d != o_dim_env:
                print(f"警告：checkpoint hunter obs_rms 维数 {d}，跳过（YAML 观测维 {o_dim_env}）")
            else:
                obs_rms_h = RunningMeanStd(d)
                obs_rms_h.set_state(st)
        if "escaper" in orm:
            st = orm["escaper"]
            d = int(np.asarray(st["mean"]).size)
            if d != o_dim_env:
                print(f"警告：checkpoint escaper obs_rms 维数 {d}，跳过（YAML 观测维 {o_dim_env}）")
            else:
                obs_rms_e = RunningMeanStd(d)
                obs_rms_e.set_state(st)

    al_h, ah_h = action_bounds_from_cfg(cfg, "hunter")
    al_e, ah_e = action_bounds_from_cfg(cfg, "escaper")
    lo_h = torch.as_tensor(al_h, dtype=torch.float32, device=device)
    hi_h = torch.as_tensor(ah_h, dtype=torch.float32, device=device)
    lo_e = torch.as_tensor(al_e, dtype=torch.float32, device=device)
    hi_e = torch.as_tensor(ah_e, dtype=torch.float32, device=device)
    c_clip = float(args.obs_clip)

    def _prep(role: str, raw: np.ndarray) -> torch.Tensor:
        r = np.asarray(raw, dtype=np.float32)
        if role == "hunter":
            m = obs_rms_h
        else:
            m = obs_rms_e
        if m is None:
            t = r
        else:
            t = np.clip(m.normalize(r), -c_clip, c_clip).astype(np.float32, copy=False)
        return torch.as_tensor(t, device=device).unsqueeze(0)

    nh = cfg.agents.n_hunters

    def build_actions(obs: dict[str, np.ndarray], agent_names: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i, name in enumerate(agent_names):
            o = obs[name]
            if i < nh:
                with torch.no_grad():
                    ot = _prep("hunter", o)
                    a_env, _, _, _ = pol_h.act(ot, lo_h, hi_h, deterministic=not args.stochastic)
                out[name] = a_env.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            else:
                with torch.no_grad():
                    ot = _prep("escaper", o)
                    a_env, _, _, _ = pol_e.act(ot, lo_e, hi_e, deterministic=not args.stochastic)
                out[name] = a_env.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
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
                print(f"已达 --record-max-frames={rec_cap}，已写入 {n} 帧，停止录屏")

    try:
        print(f"已加载双边: {path}  device={device}  stochastic={'是' if args.stochastic else '否（均值）'}")
        _lim = f"{args.max_episodes} 局" if args.max_episodes > 0 else "不限局数"
        print(f"局数上限: {_lim}。ESC / 关闭窗口退出。")

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
                hk = "hunter_0" if "hunter_0" in rews else next((k for k in rews if k.startswith("hunter_")), None)
                ek = next((k for k in rews if k.startswith("escaper_")), None)
                hr = float(rews.get(hk, 0)) if hk else 0.0
                er = float(rews.get(ek, 0)) if ek else 0.0
                print(f"局结束：{reason}，hunter≈{hr:.2f} escaper≈{er:.2f}")

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
