"""
加载预训练 / PPO checkpoint 中的猎人网络，在 `human` 窗口中可视化其策略；逃脱者默认同预训练，用 `rule_action_escaper`。

与 `viz_rule_baseline.py` 同观测、同窗口交互（ESC/关闭退出）；便于肉眼对比“学到的猎人”与规则猎人。
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

from hunt_env.config.loader import load_config
from hunt_env.config.schema import HuntEnvConfig
from hunt_env.env.hunt_parallel import HuntParallelEnv
from hunt_env.policies.rules import rule_action_escaper
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


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_cfg(ck: dict, config_path: str | None) -> HuntEnvConfig:
    raw = ck.get("cfg")
    if raw is not None:
        if config_path is not None:
            print("提示：checkpoint 内已有 cfg，已忽略 --config（需与训练一致时请直接换 checkpoint）。")
        return HuntEnvConfig.model_validate(raw)
    if not config_path:
        raise SystemExit("checkpoint 中无 cfg 字段，请用 --config 指定与训练相同的 YAML。")
    return load_config(config_path)


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
        default=None,
        help="仅当 checkpoint 无 cfg 时必需；有 cfg 时一般不需要",
    )
    p.add_argument("--seed", type=int, default=None, help="首局 reset 种子；后续局为随机")
    p.add_argument("--max-episodes", type=int, default=3, help="自动进行的局数")
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
    args = p.parse_args()

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

    cfg = _resolve_cfg(ck, args.config)
    obs_dim, hidden = _infer_backbone(sd_h)
    pol = ActorCritic(obs_dim, 2, hidden).to(device)
    pol.load_state_dict(sd_h)
    pol.eval()

    # 与 PPO/预训练推理一致：不更新 running 统计
    obs_rms: RunningMeanStd | None = None
    orm = ck.get("obs_rms")
    if isinstance(orm, dict) and "hunter" in orm:
        st = orm["hunter"]
        d = int(np.asarray(st["mean"]).size)
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

    env = HuntParallelEnv(cfg=cfg, render_mode="human")
    o_dim_env = env._obs_dim  # type: ignore[attr-defined]
    if o_dim_env != obs_dim:
        raise SystemExit(
            f"配置观测维 {o_dim_env} 与网络输入维 {obs_dim} 不一致，请用训练时同一份 config/checkpoint。"
        )

    import pygame

    pygame.init()
    clock = pygame.time.Clock()

    print(f"已加载: {path}  device={device}  确定性均值={'否' if args.stochastic else '是'}")
    print("逃脱者: rule_action_escaper（与预训练/规则对照一致）。按 ESC 或关闭窗口退出。")

    ob, infos = env.reset(seed=args.seed)
    session = 1
    print(f"--- 第 {session} 局 ---")
    env.render()

    running = True
    while running:
        if not env.agents:
            session += 1
            if session > args.max_episodes:
                break
            ob, infos = env.reset(seed=None)
            print(f"--- 第 {session} 局 ---")
            env.render()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        actions = build_actions({a: ob[a] for a in ob}, env.possible_agents)
        ob, rews, terms, truncs, infos = env.step(actions)
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
