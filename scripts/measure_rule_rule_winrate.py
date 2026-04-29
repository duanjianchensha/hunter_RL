"""
双方均用规则策略（与训练同 obs）连打若干局，统计猎人胜 / 逃脱者胜比例。

猎人胜：本局 termination（全捕获）；逃脱者胜：truncation（超时且仍有逃脱者存活）。
"""

from __future__ import annotations

import argparse

import numpy as np

from hunt_env.cli_defaults import DEFAULT_CONFIG_YAML
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_env.policies.rules import build_rule_actions_dict


def _episode_outcome(term: np.ndarray, trunc: np.ndarray) -> str:
    """单环境：任取 agent 0 的 term/trunc（全智能体一致）。"""
    t0, tr0 = bool(term[0, 0]), bool(trunc[0, 0])
    if t0 and tr0:
        return "ambiguous"
    if t0:
        return "hunter"
    if tr0:
        return "escaper"
    return "ongoing"


def run_episodes(env: HuntVectorizedEnv, *, n_episodes: int, seed_base: int) -> tuple[int, int, int]:
    nh = env.cfg.agents.n_hunters
    ne = env.cfg.agents.n_escapers
    if nh != 1 or ne != 1:
        raise ValueError(f"本脚本按 1v1 统计，当前 nh={nh}, ne={ne}")

    hunter_wins = escaper_wins = ambiguous = 0
    for ep in range(n_episodes):
        obs = env.reset(seed=None if seed_base < 0 else seed_base + ep)
        while True:
            obs_dict = {name: obs[0, i].astype(np.float32, copy=False) for i, name in enumerate(env.agent_ids)}
            act_dict = build_rule_actions_dict(obs_dict, env.cfg, list(env.agent_ids))
            actions = np.stack([act_dict[name] for name in env.agent_ids], axis=0)[np.newaxis, ...]
            obs, _rew, term, trunc, _info = env.step(actions)
            out = _episode_outcome(term, trunc)
            if out == "ongoing":
                continue
            if out == "hunter":
                hunter_wins += 1
            elif out == "escaper":
                escaper_wins += 1
            else:
                ambiguous += 1
            break
    return hunter_wins, escaper_wins, ambiguous


def main() -> None:
    p = argparse.ArgumentParser(description="规则 vs 规则 胜率统计（1v1）")
    p.add_argument("--config", type=str, default=None, help=f"YAML（默认 {DEFAULT_CONFIG_YAML}）")
    p.add_argument("--episodes", type=int, default=100, help="对局数")
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="第 i 局 reset(seed=seed_base+i)；设为 -1 则每局 reset(seed=None) 真随机",
    )
    args = p.parse_args()

    cfg_path = args.config if args.config is not None else DEFAULT_CONFIG_YAML

    env = HuntVectorizedEnv.from_yaml(cfg_path, num_envs=1)
    h, e, amb = run_episodes(env, n_episodes=args.episodes, seed_base=args.seed_base)
    total = h + e + amb
    print(f"配置: {cfg_path}")
    print(f"局数: {args.episodes}（seed_base={args.seed_base}）")
    print(f"猎人胜: {h} ({100.0 * h / total:.1f}%)" if total else "猎人胜: 0")
    print(f"逃脱者胜: {e} ({100.0 * e / total:.1f}%)" if total else "逃脱者胜: 0")
    if amb:
        print(f"异常同时 term+trunc: {amb}")


if __name__ == "__main__":
    main()
