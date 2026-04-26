"""规则基线动作在合法范围内，且仅依赖观测向量（与引擎接口解耦）。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.loader import load_config
from hunt_env.core.engine import HuntBatchEngine
from hunt_env.core.state import total_obs_dim
from hunt_env.policies.rules import build_rule_actions_dict, rule_action_escaper, rule_action_hunter


def test_rule_actions_bounded() -> None:
    cfg = load_config()
    eng = HuntBatchEngine(cfg, num_envs=1)
    obs_t = eng.reset(seed=0)
    d = total_obs_dim(cfg)
    assert obs_t.shape[2] == d
    names = [f"hunter_{i}" for i in range(cfg.agents.n_hunters)]
    names += [f"escaper_{j}" for j in range(cfg.agents.n_escapers)]
    for _ in range(20):
        obs = {names[i]: obs_t[0, i] for i in range(len(names))}
        act = build_rule_actions_dict(obs, cfg, names)
        for i, n in enumerate(names):
            a, w = act[n]
            if i < cfg.agents.n_hunters:
                lim = cfg.agents.hunter_limits
            else:
                lim = cfg.agents.escaper_limits
            assert -lim.max_accel - 1e-5 <= a <= lim.max_accel + 1e-5
            assert -lim.max_omega - 1e-5 <= w <= lim.max_omega + 1e-5
        stacked = np.array([act[n] for n in names], dtype=np.float64)[None, ...]
        obs_t, _, _, _, _ = eng.step(stacked)


def test_hunter_escaper_from_obs_shape() -> None:
    cfg = load_config()
    eng = HuntBatchEngine(cfg, num_envs=1)
    obs_t = eng.reset(seed=1)
    nh = cfg.agents.n_hunters
    ah = rule_action_hunter(obs_t[0, 0], cfg)
    ae = rule_action_escaper(obs_t[0, nh], cfg)
    assert ah.shape == (2,) and ae.shape == (2,)


def test_hunter_uses_search_when_no_visible_escaper() -> None:
    """全零有效槽时猎人走搜索回退，动作仍落在限制内。"""
    cfg = load_config()
    d = total_obs_dim(cfg)
    o = np.zeros(d, dtype=np.float32)
    o[0] = 1.0
    o[1] = 1.0
    o[6] = 0.0
    out = rule_action_hunter(o, cfg)
    assert out.shape == (2,)
    # 小线加速+慢转
    assert float(np.hypot(out[0], out[1])) > 0.01
