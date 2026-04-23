"""规则基线动作在合法范围内。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.loader import load_config
from hunt_env.core.engine import HuntBatchEngine
from hunt_env.policies.rules import build_rule_actions_dict, rule_action_escaper, rule_action_hunter


def test_rule_actions_bounded() -> None:
    cfg = load_config()
    eng = HuntBatchEngine(cfg, num_envs=1)
    eng.reset(seed=0)
    names = [f"hunter_{i}" for i in range(cfg.agents.n_hunters)]
    names += [f"escaper_{j}" for j in range(cfg.agents.n_escapers)]
    for _ in range(20):
        act = build_rule_actions_dict(eng, 0, cfg, names)
        for i, n in enumerate(names):
            a, w = act[n]
            if i < cfg.agents.n_hunters:
                lim = cfg.agents.hunter_limits
            else:
                lim = cfg.agents.escaper_limits
            assert -lim.max_accel - 1e-5 <= a <= lim.max_accel + 1e-5
            assert -lim.max_omega - 1e-5 <= w <= lim.max_omega + 1e-5
        stacked = np.array([act[n] for n in names], dtype=np.float64)[None, ...]
        eng.step(stacked)


def test_hunter_escaper_return_floats() -> None:
    cfg = load_config()
    eng = HuntBatchEngine(cfg, num_envs=1)
    eng.reset(seed=1)
    ah = rule_action_hunter(eng, 0, 0, cfg)
    ae = rule_action_escaper(eng, 0, cfg.agents.n_hunters, cfg)
    assert len(ah) == 2 and len(ae) == 2
