"""出生点间距。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.loader import load_config
from hunt_env.core.engine import _pairwise_min_dist, _spawn_positions


def test_spawn_respects_min_separation() -> None:
    cfg = load_config()
    rng = np.random.default_rng(0)
    e, n = 4, cfg.agents.n_hunters + cfg.agents.n_escapers
    sep = max(cfg.capture.capture_radius * 1.5, 0.5)
    pos = _spawn_positions(rng, cfg, e, n)
    for ei in range(e):
        assert _pairwise_min_dist(pos[ei]) >= sep - 1e-6
