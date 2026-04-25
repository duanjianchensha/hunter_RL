"""obs 解码与配置一致；错误长度须为 ValueError 而非 IndexError。"""

from __future__ import annotations

import numpy as np
import pytest

from hunt_env.config.loader import load_config
from hunt_env.core.state import total_obs_dim
from hunt_env.policies.obs_layout import decode_observation


def test_decode_wrong_length_is_valueerror_not_indexerror() -> None:
    cfg = load_config()
    d = total_obs_dim(cfg)
    for bad_len in (d - 1, d + 1):
        with pytest.raises(ValueError) as ei:
            decode_observation(np.zeros(bad_len, dtype=np.float64), cfg)
        assert "total_obs_dim" in str(ei.value) or "维数" in str(ei.value)
    # 正确长度不抛
    decode_observation(np.zeros(d, dtype=np.float64), cfg)
