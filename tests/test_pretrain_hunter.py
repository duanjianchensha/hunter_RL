"""猎人规则预训练 smoke（需 torch）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from hunt_env.config.loader import load_config
from hunt_env.env.vectorized import HuntVectorizedEnv
from hunt_rl.pretrain_hunter import HunterPretrainConfig, HunterRulePretrainer, save_hunter_pretrain
from hunt_rl.trainer import MultiAgentPPOTrainer, PPOConfig


def test_hunter_pretrain_one_step() -> None:
    cfg = load_config(None)
    cfg.vectorization.num_envs = 2  # type: ignore[misc]
    env = HuntVectorizedEnv(cfg=cfg, num_envs=2)
    p = HunterPretrainConfig(
        update_epochs=1, num_minibatches=1, hidden_sizes=(32, 32)
    )
    tr = HunterRulePretrainer(env.cfg, env, pre_cfg=p, device=torch.device("cpu"))
    obs = env.reset(seed=0)
    next_obs, tstats, m = tr.train_step(8, obs)
    assert next_obs.shape == obs.shape
    assert "bc" in tstats
    assert np.isfinite(m["hunter_rew_mean"])


def test_pretrain_save_load_ppo_hunter() -> None:
    cfg = load_config(None)
    cfg.vectorization.num_envs = 1  # type: ignore[misc]
    env = HuntVectorizedEnv(cfg=cfg, num_envs=1)
    p = HunterPretrainConfig(
        update_epochs=1, num_minibatches=1, hidden_sizes=(16, 16)
    )
    tr = HunterRulePretrainer(env.cfg, env, pre_cfg=p, device=torch.device("cpu"))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "hunter.pt"
        save_hunter_pretrain(
            path,
            cfg=env.cfg,
            pol=tr.pol,
            obs_rms=tr.obs_rms,
            rew_rms=tr.rew_rms,
        )
        ppo = PPOConfig(update_epochs=1, num_minibatches=1)
        loaded = MultiAgentPPOTrainer(
            env.cfg, env, ppo_cfg=ppo, hidden_sizes=(16, 16), use_cuda_if_available=False
        )
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(path, map_location="cpu")
        if "hunter" in loaded.policies and "hunter" in ck.get("state_dicts", {}):
            loaded.policies["hunter"].load_state_dict(ck["state_dicts"]["hunter"])
        orms = ck.get("obs_rms")
        if isinstance(orms, dict) and "hunter" in orms and "hunter" in loaded._obs_rms:
            loaded._obs_rms["hunter"].set_state(orms["hunter"])
