"""PPO 训练用在线观测与奖励规约化（Welford 式合并更新）。"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """
    按维维护 mean/var，对输入做 (x - mean) / (sqrt(var) + eps)。
    update 可批量调用（典型 shape (batch, d) 或一维 d）。
    """

    def __init__(self, size: int, *, epsilon: float = 1e-4, eps: float = 1e-8) -> None:
        self._eps = float(eps)
        self._epsilon = float(epsilon)
        self.mean = np.zeros((size,), dtype=np.float64)
        self.var = np.ones((size,), dtype=np.float64)
        self.count = float(self._epsilon)

    def _ensure_shape(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def update(self, x: np.ndarray) -> None:
        """x: (batch, d) 或 (d,) 视为 1 行。"""
        batch = self._ensure_shape(x)
        batch_size = int(batch.shape[0])
        if batch_size == 0:
            return
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = float(batch_size)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        if tot_count < self._epsilon * 0.5:
            return
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * (self.count * batch_count) / tot_count
        new_var = m2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        s = (x - self.mean) / (np.sqrt(self.var) + self._eps)
        return s.astype(np.float32, copy=False)

    def get_state(self) -> dict[str, np.ndarray | float]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
            "eps": self._eps,
        }

    def set_state(self, d: dict[str, np.ndarray | float]) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64).reshape(-1).copy()
        self.var = np.asarray(d["var"], dtype=np.float64).reshape(-1).copy()
        self.count = float(d["count"])
        self._eps = float(d.get("eps", 1e-8))


class RunningRewardRMS:
    """
    标量奖励的 running 方差，用于 r / (sqrt(var) + eps)；维护单维 mean/var。
    """

    def __init__(self, *, epsilon: float = 1e-4, eps: float = 1e-8) -> None:
        self._eps = float(eps)
        self._epsilon = float(epsilon)
        self.mean = 0.0
        self.var = 1.0
        self.count = float(self._epsilon)

    def update(self, x: np.ndarray) -> None:
        flat = np.asarray(x, dtype=np.float64).ravel()
        if flat.size == 0:
            return
        batch_mean = float(np.mean(flat))
        batch_var = float(np.var(flat)) if flat.size > 1 else 0.0
        batch_count = float(flat.size)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m2 = self.var * self.count + batch_var * batch_count + (delta**2) * (self.count * batch_count) / tot_count
        new_var = m2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, r: np.ndarray) -> np.ndarray:
        s = r / (np.sqrt(self.var) + self._eps)
        return s.astype(np.float32, copy=False)

    def get_state(self) -> dict[str, float | np.ndarray]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "count": float(self.count),
            "eps": self._eps,
        }

    def set_state(self, d: dict[str, float | np.ndarray]) -> None:
        self.mean = float(d["mean"])
        self.var = float(d["var"])
        self.count = float(d["count"])
        self._eps = float(d.get("eps", 1e-8))
