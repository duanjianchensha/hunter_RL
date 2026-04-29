"""逐智能体独立视野：距离圆 + 可选扇形；无队伍融合。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.state import wrap_angle


def visible_pair_mask(
    pos: np.ndarray,
    theta: np.ndarray,
    active: np.ndarray,
    cfg: HuntEnvConfig,
) -> np.ndarray:
    """
    返回 (E, N, N) 布尔：对观察者 i、目标 j（i!=j），j 是否对 i 可见。
    仅统计双方均 active 的对；同阵营与敌方规则相同。
    """
    e, n, _ = pos.shape
    # 相对位移 (E, N, N, 2): V[i,j] = pos[j]-pos[i]
    pi = pos[:, :, np.newaxis, :]  # (E, N, 1, 2)
    pj = pos[:, np.newaxis, :, :]  # (E, 1, N, 2)
    diff = pj - pi
    dist = np.linalg.norm(diff, axis=-1)  # (E, N, N)

    eye = np.eye(n, dtype=bool)[np.newaxis, :, :]
    both = active[:, :, np.newaxis] & active[:, np.newaxis, :]
    both = both & ~np.broadcast_to(eye, both.shape)

    rad = cfg.visibility.radii_per_observer(cfg.agents.n_hunters, n)
    in_range = dist <= rad[np.newaxis, :, np.newaxis]

    if cfg.visibility.use_sector_fov and cfg.visibility.fov_deg is not None:
        half = np.deg2rad(cfg.visibility.fov_deg) / 2.0
        # 从 i 指向 j 的方位角
        ang = np.arctan2(diff[..., 1], diff[..., 0])
        rel = wrap_angle(ang - theta[:, :, np.newaxis])
        in_fov = np.abs(rel) <= half + 1e-6
        vis = both & in_range & in_fov
    else:
        vis = both & in_range

    # 对角线与自环已在 both 中排除
    return vis


def topk_visible_indices(
    pos: np.ndarray,
    active: np.ndarray,
    visible: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个观察者 i，在 visible[i,j] 为真的 j 中取距离最近的 K 个。

    返回 dist_top (E, N, K) 与 idx_top (E, N, K)，不足位置 idx=-1, dist=inf。
    """
    e, n, _ = pos.shape
    # 距离矩阵
    diff = pos[:, :, np.newaxis, :] - pos[:, np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (E, N, N)

    dist_large = np.where(visible, dist, np.inf)
    # 将对角线设为 inf
    eye = np.eye(n, dtype=bool)
    dist_large[:, eye] = np.inf

    # 取 top-k 最小距离：对部分环境用 argpartition
    idx_top = np.full((e, n, k), -1, dtype=np.int32)
    dist_top = np.full((e, n, k), np.inf, dtype=np.float64)

    for ei in range(e):
        for i in range(n):
            row = dist_large[ei, i]
            finite = np.isfinite(row)
            if not np.any(finite):
                continue
            # 候选索引
            js = np.where(finite)[0]
            vals = row[js]
            kk = min(k, len(js))
            part = np.argpartition(vals, kk - 1)[:kk]
            chosen = js[part]
            order = np.argsort(row[chosen])
            chosen = chosen[order][:kk]
            dist_top[ei, i, :kk] = row[chosen]
            idx_top[ei, i, :kk] = chosen.astype(np.int32)

    return dist_top, idx_top
