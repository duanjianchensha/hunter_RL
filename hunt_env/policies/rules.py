"""
基于规则的策略：与 RL **同一观测向量** `obs` 与 **同一动作形状** `(2,)` float32。

决策仅使用 `decode_observation` 解析出的量 + 与动作上界相同的 `cfg` 运动学限制（与 PettingZoo 中
`action_space` 一致）；不读取 `HuntBatchEngine`。
"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import AgentLimitsConfig, HuntEnvConfig
from hunt_env.core.state import wrap_angle
from hunt_env.policies.obs_layout import DecodedObs, decode_observation, rel_to_world_delta


def _steer_to_direction(
    theta: float,
    speed: float,
    dir_x: float,
    dir_y: float,
    limits: AgentLimitsConfig,
    *,
    target_speed_frac: float = 0.85,
    turn_gain: float = 2.2,
    align_threshold: float = 0.45,
) -> tuple[float, float]:
    n = np.hypot(dir_x, dir_y)
    if n < 1e-9:
        return 0.0, 1.2

    ux, uy = dir_x / n, dir_y / n
    theta_des = float(np.arctan2(uy, ux))
    err = float(wrap_angle(np.asarray(theta_des - theta, dtype=np.float64)))
    max_a = limits.max_accel
    max_w = limits.max_omega

    omega = float(np.clip(turn_gain * err, -max_w, max_w))
    v_tgt = min(limits.max_speed * target_speed_frac, limits.max_speed)

    if abs(err) < align_threshold:
        a = float(np.clip(1.5 * (v_tgt - speed), -max_a, max_a))
    else:
        a = float(np.clip(0.25 * max_a * max(np.cos(err), -0.2), -max_a, max_a))

    return a, omega


def _hunter_search_no_target(
    theta: float,
    _speed: float,
    limits: AgentLimitsConfig,
) -> tuple[float, float]:
    max_a = limits.max_accel
    max_w = limits.max_omega
    a = float(0.2 * max_a)
    omega = float(0.35 * max_w)
    return a, omega


def rule_action_hunter(obs: np.ndarray, cfg: HuntEnvConfig) -> np.ndarray:
    """
    猎人：在观测 Top-K 槽位中，取 **mask 有效且 other 为逃脱者** 的最近槽，朝其世界系相对方向追击。
    若无：搜索回退（与空槽一致，仅自车姿态）。返回 shape (2,) float32。
    """
    dec = decode_observation(obs, cfg)
    lim = cfg.agents.hunter_limits
    th = float(dec.self_vec[6])
    vx, vy = float(dec.self_vec[2]), float(dec.self_vec[3])
    speed = float(np.hypot(vx, vy))
    use_ego = cfg.observation.use_ego_frame_for_others

    k = dec.slots.shape[0]
    best_t = -1
    best_d = np.inf
    for t in range(k):
        sl = dec.slots[t]
        if sl[5] < 0.5:
            continue
        if sl[6] < 0.5:
            continue
        rx, ry = float(sl[0]), float(sl[1])
        d = float(np.hypot(rx, ry))
        if d < best_d:
            best_d, best_t = d, t

    if best_t < 0:
        a, w = _hunter_search_no_target(th, speed, lim)
    else:
        sl = dec.slots[best_t]
        wx, wy = rel_to_world_delta(float(sl[0]), float(sl[1]), th, use_ego)
        a, w = _steer_to_direction(th, speed, wx, wy, lim)

    return np.array([a, w], dtype=np.float32)


def _escaper_center_wall_steer(dec: DecodedObs, cfg: HuntEnvConfig) -> tuple[float, float]:
    """无可见猎人：仅用 wall_dist + 自车位置衍生向心方向（与观测一致）。"""
    if dec.wall_dist is None:
        raise ValueError("逃脱者规则需要 observation.include_world_bounds=True")
    lim = cfg.agents.escaper_limits
    d_l, d_r, d_b, d_t = [float(x) for x in dec.wall_dist]
    th = float(dec.self_vec[6])
    vx, vy = float(dec.self_vec[2]), float(dec.self_vec[3])
    sp = float(np.hypot(vx, vy))

    margin = 1.0
    rx, ry = 0.0, 0.0
    if d_l < margin:
        rx += 1.0
    if d_r < margin:
        rx -= 1.0
    if d_b < margin:
        ry += 1.0
    if d_t < margin:
        ry -= 1.0
    rlen = float(np.hypot(rx, ry))
    if rlen > 1e-9:
        rx, ry = rx / rlen, ry / rlen

    d_edge = min(d_l, d_r, d_b, d_t)
    wall_w = float(np.clip((1.7 - d_edge) / 1.7, 0.0, 1.0))

    cx = (d_r - d_l) * 0.5
    cy = (d_t - d_b) * 0.5
    clen = float(np.hypot(cx, cy))
    if clen > 1e-6:
        cnx, cny = cx / clen, cy / clen
    else:
        cnx, cny = 1.0, 0.0

    mx = (1.0 - 0.42 * wall_w) * cnx + 0.42 * wall_w * rx
    my = (1.0 - 0.42 * wall_w) * cny + 0.42 * wall_w * ry
    mn = float(np.hypot(mx, my))
    if mn < 1e-9:
        fx, fy = cnx, cny
    else:
        fx, fy = mx / mn, my / mn

    return _steer_to_direction(
        th,
        sp,
        fx,
        fy,
        lim,
        target_speed_frac=0.75,
        turn_gain=2.0,
        align_threshold=0.42,
    )


def rule_action_escaper(obs: np.ndarray, cfg: HuntEnvConfig) -> np.ndarray:
    """
    逃脱者：最近 **可见猎人** 槽上逃离，启发式同原设计但威胁几何来自观测；无猎人槽时用边界/向心。返回 (2,) float32。
    """
    dec = decode_observation(obs, cfg)
    lim = cfg.agents.escaper_limits
    use_ego = cfg.observation.use_ego_frame_for_others
    k = dec.slots.shape[0]
    th = float(dec.self_vec[6])
    vx, vy = float(dec.self_vec[2]), float(dec.self_vec[3])
    sp = float(np.hypot(vx, vy))
    cap = float(cfg.capture.capture_radius)

    best_t = -1
    best_d = np.inf
    for t in range(k):
        sl = dec.slots[t]
        if sl[5] < 0.5:
            continue
        if sl[6] > 0.5:
            continue
        d = float(np.hypot(float(sl[0]), float(sl[1])))
        if d < best_d:
            best_d, best_t = d, t

    if best_t < 0:
        a, w = _escaper_center_wall_steer(dec, cfg)
        return np.array([a, w], dtype=np.float32)

    sl = dec.slots[best_t]
    wx, wy = rel_to_world_delta(float(sl[0]), float(sl[1]), th, use_ego)
    # 自车->猎人 方向为 (wx,wy)，逃离为反向
    flee_x, flee_y = -wx, -wy
    fn = float(np.hypot(flee_x, flee_y))
    if fn < 1e-9:
        a, w = 0.0, 1.8
        return np.array([a, w], dtype=np.float32)
    fx, fy = flee_x / fn, flee_y / fn

    if dec.wall_dist is None:
        raise ValueError("逃脱者规则需要 observation.include_world_bounds=True")
    d_l, d_r, d_b, d_t = [float(x) for x in dec.wall_dist]
    # 边界推开（用 wall 距离，避免依赖引擎）
    margin = 1.0
    rx, ry = 0.0, 0.0
    if d_l < margin:
        rx += 1.0
    if d_r < margin:
        rx -= 1.0
    if d_b < margin:
        ry += 1.0
    if d_t < margin:
        ry -= 1.0
    rlen = float(np.hypot(rx, ry))
    if rlen > 1e-9:
        rx, ry = rx / rlen, ry / rlen

    d_edge = min(d_l, d_r, d_b, d_t)
    wall_w = float(np.clip((1.7 - d_edge) / 1.7, 0.0, 1.0))
    danger = float(np.clip((cap * 5.0 - best_d) / (cap * 5.0 + 1e-9), 0.0, 1.0))

    px, py = -fy, fx
    cx = (d_r - d_l) * 0.5
    cy = (d_t - d_b) * 0.5
    cross_z = px * cy - py * cx
    sign = 1.0 if cross_z >= 0.0 else -1.0

    mx = (1.0 - 0.42 * wall_w) * fx + 0.42 * wall_w * rx + (0.55 * danger) * sign * px
    my = (1.0 - 0.42 * wall_w) * fy + 0.42 * wall_w * ry + (0.55 * danger) * sign * py
    clen = float(np.hypot(cx, cy))
    if clen > 1e-6:
        mx += 0.12 * (1.0 - 0.5 * danger) * (cx / clen)
        my += 0.12 * (1.0 - 0.5 * danger) * (cy / clen)

    mn = float(np.hypot(mx, my))
    if mn < 1e-9:
        dx, dy = fx, fy
    else:
        dx, dy = mx / mn, my / mn

    tg = 2.3 + 2.8 * danger
    vfrac = 0.88 + 0.1 * danger
    align = 0.38 + 0.12 * (1.0 - danger)
    a, w = _steer_to_direction(
        th, sp, dx, dy, lim, target_speed_frac=vfrac, turn_gain=tg, align_threshold=align
    )
    return np.array([a, w], dtype=np.float32)


def build_rule_actions_dict(
    obs_by_agent: dict[str, np.ndarray],
    cfg: HuntEnvConfig,
    agent_names: list[str],
) -> dict[str, np.ndarray]:
    """
    与 `HuntParallelEnv` 的 `obs` 字典、同一 `agent_names` 顺序，构造 `step` 所需动作字典。(2,) float32。
    """
    nh = cfg.agents.n_hunters
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(agent_names):
        o = np.asarray(obs_by_agent[name], dtype=np.float32)
        if i < nh:
            out[name] = rule_action_hunter(o, cfg)
        else:
            out[name] = rule_action_escaper(o, cfg)
    return out
