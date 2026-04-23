"""
基于规则的策略（教师/基线）：使用仿真器内部全局状态，不依赖神经网络观测。

适用于预训练数据收集、行为示范或与 RL 策略对比；与「仅用局部观测」的智能体不等价。
"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import AgentLimitsConfig, HuntEnvConfig
from hunt_env.core.engine import HuntBatchEngine
from hunt_env.core.state import wrap_angle


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
    """
    将车头转向世界系方向 (dir_x, dir_y)，并沿该方向加速到目标速率。
    """
    n = np.hypot(dir_x, dir_y)
    if n < 1e-9:
        return 0.0, 1.2  # 无方向时原地慢转

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
        # 大角度差时以较小线加速度配合转向，避免原地甩尾过快
        a = float(np.clip(0.25 * max_a * max(np.cos(err), -0.2), -max_a, max_a))

    return a, omega


def rule_action_hunter(
    env: HuntBatchEngine,
    env_index: int,
    hunter_index: int,
    cfg: HuntEnvConfig,
) -> tuple[float, float]:
    """
    猎人：追击距离最近的、仍存活的逃脱者（全局最近邻，非视野约束）。
    """
    nh = cfg.agents.n_hunters
    if hunter_index < 0 or hunter_index >= nh:
        return 0.0, 0.0

    pos = env.pos[env_index]
    theta = env.theta[env_index]
    speed = env.speed[env_index]
    active = env.active[env_index]

    if not active[hunter_index]:
        return 0.0, 0.0

    best_j = -1
    best_d = np.inf
    for j in range(nh, env.n_agents):
        if not active[j]:
            continue
        d = float(np.linalg.norm(pos[hunter_index] - pos[j]))
        if d < best_d:
            best_d = d
            best_j = j

    if best_j < 0:
        return 0.0, 0.0

    tx = pos[best_j, 0] - pos[hunter_index, 0]
    ty = pos[best_j, 1] - pos[hunter_index, 1]
    return _steer_to_direction(
        float(theta[hunter_index]),
        float(speed[hunter_index]),
        tx,
        ty,
        cfg.agents.hunter_limits,
    )


def rule_action_escaper(
    env: HuntBatchEngine,
    env_index: int,
    escaper_index: int,
    cfg: HuntEnvConfig,
) -> tuple[float, float]:
    """
    逃脱者：以「远离最近猎人」为主，并融合：
    - 近边界时沿墙向场内推开，减少贴边被堵；
    - 威胁距离较近时加入垂直于追逃线的侧向分量，做简单闪避；
    - 威胁越近，转向与目标速度略提高。
    """
    nh = cfg.agents.n_hunters
    if escaper_index < nh or escaper_index >= env.n_agents:
        return 0.0, 0.0

    pos = env.pos[env_index]
    theta = env.theta[env_index]
    speed = env.speed[env_index]
    active = env.active[env_index]

    if not active[escaper_index]:
        return 0.0, 0.0

    best_h = -1
    best_d = np.inf
    for h in range(nh):
        if not active[h]:
            continue
        d = float(np.linalg.norm(pos[escaper_index] - pos[h]))
        if d < best_d:
            best_d = d
            best_h = h

    if best_h < 0:
        return 0.0, 0.0

    ex = float(pos[escaper_index, 0])
    ey = float(pos[escaper_index, 1])
    hx = float(pos[best_h, 0])
    hy = float(pos[best_h, 1])

    # 单位「逃离」方向：猎人 -> 逃脱者
    tx = ex - hx
    ty = ey - hy
    tn = float(np.hypot(tx, ty))
    if tn < 1e-9:
        return 0.0, 1.8
    fx, fy = tx / tn, ty / tn

    wc = cfg.world
    ox, oy, wdim, hdim = float(wc.origin_x), float(wc.origin_y), float(wc.width), float(wc.height)
    margin = 1.0
    rx, ry = 0.0, 0.0
    if ex < ox + margin:
        rx += 1.0
    if ex > ox + wdim - margin:
        rx -= 1.0
    if ey < oy + margin:
        ry += 1.0
    if ey > oy + hdim - margin:
        ry -= 1.0
    rlen = float(np.hypot(rx, ry))
    if rlen > 1e-9:
        rx, ry = rx / rlen, ry / rlen

    d_edge = min(ex - ox, ox + wdim - ex, ey - oy, oy + hdim - ey)
    wall_w = float(np.clip((1.7 - d_edge) / 1.7, 0.0, 1.0))

    cap = float(cfg.capture.capture_radius)
    # 距离越近 danger 越大（上限 1）
    danger = float(np.clip((cap * 5.0 - best_d) / (cap * 5.0), 0.0, 1.0))

    # 侧向单位向量（在 flee 左侧为 +90°）
    px, py = -fy, fx
    cx = (ox + wdim * 0.5) - ex
    cy = (oy + hdim * 0.5) - ey
    cross_z = px * cy - py * cx
    sign = 1.0 if cross_z >= 0.0 else -1.0

    # 混合方向后归一化
    mx = (1.0 - 0.42 * wall_w) * fx + 0.42 * wall_w * rx + (0.55 * danger) * sign * px
    my = (1.0 - 0.42 * wall_w) * fy + 0.42 * wall_w * ry + (0.55 * danger) * sign * py
    # 弱向场地中心偏置，减少长期卡角（1v1 时尤其明显）
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
    return _steer_to_direction(
        float(theta[escaper_index]),
        float(speed[escaper_index]),
        dx,
        dy,
        cfg.agents.escaper_limits,
        target_speed_frac=vfrac,
        turn_gain=tg,
        align_threshold=align,
    )


def build_rule_actions_dict(
    env: HuntBatchEngine,
    env_index: int,
    cfg: HuntEnvConfig,
    agent_names: list[str],
) -> dict[str, np.ndarray]:
    """为 ParallelEnv.step 构造整局规则动作字典。"""
    nh = cfg.agents.n_hunters
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(agent_names):
        if i < nh:
            a, w = rule_action_hunter(env, env_index, i, cfg)
        else:
            a, w = rule_action_escaper(env, env_index, i, cfg)
        out[name] = np.array([a, w], dtype=np.float32)
    return out
