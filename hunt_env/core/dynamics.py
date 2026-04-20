"""2D 单轨模型：线加速度与角速度控制；矩形硬边界（钳制位置 + 消除外向法向速度）。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig, WorldConfig


def clip_actions(
    a: np.ndarray,
    omega: np.ndarray,
    max_accel: np.ndarray,
    max_omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将动作裁剪到各智能体限制内。形状广播兼容 (E,N) 与 (N,)。"""
    a_c = np.clip(a, -max_accel, max_accel)
    w_c = np.clip(omega, -max_omega, max_omega)
    return a_c, w_c


def _world_bounds(cfg: HuntEnvConfig) -> tuple[float, float, float, float]:
    w = cfg.world
    return w.origin_x, w.origin_y, w.origin_x + w.width, w.origin_y + w.height


def step_unicycle_batch(
    pos: np.ndarray,
    theta: np.ndarray,
    speed: np.ndarray,
    active: np.ndarray,
    a_cmd: np.ndarray,
    omega_cmd: np.ndarray,
    max_speed: np.ndarray,
    dt: float,
    world: WorldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对批量环境积分一步（仅更新仍 active 的实体）。

    pos: (E, N, 2), theta: (E, N), speed: (E, N), active: (E, N)
    a_cmd, omega_cmd: (E, N), max_speed: (N,) 或 (E, N)
    """
    ox, oy, x_max, y_max = (
        world.origin_x,
        world.origin_y,
        world.origin_x + world.width,
        world.origin_y + world.height,
    )
    eps = 1e-8

    vx = speed * np.cos(theta)
    vy = speed * np.sin(theta)

    # 仅 active 更新
    mask = active[..., None]
    theta_new = np.where(active, theta + omega_cmd * dt, theta)
    speed_new = np.where(active, speed + a_cmd * dt, speed)
    speed_new = np.where(active, np.clip(speed_new, 0.0, max_speed), speed_new)

    vx = speed_new * np.cos(theta_new)
    vy = speed_new * np.sin(theta_new)

    px = pos[..., 0] + vx * dt * active
    py = pos[..., 1] + vy * dt * active

    px = np.clip(px, ox, x_max)
    py = np.clip(py, oy, y_max)

    # 边界处消除外向速度分量（不反弹）
    vx = np.where(px <= ox + eps, np.maximum(vx, 0.0), vx)
    vx = np.where(px >= x_max - eps, np.minimum(vx, 0.0), vx)
    vy = np.where(py <= oy + eps, np.maximum(vy, 0.0), vy)
    vy = np.where(py >= y_max - eps, np.minimum(vy, 0.0), vy)

    speed_out = np.hypot(vx, vy)
    theta_out = np.arctan2(vy, vx)
    # 低速保持原朝向，避免数值跳变
    slow = speed_out < 1e-3
    theta_out = np.where(slow & active, theta_new, theta_out)
    speed_out = np.where(active, speed_out, speed)

    pos_new = np.stack([px, py], axis=-1)
    pos_new = np.where(active[..., None], pos_new, pos)
    theta_out = np.where(active, theta_out, theta)
    return pos_new, theta_out, speed_out
