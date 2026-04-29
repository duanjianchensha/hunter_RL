"""Pygame 可视化：世界坐标到像素映射、视野圈、智能体朝向。"""

from __future__ import annotations

import numpy as np

from hunt_env.config.schema import HuntEnvConfig
from hunt_env.core.engine import HuntBatchEngine


class PygameHuntRenderer:
    def __init__(self, cfg: HuntEnvConfig):
        import pygame

        self._pygame = pygame
        pygame.init()
        self.cfg = cfg
        r = cfg.render
        self._screen = pygame.display.set_mode((r.window_width, r.window_height))
        pygame.display.set_caption("hunt-env")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 14)
        ox = cfg.world.origin_x
        oy = cfg.world.origin_y
        w = cfg.world.width
        h = cfg.world.height
        margin = 24
        self._world_rect = (ox, oy, w, h)
        self._margin = margin
        self._sx = (r.window_width - 2 * margin) / w
        self._sy = (r.window_height - 2 * margin) / h
        nh = cfg.agents.n_hunters
        self._nh = nh
        self._trails: list[list[tuple[float, float]]] = [
            [] for _ in range(cfg.agents.n_hunters + cfg.agents.n_escapers)
        ]

    def clear_trajectories(self) -> None:
        """新一局开始时清空轨迹缓存。"""
        self._trails = [[] for _ in range(len(self._trails))]

    def close(self) -> None:
        self._pygame.quit()

    def world_to_px(self, xy: np.ndarray) -> tuple[int, int]:
        ox, oy, w, h = self._world_rect
        m = self._margin
        x, y = float(xy[0]), float(xy[1])
        px = m + (x - ox) * self._sx
        py = self._screen.get_height() - (m + (y - oy) * self._sy)
        return int(px), int(py)

    def render(self, engine: HuntBatchEngine, env_index: int = 0, return_rgb: bool = False) -> np.ndarray | None:
        pygame = self._pygame
        cfg = self.cfg
        nh = cfg.agents.n_hunters
        screen = self._screen
        screen.fill((20, 22, 30))

        ox, oy, w, h = self._world_rect
        # 边界矩形
        p0 = self.world_to_px(np.array([ox, oy]))
        p1 = self.world_to_px(np.array([ox + w, oy + h]))
        rect = pygame.Rect(min(p0[0], p1[0]), min(p0[1], p1[1]), abs(p1[0] - p0[0]), abs(p1[1] - p0[0]))
        pygame.draw.rect(screen, (60, 64, 80), rect, width=2)

        pos = engine.pos[env_index]
        theta = engine.theta[env_index]
        active = engine.active[env_index]
        n_agents = engine.n_agents
        view_radii = cfg.visibility.radii_per_observer(nh, n_agents)

        # 轨迹长度与引擎智能体数对齐（防御）
        if len(self._trails) != n_agents:
            self._trails = [[] for _ in range(n_agents)]

        # 运动轨迹（先画在底层）
        if cfg.render.draw_trajectories:
            max_pts = cfg.render.trajectory_max_points
            for i in range(n_agents):
                if active[i]:
                    self._trails[i].append((float(pos[i, 0]), float(pos[i, 1])))
                    if len(self._trails[i]) > max_pts:
                        self._trails[i] = self._trails[i][-max_pts:]
            hunter_trail_rgb = (140, 65, 75)
            escaper_trail_rgb = (55, 120, 85)
            for i in range(n_agents):
                pts = self._trails[i]
                if len(pts) < 2:
                    continue
                pixel_pts = [self.world_to_px(np.array(p)) for p in pts]
                col = hunter_trail_rgb if i < self._nh else escaper_trail_rgb
                pygame.draw.lines(screen, col, False, pixel_pts, width=2)

        # 视野（每名智能体）
        if cfg.render.draw_view_radius:
            for i in range(engine.n_agents):
                if not active[i]:
                    continue
                c = self.world_to_px(pos[i])
                r_px = int(float(view_radii[i]) * min(self._sx, self._sy))
                pygame.draw.circle(screen, (40, 55, 70), c, r_px, width=1)

        # 智能体
        for i in range(engine.n_agents):
            if not active[i]:
                continue
            c = self.world_to_px(pos[i])
            color = (220, 90, 90) if i < nh else (90, 200, 140)
            pygame.draw.circle(screen, color, c, 6)
            # 朝向
            tip = np.array(
                [
                    pos[i, 0] + 0.6 * np.cos(theta[i]),
                    pos[i, 1] + 0.6 * np.sin(theta[i]),
                ]
            )
            tpx = self.world_to_px(tip)
            pygame.draw.line(screen, (230, 230, 240), c, tpx, width=2)

        # HUD
        txt = f"step={int(engine.step_count[env_index])}/{cfg.sim.max_episode_steps}"
        surf = self._font.render(txt, True, (200, 200, 210))
        screen.blit(surf, (8, 8))

        pygame.display.flip()
        self._clock.tick(cfg.render.fps)

        if return_rgb:
            arr = pygame.surfarray.array3d(screen)
            return np.transpose(arr, (1, 0, 2))
        return None
