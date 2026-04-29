"""环境配置 Pydantic 模型（单一事实来源由 YAML 提供）。"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator


class SimConfig(BaseModel):
    """仿真时间与步数。"""

    dt: float = Field(gt=0, description="物理步长（秒）")
    max_episode_steps: int = Field(ge=1, description="单局最大步数")
    seed: int | None = Field(default=None, description="None 表示非固定种子")


class WorldConfig(BaseModel):
    """矩形世界（轴对齐），硬边界不可穿越、不反弹。"""

    width: float = Field(gt=0, description="x 方向宽度")
    height: float = Field(gt=0, description="y 方向高度")
    # 左下角为 (origin_x, origin_y)，向右 x 增大，向上 y 增大
    origin_x: float = 0.0
    origin_y: float = 0.0


class AgentLimitsConfig(BaseModel):
    """单类智能体的运动学限制。"""

    max_speed: float = Field(ge=0)
    max_accel: float = Field(ge=0)
    max_omega: float = Field(ge=0, description="角速度上限（弧度/秒）")


class SpawnConfig(BaseModel):
    """重置时初始位置采样。"""

    mode: Literal["uniform", "disk"] = "uniform"
    # uniform：在矩形内均匀；disk：在世界中心附近圆盘内均匀
    disk_radius_frac: float = Field(
        default=0.35,
        ge=0.0,
        le=0.5,
        description="mode=disk 时半径 = min(w,h)*frac",
    )
    # None 时由引擎按 capture_radius 自动取安全间距，避免首帧即被判定捕获
    min_pairwise_separation: float | None = Field(
        default=None,
        ge=0.0,
        description="智能体两两之间的最小出生距离；None 表示自动",
    )


class AgentsConfig(BaseModel):
    """智能体数量与限制。"""

    n_hunters: int = Field(ge=1)
    n_escapers: int = Field(ge=1)
    hunter_limits: AgentLimitsConfig
    escaper_limits: AgentLimitsConfig
    spawn: SpawnConfig = Field(default_factory=SpawnConfig)


class VisibilityConfig(BaseModel):
    """逐智能体独立视野，无队伍融合。"""

    view_radius: float = Field(
        gt=0,
        description="默认视野半径；未单独指定 hunter_view_radius / escaper_view_radius 时双方均用此值",
    )
    hunter_view_radius: float | None = Field(
        default=None,
        description="猎人索引所用距离圆半径；None 则用 view_radius",
    )
    escaper_view_radius: float | None = Field(
        default=None,
        description="逃脱者索引所用距离圆半径；None 则用 view_radius",
    )
    use_sector_fov: bool = False
    fov_deg: float | None = Field(
        default=None,
        ge=0.0,
        lt=360.0,
        description="扇形半角（度），从朝向左右各一半；全圆为 360 或不用扇形",
    )
    k_visible: int = Field(ge=1, description="Top-K 其他智能体槽位")

    @model_validator(mode="after")
    def _role_view_radii_positive(self) -> VisibilityConfig:
        for name, v in (
            ("hunter_view_radius", self.hunter_view_radius),
            ("escaper_view_radius", self.escaper_view_radius),
        ):
            if v is not None and v <= 0:
                raise ValueError(f"{name} 必须 > 0")
        return self

    def radii_per_observer(self, n_hunters: int, n_agents: int) -> np.ndarray:
        """每名观察者 i 的距离阈值 (N,)，与 visible_pair_mask 一致。"""
        vh = float(self.hunter_view_radius) if self.hunter_view_radius is not None else float(self.view_radius)
        ve = float(self.escaper_view_radius) if self.escaper_view_radius is not None else float(self.view_radius)
        r = np.empty(n_agents, dtype=np.float64)
        r[:n_hunters] = vh
        r[n_hunters:] = ve
        return r


class CaptureConfig(BaseModel):
    """捕获判定。"""

    capture_radius: float = Field(gt=0)
    remove_captured: bool = True


class ObservationConfig(BaseModel):
    """观测编码选项。"""

    use_ego_frame_for_others: bool = Field(
        default=False,
        description="True：他车相对量用机体坐标；False：相对位移仍用世界系，角度仍为世界角",
    )
    include_remaining_steps: bool = True
    include_captured_count: bool = True
    include_world_bounds: bool = Field(
        default=True,
        description="追加 4 维：到西/东/下/上边界距离（世界系），与规则基线及边界启发共用",
    )


class RewardsConfig(BaseModel):
    """可配置奖励权重。"""

    hunter_step: float = 0.0
    hunter_capture: float = 1.0
    hunter_win: float = 1.0
    hunter_approach_shaping_scale: float = 0.0
    # 与猎人接近 shaping 对称：本步「到最近猎人距离」相对上步增加量为正（拉远有奖）
    escaper_flee_shaping_scale: float = 0.0
    escaper_step: float = 0.0
    escaper_survive: float = 0.0
    escaper_caught_penalty: float = -1.0
    escaper_all_caught_penalty: float = 0.0


class VectorizationConfig(BaseModel):
    """训练入口向量化（环境类内部仍单配置）。"""

    num_envs: int = Field(default=1, ge=1)


class RenderConfig(BaseModel):
    """Pygame 渲染。"""

    window_width: int = Field(default=800, ge=64)
    window_height: int = Field(default=800, ge=64)
    fps: int = Field(default=60, ge=1)
    draw_view_radius: bool = True
    draw_sector: bool = False
    draw_trajectories: bool = Field(default=True, description="是否绘制每名智能体的运动轨迹折线")
    trajectory_max_points: int = Field(
        default=2000,
        ge=20,
        description="每名智能体保留的轨迹点数上限（超过则丢弃最旧点）",
    )


class HumanControlConfig(BaseModel):
    """键盘试玩。"""

    accel_step: float = Field(default=1.0, gt=0)
    omega_step: float = Field(default=1.0, gt=0)
    initial_control_agent: Literal["hunter", "escaper"] = "hunter"
    initial_agent_index: int = Field(default=0, ge=0)


class HuntEnvConfig(BaseModel):
    """根配置。"""

    sim: SimConfig
    world: WorldConfig
    agents: AgentsConfig
    visibility: VisibilityConfig
    capture: CaptureConfig
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    rewards: RewardsConfig = Field(default_factory=RewardsConfig)
    vectorization: VectorizationConfig = Field(default_factory=VectorizationConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    human_control: HumanControlConfig = Field(default_factory=HumanControlConfig)

    @model_validator(mode="after")
    def _check_sector(self) -> HuntEnvConfig:
        if self.visibility.use_sector_fov and self.visibility.fov_deg is None:
            raise ValueError("visibility.use_sector_fov 为 True 时必须设置 visibility.fov_deg")
        return self
