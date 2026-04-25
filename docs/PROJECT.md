# hunt-env / hunt_RL 项目说明

> 本文档描述「追猎式多智能体强化学习」代码库的整体设计、目录结构、公共接口、物理/观测/奖励与训练算法。  
> **与代码库对齐说明**：以仓库内源文件为准；若下文与实现不一致，以实现为准，并应回写本文档。  
> **维护要求**：对 `hunt_env/`、`hunt_rl/`、`scripts/`、`configs/` 中行为、接口、配置或算法有**实质性**修改时，请同步更新本文件对应章节。

---

## 1. 项目定位与能力边界

- **目标**：提供可配置的多智能体「猎人 vs 逃脱者」仿真，支持矩形世界、独立视野、连续控制，并配套 **PPO** 训练脚本与规则基线。
- **环境 API**：实现 **PettingZoo** `ParallelEnv`（`HuntParallelEnv`），便于标准多智能体管线对接；高吞吐训练使用 **向量化** `HuntVectorizedEnv` + `HuntBatchEngine` 的 NumPy 张量 API。
- **非目标**：不保证与任意第三方 MARL 库一键集成；不内置分布式训练。

---

## 2. 技术栈

| 类别 | 选择 |
|------|------|
| 语言 | Python ≥3.10 |
| 环境 API | [Gymnasium](https://gymnasium.farama.org/) 空间定义、[PettingZoo](https://pettingzoo.farama.org/) Parallel API |
| 配置 | YAML + [Pydantic v2](https://docs.pydantic.dev/) 校验（`HuntEnvConfig`） |
| 仿真/数值 | NumPy 向量化 |
| 渲染 | Pygame（`human` / `rgb_array`） |
| RL（可选） | PyTorch，对角高斯策略 + 价值网络，多智能体 PPO |

`pyproject.toml` 中基础依赖不含 PyTorch；训练需 `pip install -e ".[rl]"` 安装 `torch`。

---

## 3. 仓库目录结构

```
configs/           # 默认与示例 YAML（如 default.yaml）
hunt_env/          # 环境包：配置、核心仿真、包装器、规则策略、渲染
  config/          # schema.py（Pydantic 模型）, loader.py（load_config）
  core/          # 动力学、状态维度、引擎、观测、奖励、视野
  env/            # HuntParallelEnv, HuntVectorizedEnv
  policies/      # obs_layout（观测切片）, rules（与 RL 同 obs→同 shape 动作）
  render/         # Pygame 后端
hunt_rl/           # 可选：ActorCritic、MultiAgentPPOTrainer、设备选择
scripts/           # 人类试玩、规则可视化、PPO 训练入口
tests/             # pytest
docs/              # 本说明 PROJECT.md
```

**公开导出**（`hunt_env/__init__.py`）：`load_config`、`HuntParallelEnv`、`build_rule_actions_dict`；`hunt_env.policies` 另含 `decode_observation`、`DecodedObs`。

---

## 4. 配置系统

### 4.1 加载

- **函数**：`hunt_env.config.loader.load_config(path=None, *, merge=None) -> HuntEnvConfig`
- **路径**：`path` 为 `None` 时，读环境变量 `HUNT_ENV_CONFIG`；未设置则使用包内相对路径的 `configs/default.yaml`（相对 `loader.py` 上溯两级后的 `configs/default.yaml`）。
- **深合并**：`merge` 为 dict，在 YAML 解析后与根 dict **深合并**，再经 Pydantic 校验。用于实验覆盖、测试。
- **禁止**：不要写成 `load_config(overrides=...)` 传递嵌套键，会与 `**kwargs` 语义冲突（README 已说明）。

### 4.2 根配置 `HuntEnvConfig` 组成（`config/schema.py`）

| 段 | 模型 | 作用摘要 |
|----|------|----------|
| `sim` | `SimConfig` | `dt` 物理步长；`max_episode_steps`；`seed`（`None` 不固定） |
| `world` | `WorldConfig` | 轴对齐矩形：`width/height`，左下角 `origin_x/origin_y` |
| `agents` | `AgentsConfig` | `n_hunters` / `n_escapers`；`hunter_limits` / `escaper_limits`（`max_speed, max_accel, max_omega`）；`spawn`（`uniform` 或 `disk`、圆盘比例等） |
| `visibility` | `VisibilityConfig` | `view_radius`；可选扇形 `use_sector_fov` + `fov_deg`；`k_visible`（Top-K 他智能体槽位） |
| `capture` | `CaptureConfig` | `capture_radius`；`remove_captured`（捕获后从 active 移除） |
| `observation` | `ObservationConfig` | `use_ego_frame_for_others`；`include_remaining_steps`；`include_captured_count`；`include_world_bounds`（四向边界距） |
| `rewards` | `RewardsConfig` | 见第 7 节 |
| `vectorization` | `VectorizationConfig` | `num_envs`（`HuntVectorizedEnv` 默认并行数） |
| `render` | `RenderConfig` | 窗口、FPS、视野/轨迹绘制等 |
| `human_control` | `HumanControlConfig` | 人类试玩按键步长、初始控制方 |

**校验**：`use_sector_fov` 为真时必须设置 `fov_deg`。

---

## 5. 运动学与物理（`core/dynamics.py`）

- **运动模型**：二维独轮车式。**状态**：位置 `(x,y)`、朝向 `theta`、标量 `speed`。**动作**：线加速度 `a`、角速度 `omega`（每步在限制内裁剪）。
- **积分**（`step_unicycle_batch`）：`theta += omega*dt`；`speed += a*dt` 后裁剪到 `[0, max_speed]`；速度向量 `(vx,vy) = (speed*cosθ, speed*sinθ)`；位置欧拉积分。仅对 `active` 为真的实体更新。
- **边界**：轴对齐硬边界。位置夹紧到世界矩形内；在边界上**消除外向法向速度分量**（不反弹）；低速时用上一积分后的 `theta_new` 避免朝向数值跳变。

**动作空间**（每智能体 `Box(2,)`）：`[a, omega]`，上下界由猎人与逃脱者各自的 `max_accel`、`max_omega` 在 `HuntParallelEnv._build_action_spaces` 中分别设定。

---

## 6. 仿真核心 `HuntBatchEngine`（`core/engine.py`）

- **张量形状**：设 `E = num_envs`，`N = n_hunters + n_escapers`，`D = obs_dim()`。  
  - `pos`: `(E, N, 2)`；`theta`, `speed`, `prev_a`, `prev_omega`: `(E, N)`；`active`: `(E, N)`；`step_count`: `(E,)`。

### 6.1 出生（spawn）

- `_spawn_positions`：按 `spawn.mode` 在矩形内均匀或在中心圆盘内均匀；要求两两距离 ≥ `min_pairwise_separation`（`None` 时取 `max(capture_radius*1.5, 0.5)`），减少开局即被判定捕获。若多次重试仍失败，退化为无间距采样（小地图多智能体需调大场地或减员）。

### 6.2 每步流程（`step`）

1. 记录本步前逃脱方存活，用于「刚被捕获」判定。
2. 裁剪并应用动作，独轮车积分，更新 `prev_a` / `prev_omega`。
3. **捕获**：对每个仍存活的逃脱者，计算到**最近猎人**距离；若 ≤ `capture_radius` 则本步记为被捕获。若 `remove_captured`，`active` 中对应逃脱者为假；被捕获后速度置零。
4. **逐步奖励** `compute_step_rewards`、**终局奖励** `compute_terminal_rewards`（见第 7 节）。
5. 步数 +1；`all_caught` = 所有逃脱者均不存活；`timeout` = 步数 ≥ `max_episode_steps`。
6. **Gymnasium 语义**：`termination` 在「全捕获」时对所有智能体为真；`truncation` 在「超时且未全捕获」时为真。终局胜方奖励在 **本步首次** 达成全捕获时发放，避免在吸收态多步刷分（见 `just_all_caught`）。

### 6.3 `reset(seed)`

- 可重设 `numpy` 随机数生成器；位置、朝向随机；速度与历史控制清零；`active` 全真；`step_count` 清零。

**接口摘要**：

```text
HuntBatchEngine.reset(seed=None) -> obs (E, N, D)
HuntBatchEngine.step(actions) -> (obs, rew, term, trunc, info)
  actions: (E, N, 2)
  rew/term/trunc: (E, N)
info 含: just_caught, all_caught, timeout, visibility (visible_pair_mask)
```

---

## 7. 奖励（`core/rewards.py`）

在配置 `rewards` 中可调（变量名与 `default.yaml` 一致）：

- **猎人（每步）**：`hunter_step` 常数加在每猎人上；有逃脱者在本步**新被捕获**时，每名猎人得 `hunter_capture * n_new`（`n_new` 为本步新捕获人数）。
- **逃脱者（每步）**：`escaper_caught_penalty` 施加在刚被捕获者上；`escaper_step` 与 `escaper_survive` 对仍存活逃脱者每步加（后者可理解为生存 shaping）。
- **接近 shaping（猎人）**：若 `hunter_approach_shaping_scale != 0`，用「全逃脱者到最近猎人距离」的减少量对猎人做分滩 shaping（需上一步/本步距离张量，见 `engine.step` 调用处）。
- **终局一次性**：`hunter_win` 在 `just_all_caught` 时加在每名猎人上；`escaper_all_caught_penalty` 在全体逃脱者被消灭时加在每名逃脱者上（用于惩罚全灭）。

实现上 `escaper` 的 step 类奖励会乘 `escaper_alive` 掩码，避免已捕获者仍拿 step/survive 正奖励（见 `engine` 中传入的 `escaper_alive_now`）。

---

## 8. 视野与观测（`core/visibility.py`、`core/observation.py`）

### 8.1 可见性

- `visible_pair_mask(pos, theta, active, cfg)` → `(E, N, N)`：对观察者 `i`、目标 `j`（`i≠j`），当双方均 `active`、欧氏距离 ≤ `view_radius`，且（若开启扇形）相对角在 FOV 内，则 `vis[i,j]=True`。**无队伍融合**，同队与敌方规则相同。

### 8.2 Top-K 槽位

- `topk_visible_indices`：在 `vis[i,:]` 为真的目标中，按距离取最近 **K = k_visible** 个；不足填 `-1` / `inf`。
- 观测中每个槽位 7 维：`rel_x, rel_y, rel_vx, rel_vy, theta_j, mask, other_is_escaper`（`theta_j` 为世界系朝向包装到 `[-π,π]`；`mask=1` 表示该槽有目标；`other_is_escaper∈{0,1}` 标识他车身份供规则/学习共用）。

`use_ego_frame_for_others`：为真时，相对位置与相对速度在观察者机体坐标下表达；为假时相对位移与相对速度用世界系，他车朝向角仍输出为世界角（与 `state.py` 中说明一致）。

### 8.3 自身与其它与总维度

- **自身 7 维**：`x, y, vx, vy, prev_a, prev_omega, wrap_angle(theta)`。  
- **每槽 7 维** × `k_visible`。  
- 若 `include_remaining_steps`：加 1 维，为剩余步数 / `max_episode_steps`（对全体智能体相同）。  
- 若 `include_captured_count`：加 1 维，为已捕获逃脱者比例 `(ne - sum(escaper_active)) / ne`。  
- 若 `include_world_bounds`（默认建议开启）：加 4 维，为到西/东/下/上边界距离（世界系，与自车位置一致）。

总维度由 `hunt_env.core.state.total_obs_dim(cfg)` 与 `assert_obs_dim` 一致。

**函数** `build_observations_batch(...)`：输入批状态，输出 `obs (E, N, D)`。`policies/obs_layout.decode_observation` 将单智能体展平向量按当前 `HuntEnvConfig` 切回各段，供规则策略与调试。

---

## 9. 环境包装器

### 9.1 `HuntParallelEnv`（`env/hunt_parallel.py`）

- 继承 `pettingzoo.ParallelEnv`；内部固定 `HuntBatchEngine(num_envs=1)`。
- `possible_agents` 顺序：`hunter_0..` 后接 `escaper_0..`（`hunt_env.core.state.agent_names`）。
- `reset` → 观测/信息 dict；`step(actions: dict[str, np.ndarray])` 同上；episode 任一方终止时 `agents` 置空（PettingZoo 约定）。
- `engine` 属性暴露只读 `HuntBatchEngine`，供调试、可视化、规则基线对照（**规则动作应优先仅用 `obs` 字典**，与 RL 公平对齐）。
- `render()`：懒加载 `PygameHuntRenderer`；`render_mode` 为 `human` 或 `rgb_array`。

### 9.2 `HuntVectorizedEnv`（`env/vectorized.py`）

- 字段 `cfg`, `num_envs`，`engine: HuntBatchEngine`；`from_yaml(path, num_envs=None)` 会读 `load_config` 且 `num_envs` 默认取 `vectorization.num_envs`。
- `reset` → `(E, N, D)`；`step(actions)` 要求 `actions` 形状 `(E, N, 2)`。

此包装为 **RL 训练**（`hunt_rl.trainer`）所使用。

---

## 10. 规则基线（`policies/rules.py` + `policies/obs_layout.py`）

- **与 RL 相同接口（预训练/公平性）**：`rule_action_hunter(obs, cfg) -> (2,) float32`、`rule_action_escaper(obs, cfg) -> (2,) float32`，输入为**与 `HuntParallelEnv.reset/step` 返回的各智能体观测同形状、同语义的** `ndarray`；`build_rule_actions_dict(obs_by_agent, cfg, agent_names)` 与 `env.step` 的 `dict` 一致。不读取 `HuntBatchEngine`。
- **信息来源**：可见性、相对几何、阵营、边界距离均**只**能来自展平观测（Top-K 槽位含 `other_is_escaper`；`include_world_bounds` 为真时含四向边界距）。引擎内的可见性由 `build_observations_batch` 与规则「间接」一致。
- **猎人**：在 `mask` 有效且 `other_is_escaper=1` 的槽中，对相对位移范数**最小**的槽所给世界系/机体（由 `use_ego_frame_for_others` 决定）目标方向做 `_steer_to_direction`；无可用槽则 `_hunter_search_no_target`（仅自车 7 维可推断的量）。
- **逃脱者**：在 `mask` 有效且 `other_is_escaper=0` 的槽中选最近猎人，以观测相对量构造逃离+贴边+向心+闪避；无猎人槽时 `_escaper_center_wall_steer` 仅用自车 7 维 + `wall_dist` 四元组。
- **动作上界**与 RL 一样由 `cfg` 运动学 + 环境 `action_space` 保证；`hunt_rl.trainer` 在 `escaper_mode="rule"` 下对每步的 `obs_buf[t,ei,ni,:]` 调用 `rule_action_escaper`（与 PPO 分支**同一 `obs` 张量**）。

---

## 11. 强化学习模块（`hunt_rl/`）

### 11.1 网络 `ActorCritic`（`actor_critic.py`）

- 共享 MLP 骨干；**策略头**输出动作均值，**可学习** `log_std`（各维独立，对角高斯）；**价值头**标量 V(s)。
- `act`：采样或确定性取均值；**执行动作为 `clip` 到环境 `low/high` 后送入环境**；`log_prob` 在未裁剪的高斯变量上计算（PPO 常见近似）。
- `evaluate`：给定存储的**未裁剪**动作用于 PPO 的比率与熵；接口仍传入 bound 以兼容（实现与 `act` 一致）。

`action_bounds_from_cfg(cfg, role)`：`role` 为 `"hunter"` 或 `"escaper"`，返回与 YAML 中 `max_accel`/`max_omega` 一致的对称边界。

### 11.2 多智能体 PPO `MultiAgentPPOTrainer`（`trainer.py`）

- **角色分组**：**所有猎人共享**一个 `ActorCritic`（`policies["hunter"]`）；**所有逃脱者共享**一个 `ActorCritic`（`policies["escaper"]`），除非 `escaper_mode="rule"`（此时不建逃脱者网络，逃脱者动作用 `rule_action_escaper` 生成，logp/价值对该角色为 0）。
- **采样子循环**：对每个时间步、每个环境、每个智能体，按索引选策略与动作上下界，收集 `obs, raw_action, logp, value, reward, done`。
- **GAE**（`compute_gae`）：`rewards` `(T, E)`，按环境独立 bootstrap；`dones` 为标量或从 `term|trunc` 对全体 env 的掩码；实现见该文件。
- **Reset 行为**：`num_envs==1` 时，任一步 `done` 则立即 `reset` 下一步观测；`num_envs>1` 时仅当**本步所有 env 都 done** 才整批 `reset`（与注释一致）。
- **PPO 更新**：对每个要训练的智能体索引分别调用 `ppo_update_agent`（独立优化器与损失聚合）；`escaper_mode=rule` 时只更新 `0..nh-1` 猎人（即逃脱者不更新）。
- **保存/加载**：`save` 写入 `cfg.model_dump()`、`state_dicts`、**`escaper_mode`**，便于 `load` 时恢复；`load` 若 checkpoint 含 `escaper_mode` 会注入构造函数。

`PPOConfig`：学习率、γ、GAE λ、clip、熵/价值系数、epoch 与 minibatch 数、是否标准化 advantage 等（见类定义默认值）。

### 11.3 设备（`device.py`）

- `get_train_device(prefer_cuda)`：无 CUDA 时回退 CPU；可用 `CUDA_VISIBLE_DEVICES` 控制可见 GPU。

### 11.4 与环境的假设

- 观测维相同、动作维为 2；猎人与逃脱者**边界不同**时通过 `action_bounds_from_cfg` 分角色裁剪。

---

## 12. 脚本入口

| 脚本 | 作用 |
|------|------|
| `scripts/human_play.py` | 键盘控制**一名**智能体（W/S 线加减速，A/D 角速度，Q 切换智能体），其余发零动作；`render_mode=human` |
| `scripts/viz_rule_baseline.py` | 全用规则策略跑若干局并可视化 |
| `scripts/train_ppo.py` | 双端（或仅剩一侧）PPO 训练；`--total-steps`、`--rollout-len`、`--num-envs`、`--device` 等 |
| `scripts/train_hunter_ppo_rule_escaper.py` | 仅训猎人、逃脱者走规则；适合短时间验证管线 |

**人类试玩**与真实训练差异：试玩中未控制者动作为零；训练使用完整策略或规则。

---

## 13. 测试（`tests/`）

- 覆盖：配置、动力学、引擎、视野、观测维度、奖励、环境 API、PPO 冒烟、猎人 PPO+规则逃脱者等。运行：`pytest tests/ -q`（从项目根且包已安装）。

---

## 14. 渲染（`render/pygame_backend.py`）

- `PygameHuntRenderer`：世界 AABB 映射到屏幕；绘制边界、可选轨迹（猎人/逃脱者分色）、每名智能体视野圈（`draw_view_radius`）、智能体位姿等。`HuntParallelEnv` 的 `reset` 时若有 renderer 会 `clear_trajectories`。

---

## 15. 术语与团队索引约定

- 智能体展平顺序：先猎人 `0..n_hunters-1`，后逃脱者 `0..n_escapers-1`；名称字符串 `hunter_i` / `escaper_j`。
- `engine` 中前 `nh` 个索引为猎人，其后为逃脱者。

---

## 16. 文档维护清单（供修改代码时对照）

- 新增/删除配置键：更新 `config/schema.py`、`configs/*.yaml` 与 **本文件第 4 节、相关算法节**。
- 改观测/动作形状或语义：第 5、8 节 + `HuntParallelEnv` / `HuntVectorizedEnv` 说明。
- 改奖励或终止条件：第 6、7 节。
- 改 PPO/网络/采样子过程：第 11 节 + `PPOConfig` 默认值。
- 新脚本：第 12 节表格。

**文档版本**：与仓库主分支实现同步维护；大改时可在本行更新日期。  
*末次全面对齐（撰写时）：2026-04-25。*
