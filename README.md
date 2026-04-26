# hunt-env

追猎式多智能体强化学习环境（Python）：矩形硬边界、独立视野、配置文件驱动。

**详细说明**（结构、接口、物理与观测、奖励、PPO、规则基线与 `decode_observation`）：见 [docs/PROJECT.md](docs/PROJECT.md)。修改实现时请同步更新该文档（仓库内 Cursor 规则已要求）。

## 环境（推荐 Conda）

新建并安装（需在**项目根目录**执行，`environment.yml` 与 `pyproject.toml` 同级）：

```powershell
cd <你的项目根目录>
conda env create -f environment.yml
conda activate hunt-env
```

若尚未用 `environment.yml` 创建过环境，也可手动：

```powershell
conda create -n hunt-env python=3.11 pip -y
conda activate hunt-env
cd <你的项目根目录>
pip install -e ".[dev]"
```

需要训练（PPO）时再安装 PyTorch 可选依赖：`pip install -e ".[rl]"`。

程序化覆盖 YAML 片段请使用 `load_config(merge={"sim": {"max_episode_steps": 500}})`，勿写成 `load_config(overrides=...)`（会与 `**kwargs` 语义冲突）。

## 人类试玩

```powershell
python scripts/human_play.py --config configs/default.yaml
```

## 规则基线（预训练/示范/对手）

规则策略与 RL **使用同一套观测向量**与 **`(2,)` 线加速度+角速度动作**，不读 `HuntBatchEngine` 真值。可见目标仅来自 Top-K 槽位（含 `other_is_escaper`）；逃脱者无可见猎人时依赖观测中的**四向边界距**（`observation.include_world_bounds`，默认真）。

- API：`hunt_env.policies` 中 `rule_action_hunter(obs, cfg)`、`rule_action_escaper(obs, cfg)`、`build_rule_actions_dict(obs_by_agent, cfg, agent_names)`；布局切片见 `decode_observation`。
- 可视化整局：使用当前步 `env.reset`/`step` 返回的 `obs` 字典，勿传引擎状态。

```powershell
python scripts/viz_rule_baseline.py --config configs/default.yaml --max-episodes 3
```

仅训猎人、逃脱者走上述规则（与 PPO 采集中**同一** `obs_buf`）的短训验证：

```powershell
pip install -e ".[rl]"
python scripts/train_hunter_ppo_rule_escaper.py --time-sec 600
```

## 测试

```powershell
pytest tests/ -q
```
