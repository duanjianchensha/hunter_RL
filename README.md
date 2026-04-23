# hunt-env

追猎式多智能体强化学习环境（Python）：矩形硬边界、独立视野、配置文件驱动。

## 环境（推荐 Conda）

新建并安装（需在项目根目录执行，`environment.yml` 与 `pyproject.toml` 同级）：

```powershell
cd d:\work\test5
conda env create -f environment.yml
conda activate hunt-env
```

若尚未用 `environment.yml` 创建过环境，也可手动：

```powershell
conda create -n hunt-env python=3.11 pip -y
conda activate hunt-env
cd d:\work\test5
pip install -e ".[dev]"
```

程序化覆盖 YAML 片段请使用 `load_config(merge={"sim": {"max_episode_steps": 500}})`，勿写成 `load_config(overrides=...)`（会与 `**kwargs` 语义冲突）。

## 人类试玩

```powershell
python scripts/human_play.py --config configs/default.yaml
```

## 规则基线（预训练/示范）

猎人追击最近逃脱者、逃脱者逃离最近猎人（使用仿真器全局位置，非神经网络观测）：

```powershell
python scripts/viz_rule_baseline.py --config configs/default.yaml --max-episodes 3
```

## 测试

```powershell
pytest tests/ -q
```
