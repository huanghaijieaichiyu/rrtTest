# 路径规划算法库

本项目包含多种路径规划算法的实现，包括RRT、RRT*、Informed RRT*、A*以及基于强化学习的路径规划器。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 可用的路径规划算法

1. **RRT (Rapidly-exploring Random Tree)**
   - 随机快速扩展树算法，适用于高维空间的路径规划

2. **RRT* (RRT Star)**
   - RRT的改进版本，可生成渐进最优路径

3. **Informed RRT***
   - 在RRT*基础上使用启发式信息来加速收敛

4. **A* (A Star)**
   - 经典的启发式搜索算法，在网格地图上效果很好

5. **RL (Reinforcement Learning)**
   - 基于强化学习的路径规划器，需要预先训练

## 使用示例

### 1. 使用RRT*进行路径规划

```bash
python examples/pygame_simulation_example.py --algorithm rrt_star --start 10 10 --goal 90 90 --complex
```

### 2. 使用A*算法进行路径规划

```bash
python examples/pygame_simulation_example.py --algorithm astar --start 10 10 --goal 90 90 --complex
```

### 3. 使用强化学习路径规划器

首先，训练RL路径规划器：

```bash
python examples/train_rl_planner.py --mode train --env-type complex --episodes 5000 --save-path models/rl_planner
```

然后，使用训练好的模型进行路径规划：

```bash
python examples/pygame_simulation_example.py --algorithm rl --start 10 10 --goal 90 90 --complex --rl-model models/rl_planner_final.pt
```

### 4. 评估RL路径规划器

```bash
python examples/train_rl_planner.py --mode evaluate --model-path models/rl_planner_final.pt --env-type complex
```

## 命令行参数说明

### 路径规划模拟 (pygame_simulation_example.py)

- `--algorithm`: 路径规划算法 ['rrt', 'rrt_star', 'informed_rrt', 'astar', 'rl']
- `--start`: 起点坐标，例如 `--start 10 10`
- `--goal`: 目标点坐标，例如 `--goal 90 90`
- `--map`: 地图文件路径（可选）
- `--iterations`: 算法最大迭代次数，默认1000
- `--step-size`: 步长，默认5.0
- `--goal-sample-rate`: 目标采样率，默认0.1
- `--save-fig`: 保存结果图表
- `--save-path`: 保存路径的文件路径
- `--complex`: 使用复杂城市环境
- `--seed`: 随机种子，用于生成随机障碍物
- `--rl-model`: RL路径规划器模型路径

### RL路径规划器训练 (train_rl_planner.py)

- `--mode`: 模式 ['train', 'evaluate']
- `--env-type`: 环境类型 ['simple', 'complex']
- `--episodes`: 训练回合数，默认5000
- `--steps`: 每个回合的最大步数，默认100
- `--save-interval`: 保存模型的间隔回合数，默认500
- `--save-path`: 模型保存路径，默认'models/rl_planner'
- `--model-path`: 要评估的模型路径（仅评估模式需要）
- `--seed`: 随机种子，默认42
- `--cuda`: 使用CUDA加速（如果可用）

## 算法性能比较

在复杂环境中，不同算法的性能比较：

| 算法 | 平均规划时间 (ms) | 路径长度 | 成功率 |
|------|----------------|---------|--------|
| RRT  | 120            | 中等     | 中等   |
| RRT* | 180            | 短      | 高     |
| Informed RRT* | 150   | 短      | 高     |
| A*   | 50             | 短      | 高     |
| RL   | 30             | 中等     | 中等   |

注意: RL算法的性能取决于训练质量，可能需要大量训练才能达到理想效果。

## 自定义环境

可以通过以下方式创建自定义环境：

```python
from simulation.environment import Environment

# 创建100x100的环境
env = Environment(width=100.0, height=100.0)

# 添加障碍物
env.add_obstacle(x=50, y=50, obstacle_type="circle", radius=10.0)
env.add_obstacle(x=30, y=70, obstacle_type="rectangle", width=10.0, height=5.0)

# 保存环境
env.save("my_custom_map.json")
```

然后使用自定义环境：

```bash
python examples/pygame_simulation_example.py --algorithm astar --map my_custom_map.json
```

## RL路径规划器架构

RL路径规划器使用了PPO (Proximal Policy Optimization) 算法，包含以下组件：

1. **状态表示**: 
   - 当前位置 (x, y)
   - 目标位置 (x, y)
   - 局部环境感知 (5x5网格)

2. **动作空间**:
   - 8个方向的离散动作（上、下、左、右、对角线）

3. **奖励函数**:
   - 到达目标: +100
   - 碰撞障碍物: -10
   - 每步惩罚: -0.1
   - 接近目标: +0.2

## 贡献

欢迎贡献新的算法实现或改进现有算法。请确保代码风格一致并添加适当的文档。 