# RRT 路径规划与仿真系统

这是一个基于 RRT（Rapidly-exploring Random Tree）算法的路径规划系统，包含完整的路径规划和可视化仿真功能。系统使用 Pygame 实现了一个交互式的仿真环境，支持实时路径跟踪和车辆动力学模拟。

## 主要特点

- 基于 RRT* 算法的路径规划
- 实时交互式仿真环境
- 多种路径跟踪控制算法（PID、MPC、LQR）
- 完整的车辆动力学模型
- 碰撞检测和避障功能
- 支持动态路径重规划

## 许可证

本项目采用 Apache License 2.0 许可证。详细信息请参见 [LICENSE](LICENSE) 文件。

```
Copyright 2024 RRT Path Planning and Simulation System

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 安装依赖

```bash
pip install numpy pygame matplotlib shapely pyyaml
```

## 核心功能演示

系统的核心演示程序是 `simulation/pygame_simulator.py`，它提供了一个完整的交互式仿真环境。

### 基本用法

```python
from simulation.environment import Environment
from simulation.pygame_simulator import PygameSimulator
from rrt.rrt_star import RRTStar

# 创建环境
env = Environment(width=100.0, height=100.0)

# 添加一些障碍物
env.add_obstacle(x=30, y=30, obstacle_type="circle", radius=5)
env.add_obstacle(x=70, y=60, obstacle_type="rectangle", width=10, height=10)

# 创建仿真器
simulator = PygameSimulator()
simulator.set_environment(env)

# 设置起点和终点
start = (10, 10)
goal = (90, 90)

# 创建路径规划器
planner = RRTStar(
    start=start,
    goal=goal,
    env=env,
    max_iterations=1000,
    goal_sample_rate=0.1
)

# 规划路径
path = planner.plan()

# 执行仿真
if path:
    simulator.execute_path(path)
```

### 交互控制

在仿真过程中，您可以使用以下按键进行交互：

- **空格键**：暂停/继续仿真
- **R 键**：重置车辆位置到起点
- **C 键**：切换控制方法（在 default、pid、mpc、lqr 之间切换）
- **T 键**：重新规划路径并重置车辆位置
- **ESC 键**：退出仿真

### 控制方法

系统支持多种路径跟踪控制方法：

1. **默认控制器**：基础的路径跟踪算法
2. **PID 控制器**：经典 PID 控制，适合一般场景
3. **MPC 控制器**：模型预测控制，可以预测和优化未来轨迹
4. **LQR 控制器**：线性二次型调节器，提供最优控制

### 安全特性

- 实时碰撞检测
- 碰撞警告提示
- 自动停车机制
- 路径重规划功能

## 项目结构

```
.
├── rrt/                    # RRT 算法实现
│   ├── rrt_base.py         # 基础 RRT 算法
│   └── rrt_star.py         # RRT* 算法
│
├── simulation/             # 仿真环境
│   ├── environment.py      # 环境定义
│   └── pygame_simulator.py # 核心仿真器实现
│
└── examples/               # 示例脚本
    └── demo.py            # 演示脚本
```

## 高级功能

### 1. 自定义环境

您可以通过 `Environment` 类创建自定义环境：

```python
env = Environment(width=100.0, height=100.0)

# 添加圆形障碍物
env.add_obstacle(x=50, y=50, obstacle_type="circle", radius=5)

# 添加矩形障碍物
env.add_obstacle(x=30, y=30, obstacle_type="rectangle", 
                width=10, height=10, angle=0.5)
```

### 2. 调整控制参数

可以通过修改控制器参数来优化性能：

```python
simulator.follower.pid_params = {
    'kp_steer': 0.5,    # 转向角度比例系数
    'ki_steer': 0.05,   # 转向角度积分系数
    'kd_steer': 0.1,    # 转向角度微分系数
    'kp_speed': 0.3,    # 速度比例系数
    'ki_speed': 0.01,   # 速度积分系数
    'kd_speed': 0.05    # 速度微分系数
}
```

### 3. 可视化结果

仿真完成后可以查看详细的运行数据：

```python
results = simulator.get_simulation_results()
simulator.visualize_results(results)
```

## 注意事项

- 确保环境中的障碍物不会完全阻断起点到终点的可行路径
- 在复杂环境中，可能需要增加 RRT* 算法的迭代次数
- 不同的控制方法适合不同的场景，建议尝试不同的控制器
- 如果发生碰撞，系统会自动停止并提供重置选项

## 故障排除

如果遇到问题，可以尝试：

1. 增加 `max_iterations` 参数来提高路径规划成功率
2. 调整 `goal_sample_rate` 来平衡探索和利用
3. 修改控制器参数以获得更好的跟踪效果
4. 确保环境中的障碍物配置合理

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个项目。 