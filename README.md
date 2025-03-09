# RRT-PyTorch-CarSim 仿真项目

一个基于PyTorch和CarSim的RRT（快速探索随机树）路径规划算法仿真平台，用于自动驾驶车辆路径规划研究。

## 项目概述

本项目结合了RRT路径规划算法、PyTorch深度学习框架和CarSim车辆动力学仿真软件，创建了一个完整的自动驾驶路径规划仿真环境。主要功能包括：

- 基于RRT及其变种（RRT*、Informed RRT*等）的路径规划算法实现
- 使用PyTorch构建的神经网络模型，用于优化路径规划策略
- 与CarSim的接口，实现高保真度的车辆动力学仿真
- 可视化工具，展示规划路径和车辆行为

## 安装指南

### 前提条件

- Python 3.8+
- CarSim 软件（需单独安装）
- CUDA（可选，用于GPU加速）

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/huangxiaohaiaichiyu/rrt-pytorch-carsim.git
   cd rrt-pytorch-carsim
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置CarSim：
   - 在`config/carsim_config.yaml`中配置CarSim安装路径和接口设置

## 目录结构

```
rrt-pytorch-carsim/
│
├── rrt/                    # RRT算法实现
│   ├── __init__.py
│   ├── rrt_base.py         # 基础RRT算法
│   ├── rrt_star.py         # RRT*算法
│   └── informed_rrt.py     # Informed RRT*算法
│
├── ml/                     # 机器学习模型
│   ├── __init__.py
│   ├── models/             # PyTorch模型定义
│   ├── training/           # 训练脚本
│   └── inference/          # 推理代码
│
├── simulation/             # 仿真环境
│   ├── __init__.py
│   ├── carsim_interface.py # CarSim接口
│   ├── environment.py      # 仿真环境定义
│   └── visualization.py    # 可视化工具
│
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── math_utils.py       # 数学工具
│   └── data_processing.py  # 数据处理工具
│
├── config/                 # 配置文件
│   ├── default_config.yaml # 默认配置
│   └── carsim_config.yaml  # CarSim配置
│
├── examples/               # 示例脚本
│   ├── basic_rrt_demo.py   # 基础RRT演示
│   ├── ml_rrt_demo.py      # 结合机器学习的RRT演示
│   └── carsim_demo.py      # CarSim集成演示
│
├── tests/                  # 测试代码
│   ├── __init__.py
│   ├── test_rrt.py
│   └── test_carsim.py
│
├── data/                   # 数据目录
│   ├── maps/               # 地图数据
│   ├── trajectories/       # 轨迹数据
│   └── model_weights/      # 模型权重
│
├── docs/                   # 文档
│   ├── api_reference.md
│   └── tutorials/
│
├── main.py                 # 主入口
├── run_simulation.py       # 运行仿真
├── requirements.txt        # 项目依赖
└── README.md               # 本文档
```

## 使用示例

### 基础RRT路径规划

```python
from rrt.rrt_base import RRT
from simulation.environment import Environment

# 创建环境
env = Environment(map_path="data/maps/example_map.yaml")

# 初始化RRT规划器
rrt_planner = RRT(
    start=(0, 0),
    goal=(100, 100),
    env=env,
    step_size=5.0,
    max_iterations=1000
)

# 执行路径规划
path = rrt_planner.plan()

# 可视化结果
env.visualize_path(path)
```

### 与CarSim集成

```python
from simulation.carsim_interface import CarSimInterface
from rrt.rrt_star import RRTStar

# 创建CarSim接口
carsim = CarSimInterface(config_path="config/carsim_config.yaml")

# 初始化规划器
planner = RRTStar(
    start=carsim.get_vehicle_state(),
    goal=(100, 100),
    env=carsim.get_environment(),
    step_size=2.0
)

# 规划路径
path = planner.plan()

# 在CarSim中执行路径
carsim.execute_path(path)
```

## 与PyTorch结合

项目使用PyTorch实现了几种强化学习和监督学习方法来优化RRT算法，例如：

- 使用神经网络预测采样区域
- 基于深度强化学习的路径优化
- 端到端的轨迹生成

详细示例请参见`examples/ml_rrt_demo.py`。

## 贡献指南

欢迎贡献代码、报告问题或提出新功能建议。请遵循以下步骤：

1. Fork仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题，请通过以下方式联系：

- 电子邮件：huangxiaohai99@126.com
- GitHub Issues 