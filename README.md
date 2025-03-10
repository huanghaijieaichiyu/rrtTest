# RRT Path Planning Project

基于 RRT（Rapidly-exploring Random Tree）算法的路径规划项目，包含多种 RRT 变体实现和可视化工具。

## 功能特点

- 支持多种 RRT 算法变体：
  - 基础 RRT
  - RRT*
  - Informed RRT*
  - 基于深度学习的神经网络增强 RRT
- 使用 Pygame 进行实时可视化
- 支持自定义环境创建和保存
- 提供神经网络模型训练功能

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/rrt-path-planning.git
cd rrt-path-planning
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

项目提供统一的命令行接口，支持以下主要功能：

### 1. 运行路径规划算法

```bash
# 使用 RRT* 算法（默认）
python main.py run

# 指定算法、起点和终点
python main.py run --algorithm rrt_star --start 10 10 --goal 90 90

# 使用自定义地图
python main.py run --algorithm informed_rrt --map maps/custom_map.json

# 使用神经网络增强的 RRT
python main.py run --algorithm neural_rrt --model-path results/models/model.pth

# 可视化执行过程
python main.py run --visualize

# 保存规划路径
python main.py run --save-path results/paths/path.json
```

可用参数：
- `--algorithm`: 选择算法 [rrt, rrt_star, informed_rrt, neural_rrt]
- `--start`: 起点坐标，例如 "10 10"
- `--goal`: 终点坐标，例如 "90 90"
- `--map`: 地图文件路径
- `--model-path`: 神经网络模型路径（仅用于 neural_rrt）
- `--iterations`: 最大迭代次数
- `--save-path`: 保存路径的文件路径
- `--visualize`: 是否可视化执行过程

### 2. 训练神经网络增强的 RRT

```bash
# 使用默认参数训练
python main.py train

# 自定义训练参数
python main.py train --num-episodes 2000 --num-epochs 200 --batch-size 64

# 指定保存目录和设备
python main.py train --save-dir results/my_models --device cuda
```

可用参数：
- `--num-episodes`: 训练数据收集的路径数量
- `--num-epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--learning-rate`: 学习率
- `--save-dir`: 模型保存目录
- `--device`: 训练设备 (cuda/cpu)
- `--visualize`: 是否可视化训练过程

### 3. 创建测试环境

```bash
# 创建随机环境
python main.py create-env --save-path maps/random_map.json

# 自定义环境参数
python main.py create-env --width 200 --height 200 --num-circles 8 --num-rectangles 5 --save-path maps/large_map.json

# 可视化创建的环境
python main.py create-env --save-path maps/map.json --visualize
```

可用参数：
- `--width`: 环境宽度
- `--height`: 环境高度
- `--num-circles`: 圆形障碍物数量
- `--num-rectangles`: 矩形障碍物数量
- `--save-path`: 保存环境的文件路径
- `--visualize`: 是否可视化环境

## 项目结构

```
.
├── main.py              # 主程序入口
├── rrt/                 # RRT 算法实现
│   ├── rrt_base.py     # 基础 RRT
│   ├── rrt_star.py     # RRT*
│   └── informed_rrt.py # Informed RRT*
├── ml/                  # 机器学习相关
│   └── models/         # 神经网络模型
├── simulation/         # 仿真和可视化
│   ├── environment.py  # 环境定义
│   └── pygame_simulator.py # Pygame 可视化
├── examples/           # 示例脚本
├── config/            # 配置文件
├── results/           # 结果保存
│   ├── models/       # 训练模型
│   └── paths/        # 规划路径
└── maps/             # 环境地图
```

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 