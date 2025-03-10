# Pygame 路径规划仿真系统使用指南

本文档详细介绍如何使用基于 Pygame 的仿真系统进行路径规划算法测试和可视化，作为 CarSim 的轻量级替代方案。

## 1. 简介

Pygame 仿真系统提供了一个简化的 2D 环境，用于可视化和测试 RRT 系列路径规划算法。与 CarSim 相比，它具有以下优势：

- 开源免费，无需商业许可
- 安装简单，依赖少
- 跨平台兼容性好
- 可以直观地观察算法运行过程
- 提供基本的车辆动力学模型

## 2. 系统要求

### 2.1 依赖库

```bash
# 安装必要的依赖库
pip install pygame numpy matplotlib pandas
```

### 2.2 项目文件

确保以下文件存在于项目中：

- `simulation/pygame_simulator.py` - Pygame 仿真器实现
- `config/pygame_config.yaml` - 仿真器配置文件
- `examples/pygame_simulation_example.py` - 示例脚本

## 3. 快速开始

### 3.1 运行示例脚本

```bash
# 使用默认参数运行示例
python examples/pygame_simulation_example.py

# 指定起点和终点
python examples/pygame_simulation_example.py --start 5 5 --goal 95 95

# 选择不同的算法
python examples/pygame_simulation_example.py --algorithm informed_rrt
```

### 3.2 使用自定义地图

```bash
# 使用自定义地图文件
python examples/pygame_simulation_example.py --map data/maps/custom_map.yaml
```

### 3.3 保存结果

```bash
# 保存仿真结果为 CSV 文件
python examples/pygame_simulation_example.py --save-path data/results/simulation_result.csv
```

## 4. 仿真器配置

仿真器通过 `config/pygame_config.yaml` 文件进行配置，主要参数包括：

```yaml
# 窗口设置
window_width: 1200    # 窗口宽度
window_height: 900    # 窗口高度
window_title: "..."   # 窗口标题

# 仿真参数
scale: 8              # 显示比例（像素/米）
fps: 60               # 帧率
dt: 0.05              # 时间步长(秒)
lookahead: 5.0        # 路径跟踪前瞻距离

# 车辆参数
vehicle:
  length: 4.5         # 车辆长度(米)
  width: 1.8          # 车辆宽度(米)
  # ... 其他车辆参数
```

## 5. 交互控制

仿真过程中可以使用以下按键控制：

- `ESC`: 退出仿真
- `空格键`: 暂停/继续仿真
- `R`: 重置车辆位置到起点

## 6. 自定义仿真环境

### 6.1 创建自定义环境

您可以通过编程方式创建环境：

```python
from simulation.environment import Environment

# 创建环境
env = Environment(width=100.0, height=100.0)

# 添加障碍物
env.add_obstacle(x=30, y=30, obstacle_type="circle", radius=5.0)
env.add_obstacle(x=70, y=40, obstacle_type="rectangle", width=10.0, height=20.0)
```

### 6.2 保存/加载环境

```python
# 保存环境到文件
env.save_map("data/maps/my_map.yaml")

# 从文件加载环境
def load_environment(map_file):
    # ... (参见示例代码)
```

## 7. 系统架构

### 7.1 主要组件

- `PygameSimulator`: 主仿真器类，负责渲染和仿真循环
- `VehicleModel`: 车辆动力学模型
- `PathFollower`: 路径跟踪控制器

### 7.2 类关系图

```
Environment <---- PygameSimulator
                      |
                      ├---- VehicleModel
                      └---- PathFollower
```

## 8. 扩展功能

### 8.1 添加新的车辆模型

您可以通过扩展 `VehicleModel` 类来创建更复杂的车辆模型：

```python
class AdvancedVehicleModel(VehicleModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加新属性
        
    def update(self, throttle, brake, steer, dt):
        # 实现更复杂的动力学模型
```

### 8.2 添加传感器模拟

您可以添加模拟的传感器，如激光雷达、摄像头等：

```python
class LidarSensor:
    def __init__(self, vehicle, range=50.0, num_rays=180):
        self.vehicle = vehicle
        self.range = range
        self.num_rays = num_rays
        
    def scan(self, environment):
        # 实现激光雷达扫描逻辑
        # 返回距离数组
```

## 9. 常见问题

### 9.1 性能问题

- **问题**: 仿真运行缓慢
- **解决方案**: 
  - 减少窗口尺寸
  - 降低 FPS
  - 简化障碍物数量

### 9.2 碰撞检测问题

- **问题**: 车辆穿过障碍物
- **解决方案**:
  - 检查障碍物定义
  - 确保车辆尺寸设置正确
  - 减小仿真时间步长

### 9.3 路径跟踪问题

- **问题**: 车辆无法准确跟踪路径
- **解决方案**:
  - 增加前瞻距离
  - 调整车辆最大速度和转向角
  - 在路径中添加更多中间点

## 10. 未来扩展

- 添加动态障碍物
- 实现更复杂的车辆动力学模型
- 支持多车仿真
- 添加更多的环境元素（如路标、信号灯）
- 集成机器学习模型进行决策

---

通过使用这个基于 Pygame 的仿真系统，您可以方便地测试和可视化各种路径规划算法，而无需依赖商业软件。系统提供了足够的灵活性，可以根据需要进行扩展和定制。 