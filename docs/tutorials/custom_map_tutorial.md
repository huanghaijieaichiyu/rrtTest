# CarSim 自定义地图创建教程

本教程将详细介绍如何创建和使用自定义地图进行 CarSim 仿真。

## 目录

1. [环境准备](#1-环境准备)
2. [创建基础环境](#2-创建基础环境)
3. [添加静态障碍物](#3-添加静态障碍物)
4. [设置路面属性](#4-设置路面属性)
5. [添加参考线和标记点](#5-添加参考线和标记点)
6. [保存地图](#6-保存地图)
7. [转换为CarSim格式](#7-转换为carsim格式)
8. [验证地图](#8-验证地图)
9. [CarSim接口测试](#9-carsim接口测试)
10. [注意事项](#10-注意事项)

## 1. 环境准备

### 1.1 必要的库和模块
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from simulation.environment import Environment
from simulation.visualization import Visualization
from simulation.carsim_interface import CarSimInterface
```

### 1.2 工作目录结构
```
project_root/
├── data/
│   └── maps/          # 地图文件存储目录
├── simulation/        # 仿真相关模块
└── config/           # 配置文件目录
```

## 2. 创建基础环境

### 2.1 基本参数设置
```python
env = Environment(
    width=100.0,    # 地图宽度
    height=100.0,   # 地图高度
    grid_size=1.0   # 网格大小
)

vis = Visualization(env)
```

### 2.2 环境参数说明
- width: 地图宽度（米）
- height: 地图高度（米）
- grid_size: 网格大小，用于离散化地图

## 3. 添加静态障碍物

### 3.1 圆形障碍物
```python
env.add_obstacle(
    x=30.0,
    y=30.0,
    obstacle_type="circle",
    radius=5.0
)
```

### 3.2 矩形障碍物
```python
env.add_obstacle(
    x=70.0,
    y=40.0,
    obstacle_type="rectangle",
    width=10.0,
    height=20.0
)
```

## 4. 设置路面属性

### 4.1 定义物理特性
```python
road_properties = {
    'friction': 0.85,        # 摩擦系数
    'roughness': 0.02,       # 粗糙度
    'banking_angle': 0.0,    # 横向倾角
    'elevation': 0.0,        # 纵向坡度
    'lanes': {
        'number': 2,         # 车道数
        'width': 3.75        # 车道宽度(m)
    }
}
```

## 5. 添加参考线和标记点

### 5.1 添加道路中心线
```python
reference_line = [
    (0, 0),
    (20, 20),
    (40, 40),
    (60, 40),
    (80, 60),
    (100, 80)
]
env.add_reference_line(reference_line)
```

### 5.2 添加关键点标记
```python
checkpoints = [
    {'x': 20, 'y': 20, 'description': '检查点1'},
    {'x': 60, 'y': 40, 'description': '检查点2'},
    {'x': 100, 'y': 80, 'description': '终点'}
]

for point in checkpoints:
    env.add_marker(
        x=point['x'],
        y=point['y'],
        marker_type='checkpoint',
        description=point['description']
    )
```

## 6. 保存地图

### 6.1 创建配置文件
```python
map_config = {
    'environment': {
        'width': env.width,
        'height': env.height,
        'grid_size': env.grid_size,
        'obstacles': env.get_obstacles(),
        'reference_line': reference_line,
        'checkpoints': checkpoints,
        'properties': road_properties
    }
}
```

### 6.2 保存为YAML文件
```python
map_file = 'data/maps/custom_map.yaml'
os.makedirs(os.path.dirname(map_file), exist_ok=True)

with open(map_file, 'w', encoding='utf-8') as f:
    yaml.dump(map_config, f, allow_unicode=True)
```

## 7. 转换为CarSim格式

### 7.1 转换函数
```python
def convert_to_carsim_format(map_file: str) -> str:
    """将自定义地图转换为CarSim路面文件格式"""
    with open(map_file, 'r', encoding='utf-8') as f:
        map_data = yaml.safe_load(f)
    
    env_data = map_data['environment']
    reference_line = env_data['reference_line']
    properties = env_data['properties']
    
    road_data = []
    road_data.append("* Road geometry data")
    road_data.append("* Station X Y Z Bank Grade")
    road_data.append("ROAD_DZ")
    
    for i, point in enumerate(reference_line):
        station = i * 10  # 每10米一个站点
        x, y = point
        z = 0  # 高程
        bank = properties['banking_angle']
        grade = properties['elevation']
        
        road_data.append(f"{station:.1f} {x:.3f} {y:.3f} {z:.3f} {bank:.3f} {grade:.3f}")
    
    return "\n".join(road_data)
```

### 7.2 生成CarSim文件
```python
carsim_road_file = 'data/maps/carsim_road.pars'
carsim_road_data = convert_to_carsim_format(map_file)

with open(carsim_road_file, 'w', encoding='utf-8') as f:
    f.write(carsim_road_data)
```

## 8. 验证地图

### 8.1 路径碰撞检测
```python
test_path = [
    (0, 0),
    (20, 20),
    (40, 40),
    (60, 40),
    (80, 60),
    (100, 80)
]

is_valid = env.check_path_collision(test_path)
print(f"路径是否有效: {is_valid}")
```

### 8.2 可视化验证
```python
vis.plot_path(
    path=test_path,
    show_environment=True,
    title="测试路径验证"
)
plt.show()
```

## 9. CarSim接口测试

### 9.1 初始化接口
```python
carsim = CarSimInterface(config_path="config/carsim_config.yaml")
```

### 9.2 加载和测试地图
```python
# 加载自定义地图
carsim.load_custom_road(carsim_road_file)

# 测试路径执行
success = carsim.execute_path(test_path)
print(f"路径执行状态: {'成功' if success else '失败'}")

# 获取和可视化结果
results = carsim.get_simulation_results()
carsim.visualize_results(results)
```

## 10. 注意事项

### 10.1 文件编码
- 所有文本文件使用 UTF-8 编码
- 确保路径中不含特殊字符

### 10.2 坐标系统
- 使用米为单位
- 原点在地图左下角
- X轴向右为正
- Y轴向上为正

### 10.3 性能考虑
- 合理设置地图大小
- 避免过多的障碍物
- 优化采样点密度

### 10.4 调试建议
- 先在程序中完全验证
- 分步测试各个功能
- 保存中间结果 