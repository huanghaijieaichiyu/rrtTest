"""
仿真模块

包含环境定义和可视化工具：
- Environment: 路径规划环境
- PygameSimulator: 基于 Pygame 的可视化工具
"""

from simulation.environment import Environment
from simulation.pygame_simulator import PygameSimulator

__all__ = ['Environment', 'PygameSimulator']
