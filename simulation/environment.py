#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
环境定义模块

提供路径规划环境的定义，包括：
- 环境边界
- 障碍物管理
- 碰撞检测
- 栅格化表示
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
from shapely.geometry import Point, LineString
from matplotlib.patches import Circle, Polygon
from dataclasses import dataclass


@dataclass
class CircleObstacle:
    """圆形障碍物"""
    x: float
    y: float
    radius: float
    type: str = 'circle'


@dataclass
class RectangleObstacle:
    """矩形障碍物"""
    x: float
    y: float
    width: float
    height: float
    angle: float = 0.0
    type: str = 'rectangle'


class Obstacle:
    """障碍物基类"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def check_collision(self, point: Tuple[float, float]) -> bool:
        """检查点是否与障碍物碰撞"""
        raise NotImplementedError("子类必须实现此方法")

    def check_line_collision(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> bool:
        """检查线段是否与障碍物碰撞"""
        raise NotImplementedError("子类必须实现此方法")

    def plot(self, ax) -> None:
        """在给定的坐标轴上绘制障碍物"""
        raise NotImplementedError("子类必须实现此方法")


class CircleObstacle(Obstacle):
    """圆形障碍物"""

    def __init__(self, x: float, y: float, radius: float):
        super().__init__(x, y)
        self.radius = radius
        # 使用Shapely创建几何表示
        self.geometry = Point(x, y).buffer(radius)

    def check_collision(self, point: Tuple[float, float]) -> bool:
        """检查点是否与圆形障碍物碰撞"""
        # 计算点到障碍物中心的距离
        dist = np.hypot(point[0] - self.x, point[1] - self.y)
        return dist <= self.radius

    def check_line_collision(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """检查线段是否与圆形障碍物碰撞"""
        # 使用Shapely检查线段与圆的相交
        line = LineString([start, end])
        return line.intersects(self.geometry)

    def plot(self, ax) -> None:
        """在给定的坐标轴上绘制圆形障碍物"""
        circle = plt.Circle((self.x, self.y), self.radius,
                            color='r', alpha=0.5)
        ax.add_patch(circle)


class RectangleObstacle(Obstacle):
    """矩形障碍物"""

    def __init__(self, x: float, y: float, width: float, height: float, angle: float = 0.0):
        super().__init__(x, y)
        self.width = width
        self.height = height
        self.angle = angle  # 弧度

        # 计算四个角点
        self._compute_corners()

        # 使用Shapely创建几何表示
        from shapely.geometry import Polygon
        self.geometry = Polygon(self.corners)

    def _compute_corners(self) -> None:
        """计算矩形的四个角点坐标"""
        # 未旋转时的半宽和半高
        hw, hh = self.width / 2, self.height / 2

        # 未旋转时的四个角点（相对于中心）
        corners_rel = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

        # 应用旋转
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        corners = []
        for x_rel, y_rel in corners_rel:
            # 旋转变换
            x_rot = x_rel * cos_a - y_rel * sin_a
            y_rot = x_rel * sin_a + y_rel * cos_a
            # 平移到实际位置
            corners.append((self.x + x_rot, self.y + y_rot))

        self.corners = corners

    def check_collision(self, point: Tuple[float, float]) -> bool:
        """检查点是否与矩形障碍物碰撞"""
        # 使用Shapely检查点是否在多边形内
        p = Point(point)
        return p.within(self.geometry) or p.touches(self.geometry)

    def check_line_collision(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """检查线段是否与矩形障碍物碰撞"""
        # 使用Shapely检查线段与多边形的相交
        line = LineString([start, end])
        return line.intersects(self.geometry)

    def plot(self, ax) -> None:
        """在给定的坐标轴上绘制矩形障碍物"""
        from matplotlib.patches import Polygon
        rect = Polygon(self.corners, closed=True, color='r', alpha=0.5)
        ax.add_patch(rect)


class Environment:
    """
    路径规划环境类

    提供环境表示、碰撞检测和可视化功能。
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        map_path: Optional[str] = None
    ):
        """
        初始化环境

        参数:
            width: 环境宽度
            height: 环境高度
            map_path: 地图文件路径（可选）
        """
        self.width = width
        self.height = height
        self.obstacles: List[Union[CircleObstacle, RectangleObstacle]] = []

        # 如果提供了地图文件，从文件加载环境
        if map_path:
            self.load_map(map_path)

    def add_obstacle(
        self,
        x: float,
        y: float,
        obstacle_type: str = "circle",
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        angle: float = 0.0
    ) -> None:
        """
        添加障碍物

        参数:
            x: 障碍物中心x坐标
            y: 障碍物中心y坐标
            obstacle_type: 障碍物类型 ("circle" 或 "rectangle")
            radius: 圆形障碍物半径
            width: 矩形障碍物宽度
            height: 矩形障碍物高度
            angle: 矩形障碍物旋转角度（弧度）
        """
        if obstacle_type == "circle" and radius is not None:
            self.obstacles.append(CircleObstacle(x=x, y=y, radius=radius))
        elif obstacle_type == "rectangle" and width is not None and height is not None:
            self.obstacles.append(
                RectangleObstacle(
                    x=x, y=y,
                    width=width,
                    height=height,
                    angle=angle
                )
            )
        else:
            raise ValueError("无效的障碍物参数")

    def check_collision(self, point: Tuple[float, float]) -> bool:
        """
        检查点是否与障碍物碰撞

        参数:
            point: 待检查的点坐标 (x, y)

        返回:
            是否发生碰撞
        """
        x, y = point
        
        # 检查是否在环境边界内
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return True
        
        # 检查是否与障碍物碰撞
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                # 圆形障碍物碰撞检测
                dist = np.sqrt(
                    (x - obstacle.x) ** 2 + (y - obstacle.y) ** 2
                )
                if dist <= obstacle.radius:
                    return True
            else:
                # 矩形障碍物碰撞检测（简化版，不考虑旋转）
                if (abs(x - obstacle.x) <= obstacle.width / 2 and
                    abs(y - obstacle.y) <= obstacle.height / 2):
                    return True
        
        return False

    def check_segment_collision(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """
        检查线段是否与任意障碍物碰撞

        参数:
            start: 线段起点坐标 (x, y)
            end: 线段终点坐标 (x, y)

        返回:
            如果无碰撞返回True，否则返回False
        """
        # 检查端点是否超出边界
        if (start[0] < 0 or start[0] > self.width or
            start[1] < 0 or start[1] > self.height or
            end[0] < 0 or end[0] > self.width or
                end[1] < 0 or end[1] > self.height):
            return False

        # 检查是否与任意障碍物碰撞
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                # 圆形障碍物碰撞检测
                dist = np.sqrt(
                    (start[0] - obs.x) ** 2 + (start[1] - obs.y) ** 2
                )
                if dist <= obs.radius:
                    return True
            elif isinstance(obs, RectangleObstacle):
                # 矩形障碍物碰撞检测（简化版，不考虑旋转）
                if (abs(start[0] - obs.x) <= obs.width / 2 and
                    abs(start[1] - obs.y) <= obs.height / 2) or \
                   (abs(end[0] - obs.x) <= obs.width / 2 and
                    abs(end[1] - obs.y) <= obs.height / 2):
                    return True

        return False

    def load_map(self, map_path: str) -> None:
        """
        从文件加载地图

        参数:
            map_path: 地图文件路径
        """
        # 暂时简单实现，根据实际需求扩展
        # 假设地图文件是YAML格式
        import yaml
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                map_data = yaml.safe_load(f)

            # 解析地图数据
            self.width = map_data.get('width', self.width)
            self.height = map_data.get('height', self.height)

            # 加载障碍物
            for obs_data in map_data.get('obstacles', []):
                self.add_obstacle(**obs_data)

        except Exception as e:
            print(f"加载地图失败: {e}")

    def save_map(self, map_path: str) -> None:
        """
        保存地图到文件

        参数:
            map_path: 地图文件路径
        """
        # 准备地图数据
        map_data = {
            'width': self.width,
            'height': self.height,
            'obstacles': []
        }

        # 保存障碍物数据
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                obs_data = {
                    'x': obs.x,
                    'y': obs.y,
                    'obstacle_type': 'circle',
                    'radius': obs.radius
                }
            elif isinstance(obs, RectangleObstacle):
                obs_data = {
                    'x': obs.x,
                    'y': obs.y,
                    'obstacle_type': 'rectangle',
                    'width': obs.width,
                    'height': obs.height,
                    'angle': obs.angle
                }
            else:
                continue

            map_data['obstacles'].append(obs_data)

        # 保存到文件
        import yaml
        try:
            with open(map_path, 'w', encoding='utf-8') as f:
                yaml.dump(map_data, f, default_flow_style=False,
                          allow_unicode=True)
        except Exception as e:
            print(f"保存地图失败: {e}")

    def plot_obstacles(self, ax) -> None:
        """绘制所有障碍物"""
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                obs.plot(ax)
            elif isinstance(obs, RectangleObstacle):
                obs.plot(ax)

    def visualize(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """可视化整个环境"""
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制边界
        ax.plot([0, self.width, self.width, 0, 0],
                [0, 0, self.height, self.height, 0], 'k-')

        # 绘制障碍物
        self.plot_obstacles(ax)

        # 设置坐标轴
        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('环境地图')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.show()

    def visualize_path(
        self,
        path: List[Tuple[float, float]],
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        可视化路径

        参数:
            path: 路径点列表
            figsize: 图形大小
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制边界
        ax.plot([0, self.width, self.width, 0, 0],
                [0, 0, self.height, self.height, 0], 'k-')

        # 绘制障碍物
        self.plot_obstacles(ax)

        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, '-b', linewidth=2, label='路径')

            # 绘制起点和终点
            ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='起点')
            ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='终点')

        # 设置坐标轴
        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('规划路径')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()

        plt.show()

    def to_grid(
        self,
        grid_size: Tuple[int, int] = (64, 64)
    ) -> np.ndarray:
        """
        将环境转换为栅格表示

        参数:
            grid_size: 栅格大小 (width, height)

        返回:
            栅格化的环境表示，0表示空闲，1表示障碍物
        """
        grid_width, grid_height = grid_size
        grid = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # 计算栅格分辨率
        cell_width = self.width / grid_width
        cell_height = self.height / grid_height
        
        # 对每个栅格进行采样
        for i in range(grid_height):
            for j in range(grid_width):
                # 计算栅格中心点坐标
                x = (j + 0.5) * cell_width
                y = (i + 0.5) * cell_height
                
                # 检查是否碰撞
                if self.check_collision((x, y)):
                    grid[i, j] = 1.0
        
        return grid

    def save(self, filepath: str) -> None:
        """
        保存环境到文件

        参数:
            filepath: 保存路径
        """
        import json
        
        # 将障碍物转换为字典
        obstacles_dict = []
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                obstacles_dict.append({
                    'type': 'circle',
                    'x': obs.x,
                    'y': obs.y,
                    'radius': obs.radius
                })
            else:
                obstacles_dict.append({
                    'type': 'rectangle',
                    'x': obs.x,
                    'y': obs.y,
                    'width': obs.width,
                    'height': obs.height,
                    'angle': obs.angle
                })
        
        # 保存环境配置
        env_dict = {
            'width': self.width,
            'height': self.height,
            'obstacles': obstacles_dict
        }
        
        with open(filepath, 'w') as f:
            json.dump(env_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Environment':
        """
        从文件加载环境

        参数:
            filepath: 文件路径

        返回:
            Environment 对象
        """
        import json
        
        with open(filepath, 'r') as f:
            env_dict = json.load(f)
        
        # 创建环境
        env = cls(
            width=env_dict['width'],
            height=env_dict['height']
        )
        
        # 添加障碍物
        for obs_dict in env_dict['obstacles']:
            if obs_dict['type'] == 'circle':
                env.add_obstacle(
                    x=obs_dict['x'],
                    y=obs_dict['y'],
                    obstacle_type='circle',
                    radius=obs_dict['radius']
                )
            else:
                env.add_obstacle(
                    x=obs_dict['x'],
                    y=obs_dict['y'],
                    obstacle_type='rectangle',
                    width=obs_dict['width'],
                    height=obs_dict['height'],
                    angle=obs_dict.get('angle', 0.0)
                )
        
        return env


if __name__ == "__main__":
    # 测试环境
    env = Environment(width=100, height=100)

    # 添加一些随机障碍物
    np.random.seed(42)  # 设置随机种子以便重现结果

    # 添加圆形障碍物
    for _ in range(10):
        env.add_obstacle(
            x=np.random.uniform(10, 90),
            y=np.random.uniform(10, 90),
            obstacle_type="circle",
            radius=np.random.uniform(2, 8)
        )

    # 添加矩形障碍物
    for _ in range(5):
        env.add_obstacle(
            x=np.random.uniform(10, 90),
            y=np.random.uniform(10, 90),
            obstacle_type="rectangle",
            width=np.random.uniform(5, 15),
            height=np.random.uniform(5, 15),
            angle=np.random.uniform(0, 2 * np.pi)
        )

    # 可视化环境
    env.visualize()

    # 保存和加载地图测试
    env.save_map("test_map.yaml")
    new_env = Environment()
    new_env.load_map("test_map.yaml")
    new_env.visualize()
