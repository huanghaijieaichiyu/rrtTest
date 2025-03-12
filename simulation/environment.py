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
from matplotlib.patches import Circle, Polygon
from typing import List, Optional, Tuple, Union
from shapely.geometry import Point, LineString
from dataclasses import dataclass
import math


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

    def get_min_distance(self, x: float, y: float) -> float:
        """
        计算点到所有障碍物的最小距离

        参数:
            x: 点的x坐标
            y: 点的y坐标

        返回:
            到最近障碍物的距离
        """
        if not self.obstacles:
            return float('inf')

        min_dist = float('inf')
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                # 计算点到圆心的距离
                dist = ((x - obstacle.x) ** 2 + (y - obstacle.y) ** 2) ** 0.5
                # 减去圆的半径得到到圆边界的距离
                dist = max(0.0, dist - obstacle.radius)
                min_dist = min(min_dist, dist)
            elif isinstance(obstacle, RectangleObstacle):
                # 计算点到矩形中心的距离
                dx = abs(x - obstacle.x) - obstacle.width / 2
                dy = abs(y - obstacle.y) - obstacle.height / 2

                if dx <= 0 and dy <= 0:
                    # 点在矩形内部
                    min_dist = 0.0
                    break
                elif dx <= 0:
                    # 点在矩形的上方或下方
                    min_dist = min(min_dist, max(0.0, dy))
                elif dy <= 0:
                    # 点在矩形的左侧或右侧
                    min_dist = min(min_dist, max(0.0, dx))
                else:
                    # 点在矩形的对角方向
                    min_dist = min(min_dist, (dx * dx + dy * dy) ** 0.5)

        return min_dist

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

    def check_segment_collision(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        vehicle_width: float = 0.0,
        vehicle_length: float = 0.0
    ) -> bool:
        """
        检查线段是否与任意障碍物碰撞，考虑车辆尺寸

        参数:
            start: 线段起点坐标 (x, y)
            end: 线段终点坐标 (x, y)
            vehicle_width: 车辆宽度，默认为0（点）
            vehicle_length: 车辆长度，默认为0（点）

        返回:
            是否发生碰撞
        """
        # 如果没有指定车辆尺寸，使用点碰撞检测
        if vehicle_width == 0 or vehicle_length == 0:
            # 创建线段的几何表示
            line = LineString([start, end])

            # 检查是否与任意障碍物碰撞
            for obstacle in self.obstacles:
                if isinstance(obstacle, CircleObstacle):
                    # 创建圆形障碍物的几何表示
                    circle = Point(obstacle.x, obstacle.y).buffer(
                        obstacle.radius)
                    if line.intersects(circle):
                        return True
                else:
                    # 创建矩形障碍物的几何表示
                    x_min = obstacle.x - obstacle.width / 2
                    x_max = obstacle.x + obstacle.width / 2
                    y_min = obstacle.y - obstacle.height / 2
                    y_max = obstacle.y + obstacle.height / 2
                    rect = Polygon([
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max)
                    ])
                    if line.intersects(rect):
                        return True
        else:
            # 考虑车辆尺寸的碰撞检测
            # 计算路径方向
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            heading = math.atan2(dy, dx)

            # 创建车辆多边形
            # 在路径的多个点上检查车辆碰撞
            steps = 5  # 采样点数量
            for i in range(steps + 1):
                t = i / steps
                x = start[0] + t * dx
                y = start[1] + t * dy

                # 计算车辆四个角的坐标
                half_length = vehicle_length / 2
                half_width = vehicle_width / 2
                cos_h = math.cos(heading)
                sin_h = math.sin(heading)

                corners = [
                    (x + half_length * cos_h - half_width * sin_h,
                     y + half_length * sin_h + half_width * cos_h),
                    (x + half_length * cos_h + half_width * sin_h,
                     y + half_length * sin_h - half_width * cos_h),
                    (x - half_length * cos_h + half_width * sin_h,
                     y - half_length * sin_h - half_width * cos_h),
                    (x - half_length * cos_h - half_width * sin_h,
                     y - half_length * sin_h + half_width * cos_h)
                ]

                vehicle_polygon = Polygon(corners)

                # 检查是否与任意障碍物碰撞
                for obstacle in self.obstacles:
                    if isinstance(obstacle, CircleObstacle):
                        # 圆形障碍物
                        circle = Point(obstacle.x, obstacle.y).buffer(
                            obstacle.radius)
                        if vehicle_polygon.intersects(circle):
                            return True
                    else:
                        # 矩形障碍物
                        x_min = obstacle.x - obstacle.width / 2
                        x_max = obstacle.x + obstacle.width / 2
                        y_min = obstacle.y - obstacle.height / 2
                        y_max = obstacle.y + obstacle.height / 2

                        # 考虑障碍物的旋转角度
                        if hasattr(obstacle, 'angle') and obstacle.angle != 0:
                            # 创建旋转后的矩形
                            rect_corners = [
                                (x_min, y_min),
                                (x_max, y_min),
                                (x_max, y_max),
                                (x_min, y_max)
                            ]

                            # 旋转矩形的角点
                            cos_angle = math.cos(-obstacle.angle)
                            sin_angle = math.sin(-obstacle.angle)
                            rotated_corners = []

                            for x, y in rect_corners:
                                # 平移到原点
                                tx = x - obstacle.x
                                ty = y - obstacle.y
                                # 旋转
                                rx = tx * cos_angle - ty * sin_angle
                                ry = tx * sin_angle + ty * cos_angle
                                # 平移回原位置
                                rotated_corners.append(
                                    (rx + obstacle.x, ry + obstacle.y))

                            obstacle_polygon = Polygon(rotated_corners)
                        else:
                            # 不旋转的矩形
                            obstacle_polygon = Polygon([
                                (x_min, y_min),
                                (x_max, y_min),
                                (x_max, y_max),
                                (x_min, y_max)
                            ])

                        if vehicle_polygon.intersects(obstacle_polygon):
                            return True

        return False

    def to_grid(self, grid_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        将环境转换为栅格表示

        参数:
            grid_size: 栅格大小 (width, height)

        返回:
            栅格地图，0表示空闲，1表示障碍物
        """
        grid_width, grid_height = grid_size
        grid = np.zeros((grid_height, grid_width))

        # 计算栅格分辨率
        res_x = self.width / grid_width
        res_y = self.height / grid_height

        # 遍历每个栅格
        for i in range(grid_height):
            for j in range(grid_width):
                # 计算栅格中心点坐标
                x = (j + 0.5) * res_x
                y = (i + 0.5) * res_y

                # 检查是否与障碍物碰撞
                if self.check_collision((x, y)):
                    grid[i, j] = 1

        return grid

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
                circle = plt.Circle((obs.x, obs.y), obs.radius,
                                    color='r', alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obs, RectangleObstacle):
                from matplotlib.patches import Polygon
                rect = Polygon(self._compute_corners(
                    obs), closed=True, color='r', alpha=0.5)
                ax.add_patch(rect)

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

    def _compute_corners(self, obstacle: RectangleObstacle) -> List[Tuple[float, float]]:
        """计算矩形的四个角点坐标"""
        # 未旋转时的半宽和半高
        hw, hh = obstacle.width / 2, obstacle.height / 2

        # 未旋转时的四个角点（相对于中心）
        corners_rel = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

        # 应用旋转
        cos_a, sin_a = np.cos(obstacle.angle), np.sin(obstacle.angle)
        corners = []
        for x_rel, y_rel in corners_rel:
            # 旋转变换
            x_rot = x_rel * cos_a - y_rel * sin_a
            y_rot = x_rel * sin_a + y_rel * cos_a
            # 平移到实际位置
            corners.append((obstacle.x + x_rot, obstacle.y + y_rot))

        return corners


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
