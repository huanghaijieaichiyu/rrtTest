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
from matplotlib.patches import Circle as MplCircle
from typing import List, Optional, Tuple, Union
from shapely.geometry import Point, LineString, Polygon
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

    def check_collision(self, point: Tuple[float, float], vehicle_width: float = 0.0, vehicle_length: float = 0.0) -> bool:
        """
        检查点是否与障碍物碰撞

        参数:
            point: 待检查的点坐标 (x, y)
            vehicle_width: 车辆宽度，默认为0（点）
            vehicle_length: 车辆长度，默认为0（点）

        返回:
            是否发生碰撞
        """
        x, y = point

        # 检查是否在环境边界内
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return True

        # 如果提供了车辆尺寸，使用check_segment_collision方法
        if vehicle_width > 0 and vehicle_length > 0 and hasattr(self, 'check_segment_collision'):
            return self.check_segment_collision(point, point, vehicle_width, vehicle_length)

        # 否则使用点碰撞检测
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


class Obstacle:
    def __init__(self, x, y, color=(0, 0, 0, 255), is_filled=True, line_width=2):
        self.x = x
        self.y = y
        self.color = color
        self.is_filled = is_filled
        self.line_width = line_width

    def check_collision(self, x, y):
        """检查点(x, y)是否与障碍物碰撞"""
        raise NotImplementedError("子类必须实现此方法")

    def check_line_collision(self, x1, y1, x2, y2):
        """检查线段是否与障碍物碰撞"""
        raise NotImplementedError("子类必须实现此方法")

# 矩形障碍物


class RectangleObstacle(Obstacle):
    def __init__(self, x, y, width, height, angle=0.0, color=(0, 0, 0, 255), is_filled=True, line_width=2):
        super().__init__(x, y, color, is_filled, line_width)
        self.width = width
        self.height = height
        self.angle = angle  # 角度，单位为度
        self.angle_rad = math.radians(angle)  # 转换为弧度
        # 兼容旧代码的属性
        self.type = "rectangle"
        self.radius = 0.0
        self.is_parking_spot = False
        self.occupied = False

    def check_collision(self, x, y):
        """检查点(x, y)是否与矩形障碍物碰撞"""
        # 如果是未占用的停车位，不视为障碍物
        if self.is_parking_spot and not self.occupied:
            return False

        # 将点转换到矩形的局部坐标系
        dx = x - self.x
        dy = y - self.y

        # 旋转点
        rotated_x = dx * math.cos(-self.angle_rad) - \
            dy * math.sin(-self.angle_rad)
        rotated_y = dx * math.sin(-self.angle_rad) + \
            dy * math.cos(-self.angle_rad)

        # 检查点是否在矩形内
        return (abs(rotated_x) <= self.width / 2 and abs(rotated_y) <= self.height / 2)

    def check_line_collision(self, x1, y1, x2, y2):
        """检查线段是否与矩形障碍物碰撞"""
        # 如果是未占用的停车位，不视为障碍物
        if self.is_parking_spot and not self.occupied:
            return False

        # 将线段的两个端点转换到矩形的局部坐标系
        dx1 = x1 - self.x
        dy1 = y1 - self.y
        dx2 = x2 - self.x
        dy2 = y2 - self.y

        # 旋转点
        rotated_x1 = dx1 * math.cos(-self.angle_rad) - \
            dy1 * math.sin(-self.angle_rad)
        rotated_y1 = dx1 * math.sin(-self.angle_rad) + \
            dy1 * math.cos(-self.angle_rad)
        rotated_x2 = dx2 * math.cos(-self.angle_rad) - \
            dy2 * math.sin(-self.angle_rad)
        rotated_y2 = dx2 * math.sin(-self.angle_rad) + \
            dy2 * math.cos(-self.angle_rad)

        # 矩形的边界
        left = -self.width / 2
        right = self.width / 2
        bottom = -self.height / 2
        top = self.height / 2

        # 检查线段是否与矩形相交
        # 使用Cohen-Sutherland算法的思想

        # 如果两个端点都在矩形内，则线段与矩形相交
        if (left <= rotated_x1 <= right and bottom <= rotated_y1 <= top) or \
           (left <= rotated_x2 <= right and bottom <= rotated_y2 <= top):
            return True

        # 检查线段是否与矩形的四条边相交
        # 左边
        if self._check_line_intersection(rotated_x1, rotated_y1, rotated_x2, rotated_y2, left, bottom, left, top):
            return True
        # 右边
        if self._check_line_intersection(rotated_x1, rotated_y1, rotated_x2, rotated_y2, right, bottom, right, top):
            return True
        # 下边
        if self._check_line_intersection(rotated_x1, rotated_y1, rotated_x2, rotated_y2, left, bottom, right, bottom):
            return True
        # 上边
        if self._check_line_intersection(rotated_x1, rotated_y1, rotated_x2, rotated_y2, left, top, right, top):
            return True

        return False

    def _check_line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """检查两条线段是否相交"""
        # 计算分母
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        # 如果分母为0，则线段平行或共线
        if den == 0:
            return False

        # 计算分子
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

        # 如果ua和ub都在[0,1]范围内，则线段相交
        return (0 <= ua <= 1) and (0 <= ub <= 1)

# 圆形障碍物


class CircleObstacle(Obstacle):
    def __init__(self, x, y, radius, color=(0, 0, 0, 255), is_filled=True, line_width=2):
        super().__init__(x, y, color, is_filled, line_width)
        self.radius = radius
        # 兼容旧代码的属性
        self.type = "circle"
        self.width = radius * 2
        self.height = radius * 2
        self.angle = 0.0
        self.is_parking_spot = False
        self.occupied = False

    def check_collision(self, x, y):
        """检查点(x, y)是否与圆形障碍物碰撞"""
        # 如果是未占用的停车位，不视为障碍物
        if self.is_parking_spot and not self.occupied:
            return False

        # 计算点到圆心的距离
        distance = math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

        # 如果距离小于等于半径，则点在圆内
        return distance <= self.radius

    def check_line_collision(self, x1, y1, x2, y2):
        """检查线段是否与圆形障碍物碰撞"""
        # 如果是未占用的停车位，不视为障碍物
        if self.is_parking_spot and not self.occupied:
            return False

        # 计算线段的方向向量
        dx = x2 - x1
        dy = y2 - y1

        # 计算线段长度的平方
        length_squared = dx ** 2 + dy ** 2

        # 如果线段长度为0，则检查端点是否在圆内
        if length_squared == 0:
            return self.check_collision(x1, y1)

        # 计算从线段起点到圆心的向量
        cx = self.x - x1
        cy = self.y - y1

        # 计算线段上最接近圆心的点的参数t
        t = max(0, min(1, (cx * dx + cy * dy) / length_squared))

        # 计算线段上最接近圆心的点
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # 计算最接近点到圆心的距离
        distance = math.sqrt((closest_x - self.x) ** 2 +
                             (closest_y - self.y) ** 2)

        # 如果距离小于等于半径，则线段与圆相交
        return distance <= self.radius


class DynamicObstacle:
    def __init__(self, x0, y0, vx, vy, width, height):
        """初始化动态障碍物
        参数:
            x0, y0: 初始位置
            vx, vy: 速度分量
            width, height: 宽度和高度
        """
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.width = width
        self.height = height

    def get_position_at_time(self, t):
        """计算在时间t时的位置"""
        x = self.x0 + self.vx * t
        y = self.y0 + self.vy * t
        return x, y


class Vehicle(RectangleObstacle):
    """
    车辆类，继承自矩形障碍物
    用于静态车辆表示和渲染
    """

    def __init__(self, x, y, length, width, orientation=0, color=(50, 50, 50, 230)):
        # 车辆是一个特殊的矩形障碍物
        super().__init__(x, y, length, width, orientation, color)
        self.length = length
        self.width = width
        self.orientation = orientation
        self.window_color = (150, 150, 150, 180)
        self.highlight_color = (200, 200, 200, 200)

        # 传感器配置
        self.sensors = {
            'fisheye_cameras': [],  # 环视摄像头 (黄色)
            'front_camera': None,   # 前视摄像头 (红色)
            'ultrasonic': [],       # 超声波雷达 (紫色)
            'imu': None,            # 消费级IMU (绿色)
            'gps': None             # 消费级GPS (绿色)
        }

        # 初始化传感器位置
        self._init_sensors()

    def _init_sensors(self):
        """初始化传感器位置"""
        half_length = self.length / 2
        half_width = self.width / 2

        # 环视摄像头 (4个，黄色)
        # 车头/车尾各2个，左/右后视镜各1个
        fisheye_positions = [
            (half_length, 0),           # 车头中央
            (-half_length, 0),          # 车尾中央
            (0, half_width),            # 右侧中央
            (0, -half_width)            # 左侧中央
        ]

        for pos in fisheye_positions:
            self.sensors['fisheye_cameras'].append({
                'local_pos': pos,
                'color': (255, 255, 0)  # 黄色
            })

        # 前视摄像头 (1个，红色)
        # 前挡风玻璃
        self.sensors['front_camera'] = {
            'local_pos': (half_length * 0.5, 0),
            'color': (255, 0, 0)  # 红色
        }

        # 超声波雷达 (12个，紫色)
        # 车前保4个(短距)，车后保4个(短距)，车眉处4个(长距)
        ultrasonic_positions = []

        # 前保险杠 (4个)
        front_spacing = half_width / 2
        for i in range(4):
            x = half_length
            y = -half_width + i * front_spacing
            ultrasonic_positions.append((x, y))

        # 后保险杠 (4个)
        rear_spacing = half_width / 2
        for i in range(4):
            x = -half_length
            y = -half_width + i * rear_spacing
            ultrasonic_positions.append((x, y))

        # 侧面 (4个)
        side_spacing = half_length / 2
        for i in range(2):
            # 左侧
            x = -half_length + i * side_spacing * 2
            y = -half_width
            ultrasonic_positions.append((x, y))

            # 右侧
            x = -half_length + i * side_spacing * 2
            y = half_width
            ultrasonic_positions.append((x, y))

        for pos in ultrasonic_positions:
            self.sensors['ultrasonic'].append({
                'local_pos': pos,
                'color': (128, 0, 128)  # 紫色
            })

        # 消费级IMU (1个，绿色)
        # 推荐嵌入摄像头
        self.sensors['imu'] = {
            'local_pos': (0, 0),
            'color': (0, 128, 0)  # 绿色
        }

        # 消费级GPS (1个，绿色)
        self.sensors['gps'] = {
            'local_pos': (0, half_width * 0.5),
            'color': (0, 200, 0)  # 浅绿色
        }

    def get_corners(self):
        """获取车辆四个角的坐标"""
        # 计算车辆四个角的局部坐标
        half_length = self.length / 2
        half_width = self.width / 2

        # 局部坐标系中的四个角
        corners_local = [
            (-half_length, -half_width),  # 左下
            (half_length, -half_width),   # 右下
            (half_length, half_width),    # 右上
            (-half_length, half_width)    # 左上
        ]

        # 将局部坐标转换为全局坐标
        angle_rad = math.radians(self.orientation)
        corners_global = []

        for x_local, y_local in corners_local:
            # 旋转
            x_rotated = x_local * \
                math.cos(angle_rad) - y_local * math.sin(angle_rad)
            y_rotated = x_local * \
                math.sin(angle_rad) + y_local * math.cos(angle_rad)

            # 平移
            x_global = x_rotated + self.x
            y_global = y_rotated + self.y

            corners_global.append((x_global, y_global))

        return corners_global

    def get_sensor_positions(self):
        """获取传感器的全局坐标位置"""
        cos_h = math.cos(self.orientation)
        sin_h = math.sin(self.orientation)

        sensor_positions = {
            'fisheye_cameras': [],
            'front_camera': None,
            'ultrasonic': [],
            'imu': None,
            'gps': None
        }

        # 环视摄像头
        for camera in self.sensors['fisheye_cameras']:
            lx, ly = camera['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['fisheye_cameras'].append({
                'pos': (x, y),
                'color': camera['color']
            })

        # 前视摄像头
        if self.sensors['front_camera']:
            lx, ly = self.sensors['front_camera']['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['front_camera'] = {
                'pos': (x, y),
                'color': self.sensors['front_camera']['color']
            }

        # 超声波雷达
        for sensor in self.sensors['ultrasonic']:
            lx, ly = sensor['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['ultrasonic'].append({
                'pos': (x, y),
                'color': sensor['color']
            })

        # IMU
        if self.sensors['imu']:
            lx, ly = self.sensors['imu']['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['imu'] = {
                'pos': (x, y),
                'color': self.sensors['imu']['color']
            }

        # GPS
        if self.sensors['gps']:
            lx, ly = self.sensors['gps']['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['gps'] = {
                'pos': (x, y),
                'color': self.sensors['gps']['color']
            }

        return sensor_positions
