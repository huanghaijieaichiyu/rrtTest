#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境定义模块。

提供：
- 环境边界
- 障碍物管理
- 碰撞检测
- 栅格化表示

公共角度语义统一使用“度”，内部几何计算按需转换为弧度。
"""

from __future__ import annotations

import json
import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml
from shapely.geometry import LineString, Point, Polygon

from .obstacles import CircleObstacle, DynamicObstacle, RectangleObstacle, Vehicle

ObstacleLike = Union[CircleObstacle, RectangleObstacle]


class Environment:
    """
    路径规划环境类。

    提供环境表示、碰撞检测和可视化功能。
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        map_path: Optional[str] = None,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.obstacles: List[ObstacleLike] = []

        if map_path:
            self.load_map(map_path)

    def get_min_distance(self, x: float, y: float) -> float:
        """计算点到所有阻挡型障碍物的最小距离。"""
        if not self.obstacles:
            return float("inf")

        min_distance = float("inf")
        for obstacle in self.obstacles:
            min_distance = min(min_distance, obstacle.distance_to_point(x, y))
        return min_distance

    def add_obstacle(
        self,
        x: float,
        y: float,
        obstacle_type: str = "circle",
        radius: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        angle: float = 0.0,
    ) -> None:
        """
        添加障碍物。

        参数:
            x: 障碍物中心 x 坐标
            y: 障碍物中心 y 坐标
            obstacle_type: 障碍物类型 ("circle" 或 "rectangle")
            radius: 圆形障碍物半径
            width: 矩形障碍物宽度
            height: 矩形障碍物高度
            angle: 矩形障碍物旋转角度（度）
        """
        if obstacle_type == "circle" and radius is not None:
            self.obstacles.append(CircleObstacle(x=x, y=y, radius=radius))
            return
        if obstacle_type == "rectangle" and width is not None and height is not None:
            self.obstacles.append(
                RectangleObstacle(x=x, y=y, width=width, height=height, angle=angle)
            )
            return
        raise ValueError("无效的障碍物参数")

    def _point_in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def _polygon_within_bounds(self, polygon: Polygon) -> bool:
        min_x, min_y, max_x, max_y = polygon.bounds
        return min_x >= 0.0 and min_y >= 0.0 and max_x <= self.width and max_y <= self.height

    def _vehicle_polygon(
        self,
        x: float,
        y: float,
        heading: float,
        vehicle_width: float,
        vehicle_length: float,
    ) -> Polygon:
        half_length = vehicle_length / 2
        half_width = vehicle_width / 2
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)
        corners = [
            (
                x + half_length * cos_heading - half_width * sin_heading,
                y + half_length * sin_heading + half_width * cos_heading,
            ),
            (
                x + half_length * cos_heading + half_width * sin_heading,
                y + half_length * sin_heading - half_width * cos_heading,
            ),
            (
                x - half_length * cos_heading + half_width * sin_heading,
                y - half_length * sin_heading - half_width * cos_heading,
            ),
            (
                x - half_length * cos_heading - half_width * sin_heading,
                y - half_length * sin_heading + half_width * cos_heading,
            ),
        ]
        return Polygon(corners)

    def check_collision(
        self,
        point: Tuple[float, float],
        vehicle_width: float = 0.0,
        vehicle_length: float = 0.0,
    ) -> bool:
        """
        检查点或车辆占位是否与障碍物碰撞。
        """
        x, y = point
        if not self._point_in_bounds(x, y):
            return True

        if vehicle_width > 0 and vehicle_length > 0:
            vehicle_polygon = self._vehicle_polygon(x, y, 0.0, vehicle_width, vehicle_length)
            if not self._polygon_within_bounds(vehicle_polygon):
                return True

            for obstacle in self.obstacles:
                if not obstacle.blocks_motion:
                    continue
                if vehicle_polygon.intersects(obstacle.to_geometry()):
                    return True
            return False

        for obstacle in self.obstacles:
            if obstacle.check_collision(x, y):
                return True
        return False

    def check_segment_collision(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        vehicle_width: float = 0.0,
        vehicle_length: float = 0.0,
    ) -> bool:
        """
        检查线段是否与任意障碍物碰撞，支持车辆尺寸。
        """
        if not self._point_in_bounds(*start) or not self._point_in_bounds(*end):
            return True

        if vehicle_width <= 0 or vehicle_length <= 0:
            line = LineString([start, end])
            for obstacle in self.obstacles:
                if not obstacle.blocks_motion:
                    continue
                if line.intersects(obstacle.to_geometry()):
                    return True
            return False

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        path_length = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) if path_length > 1e-9 else 0.0
        steps = 1 if path_length <= 1e-9 else max(3, int(path_length / max(vehicle_width / 2, 0.5)) + 1)

        for index in range(steps):
            t = 0.0 if steps == 1 else index / (steps - 1)
            x = start[0] + t * dx
            y = start[1] + t * dy
            vehicle_polygon = self._vehicle_polygon(x, y, heading, vehicle_width, vehicle_length)

            if not self._polygon_within_bounds(vehicle_polygon):
                return True

            for obstacle in self.obstacles:
                if not obstacle.blocks_motion:
                    continue
                if vehicle_polygon.intersects(obstacle.to_geometry()):
                    return True

        return False

    def to_grid(self, grid_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """将环境转换为栅格表示。"""
        grid_width, grid_height = grid_size
        grid = np.zeros((grid_height, grid_width))
        resolution_x = self.width / grid_width
        resolution_y = self.height / grid_height

        for row in range(grid_height):
            for column in range(grid_width):
                x = (column + 0.5) * resolution_x
                y = (row + 0.5) * resolution_y
                if self.check_collision((x, y)):
                    grid[row, column] = 1

        return grid

    def load_map(self, map_path: str) -> None:
        """从 YAML 文件加载地图。"""
        try:
            with open(map_path, "r", encoding="utf-8") as handle:
                map_data = yaml.safe_load(handle) or {}

            self.width = float(map_data.get("width", self.width))
            self.height = float(map_data.get("height", self.height))
            self.obstacles.clear()

            for obstacle_data in map_data.get("obstacles", []):
                self.add_obstacle(**obstacle_data)
        except Exception as exc:
            print(f"加载地图失败: {exc}")

    def save_map(self, map_path: str) -> None:
        """保存地图到 YAML 文件。"""
        map_data = {
            "width": self.width,
            "height": self.height,
            "obstacles": [],
        }

        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                map_data["obstacles"].append(
                    {
                        "x": obstacle.x,
                        "y": obstacle.y,
                        "obstacle_type": "circle",
                        "radius": obstacle.radius,
                    }
                )
            elif isinstance(obstacle, RectangleObstacle):
                map_data["obstacles"].append(
                    {
                        "x": obstacle.x,
                        "y": obstacle.y,
                        "obstacle_type": "rectangle",
                        "width": obstacle.width,
                        "height": obstacle.height,
                        "angle": obstacle.angle,
                    }
                )

        try:
            with open(map_path, "w", encoding="utf-8") as handle:
                yaml.dump(map_data, handle, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            print(f"保存地图失败: {exc}")

    def plot_obstacles(self, ax) -> None:
        """绘制所有障碍物。"""
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                circle = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, color="r", alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obstacle, RectangleObstacle):
                polygon = plt.Polygon(self._compute_corners(obstacle), closed=True, color="r", alpha=0.5)
                ax.add_patch(polygon)

    def visualize(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """可视化整个环境。"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0], "k-")
        self.plot_obstacles(ax)
        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title("环境地图")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()

    def visualize_path(self, path: List[Tuple[float, float]], figsize: Tuple[int, int] = (10, 8)) -> None:
        """可视化路径。"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0], "k-")
        self.plot_obstacles(ax)

        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            ax.plot(path_x, path_y, "-b", linewidth=2, label="路径")
            ax.plot(path_x[0], path_y[0], "go", markersize=10, label="起点")
            ax.plot(path_x[-1], path_y[-1], "ro", markersize=10, label="终点")

        ax.set_xlim(-5, self.width + 5)
        ax.set_ylim(-5, self.height + 5)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title("规划路径")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.show()

    def save(self, filepath: str) -> None:
        """保存环境到 JSON 文件。"""
        obstacles_payload = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                obstacles_payload.append(
                    {
                        "type": "circle",
                        "x": obstacle.x,
                        "y": obstacle.y,
                        "radius": obstacle.radius,
                    }
                )
            elif isinstance(obstacle, RectangleObstacle):
                obstacles_payload.append(
                    {
                        "type": "rectangle",
                        "x": obstacle.x,
                        "y": obstacle.y,
                        "width": obstacle.width,
                        "height": obstacle.height,
                        "angle": obstacle.angle,
                    }
                )

        payload = {
            "width": self.width,
            "height": self.height,
            "obstacles": obstacles_payload,
        }

        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Environment":
        """从 JSON 文件加载环境。"""
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        environment = cls(width=payload["width"], height=payload["height"])
        for obstacle_data in payload["obstacles"]:
            if obstacle_data["type"] == "circle":
                environment.add_obstacle(
                    x=obstacle_data["x"],
                    y=obstacle_data["y"],
                    obstacle_type="circle",
                    radius=obstacle_data["radius"],
                )
            else:
                environment.add_obstacle(
                    x=obstacle_data["x"],
                    y=obstacle_data["y"],
                    obstacle_type="rectangle",
                    width=obstacle_data["width"],
                    height=obstacle_data["height"],
                    angle=obstacle_data.get("angle", 0.0),
                )
        return environment

    def _compute_corners(self, obstacle: RectangleObstacle) -> List[Tuple[float, float]]:
        """计算矩形的四个角点坐标。"""
        return obstacle.to_corners()


__all__ = [
    "CircleObstacle",
    "DynamicObstacle",
    "Environment",
    "RectangleObstacle",
    "Vehicle",
]

