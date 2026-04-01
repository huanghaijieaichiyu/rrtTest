#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享障碍物模型。

公共角度语义统一使用“度”，仅在几何计算时局部转换为弧度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from shapely.geometry import LineString, Point, Polygon


class Obstacle:
    """障碍物基类。"""

    def __init__(
        self,
        x: float,
        y: float,
        color: Tuple[int, ...] = (0, 0, 0, 255),
        is_filled: bool = True,
        line_width: int = 2,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.is_filled = is_filled
        self.line_width = line_width
        self.is_parking_spot = False
        self.occupied = False

    @property
    def blocks_motion(self) -> bool:
        """未占用停车位在碰撞检测中应视为可通行。"""
        return not (self.is_parking_spot and not self.occupied)

    def to_geometry(self) -> Polygon:
        raise NotImplementedError("子类必须实现 to_geometry")

    def check_collision(self, x: float, y: float) -> bool:
        if not self.blocks_motion:
            return False
        return self.to_geometry().covers(Point(float(x), float(y)))

    def check_line_collision(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        if not self.blocks_motion:
            return False
        return self.to_geometry().intersects(
            LineString([(float(x1), float(y1)), (float(x2), float(y2))])
        )

    def distance_to_point(self, x: float, y: float) -> float:
        if not self.blocks_motion:
            return float("inf")
        return float(self.to_geometry().distance(Point(float(x), float(y))))


class RectangleObstacle(Obstacle):
    """矩形障碍物，角度单位为度。"""

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        angle: float = 0.0,
        color: Tuple[int, ...] = (0, 0, 0, 255),
        is_filled: bool = True,
        line_width: int = 2,
    ) -> None:
        super().__init__(x, y, color, is_filled, line_width)
        self.width = float(width)
        self.height = float(height)
        self.angle = float(angle)
        self.type = "rectangle"
        self.radius = 0.0

    @property
    def angle_rad(self) -> float:
        return math.radians(self.angle)

    def to_corners(self) -> List[Tuple[float, float]]:
        half_width = self.width / 2
        half_height = self.height / 2
        cos_angle = math.cos(self.angle_rad)
        sin_angle = math.sin(self.angle_rad)
        local_corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]

        corners: List[Tuple[float, float]] = []
        for local_x, local_y in local_corners:
            world_x = self.x + local_x * cos_angle - local_y * sin_angle
            world_y = self.y + local_x * sin_angle + local_y * cos_angle
            corners.append((world_x, world_y))
        return corners

    def to_geometry(self) -> Polygon:
        return Polygon(self.to_corners())


class CircleObstacle(Obstacle):
    """圆形障碍物。"""

    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        color: Tuple[int, ...] = (0, 0, 0, 255),
        is_filled: bool = True,
        line_width: int = 2,
    ) -> None:
        super().__init__(x, y, color, is_filled, line_width)
        self.radius = float(radius)
        self.type = "circle"
        self.width = self.radius * 2
        self.height = self.radius * 2
        self.angle = 0.0

    def to_geometry(self) -> Polygon:
        return Point(self.x, self.y).buffer(self.radius)


@dataclass
class DynamicObstacle:
    """简单匀速动态障碍物。"""

    x0: float
    y0: float
    vx: float
    vy: float
    width: float
    height: float

    def get_position_at_time(self, t: float) -> Tuple[float, float]:
        return self.x0 + self.vx * t, self.y0 + self.vy * t


class Vehicle(RectangleObstacle):
    """
    用于静态渲染/分析的车辆表示。

    这里保留旧接口，但统一使用 orientation 的“度”语义。
    """

    def __init__(
        self,
        x: float,
        y: float,
        length: float,
        width: float,
        orientation: float = 0.0,
        color: Tuple[int, ...] = (50, 50, 50, 230),
    ) -> None:
        super().__init__(x, y, length, width, orientation, color)
        self.length = float(length)
        self.width = float(width)
        self.orientation = float(orientation)
        self.window_color = (150, 150, 150, 180)
        self.highlight_color = (200, 200, 200, 200)
        self.sensors: Dict[str, object] = {
            "fisheye_cameras": [],
            "front_camera": None,
            "ultrasonic": [],
            "imu": None,
            "gps": None,
        }
        self._init_sensors()

    def _init_sensors(self) -> None:
        half_length = self.length / 2
        half_width = self.width / 2

        for position in (
            (half_length, 0),
            (-half_length, 0),
            (0, half_width),
            (0, -half_width),
        ):
            self.sensors["fisheye_cameras"].append(  # type: ignore[union-attr]
                {"local_pos": position, "color": (255, 255, 0)}
            )

        self.sensors["front_camera"] = {  # type: ignore[index]
            "local_pos": (half_length * 0.5, 0),
            "color": (255, 0, 0),
        }

        ultrasonic_positions = []
        front_spacing = half_width / 2
        for index in range(4):
            ultrasonic_positions.append((half_length, -half_width + index * front_spacing))
            ultrasonic_positions.append((-half_length, -half_width + index * front_spacing))

        side_spacing = half_length / 2
        for index in range(2):
            ultrasonic_positions.append((-half_length + index * side_spacing * 2, -half_width))
            ultrasonic_positions.append((-half_length + index * side_spacing * 2, half_width))

        for position in ultrasonic_positions:
            self.sensors["ultrasonic"].append(  # type: ignore[union-attr]
                {"local_pos": position, "color": (128, 0, 128)}
            )

        self.sensors["imu"] = {"local_pos": (0, 0), "color": (0, 128, 0)}  # type: ignore[index]
        self.sensors["gps"] = {  # type: ignore[index]
            "local_pos": (0, half_width * 0.5),
            "color": (0, 200, 0),
        }

    def get_corners(self) -> List[Tuple[float, float]]:
        self.angle = self.orientation
        return self.to_corners()

    def get_sensor_positions(self) -> Dict[str, object]:
        angle_rad = math.radians(self.orientation)
        cos_heading = math.cos(angle_rad)
        sin_heading = math.sin(angle_rad)

        sensor_positions: Dict[str, object] = {
            "fisheye_cameras": [],
            "front_camera": None,
            "ultrasonic": [],
            "imu": None,
            "gps": None,
        }

        for key in ("fisheye_cameras", "ultrasonic"):
            for sensor in self.sensors[key]:  # type: ignore[index]
                local_x, local_y = sensor["local_pos"]
                x = self.x + local_x * cos_heading - local_y * sin_heading
                y = self.y + local_x * sin_heading + local_y * cos_heading
                sensor_positions[key].append({"pos": (x, y), "color": sensor["color"]})  # type: ignore[index]

        for key in ("front_camera", "imu", "gps"):
            sensor = self.sensors[key]  # type: ignore[index]
            if sensor is None:
                continue
            local_x, local_y = sensor["local_pos"]
            x = self.x + local_x * cos_heading - local_y * sin_heading
            y = self.y + local_x * sin_heading + local_y * cos_heading
            sensor_positions[key] = {"pos": (x, y), "color": sensor["color"]}

        return sensor_positions

