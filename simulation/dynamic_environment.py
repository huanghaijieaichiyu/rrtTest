#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动态环境模块

提供用于RRT路径规划的动态城市环境，包括：
- 静态障碍物（建筑物、墙壁等）
- 动态障碍物（行人、车辆等）
- 交通信号灯
- 道路网络
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
from simulation.environment import Environment, CircleObstacle, RectangleObstacle


class TrafficLightState(Enum):
    """交通信号灯状态"""
    RED = 1
    YELLOW = 2
    GREEN = 3


@dataclass
class TrafficLight:
    """交通信号灯"""
    x: float
    y: float
    radius: float = 2.0
    state: TrafficLightState = TrafficLightState.RED
    cycle_time: Dict[TrafficLightState, float] = None
    last_change_time: float = 0.0

    def __post_init__(self):
        if self.cycle_time is None:
            # 默认周期：红灯30秒，绿灯25秒，黄灯5秒
            self.cycle_time = {
                TrafficLightState.RED: 30.0,
                TrafficLightState.GREEN: 25.0,
                TrafficLightState.YELLOW: 5.0
            }
            self.last_change_time = time.time()

    def update(self):
        """更新交通信号灯状态"""
        current_time = time.time()
        elapsed_time = current_time - self.last_change_time

        # 根据经过的时间切换状态
        if elapsed_time >= self.cycle_time[self.state]:
            if self.state == TrafficLightState.RED:
                self.state = TrafficLightState.GREEN
            elif self.state == TrafficLightState.GREEN:
                self.state = TrafficLightState.YELLOW
            else:  # YELLOW
                self.state = TrafficLightState.RED

            self.last_change_time = current_time


class MovementPattern(Enum):
    """移动模式"""
    LINEAR = 1       # 直线移动
    CIRCULAR = 2     # 圆形轨迹
    RANDOM = 3       # 随机移动
    WAYPOINT = 4     # 路径点移动


@dataclass
class MovingObstacle:
    """移动障碍物基类"""
    x: float
    y: float
    width: float
    height: float
    speed: float
    direction: float  # 移动方向（弧度）
    pattern: MovementPattern
    pattern_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.pattern_params is None:
            self.pattern_params = {}

        # 为不同的移动模式设置默认参数
        if self.pattern == MovementPattern.CIRCULAR:
            if 'center_x' not in self.pattern_params:
                self.pattern_params['center_x'] = self.x
            if 'center_y' not in self.pattern_params:
                self.pattern_params['center_y'] = self.y
            if 'radius' not in self.pattern_params:
                self.pattern_params['radius'] = 20.0
            if 'angle' not in self.pattern_params:
                self.pattern_params['angle'] = 0.0

        elif self.pattern == MovementPattern.WAYPOINT:
            if 'waypoints' not in self.pattern_params:
                self.pattern_params['waypoints'] = [(self.x, self.y)]
            if 'current_waypoint' not in self.pattern_params:
                self.pattern_params['current_waypoint'] = 0

        elif self.pattern == MovementPattern.RANDOM:
            if 'change_dir_prob' not in self.pattern_params:
                self.pattern_params['change_dir_prob'] = 0.05

    def update(self, dt: float, env_width: float, env_height: float):
        """
        更新障碍物位置

        参数:
            dt: 时间增量（秒）
            env_width: 环境宽度
            env_height: 环境高度
        """
        if self.pattern == MovementPattern.LINEAR:
            # 直线移动
            self.x += self.speed * np.cos(self.direction) * dt
            self.y += self.speed * np.sin(self.direction) * dt

            # 边界反射
            if self.x < 0 or self.x > env_width:
                self.direction = np.pi - self.direction
                self.x = np.clip(self.x, 0, env_width)
            if self.y < 0 or self.y > env_height:
                self.direction = -self.direction
                self.y = np.clip(self.y, 0, env_height)

        elif self.pattern == MovementPattern.CIRCULAR:
            # 圆形轨迹移动
            center_x = self.pattern_params['center_x']
            center_y = self.pattern_params['center_y']
            radius = self.pattern_params['radius']
            angle = self.pattern_params['angle']

            # 更新角度
            angle += (self.speed / radius) * dt
            angle = angle % (2 * np.pi)

            # 更新位置
            self.x = center_x + radius * np.cos(angle)
            self.y = center_y + radius * np.sin(angle)

            # 更新方向（切线方向）
            self.direction = angle + np.pi / 2

            # 更新参数
            self.pattern_params['angle'] = angle

        elif self.pattern == MovementPattern.RANDOM:
            # 随机移动
            change_dir_prob = self.pattern_params['change_dir_prob']

            # 有一定概率改变方向
            if np.random.random() < change_dir_prob:
                self.direction = np.random.uniform(0, 2 * np.pi)

            # 直线移动
            self.x += self.speed * np.cos(self.direction) * dt
            self.y += self.speed * np.sin(self.direction) * dt

            # 边界反射
            if self.x < 0 or self.x > env_width:
                self.direction = np.pi - self.direction
                self.x = np.clip(self.x, 0, env_width)
            if self.y < 0 or self.y > env_height:
                self.direction = -self.direction
                self.y = np.clip(self.y, 0, env_height)

        elif self.pattern == MovementPattern.WAYPOINT:
            # 路径点移动
            waypoints = self.pattern_params['waypoints']
            current_idx = self.pattern_params['current_waypoint']

            if len(waypoints) <= 1:
                return

            # 当前目标路径点
            target_x, target_y = waypoints[current_idx]

            # 计算到目标点的向量
            dx = target_x - self.x
            dy = target_y - self.y
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < 1.0:  # 到达当前路径点
                # 移动到下一个路径点
                current_idx = (current_idx + 1) % len(waypoints)
                self.pattern_params['current_waypoint'] = current_idx
                target_x, target_y = waypoints[current_idx]
                dx = target_x - self.x
                dy = target_y - self.y
                distance = np.sqrt(dx*dx + dy*dy)

            # 计算移动方向
            if distance > 0:
                self.direction = np.arctan2(dy, dx)

            # 移动
            move_dist = min(self.speed * dt, distance)
            self.x += move_dist * np.cos(self.direction)
            self.y += move_dist * np.sin(self.direction)


@dataclass
class Pedestrian(MovingObstacle):
    """行人"""

    def __init__(
        self,
        x: float,
        y: float,
        width: float = 1.0,
        height: float = 1.0,
        speed: float = 1.5,
        direction: float = 0.0,
        pattern: MovementPattern = MovementPattern.RANDOM,
        pattern_params: Dict[str, Any] = None
    ):
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            speed=speed,
            direction=direction,
            pattern=pattern,
            pattern_params=pattern_params
        )


@dataclass
class Vehicle(MovingObstacle):
    """车辆"""

    def __init__(
        self,
        x: float,
        y: float,
        width: float = 4.0,  # 车宽
        height: float = 2.0,  # 车长
        speed: float = 5.0,
        direction: float = 0.0,
        pattern: MovementPattern = MovementPattern.LINEAR,
        pattern_params: Dict[str, Any] = None
    ):
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            speed=speed,
            direction=direction,
            pattern=pattern,
            pattern_params=pattern_params
        )


@dataclass
class MovingTarget(MovingObstacle):
    """移动目标 - 用于跟随的目标点"""

    def __init__(
        self,
        x: float,
        y: float,
        width: float = 2.0,
        height: float = 2.0,
        speed: float = 3.0,
        direction: float = 0.0,
        pattern: MovementPattern = MovementPattern.CIRCULAR,
        pattern_params: Dict[str, Any] = None,
        color: Tuple[int, int, int] = (0, 255, 0)  # 默认绿色
    ):
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            speed=speed,
            direction=direction,
            pattern=pattern,
            pattern_params=pattern_params
        )
        self.color = color  # 目标颜色
        self.is_target = True  # 标记为目标


class DynamicEnvironment(Environment):
    """动态环境类，扩展了基本环境类，添加动态障碍物支持"""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        map_path: Optional[str] = None
    ):
        super().__init__(width, height, map_path)
        self.moving_obstacles: List[MovingObstacle] = []
        self.traffic_lights: List[TrafficLight] = []
        self.last_update_time = time.time()
        self.moving_target: Optional[MovingTarget] = None  # 移动目标

    def add_pedestrian(
        self,
        x: float,
        y: float,
        speed: float = 1.5,
        direction: float = 0.0,
        pattern: MovementPattern = MovementPattern.RANDOM,
        pattern_params: Dict[str, Any] = None
    ) -> None:
        """
        添加行人

        参数:
            x: x坐标
            y: y坐标
            speed: 移动速度
            direction: 移动方向（弧度）
            pattern: 移动模式
            pattern_params: 移动模式参数
        """
        pedestrian = Pedestrian(
            x=x,
            y=y,
            speed=speed,
            direction=direction,
            pattern=pattern,
            pattern_params=pattern_params
        )
        self.moving_obstacles.append(pedestrian)

    def add_vehicle(
        self,
        x: float,
        y: float,
        width: float = 4.0,
        height: float = 2.0,
        speed: float = 5.0,
        direction: float = 0.0,
        pattern: MovementPattern = MovementPattern.LINEAR,
        pattern_params: Dict[str, Any] = None
    ) -> None:
        """
        添加车辆

        参数:
            x: x坐标
            y: y坐标
            width: 车宽
            height: 车长
            speed: 移动速度
            direction: 移动方向（弧度）
            pattern: 移动模式
            pattern_params: 移动模式参数
        """
        vehicle = Vehicle(
            x=x,
            y=y,
            width=width,
            height=height,
            speed=speed,
            direction=direction,
            pattern=pattern,
            pattern_params=pattern_params
        )
        self.moving_obstacles.append(vehicle)

    def add_traffic_light(
        self,
        x: float,
        y: float,
        radius: float = 2.0,
        state: TrafficLightState = TrafficLightState.RED,
        cycle_time: Dict[TrafficLightState, float] = None
    ) -> None:
        """
        添加交通信号灯

        参数:
            x: x坐标
            y: y坐标
            radius: 半径
            state: 初始状态
            cycle_time: 周期时间，例如 {RED: 30, GREEN: 25, YELLOW: 5}
        """
        traffic_light = TrafficLight(
            x=x,
            y=y,
            radius=radius,
            state=state,
            cycle_time=cycle_time
        )
        self.traffic_lights.append(traffic_light)

    def add_moving_target(
        self,
        x: float,
        y: float,
        speed: float = 3.0,
        pattern: MovementPattern = MovementPattern.CIRCULAR,
        pattern_params: Dict[str, Any] = None,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> MovingTarget:
        """
        添加移动目标

        参数:
            x: x坐标
            y: y坐标
            speed: 移动速度
            pattern: 移动模式
            pattern_params: 移动模式参数
            color: 目标颜色

        返回:
            创建的移动目标对象
        """
        # 如果已有移动目标，先移除
        if self.moving_target:
            if self.moving_target in self.moving_obstacles:
                self.moving_obstacles.remove(self.moving_target)
            self.moving_target = None

        # 创建新的移动目标
        target = MovingTarget(
            x=x,
            y=y,
            speed=speed,
            pattern=pattern,
            pattern_params=pattern_params,
            color=color
        )
        self.moving_obstacles.append(target)
        self.moving_target = target
        return target

    def get_moving_target_position(self) -> Optional[Tuple[float, float]]:
        """
        获取移动目标的当前位置

        返回:
            移动目标的坐标 (x, y)，如果没有移动目标则返回None
        """
        if self.moving_target:
            return (self.moving_target.x, self.moving_target.y)
        return None

    def update(self, dt: Optional[float] = None) -> None:
        """
        更新环境中所有动态对象

        参数:
            dt: 时间增量，如果为None则使用当前时间与上次更新时间之差
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update_time

        # 更新所有移动障碍物
        for obstacle in self.moving_obstacles:
            obstacle.update(dt, self.width, self.height)

        # 更新所有交通信号灯
        for light in self.traffic_lights:
            light.update()

        self.last_update_time = current_time

    def check_collision_with_moving_obstacles(self, point: Tuple[float, float]) -> bool:
        """
        检查点是否与移动障碍物碰撞

        参数:
            point: 待检查的点坐标 (x, y)

        返回:
            是否发生碰撞
        """
        x, y = point

        for obstacle in self.moving_obstacles:
            # 简化为矩形碰撞检测
            if (abs(x - obstacle.x) <= obstacle.width / 2 and
                    abs(y - obstacle.y) <= obstacle.height / 2):
                return True

        return False

    def check_collision_with_traffic_lights(
        self,
        point: Tuple[float, float],
        respect_red_light: bool = True
    ) -> bool:
        """
        检查点是否与交通信号灯碰撞

        参数:
            point: 待检查的点坐标 (x, y)
            respect_red_light: 是否遵守红灯

        返回:
            是否发生碰撞
        """
        x, y = point

        for light in self.traffic_lights:
            # 检查是否在信号灯范围内
            dist = np.sqrt((x - light.x) ** 2 + (y - light.y) ** 2)
            if dist <= light.radius:
                return True

            # 如果需要遵守红灯
            if respect_red_light and light.state == TrafficLightState.RED:
                # 检查是否在红灯影响区域
                # 这里简化为红灯前方10米范围
                influence_radius = 10.0
                if dist <= influence_radius:
                    return True

        return False

    def check_collision(
        self,
        point: Tuple[float, float],
        respect_red_light: bool = True
    ) -> bool:
        """
        检查点是否与任何障碍物（静态或动态）碰撞

        参数:
            point: 待检查的点坐标 (x, y)
            respect_red_light: 是否遵守红灯

        返回:
            是否发生碰撞
        """
        # 首先检查是否与静态障碍物碰撞
        if super().check_collision(point):
            return True

        # 然后检查是否与移动障碍物碰撞
        if self.check_collision_with_moving_obstacles(point):
            return True

        # 最后检查是否与交通信号灯碰撞
        if self.check_collision_with_traffic_lights(point, respect_red_light):
            return True

        return False

    def create_urban_environment(self) -> None:
        """创建城市环境，包括道路、建筑物、十字路口等"""
        # 清空现有障碍物
        self.obstacles.clear()
        self.moving_obstacles.clear()
        self.traffic_lights.clear()

        # 设置边界
        width, height = self.width, self.height

        # 添加建筑物（用矩形表示）
        # 左上角的建筑群
        self.add_obstacle(
            x=15, y=15, obstacle_type="rectangle", width=20, height=20)
        self.add_obstacle(
            x=15, y=45, obstacle_type="rectangle", width=20, height=20)

        # 右上角的建筑群
        self.add_obstacle(x=width-15, y=15,
                          obstacle_type="rectangle", width=20, height=20)
        self.add_obstacle(x=width-15, y=45,
                          obstacle_type="rectangle", width=20, height=20)

        # 左下角的建筑群
        self.add_obstacle(x=15, y=height-15,
                          obstacle_type="rectangle", width=20, height=20)
        self.add_obstacle(x=15, y=height-45,
                          obstacle_type="rectangle", width=20, height=20)

        # 右下角的建筑群
        self.add_obstacle(x=width-15, y=height-15,
                          obstacle_type="rectangle", width=20, height=20)
        self.add_obstacle(x=width-15, y=height-45,
                          obstacle_type="rectangle", width=20, height=20)

        # 中央的建筑物
        self.add_obstacle(x=width/2, y=height/2,
                          obstacle_type="rectangle", width=10, height=10)

        # 添加交通信号灯（在十字路口处）
        self.add_traffic_light(x=width/2-15, y=height /
                               2-15, state=TrafficLightState.RED)
        self.add_traffic_light(x=width/2+15, y=height /
                               2-15, state=TrafficLightState.GREEN)
        self.add_traffic_light(x=width/2-15, y=height /
                               2+15, state=TrafficLightState.GREEN)
        self.add_traffic_light(x=width/2+15, y=height /
                               2+15, state=TrafficLightState.RED)

        # 添加行人
        # 随机在环境中添加10个行人
        for _ in range(10):
            x = np.random.uniform(10, width-10)
            y = np.random.uniform(10, height-10)
            # 确保行人不在建筑物内
            while self.check_collision((x, y)):
                x = np.random.uniform(10, width-10)
                y = np.random.uniform(10, height-10)
            self.add_pedestrian(
                x=x,
                y=y,
                speed=np.random.uniform(0.8, 2.0),
                direction=np.random.uniform(0, 2*np.pi)
            )

        # 添加车辆
        # 水平方向的车辆
        for i in range(3):
            self.add_vehicle(
                x=np.random.uniform(30, width-30),
                y=height/4,
                speed=np.random.uniform(3.0, 7.0),
                direction=0 if np.random.random() > 0.5 else np.pi
            )

        # 垂直方向的车辆
        for i in range(3):
            self.add_vehicle(
                x=width/4,
                y=np.random.uniform(30, height-30),
                speed=np.random.uniform(3.0, 7.0),
                direction=np.pi/2 if np.random.random() > 0.5 else 3*np.pi/2
            )

        # 圆形轨迹的车辆
        self.add_vehicle(
            x=width/2,
            y=height/2 + 25,
            speed=5.0,
            pattern=MovementPattern.CIRCULAR,
            pattern_params={
                'center_x': width/2,
                'center_y': height/2,
                'radius': 25.0
            }
        )

        # 路径点车辆
        self.add_vehicle(
            x=width/2-25,
            y=height/2,
            speed=5.0,
            pattern=MovementPattern.WAYPOINT,
            pattern_params={
                'waypoints': [
                    (width/2-25, height/2),
                    (width/2, height/2-25),
                    (width/2+25, height/2),
                    (width/2, height/2+25)
                ]
            }
        )
