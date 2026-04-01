#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pygame 车辆仿真器

使用 Pygame 实现简单的车辆动力学模型和仿真环境，
作为 CarSim 的轻量级替代方案。
"""

import os
import math
import platform
import sys
import time
import matplotlib
import numpy as np
import pygame
import yaml
from typing import List, Tuple, Dict, Optional, Union, Any
from shapely.geometry import Point, Polygon, LineString

from .environment import Environment
from .obstacles import CircleObstacle, DynamicObstacle, RectangleObstacle

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# 字体设置


_FONT_CACHE: Dict[Tuple[int, str], pygame.font.Font] = {}
_FONT_RESOLUTION: Dict[str, Dict[str, str]] = {}
_FONT_OVERRIDE_CANDIDATES: Dict[str, Dict[str, List[str]]] = {}
_FONT_SAMPLE_TEXTS = (
    "停车位姿A9",
    "目标车位",
    "状态提示",
)
_FONT_PLACEHOLDER_SAMPLES = ("□□□□", "????", "口口口口")
_BUNDLED_FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")
_DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")
_DEFAULT_LOG_FILE = os.path.join(_DEFAULT_LOG_DIR, "parking_demo.log")
_FONT_ROLE_CANDIDATES = {
    "ui": {
        "windows": [
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "SimHei",
            "DengXian",
            "SimSun",
        ],
        "linux": [
            "Noto Sans CJK SC",
            "WenQuanYi Micro Hei",
            "Source Han Sans SC",
            "Droid Sans Fallback",
        ],
        "darwin": [
            "PingFang SC",
            "Hiragino Sans GB",
            "STHeiti",
        ],
        "generic": ["Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"],
        "bundled": [
            "NotoSansCJKsc-Regular.otf",
        ],
    },
    "title": {
        "windows": [
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "DengXian",
            "SimHei",
        ],
        "linux": [
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Micro Hei",
        ],
        "darwin": ["PingFang SC", "STKaiti"],
        "generic": ["Noto Sans CJK SC", "Source Han Sans SC"],
        "bundled": ["STKAITI.TTF", "NotoSansCJKsc-Regular.otf"],
    },
    "mono": {
        "windows": [
            "Sarasa Mono SC",
            "Microsoft YaHei UI",
            "Consolas",
        ],
        "linux": [
            "Sarasa Mono SC",
            "WenQuanYi Zen Hei Mono",
            "Noto Sans Mono CJK SC",
            "Noto Sans CJK SC",
        ],
        "darwin": [
            "PingFang SC",
            "Menlo",
        ],
        "generic": ["Sarasa Mono SC", "Noto Sans CJK SC", "DejaVu Sans Mono"],
        "bundled": [
            "NotoSansCJKsc-Regular.otf",
        ],
    },
}


def _merge_unique(items: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        merged.append(item)
        seen.add(item)
    return merged


def _coerce_font_entries(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def configure_font_preferences(font_config: Optional[Dict[str, Any]] = None) -> None:
    """根据配置覆盖字体候选列表。"""
    global _FONT_OVERRIDE_CANDIDATES

    font_config = font_config or {}
    if not isinstance(font_config, dict):
        return

    overrides: Dict[str, Dict[str, List[str]]] = {}
    for role in ("ui", "title", "mono"):
        role_config: Any = font_config.get(role, {})
        if role == "ui" and not role_config and any(
            key in font_config for key in ("preferred", "names", "system", "bundled", "files")
        ):
            role_config = font_config

        if isinstance(role_config, (str, list, tuple)):
            role_config = {"preferred": role_config}
        if not isinstance(role_config, dict):
            continue

        names = _merge_unique(
            _coerce_font_entries(role_config.get("preferred"))
            + _coerce_font_entries(role_config.get("names"))
            + _coerce_font_entries(role_config.get("system"))
        )
        bundled = _merge_unique(
            _coerce_font_entries(role_config.get("bundled"))
            + _coerce_font_entries(role_config.get("files"))
        )
        if names or bundled:
            overrides[role] = {"names": names, "bundled": bundled}

    if overrides == _FONT_OVERRIDE_CANDIDATES:
        return

    _FONT_OVERRIDE_CANDIDATES = overrides
    _FONT_CACHE.clear()
    _FONT_RESOLUTION.clear()


def _font_signature(font: pygame.font.Font, sample_text: str) -> Tuple[Tuple[int, int], int, int]:
    surface = font.render(sample_text, True, WHITE)
    alpha = pygame.surfarray.array_alpha(surface)
    return alpha.shape, int(alpha.sum()), int(np.count_nonzero(alpha))


def _font_supports_chinese(font: pygame.font.Font) -> bool:
    try:
        placeholder_signatures = {
            _font_signature(font, placeholder)
            for placeholder in _FONT_PLACEHOLDER_SAMPLES
        }
        for sample_text in _FONT_SAMPLE_TEXTS:
            sample_signature = _font_signature(font, sample_text)
            if sample_signature[1] <= 0 or sample_signature[2] <= 0:
                return False
            if sample_signature in placeholder_signatures:
                return False
        return True
    except Exception:
        return False


def _font_platform_key() -> str:
    system_name = platform.system().lower()
    if system_name.startswith("win"):
        return "windows"
    if system_name.startswith("darwin"):
        return "darwin"
    return "linux"


def _candidate_font_names(role: str) -> List[str]:
    role_config = _FONT_ROLE_CANDIDATES.get(role, _FONT_ROLE_CANDIDATES["ui"])
    platform_key = _font_platform_key()
    override = _FONT_OVERRIDE_CANDIDATES.get(role, {})
    return _merge_unique(
        override.get("names", [])
        + role_config.get(platform_key, [])
        + role_config.get("generic", [])
    )


def _candidate_font_files(role: str) -> List[str]:
    role_config = _FONT_ROLE_CANDIDATES.get(role, _FONT_ROLE_CANDIDATES["ui"])
    override = _FONT_OVERRIDE_CANDIDATES.get(role, {})
    files = override.get("bundled", []) + role_config.get("bundled", [])
    resolved: List[str] = []
    for filename in _merge_unique(files):
        resolved.append(filename if os.path.isabs(filename) else os.path.join(_BUNDLED_FONTS_DIR, filename))
    return resolved


def get_font_resolution(role: str = "ui") -> Dict[str, str]:
    """返回指定字体角色的解析结果。"""
    if role not in _FONT_RESOLUTION:
        get_font(16, role=role)
    return dict(_FONT_RESOLUTION.get(role, {}))


def get_font(size: int = 24, role: str = "ui") -> pygame.font.Font:
    """按角色解析字体，并优先选择可正常渲染中文的字体。"""
    cache_key = (size, role)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    if not pygame.font.get_init():
        pygame.font.init()

    for font_name in _candidate_font_names(role):
        matched_path = None
        try:
            matched_path = pygame.font.match_font(font_name)
        except Exception:
            matched_path = None
        try:
            font = (
                pygame.font.Font(matched_path, size)
                if matched_path and os.path.exists(matched_path)
                else pygame.font.SysFont(font_name, size)
            )
        except Exception:
            continue
        if not _font_supports_chinese(font):
            continue
        _FONT_CACHE[cache_key] = font
        _FONT_RESOLUTION[role] = {
            "role": role,
            "source": "system",
            "name": font_name,
            "path": matched_path or "",
        }
        return font

    for font_path in _candidate_font_files(role):
        if not os.path.exists(font_path):
            continue
        try:
            font = pygame.font.Font(font_path, size)
        except Exception:
            continue
        if not _font_supports_chinese(font):
            continue
        _FONT_CACHE[cache_key] = font
        _FONT_RESOLUTION[role] = {
            "role": role,
            "source": "bundled",
            "name": os.path.basename(font_path),
            "path": font_path,
        }
        return font

    fallback_font = pygame.font.Font(None, size)
    _FONT_CACHE[cache_key] = fallback_font
    _FONT_RESOLUTION[role] = {
        "role": role,
        "source": "pygame-default",
        "name": "default",
    }
    return fallback_font


class VehicleModel:
    """增强的车辆动力学模型 - 四轮转向(4WS)模型，带传感器"""

    def __init__(self, x: float = 0, y: float = 0, heading: float = 0, length: float = 4.5, width: float = 1.8):
        """
        初始化车辆模型

        参数:
            x: 初始x坐标
            y: 初始y坐标
            heading: 初始朝向角度(弧度)
            length: 车辆长度(米)
            width: 车辆宽度(米)
        """
        self.x = x
        self.y = y
        self.heading = heading  # 弧度
        self.length = length
        self.width = width
        self.speed = 0.0  # m/s
        self.acceleration = 0.0  # m/s^2
        self.front_steer_angle = 0.0  # 前轮转向角(弧度)
        self.rear_steer_angle = 0.0  # 后轮转向角(弧度)
        self.wheelbase = 2.7  # 轴距(米)
        # 兼容旧仿真代码中常用的字段命名
        self.v = self.speed
        self.a = self.acceleration
        self.steer_angle = self.front_steer_angle

        # 车轮参数
        self.wheel_width = 0.25 * width  # 车轮宽度
        self.wheel_length = 0.5  # 车轮长度

        # 记录轨迹
        self.trajectory = [(x, y)]

        # 车辆控制参数
        self.max_speed = 5.0  # m/s
        self.max_reverse_speed = 2.5  # m/s
        self.max_accel = 2.0  # m/s^2
        self.max_brake = 4.0  # m/s^2
        self.max_steer = math.pi / 4  # 最大转向角(弧度)
        self.steer_response = math.pi  # 转向响应速度(rad/s)
        self.rolling_resistance = 0.2  # 简单滚阻(m/s^2)
        self.drag_coefficient = 0.015  # 近似空气阻力
        self.creep_speed = 0.35  # 松油门蠕行目标速度
        self.creep_accel = 0.45  # 蠕行加速度
        self.jerk_limit = 6.0  # 纵向 jerk 上限
        self.throttle_response = 2.4  # 油门建立速度
        self.brake_response = 4.0  # 制动建立速度
        self.steer_speed_sensitivity = 0.08  # 高速时削弱最大转角
        self.reverse = False  # 当前是否挂入倒挡
        self.applied_throttle = 0.0
        self.applied_brake = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_steer = 0.0

        # 四轮转向模式
        self.steering_mode = "normal"  # 可选: "normal", "counter", "crab"
        self.rear_steer_ratio = 0.5  # 后轮转向比例 (相对于前轮)

        # 传感器配置
        self.sensors = {
            'fisheye_cameras': [],  # 环视摄像头 (黄色)
            'front_camera': None,  # 前视摄像头 (红色)
            'ultrasonic': [],  # 超声波雷达 (紫色)
            'imu': None,  # 消费级IMU (绿色)
            'gps': None  # 消费级GPS (绿色)
        }

        # 传感器显示控制
        self.show_sensors = False  # 默认不显示传感器

        # 初始化传感器位置
        self._init_sensors()

    def _init_sensors(self):
        """初始化传感器位置"""
        half_length = self.length / 2
        half_width = self.width / 2

        # 环视摄像头 (4个，黄色)
        # 车头/车尾各2个，左/右后视镜各1个
        fisheye_positions = [
            (half_length, 0),  # 车头中央
            (-half_length, 0),  # 车尾中央
            (0, half_width),  # 右侧中央
            (0, -half_width)  # 左侧中央
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
        """获取车辆四个角的坐标(用于碰撞检测和渲染)"""
        half_length = self.length / 2
        half_width = self.width / 2

        # 车辆本地坐标系中的四个角
        corners_local = [
            (half_length, half_width),  # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width),  # 左后
            (-half_length, half_width)  # 右后
        ]

        # 转换到世界坐标系
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        corners_world = []
        for lx, ly in corners_local:
            wx = self.x + lx * cos_h - ly * sin_h
            wy = self.y + lx * sin_h + ly * cos_h
            corners_world.append((wx, wy))

        return corners_world

    def get_wheel_positions(self) -> List[Tuple[float, float, float]]:
        """
        获取四个车轮的位置和角度

        返回:
            wheels: 列表，每个元素为(x, y, angle) 表示车轮的位置和朝向
        """
        half_length = self.length / 2 * 0.8  # 车轮位置略微内缩
        half_width = self.width / 2 * 0.9

        # 车轮在车身坐标系中的位置
        wheel_positions_local = [
            (half_length, half_width, self.front_steer_angle),  # 右前轮
            (half_length, -half_width, self.front_steer_angle),  # 左前轮
            (-half_length, -half_width, self.rear_steer_angle),  # 左后轮
            (-half_length, half_width, self.rear_steer_angle)  # 右后轮
        ]

        # 转换到世界坐标系
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        wheel_positions_world = []
        for lx, ly, angle in wheel_positions_local:
            wx = self.x + lx * cos_h - ly * sin_h
            wy = self.y + lx * sin_h + ly * cos_h
            wheel_angle = self.heading + angle
            wheel_positions_world.append((wx, wy, wheel_angle))

        return wheel_positions_world

    def get_sensor_positions(self):
        """获取传感器的全局坐标位置"""
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        sensor_positions = {'fisheye_cameras': [
        ], 'front_camera': None, 'ultrasonic': [], 'imu': None, 'gps': None}

        # 环视摄像头
        for camera in self.sensors['fisheye_cameras']:
            lx, ly = camera['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['fisheye_cameras'].append(
                {'pos': (x, y), 'color': camera['color']})

        # 前视摄像头
        if self.sensors['front_camera']:
            lx, ly = self.sensors['front_camera']['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['front_camera'] = {
                'pos': (x, y), 'color': self.sensors['front_camera']['color']}

        # 超声波雷达
        for sensor in self.sensors['ultrasonic']:
            lx, ly = sensor['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['ultrasonic'].append(
                {'pos': (x, y), 'color': sensor['color']})

        # IMU
        if self.sensors['imu']:
            lx, ly = self.sensors['imu']['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['imu'] = {
                'pos': (x, y), 'color': self.sensors['imu']['color']}

        # GPS
        if self.sensors['gps']:
            lx, ly = self.sensors['gps']['local_pos']
            x = self.x + lx * cos_h - ly * sin_h
            y = self.y + lx * sin_h + ly * cos_h
            sensor_positions['gps'] = {
                'pos': (x, y), 'color': self.sensors['gps']['color']}

        return sensor_positions

    def update(self, throttle: float, brake: float, steer: float, dt: float) -> None:
        """
        更新车辆状态

        参数:
            throttle: 油门输入[0, 1]
            brake: 制动输入[0, 1]
            steer: 转向输入[-1, 1]
            dt: 时间步长(秒)
        """
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        steer = max(-1.0, min(1.0, steer))
        self.last_throttle = throttle
        self.last_brake = brake
        self.last_steer = steer

        throttle_delta_limit = self.throttle_response * dt
        brake_delta_limit = self.brake_response * dt
        self.applied_throttle += max(
            -throttle_delta_limit,
            min(throttle - self.applied_throttle, throttle_delta_limit),
        )
        self.applied_brake += max(
            -brake_delta_limit,
            min(brake - self.applied_brake, brake_delta_limit),
        )

        speed_factor = 1.0 / (1.0 + self.steer_speed_sensitivity * abs(self.speed))
        effective_max_steer = max(math.radians(8.0), self.max_steer * speed_factor)

        # 计算目标前轮转向角，并施加转向速率限制，避免瞬时满打方向
        target_front_steer = steer * effective_max_steer
        max_steer_delta = self.steer_response * dt
        steer_delta = target_front_steer - self.front_steer_angle
        steer_delta = max(-max_steer_delta, min(max_steer_delta, steer_delta))
        self.front_steer_angle += steer_delta

        # 根据转向模式更新后轮转向角
        if self.steering_mode == "normal":
            # 普通模式：后轮不转向
            target_rear_steer = 0.0
        elif self.steering_mode == "counter":
            # 反向模式：后轮反向转向，提高转弯半径
            target_rear_steer = -self.front_steer_angle * self.rear_steer_ratio
        elif self.steering_mode == "crab":
            # 蟹行模式：后轮同向转向，实现横向移动
            target_rear_steer = self.front_steer_angle * self.rear_steer_ratio
        else:
            target_rear_steer = 0.0

        rear_steer_delta = target_rear_steer - self.rear_steer_angle
        rear_steer_delta = max(-max_steer_delta, min(max_steer_delta, rear_steer_delta))
        self.rear_steer_angle += rear_steer_delta

        # 计算纵向加速度，支持倒车和符号感知制动
        drive_acceleration = self.applied_throttle * self.max_accel
        target_acceleration = -drive_acceleration if self.reverse else drive_acceleration

        if (
            not self.reverse
            and self.applied_throttle < 0.03
            and self.applied_brake < 0.05
            and self.speed >= -0.05
            and self.speed < self.creep_speed
        ):
            target_acceleration = max(target_acceleration, self.creep_accel)

        if self.applied_brake > 0:
            if abs(self.speed) > 1e-4:
                target_acceleration += -math.copysign(self.applied_brake * self.max_brake, self.speed)
            else:
                target_acceleration = 0.0
        elif abs(self.speed) > 1e-4 and self.applied_throttle < 0.02:
            drag = self.rolling_resistance + self.drag_coefficient * abs(self.speed) * abs(self.speed)
            target_acceleration += -math.copysign(drag, self.speed)

        max_accel_delta = self.jerk_limit * dt
        accel_delta = target_acceleration - self.acceleration
        accel_delta = max(-max_accel_delta, min(max_accel_delta, accel_delta))
        self.acceleration += accel_delta

        # 更新速度，并支持倒车
        self.speed += self.acceleration * dt
        self.speed = max(-self.max_reverse_speed, min(self.speed, self.max_speed))

        # 制动至接近零速时，直接置零，避免车辆抖动
        if self.applied_brake > 0 and abs(self.speed) < 0.05:
            self.speed = 0.0
        elif self.applied_throttle < 0.02 and self.applied_brake < 0.02 and abs(self.speed) < 0.02:
            self.speed = 0.0

        # 低速四轮转向运动学模型
        if abs(self.speed) > 1e-3:
            front_angle = self.front_steer_angle
            rear_angle = self.rear_steer_angle
            wheelbase = max(self.wheelbase, 1e-3)
            lf = lr = wheelbase / 2.0

            # beta 决定速度方向，yaw_rate 决定航向变化；
            # 对 crab 模式，同向转角会产生近似零航向变化但允许斜向位移。
            beta = math.atan2(
                lr * math.tan(front_angle) + lf * math.tan(rear_angle),
                wheelbase
            )
            yaw_rate = self.speed * (
                math.tan(front_angle) - math.tan(rear_angle)
            ) / wheelbase

            self.heading = (self.heading + yaw_rate * dt) % (2 * math.pi)
            travel_heading = self.heading + beta
        else:
            travel_heading = self.heading

        # 根据当前速度方向更新位置，负速度自然表示倒车
        self.x += self.speed * math.cos(travel_heading) * dt
        self.y += self.speed * math.sin(travel_heading) * dt

        # 记录轨迹
        self.trajectory.append((self.x, self.y))
        self.v = self.speed
        self.a = self.acceleration
        self.steer_angle = self.front_steer_angle

    def get_gear_label(self) -> str:
        if abs(self.speed) < 0.05 and self.applied_brake > 0.2:
            return "P"
        return "R" if self.reverse else "D"

    def set_steering_mode(self, mode: str) -> None:
        """
        设置转向模式

        参数:
            mode: 转向模式，可选 "normal", "counter", "crab"
        """
        if mode in ["normal", "counter", "crab"]:
            self.steering_mode = mode
            print(f"已切换到{mode}转向模式")
        else:
            print(f"无效的转向模式: {mode}")

    def get_steering_mode(self) -> str:
        """获取当前转向模式"""
        return self.steering_mode


class PathFollower:
    """路径跟踪控制器"""

    def __init__(self, lookahead=5.0, control_method='default'):
        """
        初始化路径跟踪控制器

        参数:
            lookahead: 前瞻距离(米)
            control_method: 控制方法('default', 'pid', 'mpc', 'lqr', 'parking')
        """
        self.path = []
        self.lookahead = lookahead
        self.current_target_idx = 0
        self.control_method = control_method
        self.target_speed = 5.0  # 目标速度(m/s)

        # 泊车相关参数
        self.parking_phase = 'approach'  # 泊车阶段：approach, reverse, adjust
        self.parking_type = None  # 停车类型：parallel, perpendicular
        self.reverse_gear = False  # 是否处于倒车状态
        self.min_parking_speed = 1.0  # 最小泊车速度(m/s)
        self.max_parking_speed = 2.0  # 最大泊车速度(m/s)
        self.safe_distance = 0.5  # 安全距离(m)
        self.goal_pose: Optional[Dict[str, Any]] = None
        self.goal_slot_id: Optional[Union[str, int]] = None
        self.goal_slot_type = "perpendicular"
        self.auto_terminal_parking = True
        self.terminal_trigger_distance = 3.2
        self.terminal_path_tail = 6
        self.position_tolerance = 0.25
        self.heading_tolerance = math.radians(5.0)
        self.stop_speed_tolerance = 0.05
        self.hold_time = 0.35
        self.max_terminal_time = 18.0
        self.max_stagnation_time = 4.0
        self.max_gear_switches = 2
        self.minimum_clearance = 0.2
        self.stage_offset = 1.15
        self.align_heading_threshold = math.radians(12.0)
        self.max_stage_speed = 1.0
        self.max_dock_speed = 0.75
        self.max_adjust_speed = 0.45
        self.terminal_mode_active = False
        self.terminal_phase = "idle"
        self.terminal_status = "路径跟踪中"
        self.terminal_failure_reason: Optional[str] = None
        self.terminal_success = False
        self.terminal_elapsed = 0.0
        self.terminal_hold_elapsed = 0.0
        self.terminal_clearance = float("inf")
        self.terminal_errors: Dict[str, float] = {}
        self._gear_switch_count = 0
        self._last_reverse_command: Optional[bool] = None
        self._last_progress_score: Optional[float] = None
        self._stagnation_time = 0.0

        # PID控制参数
        self.pid_params = {
            'kp_steer': 0.7,  # 转向比例系数
            'ki_steer': 0.01,  # 转向积分系数
            'kd_steer': 0.1,  # 转向微分系数
            'kp_speed': 0.5,  # 速度比例系数
            'ki_speed': 0.01,  # 速度积分系数
            'kd_speed': 0.05  # 速度微分系数
        }
        self.steer_error_prev = 0.0
        self.steer_error_sum = 0.0
        self.speed_error_prev = 0.0
        self.speed_error_sum = 0.0

        # MPC控制参数
        self.mpc_params = {
            'horizon': 10,  # 预测步长
            'dt': 0.1,  # 时间步长
            'q_x': 1.0,  # 纵向误差权重
            'q_y': 2.0,  # 横向误差权重
            'q_heading': 3.0,  # 朝向误差权重
            'r_steer': 1.0,  # 转向输入权重
            'r_accel': 0.5  # 加速度输入权重
        }

        # LQR控制参数
        self.lqr_params = {
            'q_y': 1.0,  # 横向误差权重
            'q_heading': 2.0,  # 朝向误差权重
            'q_speed': 0.5,  # 速度误差权重
            'r_steer': 0.1,  # 转向输入权重
            'r_accel': 0.1  # 加速度输入权重
        }

    def configure_terminal_parking(self, config: Optional[Dict[str, Any]] = None) -> None:
        """配置终端泊车控制参数。"""
        config = config or {}
        self.auto_terminal_parking = bool(config.get("enabled", True))
        self.terminal_trigger_distance = float(config.get("trigger_distance", 3.2))
        self.position_tolerance = float(config.get("position_tolerance", 0.25))
        self.heading_tolerance = math.radians(float(config.get("heading_tolerance_deg", 5.0)))
        self.stop_speed_tolerance = float(config.get("stop_speed_tolerance", 0.05))
        self.hold_time = float(config.get("hold_time", 0.35))
        self.max_terminal_time = float(config.get("max_duration", 18.0))
        self.max_stagnation_time = float(config.get("max_stagnation_time", 4.0))
        self.max_gear_switches = int(config.get("max_gear_switches", 2))
        self.minimum_clearance = float(config.get("minimum_clearance", 0.2))
        self.stage_offset = float(config.get("staging_offset", 1.15))
        self.align_heading_threshold = math.radians(float(config.get("align_heading_deg", 12.0)))
        self.max_stage_speed = float(config.get("max_stage_speed", 1.0))
        self.max_dock_speed = float(config.get("max_dock_speed", 0.75))
        self.max_adjust_speed = float(config.get("max_adjust_speed", 0.45))
        self.terminal_path_tail = max(2, int(config.get("path_tail_samples", 6)))

    def _reset_terminal_state(self, keep_goal: bool = True) -> None:
        self.terminal_mode_active = False
        self.terminal_phase = "idle"
        self.terminal_status = "路径跟踪中"
        self.terminal_failure_reason = None
        self.terminal_success = False
        self.terminal_elapsed = 0.0
        self.terminal_hold_elapsed = 0.0
        self.terminal_clearance = float("inf")
        self.terminal_errors = {}
        self._gear_switch_count = 0
        self._last_reverse_command = None
        self._last_progress_score = None
        self._stagnation_time = 0.0
        if not keep_goal:
            self.goal_pose = None
            self.goal_slot_id = None
            self.goal_slot_type = "perpendicular"

    def clear_goal_pose(self) -> None:
        self._reset_terminal_state(keep_goal=False)

    def set_goal_pose(
        self,
        x: float,
        y: float,
        heading_deg: float,
        slot_id: Optional[Union[str, int]] = None,
        slot_type: str = "perpendicular",
    ) -> None:
        """设置终端泊车目标位姿。"""
        self.goal_pose = {
            "x": float(x),
            "y": float(y),
            "heading_deg": float(heading_deg),
        }
        self.goal_slot_id = slot_id
        self.goal_slot_type = slot_type or "perpendicular"
        self._reset_terminal_state(keep_goal=True)
        self.terminal_status = "已锁定目标位姿"

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _goal_error_metrics(self, vehicle) -> Optional[Dict[str, float]]:
        if not self.goal_pose:
            return None

        goal_heading = math.radians(self.goal_pose["heading_deg"])
        goal_x = self.goal_pose["x"]
        goal_y = self.goal_pose["y"]

        rear_offset = vehicle.wheelbase / 2.0
        vehicle_rear_x = vehicle.x - math.cos(vehicle.heading) * rear_offset
        vehicle_rear_y = vehicle.y - math.sin(vehicle.heading) * rear_offset
        goal_rear_x = goal_x - math.cos(goal_heading) * rear_offset
        goal_rear_y = goal_y - math.sin(goal_heading) * rear_offset

        center_dx = goal_x - vehicle.x
        center_dy = goal_y - vehicle.y
        rear_dx = goal_rear_x - vehicle_rear_x
        rear_dy = goal_rear_y - vehicle_rear_y

        longitudinal_error = rear_dx * math.cos(goal_heading) + rear_dy * math.sin(goal_heading)
        lateral_error = -rear_dx * math.sin(goal_heading) + rear_dy * math.cos(goal_heading)
        yaw_error = self._normalize_angle(goal_heading - vehicle.heading)

        metrics = {
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_heading": goal_heading,
            "goal_rear_x": goal_rear_x,
            "goal_rear_y": goal_rear_y,
            "rear_x": vehicle_rear_x,
            "rear_y": vehicle_rear_y,
            "center_dx": center_dx,
            "center_dy": center_dy,
            "center_distance": math.hypot(center_dx, center_dy),
            "rear_distance": math.hypot(rear_dx, rear_dy),
            "longitudinal_error": longitudinal_error,
            "lateral_error": lateral_error,
            "yaw_error": yaw_error,
            "yaw_error_deg": math.degrees(yaw_error),
        }
        self.terminal_errors = metrics
        return metrics

    def _terminal_progress_score(self, metrics: Dict[str, float]) -> float:
        return (
            metrics["rear_distance"]
            + 0.6 * abs(metrics["lateral_error"])
            + 0.03 * abs(metrics["yaw_error_deg"])
        )

    def _update_terminal_progress(self, metrics: Dict[str, float], dt: float) -> None:
        score = self._terminal_progress_score(metrics)
        if self._last_progress_score is None or score < self._last_progress_score - 0.02:
            self._last_progress_score = score
            self._stagnation_time = 0.0
            return
        self._stagnation_time += dt
        self._last_progress_score = min(self._last_progress_score, score)

    def _set_reverse_command(self, reverse: bool) -> None:
        if self._last_reverse_command is None:
            self._last_reverse_command = reverse
        elif reverse != self._last_reverse_command:
            self._gear_switch_count += 1
            self._last_reverse_command = reverse
        self.reverse_gear = reverse

    def update_terminal_clearance(self, clearance: float) -> None:
        self.terminal_clearance = float(clearance)
        if (
            self.terminal_mode_active
            and clearance < self.minimum_clearance
            and not self.terminal_failure_reason
        ):
            self.terminal_failure_reason = "终端泊车安全间隙不足"
            self.terminal_phase = "failed"
            self.terminal_status = self.terminal_failure_reason

    def _speed_command(self, vehicle, target_speed: float, reverse: bool) -> Tuple[float, float]:
        direction_speed = -vehicle.speed if reverse else vehicle.speed
        speed_error = target_speed - direction_speed
        if target_speed <= self.stop_speed_tolerance:
            return 0.0, 0.45 if abs(vehicle.speed) > self.stop_speed_tolerance else 0.18
        if speed_error >= 0:
            return min(0.65, 0.65 * speed_error + 0.08), 0.0
        return 0.0, min(0.8, -0.85 * speed_error)

    def _pursuit_to_point(
        self,
        vehicle,
        target_point: Tuple[float, float],
        target_speed: float,
        reverse: bool = False,
        yaw_correction: float = 0.0,
    ) -> Tuple[float, float, float]:
        dx = target_point[0] - vehicle.x
        dy = target_point[1] - vehicle.y
        dx_local = dx * math.cos(vehicle.heading) + dy * math.sin(vehicle.heading)
        dy_local = -dx * math.sin(vehicle.heading) + dy * math.cos(vehicle.heading)
        distance = math.hypot(dx_local, dy_local)
        curvature = 0.0 if distance < 1e-6 else (2.0 * dy_local) / max(distance * distance, 0.8)
        steer_angle = math.atan(vehicle.wheelbase * curvature)
        steer = steer_angle / max(vehicle.max_steer, math.radians(8.0))
        if reverse:
            steer = -steer
        steer += yaw_correction
        steer = max(-1.0, min(1.0, steer))

        self._set_reverse_command(reverse)
        if hasattr(vehicle, 'reverse'):
            vehicle.reverse = reverse

        throttle, brake = self._speed_command(vehicle, target_speed, reverse)
        if distance < 0.35:
            throttle *= 0.55
        return throttle, brake, steer

    def _should_use_terminal_parking(self, vehicle) -> bool:
        metrics = self._goal_error_metrics(vehicle)
        if not metrics or not self.goal_pose or not self.auto_terminal_parking:
            return False
        near_end = self.current_target_idx >= max(len(self.path) - self.terminal_path_tail, 0)
        return near_end or metrics["center_distance"] <= self.terminal_trigger_distance

    def _terminal_parking_control(self, vehicle, dt: float) -> Tuple[float, float, float]:
        metrics = self._goal_error_metrics(vehicle)
        if not metrics:
            self._reset_terminal_state(keep_goal=True)
            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = False
            return 0.0, 0.0, 0.0

        self.terminal_mode_active = True
        self.terminal_elapsed += dt

        if self.terminal_failure_reason:
            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = False
            return 0.0, 0.85, 0.0

        self._update_terminal_progress(metrics, dt)

        if self.terminal_elapsed > self.max_terminal_time:
            self.terminal_failure_reason = "终端泊车超时"
        elif self._stagnation_time > self.max_stagnation_time and metrics["center_distance"] > self.position_tolerance * 1.5:
            self.terminal_failure_reason = "终端泊车未继续收敛"
        elif self._gear_switch_count > self.max_gear_switches:
            self.terminal_failure_reason = "终端泊车换挡次数过多"

        if self.terminal_failure_reason:
            self.terminal_phase = "failed"
            self.terminal_status = self.terminal_failure_reason
            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = False
            return 0.0, 0.85, 0.0

        if (
            metrics["center_distance"] <= self.position_tolerance
            and abs(metrics["yaw_error"]) <= self.heading_tolerance
        ):
            self.terminal_phase = "hold"
            self.terminal_hold_elapsed += dt
            self.terminal_status = "终端泊车保持中"
            self._set_reverse_command(False)
            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = False
            if (
                self.terminal_hold_elapsed >= self.hold_time
                and abs(vehicle.speed) <= self.stop_speed_tolerance
            ):
                self.terminal_success = True
                self.terminal_status = "终端泊车完成"
            return 0.0, 0.55 if abs(vehicle.speed) > self.stop_speed_tolerance else 0.25, 0.0

        self.terminal_hold_elapsed = 0.0

        stage_point = (
            metrics["goal_x"] - math.cos(metrics["goal_heading"]) * self.stage_offset,
            metrics["goal_y"] - math.sin(metrics["goal_heading"]) * self.stage_offset,
        )
        yaw_correction = max(
            -0.45,
            min(0.45, metrics["yaw_error"] / math.radians(28.0)),
        )

        if (
            metrics["rear_distance"] > max(self.stage_offset * 0.9, 0.9)
            or abs(metrics["yaw_error"]) > self.align_heading_threshold
        ):
            self.terminal_phase = "stage"
            self.terminal_status = "终端泊车引导入位"
            return self._pursuit_to_point(
                vehicle,
                stage_point,
                self.max_stage_speed,
                reverse=False,
                yaw_correction=yaw_correction * 0.45,
            )

        reverse = metrics["longitudinal_error"] < -0.08
        target_speed = self.max_adjust_speed if abs(metrics["longitudinal_error"]) < 0.45 else self.max_dock_speed
        self.terminal_phase = "adjust" if target_speed == self.max_adjust_speed else "dock"
        self.terminal_status = "终端泊车精调中" if self.terminal_phase == "adjust" else "终端泊车对位中"
        return self._pursuit_to_point(
            vehicle,
            (metrics["goal_rear_x"], metrics["goal_rear_y"]),
            target_speed,
            reverse=reverse,
            yaw_correction=yaw_correction * (-0.55 if reverse else 0.55),
        )

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            "terminal_active": self.terminal_mode_active,
            "phase": self.terminal_phase,
            "status": self.terminal_status,
            "failure_reason": self.terminal_failure_reason,
            "success": self.terminal_success,
            "gear": "R" if self.reverse_gear else "D",
            "gear_switches": self._gear_switch_count,
            "clearance": self.terminal_clearance,
            "goal_slot_id": self.goal_slot_id,
            "goal_slot_type": self.goal_slot_type,
            "elapsed": self.terminal_elapsed,
            "errors": dict(self.terminal_errors),
        }

    def set_path(self, path):
        """设置跟踪路径"""
        self.path = path or []
        self.current_target_idx = 0
        self._reset_terminal_state(keep_goal=True)
        self.steer_error_prev = 0.0
        self.steer_error_sum = 0.0
        self.speed_error_prev = 0.0
        self.speed_error_sum = 0.0

    def set_control_method(self, method):
        """设置控制方法"""
        if method in ['default', 'pid', 'mpc', 'lqr', 'parking']:
            self.control_method = method
        else:
            print(f"不支持的控制方法: {method}，使用默认方法")
            self.control_method = 'default'

    def get_control(self, vehicle, dt: float):
        """获取控制输入"""
        dt = max(float(dt), 0.0)
        if not self.path:
            self._reset_terminal_state(keep_goal=True)
            self.terminal_status = "等待路径"
            return 0.0, 0.0, 0.0

        if self.control_method == 'parking':
            self.terminal_mode_active = False
            return self._parking_control(vehicle)

        if self._should_use_terminal_parking(vehicle):
            return self._terminal_parking_control(vehicle, dt)

        if self.terminal_mode_active:
            self._reset_terminal_state(keep_goal=True)
            self.terminal_status = "路径跟踪中"

        if hasattr(vehicle, 'reverse'):
            vehicle.reverse = False

        if self.control_method == 'pid':
            return self._pid_control(vehicle)
        if self.control_method == 'mpc':
            return self._mpc_control(vehicle)
        if self.control_method == 'lqr':
            return self._lqr_control(vehicle)

        return self._default_control(vehicle)

    def _default_control(self, vehicle):
        """默认控制方法"""
        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')

        # 向前找到一个在前瞻距离范围内的点
        for i in range(self.current_target_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                target_idx = i

            if dist > self.lookahead:
                break

        # 更新当前目标点索引
        self.current_target_idx = target_idx

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算车辆到目标点的向量
        dx = tx - vehicle.x
        dy = ty - vehicle.y

        # 计算目标点相对于车头的角度
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle.heading

        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # 根据角度误差计算转向
        steer = heading_error / (math.pi / 4)  # 假设最大转向角为π/4
        steer = max(-1.0, min(1.0, steer))  # 限制在 [-1, 1] 范围内

        # 简单的速度控制：根据转向角的大小调整速度
        throttle = 0.5 * (1.0 - 0.5 * abs(steer))
        brake = 0.0

        # 如果即将转弯，提前减速
        if abs(steer) > 0.5:
            throttle *= 0.5

        return throttle, brake, steer

    def _pid_control(self, vehicle):
        """PID控制方法"""
        # 动态调整前瞻距离 - 根据车速调整
        dynamic_lookahead = max(3.0, min(self.lookahead, vehicle.speed * 0.8))

        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')
        closest_idx = target_idx

        # 首先找到最近点
        for i in range(self.current_target_idx, min(self.current_target_idx + 30, len(self.path))):
            if i >= len(self.path):
                break

            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 从最近点开始，找到前瞻距离范围内的目标点
        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist > dynamic_lookahead:
                target_idx = i
                break

        # 确保目标点不会超出路径范围
        target_idx = min(target_idx, len(self.path) - 1)

        # 更新当前目标点索引，但不要后退
        self.current_target_idx = max(self.current_target_idx, closest_idx)

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算车辆到目标点的向量
        dx = tx - vehicle.x
        dy = ty - vehicle.y

        # 计算目标点相对于车头的角度
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle.heading

        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # PID控制 - 转向
        # 限制积分项，防止积分饱和
        self.steer_error_sum = max(-3.0, min(3.0,
                                   self.steer_error_sum + heading_error))
        steer_error_diff = heading_error - self.steer_error_prev
        self.steer_error_prev = heading_error

        # 计算PID控制输出
        steer = (self.pid_params['kp_steer'] * heading_error + self.pid_params['ki_steer'] * self.steer_error_sum +
                 self.pid_params['kd_steer'] * steer_error_diff)

        # 限制在 [-1, 1] 范围内
        steer = max(-1.0, min(1.0, steer))

        # PID控制 - 速度
        speed_error = self.target_speed - vehicle.speed

        # 限制积分项，防止积分饱和
        self.speed_error_sum = max(-5.0, min(5.0,
                                   self.speed_error_sum + speed_error))
        speed_error_diff = speed_error - self.speed_error_prev
        self.speed_error_prev = speed_error

        # 计算PID控制输出
        throttle_brake = (self.pid_params['kp_speed'] * speed_error +
                          self.pid_params['ki_speed'] * self.speed_error_sum +
                          self.pid_params['kd_speed'] * speed_error_diff)

        # 将输出转换为油门和刹车
        if throttle_brake >= 0:
            throttle = min(1.0, throttle_brake)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -throttle_brake)

        # 如果即将转弯，减速
        if abs(steer) > 0.5:
            throttle *= 0.5
        elif abs(steer) > 0.3:
            throttle *= 0.7

        return throttle, brake, steer

    def _mpc_control(self, vehicle):
        """简化的模型预测控制方法"""
        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')

        # 向前找到一个在前瞻距离范围内的点
        for i in range(self.current_target_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                target_idx = i

            if dist > self.lookahead:
                break

        # 更新当前目标点索引
        self.current_target_idx = target_idx

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算车辆到目标点的向量
        dx = tx - vehicle.x
        dy = ty - vehicle.y

        # 计算目标点相对于车头的角度
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle.heading

        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # 简化的MPC控制 - 使用预测模型计算最优控制输入
        # 这里使用简化方法：将状态误差与权重相乘作为控制输入
        steer = (self.mpc_params['q_y'] * math.sin(heading_error) +
                 self.mpc_params['q_heading'] * heading_error) / self.mpc_params['r_steer']

        # 限制在 [-1, 1] 范围内
        steer = max(-1.0, min(1.0, steer))

        # 计算速度误差
        speed_error = self.target_speed - vehicle.speed

        # 计算油门和刹车
        accel_cmd = self.mpc_params['q_x'] * \
            speed_error / self.mpc_params['r_accel']

        if accel_cmd >= 0:
            throttle = min(1.0, accel_cmd)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -accel_cmd)

        # 如果即将转弯，减速
        if abs(steer) > 0.5:
            throttle *= 0.5
        elif abs(steer) > 0.3:
            throttle *= 0.7

        return throttle, brake, steer

    def _lqr_control(self, vehicle):
        """简化的线性二次型调节器控制方法"""
        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')

        # 向前找到一个在前瞻距离范围内的点
        for i in range(self.current_target_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                target_idx = i

            if dist > self.lookahead:
                break

        # 更新当前目标点索引
        self.current_target_idx = target_idx

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算参考路径的切线方向（简化版）
        next_idx = min(target_idx + 1, len(self.path) - 1)
        next_x, next_y = self.path[next_idx]
        ref_heading = math.atan2(next_y - ty, next_x - tx)

        # 计算状态误差
        dx = tx - vehicle.x
        dy = ty - vehicle.y
        dheading = ref_heading - vehicle.heading

        # 规范化到 [-π, π]
        while dheading > math.pi:
            dheading -= 2 * math.pi
        while dheading < -math.pi:
            dheading += 2 * math.pi

        # 计算横向误差（车辆坐标系中）
        cos_heading = math.cos(vehicle.heading)
        sin_heading = math.sin(vehicle.heading)
        lateral_error = -dx * sin_heading + dy * cos_heading

        # 简化的LQR控制 - 在实际应用中应求解Riccati方程
        # 这里使用简化方法：将状态误差与权重相乘作为控制输入
        steer = (self.lqr_params['q_y'] * lateral_error +
                 self.lqr_params['q_heading'] * dheading) / self.lqr_params['r_steer']

        # 限制在 [-1, 1] 范围内
        steer = max(-1.0, min(1.0, steer))

        # 计算速度误差
        speed_error = self.target_speed - vehicle.speed

        # 计算油门和刹车
        accel_cmd = self.lqr_params['q_speed'] * \
            speed_error / self.lqr_params['r_accel']

        if accel_cmd >= 0:
            throttle = min(1.0, accel_cmd)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -accel_cmd)

        # 如果即将转弯，减速
        if abs(steer) > 0.5:
            throttle *= 0.5

        return throttle, brake, steer

    def _parking_control(self, vehicle):
        """泊车专用控制方法"""
        if not self.path or not self.parking_type:
            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = False
            return 0.0, 0.0, 0.0

        # 获取目标点
        target_idx = self._find_target_point(vehicle)
        if target_idx >= len(self.path):
            return 0.0, 0.0, 0.0

        tx, ty = self.path[target_idx]

        # 计算到目标点的距离和角度
        dx = tx - vehicle.x
        dy = ty - vehicle.y
        distance = math.sqrt(dx * dx + dy * dy)

        # 计算目标航向角（根据路径的下一个点）
        next_idx = min(target_idx + 1, len(self.path) - 1)
        next_x, next_y = self.path[next_idx]
        path_heading = math.atan2(next_y - ty, next_x - tx)

        # 计算航向误差
        heading_error = path_heading - vehicle.heading
        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # 计算预瞄点 - 根据当前阶段和速度动态调整预瞄距离
        preview_distance = 0.0
        if self.parking_phase == 'approach':
            # 接近阶段使用较远的预瞄点
            preview_distance = max(2.0, min(5.0, vehicle.speed * 1.0))
        elif self.parking_phase == 'reverse':
            # 倒车阶段使用较近的预瞄点
            preview_distance = max(1.0, min(3.0, abs(vehicle.speed) * 0.8))
        else:  # adjust phase
            # 微调阶段使用非常近的预瞄点
            preview_distance = max(0.5, min(1.5, abs(vehicle.speed) * 0.5))

        # 寻找预瞄点
        preview_idx = target_idx
        preview_distance_sum = 0.0
        for i in range(target_idx, len(self.path) - 1):
            segment_length = math.sqrt((self.path[i + 1][0] - self.path[i][0])**2 +
                                       (self.path[i + 1][1] - self.path[i][1])**2)
            preview_distance_sum += segment_length
            if preview_distance_sum >= preview_distance:
                preview_idx = i + 1
                break

        # 获取预瞄点坐标
        preview_x, preview_y = self.path[preview_idx]

        # 计算预瞄点相对于车辆的位置（车辆坐标系）
        dx_local = (preview_x - vehicle.x) * math.cos(vehicle.heading) + \
            (preview_y - vehicle.y) * math.sin(vehicle.heading)
        dy_local = -(preview_x - vehicle.x) * math.sin(vehicle.heading) + \
            (preview_y - vehicle.y) * math.cos(vehicle.heading)

        # 计算到预瞄点的距离
        preview_distance_actual = math.sqrt(dx_local**2 + dy_local**2)

        # 计算预瞄角度
        preview_angle = math.atan2(dy_local, dx_local)

        # 根据泊车阶段决定控制策略
        if self.parking_phase == 'approach':
            # 接近阶段：缓慢前进到泊车起始点
            if distance < 1.0:  # 到达泊车起始点
                self.parking_phase = 'reverse'
                self.reverse_gear = True
                if hasattr(vehicle, 'reverse'):
                    vehicle.reverse = False
                return 0.0, 0.3, 0.0  # 轻踩刹车准备倒车

            # 横向控制 - 使用PID控制器
            # 计算横向误差
            lateral_error = dy_local

            # 更新PID控制器状态
            self.steer_error_sum = max(-3.0, min(3.0,
                                       self.steer_error_sum + lateral_error))
            steer_error_diff = lateral_error - self.steer_error_prev
            self.steer_error_prev = lateral_error

            # 计算PID控制输出
            steer = (self.pid_params['kp_steer'] * lateral_error + self.pid_params['ki_steer'] * self.steer_error_sum +
                     self.pid_params['kd_steer'] * steer_error_diff)

            # 限制在 [-1, 1] 范围内
            steer = max(-1.0, min(1.0, steer))

            # 纵向控制 - 使用LQR控制器
            # 根据距离和预瞄点计算目标速度
            speed_factor = min(1.0, distance / 5.0)
            target_speed = self.min_parking_speed + speed_factor * \
                (self.max_parking_speed - self.min_parking_speed)

            # 计算速度误差
            speed_error = target_speed - vehicle.speed

            # 使用LQR参数计算加速度命令
            accel_cmd = self.lqr_params['q_speed'] * \
                speed_error / self.lqr_params['r_accel']

            # 将加速度命令转换为油门和刹车
            if accel_cmd >= 0:
                throttle = min(0.5, accel_cmd)  # 限制最大油门
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(0.3, -accel_cmd)  # 限制最大刹车

            # 如果转向角大，减速
            if abs(steer) > 0.5:
                throttle *= 0.5

            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = False
            return throttle, brake, steer

        elif self.parking_phase == 'reverse':
            # 倒车入库阶段

            # 横向控制 - 使用PID控制器
            # 计算横向误差 - 倒车时需要反向
            lateral_error = -dy_local  # 倒车时横向误差取反

            # 更新PID控制器状态
            self.steer_error_sum = max(-3.0, min(3.0,
                                       self.steer_error_sum + lateral_error))
            steer_error_diff = lateral_error - self.steer_error_prev
            self.steer_error_prev = lateral_error

            # 计算PID控制输出
            steer = (self.pid_params['kp_steer'] * lateral_error + self.pid_params['ki_steer'] * self.steer_error_sum +
                     self.pid_params['kd_steer'] * steer_error_diff)

            # 限制在 [-1, 1] 范围内
            steer = max(-1.0, min(1.0, steer))

            # 根据停车类型调整转向策略
            if self.parking_type == 'parallel':
                # 侧方停车：需要先倒车转向，然后回正
                if abs(heading_error) > math.pi / 6:  # 如果航向偏差大，优先调整航向
                    # 增加转向力度
                    steer = max(-1.0, min(1.0, steer * 1.5))

            # 如果接近目标点，进入微调阶段
            if distance < 1.0:
                self.parking_phase = 'adjust'
                self.reverse_gear = False
                if hasattr(vehicle, 'reverse'):
                    vehicle.reverse = True
                return 0.0, 0.3, 0.0

            # 纵向控制 - 使用LQR控制器
            # 倒车速度控制 - 使用较小的目标速度
            target_speed = -self.min_parking_speed * 0.8  # 负值表示倒车

            # 计算速度误差
            speed_error = target_speed - vehicle.speed

            # 使用LQR参数计算加速度命令
            accel_cmd = self.lqr_params['q_speed'] * \
                speed_error / self.lqr_params['r_accel']

            # 倒车时油门和刹车的处理方式不同
            if self.reverse_gear:
                if accel_cmd <= 0:  # 需要减速或保持当前倒车速度
                    throttle = min(0.3, -accel_cmd)  # 倒车时，负的加速度命令对应油门
                    brake = 0.0
                else:  # 需要减小倒车速度
                    throttle = 0.0
                    brake = min(0.3, accel_cmd)  # 倒车时，正的加速度命令对应刹车
            else:
                throttle = 0.0
                brake = 0.3  # 如果不是倒车状态但在倒车阶段，使用刹车

            # 如果转向角大，减小倒车速度
            if abs(steer) > 0.5:
                throttle *= 0.7

            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = self.reverse_gear
            return throttle, brake, steer

        else:  # adjust phase
            # 微调阶段：精确调整到目标位置
            if distance < self.safe_distance:
                if hasattr(vehicle, 'reverse'):
                    vehicle.reverse = False
                return 0.0, 0.3, 0.0  # 停车

            # 横向控制 - 使用PID控制器
            # 计算横向误差
            lateral_error = dy_local
            if self.reverse_gear:
                lateral_error = -lateral_error  # 倒车时横向误差取反

            # 更新PID控制器状态
            self.steer_error_sum = max(-3.0, min(3.0,
                                       self.steer_error_sum + lateral_error))
            steer_error_diff = lateral_error - self.steer_error_prev
            self.steer_error_prev = lateral_error

            # 计算PID控制输出 - 微调阶段使用较小的增益
            steer = (0.5 * self.pid_params['kp_steer'] * lateral_error +
                     0.3 * self.pid_params['ki_steer'] * self.steer_error_sum +
                     0.7 * self.pid_params['kd_steer'] * steer_error_diff)

            # 限制在 [-0.5, 0.5] 范围内，微调阶段使用较小的转向角
            steer = max(-0.5, min(0.5, steer))

            # 计算纵向误差
            longitudinal_error = dx_local
            if self.reverse_gear:
                longitudinal_error = -longitudinal_error  # 倒车时纵向误差取反

            # 根据纵向误差决定前进还是倒车
            self.reverse_gear = longitudinal_error < 0

            # 纵向控制 - 使用LQR控制器
            # 根据距离计算目标速度
            speed_factor = min(1.0, distance / 2.0)
            target_speed = 0.5 * self.min_parking_speed * speed_factor
            if self.reverse_gear:
                target_speed = -target_speed  # 倒车时目标速度为负

            # 计算速度误差
            speed_error = target_speed - vehicle.speed

            # 使用LQR参数计算加速度命令
            accel_cmd = self.lqr_params['q_speed'] * \
                speed_error / self.lqr_params['r_accel']

            # 将加速度命令转换为油门和刹车
            if self.reverse_gear:
                if accel_cmd <= 0:  # 需要加大倒车速度
                    throttle = min(0.2, -accel_cmd)
                    brake = 0.0
                else:  # 需要减小倒车速度
                    throttle = 0.0
                    brake = min(0.2, accel_cmd)
            else:
                if accel_cmd >= 0:  # 需要加速
                    throttle = min(0.2, accel_cmd)
                    brake = 0.0
                else:  # 需要减速
                    throttle = 0.0
                    brake = min(0.2, -accel_cmd)

            if hasattr(vehicle, 'reverse'):
                vehicle.reverse = self.reverse_gear
            return throttle, brake, steer

    def _find_target_point(self, vehicle):
        """寻找合适的目标点"""
        # 动态调整前瞻距离 - 根据车速和泊车阶段调整
        if self.parking_phase == 'approach':
            dynamic_lookahead = max(
                3.0, min(self.lookahead, vehicle.speed * 0.8))
        elif self.parking_phase == 'reverse':
            dynamic_lookahead = max(2.0, min(4.0, vehicle.speed * 0.6))
        else:  # adjust phase
            dynamic_lookahead = max(1.0, min(2.0, vehicle.speed * 0.5))

        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')
        closest_idx = target_idx

        # 首先找到最近点
        for i in range(self.current_target_idx, min(self.current_target_idx + 30, len(self.path))):
            if i >= len(self.path):
                break

            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 从最近点开始，找到前瞻距离范围内的目标点
        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist > dynamic_lookahead:
                target_idx = i
                break

        # 确保目标点不会超出路径范围
        target_idx = min(target_idx, len(self.path) - 1)

        # 更新当前目标点索引，但不要后退
        self.current_target_idx = max(self.current_target_idx, closest_idx)

        return target_idx

    def set_parking_type(self, parking_type):
        """设置停车类型"""
        if parking_type in ['parallel', 'perpendicular']:
            self.parking_type = parking_type
            self.parking_phase = 'approach'
            self.reverse_gear = False
            return True
        return False


class ParkingEnvironment(Environment):

    def __init__(self, width, height):
        """初始化停车场环境"""
        super().__init__(width, height)
        self.dynamic_obstacles = []
        self.vehicle_width = 1.8  # 车辆宽度

    def add_obstacle(
            self,
            x,
            y,
            obstacle_type="rectangle",
            width=1.0,
            height=1.0,
            radius=0.5,
            angle=0.0,
            color=(100, 100, 100, 200),  # 默认颜色：灰色半透明
            is_parking_spot=False,  # 新增停车位标识
            occupied=False,  # 新增占用状态
            is_filled=True,  # 是否填充
            line_width=1  # 线宽
    ):
        """添加障碍物，支持颜色属性和停车位状态"""
        # 根据占用状态设置颜色
        if is_parking_spot:
            color = (255, 0, 0, 200) if occupied else (
                0, 255, 0, 200)  # 红色表示占用，绿色表示空闲

        # 创建一个新的障碍物对象
        if obstacle_type == "rectangle":
            obstacle = RectangleObstacle(
                x, y, width, height, angle, color, is_filled, line_width)
        else:  # circle
            obstacle = CircleObstacle(
                x, y, radius, color, is_filled, line_width)

        # 为了兼容旧代码，添加额外属性
        obstacle.type = obstacle_type
        obstacle.is_parking_spot = is_parking_spot
        obstacle.occupied = occupied

        # 添加安全边界（蓝色半透明）
        safety_margin = 0.25 * self.vehicle_width
        if is_parking_spot:
            # 生成停车位扩展区域
            expansion_factor = 1.3  # 区域扩展系数
            safety_color = (255, 0, 0, 40) if occupied else (
                0, 255, 0, 40)  # 匹配停车位颜色

            if obstacle_type == "rectangle":
                safety_obstacle = RectangleObstacle(x, y, width * expansion_factor, height * expansion_factor, angle,
                                                    safety_color, True, 1)
            else:  # circle
                safety_obstacle = CircleObstacle(
                    x, y, radius * expansion_factor, safety_color, True, 1)
        else:
            safety_color = (0, 0, 255, 40)
            if obstacle_type == "rectangle":
                safety_obstacle = RectangleObstacle(x, y, width + 2 * safety_margin, height + 2 * safety_margin, angle,
                                                    safety_color, True, 1)
            else:  # circle
                safety_obstacle = CircleObstacle(
                    x, y, radius + safety_margin, safety_color, True, 1)

        # 为了兼容旧代码，添加额外属性
        safety_obstacle.type = obstacle_type
        safety_obstacle.is_parking_spot = is_parking_spot
        safety_obstacle.occupied = occupied

        # 建立停车位与安全边界的引用关系，便于交互式编辑场景
        obstacle.safety_obstacle = safety_obstacle
        safety_obstacle.linked_obstacle = obstacle

        # 先添加安全边界
        self.obstacles.append(safety_obstacle)

        # 再添加实际障碍物
        self.obstacles.append(obstacle)
        return obstacle

    def add_dynamic_obstacle(self, x0, y0, vx, vy, width, height):
        """添加动态障碍物"""
        self.dynamic_obstacles.append(
            DynamicObstacle(x0, y0, vx, vy, width, height))

    def find_parking_spot(self, point, include_occupied: bool = False):
        """
        查找点所在的未占用停车位

        参数:
            point: 坐标点 (x, y)

        返回:
            如果点在未占用的停车位内，返回停车位对象；否则返回None
        """
        for i in range(0, len(self.obstacles), 2):
            obstacle = self.obstacles[i + 1]  # 实际障碍物（非安全边界）
            if hasattr(obstacle, 'is_parking_spot') and obstacle.is_parking_spot:
                if obstacle.occupied and not include_occupied:
                    continue
                # 检查点是否在这个未占用的停车位内
                if obstacle.type == "rectangle":
                    # 将点转换到矩形的局部坐标系
                    dx = point[0] - obstacle.x
                    dy = point[1] - obstacle.y

                    # 旋转点
                    angle_rad = math.radians(obstacle.angle)
                    rotated_x = dx * \
                        math.cos(-angle_rad) - dy * math.sin(-angle_rad)
                    rotated_y = dx * \
                        math.sin(-angle_rad) + dy * math.cos(-angle_rad)

                    # 检查点是否在矩形内
                    if (abs(rotated_x) <= obstacle.width / 2 and abs(rotated_y) <= obstacle.height / 2):
                        return obstacle
                elif obstacle.type == "circle":
                    # 计算点到圆心的距离
                    distance = math.sqrt(
                        (point[0] - obstacle.x)**2 + (point[1] - obstacle.y)**2)

                    # 如果距离小于等于半径，则点在圆内
                    if distance <= obstacle.radius:
                        return obstacle

        return None

    def check_segment_collision(self, start, end, vehicle_width=0.0, vehicle_length=0.0):
        """
        检查线段是否与任意障碍物碰撞，考虑车辆尺寸
        重写父类方法以支持停车位逻辑

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
            for i in range(0, len(self.obstacles), 2):
                safety_obstacle = self.obstacles[i]  # 安全边界
                obstacle = self.obstacles[i+1] if i + \
                    1 < len(self.obstacles) else None  # 实际障碍物

                # 跳过未占用的停车位及其安全边界
                if obstacle and hasattr(obstacle,
                                        'is_parking_spot') and obstacle.is_parking_spot and not obstacle.occupied:
                    continue

                # 检查实际障碍物
                if obstacle:
                    obstacle_polygon = None
                    if obstacle.type == 'circle':
                        obstacle_polygon = Point(
                            obstacle.x, obstacle.y).buffer(obstacle.radius)
                    else:  # rectangle
                        # 计算矩形的角点
                        x_min = obstacle.x - obstacle.width / 2
                        x_max = obstacle.x + obstacle.width / 2
                        y_min = obstacle.y - obstacle.height / 2
                        y_max = obstacle.y + obstacle.height / 2
                        corners = [(x_min, y_min), (x_max, y_min),
                                   (x_max, y_max), (x_min, y_max)]

                        # 如果有角度，旋转角点
                        if hasattr(obstacle, 'angle') and obstacle.angle != 0:
                            angle_rad = math.radians(obstacle.angle)
                            cos_angle = math.cos(-angle_rad)
                            sin_angle = math.sin(-angle_rad)
                            rotated_corners = []
                            for x, y in corners:
                                tx = x - obstacle.x
                                ty = y - obstacle.y
                                rx = tx * cos_angle - ty * sin_angle
                                ry = tx * sin_angle + ty * cos_angle
                                rotated_corners.append(
                                    (rx + obstacle.x, ry + obstacle.y))
                            corners = rotated_corners

                        obstacle_polygon = Polygon(corners)

                    if obstacle_polygon and line.intersects(obstacle_polygon):
                        return True

        else:
            # 使用车辆尺寸进行碰撞检测
            # 计算路径方向
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            path_length = math.sqrt(dx * dx + dy * dy)

            if path_length < 1e-6:
                temp_vehicle = VehicleModel(
                    start[0],
                    start[1],
                    0.0,
                    vehicle_length,
                    vehicle_width,
                )
                collision_info = check_vehicle_collision(temp_vehicle, self)
                return collision_info['collision']

            # 计算采样点数量（根据路径长度动态调整）
            steps = max(3, int(path_length / (vehicle_width / 2)))

            # 在路径上采样点进行碰撞检测
            for i in range(steps):
                t = i / (steps - 1)
                x = start[0] + t * dx
                y = start[1] + t * dy
                angle = math.atan2(dy, dx)

                # 创建临时车辆模型
                temp_vehicle = VehicleModel(
                    x, y, angle, vehicle_length, vehicle_width)

                # 检查碰撞
                collision_info = check_vehicle_collision(temp_vehicle, self)
                if collision_info['collision']:
                    return True

        return False

    def check_segment_collision_with_time(self, start, end, start_time, end_time):
        """
        检查路径段在时间区间内是否发生碰撞

        参数:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            start_time: 起始时间
            end_time: 结束时间

        返回:
            是否发生碰撞
        """
        # 检查静态障碍物 - 使用当前类的方法而不是父类的方法
        if self.check_segment_collision(start, end):
            return True

        # 检查动态障碍物
        for dyn_obs in self.dynamic_obstacles:
            for t in np.arange(start_time, end_time, 0.1):
                robot_pos = interpolate_position(
                    start, end, start_time, end_time, t)
                obs_pos = dyn_obs.get_position_at_time(t)
                if check_collision(robot_pos, obs_pos, dyn_obs.width, dyn_obs.height):
                    return True

        return False


def interpolate_position(start, end, start_time, end_time, t):
    """插值计算机器人当前位置"""
    ratio = (t - start_time) / (end_time - start_time)
    x = start[0] + ratio * (end[0] - start[0])
    y = start[1] + ratio * (end[1] - start[1])
    return x, y


def check_collision(robot_pos, obs_pos, obs_width, obs_height):
    """简化的碰撞检测：使用圆形近似"""
    robot_radius = 2.5  # 假设机器人为圆形，半径为2.5
    obs_radius = np.hypot(obs_width / 2, obs_height / 2)
    dist = np.hypot(robot_pos[0] - obs_pos[0], robot_pos[1] - obs_pos[1])
    return dist < (robot_radius + obs_radius)


def check_vehicle_collision(vehicle, env):
    """
    检查车辆与环境中障碍物的碰撞

    参数:
        vehicle: 车辆模型
        env: 环境对象

    返回:
        碰撞信息字典
    """
    from shapely.geometry import Point, Polygon
    import math

    collision_info = {
        'collision': False,
        'position': None,
        'obstacle': None,
        'distance': float('inf'),
        'safety_warning': False,
        'clearance': float('inf'),
    }

    # 获取车辆四个角的坐标
    corners = vehicle.get_corners()

    # 创建车辆多边形
    vehicle_polygon = Polygon(corners)

    # 检查车辆与每个障碍物的碰撞
    for i in range(0, len(env.obstacles), 2):
        safety_obstacle = env.obstacles[i]  # 安全边界
        obstacle = env.obstacles[i+1] if i + \
            1 < len(env.obstacles) else None  # 实际障碍物

        # 跳过未占用的停车位及其安全边界
        if obstacle and hasattr(obstacle, 'is_parking_spot') and obstacle.is_parking_spot and not obstacle.occupied:
            continue

        # 首先检查实际障碍物
        if obstacle:
            obstacle_polygon = None
            if obstacle.type == 'circle':
                obstacle_polygon = Point(
                    obstacle.x, obstacle.y).buffer(obstacle.radius)
            else:  # rectangle
                # 计算矩形的角点
                x_min = obstacle.x - obstacle.width / 2
                x_max = obstacle.x + obstacle.width / 2
                y_min = obstacle.y - obstacle.height / 2
                y_max = obstacle.y + obstacle.height / 2
                corners = [(x_min, y_min), (x_max, y_min),
                           (x_max, y_max), (x_min, y_max)]

                # 如果有角度，旋转角点
                if hasattr(obstacle, 'angle') and obstacle.angle != 0:
                    angle_rad = math.radians(obstacle.angle)
                    cos_angle = math.cos(-angle_rad)
                    sin_angle = math.sin(-angle_rad)
                    rotated_corners = []

                    for x, y in corners:
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
                    obstacle_polygon = Polygon(
                        [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

            if obstacle_polygon and vehicle_polygon.intersects(obstacle_polygon):
                collision_info['collision'] = True
                collision_info['position'] = (obstacle.x, obstacle.y)
                collision_info['obstacle'] = obstacle
                collision_info['distance'] = np.hypot(
                    vehicle.x - obstacle.x, vehicle.y - obstacle.y)
                collision_info['clearance'] = 0.0
                return collision_info  # 发生实际碰撞立即返回

            if obstacle_polygon is not None:
                collision_info['clearance'] = min(
                    collision_info['clearance'],
                    float(vehicle_polygon.distance(obstacle_polygon)),
                )

        # 检查安全边界（如果没有发生实际碰撞）
        if hasattr(safety_obstacle, 'type') and safety_obstacle.type == 'circle':
            # 圆形安全边界
            safety_circle = Point(safety_obstacle.x, safety_obstacle.y).buffer(
                safety_obstacle.radius)
            if vehicle_polygon.intersects(safety_circle):
                if not collision_info['collision']:  # 只有在没有实际碰撞的情况下才更新
                    collision_info['safety_warning'] = True
                    collision_info['position'] = (
                        safety_obstacle.x, safety_obstacle.y)
                    collision_info['obstacle'] = safety_obstacle
                    collision_info['distance'] = np.hypot(
                        vehicle.x - safety_obstacle.x, vehicle.y - safety_obstacle.y)
                    collision_info['clearance'] = min(collision_info['clearance'], 0.0)
            else:
                collision_info['clearance'] = min(
                    collision_info['clearance'],
                    float(vehicle_polygon.distance(safety_circle)),
                )

    return collision_info


def check_path_collision(path, env, vehicle_length, vehicle_width, steps=10):
    """
    检查路径是否与障碍物碰撞

    参数:
        path: 路径点列表
        env: 环境对象
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
        steps: 每段路径的采样点数

    返回:
        collision_info: 碰撞信息字典
    """
    import math

    if len(path) < 2:
        return {
            'collision': False,
            'position': None,
            'obstacle': None,
            'distance': float('inf'),
            'safety_warning': False
        }

    collision_result = {
        'collision': False,
        'position': None,
        'obstacle': None,
        'distance': float('inf'),
        'safety_warning': False
    }

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]

        # 计算当前段的方向
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.sqrt(dx * dx + dy * dy)

        if segment_length < 1e-6:  # 避免除以零
            continue

        # 计算方向角度
        angle = math.atan2(dy, dx)

        # 在路径段上采样点进行碰撞检测
        for j in range(steps):
            t = j / (steps - 1) if steps > 1 else 0
            x = start[0] + t * dx
            y = start[1] + t * dy

            # 创建临时车辆模型进行碰撞检测
            temp_vehicle = VehicleModel(
                x, y, angle, vehicle_length, vehicle_width)
            temp_result = check_vehicle_collision(temp_vehicle, env)

            if temp_result['collision']:
                return temp_result

            # 更新安全警告状态
            if temp_result['safety_warning']:
                collision_result = temp_result
                # 在当前点附近增加额外的检查点
                for angle_offset in [-0.1, 0.1]:  # 小角度偏移
                    temp_angle = angle + angle_offset
                    temp_vehicle = VehicleModel(
                        x, y, temp_angle, vehicle_length, vehicle_width)
                    detailed_result = check_vehicle_collision(
                        temp_vehicle, env)
                    if detailed_result['collision']:
                        return detailed_result

    return collision_result


def check_segment_collision(start: Tuple[float, float],
                            end: Tuple[float, float],
                            env: Environment,
                            vehicle_length: float,
                            vehicle_width: float,
                            steps: int = 5) -> bool:
    """
    检查单个直线段是否与环境中的障碍物碰撞，考虑车辆尺寸。
    通过在段上采样点并检查每个点的车辆碰撞来实现。

    参数:
        start: 线段起点坐标 (x, y)
        end: 线段终点坐标 (x, y)
        env: 环境对象
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
        steps: 在段上采样的点数 (至少为2)

    返回:
        如果发生碰撞则返回 True，否则返回 False
    """
    import math
    # 确保steps至少为2，以检查起点和终点
    steps = max(2, steps)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    segment_length = math.sqrt(dx * dx + dy * dy)

    if segment_length < 1e-6:  # 如果段长度几乎为零，则不碰撞
        return False

    # 计算方向角度
    angle = math.atan2(dy, dx)

    # 在路径段上采样点进行碰撞检测
    for i in range(steps):
        t = i / (steps - 1)  # t 从 0 到 1
        x = start[0] + t * dx
        y = start[1] + t * dy

        # 创建临时车辆模型进行碰撞检测
        # 注意：这里假设车辆沿直线段移动，朝向固定为段的方向
        temp_vehicle = VehicleModel(x, y, angle, vehicle_length, vehicle_width)
        collision_info = check_vehicle_collision(temp_vehicle, env)

        if collision_info['collision']:
            # print(f"Segment collision detected at point ({x:.2f}, {y:.2f}) on segment {start} -> {end}") # Debugging
            return True  # 发生碰撞立即返回True

    return False  # 循环完成没有碰撞


class PygameSimulator:
    """基于Pygame的车辆仿真器"""

    @staticmethod
    def configure_rendering_environment() -> None:
        """优先启用更稳妥的 SDL 软件渲染配置。"""
        os.environ.setdefault('SDL_RENDER_DRIVER', 'software')
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.environ.setdefault('SDL_VIDEO_X11_FORCE_EGL', '0')

    @staticmethod
    def configure_text_output(log_path: Optional[str] = None) -> Optional[str]:
        """尽量统一控制台输出为 UTF-8。"""
        platform_is_windows = platform.system().lower().startswith("win")
        has_console = False
        for stream in (sys.stdout, sys.stderr):
            if stream is None:
                continue
            isatty = getattr(stream, "isatty", None)
            if callable(isatty):
                try:
                    has_console = has_console or bool(isatty())
                except Exception:
                    pass
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    continue
        if platform_is_windows and not has_console:
            log_path = log_path or _DEFAULT_LOG_FILE
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            try:
                log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            except OSError:
                return None
            if getattr(sys, "stdout", None) is None:
                sys.stdout = log_handle
            if getattr(sys, "stderr", None) is None:
                sys.stderr = log_handle
            return log_path
        return None

    def __init__(self, config_input: Optional[Union[str, Dict]] = None):
        """
        初始化仿真器

        参数:
            config_input: 配置文件路径(str)或配置字典(Dict)
        """
        self.log_path = self.configure_text_output()
        # 加载配置
        self.config = self._load_config(config_input)
        configure_font_preferences(self.config.get('ui', {}).get('fonts', {}))
        self.configure_rendering_environment()

        # 初始化pygame
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()

        # 设置窗口尺寸和比例
        self.scale = self.config.get('scale', 5)  # 像素/米
        self.width = self.config.get('window_width', 1000)
        self.height = self.config.get('window_height', 800)

        # 创建窗口
        try:
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                pygame.SWSURFACE,
            )
        except pygame.error as exc:
            raise RuntimeError(
                f"无法创建Pygame窗口（已尝试 SDL 软件渲染兼容模式）: {exc}"
            ) from exc
        pygame.display.set_caption(self.config.get(
            'window_title', 'RRT-Pygame 仿真器'))

        # 创建时钟对象
        self.clock = pygame.time.Clock()

        # 获取车辆配置
        vehicle_config = self.config.get('vehicle', {})

        # 初始化车辆和环境
        self.environment = None
        self.vehicle = VehicleModel(length=vehicle_config.get(
            'length', 4.5), width=vehicle_config.get('width', 1.8))
        self._apply_vehicle_config(vehicle_config)

        self.follower = PathFollower(lookahead=self.config.get('lookahead', 5.0),
                                     control_method=self.config.get('control_method', 'default'))
        self.follower.configure_terminal_parking(
            self.config.get('parking', {}).get('final_pose', {})
        )

        # 仿真状态
        self.running = False
        self.paused = False
        self.collision_detected = False
        self.status_text = ""
        self.status_color = (0, 0, 0)

        # 控制方法
        self.control_methods = ["default", "pid", "mpc", "lqr"]
        self.current_control_method = self.config.get(
            'control_method', 'default')

        # 记录数据
        self.simulation_data = {
            'time': [],
            'position_x': [],
            'position_y': [],
            'heading': [],
            'speed': [],
            'steer_angle': [],
            'acceleration': [],
            'path': []
        }

        # 坐标转换参数
        self.offset_x = self.width / 2
        self.offset_y = self.height / 2

        # 添加按键提示信息
        self.key_hints = ["R: 重置车辆", "C: 切换控制方法",
                          "P: 切换规划算法", "S: 切换转向模式", "空格: 暂停/继续", "右键: 选择目标点"]
        self.hint_color = (50, 50, 50)  # 深灰色
        self.hint_font_size = 20

    def _apply_vehicle_config(self, vehicle_config: Dict[str, Any]) -> None:
        dynamics = vehicle_config.get('dynamics', {})
        render = vehicle_config.get('render', {})
        wheelbase = vehicle_config.get(
            'wheelbase', vehicle_config.get('wheel_base', 2.7))
        max_brake = dynamics.get(
            'max_brake',
            vehicle_config.get('max_brake', vehicle_config.get('max_decel', 4.0)),
        )
        max_steer = dynamics.get(
            'max_steer',
            vehicle_config.get('max_steer', vehicle_config.get('max_steer_angle', 0.7854)),
        )

        self.vehicle.wheelbase = wheelbase
        self.vehicle.wheel_base = wheelbase
        self.vehicle.max_speed = dynamics.get('max_speed', vehicle_config.get('max_speed', 5.0))
        self.vehicle.max_reverse_speed = dynamics.get('max_reverse_speed', vehicle_config.get('max_reverse_speed', 2.5))
        self.vehicle.max_accel = dynamics.get('max_accel', vehicle_config.get('max_accel', 2.0))
        self.vehicle.max_brake = max_brake
        self.vehicle.max_decel = max_brake
        self.vehicle.max_steer = max_steer
        self.vehicle.max_steer_angle = max_steer
        self.vehicle.steer_response = dynamics.get('steer_rate', dynamics.get('steer_response', self.vehicle.steer_response))
        self.vehicle.rolling_resistance = dynamics.get('rolling_resistance', self.vehicle.rolling_resistance)
        self.vehicle.drag_coefficient = dynamics.get('drag_coefficient', self.vehicle.drag_coefficient)
        self.vehicle.creep_speed = dynamics.get('creep_speed', self.vehicle.creep_speed)
        self.vehicle.creep_accel = dynamics.get('creep_accel', self.vehicle.creep_accel)
        self.vehicle.jerk_limit = dynamics.get('jerk_limit', self.vehicle.jerk_limit)
        self.vehicle.throttle_response = dynamics.get('throttle_response', self.vehicle.throttle_response)
        self.vehicle.brake_response = dynamics.get('brake_response', self.vehicle.brake_response)
        self.vehicle.steer_speed_sensitivity = dynamics.get(
            'steer_speed_sensitivity',
            self.vehicle.steer_speed_sensitivity,
        )
        self.vehicle.render_style = render

    def _load_config(self, config_input: Optional[Union[str, Dict]]) -> Dict:
        """加载配置文件或配置字典"""
        default_config = {
            'scale': 5,
            'window_width': 1000,
            'window_height': 800,
            'window_title': 'RRT-Pygame 仿真器',
            'fps': 60,
            'dt': 0.05,  # 仿真时间步长(秒)
            'lookahead': 5.0,  # 路径跟踪前瞻距离
            'control_method': 'default',  # 控制方法: default, pid, mpc, lqr, parking
            'ui': {
                'fonts': {},
            },
            'parking': {
                'final_pose': {
                    'enabled': True,
                    'trigger_distance': 3.2,
                    'position_tolerance': 0.25,
                    'heading_tolerance_deg': 5.0,
                    'stop_speed_tolerance': 0.05,
                    'max_duration': 18.0,
                    'minimum_clearance': 0.2,
                },
            },
            'vehicle': {
                'length': 4.5,
                'width': 1.8,
                'wheelbase': 2.7,
                'wheel_base': 2.7,
                'max_speed': 20.0,
                'max_accel': 2.0,
                'max_brake': 4.0,
                'max_decel': 4.0,
                'max_steer': 0.7854,
                'max_steer_angle': 0.7854,  # π/4
                'dynamics': {
                    'max_speed': 20.0,
                    'max_reverse_speed': 2.5,
                    'max_accel': 2.0,
                    'max_brake': 4.0,
                    'max_steer': 0.7854,
                    'steer_rate': math.pi,
                    'rolling_resistance': 0.2,
                    'drag_coefficient': 0.015,
                    'creep_speed': 0.35,
                    'creep_accel': 0.45,
                    'jerk_limit': 6.0,
                    'throttle_response': 2.4,
                    'brake_response': 4.0,
                    'steer_speed_sensitivity': 0.08,
                },
                'render': {
                    'body_color': [46, 160, 109, 255],
                    'roof_color': [73, 178, 128, 245],
                    'window_color': [200, 225, 234, 220],
                    'window_shadow': [93, 121, 138, 185],
                    'trim_color': [25, 34, 42, 255],
                    'wheel_color': [32, 35, 38, 255],
                    'wheel_hub_color': [184, 191, 198, 255],
                    'shadow_color': [18, 22, 24, 70],
                    'headlight_color': [255, 244, 196, 230],
                    'taillight_color': [220, 63, 52, 230],
                    'brake_color': [255, 76, 76, 255],
                    'reverse_color': [146, 228, 255, 255],
                    'outline_color': [15, 26, 31, 255],
                },
            }
        }

        if isinstance(config_input, str) and os.path.exists(config_input):
            try:
                with open(config_input, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    # 更新默认配置
                    self._update_config(default_config, user_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        elif isinstance(config_input, dict):
            # 直接使用配置字典更新默认配置
            self._update_config(default_config, config_input)

        return default_config

    def _update_config(self, default_config: Dict, user_config: Dict) -> None:
        """递归更新配置字典"""
        for key, value in user_config.items():
            if isinstance(value, dict) and key in default_config:
                self._update_config(default_config[key], value)
            else:
                default_config[key] = value

    def set_environment(self, env: Environment) -> None:
        """设置环境"""
        self.environment = env

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """将世界坐标转换为屏幕坐标"""
        # 坐标系转换：原点移到屏幕中心，y轴朝上
        screen_x = int(x * self.scale + self.width / 2)
        screen_y = int(self.height / 2 - y * self.scale)
        return screen_x, screen_y

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """将屏幕坐标转换为世界坐标"""
        x = (screen_x - self.width / 2) / self.scale
        y = (self.height / 2 - screen_y) / self.scale
        return x, y

    def _draw_vehicle(self, screen, vehicle, scale=1.0, offset_x=0, offset_y=0, color=None):
        """
        绘制车辆
        """
        render_style = getattr(vehicle, "render_style", None) or self.config.get("vehicle", {}).get("render", {})

        def rgb(name: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
            value = render_style.get(name)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return tuple(int(component) for component in value[:3])
            return fallback

        def rgba(name: str, fallback: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
            value = render_style.get(name)
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                return tuple(int(component) for component in value[:4])
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return tuple(int(component) for component in value[:3]) + (fallback[3],)
            return fallback

        def local_to_screen(local_x: float, local_y: float) -> Tuple[int, int]:
            world_x = vehicle.x + local_x * cos_h - local_y * sin_h
            world_y = vehicle.y + local_x * sin_h + local_y * cos_h
            return (
                int(round(world_x * scale + offset_x)),
                int(round(world_y * scale + offset_y)),
            )

        cos_h = math.cos(vehicle.heading)
        sin_h = math.sin(vehicle.heading)
        body_corners = [
            (
                int(round(world_x * scale + offset_x)),
                int(round(world_y * scale + offset_y)),
            )
            for world_x, world_y in vehicle.get_corners()
        ]

        body_color = tuple(int(component) for component in color[:3]) if color is not None else rgb("body_color", (46, 160, 109))
        roof_color = rgb("roof_color", (73, 178, 128))
        window_color = rgb("window_color", (200, 225, 234))
        window_shadow = rgb("window_shadow", (93, 121, 138))
        outline_color = rgb("outline_color", (15, 26, 31))
        trim_color = rgb("trim_color", (25, 34, 42))
        wheel_color = rgb("wheel_color", (32, 35, 38))
        wheel_hub_color = rgb("wheel_hub_color", (184, 191, 198))
        headlight_color = rgb("headlight_color", (255, 244, 196))
        taillight_color = rgb("taillight_color", (220, 63, 52))
        brake_color = rgb("brake_color", (255, 76, 76))
        reverse_color = rgb("reverse_color", (146, 228, 255))
        shadow_color = rgba("shadow_color", (18, 22, 24, 70))

        shadow_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        shadow_offset_x = max(3, int(round(scale * 0.18)))
        shadow_offset_y = max(5, int(round(scale * 0.28)))
        shadow_points = [(x + shadow_offset_x, y + shadow_offset_y) for x, y in body_corners]
        pygame.draw.polygon(shadow_surface, shadow_color, shadow_points)
        screen.blit(shadow_surface, (0, 0))

        pygame.draw.polygon(screen, body_color, body_corners)
        pygame.draw.polygon(screen, outline_color, body_corners, max(2, int(round(scale * 0.04))))

        roof_local = [
            (vehicle.length * 0.22, vehicle.width * 0.34),
            (vehicle.length * 0.30, -vehicle.width * 0.34),
            (-vehicle.length * 0.02, -vehicle.width * 0.30),
            (-vehicle.length * 0.12, vehicle.width * 0.30),
        ]
        roof_screen = [local_to_screen(lx, ly) for lx, ly in roof_local]
        pygame.draw.polygon(screen, roof_color, roof_screen)
        pygame.draw.polygon(screen, outline_color, roof_screen, 2)

        windshield_local = [
            (vehicle.length * 0.31, vehicle.width * 0.26),
            (vehicle.length * 0.34, -vehicle.width * 0.26),
            (vehicle.length * 0.18, -vehicle.width * 0.28),
            (vehicle.length * 0.14, vehicle.width * 0.28),
        ]
        rear_glass_local = [
            (-vehicle.length * 0.01, vehicle.width * 0.28),
            (vehicle.length * 0.08, -vehicle.width * 0.28),
            (-vehicle.length * 0.10, -vehicle.width * 0.26),
            (-vehicle.length * 0.18, vehicle.width * 0.26),
        ]
        pygame.draw.polygon(screen, window_shadow, [local_to_screen(lx, ly) for lx, ly in windshield_local])
        pygame.draw.polygon(screen, window_color, [local_to_screen(lx, ly) for lx, ly in rear_glass_local])
        pygame.draw.line(
            screen,
            trim_color,
            local_to_screen(vehicle.length * 0.11, vehicle.width * 0.30),
            local_to_screen(vehicle.length * 0.03, -vehicle.width * 0.30),
            2,
        )

        # 获取车轮位置和角度
        wheels = vehicle.get_wheel_positions()

        # 绘制车轮
        for wheel_x, wheel_y, wheel_angle in wheels:
            # 计算车轮的四个角
            wheel_half_length = vehicle.wheel_length / 2
            wheel_half_width = vehicle.wheel_width / 2

            # 车轮在自身坐标系中的四个角
            wheel_corners_local = [
                (wheel_half_length, wheel_half_width),  # 右前
                (wheel_half_length, -wheel_half_width),  # 左前
                (-wheel_half_length, -wheel_half_width),  # 左后
                (-wheel_half_length, wheel_half_width)  # 右后
            ]

            # 转换到世界坐标系，考虑车轮自身的转向角
            cos_w = math.cos(wheel_angle)
            sin_w = math.sin(wheel_angle)

            wheel_corners_screen = []
            for lx, ly in wheel_corners_local:
                wx = wheel_x + lx * cos_w - ly * sin_w
                wy = wheel_y + lx * sin_w + ly * cos_w
                sx = wx * scale + offset_x
                sy = wy * scale + offset_y
                wheel_corners_screen.append((int(sx), int(sy)))

            pygame.draw.polygon(screen, wheel_color, wheel_corners_screen)
            pygame.draw.polygon(screen, outline_color, wheel_corners_screen, 1)

            hub_x = int(wheel_x * scale + offset_x)
            hub_y = int(wheel_y * scale + offset_y)
            pygame.draw.circle(screen, wheel_hub_color, (hub_x, hub_y), max(2, int(round(scale * 0.06))))

        # 绘制车灯
        light_radius = vehicle.width * 0.1
        light_offset_y = vehicle.width * 0.3

        # 前灯位置 (黄色)
        front_light_local = [
            (vehicle.length / 2 - light_radius, light_offset_y),  # 右前灯
            (vehicle.length / 2 - light_radius, -light_offset_y)  # 左前灯
        ]

        for lx, ly in front_light_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = int(wx * scale + offset_x)
            sy = int(wy * scale + offset_y)
            pygame.draw.circle(screen, headlight_color,
                               (sx, sy), int(light_radius * scale))

        # 后灯位置 (红色)
        rear_light_local = [
            (-vehicle.length / 2 + light_radius, light_offset_y),  # 右后灯
            (-vehicle.length / 2 + light_radius, -light_offset_y)  # 左后灯
        ]

        for lx, ly in rear_light_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = int(wx * scale + offset_x)
            sy = int(wy * scale + offset_y)
            pygame.draw.circle(
                screen,
                brake_color if getattr(vehicle, "last_brake", 0.0) > 0.1 else taillight_color,
                (sx, sy),
                int(light_radius * scale),
            )
            if getattr(vehicle, "reverse", False):
                pygame.draw.circle(screen, reverse_color, (sx, sy), max(2, int(light_radius * scale * 0.55)))

        gear_label = vehicle.get_gear_label() if hasattr(vehicle, "get_gear_label") else ("R" if getattr(vehicle, "reverse", False) else "D")
        gear_color = {
            "D": (73, 190, 122),
            "R": (92, 184, 255),
            "P": (255, 188, 79),
        }.get(gear_label, (73, 190, 122))
        center_screen = (int(vehicle.x * scale + offset_x), int(vehicle.y * scale + offset_y))
        pygame.draw.circle(screen, gear_color, center_screen, max(4, int(round(scale * 0.09))))
        pygame.draw.circle(screen, outline_color, center_screen, max(4, int(round(scale * 0.09))), 1)
        gear_font = get_font(max(12, int(round(scale * 0.22))), role="mono")
        gear_surface = gear_font.render(gear_label, True, WHITE)
        gear_rect = gear_surface.get_rect(center=center_screen)
        screen.blit(gear_surface, gear_rect)

        # 仅当show_sensors为True时绘制传感器
        if hasattr(vehicle, 'show_sensors') and vehicle.show_sensors:
            # 绘制传感器
            sensor_positions = vehicle.get_sensor_positions()

            # 绘制环视摄像头 (黄色)
            for camera in sensor_positions['fisheye_cameras']:
                pos = camera['pos']
                color = camera['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                pygame.draw.circle(screen, color, (sx, sy), 5)
                # 绘制摄像头视野范围指示
                pygame.draw.circle(screen, color, (sx, sy), 15, 1)

            # 绘制前视摄像头 (红色)
            if sensor_positions['front_camera']:
                pos = sensor_positions['front_camera']['pos']
                color = sensor_positions['front_camera']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                pygame.draw.circle(screen, color, (sx, sy), 4)
                # 绘制摄像头视野范围
                view_length = vehicle.length * 0.8
                view_x = pos[0] + math.cos(vehicle.heading) * view_length
                view_y = pos[1] + math.sin(vehicle.heading) * view_length
                view_sx = int(view_x * scale + offset_x)
                view_sy = int(view_y * scale + offset_y)
                pygame.draw.line(screen, color, (sx, sy),
                                 (view_sx, view_sy), 1)

            # 绘制超声波雷达 (紫色)
            for sensor in sensor_positions['ultrasonic']:
                pos = sensor['pos']
                color = sensor['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制超声波雷达点
                pygame.draw.circle(screen, color, (sx, sy), 3)

                # 计算超声波雷达方向 - 从车辆中心指向传感器
                sensor_angle = math.atan2(
                    pos[1] - vehicle.y, pos[0] - vehicle.x)
                # 绘制超声波雷达探测范围
                range_length = 1.0  # 探测范围1米
                range_x = pos[0] + math.cos(sensor_angle) * range_length
                range_y = pos[1] + math.sin(sensor_angle) * range_length
                range_sx = int(range_x * scale + offset_x)
                range_sy = int(range_y * scale + offset_y)
                pygame.draw.line(screen, color, (sx, sy),
                                 (range_sx, range_sy), 1)

            # 绘制IMU (绿色)
            if sensor_positions['imu']:
                pos = sensor_positions['imu']['pos']
                color = sensor_positions['imu']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制IMU为一个小方块
                imu_size = 4
                pygame.draw.rect(
                    screen, color, (sx - imu_size // 2, sy - imu_size // 2, imu_size, imu_size))

            # 绘制GPS (浅绿色)
            if sensor_positions['gps']:
                pos = sensor_positions['gps']['pos']
                color = sensor_positions['gps']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制GPS为一个十字形
                cross_size = 5
                pygame.draw.line(
                    screen, color, (sx - cross_size, sy), (sx + cross_size, sy), 2)
                pygame.draw.line(
                    screen, color, (sx, sy - cross_size), (sx, sy + cross_size), 2)

    def _draw_environment(self) -> None:
        """绘制环境"""
        if not self.environment:
            return

        # 绘制边界
        border_width = 2
        pygame.draw.rect(self.screen, BLACK, (0, 0, self.width, border_width))
        pygame.draw.rect(self.screen, BLACK, (0, 0, border_width, self.height))
        pygame.draw.rect(self.screen, BLACK, (0, self.height -
                         border_width, self.width, border_width))
        pygame.draw.rect(self.screen, BLACK, (self.width -
                         border_width, 0, border_width, self.height))

        # 绘制网格
        grid_size = 10 * self.scale  # 10米一格

        for x in range(0, self.width, int(grid_size)):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.height), 1)

        for y in range(0, self.height, int(grid_size)):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.width, y), 1)

        # 绘制坐标轴
        origin = self.world_to_screen(0, 0)
        x_axis = self.world_to_screen(10, 0)
        y_axis = self.world_to_screen(0, 10)

        pygame.draw.line(self.screen, RED, origin, x_axis, 2)
        pygame.draw.line(self.screen, GREEN, origin, y_axis, 2)

        # 绘制障碍物
        for obstacle in self.environment.obstacles:
            if hasattr(obstacle, 'radius'):  # 圆形障碍物
                center = self.world_to_screen(obstacle.x, obstacle.y)
                radius = int(obstacle.radius * self.scale)
                pygame.draw.circle(self.screen, BLACK, center, radius)
            elif hasattr(obstacle, 'width') and hasattr(obstacle, 'height'):  # 矩形障碍物
                top_left = self.world_to_screen(
                    obstacle.x - obstacle.width / 2, obstacle.y + obstacle.height / 2)
                rect_width = int(obstacle.width * self.scale)
                rect_height = int(obstacle.height * self.scale)
                pygame.draw.rect(
                    self.screen, BLACK, (top_left[0], top_left[1], rect_width, rect_height))

    def _draw_path(self, path: List[Tuple[float, float]], color: Tuple[int, int, int] = BLUE, width: int = 2) -> None:
        """绘制路径"""
        if not path:
            return

        screen_points = [self.world_to_screen(x, y) for x, y in path]

        # 绘制路径线
        if len(screen_points) > 1:
            pygame.draw.lines(self.screen, color, False, screen_points, width)

        # 绘制路径点
        for point in screen_points:
            pygame.draw.circle(self.screen, color, point, 3)

    def _draw_trajectory(self) -> None:
        """绘制车辆轨迹"""
        if not self.vehicle.trajectory:
            return

        screen_points = [self.world_to_screen(
            x, y) for x, y in self.vehicle.trajectory]

        # 绘制轨迹线
        if len(screen_points) > 1:
            pygame.draw.lines(self.screen, GREEN, False, screen_points, 2)

    def _draw_info(self) -> None:
        """绘制信息"""
        font = get_font(18, role="ui")

        # 绘制控制方法信息
        control_text = f"控制方法: {self.current_control_method}"
        control_surface = font.render(control_text, True, BLACK)
        self.screen.blit(control_surface, (10, 10))

        # 绘制车辆信息
        speed_text = f"速度: {self.vehicle.speed:.2f} m/s"
        speed_surface = font.render(speed_text, True, BLACK)
        self.screen.blit(speed_surface, (10, 40))

        steer_text = f"转向角: {self.vehicle.steer_angle:.2f} rad"
        steer_surface = font.render(steer_text, True, BLACK)
        self.screen.blit(steer_surface, (10, 70))

        # 绘制碰撞状态
        if self.collision_detected:
            collision_text = "碰撞警告！"
            collision_surface = font.render(collision_text, True, RED)
            self.screen.blit(collision_surface, (10, 100))

        # 绘制操作提示
        help_text = "空格: 暂停/继续 | R: 重置位置 | C: 切换控制方法 | T: 重新规划路径并重置位置"
        help_surface = font.render(help_text, True, BLACK)
        self.screen.blit(help_surface, (10, self.height - 30))

        # 显示临时消息
        if self.message and time.time() - self.message_time < self.message_duration:
            message_surface = font.render(self.message, True, RED)
            self.screen.blit(message_surface, (self.width // 2 - message_surface.get_width() // 2,
                                               self.height // 2 - message_surface.get_height() // 2))

    def load_custom_road(self, road_file: str) -> bool:
        """加载自定义路面文件（兼容接口）"""
        print(f"注意: 在Pygame仿真中，直接使用环境对象，无需加载CarSim路面文件")
        return True

    def execute_path(self, path: List[Tuple[float, float]]) -> bool:
        """
        执行路径跟踪

        参数:
            path: 路径点列表，每个点为(x, y)坐标

        返回:
            执行成功与否的布尔值
        """
        if not path:
            print("路径为空，无法执行")
            return False

        # 设置初始位置和朝向
        self.vehicle.x, self.vehicle.y = path[0]

        if len(path) > 1:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            self.vehicle.heading = math.atan2(dy, dx)

        # 重置轨迹
        self.vehicle.trajectory = [(self.vehicle.x, self.vehicle.y)]

        # 设置路径
        self.follower.set_path(path)

        # 保存原始路径用于重新规划
        self.original_path = path.copy()
        self.start_point = path[0]
        self.goal_point = path[-1]

        # 开始仿真
        self.running = True
        self.paused = False

        # 重置数据记录
        self.simulation_data = {
            'time': [],
            'position_x': [],
            'position_y': [],
            'heading': [],
            'speed': [],
            'steer_angle': [],
            'acceleration': []
        }

        self.start_time = time.time()

        # 运行主循环
        return self._run_simulation()

    def regenerate_path(self) -> bool:
        """
        重新规划路径

        使用原始起点作为新的起点，原始目标点作为终点，重新规划路径
        并将车辆位置重置到原始起点

        返回:
            重新规划是否成功
        """
        if not hasattr(self, 'original_path') or not self.original_path:
            self.message = "没有原始路径，无法重新规划"
            self.message_time = time.time()
            print(self.message)
            return False

        if not self.environment:
            self.message = "没有设置环境，无法重新规划"
            self.message_time = time.time()
            print(self.message)
            return False

        # 使用原始起点作为新的起点
        start = self.start_point

        # 使用原始目标点作为终点
        goal = self.goal_point

        self.message = "正在重新规划路径..."
        self.message_time = time.time()
        print(f"重新规划路径: 从 {start} 到 {goal}")

        try:
            # 导入RRT*算法
            from rrt.rrt_star import RRTStar

            # 创建规划器
            planner = RRTStar(
                start=start,
                goal=goal,
                env=self.environment,
                max_iterations=1000,  # 可以根据需要调整
                goal_sample_rate=0.1  # 增加目标采样率以加快规划
            )

            # 规划新路径
            new_path = planner.plan()

            if not new_path:
                self.message = "重新规划失败，未找到可行路径"
                self.message_time = time.time()
                print(self.message)
                return False

            self.message = f"重新规划成功，新路径包含 {len(new_path)} 个点"
            self.message_time = time.time()
            print(self.message)

            # 更新路径
            self.follower.set_path(new_path)

            # 重置车辆位置到原始起点
            self.vehicle.x, self.vehicle.y = self.start_point

            # 设置车辆朝向
            if len(new_path) > 1:
                dx = new_path[1][0] - new_path[0][0]
                dy = new_path[1][1] - new_path[0][1]
                self.vehicle.heading = math.atan2(dy, dx)

            # 重置车辆速度和加速度
            self.vehicle.v = 0.0
            self.vehicle.a = 0.0
            self.vehicle.steer_angle = 0.0

            # 重置轨迹
            self.vehicle.trajectory = [(self.vehicle.x, self.vehicle.y)]

            # 重置路径跟踪器的目标点索引
            self.follower.current_target_idx = 0

            return True

        except ImportError:
            self.message = "无法导入RRT*算法，请确保rrt模块可用"
            self.message_time = time.time()
            print(self.message)
            return False
        except Exception as e:
            self.message = f"重新规划路径时出错: {e}"
            self.message_time = time.time()
            print(self.message)
            return False

    def _run_simulation(self) -> bool:
        """运行仿真主循环"""
        dt = self.config.get('dt', 0.05)
        fps = self.config.get('fps', 60)

        try:
            while self.running:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_r:
                            # 重置车辆位置
                            if self.follower.path:
                                self.vehicle.x, self.vehicle.y = self.follower.path[0]
                                self.vehicle.trajectory = [
                                    (self.vehicle.x, self.vehicle.y)]
                                self.follower.current_target_idx = 0
                                self.collision_detected = False  # 重置碰撞状态
                        elif event.key == pygame.K_c:
                            # 切换控制方法
                            self.current_control_method = self.control_methods[
                                (self.control_methods.index(self.current_control_method) + 1) %
                                len(self.control_methods)]
                            self.follower.set_control_method(
                                self.current_control_method)
                        elif event.key == pygame.K_t:
                            # 重新规划路径并重置车辆位置
                            if not self.paused:
                                self.message = "重新规划路径并重置车辆位置..."
                                self.message_time = time.time()
                                if self.regenerate_path():
                                    self.collision_detected = False  # 重置碰撞状态
                            else:
                                self.message = "请先取消暂停再重新规划路径"
                                self.message_time = time.time()

                if not self.paused and not self.collision_detected:
                    # 检查碰撞
                    if self.environment and self.environment.check_collision((self.vehicle.x, self.vehicle.y)):
                        self.collision_detected = True
                        self.message = "警告：发生碰撞！按R重置位置或按T重新规划路径"
                        self.message_time = time.time()
                        self.message_duration = 5  # 延长碰撞警告的显示时间
                        print(self.message)
                    else:
                        # 计算控制输入
                        throttle, brake, steer = self.follower.get_control(
                            self.vehicle,
                            dt,
                        )

                        # 更新车辆状态
                        self.vehicle.update(throttle, brake, steer, dt)

                        # 记录数据
                        current_time = time.time() - (self.start_time or time.time())
                        self.simulation_data['time'].append(current_time)
                        self.simulation_data['position_x'].append(
                            self.vehicle.x)
                        self.simulation_data['position_y'].append(
                            self.vehicle.y)
                        self.simulation_data['heading'].append(
                            self.vehicle.heading)
                        self.simulation_data['speed'].append(self.vehicle.v)
                        self.simulation_data['steer_angle'].append(
                            self.vehicle.steer_angle)
                        self.simulation_data['acceleration'].append(
                            self.vehicle.a)

                        # 检查是否到达终点
                        if self.follower.current_target_idx >= len(self.follower.path) - 1 and self.vehicle.v < 0.1:
                            print("已到达终点")
                            time.sleep(1)  # 短暂停留
                            break

                # 绘制场景
                self.screen.fill(WHITE)
                self._draw_environment()
                self._draw_path(self.follower.path)
                self._draw_trajectory()
                self._draw_vehicle(self.vehicle)
                self._draw_info()

                # 更新屏幕
                pygame.display.flip()

                # 控制帧率
                self.clock.tick(fps)

            pygame.quit()
            return True

        except Exception as e:
            print(f"仿真执行异常: {e}")
            pygame.quit()
            return False

    def get_simulation_results(self) -> Dict[str, List[float]]:
        """获取仿真结果数据"""
        return self.simulation_data

    def visualize_results(self, results: Dict[str, List[float]]) -> None:
        """可视化仿真结果"""
        try:
            import matplotlib.pyplot as plt
            matplotlib.rc("font", family="Microsoft YaHei")

            fig, axs = plt.subplots(3, 2, figsize=(12, 10))

            # 绘制位置
            axs[0, 0].plot(results['position_x'], results['position_y'])
            axs[0, 0].set_title('车辆轨迹')
            axs[0, 0].set_xlabel('X 位置 (m)')
            axs[0, 0].set_ylabel('Y 位置 (m)')
            axs[0, 0].grid(True)

            # 绘制速度
            axs[0, 1].plot(results['time'], results['speed'])
            axs[0, 1].set_title('车速')
            axs[0, 1].set_xlabel('时间 (s)')
            axs[0, 1].set_ylabel('速度 (m/s)')
            axs[0, 1].grid(True)

            # 绘制朝向
            axs[1, 0].plot(results['time'], [math.degrees(h)
                           for h in results['heading']])
            axs[1, 0].set_title('车辆朝向')
            axs[1, 0].set_xlabel('时间 (s)')
            axs[1, 0].set_ylabel('朝向角度 (度)')
            axs[1, 0].grid(True)

            # 绘制转向角
            axs[1, 1].plot(results['time'], [math.degrees(a)
                           for a in results['steer_angle']])
            axs[1, 1].set_title('转向角')
            axs[1, 1].set_xlabel('时间 (s)')
            axs[1, 1].set_ylabel('转向角度 (度)')
            axs[1, 1].grid(True)

            # 绘制加速度
            axs[2, 0].plot(results['time'], results['acceleration'])
            axs[2, 0].set_title('加速度')
            axs[2, 0].set_xlabel('时间 (s)')
            axs[2, 0].set_ylabel('加速度 (m/s²)')
            axs[2, 0].grid(True)

            # 绘制不同规划算法的路径
            axs[2, 1].plot(results['position_x'], results['position_y'])
            axs[2, 1].set_title('车辆轨迹')
            axs[2, 1].set_xlabel('X 位置 (m)')
            axs[2, 1].set_ylabel('Y 位置 (m)')
            axs[2, 1].grid(True)

            # 保留一个空白区域
            axs[2, 1].axis('off')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("无法导入matplotlib进行可视化，请确保已安装该库")

    def disconnect(self) -> None:
        """断开连接（兼容接口）"""
        if pygame.get_init():
            pygame.quit()

    def _draw_status_text(self):
        """绘制状态文本"""
        try:
            font = get_font(24, role="ui")
            if font and self.status_text:
                text_surface = font.render(
                    self.status_text, True, self.status_color)
                text_rect = text_surface.get_rect()
                text_rect.centerx = self.screen.get_rect().centerx
                text_rect.top = 10
                self.screen.blit(text_surface, text_rect)
        except Exception as e:
            print(f"字体渲染错误: {e}")

    def _handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:  # 重置
                    self._reset_simulation()
                    return True
                elif event.key == pygame.K_c:  # 切换控制方法
                    self._switch_control_method()
                    return True
                elif event.key == pygame.K_p:  # 切换规划算法
                    if hasattr(self, '_switch_planning_algorithm'):
                        self._switch_planning_algorithm()
                    return True
                elif event.key == pygame.K_s:  # 切换转向模式
                    if hasattr(self.vehicle, 'set_steering_mode'):
                        self._switch_steering_mode()
                    return True

        return True

    def _reset_simulation(self):
        """重置仿真"""
        if hasattr(self, '_cancel_planning_task'):
            try:
                self._cancel_planning_task(reason="基础重置触发，取消规划任务")
            except Exception:
                pass
        if hasattr(self, '_reset_vehicle'):
            self._reset_vehicle()
        self.paused = False
        self.collision_detected = False
        self.status_text = "仿真已重置"
        self.status_color = (0, 128, 0)

    def _switch_control_method(self):
        """切换控制方法"""
        try:
            current_index = self.control_methods.index(
                self.current_control_method)
            next_index = (current_index + 1) % len(self.control_methods)
            self.current_control_method = self.control_methods[next_index]
            self.follower.set_control_method(self.current_control_method)
            self.status_text = f"已切换到{self.current_control_method}控制方法"
            self.status_color = (0, 128, 0)
        except ValueError:
            self.current_control_method = "default"
            self.follower.set_control_method("default")
            self.status_text = "已重置为默认控制方法"
            self.status_color = (255, 165, 0)

    def _draw_hints(self):
        """绘制按键提示信息"""
        try:
            font = get_font(self.hint_font_size, role="ui")
            if not font:
                return

            # 计算所有提示的总宽度
            total_width = 0
            surfaces = []
            for hint in self.key_hints:
                surface = font.render(hint, True, self.hint_color)
                surfaces.append(surface)
                total_width += surface.get_width() + 20  # 20像素的间距

            # 计算起始x坐标，使提示居中
            start_x = (self.width - total_width) / 2
            current_x = start_x

            # 在底部绘制提示，留出20像素的边距
            y = self.height - self.hint_font_size - 20

            # 绘制每个提示，用竖线分隔
            for i, surface in enumerate(surfaces):
                self.screen.blit(surface, (current_x, y))
                current_x += surface.get_width()

                # 如果不是最后一个提示，添加分隔符
                if i < len(surfaces) - 1:
                    separator = font.render("|", True, self.hint_color)
                    current_x += 10  # 分隔符前的间距
                    self.screen.blit(separator, (current_x, y))
                    current_x += 10  # 分隔符后的间距

        except Exception as e:
            print(f"提示信息渲染错误: {e}")

    def draw(self):
        """绘制场景"""
        # 清空屏幕
        self.screen.fill((255, 255, 255))

        # 绘制环境
        if self.environment is not None:
            self._draw_environment()

        # 绘制车辆
        if hasattr(self, 'vehicle'):
            self._draw_vehicle(self.screen, self.vehicle)

        # 绘制状态文本
        self._draw_status_text()

        # 绘制按键提示
        self._draw_hints()

        # 更新显示
        pygame.display.flip()
