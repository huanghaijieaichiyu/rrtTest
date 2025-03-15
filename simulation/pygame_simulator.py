#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pygame 车辆仿真器

使用 Pygame 实现简单的车辆动力学模型和仿真环境，
作为 CarSim 的轻量级替代方案。
"""

import os
import time
import math
import matplotlib
import numpy as np
import pygame
import yaml
from typing import List, Tuple, Dict, Optional, Union, Any
from shapely.geometry import Point, Polygon, LineString

from .environment import CircleObstacle, Environment, RectangleObstacle

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# 字体设置


def get_font(size: int = 24) -> pygame.font.Font:
    """获取支持中文的字体"""
    # 尝试加载系统字体
    system_fonts = [
        # Windows 字体
        "SimHei",  # 黑体
        "Microsoft YaHei",  # 微软雅黑
        "SimSun",  # 宋体
        # Linux 字体
        "WenQuanYi Micro Hei",  # 文泉驿微米黑
        "Noto Sans CJK SC",  # Google Noto 字体
        "Droid Sans Fallback",  # Android 默认字体
        # macOS 字体
        "PingFang SC",  # 苹方
        "STHeiti"  # 华文黑体
    ]

    # 尝试按优先级加载字体
    for font_name in system_fonts:
        try:
            return pygame.font.SysFont(font_name, size)
        except:
            continue

    # 如果系统字体都不可用，尝试加载自带字体文件
    try:
        font_path = os.path.join(os.path.dirname(
            __file__), "fonts", "simhei.ttf")
        if os.path.exists(font_path):
            return pygame.font.Font(font_path, size)
    except:
        pass

    # 如果都失败了，使用默认字体
    return pygame.font.Font(None, size)


class VehicleModel:
    """增强的车辆动力学模型 - 四轮转向(4WS)模型，带传感器"""

    def __init__(self, x: float = 0, y: float = 0, heading: float = 0,
                 length: float = 4.5, width: float = 1.8):
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
        self.rear_steer_angle = 0.0   # 后轮转向角(弧度)
        self.wheelbase = 2.7  # 轴距(米)

        # 车轮参数
        self.wheel_width = 0.25 * width  # 车轮宽度
        self.wheel_length = 0.5  # 车轮长度

        # 记录轨迹
        self.trajectory = [(x, y)]

        # 车辆控制参数
        self.max_speed = 5.0  # m/s
        self.max_accel = 2.0   # m/s^2
        self.max_brake = 4.0   # m/s^2
        self.max_steer = math.pi/4  # 最大转向角(弧度)

        # 四轮转向模式
        self.steering_mode = "normal"  # 可选: "normal", "counter", "crab"
        self.rear_steer_ratio = 0.5  # 后轮转向比例 (相对于前轮)

        # 传感器配置
        self.sensors = {
            'fisheye_cameras': [],  # 环视摄像头 (黄色)
            'front_camera': None,   # 前视摄像头 (红色)
            'ultrasonic': [],       # 超声波雷达 (紫色)
            'imu': None,            # 消费级IMU (绿色)
            'gps': None             # 消费级GPS (绿色)
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
        """获取车辆四个角的坐标(用于碰撞检测和渲染)"""
        half_length = self.length / 2
        half_width = self.width / 2

        # 车辆本地坐标系中的四个角
        corners_local = [
            (half_length, half_width),   # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width),  # 左后
            (-half_length, half_width)   # 右后
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
            (half_length, half_width, self.front_steer_angle),    # 右前轮
            (half_length, -half_width, self.front_steer_angle),   # 左前轮
            (-half_length, -half_width, self.rear_steer_angle),   # 左后轮
            (-half_length, half_width, self.rear_steer_angle)     # 右后轮
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

    def update(self, throttle: float, brake: float, steer: float, dt: float) -> None:
        """
        更新车辆状态

        参数:
            throttle: 油门输入[0, 1]
            brake: 制动输入[0, 1]
            steer: 转向输入[-1, 1]
            dt: 时间步长(秒)
        """
        # 计算加速度
        if throttle > 0:
            self.acceleration = throttle * self.max_accel
        else:
            self.acceleration = 0

        if brake > 0:
            self.acceleration -= brake * self.max_brake

        # 更新速度
        self.speed += self.acceleration * dt
        self.speed = max(0, min(self.speed, self.max_speed))  # 限制速度范围

        # 更新前轮转向角
        self.front_steer_angle = steer * self.max_steer

        # 根据转向模式更新后轮转向角
        if self.steering_mode == "normal":
            # 普通模式：后轮不转向
            self.rear_steer_angle = 0
        elif self.steering_mode == "counter":
            # 反向模式：后轮反向转向，提高转弯半径
            self.rear_steer_angle = -self.front_steer_angle * self.rear_steer_ratio
        elif self.steering_mode == "crab":
            # 蟹行模式：后轮同向转向，实现横向移动
            self.rear_steer_angle = self.front_steer_angle * self.rear_steer_ratio

        # 四轮转向模型
        if abs(self.speed) > 0.1:  # 当速度足够大时才转向
            # 计算前后轮的转向角度
            front_angle = self.front_steer_angle
            rear_angle = self.rear_steer_angle

            # 计算前后轮的转向半径
            if abs(front_angle) > 1e-6:
                front_radius = self.wheelbase / math.tan(abs(front_angle))
                front_sign = 1 if front_angle > 0 else -1
            else:
                front_radius = float('inf')
                front_sign = 0

            if abs(rear_angle) > 1e-6:
                rear_radius = self.wheelbase / math.tan(abs(rear_angle))
                rear_sign = 1 if rear_angle > 0 else -1
            else:
                rear_radius = float('inf')
                rear_sign = 0

            # 计算瞬时旋转中心
            if front_radius == float('inf') and rear_radius == float('inf'):
                # 直线行驶
                angular_velocity = 0
            else:
                # 计算等效转向角和转向半径
                if rear_radius == float('inf'):
                    # 只有前轮转向
                    effective_radius = front_radius
                    effective_sign = front_sign
                elif front_radius == float('inf'):
                    # 只有后轮转向
                    effective_radius = rear_radius
                    effective_sign = rear_sign
                else:
                    # 前后轮都转向
                    # 计算等效转向半径 (简化模型)
                    if front_sign == rear_sign:
                        # 同向转向 (蟹行模式)
                        effective_radius = (
                            front_radius * rear_radius) / (front_radius + rear_radius)
                    else:
                        # 反向转向 (提高转弯半径)
                        effective_radius = (
                            front_radius * rear_radius) / abs(front_radius - rear_radius)
                    effective_sign = front_sign

                # 计算角速度
                angular_velocity = (
                    self.speed * effective_sign) / effective_radius

            # 更新位置和朝向
            self.heading += angular_velocity * dt
            self.heading = self.heading % (2 * math.pi)  # 规范化到 [0, 2π]

        # 根据当前朝向和速度更新位置
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

        # 记录轨迹
        self.trajectory.append((self.x, self.y))

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

        # PID控制参数
        self.pid_params = {
            'kp_steer': 0.7,   # 转向比例系数
            'ki_steer': 0.01,  # 转向积分系数
            'kd_steer': 0.1,   # 转向微分系数
            'kp_speed': 0.5,   # 速度比例系数
            'ki_speed': 0.01,  # 速度积分系数
            'kd_speed': 0.05   # 速度微分系数
        }
        self.steer_error_prev = 0.0
        self.steer_error_sum = 0.0
        self.speed_error_prev = 0.0
        self.speed_error_sum = 0.0

        # MPC控制参数
        self.mpc_params = {
            'horizon': 10,     # 预测步长
            'dt': 0.1,         # 时间步长
            'q_x': 1.0,        # 纵向误差权重
            'q_y': 2.0,        # 横向误差权重
            'q_heading': 3.0,  # 朝向误差权重
            'r_steer': 1.0,    # 转向输入权重
            'r_accel': 0.5     # 加速度输入权重
        }

        # LQR控制参数
        self.lqr_params = {
            'q_y': 1.0,        # 横向误差权重
            'q_heading': 2.0,  # 朝向误差权重
            'q_speed': 0.5,    # 速度误差权重
            'r_steer': 0.1,    # 转向输入权重
            'r_accel': 0.1     # 加速度输入权重
        }

    def set_path(self, path):
        """设置跟踪路径"""
        self.path = path
        self.current_target_idx = 0

    def set_control_method(self, method):
        """设置控制方法"""
        if method in ['default', 'pid', 'mpc', 'lqr', 'parking']:
            self.control_method = method
        else:
            print(f"不支持的控制方法: {method}，使用默认方法")
            self.control_method = 'default'

    def get_control(self, vehicle):
        """获取控制输入"""
        if not self.path:
            return 0.0, 0.0, 0.0  # 无路径时不动作

        if self.control_method == 'parking':
            return self._parking_control(vehicle)
        elif self.control_method == 'pid':
            return self._pid_control(vehicle)
        elif self.control_method == 'mpc':
            return self._mpc_control(vehicle)
        elif self.control_method == 'lqr':
            return self._lqr_control(vehicle)
        else:
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
        steer = heading_error / (math.pi/4)  # 假设最大转向角为π/4
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
        for i in range(self.current_target_idx,
                       min(self.current_target_idx + 30, len(self.path))):
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
        steer = (self.pid_params['kp_steer'] * heading_error +
                 self.pid_params['ki_steer'] * self.steer_error_sum +
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
            return 0.0, 0.0, 0.0

        # 获取目标点
        target_idx = self._find_target_point(vehicle)
        if target_idx >= len(self.path):
            return 0.0, 0.0, 0.0

        tx, ty = self.path[target_idx]

        # 计算到目标点的距离和角度
        dx = tx - vehicle.x
        dy = ty - vehicle.y
        distance = math.sqrt(dx*dx + dy*dy)

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
            segment_length = math.sqrt(
                (self.path[i+1][0] - self.path[i][0])**2 +
                (self.path[i+1][1] - self.path[i][1])**2
            )
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
            steer = (self.pid_params['kp_steer'] * lateral_error +
                     self.pid_params['ki_steer'] * self.steer_error_sum +
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
            steer = (self.pid_params['kp_steer'] * lateral_error +
                     self.pid_params['ki_steer'] * self.steer_error_sum +
                     self.pid_params['kd_steer'] * steer_error_diff)

            # 限制在 [-1, 1] 范围内
            steer = max(-1.0, min(1.0, steer))

            # 根据停车类型调整转向策略
            if self.parking_type == 'parallel':
                # 侧方停车：需要先倒车转向，然后回正
                if abs(heading_error) > math.pi/6:  # 如果航向偏差大，优先调整航向
                    # 增加转向力度
                    steer = max(-1.0, min(1.0, steer * 1.5))

            # 如果接近目标点，进入微调阶段
            if distance < 1.0:
                self.parking_phase = 'adjust'
                self.reverse_gear = False
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

            return throttle, brake, steer

        else:  # adjust phase
            # 微调阶段：精确调整到目标位置
            if distance < self.safe_distance:
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
        occupied=False,         # 新增占用状态
        is_filled=True,         # 是否填充
        line_width=1            # 线宽
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
                safety_obstacle = RectangleObstacle(
                    x, y,
                    width * expansion_factor,
                    height * expansion_factor,
                    angle,
                    safety_color,
                    True,
                    1
                )
            else:  # circle
                safety_obstacle = CircleObstacle(
                    x, y,
                    radius * expansion_factor,
                    safety_color,
                    True,
                    1
                )
        else:
            safety_color = (0, 0, 255, 40)
            if obstacle_type == "rectangle":
                safety_obstacle = RectangleObstacle(
                    x, y,
                    width + 2 * safety_margin,
                    height + 2 * safety_margin,
                    angle,
                    safety_color,
                    True,
                    1
                )
            else:  # circle
                safety_obstacle = CircleObstacle(
                    x, y,
                    radius + safety_margin,
                    safety_color,
                    True,
                    1
                )

        # 为了兼容旧代码，添加额外属性
        safety_obstacle.type = obstacle_type
        safety_obstacle.is_parking_spot = is_parking_spot
        safety_obstacle.occupied = occupied

        # 先添加安全边界
        self.obstacles.append(safety_obstacle)

        # 再添加实际障碍物
        self.obstacles.append(obstacle)
        return obstacle

    def add_dynamic_obstacle(self, x0, y0, vx, vy, width, height):
        """添加动态障碍物"""
        self.dynamic_obstacles.append(
            DynamicObstacle(x0, y0, vx, vy, width, height))

    def find_parking_spot(self, point):
        """
        查找点所在的未占用停车位

        参数:
            point: 坐标点 (x, y)

        返回:
            如果点在未占用的停车位内，返回停车位对象；否则返回None
        """
        for i in range(0, len(self.obstacles), 2):
            obstacle = self.obstacles[i+1]  # 实际障碍物（非安全边界）
            if hasattr(obstacle, 'is_parking_spot') and obstacle.is_parking_spot and not obstacle.occupied:
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
                        (point[0] - obstacle.x) ** 2 + (point[1] - obstacle.y) ** 2)

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
                if obstacle and hasattr(obstacle, 'is_parking_spot') and obstacle.is_parking_spot and not obstacle.occupied:
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

            if path_length < 1e-6:  # 避免除以零
                return False

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
        'safety_warning': False
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
                    obstacle_polygon = Polygon([
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max)
                    ])

            if obstacle_polygon and vehicle_polygon.intersects(obstacle_polygon):
                collision_info['collision'] = True
                collision_info['position'] = (obstacle.x, obstacle.y)
                collision_info['obstacle'] = obstacle
                collision_info['distance'] = np.hypot(
                    vehicle.x - obstacle.x, vehicle.y - obstacle.y)
                return collision_info  # 发生实际碰撞立即返回

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


class PygameSimulator:
    """基于Pygame的车辆仿真器"""

    def __init__(self, config_input: Optional[Union[str, Dict]] = None):
        """
        初始化仿真器

        参数:
            config_input: 配置文件路径(str)或配置字典(Dict)
        """
        # 加载配置
        self.config = self._load_config(config_input)

        # 初始化pygame
        if not pygame.get_init():
            pygame.init()

        # 设置窗口尺寸和比例
        self.scale = self.config.get('scale', 5)  # 像素/米
        self.width = self.config.get('window_width', 1000)
        self.height = self.config.get('window_height', 800)

        # 创建窗口
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.config.get(
            'window_title', 'RRT-Pygame 仿真器'))

        # 创建时钟对象
        self.clock = pygame.time.Clock()

        # 获取车辆配置
        vehicle_config = self.config.get('vehicle', {})

        # 初始化车辆和环境
        self.environment = None
        self.vehicle = VehicleModel(
            length=vehicle_config.get('length', 4.5),
            width=vehicle_config.get('width', 1.8)
        )

        # 更新车辆参数
        self.vehicle.wheel_base = vehicle_config.get('wheel_base', 2.7)
        self.vehicle.max_speed = vehicle_config.get('max_speed', 5.0)
        self.vehicle.max_accel = vehicle_config.get('max_accel', 2.0)
        self.vehicle.max_decel = vehicle_config.get('max_decel', 4.0)
        self.vehicle.max_steer_angle = vehicle_config.get(
            'max_steer_angle', 0.7854)

        self.follower = PathFollower(
            lookahead=self.config.get('lookahead', 5.0),
            control_method=self.config.get('control_method', 'default')
        )

        # 仿真状态
        self.running = False
        self.paused = False
        self.collision_detected = False
        self.status_text = ""
        self.status_color = (0, 0, 0)

        # 控制方法
        self.control_methods = ["default", "pid", "mpc", "lqr", "parking"]
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
            'acceleration': []
        }

        # 坐标转换参数
        self.offset_x = self.width / 2
        self.offset_y = self.height / 2

        # 添加按键提示信息
        self.key_hints = [
            "R: 重置车辆",
            "C: 切换控制方法",
            "P: 切换规划算法",
            "S: 切换转向模式",
            "空格: 暂停/继续",
            "右键: 选择目标点"
        ]
        self.hint_color = (50, 50, 50)  # 深灰色
        self.hint_font_size = 20

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
            'vehicle': {
                'length': 4.5,
                'width': 1.8,
                'wheel_base': 2.7,
                'max_speed': 20.0,
                'max_accel': 2.0,
                'max_decel': 4.0,
                'max_steer_angle': 0.7854  # π/4
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
        # 获取车辆四个角的坐标
        corners = vehicle.get_corners()

        # 转换到屏幕坐标
        screen_corners = []
        for x, y in corners:
            sx = x * scale + offset_x
            sy = y * scale + offset_y
            screen_corners.append((int(sx), int(sy)))

        # 设置车身颜色
        if color is None:
            car_color = (0, 128, 0)  # 默认绿色
        else:
            car_color = color

        # 绘制车身
        pygame.draw.polygon(screen, car_color, screen_corners)

        # 绘制车窗 (内部区域)
        window_inset = 0.2  # 车窗内缩比例
        half_length = vehicle.length / 2 * (1 - window_inset)
        half_width = vehicle.width / 2 * (1 - window_inset)

        # 车窗在车身坐标系中的位置
        window_local = [
            (half_length, half_width),   # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width),  # 左后
            (-half_length, half_width)   # 右后
        ]

        # 转换到世界坐标系，再转换到屏幕坐标系
        cos_h = math.cos(vehicle.heading)
        sin_h = math.sin(vehicle.heading)

        window_screen = []
        for lx, ly in window_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = wx * scale + offset_x
            sy = wy * scale + offset_y
            window_screen.append((int(sx), int(sy)))

        # 绘制车窗 (深蓝色半透明)
        pygame.draw.polygon(screen, (30, 30, 80, 180), window_screen)

        # 获取车轮位置和角度
        wheels = vehicle.get_wheel_positions()

        # 绘制车轮
        for wheel_x, wheel_y, wheel_angle in wheels:
            # 计算车轮的四个角
            wheel_half_length = vehicle.wheel_length / 2
            wheel_half_width = vehicle.wheel_width / 2

            # 车轮在自身坐标系中的四个角
            wheel_corners_local = [
                (wheel_half_length, wheel_half_width),   # 右前
                (wheel_half_length, -wheel_half_width),  # 左前
                (-wheel_half_length, -wheel_half_width),  # 左后
                (-wheel_half_length, wheel_half_width)   # 右后
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

            # 绘制车轮 (黑色)
            pygame.draw.polygon(screen, (20, 20, 20), wheel_corners_screen)

            # 绘制车轮中心点 (轮毂)
            hub_x = int(wheel_x * scale + offset_x)
            hub_y = int(wheel_y * scale + offset_y)
            pygame.draw.circle(screen, (150, 150, 150), (hub_x, hub_y), 2)

        # 绘制车头方向
        head_x = vehicle.x + math.cos(vehicle.heading) * vehicle.length / 2
        head_y = vehicle.y + math.sin(vehicle.heading) * vehicle.length / 2
        center_screen = (int(vehicle.x * scale + offset_x),
                         int(vehicle.y * scale + offset_y))
        head_screen = (int(head_x * scale + offset_x),
                       int(head_y * scale + offset_y))
        pygame.draw.line(screen, (0, 0, 255), center_screen, head_screen, 2)

        # 绘制车灯
        light_radius = vehicle.width * 0.1
        light_offset_y = vehicle.width * 0.3

        # 前灯位置 (黄色)
        front_light_local = [
            (vehicle.length/2 - light_radius, light_offset_y),  # 右前灯
            (vehicle.length/2 - light_radius, -light_offset_y)  # 左前灯
        ]

        for lx, ly in front_light_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = int(wx * scale + offset_x)
            sy = int(wy * scale + offset_y)
            pygame.draw.circle(screen, (255, 255, 0),
                               (sx, sy), int(light_radius * scale))

        # 后灯位置 (红色)
        rear_light_local = [
            (-vehicle.length/2 + light_radius, light_offset_y),  # 右后灯
            (-vehicle.length/2 + light_radius, -light_offset_y)  # 左后灯
        ]

        for lx, ly in rear_light_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = int(wx * scale + offset_x)
            sy = int(wy * scale + offset_y)
            pygame.draw.circle(screen, (255, 0, 0), (sx, sy),
                               int(light_radius * scale))

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
                pygame.draw.rect(screen, color, (sx - imu_size//2,
                                                 sy - imu_size//2, imu_size, imu_size))

            # 绘制GPS (浅绿色)
            if sensor_positions['gps']:
                pos = sensor_positions['gps']['pos']
                color = sensor_positions['gps']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制GPS为一个十字形
                cross_size = 5
                pygame.draw.line(screen, color, (sx - cross_size,
                                                 sy), (sx + cross_size, sy), 2)
                pygame.draw.line(screen, color, (sx, sy - cross_size),
                                 (sx, sy + cross_size), 2)

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
                    obstacle.x - obstacle.width / 2,
                    obstacle.y + obstacle.height / 2
                )
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
        font = get_font(18)

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
                            self.current_control_method = self.control_methods[(self.control_methods.index(
                                self.current_control_method) + 1) % len(self.control_methods)]
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
                            self.vehicle)

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
                        self.simulation_data['speed'].append(
                            self.vehicle.v)
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
            font = get_font(24)
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
            font = get_font(self.hint_font_size)
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
