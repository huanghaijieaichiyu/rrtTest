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
import numpy as np
import pygame
import yaml
from typing import List, Tuple, Dict, Optional, Any

from .environment import Environment

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
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "simhei.ttf")
        if os.path.exists(font_path):
            return pygame.font.Font(font_path, size)
    except:
        pass
        
    # 如果都失败了，使用默认字体
    return pygame.font.Font(None, size)

class VehicleModel:
    """简化的车辆动力学模型"""

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
        self.steer_angle = 0.0  # 转向角(弧度)
        self.wheelbase = 2.7  # 轴距(米)

        # 记录轨迹
        self.trajectory = [(x, y)]

        # 车辆控制参数
        self.max_speed = 20.0  # m/s
        self.max_accel = 2.0   # m/s^2
        self.max_brake = 4.0   # m/s^2
        self.max_steer = math.pi/4  # 最大转向角(弧度)

    def get_corners(self) -> List[Tuple[float, float]]:
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

    def update(self, throttle: float, brake: float, steer: float, dt: float) -> None:
        """
        更新车辆状态

        参数:
            throttle: 油门输入 [0, 1]
            brake: 制动输入 [0, 1]
            steer: 转向输入 [-1, 1]
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

        # 更新转向角
        self.steer_angle = steer * self.max_steer

        # 简化的自行车模型
        if abs(self.speed) > 0.1:  # 当速度足够大时才转向
            turn_radius = self.wheelbase / \
                math.tan(abs(self.steer_angle) + 1e-10)
            angular_velocity = self.speed / turn_radius if self.steer_angle != 0 else 0

            if self.steer_angle < 0:
                angular_velocity = -angular_velocity

            # 更新位置和朝向
            self.heading += angular_velocity * dt
            self.heading = self.heading % (2 * math.pi)  # 规范化到 [0, 2π]

        # 根据当前朝向和速度更新位置
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

        # 记录轨迹
        self.trajectory.append((self.x, self.y))


class PathFollower:
    """路径跟踪控制器"""

    def __init__(self, lookahead: float = 5.0):
        """
        初始化路径跟踪器

        参数:
            lookahead: 前瞻距离(米)
        """
        self.lookahead = lookahead
        self.path = []
        self.current_target_idx = 0

    def set_path(self, path: List[Tuple[float, float]]) -> None:
        """设置要跟踪的路径"""
        self.path = path
        self.current_target_idx = 0

    def get_control(self, vehicle: VehicleModel) -> Tuple[float, float, float]:
        """
        计算控制输入

        参数:
            vehicle: 车辆模型实例

        返回:
            throttle, brake, steer 三个控制输入
        """
        if not self.path or self.current_target_idx >= len(self.path):
            return 0.0, 0.0, 0.0  # 没有路径或已到达终点，停车

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
        steer = heading_error / vehicle.max_steer
        steer = max(-1.0, min(1.0, steer))  # 限制在 [-1, 1] 范围内

        # 简单的速度控制：根据转向角的大小调整速度
        throttle = 0.5 * (1.0 - 0.5 * abs(steer))
        brake = 0.0

        # 如果即将转弯，提前减速
        if abs(steer) > 0.5:
            throttle *= 0.5

        return throttle, brake, steer


class PygameSimulator:
    """基于Pygame的车辆仿真器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化仿真器

        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)

        # 初始化pygame
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

        # 初始化车辆和环境
        self.environment = None
        self.vehicle = VehicleModel()
        self.follower = PathFollower(
            lookahead=self.config.get('lookahead', 5.0))

        # 仿真状态
        self.running = False
        self.paused = False

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
        self.start_time = None

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            'scale': 5,
            'window_width': 1000,
            'window_height': 800,
            'window_title': 'RRT-Pygame 仿真器',
            'fps': 60,
            'dt': 0.05,  # 仿真时间步长(秒)
            'lookahead': 5.0,  # 路径跟踪前瞻距离
            'vehicle': {
                'length': 4.5,
                'width': 1.8,
                'wheelbase': 2.7,
                'max_speed': 20.0,
                'max_accel': 2.0,
                'max_brake': 4.0,
                'max_steer': 0.7854  # π/4
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    # 更新默认配置
                    self._update_config(default_config, user_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")

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

    def _draw_vehicle(self) -> None:
        """绘制车辆"""
        corners = self.vehicle.get_corners()
        screen_corners = [self.world_to_screen(x, y) for x, y in corners]

        # 绘制车身
        pygame.draw.polygon(self.screen, RED, screen_corners)

        # 绘制车头方向
        head_x = self.vehicle.x + \
            math.cos(self.vehicle.heading) * self.vehicle.length * 0.5
        head_y = self.vehicle.y + \
            math.sin(self.vehicle.heading) * self.vehicle.length * 0.5

        center_screen = self.world_to_screen(self.vehicle.x, self.vehicle.y)
        head_screen = self.world_to_screen(head_x, head_y)

        pygame.draw.line(self.screen, BLUE, center_screen, head_screen, 2)

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
        """绘制车辆信息"""
        font = get_font(24)
        
        # 车辆状态信息
        info_list = [
            f"速度: {self.vehicle.speed:.1f} m/s",
            f"加速度: {self.vehicle.acceleration:.1f} m/s²",
            f"转向角: {math.degrees(self.vehicle.steer_angle):.1f}°",
            f"位置: ({self.vehicle.x:.1f}, {self.vehicle.y:.1f})",
            f"朝向: {math.degrees(self.vehicle.heading):.1f}°"
        ]
        
        y = 10
        for info in info_list:
            text = font.render(info, True, BLACK)
            self.screen.blit(text, (10, y))
            y += 30

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

                if not self.paused:
                    # 计算控制输入
                    throttle, brake, steer = self.follower.get_control(
                        self.vehicle)

                    # 更新车辆状态
                    self.vehicle.update(throttle, brake, steer, dt)

                    # 记录数据
                    current_time = time.time() - self.start_time
                    self.simulation_data['time'].append(current_time)
                    self.simulation_data['position_x'].append(self.vehicle.x)
                    self.simulation_data['position_y'].append(self.vehicle.y)
                    self.simulation_data['heading'].append(
                        self.vehicle.heading)
                    self.simulation_data['speed'].append(self.vehicle.speed)
                    self.simulation_data['steer_angle'].append(
                        self.vehicle.steer_angle)
                    self.simulation_data['acceleration'].append(
                        self.vehicle.acceleration)

                    # 检查是否到达终点
                    if self.follower.current_target_idx >= len(self.follower.path) - 1 and self.vehicle.speed < 0.1:
                        print("已到达终点")
                        time.sleep(1)  # 短暂停留
                        break

                # 绘制场景
                self.screen.fill(WHITE)
                self._draw_environment()
                self._draw_path(self.follower.path)
                self._draw_trajectory()
                self._draw_vehicle()
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
