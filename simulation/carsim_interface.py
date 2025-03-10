#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CarSim接口模块

提供与CarSim软件的接口，支持路径规划算法在CarSim中的验证和仿真。
"""

import os
import numpy as np
import yaml
from typing import Tuple, List, Dict, Any, Optional, Sequence
import time
import matplotlib.pyplot as plt

from .environment import Environment


class CarSimInterface:
    """
    CarSim接口类

    提供与CarSim软件的交互功能，包括：
    1. 初始化CarSim仿真环境
    2. 发送控制命令到CarSim
    3. 从CarSim获取车辆状态
    4. 在CarSim中执行规划路径
    """

    def __init__(self, config_path: str):
        """
        初始化CarSim接口

        参数:
            config_path: CarSim配置文件路径
        """
        self.config = self._load_config(config_path)
        self.carsim_path = self.config.get('carsim_path', '')
        self.vehicle_params = self.config.get('vehicle_params', {})
        self.simulation_params = self.config.get('simulation_params', {})

        # 连接状态
        self.connected = False

        # 仿真环境
        self.environment = self._create_environment()

        # 初始化API连接
        self._initialize_connection()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件

        参数:
            config_path: 配置文件路径

        返回:
            配置数据字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}

    def _create_environment(self) -> Environment:
        """
        创建仿真环境

        返回:
            环境对象
        """
        # 从配置中获取环境参数
        env_params = self.config.get('environment', {})
        width = env_params.get('width', 100.0)
        height = env_params.get('height', 100.0)

        # 创建环境
        env = Environment(width=width, height=height)

        # 添加障碍物
        for obs in env_params.get('obstacles', []):
            env.add_obstacle(**obs)

        return env

    def _initialize_connection(self) -> bool:
        """
        初始化与CarSim的连接

        返回:
            连接是否成功
        """
        # 实际实现中，应该调用CarSim API进行连接
        # 这里简单模拟连接过程
        try:
            # 检查CarSim路径是否有效
            if not os.path.exists(self.carsim_path):
                print(f"CarSim路径不存在: {self.carsim_path}")
                return False

            # 模拟连接过程
            print("正在连接CarSim...")
            time.sleep(0.5)  # 模拟连接延迟

            # 设置连接状态
            self.connected = True
            print("CarSim连接成功")

            return True
        except Exception as e:
            print(f"连接CarSim失败: {e}")
            self.connected = False
            return False

    def get_vehicle_state(self) -> Tuple[float, float]:
        """
        获取当前车辆状态

        返回:
            车辆位置 (x, y)
        """
        # 实际实现中，应该从CarSim获取车辆状态
        # 这里简单返回一个固定位置
        if not self.connected:
            print("未连接到CarSim，无法获取车辆状态")
            return (0.0, 0.0)

        # 模拟从CarSim获取车辆位置
        # 实际应通过API调用获取
        x = 10.0  # 假设的车辆x坐标
        y = 10.0  # 假设的车辆y坐标

        return (x, y)

    def get_environment(self) -> Environment:
        """
        获取当前仿真环境

        返回:
            环境对象
        """
        return self.environment

    def execute_path(self, path: Sequence[Tuple[float, float]]) -> bool:
        """
        在CarSim中执行规划路径

        参数:
            path: 路径点列表

        返回:
            执行是否成功
        """
        if not self.connected:
            print("未连接到CarSim，无法执行路径")
            return False

        if not path:
            print("路径为空，无法执行")
            return False

        try:
            print(f"开始在CarSim中执行路径，包含 {len(path)} 个点")

            # 实际实现中，应该将路径发送到CarSim执行
            # 这里简单模拟执行过程

            # 模拟执行延迟
            for i, point in enumerate(path):
                print(
                    f"执行路径点 {i+1}/{len(path)}: "
                    f"({point[0]:.2f}, {point[1]:.2f})"
                )
                time.sleep(0.1)  # 模拟执行延迟

            print("路径执行完成")
            return True

        except Exception as e:
            print(f"执行路径失败: {e}")
            return False

    def set_vehicle_params(self, params: Dict[str, Any]) -> bool:
        """
        设置车辆参数

        参数:
            params: 车辆参数字典

        返回:
            设置是否成功
        """
        if not self.connected:
            print("未连接到CarSim，无法设置车辆参数")
            return False

        try:
            # 更新本地参数
            self.vehicle_params.update(params)

            # 实际实现中，应该将参数发送到CarSim
            # 这里简单打印参数
            print("设置车辆参数:")
            for key, value in params.items():
                print(f"  {key}: {value}")

            return True

        except Exception as e:
            print(f"设置车辆参数失败: {e}")
            return False

    def start_simulation(self, duration: float = 60.0) -> bool:
        """
        开始CarSim仿真

        参数:
            duration: 仿真时长（秒）

        返回:
            仿真是否成功启动
        """
        if not self.connected:
            print("未连接到CarSim，无法启动仿真")
            return False

        try:
            print(f"开始CarSim仿真，时长: {duration} 秒")

            # 实际实现中，应该调用CarSim API启动仿真
            # 这里简单模拟仿真过程

            # 设置仿真参数
            sim_params = {
                'duration': duration,
                'step_size': self.simulation_params.get('step_size', 0.01),
                'real_time_factor': self.simulation_params.get('real_time_factor', 1.0)
            }

            print("仿真参数:")
            for key, value in sim_params.items():
                print(f"  {key}: {value}")

            # 模拟仿真启动
            print("仿真已启动，实际运行将在CarSim中进行")

            return True

        except Exception as e:
            print(f"启动仿真失败: {e}")
            return False

    def stop_simulation(self) -> bool:
        """
        停止CarSim仿真

        返回:
            仿真是否成功停止
        """
        if not self.connected:
            print("未连接到CarSim，无法停止仿真")
            return False

        try:
            print("停止CarSim仿真")

            # 实际实现中，应该调用CarSim API停止仿真
            # 这里简单模拟停止过程

            print("仿真已停止")

            return True

        except Exception as e:
            print(f"停止仿真失败: {e}")
            return False

    def get_simulation_results(self) -> Dict[str, Any]:
        """
        获取仿真结果

        返回:
            仿真结果字典
        """
        if not self.connected:
            print("未连接到CarSim，无法获取仿真结果")
            return {}

        try:
            print("获取仿真结果")

            # 实际实现中，应该从CarSim获取结果数据
            # 这里简单返回一些模拟数据

            # 模拟车辆轨迹
            time_steps = np.linspace(0, 10, 100)
            positions_x = 10 + time_steps * 2  # 简单的x轨迹
            positions_y = 10 + np.sin(time_steps) * 5  # 简单的y轨迹

            # 模拟速度和加速度
            velocities = 10 + np.sin(time_steps) * 2
            accelerations = np.cos(time_steps) * 2

            # 组织结果数据
            results = {
                'time': time_steps.tolist(),
                'position_x': positions_x.tolist(),
                'position_y': positions_y.tolist(),
                'velocity': velocities.tolist(),
                'acceleration': accelerations.tolist()
            }

            return results

        except Exception as e:
            print(f"获取仿真结果失败: {e}")
            return {}

    def visualize_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """
        可视化仿真结果

        参数:
            results: 仿真结果字典，如果为None则获取最新结果
        """
        # 如果未提供结果，获取最新结果
        if results is None:
            results = self.get_simulation_results()

        if not results:
            print("没有可视化的结果数据")
            return

        # 创建可视化图表
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # 1. 轨迹图
        ax1 = axes[0]
        if 'position_x' in results and 'position_y' in results:
            ax1.plot(results['position_x'], results['position_y'], 'b-')
            ax1.scatter(results['position_x'][0], results['position_y'][0],
                        color='g', s=100, label='起点')
            ax1.scatter(results['position_x'][-1], results['position_y'][-1],
                        color='r', s=100, label='终点')

            # 绘制环境（如果需要）
            self.environment.plot_obstacles(ax1)

            ax1.set_title('车辆轨迹')
            ax1.set_xlabel('X 位置 (m)')
            ax1.set_ylabel('Y 位置 (m)')
            ax1.grid(True)
            ax1.legend()
            ax1.set_aspect('equal')

        # 2. 速度图
        ax2 = axes[1]
        if 'time' in results and 'velocity' in results:
            ax2.plot(results['time'], results['velocity'], 'g-')
            ax2.set_title('车辆速度')
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('速度 (m/s)')
            ax2.grid(True)

        # 3. 加速度图
        ax3 = axes[2]
        if 'time' in results and 'acceleration' in results:
            ax3.plot(results['time'], results['acceleration'], 'r-')
            ax3.set_title('车辆加速度')
            ax3.set_xlabel('时间 (s)')
            ax3.set_ylabel('加速度 (m/s²)')
            ax3.grid(True)

        plt.tight_layout()
        plt.show()

    def disconnect(self) -> bool:
        """
        断开与CarSim的连接

        返回:
            断开是否成功
        """
        if not self.connected:
            print("未连接到CarSim")
            return True

        try:
            print("断开与CarSim的连接")

            # 实际实现中，应该调用CarSim API断开连接
            # 这里简单模拟断开过程

            # 设置连接状态
            self.connected = False
            print("已断开与CarSim的连接")

            return True

        except Exception as e:
            print(f"断开连接失败: {e}")
            return False


# 简单的用法示例
if __name__ == "__main__":
    # 配置文件路径
    config_path = "../config/carsim_config.yaml"

    # 创建CarSim接口
    try:
        carsim = CarSimInterface(config_path)

        # 获取车辆状态
        vehicle_state = carsim.get_vehicle_state()
        print(f"车辆当前位置: {vehicle_state}")

        # 设置车辆参数
        carsim.set_vehicle_params({
            'mass': 1500,
            'wheelbase': 2.8,
            'max_steer_angle': 0.6
        })

        # 创建一条简单的路径
        path = [(10, 10), (20, 15), (30, 25), (40, 30), (50, 30)]

        # 执行路径
        carsim.execute_path(path)

        # 启动仿真
        carsim.start_simulation(duration=10.0)

        # 获取和可视化结果
        results = carsim.get_simulation_results()
        carsim.visualize_results(results)

        # 断开连接
        carsim.disconnect()

    except Exception as e:
        print(f"运行示例失败: {e}")
