#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
控制方法比较示例

展示不同控制方法（默认、PID、MPC、LQR）的性能比较
"""

from rrt.rrt_star import RRTStar
from simulation.pygame_simulator import PygameSimulator
from simulation.environment import Environment
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)


def create_test_environment():
    """创建测试环境"""
    env = Environment(width=500.0, height=300.0)

    # 添加一些障碍物
    env.add_obstacle(x=150, y=150, obstacle_type="circle", radius=30)
    env.add_obstacle(x=300, y=100, obstacle_type="rectangle",
                     width=50, height=40)
    env.add_obstacle(x=250, y=200, obstacle_type="circle", radius=25)
    env.add_obstacle(x=400, y=200, obstacle_type="rectangle",
                     width=30, height=60)

    return env


def generate_test_path(env):
    """生成测试路径"""
    # 设置起点和终点
    start_point = (50, 50)
    goal_point = (450, 250)

    # 使用RRT*生成路径
    planner = RRTStar(
        start=start_point,
        goal=goal_point,
        env=env,
        max_iterations=2000,
        step_size=10.0,
        goal_sample_rate=0.2
    )

    path = planner.plan()
    if not path:
        print("路径规划失败")
        return None

    return path


def run_simulation_with_control_method(env, path, control_method):
    """使用指定控制方法运行仿真"""
    # 创建临时配置文件
    config_path = os.path.join(root_dir, f"temp_{control_method}_config.yaml")
    with open(config_path, 'w') as f:
        f.write(f"control_method: {control_method}\n")
        title = "window_title: 'RRT-Pygame 仿真器 - "
        title += f"{control_method.upper()} 控制'\n"
        f.write(title)

    # 创建仿真器
    simulator = PygameSimulator(config_path)

    # 设置环境
    simulator.set_environment(env)

    # 执行路径
    print(f"开始 {control_method.upper()} 控制方法的路径仿真")
    print("按ESC退出，空格键暂停/继续，R键重置，C键切换控制方法")
    simulator.execute_path(path)

    # 获取仿真结果
    results = simulator.get_simulation_results()

    # 删除临时配置文件
    if os.path.exists(config_path):
        os.remove(config_path)

    return results


def compare_control_methods(env, path):
    """比较不同控制方法的性能"""
    control_methods = ["default", "pid", "mpc", "lqr"]
    results = {}

    for method in control_methods:
        print(f"\n正在测试 {method.upper()} 控制方法...")
        method_results = run_simulation_with_control_method(env, path, method)
        results[method] = method_results

    # 可视化比较结果
    visualize_comparison(results)


def visualize_comparison(results):
    """可视化不同控制方法的比较结果"""
    control_methods = list(results.keys())
    colors = ['b', 'g', 'r', 'c']

    # 设置字体，避免中文乱码
    plt.rc("font", family="Microsoft YaHei")

    plt.figure(figsize=(15, 10))

    # 绘制轨迹比较
    plt.subplot(2, 2, 1)
    for i, method in enumerate(control_methods):
        plt.plot(results[method]['position_x'], results[method]['position_y'],
                 color=colors[i], label=method.upper())
    plt.title('轨迹比较')
    plt.xlabel('X 位置 (m)')
    plt.ylabel('Y 位置 (m)')
    plt.grid(True)
    plt.legend()

    # 绘制速度比较
    plt.subplot(2, 2, 2)
    for i, method in enumerate(control_methods):
        plt.plot(results[method]['time'], results[method]['speed'],
                 color=colors[i], label=method.upper())
    plt.title('速度比较')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.grid(True)
    plt.legend()

    # 绘制转向角比较
    plt.subplot(2, 2, 3)
    for i, method in enumerate(control_methods):
        plt.plot(results[method]['time'],
                 [np.degrees(a) for a in results[method]['steer_angle']],
                 color=colors[i], label=method.upper())
    plt.title('转向角比较')
    plt.xlabel('时间 (s)')
    plt.ylabel('转向角度 (度)')
    plt.grid(True)
    plt.legend()

    # 绘制加速度比较
    plt.subplot(2, 2, 4)
    for i, method in enumerate(control_methods):
        plt.plot(results[method]['time'], results[method]['acceleration'],
                 color=colors[i], label=method.upper())
    plt.title('加速度比较')
    plt.xlabel('时间 (s)')
    plt.ylabel('加速度 (m/s²)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # 保存图表
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "control_methods_comparison.png"))

    plt.show()


def main():
    """主函数"""
    # 创建测试环境
    env = create_test_environment()

    # 生成测试路径
    path = generate_test_path(env)
    if not path:
        return

    # 比较不同控制方法
    compare_control_methods(env, path)


if __name__ == "__main__":
    main()
