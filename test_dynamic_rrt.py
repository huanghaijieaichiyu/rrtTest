#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
动态环境下的RRT*路径规划测试

演示在城市场景下的RRT*路径规划，包括避开静态障碍和动态移动的障碍。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon

from rrt.rrt_star import RRTStar
from simulation.dynamic_environment import (
    DynamicEnvironment,
    MovementPattern,
    TrafficLightState
)


def run_rrt_in_dynamic_environment():
    """在动态环境中运行RRT*路径规划"""
    # 创建动态环境
    env = DynamicEnvironment(width=100.0, height=100.0)

    # 生成城市环境
    print("创建城市环境...")
    env.create_urban_environment()

    # 设置起点和终点
    start = (10.0, 10.0)
    goal = (90.0, 90.0)

    print("环境设置:")
    print(f"- 环境大小: {env.width} x {env.height}")
    print(f"- 起点: {start}")
    print(f"- 终点: {goal}")
    print(f"- 静态障碍物数量: {len(env.obstacles)}")
    print(f"- 移动障碍物数量: {len(env.moving_obstacles)}")
    print(f"- 交通信号灯数量: {len(env.traffic_lights)}")

    # 创建RRT*规划器
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        env=env,
        step_size=5.0,
        max_iterations=1000,
        goal_sample_rate=0.1,
        search_radius=20.0,
        rewire_factor=1.5
    )

    # 执行路径规划
    print("\n开始路径规划...")
    start_time = time.time()
    path = rrt_star.plan()
    end_time = time.time()

    # 输出规划结果
    if path:
        print("\n成功找到路径!")
        print(f"- 规划用时: {end_time - start_time:.2f} 秒")
        print(f"- 路径长度: {len(path)}")
        print("- 路径节点:")
        for i, point in enumerate(path):
            print(f"  {i+1}. ({point[0]:.2f}, {point[1]:.2f})")
    else:
        print("\n未找到可行路径")
        print(f"- 规划用时: {end_time - start_time:.2f} 秒")
        print(f"- 探索节点数: {len(rrt_star.node_list)}")

    return env, path


def visualize_dynamic_environment(env, path=None, duration=10.0, interval=100):
    """
    可视化动态环境和路径

    参数:
        env: 动态环境对象
        path: 规划的路径
        duration: 动画持续时间 (秒)
        interval: 动画间隔 (毫秒)
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title('城市环境中的RRT*路径规划')

    # 用于存储动态对象的图形引用
    moving_patches = []
    traffic_light_patches = []

    # 绘制静态障碍物
    for obstacle in env.obstacles:
        if hasattr(obstacle, 'type') and obstacle.type == 'circle':
            patch = Circle(
                (obstacle.x, obstacle.y),
                radius=obstacle.radius,
                color='grey',
                alpha=0.7
            )
        else:  # 矩形障碍物
            patch = Rectangle(
                (obstacle.x - obstacle.width/2, obstacle.y - obstacle.height/2),
                obstacle.width,
                obstacle.height,
                color='grey',
                alpha=0.7
            )
        ax.add_patch(patch)

    # 创建移动障碍物图形
    for obstacle in env.moving_obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) >= 3:
            # 对于圆形障碍物
            patch = Circle(
                (obstacle[0], obstacle[1]),
                radius=obstacle[2],
                color='blue',
                alpha=0.5
            )
        else:
            # 对于矩形障碍物
            patch = Rectangle(
                (obstacle.x - obstacle.width/2, obstacle.y - obstacle.height/2),
                obstacle.width,
                obstacle.height,
                color='red' if hasattr(
                    obstacle, 'speed') and obstacle.speed > 2.0 else 'blue',
                alpha=0.7
            )
        moving_patches.append(patch)
        ax.add_patch(patch)

    # 创建交通信号灯图形
    for light in env.traffic_lights:
        color = 'red'
        if light.state == TrafficLightState.GREEN:
            color = 'green'
        elif light.state == TrafficLightState.YELLOW:
            color = 'yellow'

        patch = Circle(
            (light.x, light.y),
            radius=light.radius,
            color=color,
            alpha=0.7
        )
        traffic_light_patches.append((patch, light))
        ax.add_patch(patch)

    # 绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        line, = ax.plot(path_x, path_y, 'g-', linewidth=2, label='规划路径')

        # 绘制起点和终点
        ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='起点')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='终点')

    # 设置网格
    ax.grid(True)
    ax.legend()

    # 计算总的帧数
    total_frames = int(duration * 1000 / interval)

    def update(frame):
        # 更新环境
        env.update(dt=interval/1000.0)

        # 更新移动障碍物位置
        for i, obstacle in enumerate(env.moving_obstacles):
            if i < len(moving_patches):
                patch = moving_patches[i]
                patch.set_xy((obstacle.x - obstacle.width/2,
                             obstacle.y - obstacle.height/2))

        # 更新交通信号灯颜色
        for patch, light in traffic_light_patches:
            if light.state == TrafficLightState.RED:
                patch.set_color('red')
            elif light.state == TrafficLightState.GREEN:
                patch.set_color('green')
            else:  # YELLOW
                patch.set_color('yellow')

        # 返回所有需要更新的图形对象
        return moving_patches + [patch for patch, _ in traffic_light_patches]

    # 创建动画
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=True
    )

    # 显示动画
    plt.show()


def main():
    """主函数"""
    print("开始城市环境下的RRT*路径规划测试...")
    env, path = run_rrt_in_dynamic_environment()

    # 可视化结果
    if path:
        print("\n开始3D可视化...")
        visualize_dynamic_environment(env, path, duration=30.0)


if __name__ == "__main__":
    main()
