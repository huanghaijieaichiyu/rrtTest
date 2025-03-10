#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
固定环境数据生成器

生成一个固定的简单环境和路径，用于调试
"""

import os
import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from simulation.environment import Environment
    from rrt.rrt_star import RRTStar
    print("成功导入基础模块")
except ImportError as e:
    print(f"导入基础模块失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 创建一个非常简单的环境
env = Environment(width=100.0, height=100.0)

# 添加一个障碍物
env.add_obstacle(x=50, y=50, obstacle_type="circle", radius=10)
print("成功创建环境和障碍物")

# 设置起点和终点
start = (10, 10)
goal = (90, 90)

# 检查起点和终点是否有效
print(f"起点碰撞检测: {env.check_collision(start)}")
print(f"终点碰撞检测: {env.check_collision(goal)}")

# 使用 RRT* 生成路径
print("开始路径规划...")
planner = RRTStar(
    start=start,
    goal=goal,
    env=env,
    step_size=10.0,  # 增大步长
    max_iterations=100,  # 减少迭代次数，便于调试
    goal_sample_rate=0.3,  # 增加目标采样率
    search_radius=50.0  # 增大搜索半径
)

# 打印规划器参数
print(f"规划器参数:")
print(f"  步长: {planner.step_size}")
print(f"  最大迭代次数: {planner.max_iterations}")
print(f"  目标采样率: {planner.goal_sample_rate}")
print(f"  搜索半径: {planner.search_radius}")

# 执行规划
path = planner.plan()

if path:
    print(f"成功规划路径，路径点数: {len(path)}")
    print("路径点:")
    for i, point in enumerate(path):
        print(f"  点 {i+1}: {point}")

    # 可视化环境和路径
    fig, ax = plt.subplots(figsize=(8, 8))
    env.plot_obstacles(ax)

    # 绘制起点和终点
    ax.plot(start[0], start[1], 'go', markersize=10, label='起点')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='终点')

    # 绘制路径
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, 'b-', linewidth=2, label='路径')

    # 设置图表
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('RRT* 路径规划结果')

    # 保存图表
    save_dir = os.path.join(root_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fixed_environment_path.png"))
    print(f"图表已保存到 {os.path.join(save_dir, 'fixed_environment_path.png')}")

    # 显示图表
    plt.show()
else:
    print("路径规划失败")

    # 打印规划器内部状态
    print(f"节点数量: {len(planner.node_list)}")
    print(
        f"最近节点到目标的距离: {np.hypot(planner.node_list[-1].x - goal[0], planner.node_list[-1].y - goal[1])}")

    # 可视化环境和节点
    fig, ax = plt.subplots(figsize=(8, 8))
    env.plot_obstacles(ax)

    # 绘制起点和终点
    ax.plot(start[0], start[1], 'go', markersize=10, label='起点')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='终点')

    # 绘制所有节点
    for node in planner.node_list:
        ax.plot(node.x, node.y, 'bx', markersize=3)
        if node.parent:
            ax.plot([node.x, node.parent.x], [
                    node.y, node.parent.y], 'b-', linewidth=0.5)

    # 设置图表
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('RRT* 规划失败')

    # 保存图表
    save_dir = os.path.join(root_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "fixed_environment_failure.png"))
    print(f"图表已保存到 {os.path.join(save_dir, 'fixed_environment_failure.png')}")

    # 显示图表
    plt.show()
