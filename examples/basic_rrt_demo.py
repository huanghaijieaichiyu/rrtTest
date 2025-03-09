#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基本RRT算法演示脚本

演示如何使用基本RRT算法规划路径，并可视化结果。
"""

from simulation.visualization import Visualization
from simulation.environment import Environment
from rrt.rrt_base import RRT
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块


def main():
    """主函数"""
    print("基本RRT算法演示")

    # 创建环境
    env = Environment(width=100, height=100)

    # 添加障碍物
    # 设置随机种子，以便结果可复现
    np.random.seed(42)

    # 添加一些随机圆形障碍物
    for _ in range(10):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        radius = np.random.uniform(3, 8)
        env.add_obstacle(x, y, obstacle_type="circle", radius=radius)

    # 添加一些随机矩形障碍物
    for _ in range(5):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        width = np.random.uniform(5, 15)
        height = np.random.uniform(5, 15)
        angle = np.random.uniform(0, 2 * np.pi)
        env.add_obstacle(
            x, y, obstacle_type="rectangle",
            width=width, height=height, angle=angle
        )

    # 设置起点和终点
    start = (10, 10)
    goal = (90, 90)

    # 可视化环境
    vis = Visualization(env)
    vis.plot_path(
        path=[],  # 空路径，只展示环境
        start=start,
        goal=goal,
        title="RRT规划环境"
    )
    plt.show()

    # 创建RRT规划器
    rrt = RRT(
        start=start,
        goal=goal,
        env=env,
        step_size=5.0,
        max_iterations=1000,
        goal_sample_rate=0.1,
        search_radius=50.0
    )

    # 执行规划
    print("开始RRT路径规划...")
    start_time = time.time()
    path = rrt.plan()
    elapsed_time = time.time() - start_time

    if path:
        print(f"找到路径，包含 {len(path)} 个节点，用时 {elapsed_time:.3f} 秒")
    else:
        print(f"未能找到路径，用时 {elapsed_time:.3f} 秒")
        return

    # 可视化路径
    print("可视化规划结果...")

    # 1. 绘制路径
    vis.plot_path(
        path=path,
        start=start,
        goal=goal,
        title="RRT规划路径"
    )
    plt.show()

    # 2. 绘制搜索树
    vis.plot_search_tree(
        nodes=rrt.node_list,
        path=path,
        title="RRT搜索树和规划路径"
    )
    plt.show()

    # 3. 创建路径动画
    print("创建路径动画...")
    anim = vis.animate_path(path)
    plt.show()

    print("演示完成!")


if __name__ == "__main__":
    main()
