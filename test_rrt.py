#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 RRTStar 类的初始化和规划功能
"""

import time
from rrt.rrt_star import RRTStar
from simulation.environment import Environment


def test_rrt_star():
    # 创建环境
    env = Environment(width=100.0, height=100.0)

    # 添加一些障碍物
    env.add_obstacle(x=30.0, y=30.0, obstacle_type="circle", radius=10.0)
    env.add_obstacle(x=70.0, y=70.0, obstacle_type="rectangle",
                     width=20.0, height=20.0)

    # 设置起点和终点
    start = (10.0, 10.0)
    goal = (90.0, 90.0)

    print("环境设置:")
    print(f"- 环境大小: {env.width} x {env.height}")
    print(f"- 起点: {start}")
    print(f"- 终点: {goal}")
    print(f"- 障碍物数量: {len(env.obstacles)}")

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


if __name__ == "__main__":
    test_rrt_star()
