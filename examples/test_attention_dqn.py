#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的Attention DQN RRT模型
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
import matplotlib.pyplot as plt
from simulation.pygame_simulator import ParkingEnvironment
from rrt.attention_dqn_rrt import AttentionDQNRRT


def main():
    # 创建一个简单的环境
    env = ParkingEnvironment(width=100, height=100)

    # 添加一些障碍物
    env.add_obstacle(x=30, y=30, obstacle_type="rectangle", width=10, height=10)
    env.add_obstacle(x=70, y=70, obstacle_type="rectangle", width=10, height=10)

    # 设置起点和终点
    start = (10, 10)
    goal = (90, 90)

    # 创建AttentionDQNRRT实例
    model = AttentionDQNRRT(start=start,
                            goal=goal,
                            env=env,
                            vehicle_width=2.0,
                            vehicle_length=4.0,
                            step_size=2.0,
                            max_iterations=200,
                            rewire_factor=1.5,
                            learning_rate=0.0005,
                            gamma=0.99,
                            epsilon=0.2,
                            buffer_capacity=10000,
                            batch_size=32,
                            hidden_dim=256,
                            prediction_horizon=5)

    # 尝试规划路径
    print("开始规划路径...")
    path = model.plan()

    if path:
        print(f"成功规划路径，路径长度: {len(path)}")

        # 绘制环境和路径
        plt.figure(figsize=(10, 10))

        # 绘制障碍物
        for obs in env.obstacles:
            if obs.type == "rectangle":
                x = obs.x - obs.width / 2
                y = obs.y - obs.height / 2
                rect = plt.Rectangle((x, y), obs.width, obs.height, color='red', alpha=0.5)
                plt.gca().add_patch(rect)

        # 绘制路径
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2)

        # 绘制起点和终点
        plt.plot(start[0], start[1], 'go', markersize=10)
        plt.plot(goal[0], goal[1], 'ro', markersize=10)

        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        plt.grid(True)
        plt.title("规划路径")

        # 保存图像
        plt.savefig("planned_path.png")
        print("已保存路径图像到planned_path.png")

        # 显示图像
        plt.show()
    else:
        print("路径规划失败")


if __name__ == "__main__":
    main()
