#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Attention DQN RRT模型加载

这个脚本演示如何加载预训练的Attention DQN RRT模型进行路径规划。
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulation.pygame_simulator import ParkingEnvironment
from rrt.attention_dqn_rrt import AttentionDQNRRT


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试加载预训练Attention DQN RRT模型')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--iterations', type=int, default=200, help='最大迭代次数')
    parser.add_argument('--step_size', type=float, default=2.0, help='步长')
    parser.add_argument('--epsilon', type=float, default=0.05, help='探索率')
    return parser.parse_args()


def create_test_environment():
    """创建测试环境"""
    env = ParkingEnvironment(width=100, height=100)

    # 添加障碍物
    env.add_obstacle(x=30, y=30, obstacle_type="rectangle", width=10, height=10)
    env.add_obstacle(x=70, y=70, obstacle_type="rectangle", width=10, height=10)
    env.add_obstacle(x=30, y=70, obstacle_type="rectangle", width=15, height=8)
    env.add_obstacle(x=60, y=40, obstacle_type="rectangle", width=12, height=6)

    # 添加一些更小的障碍物
    for i in range(5):
        x = np.random.uniform(20, 80)
        y = np.random.uniform(20, 80)
        width = np.random.uniform(3, 6)
        height = np.random.uniform(3, 6)
        env.add_obstacle(x=x, y=y, obstacle_type="rectangle", width=width, height=height)

    return env


def main():
    """主函数"""
    args = parse_args()

    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return

    # 创建环境
    env = create_test_environment()

    # 设置起点和终点
    start = (10, 10)
    goal = (90, 90)

    print(f"模型路径: {args.model_path}")
    print(f"起点: {start}, 终点: {goal}")
    print(f"最大迭代次数: {args.iterations}, 步长: {args.step_size}")

    # 创建规划器并加载模型
    planner = AttentionDQNRRT(start=start,
                              goal=goal,
                              env=env,
                              vehicle_width=2.0,
                              vehicle_length=4.0,
                              step_size=args.step_size,
                              max_iterations=args.iterations,
                              rewire_factor=1.5,
                              learning_rate=0.0005,
                              gamma=0.99,
                              epsilon=args.epsilon,
                              buffer_capacity=10000,
                              batch_size=32,
                              hidden_dim=256,
                              prediction_horizon=5,
                              model_path=args.model_path)

    # 规划路径
    print("\n开始规划路径...")
    path = planner.plan()

    if path:
        print(f"成功规划路径! 路径长度: {len(path)} 个节点")

        # 可视化结果
        plt.figure(figsize=(10, 10))

        # 绘制障碍物
        for obs in env.obstacles:
            if obs.type == "rectangle":
                x = obs.x - obs.width / 2
                y = obs.y - obs.height / 2
                rect = patches.Rectangle((x, y), obs.width, obs.height, color='red', alpha=0.5)
                plt.gca().add_patch(rect)

        # 绘制规划路径
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')

        # 绘制起点和终点
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

        # 设置图表属性
        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        plt.grid(True)
        plt.title('Attention DQN RRT 路径规划')
        plt.legend()

        # 保存图像
        plt.savefig('attention_dqn_rrt_path.png')
        print("已保存路径图像到 attention_dqn_rrt_path.png")

        # 显示图像
        plt.show()
    else:
        print("路径规划失败!")


if __name__ == "__main__":
    main()
