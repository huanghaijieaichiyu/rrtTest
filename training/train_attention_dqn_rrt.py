#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预训练 Attention DQN RRT 模型

使用模拟环境和经验回放来预训练 Attention DQN RRT 模型，提高其初始性能。
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Tuple

from rrt.attention_dqn_rrt import AttentionDQNRRT
from simulation.environment import Environment
from simulation.pygame_simulator import ParkingEnvironment


def create_training_environments(num_envs: int = 10) -> List[Environment]:
    """创建多个训练环境"""
    environments = []
    for _ in range(num_envs):
        env = ParkingEnvironment(width=100, height=100)
        # 随机添加障碍物
        num_obstacles = random.randint(5, 15)
        for _ in range(num_obstacles):
            x = random.uniform(10, 90)
            y = random.uniform(10, 90)
            width = random.uniform(2, 8)
            height = random.uniform(2, 8)
            env.add_obstacle(x=x, y=y, obstacle_type="rectangle", width=width, height=height)
        environments.append(env)
    return environments


def generate_training_episodes(env: Environment, num_episodes: int = 1000) -> List[Tuple]:
    """生成训练数据"""
    episodes = []
    for _ in range(num_episodes):
        # 随机生成起点和终点
        start = (random.uniform(5, 95), random.uniform(5, 95))
        goal = (random.uniform(5, 95), random.uniform(5, 95))
        episodes.append((start, goal))
    return episodes


def pretrain_model(model: AttentionDQNRRT,
                   env: Environment,
                   episodes: List[Tuple],
                   num_epochs: int = 10,
                   batch_size: int = 32,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """预训练模型"""
    print(f"开始预训练，设备: {device}")

    # 训练循环
    for epoch in range(num_epochs):
        total_reward = 0
        success_count = 0

        # 设置网络为评估模式进行路径规划
        model.q_network.eval()
        model.target_network.eval()

        # 遍历每个训练样本
        for start, goal in tqdm(episodes, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 重置模型状态
            model.start = start
            model.goal = goal

            # 尝试规划路径
            path = model.plan()

            if path:
                success_count += 1
                # 计算路径长度作为奖励的一部分
                path_length = 0
                for i in range(len(path) - 1):
                    node1 = path[i]
                    node2 = path[i + 1]
                    path_length += np.hypot(node2.x - node1.x, node2.y - node1.y)
                reward = 100 - path_length * 0.1  # 奖励更短的路径
            else:
                reward = -50  # 惩罚失败的规划

            total_reward += reward

        # 设置网络为训练模式进行批量更新
        model.q_network.train()

        # 如果有足够的经验数据，执行批量更新
        if len(model.replay_buffer) >= batch_size:
            # 执行多次更新以充分利用收集的经验
            for _ in range(10):  # 每个epoch更新多次
                model.update_network()
                model.update_prediction_network()

        # 打印训练统计
        avg_reward = total_reward / len(episodes) if episodes else 0
        success_rate = success_count / len(episodes) if episodes else 0
        print(f"Epoch {epoch+1}: Avg Reward = {avg_reward:.2f}, "
              f"Success Rate = {success_rate:.2%}")

        # 保存检查点
        if (epoch + 1) % 2 == 0:  # 更频繁地保存检查点
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/attention_dqn_rrt_epoch_{epoch+1}.pt"
            model.save_model(checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预训练 Attention DQN RRT 模型')
    parser.add_argument('--num_envs', type=int, default=5, help='训练环境数量')
    parser.add_argument('--num_episodes', type=int, default=100, help='每个环境的训练样本数')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    args = parser.parse_args()

    # 创建训练环境
    print("创建训练环境...")
    environments = create_training_environments(args.num_envs)

    # 为每个环境生成训练样本
    all_episodes = []
    for env in environments:
        episodes = generate_training_episodes(env, args.num_episodes // args.num_envs)
        all_episodes.extend([(env, start, goal) for start, goal in episodes])

    # 创建模型
    env = environments[0]  # 使用第一个环境初始化模型
    start = (50, 50)  # 临时起点
    goal = (80, 80)  # 临时终点
    params = {
        'learning_rate': 0.0005,  # 降低学习率以提高稳定性
        'gamma': 0.99,
        'epsilon': 0.2,  # 增加探索率
        'buffer_capacity': 10000,
        'batch_size': args.batch_size,
        'hidden_dim': 256,
        'prediction_horizon': 5
    }
    model = AttentionDQNRRT(
        start=start,
        goal=goal,
        env=env,
        vehicle_width=2.0,
        vehicle_length=4.0,
        step_size=2.0,
        max_iterations=200,  # 减少最大迭代次数以加快训练
        rewire_factor=1.5,
        **params)

    # 预训练模型
    extracted_episodes = [(start, goal) for _, start, goal in all_episodes[:args.num_episodes]]
    pretrain_model(model=model,
                   env=env,
                   episodes=extracted_episodes,
                   num_epochs=args.num_epochs,
                   batch_size=args.batch_size,
                   device=args.device)

    print("预训练完成!")


if __name__ == "__main__":
    main()
