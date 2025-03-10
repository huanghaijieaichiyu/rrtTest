#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
神经网络增强的 RRT 算法训练脚本

此脚本用于训练基于深度学习的 RRT 算法，包括：
1. 数据收集：在不同环境中收集路径规划数据
2. 模型训练：训练采样网络、评估网络和优化网络
3. 模型保存：将训练好的模型保存到文件
4. 可视化：使用 Pygame 可视化训练过程和结果
"""

from rrt.rrt_base import Node
from ml.models.rrt_nn import NeuralRRT, train_neural_rrt
from simulation.pygame_simulator import PygameSimulator
from simulation.environment import Environment
import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 添加调试信息
print("当前工作目录:", os.getcwd())
print("Python 路径:", sys.path)

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print("添加到 Python 路径:", project_root)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练神经网络增强的 RRT 算法')

    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='训练数据收集的路径数量')

    parser.add_argument('--num-epochs', type=int, default=100,
                        help='训练轮数')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')

    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='学习率')

    parser.add_argument('--save-dir', type=str, default='results/models',
                        help='模型保存目录')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')

    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化训练过程')

    return parser.parse_args()


def create_training_environments(num_envs: int = 5) -> list:
    """
    创建训练环境集合

    参数:
        num_envs: 环境数量

    返回:
        环境列表
    """
    environments = []

    # 基本参数
    width = 100.0
    height = 100.0

    for _ in range(num_envs):
        env = Environment(width=width, height=height)

        # 随机添加圆形障碍物
        num_circles = np.random.randint(3, 7)
        for _ in range(num_circles):
            x = np.random.uniform(10, width-10)
            y = np.random.uniform(10, height-10)
            radius = np.random.uniform(3, 8)
            env.add_obstacle(x=x, y=y, obstacle_type="circle", radius=radius)

        # 随机添加矩形障碍物
        num_rectangles = np.random.randint(2, 5)
        for _ in range(num_rectangles):
            x = np.random.uniform(10, width-10)
            y = np.random.uniform(10, height-10)
            w = np.random.uniform(5, 15)
            h = np.random.uniform(5, 15)
            env.add_obstacle(x=x, y=y, obstacle_type="rectangle",
                             width=w, height=h)

        environments.append(env)

    return environments


def save_model(model_dict, save_dir: str):
    """
    保存模型

    参数:
        model_dict: 训练好的模型字典，包含三个网络
        save_dir: 保存目录
    """
    if model_dict is None:
        print("没有可保存的模型")
        return

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存各个网络
    torch.save(model_dict['sampling_net'].state_dict(),
               os.path.join(save_dir, f'sampling_net_{timestamp}.pth'))
    torch.save(model_dict['evaluation_net'].state_dict(),
               os.path.join(save_dir, f'evaluation_net_{timestamp}.pth'))
    torch.save(model_dict['optimization_net'].state_dict(),
               os.path.join(save_dir, f'optimization_net_{timestamp}.pth'))

    print(f"模型已保存到: {save_dir}")


def visualize_training(env: Environment, model_dict):
    """
    可视化训练过程

    参数:
        env: 环境对象
        model_dict: 训练好的模型字典
    """
    if model_dict is None:
        print("没有可视化的模型")
        return

    # 创建 Pygame 模拟器
    simulator = PygameSimulator()
    simulator.set_environment(env)

    # 随机生成起点和终点
    start_x, start_y = np.random.uniform(
        0, env.width), np.random.uniform(0, env.height)
    goal_x, goal_y = np.random.uniform(
        0, env.width), np.random.uniform(0, env.height)

    # 确保起点和终点不在障碍物内
    while env.check_collision((start_x, start_y)):
        start_x, start_y = np.random.uniform(
            0, env.width), np.random.uniform(0, env.height)

    while env.check_collision((goal_x, goal_y)):
        goal_x, goal_y = np.random.uniform(
            0, env.width), np.random.uniform(0, env.height)

    # 创建神经网络增强的 RRT
    neural_rrt = NeuralRRT(
        sampling_net=model_dict['sampling_net'],
        evaluation_net=model_dict['evaluation_net'],
        optimization_net=model_dict['optimization_net'],
        start=(start_x, start_y),
        goal=(goal_x, goal_y),
        env=env
    )

    # 规划路径
    print("使用训练好的模型规划路径...")
    path = neural_rrt.plan()

    if path:
        # 将 Node 对象转换为坐标元组
        path_coords = [(node.x, node.y) for node in path]
        # 执行路径
        simulator.execute_path(path_coords)
    else:
        print("未找到可行路径")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    print("命令行参数:", args)

    # 创建训练环境
    print("创建训练环境...")
    environments = create_training_environments()
    print(f"创建了 {len(environments)} 个训练环境")

    # 训练模型
    for i, env in enumerate(environments):
        print(f"\n训练环境 {i+1}/{len(environments)}")
        print(f"环境大小: {env.width}x{env.height}")
        print(f"障碍物数量: {len(env.obstacles)}")

        # 训练模型
        try:
            print("开始训练模型...")
            model_dict = train_neural_rrt(
                env=env,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            print("模型训练完成")

            # 可视化训练结果
            if args.visualize:
                print("\n可视化训练结果...")
                visualize_training(env, model_dict)

            # 保存模型
            if i == len(environments) - 1:  # 只保存最后一个环境训练的模型
                save_model(model_dict, args.save_dir)
        except Exception as e:
            import traceback
            print(f"训练过程中出错: {e}")
            traceback.print_exc()
            continue

    print("\n训练完成!")


if __name__ == "__main__":
    main()
