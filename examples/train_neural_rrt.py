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

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from simulation.environment import Environment
from simulation.pygame_simulator import PygameSimulator
from ml.models.rrt_nn import NeuralRRT, train_neural_rrt


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


def save_model(model: NeuralRRT, save_dir: str):
    """
    保存模型
    
    参数:
        model: 训练好的模型
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存各个网络
    torch.save(model.sampling_net.state_dict(),
              os.path.join(save_dir, f'sampling_net_{timestamp}.pth'))
    torch.save(model.evaluation_net.state_dict(),
              os.path.join(save_dir, f'evaluation_net_{timestamp}.pth'))
    torch.save(model.optimization_net.state_dict(),
              os.path.join(save_dir, f'optimization_net_{timestamp}.pth'))
    
    print(f"模型已保存到: {save_dir}")


def visualize_training(env: Environment, model: NeuralRRT):
    """
    可视化训练过程
    
    参数:
        env: 环境对象
        model: 当前模型
    """
    # 创建仿真器
    simulator = PygameSimulator()
    simulator.set_environment(env)
    
    # 随机生成起点和终点
    start = (np.random.uniform(10, env.width-10),
            np.random.uniform(10, env.height-10))
    goal = (np.random.uniform(10, env.width-10),
           np.random.uniform(10, env.height-10))
    
    # 设置模型的起点和终点
    model.start = start
    model.goal = goal
    
    # 规划路径
    path = model.plan()
    
    if path:
        # 执行路径
        simulator.execute_path(path)
    else:
        print("未找到可行路径")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建训练环境
    print("创建训练环境...")
    environments = create_training_environments()
    
    # 训练模型
    for i, env in enumerate(environments):
        print(f"\n训练环境 {i+1}/{len(environments)}")
        
        # 训练模型
        model = train_neural_rrt(
            env=env,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # 可视化训练结果
        if args.visualize:
            print("\n可视化训练结果...")
            visualize_training(env, model)
        
        # 保存模型
        if i == len(environments) - 1:  # 只保存最后一个环境训练的模型
            save_model(model, args.save_dir)
    
    print("\n训练完成!")


if __name__ == "__main__":
    main() 