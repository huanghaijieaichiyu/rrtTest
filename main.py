#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RRT 路径规划算法集成工具

提供统一的命令行接口，用于：
1. 运行不同版本的 RRT 算法（RRT、RRT*、Informed RRT*）
2. 训练和测试基于深度学习的 RRT
3. 可视化和仿真路径规划结果
4. 创建和管理测试环境
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

from simulation.environment import Environment
from simulation.pygame_simulator import PygameSimulator
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar
from rrt.informed_rrt import InformedRRTStar
from ml.models.rrt_nn import NeuralRRT, train_neural_rrt


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='RRT 路径规划算法集成工具',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 运行 RRT 算法
    run_parser = subparsers.add_parser('run', help='运行路径规划算法')
    run_parser.add_argument(
        '--algorithm',
        type=str,
        choices=['rrt', 'rrt_star', 'informed_rrt', 'neural_rrt'],
        default='rrt_star',
        help='选择算法: rrt, rrt_star, informed_rrt, neural_rrt'
    )
    run_parser.add_argument(
        '--start',
        type=float,
        nargs=2,
        default=[10, 10],
        help='起点坐标，例如: "10 10"'
    )
    run_parser.add_argument(
        '--goal',
        type=float,
        nargs=2,
        default=[90, 90],
        help='终点坐标，例如: "90 90"'
    )
    run_parser.add_argument(
        '--map',
        type=str,
        default=None,
        help='地图文件路径'
    )
    run_parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='神经网络模型路径（仅用于 neural_rrt）'
    )
    run_parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='最大迭代次数'
    )
    run_parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='保存路径的文件路径'
    )

    # 训练神经网络 RRT
    train_parser = subparsers.add_parser('train', help='训练神经网络增强的 RRT')
    train_parser.add_argument(
        '--num-episodes',
        type=int,
        default=1000,
        help='训练数据收集的路径数量'
    )
    train_parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='训练轮数'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='学习率'
    )
    train_parser.add_argument(
        '--save-dir',
        type=str,
        default='results/models',
        help='模型保存目录'
    )
    train_parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='训练设备 (cuda/cpu)'
    )

    # 创建测试环境
    env_parser = subparsers.add_parser('create-env', help='创建测试环境')
    env_parser.add_argument(
        '--width',
        type=float,
        default=100.0,
        help='环境宽度'
    )
    env_parser.add_argument(
        '--height',
        type=float,
        default=100.0,
        help='环境高度'
    )
    env_parser.add_argument(
        '--num-circles',
        type=int,
        default=5,
        help='圆形障碍物数量'
    )
    env_parser.add_argument(
        '--num-rectangles',
        type=int,
        default=3,
        help='矩形障碍物数量'
    )
    env_parser.add_argument(
        '--save-path',
        type=str,
        required=True,
        help='保存环境的文件路径'
    )

    # 可视化选项
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='是否可视化执行过程'
    )

    return parser.parse_args()


def create_environment(args) -> Environment:
    """创建或加载环境"""
    if args.map and os.path.exists(args.map):
        # 从文件加载环境
        env = Environment.load(args.map)
        print(f"已加载环境: {args.map}")
    else:
        # 创建默认环境
        env = Environment(width=100.0, height=100.0)
        
        # 添加一些默认障碍物
        env.add_obstacle(x=30, y=30, obstacle_type="circle", radius=5.0)
        env.add_obstacle(x=50, y=50, obstacle_type="circle", radius=7.0)
        env.add_obstacle(x=70, y=40, obstacle_type="rectangle",
                        width=10.0, height=20.0)
        print("使用默认环境")

    return env


def create_random_environment(args) -> Environment:
    """创建随机环境"""
    env = Environment(width=args.width, height=args.height)

    # 添加随机圆形障碍物
    for _ in range(args.num_circles):
        x = np.random.uniform(10, args.width-10)
        y = np.random.uniform(10, args.height-10)
        radius = np.random.uniform(3, 8)
        env.add_obstacle(x=x, y=y, obstacle_type="circle", radius=radius)

    # 添加随机矩形障碍物
    for _ in range(args.num_rectangles):
        x = np.random.uniform(10, args.width-10)
        y = np.random.uniform(10, args.height-10)
        width = np.random.uniform(5, 15)
        height = np.random.uniform(5, 15)
        env.add_obstacle(x=x, y=y, obstacle_type="rectangle",
                        width=width, height=height)

    return env


def run_algorithm(args):
    """运行路径规划算法"""
    # 创建环境
    env = create_environment(args)

    # 选择算法
    if args.algorithm == 'rrt':
        planner = RRT(
            start=args.start,
            goal=args.goal,
            env=env,
            max_iterations=args.iterations
        )
    elif args.algorithm == 'rrt_star':
        planner = RRTStar(
            start=args.start,
            goal=args.goal,
            env=env,
            max_iterations=args.iterations
        )
    elif args.algorithm == 'informed_rrt':
        planner = InformedRRTStar(
            start=args.start,
            goal=args.goal,
            env=env,
            max_iterations=args.iterations
        )
    elif args.algorithm == 'neural_rrt':
        if not args.model_path:
            print("错误: 使用 neural_rrt 时必须提供模型路径")
            return
        planner = NeuralRRT(
            start=args.start,
            goal=args.goal,
            env=env,
            max_iterations=args.iterations
        )
        # 加载模型权重
        planner.load_state_dict(torch.load(args.model_path))

    # 规划路径
    print(f"使用 {args.algorithm} 算法规划路径...")
    path = planner.plan()

    if not path:
        print("未找到可行路径")
        return

    print(f"找到路径，长度: {len(path)} 个点")

    # 可视化
    if args.visualize:
        simulator = PygameSimulator()
        simulator.set_environment(env)
        simulator.execute_path(path)

    # 保存路径
    if args.save_path:
        import json
        with open(args.save_path, 'w') as f:
            json.dump(path, f)
        print(f"路径已保存到: {args.save_path}")


def train_neural_rrt_model(args):
    """训练神经网络增强的 RRT"""
    from examples.train_neural_rrt import train_neural_rrt, create_training_environments

    # 创建训练环境
    print("创建训练环境...")
    environments = create_training_environments()

    # 训练模型
    for i, env in enumerate(environments):
        print(f"\n训练环境 {i+1}/{len(environments)}")
        model = train_neural_rrt(
            env=env,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        # 可视化训练结果
        if args.visualize and i == len(environments) - 1:
            simulator = PygameSimulator()
            simulator.set_environment(env)
            
            # 测试训练好的模型
            start = (10, 10)
            goal = (90, 90)
            model.start = start
            model.goal = goal
            path = model.plan()
            
            if path:
                simulator.execute_path(path)

    print("训练完成!")


def create_test_environment(args):
    """创建测试环境"""
    # 创建随机环境
    env = create_random_environment(args)

    # 保存环境
    env.save(args.save_path)
    print(f"环境已保存到: {args.save_path}")

    # 可视化
    if args.visualize:
        simulator = PygameSimulator()
        simulator.set_environment(env)
        simulator.execute_path([(0, 0), (0, 0)])  # 显示空环境


def main():
    """主函数"""
    args = parse_args()

    if args.command == 'run':
        run_algorithm(args)
    elif args.command == 'train':
        train_neural_rrt_model(args)
    elif args.command == 'create-env':
        create_test_environment(args)
    else:
        print("请指定要执行的命令: run, train, 或 create-env")


if __name__ == "__main__":
    main()
