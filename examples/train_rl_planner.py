#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
强化学习路径规划器训练脚本

简化的训练脚本，用于训练RL路径规划器。
"""

from training.train_rl_planner import train, evaluate
import os
import sys
import argparse
import time

# 添加项目根目录到路径，确保能够导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='强化学习路径规划器训练')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='模式：train（训练）或 evaluate（评估）')

    parser.add_argument('--env-type', type=str, default='complex',
                        choices=['simple', 'complex'],
                        help='环境类型')

    parser.add_argument('--episodes', type=int, default=5000,
                        help='训练回合数')

    parser.add_argument('--steps', type=int, default=100,
                        help='每个回合的最大步数')

    parser.add_argument('--save-interval', type=int, default=500,
                        help='保存模型的间隔回合数')

    parser.add_argument('--save-path', type=str, default='models/rl_planner',
                        help='模型保存路径')

    parser.add_argument('--model-path', type=str, default=None,
                        help='要评估的模型路径')

    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    parser.add_argument('--cuda', action='store_true',
                        help='使用CUDA加速')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置设备
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    if args.mode == 'train':
        print("开始训练强化学习路径规划器...")
        print(f"环境类型: {args.env_type}")
        print(f"训练回合数: {args.episodes}")
        print(f"每回合最大步数: {args.steps}")
        print(f"模型保存间隔: {args.save_interval}")
        print(f"模型保存路径: {args.save_path}")
        print(f"设备: {device}")

        # 记录开始时间
        start_time = time.time()

        # 训练模型
        train(
            env_type=args.env_type,
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            save_interval=args.save_interval,
            eval_interval=args.save_interval // 5,
            save_path=args.save_path,
            seed=args.seed,
            device=device
        )

        # 记录结束时间
        end_time = time.time()
        training_time = end_time - start_time

        print(f"训练完成！总耗时: {training_time:.2f}秒 ({training_time/3600:.2f}小时)")

    else:
        if args.model_path is None:
            print("请提供要评估的模型路径")
            return

        print("开始评估强化学习路径规划器...")
        print(f"模型路径: {args.model_path}")
        print(f"环境类型: {args.env_type}")

        # 评估模型
        stats = evaluate(
            model_path=args.model_path,
            env_type=args.env_type,
            num_episodes=100,
            render=False,
            seed=args.seed,
            device=device
        )

        print("\n评估结果:")
        print(f"成功率: {stats['success_rate']:.4f}")
        print(f"平均奖励: {stats['avg_reward']:.4f}")
        print(f"平均路径长度: {stats['avg_path_length']:.4f}")


if __name__ == "__main__":
    # 确保可以导入torch
    try:
        import torch
    except ImportError:
        print("错误: 未找到PyTorch。请安装: pip install torch")
        sys.exit(1)

    main()
