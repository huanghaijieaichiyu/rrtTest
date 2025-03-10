#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练数据准备示例脚本

展示如何使用数据生成器创建训练数据集，包括：
1. 生成随机环境和路径规划数据
2. 保存和加载数据集
3. 数据集的基本统计信息
"""

from ml.data import (
    DataGenerator,
    create_data_loaders
)
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='训练数据准备工具',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--num-examples',
        type=int,
        default=1000,
        help='要生成的训练样本数量'
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/training',
        help='数据保存目录'
    )

    parser.add_argument(
        '--env-width',
        type=float,
        default=1000.0,
        help='环境宽度'
    )

    parser.add_argument(
        '--env-height',
        type=float,
        default=1000.0,
        help='环境高度'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        nargs=2,
        default=[64, 64],
        help='环境栅格化大小'
    )

    parser.add_argument(
        '--min-obstacles',
        type=int,
        default=10,
        help='最少障碍物数量'
    )

    parser.add_argument(
        '--max-obstacles',
        type=int,
        default=20,
        help='最多障碍物数量'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='每个环境的采样点数量'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小'
    )

    return parser.parse_args()


def print_dataset_stats(dataset):
    """打印数据集统计信息"""
    print("\n数据集统计信息:")
    print(f"样本数量: {len(dataset)}")

    # 计算路径长度统计
    path_lengths = [ex.path_length for ex in dataset.examples]
    print("\n路径长度统计:")
    print(f"  最小值: {min(path_lengths):.2f}")
    print(f"  最大值: {max(path_lengths):.2f}")
    print(f"  平均值: {np.mean(path_lengths):.2f}")
    print(f"  标准差: {np.std(path_lengths):.2f}")

    # 计算平滑度统计
    smoothness = [ex.smoothness for ex in dataset.examples]
    print("\n平滑度统计:")
    print(f"  最小值: {min(smoothness):.2f}")
    print(f"  最大值: {max(smoothness):.2f}")
    print(f"  平均值: {np.mean(smoothness):.2f}")
    print(f"  标准差: {np.std(smoothness):.2f}")

    # 计算间隙统计
    clearance = [ex.clearance for ex in dataset.examples]
    print("\n障碍物间隙统计:")
    print(f"  最小值: {min(clearance):.2f}")
    print(f"  最大值: {max(clearance):.2f}")
    print(f"  平均值: {np.mean(clearance):.2f}")
    print(f"  标准差: {np.std(clearance):.2f}")

    # 计算采样点统计
    valid_samples = [len(ex.valid_samples) for ex in dataset.examples]
    invalid_samples = [len(ex.invalid_samples) for ex in dataset.examples]
    total_samples = np.mean(valid_samples) + np.mean(invalid_samples)
    valid_ratio = np.mean(valid_samples) / \
        total_samples if total_samples > 0 else 0

    print("\n采样点统计:")
    print(f"  平均有效采样点数: {np.mean(valid_samples):.2f}")
    print(f"  平均无效采样点数: {np.mean(invalid_samples):.2f}")
    print(f"  有效采样率: {valid_ratio:.2%}")


def main():
    """主函数"""
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    print("初始化数据生成器...")
    generator = DataGenerator(
        env_width=500.0,  # 减小环境尺寸
        env_height=300.0,  # 减小环境尺寸
        grid_size=tuple(args.grid_size),
        min_obstacles=3,    # 减少障碍物数量以提高成功率
        max_obstacles=8,
        num_samples=100,    # 减少采样点数量
        rrt_iterations=2000  # 增加迭代次数
    )

    print(f"\n开始生成 {args.num_examples} 个训练样本...")
    examples = []
    pbar = tqdm(total=args.num_examples, desc="生成样本")

    failed_attempts = 0  # 记录失败次数
    max_attempts = args.num_examples * 10  # 最大尝试次数
    attempt = 0

    while len(examples) < args.num_examples and attempt < max_attempts:
        attempt += 1
        try:
            example = generator.generate_example()
            if example is not None:
                examples.append(example)
                pbar.update(1)
                failed_attempts = 0  # 重置失败计数
            else:
                failed_attempts += 1
                if failed_attempts % 10 == 0:  # 每10次失败打印一次
                    print(f"\n连续失败 {failed_attempts} 次，已尝试 {attempt} 次")
        except Exception as e:
            print(f"\n生成样本时出错: {str(e)}")
            failed_attempts += 1

    pbar.close()

    if len(examples) < args.num_examples:
        print(f"\n警告: 只生成了 {len(examples)} 个有效样本")
        if not examples:
            print("错误: 未能生成任何有效样本")
            return

    print(f"\n成功生成 {len(examples)} 个训练样本")
    print(f"总尝试次数: {attempt}")
    print(f"成功率: {len(examples)/attempt*100:.2f}%")

    # 创建数据集
    dataset = generator.generate_dataset(examples, save_dir=args.save_dir)

    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size
    )

    print("\n数据集创建完成!")
    print(f"保存路径: {args.save_dir}")

    # 修复 linter 错误
    train_size = len(train_loader) * args.batch_size
    val_size = len(val_loader) * args.batch_size
    print(f"训练集大小: 约 {train_size} 个样本")
    print(f"验证集大小: 约 {val_size} 个样本")

    # 打印统计信息
    print_dataset_stats(dataset)

    print("\n数据集示例:")
    batch = next(iter(train_loader))
    print("批次数据形状:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    main()
