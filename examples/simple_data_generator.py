#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版数据生成器

专注于生成少量但高质量的训练样本
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import traceback

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from simulation.environment import Environment
    from rrt.rrt_star import RRTStar
    from ml.data.data_generator import DataGenerator, obstacle_to_dict, TrainingExample

    print("成功导入所需模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 创建保存目录
save_dir = os.path.join(root_dir, "data/training")
os.makedirs(save_dir, exist_ok=True)

print("初始化数据生成器...")
generator = DataGenerator(
    env_width=500.0,  # 减小环境尺寸
    env_height=300.0,  # 减小环境尺寸
    grid_size=(64, 64),
    min_obstacles=3,   # 减少障碍物数量
    max_obstacles=8,
    num_samples=100,   # 减少采样点数量
    rrt_iterations=3000
)

# 目标样本数量
num_examples = 10
print(f"\n开始生成 {num_examples} 个训练样本...")

examples = []
max_attempts = 100  # 最大尝试次数

with tqdm(total=num_examples, desc="生成样本") as pbar:
    attempt = 0
    while len(examples) < num_examples and attempt < max_attempts:
        attempt += 1
        try:
            # 手动创建简单环境
            env = Environment(width=500.0, height=300.0)

            # 添加少量障碍物
            env.add_obstacle(x=150, y=150, obstacle_type="circle", radius=30)
            env.add_obstacle(
                x=300, y=100, obstacle_type="rectangle", width=50, height=40)

            # 设置起点和终点
            start = (50, 50)
            goal = (450, 250)

            # 使用 RRT* 生成路径
            planner = RRTStar(
                start=start,
                goal=goal,
                env=env,
                step_size=20.0,  # 增大步长
                max_iterations=3000,
                goal_sample_rate=0.2  # 增加目标采样率
            )
            path = planner.plan()

            if path and len(path) >= 3:
                # 收集采样数据
                valid_samples, invalid_samples = generator.collect_samples(
                    env, start, goal)

                # 计算路径指标
                path_length, smoothness, clearance = generator.calculate_path_metrics(
                    path, env
                )

                # 创建环境的栅格化表示
                env_state = env.to_grid((64, 64))

                # 将障碍物转换为字典形式
                obstacle_dicts = [obstacle_to_dict(
                    obs) for obs in env.obstacles]

                # 创建训练样本
                example = TrainingExample(
                    env_state=env_state,
                    obstacles=obstacle_dicts,
                    start=start,
                    goal=goal,
                    path=path,
                    valid_samples=valid_samples,
                    invalid_samples=invalid_samples,
                    path_length=path_length,
                    smoothness=smoothness,
                    clearance=clearance
                )

                examples.append(example)
                pbar.update(1)
                print(f"\n成功生成样本 #{len(examples)}")
                print(f"  路径长度: {path_length:.2f}")
                print(f"  平滑度: {smoothness:.2f}")
                print(f"  间隙: {clearance:.2f}")
            else:
                print(f"\r尝试 #{attempt}: 路径规划失败", end="")
        except Exception as e:
            print(f"\n生成样本时出错: {str(e)}")
            traceback.print_exc()

print(f"\n成功生成 {len(examples)} 个训练样本")
print(f"总尝试次数: {attempt}")
print(f"成功率: {len(examples)/attempt*100:.2f}%")

if examples:
    # 创建数据集
    dataset = generator.generate_dataset(examples, save_dir=save_dir)
    print(f"\n数据集保存到: {save_dir}")
else:
    print("\n未能生成任何有效样本")
