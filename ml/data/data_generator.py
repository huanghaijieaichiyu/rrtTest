#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练数据生成器

用于生成神经网络增强 RRT 算法的训练数据，包括：
1. 采样数据：用于训练采样网络
2. 路径评估数据：用于训练评估网络
3. 路径优化数据：用于训练优化网络
"""

import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from simulation.environment import Environment
from rrt.rrt_star import RRTStar


@dataclass
class TrainingExample:
    """单个训练样本的数据结构"""
    # 环境状态
    env_state: np.ndarray  # 环境的栅格化表示
    obstacles: List[Dict[str, Any]]  # 障碍物列表，每个障碍物转换为字典形式

    # 路径规划相关
    start: Tuple[float, float]  # 起点
    goal: Tuple[float, float]   # 终点
    path: List[Tuple[float, float]]  # RRT* 生成的路径

    # 采样点相关
    valid_samples: List[Tuple[float, float]]  # 有效的采样点
    invalid_samples: List[Tuple[float, float]]  # 无效的采样点

    # 评估指标
    path_length: float  # 路径长度
    smoothness: float   # 平滑度
    clearance: float    # 与障碍物的间距


def obstacle_to_dict(obstacle) -> Dict[str, Any]:
    """将障碍物对象转换为字典形式"""
    if hasattr(obstacle, 'radius'):  # 圆形障碍物
        return {
            'type': 'circle',
            'x': obstacle.x,
            'y': obstacle.y,
            'radius': obstacle.radius
        }
    else:  # 矩形障碍物
        return {
            'type': 'rectangle',
            'x': obstacle.x,
            'y': obstacle.y,
            'width': obstacle.width,
            'height': obstacle.height,
            'angle': obstacle.angle if hasattr(obstacle, 'angle') else 0.0
        }


class RRTDataset(Dataset):
    """RRT 训练数据集"""

    def __init__(
        self,
        examples: List[TrainingExample],
        grid_size: Tuple[int, int] = (64, 64),
        transform=None
    ):
        self.examples = examples
        self.grid_size = grid_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # 将环境转换为张量
        env_tensor = torch.from_numpy(example.env_state).float()

        # 构建输入特征
        features = {
            'env_state': env_tensor,
            'start': torch.tensor(example.start).float(),
            'goal': torch.tensor(example.goal).float(),
            'path': torch.tensor(example.path).float(),
            'valid_samples': torch.tensor(example.valid_samples).float(),
            'invalid_samples': torch.tensor(example.invalid_samples).float(),
            'path_length': torch.tensor(example.path_length).float(),
            'smoothness': torch.tensor(example.smoothness).float(),
            'clearance': torch.tensor(example.clearance).float()
        }

        if self.transform:
            features = self.transform(features)

        return features


class DataGenerator:
    """训练数据生成器"""

    def __init__(
        self,
        env_width: float = 100.0,
        env_height: float = 100.0,
        grid_size: Tuple[int, int] = (64, 64),
        min_obstacles: int = 3,
        max_obstacles: int = 8,
        num_samples: int = 1000,
        rrt_iterations: int = 2000
    ):
        self.env_width = env_width
        self.env_height = env_height
        self.grid_size = grid_size
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.num_samples = num_samples
        self.rrt_iterations = rrt_iterations

    def generate_random_environment(self) -> Environment:
        """生成随机环境"""
        env = Environment(width=self.env_width, height=self.env_height)

        # 随机生成障碍物数量
        num_obstacles = np.random.randint(
            self.min_obstacles, self.max_obstacles + 1
        )

        for _ in range(num_obstacles):
            if np.random.random() < 0.7:  # 70% 概率生成圆形障碍物
                x = np.random.uniform(10, self.env_width - 10)
                y = np.random.uniform(10, self.env_height - 10)
                radius = np.random.uniform(3, 8)
                env.add_obstacle(
                    x=x, y=y,
                    obstacle_type="circle",
                    radius=radius
                )
            else:  # 30% 概率生成矩形障碍物
                x = np.random.uniform(10, self.env_width - 10)
                y = np.random.uniform(10, self.env_height - 10)
                width = np.random.uniform(5, 15)
                height = np.random.uniform(5, 15)
                angle = np.random.uniform(0, 2 * np.pi)
                env.add_obstacle(
                    x=x, y=y,
                    obstacle_type="rectangle",
                    width=width, height=height,
                    angle=angle
                )

        return env

    def generate_random_points(
        self,
        env: Environment
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """生成随机的起点和终点"""
        while True:
            start = (
                np.random.uniform(5, self.env_width - 5),
                np.random.uniform(5, self.env_height - 5)
            )
            goal = (
                np.random.uniform(5, self.env_width - 5),
                np.random.uniform(5, self.env_height - 5)
            )

            # 确保起点和终点不在障碍物内
            if (not env.check_collision(start) and
                not env.check_collision(goal) and
                    np.linalg.norm(np.array(goal) - np.array(start)) > 20.0):
                return start, goal

    def collect_samples(
        self,
        env: Environment,
        start: Tuple[float, float],
        goal: Tuple[float, float]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """收集采样点数据"""
        valid_samples = []
        invalid_samples = []

        for _ in range(self.num_samples):
            # 随机采样
            sample = (
                np.random.uniform(0, self.env_width),
                np.random.uniform(0, self.env_height)
            )

            # 检查采样点是否有效
            if env.check_collision(sample):
                invalid_samples.append(sample)
            else:
                valid_samples.append(sample)

        return valid_samples, invalid_samples

    def calculate_path_metrics(
        self,
        path: List[Tuple[float, float]],
        env: Environment
    ) -> Tuple[float, float, float]:
        """计算路径的评估指标"""
        if not path:
            return 0.0, 0.0, 0.0

        # 计算路径长度
        path_length = 0.0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(
                np.array(path[i+1]) - np.array(path[i])
            )

        # 计算平滑度（相邻路径段的角度变化）
        smoothness = 0.0
        if len(path) > 2:
            for i in range(len(path) - 2):
                v1 = np.array(path[i+1]) - np.array(path[i])
                v2 = np.array(path[i+2]) - np.array(path[i+1])
                angle = np.arccos(np.clip(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                    -1.0, 1.0
                ))
                smoothness += angle
            smoothness /= (len(path) - 2)

        # 计算与障碍物的最小间距
        clearance = float('inf')
        for point in path:
            point_array = np.array(point)
            for obstacle in env.obstacles:
                obstacle_dict = obstacle_to_dict(obstacle)
                if obstacle_dict['type'] == 'circle':
                    dist = np.linalg.norm(
                        point_array - np.array((
                            obstacle_dict['x'],
                            obstacle_dict['y']
                        ))
                    ) - obstacle_dict['radius']
                else:  # rectangle
                    # 简化处理，使用到中心点的距离
                    dist = np.linalg.norm(
                        point_array - np.array((
                            obstacle_dict['x'],
                            obstacle_dict['y']
                        ))
                    ) - max(
                        obstacle_dict['width'],
                        obstacle_dict['height']
                    ) / 2
                clearance = min(clearance, dist)

        return path_length, smoothness, clearance

    def generate_example(self) -> Optional[TrainingExample]:
        """生成单个训练样本"""
        # 生成随机环境
        env = self.generate_random_environment()

        # 生成随机起点和终点
        start, goal = self.generate_random_points(env)

        # 使用 RRT* 生成基准路径
        planner = RRTStar(
            start=start,
            goal=goal,
            env=env,
            max_iterations=self.rrt_iterations,
            step_size=50.0,  # 增加步长以提高成功率
            goal_sample_rate=0.2  # 增加目标采样率
        )
        path = planner.plan()

        if not path:
            # 路径规划失败，返回 None
            return None

        # 确保路径至少有 3 个点，否则无法计算平滑度
        if len(path) < 3:
            return None

        # 收集采样数据
        valid_samples, invalid_samples = self.collect_samples(env, start, goal)

        # 计算路径指标
        path_length, smoothness, clearance = self.calculate_path_metrics(
            path, env
        )

        # 创建环境的栅格化表示
        env_state = env.to_grid(self.grid_size)

        # 将障碍物转换为字典形式
        obstacle_dicts = [obstacle_to_dict(obs) for obs in env.obstacles]

        return TrainingExample(
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

    def generate_dataset(
        self,
        examples_or_num: Union[List[TrainingExample], int],
        save_dir: Optional[str] = None
    ) -> RRTDataset:
        """
        生成训练数据集

        参数:
            examples_or_num: 预生成的示例列表或要生成的示例数量
            save_dir: 保存数据集的目录路径

        返回:
            RRTDataset 对象
        """
        if isinstance(examples_or_num, list):
            examples = examples_or_num
        else:
            examples = []
            for i in range(examples_or_num):
                print(f"\r生成样本 {i+1}/{examples_or_num}", end="")
                example = self.generate_example()
                if example is not None:
                    examples.append(example)
            print()  # 换行

        # 创建数据集
        dataset = RRTDataset(examples, self.grid_size)

        # 保存数据集
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(dataset, os.path.join(save_dir, 'rrt_dataset.pt'))

        return dataset


def create_data_loaders(
    dataset: RRTDataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """创建训练集和验证集的数据加载器"""
    # 划分数据集
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
