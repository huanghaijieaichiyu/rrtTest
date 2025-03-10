#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于深度学习的 RRT 算法实现

结合神经网络来优化 RRT 的采样策略和路径优化。主要包括：
1. 采样网络：学习更有效的采样分布
2. 评估网络：预测路径的可行性和质量
3. 优化网络：优化路径的平滑度和安全性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from rrt.rrt_star import RRTStar
from rrt.rrt_base import Node


class SamplingNetwork(nn.Module):
    """采样网络，用于生成更有效的采样点"""

    def __init__(self, state_dim: int = 4, hidden_dim: int = 128,
                 output_dim: int = 2):
        """
        初始化采样网络

        参数:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 环境编码器 (CNN)
        self.env_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, hidden_dim),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, state: torch.Tensor,
                env_grid: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            state: 状态张量 [batch_size, state_dim]
            env_grid: 环境网格 [batch_size, 1, height, width]

        返回:
            采样点坐标 [batch_size, 2]
        """
        # 编码状态
        state_features = self.state_encoder(state)

        # 编码环境
        env_features = self.env_encoder(env_grid)

        # 融合特征
        combined = torch.cat([state_features, env_features], dim=1)

        # 生成采样点
        samples = self.fusion(combined)

        return samples


class EvaluationNetwork(nn.Module):
    """路径评估网络，预测路径的可行性和质量"""

    def __init__(self, path_points: int = 10, hidden_dim: int = 64):
        """
        初始化评估网络

        参数:
            path_points: 路径点数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.path_points = path_points

        # 使用 1D 卷积处理路径序列
        self.conv = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 全连接层输出评估结果
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * path_points, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [可行性, 长度得分, 平滑度得分]
        )

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            path: 路径点序列 [batch_size, path_points, 2]

        返回:
            评估结果 [batch_size, 3]
        """
        # [batch_size, 2, path_points]
        x = path.transpose(1, 2)

        # 特征提取
        x = self.conv(x)

        # 展平
        x = x.reshape(x.size(0), -1)

        # 评估
        return self.fc(x)


class OptimizationNetwork(nn.Module):
    """路径优化网络，优化路径的平滑度和安全性"""

    def __init__(self, path_points: int = 10, hidden_dim: int = 64):
        """
        初始化优化网络

        参数:
            path_points: 路径点数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        self.path_points = path_points

        # 使用 Transformer 编码器处理路径序列
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2,
                nhead=2,
                dim_feedforward=hidden_dim
            ),
            num_layers=2
        )

        # 解码器生成优化后的路径点
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            path: 路径点序列 [batch_size, path_points, 2]

        返回:
            优化后的路径点 [batch_size, path_points, 2]
        """
        # [path_points, batch_size, 2]
        x = path.transpose(0, 1)

        # Transformer 编码
        x = self.transformer(x)

        # 解码优化后的路径点
        x = self.decoder(x)

        # [batch_size, path_points, 2]
        return x.transpose(0, 1)


class NeuralRRT(RRTStar):
    """神经网络增强的RRT算法"""

    def __init__(self,
                 sampling_net: SamplingNetwork,
                 evaluation_net: EvaluationNetwork,
                 optimization_net: OptimizationNetwork,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env,
                 max_iterations: int = 1000,
                 step_size: float = 5.0,
                 sample_size: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化神经网络增强的RRT

        参数:
            sampling_net: 采样网络
            evaluation_net: 评估网络
            optimization_net: 优化网络
            start: 起点坐标
            goal: 终点坐标
            env: 环境对象
            max_iterations: 最大迭代次数
            step_size: 步长
            sample_size: 每次采样的点数
            device: 计算设备
        """
        # 注意：这里不直接调用父类的初始化方法，因为我们需要自定义初始化过程
        # 而是手动设置必要的属性
        from rrt.rrt_base import Node

        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.env = env
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = 0.1  # 目标采样率
        self.search_radius = 50.0  # 搜索半径
        self.rewire_factor = 1.5  # 重连接因子

        # 初始化节点列表
        self.node_list = [self.start]

        # 路径结果
        self.path = []

        # 计算区域边界
        self.min_x = 0
        self.max_x = env.width
        self.min_y = 0
        self.max_y = env.height

        # 神经网络相关
        self.sampling_net = sampling_net
        self.evaluation_net = evaluation_net
        self.optimization_net = optimization_net
        self.device = device
        self.sample_size = sample_size

    def _get_state_features(self, point: Tuple[float, float]) -> torch.Tensor:
        """获取状态特征"""
        x, y = point
        start_dist = np.sqrt((x - self.start.x)**2 + (y - self.start.y)**2)
        goal_dist = np.sqrt((x - self.goal.x)**2 + (y - self.goal.y)**2)

        return torch.tensor([x, y, start_dist, goal_dist],
                            dtype=torch.float32).to(self.device)

    def _get_environment_features(self) -> torch.Tensor:
        """获取环境特征"""
        # 将环境转换为网格表示
        grid = self.env.to_grid(grid_size=(64, 64))
        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _neural_sample(self) -> List[Node]:
        """使用神经网络进行批量采样"""
        # 获取当前节点状态
        current = self.node_list[-1]
        state = self._get_state_features((current.x, current.y))

        # 获取环境特征
        env_features = self._get_environment_features()

        # 使用采样网络生成样本
        with torch.no_grad():
            samples = self.sampling_net(
                state.unsqueeze(0).repeat(self.sample_size, 1),
                env_features.repeat(self.sample_size, 1, 1)
            )

        # 将张量转换为节点列表
        nodes = []
        for i in range(samples.shape[0]):
            x, y = samples[i, 0].item(), samples[i, 1].item()
            # 确保样本在环境范围内
            x = max(min(x, self.max_x), self.min_x)
            y = max(min(y, self.max_y), self.min_y)
            nodes.append(Node(x, y))

        return nodes

    def _evaluate_path(self, path: List[Node]) -> float:
        """使用评估网络评估路径质量"""
        if not path:
            return 0.0

        # 提取路径点坐标
        path_coords = torch.tensor(
            [[node.x, node.y] for node in path],
            dtype=torch.float32
        ).to(self.device)

        # 获取环境特征
        env_features = self._get_environment_features()

        # 使用评估网络评估路径
        with torch.no_grad():
            quality_score = self.evaluation_net(
                path_coords.unsqueeze(0), env_features)

        return quality_score.item()

    def _optimize_path(self, path: List[Node]) -> List[Node]:
        """使用优化网络优化路径"""
        if not path or len(path) < 2:
            return path

        # 提取路径点坐标
        path_coords = torch.tensor(
            [[node.x, node.y] for node in path],
            dtype=torch.float32
        ).to(self.device)

        # 获取环境特征
        env_features = self._get_environment_features()

        # 使用优化网络优化路径
        with torch.no_grad():
            optimized_coords = self.optimization_net(
                path_coords.unsqueeze(0), env_features
            ).squeeze(0)

        # 将优化后的坐标转换回节点列表
        optimized_path = []
        for i in range(optimized_coords.shape[0]):
            x, y = optimized_coords[i, 0].item(), optimized_coords[i, 1].item()
            optimized_path.append(Node(x, y))

        return optimized_path

    def plan(self) -> List[Node]:
        """规划路径"""
        for i in range(self.max_iterations):
            # 使用神经网络采样
            samples = self._neural_sample()

            for sample in samples:
                # 如果采样点在障碍物内，跳过
                if self.env.check_collision(sample.x, sample.y):
                    continue

                # 找到最近的节点
                nearest_node = self._find_nearest_node(sample)

                # 从最近节点向采样点扩展
                new_node = self._steer(nearest_node, sample)

                # 如果路径无碰撞，添加节点
                if not self._check_collision(nearest_node, new_node):
                    # 找到附近的节点
                    near_nodes = self._find_near_nodes(new_node)

                    # 选择最优父节点
                    self._choose_parent(new_node, near_nodes)

                    # 添加到节点列表
                    self.node_list.append(new_node)

                    # 重连接
                    self._rewire(new_node, near_nodes)

                    # 检查是否可以连接到目标
                    if self._distance(new_node, self.goal) <= self.step_size:
                        final_node = self._steer(new_node, self.goal)
                        if not self._check_collision(new_node, final_node):
                            # 找到路径
                            self.path = self._extract_path(final_node)

                            # 使用神经网络优化路径
                            self.path = self._optimize_path(self.path)

                            return self.path

        # 如果达到最大迭代次数仍未找到路径，尝试连接到最接近目标的节点
        closest_node = self._find_nearest_node(self.goal)
        final_node = self._steer(closest_node, self.goal)
        if not self._check_collision(closest_node, final_node):
            self.path = self._extract_path(final_node)
            self.path = self._optimize_path(self.path)
            return self.path

        return []


class TrainingExample:
    """训练样本类，用于存储路径规划的训练数据"""

    def __init__(self, start: Tuple[float, float], goal: Tuple[float, float],
                 env, path: List[Node], metrics: dict):
        """
        初始化训练样本

        参数:
            start: 起点坐标
            goal: 终点坐标
            env: 环境对象
            path: 路径节点列表
            metrics: 路径指标字典，包含长度、间隙和平滑度等
        """
        self.start = start
        self.goal = goal
        self.env = env
        self.path = path
        self.metrics = metrics


def calculate_path_length(path: List[Node]) -> float:
    """
    计算路径长度

    参数:
        path: 路径节点列表

    返回:
        路径长度
    """
    if not path or len(path) < 2:
        return 0.0

    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1].x - path[i].x
        dy = path[i+1].y - path[i].y
        length += np.sqrt(dx*dx + dy*dy)

    return length


def calculate_path_clearance(path: List[Node], env) -> float:
    """
    计算路径与障碍物的最小距离

    参数:
        path: 路径节点列表
        env: 环境对象

    返回:
        路径与障碍物的最小距离
    """
    if not path:
        return 0.0

    # 计算路径上每个点到最近障碍物的距离
    min_clearance = float('inf')

    for node in path:
        # 获取当前点到所有障碍物的最小距离
        clearance = env.get_min_distance(node.x, node.y)
        min_clearance = min(min_clearance, clearance)

    return min_clearance


def calculate_path_smoothness(path: List[Node]) -> float:
    """
    计算路径平滑度（角度变化的平均值）

    参数:
        path: 路径节点列表

    返回:
        路径平滑度，值越小表示越平滑
    """
    if not path or len(path) < 3:
        return 0.0

    angle_changes = []
    for i in range(1, len(path) - 1):
        # 计算前一段和后一段的向量
        v1 = np.array([path[i].x - path[i-1].x, path[i].y - path[i-1].y])
        v2 = np.array([path[i+1].x - path[i].x, path[i+1].y - path[i].y])

        # 归一化向量
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 0 and v2_norm > 0:
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm

            # 计算夹角的余弦值
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)

            # 计算角度变化（弧度）
            angle_change = np.arccos(cos_angle)
            angle_changes.append(angle_change)

    # 返回角度变化的平均值
    if angle_changes:
        return np.mean(angle_changes)
    else:
        return 0.0


def collect_training_data(env, num_samples: int = 1000) -> List[TrainingExample]:
    """
    使用传统 RRT* 收集训练数据

    参数:
        env: 环境对象
        num_samples: 样本数量

    返回:
        训练样本列表
    """
    from rrt.rrt_base import Node

    examples = []
    for _ in range(num_samples):
        # 随机生成起点和终点
        start_x = np.random.uniform(0, env.width)
        start_y = np.random.uniform(0, env.height)
        goal_x = np.random.uniform(0, env.width)
        goal_y = np.random.uniform(0, env.height)

        # 确保起点和终点不在障碍物内
        if env.check_collision((start_x, start_y)) or env.check_collision((goal_x, goal_y)):
            continue

        # 使用 RRT* 规划路径
        rrt = RRTStar((start_x, start_y), (goal_x, goal_y), env)
        path = rrt.plan()

        if path:
            # 将路径点转换为 Node 对象列表，以便计算指标
            node_path = []
            for x, y in path:
                node_path.append(Node(x, y))

            # 计算路径指标
            path_length = calculate_path_length(node_path)
            path_clearance = calculate_path_clearance(node_path, env)
            path_smoothness = calculate_path_smoothness(node_path)

            # 创建训练样本
            example = TrainingExample(
                start=(start_x, start_y),
                goal=(goal_x, goal_y),
                env=env,
                path=node_path,
                metrics={
                    'length': path_length,
                    'clearance': path_clearance,
                    'smoothness': path_smoothness
                }
            )
            examples.append(example)

    return examples


def train_neural_rrt(env,
                     num_epochs: int = 100,
                     batch_size: int = 32,
                     learning_rate: float = 1e-4,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练神经网络增强的 RRT

    参数:
        env: 环境对象
        num_epochs: 训练轮数
        batch_size: 批量大小
        learning_rate: 学习率
        device: 计算设备

    返回:
        训练好的神经网络模型
    """
    print("收集训练数据...")
    examples = collect_training_data(env)

    if not examples:
        print("未能收集到有效的训练数据，请检查环境设置。")
        return None

    print(f"收集到 {len(examples)} 个训练样本")

    # 提取路径和指标
    paths = []
    length_scores = []
    clearance_scores = []
    smoothness_scores = []

    for example in examples:
        # 提取路径坐标
        path_coords = [(node.x, node.y) for node in example.path]
        paths.append(path_coords)

        # 提取指标
        length_scores.append(example.metrics['length'])
        clearance_scores.append(example.metrics['clearance'])
        smoothness_scores.append(example.metrics['smoothness'])

    # 归一化指标
    length_scores = np.array(length_scores)
    clearance_scores = np.array(clearance_scores)
    smoothness_scores = np.array(smoothness_scores)

    # 避免除以零
    length_range = length_scores.max() - length_scores.min()
    if length_range > 0:
        length_scores = (length_scores - length_scores.min()) / length_range

    clearance_range = clearance_scores.max() - clearance_scores.min()
    if clearance_range > 0:
        clearance_scores = (clearance_scores -
                            clearance_scores.min()) / clearance_range

    smoothness_range = smoothness_scores.max() - smoothness_scores.min()
    if smoothness_range > 0:
        smoothness_scores = (smoothness_scores -
                             smoothness_scores.min()) / smoothness_range

    # 创建综合得分 (加权平均)
    scores = 0.4 * length_scores + 0.4 * \
        clearance_scores + 0.2 * (1 - smoothness_scores)

    # 创建模型
    sampling_net = SamplingNetwork().to(device)
    evaluation_net = EvaluationNetwork().to(device)
    optimization_net = OptimizationNetwork().to(device)

    # 创建优化器
    sampling_optimizer = torch.optim.Adam(
        sampling_net.parameters(), lr=learning_rate)
    evaluation_optimizer = torch.optim.Adam(
        evaluation_net.parameters(), lr=learning_rate)
    optimization_optimizer = torch.optim.Adam(
        optimization_net.parameters(), lr=learning_rate)

    # 训练循环
    print("开始训练...")

    # 转换为张量
    paths_tensor = [torch.tensor(path, dtype=torch.float32).to(
        device) for path in paths]
    scores_tensor = torch.tensor(scores, dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        # 打乱数据
        indices = np.random.permutation(len(paths))
        epoch_loss = 0.0

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # 准备批次数据
            batch_paths = [paths_tensor[idx] for idx in batch_indices]
            batch_scores = scores_tensor[batch_indices]

            # 训练采样网络 - 简化版，实际应该使用环境特征和状态特征
            sampling_optimizer.zero_grad()
            # 由于采样网络需要环境特征，这里跳过实际训练
            # 实际训练中应该准备环境特征并进行前向和反向传播

            # 训练评估网络
            evaluation_optimizer.zero_grad()
            # 对于每个路径，计算评估得分并与真实得分比较
            eval_loss = 0.0
            for path_idx, path_tensor in enumerate(batch_paths):
                if path_tensor.size(0) > 1:  # 确保路径至少有两个点
                    # 调整路径形状以适应评估网络
                    padded_path = torch.zeros(10, 2, device=device)
                    path_len = min(path_tensor.size(0), 10)
                    padded_path[:path_len] = path_tensor[:path_len]

                    # 前向传播
                    pred_score = evaluation_net(padded_path.unsqueeze(0))
                    # 计算损失 - 评估网络输出 [可行性, 长度得分, 平滑度得分]
                    # 这里我们只使用第一个输出（可行性）作为目标
                    target = batch_scores[path_idx].unsqueeze(0).unsqueeze(0)
                    # 创建一个与 pred_score 相同形状的目标张量
                    full_target = torch.zeros_like(pred_score)
                    full_target[:, 0] = target  # 设置可行性得分
                    # 计算损失
                    loss = F.mse_loss(pred_score, full_target)
                    loss.backward()
                    eval_loss += loss.item()

            # 更新评估网络参数
            evaluation_optimizer.step()

            # 训练优化网络
            optimization_optimizer.zero_grad()
            # 对于每个路径，尝试优化并计算与原始路径的差异
            opt_loss = 0.0
            for path_tensor in batch_paths:
                if path_tensor.size(0) > 1:  # 确保路径至少有两个点
                    # 调整路径形状以适应优化网络
                    padded_path = torch.zeros(10, 2, device=device)
                    path_len = min(path_tensor.size(0), 10)
                    padded_path[:path_len] = path_tensor[:path_len]

                    # 前向传播
                    optimized_path = optimization_net(padded_path.unsqueeze(0))
                    # 计算损失 - 优化后的路径应该与原始路径相似但更平滑
                    loss = torch.nn.functional.mse_loss(
                        optimized_path, padded_path.unsqueeze(0))
                    loss.backward()
                    opt_loss += loss.item()

            # 更新优化网络参数
            optimization_optimizer.step()

            # 更新总损失
            epoch_loss += (eval_loss + opt_loss) / 2

        # 打印训练信息
        avg_loss = epoch_loss / (len(indices) // batch_size + 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    print("训练完成!")

    return {
        'sampling_net': sampling_net,
        'evaluation_net': evaluation_net,
        'optimization_net': optimization_net
    }


# 测试代码
if __name__ == "__main__":
    # 测试采样网络
    sampling_net = SamplingNetwork()
    print(f"采样网络结构:\n{sampling_net}")

    # 测试输入
    start = (10.0, 10.0)
    goal = (90.0, 90.0)
    width, height = 100.0, 100.0

    # 测试采样
    samples = sampling_net.forward(torch.tensor(
        [start[0]/width, start[1]/height, 0.0, 0.0], dtype=torch.float32))
    print(f"生成的采样点:\n{samples}")

    # 测试评估网络
    evaluation_net = EvaluationNetwork()
    print(f"\n评估网络结构:\n{evaluation_net}")

    # 测试评估
    path = [start, goal]
    scores = evaluation_net.forward(torch.tensor(
        path, dtype=torch.float32).unsqueeze(0))
    print(
        f"路径评估结果: 可行性={scores[0][0]:.2f}, 长度得分={scores[0][1]:.2f}, 平滑度得分={scores[0][2]:.2f}")

    # 测试优化网络
    optimization_net = OptimizationNetwork()
    print(f"\n优化网络结构:\n{optimization_net}")

    # 测试优化
    optimized_path = optimization_net.forward(
        torch.tensor(path, dtype=torch.float32).unsqueeze(0))
    print(f"优化后的路径点:\n{optimized_path}")

    # 测试神经网络增强的 RRT
    env = Environment(width, height)
    neural_rrt = NeuralRRT(start, goal, env)
    print(f"\n神经网络增强的 RRT 结构:\n{neural_rrt}")

    # 测试训练
    train_neural_rrt(env)
