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
from typing import List, Tuple, Dict, Any, Optional
from rrt.rrt_star import RRTStar


class SamplingNetwork(nn.Module):
    """采样策略网络，学习更有效的采样分布"""
    
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64):
        """
        初始化采样网络
        
        参数:
            state_dim: 状态空间维度 (x, y, 起点距离, 终点距离)
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 输出采样点的 (x, y) 坐标
            nn.Tanh()  # 将输出映射到 [-1, 1]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            state: 当前状态 [batch_size, state_dim]
            
        返回:
            采样点坐标 [batch_size, 2]
        """
        return self.net(state)


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
    """基于神经网络增强的 RRT 算法"""
    
    def __init__(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 env: 'Environment',
                 max_iterations: int = 1000,
                 step_size: float = 5.0,
                 sample_size: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化神经网络增强的 RRT
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            env: 环境对象
            max_iterations: 最大迭代次数
            step_size: 步长
            sample_size: 每次采样的批量大小
            device: 计算设备
        """
        super().__init__(start, goal, env, max_iterations, step_size)
        
        self.sample_size = sample_size
        self.device = device
        
        # 初始化神经网络
        self.sampling_net = SamplingNetwork().to(device)
        self.evaluation_net = EvaluationNetwork().to(device)
        self.optimization_net = OptimizationNetwork().to(device)
        
    def _get_state_features(self, point: Tuple[float, float]) -> torch.Tensor:
        """获取状态特征"""
        x, y = point
        start_dist = np.sqrt((x - self.start[0])**2 + (y - self.start[1])**2)
        goal_dist = np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)
        
        return torch.tensor([x, y, start_dist, goal_dist], 
                          device=self.device)
    
    def _sample_batch(self) -> torch.Tensor:
        """使用神经网络进行批量采样"""
        # 获取当前节点状态
        current = self.nodes[-1]
        state = self._get_state_features((current.x, current.y))
        
        # 扩展为批量
        state = state.expand(self.sample_size, -1)
        
        # 采样新点
        with torch.no_grad():
            samples = self.sampling_net(state)
            
        # 将采样点映射到环境范围
        samples = samples * torch.tensor([self.env.width/2, self.env.height/2],
                                       device=self.device)
        
        return samples
    
    def _evaluate_path(self, path: List[Tuple[float, float]]) -> torch.Tensor:
        """评估路径质量"""
        # 转换为张量
        path_tensor = torch.tensor(path, device=self.device)
        path_tensor = path_tensor.unsqueeze(0)  # 添加批次维度
        
        # 评估路径
        with torch.no_grad():
            scores = self.evaluation_net(path_tensor)
            
        return scores[0]  # 返回单个路径的评估结果
    
    def _optimize_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """优化路径"""
        # 转换为张量
        path_tensor = torch.tensor(path, device=self.device)
        path_tensor = path_tensor.unsqueeze(0)  # 添加批次维度
        
        # 优化路径
        with torch.no_grad():
            optimized = self.optimization_net(path_tensor)
            
        # 转换回列表
        return optimized[0].cpu().numpy().tolist()
    
    def train_step(self, 
                  paths: List[List[Tuple[float, float]]], 
                  labels: List[float],
                  optimizer: torch.optim.Optimizer) -> float:
        """
        训练一个批次
        
        参数:
            paths: 路径列表
            labels: 标签列表（路径质量得分）
            optimizer: 优化器
            
        返回:
            损失值
        """
        # 转换为张量
        paths = torch.tensor(paths, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        
        # 评估路径
        scores = self.evaluation_net(paths)
        
        # 计算损失
        loss = nn.MSELoss()(scores[:, 1], labels)  # 使用长度得分
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """规划路径"""
        path = super().plan()
        
        if path:
            # 评估路径质量
            scores = self._evaluate_path(path)
            print(f"路径评估得分: 可行性={scores[0]:.2f}, " 
                  f"长度={scores[1]:.2f}, 平滑度={scores[2]:.2f}")
            
            # 优化路径
            optimized_path = self._optimize_path(path)
            
            # 验证优化后的路径
            if all(not self.env.check_collision(p) for p in optimized_path):
                return optimized_path
        
        return path


def collect_training_data(env: 'Environment',
                         num_episodes: int = 1000) -> Tuple[List, List]:
    """
    收集训练数据
    
    参数:
        env: 环境对象
        num_episodes: 收集的路径数量
        
    返回:
        paths: 路径列表
        scores: 得分列表
    """
    paths = []
    scores = []
    
    rrt = RRTStar(start=(0, 0), goal=(100, 100), env=env)
    
    for _ in range(num_episodes):
        # 随机生成起点和终点
        start = (np.random.uniform(0, env.width),
                np.random.uniform(0, env.height))
        goal = (np.random.uniform(0, env.width),
               np.random.uniform(0, env.height))
        
        rrt.start = start
        rrt.goal = goal
        
        # 规划路径
        path = rrt.plan()
        
        if path:
            # 计算路径得分（示例：使用路径长度的倒数作为得分）
            length = sum(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                        for p1, p2 in zip(path[:-1], path[1:]))
            score = 1.0 / (1.0 + length)
            
            paths.append(path)
            scores.append(score)
    
    return paths, scores


def train_neural_rrt(env: 'Environment',
                    num_epochs: int = 100,
                    batch_size: int = 32,
                    learning_rate: float = 1e-4):
    """
    训练神经网络增强的 RRT
    
    参数:
        env: 环境对象
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 创建神经网络 RRT
    neural_rrt = NeuralRRT(start=(0, 0), goal=(100, 100), env=env)
    
    # 收集训练数据
    print("收集训练数据...")
    paths, scores = collect_training_data(env)
    
    # 创建数据加载器
    dataset = list(zip(paths, scores))
    
    # 创建优化器
    optimizer = torch.optim.Adam(neural_rrt.parameters(), lr=learning_rate)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        # 打乱数据
        np.random.shuffle(dataset)
        
        # 批次训练
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_paths, batch_scores = zip(*batch)
            
            # 训练一个批次
            loss = neural_rrt.train_step(batch_paths, batch_scores, optimizer)
            
            total_loss += loss
            num_batches += 1
        
        # 打印训练信息
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("训练完成!")
    return neural_rrt


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
    samples = sampling_net.forward(torch.tensor([start[0]/width, start[1]/height, 0.0, 0.0], dtype=torch.float32))
    print(f"生成的采样点:\n{samples}")

    # 测试评估网络
    evaluation_net = EvaluationNetwork()
    print(f"\n评估网络结构:\n{evaluation_net}")

    # 测试评估
    path = [start, goal]
    scores = evaluation_net.forward(torch.tensor(path, dtype=torch.float32).unsqueeze(0))
    print(f"路径评估结果: 可行性={scores[0][0]:.2f}, 长度得分={scores[0][1]:.2f}, 平滑度得分={scores[0][2]:.2f}")

    # 测试优化网络
    optimization_net = OptimizationNetwork()
    print(f"\n优化网络结构:\n{optimization_net}")

    # 测试优化
    optimized_path = optimization_net.forward(torch.tensor(path, dtype=torch.float32).unsqueeze(0))
    print(f"优化后的路径点:\n{optimized_path}")

    # 测试神经网络增强的 RRT
    env = Environment(width, height)
    neural_rrt = NeuralRRT(start, goal, env)
    print(f"\n神经网络增强的 RRT 结构:\n{neural_rrt}")

    # 测试训练
    train_neural_rrt(env)
