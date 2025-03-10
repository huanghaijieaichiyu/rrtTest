#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于强化学习的路径规划器

实现一个使用强化学习进行路径规划的类，可以：
1. 将环境转换为RL代理的状态表示
2. 定义动作空间和奖励函数
3. 使用预训练的RL模型进行路径规划
4. 与其他路径规划算法保持兼容的接口
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# 导入基础RRT类用于保持接口一致
from .rrt_base import Node


@dataclass
class RLNode(Node):
    """强化学习路径规划节点"""
    value: float = 0.0  # 节点的价值估计


class PolicyNetwork(nn.Module):
    """强化学习策略网络"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化策略网络

        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（动作数量）
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNetwork(nn.Module):
    """强化学习价值网络"""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        初始化价值网络

        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RLPathPlanner:
    """
    基于强化学习的路径规划器

    使用预训练的强化学习模型进行路径规划。
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env,
        model_path: Optional[str] = None,
        resolution: float = 1.0,
        max_steps: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化强化学习路径规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象
            model_path: 预训练模型路径
            resolution: 网格分辨率
            max_steps: 最大步数
            device: 计算设备
        """
        self.start = RLNode(start[0], start[1])
        self.goal = RLNode(goal[0], goal[1])
        self.env = env
        self.resolution = resolution
        self.max_steps = max_steps
        self.device = device

        # 计算区域边界
        self.min_x = 0
        self.max_x = env.width
        self.min_y = 0
        self.max_y = env.height

        # 初始化路径
        self.start.path_x = [self.start.x]
        self.start.path_y = [self.start.y]

        # 动作空间（8个方向）
        self.actions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        # 加载模型
        self.policy_net = PolicyNetwork(
            input_dim=self._get_state_dim(),
            hidden_dim=128,
            output_dim=len(self.actions)
        )
        self.value_net = ValueNetwork(
            input_dim=self._get_state_dim(),
            hidden_dim=128
        )

        if model_path:
            self.load_model(model_path)

        # 搜索过程中的统计信息
        self.node_list = []

    def _get_state_dim(self) -> int:
        """
        获取状态空间维度

        返回:
            状态空间维度
        """
        # 基础状态：当前位置(2) + 目标位置(2) = 4
        # 加上局部环境感知（以当前位置为中心的局部网格，例如5x5=25）
        # 注意：实际应用中可能需要更复杂的状态表示
        return 4 + 25

    def _get_state(self, node: RLNode) -> torch.Tensor:
        """
        获取节点的状态表示

        参数:
            node: 当前节点

        返回:
            状态张量
        """
        # 基础状态：当前位置和目标位置
        state = [
            # 归一化当前位置
            node.x / self.max_x,
            node.y / self.max_y,
            # 归一化目标位置
            self.goal.x / self.max_x,
            self.goal.y / self.max_y
        ]

        # 添加局部环境感知（5x5网格）
        grid_size = 5
        grid_radius = grid_size // 2

        for i in range(-grid_radius, grid_radius + 1):
            for j in range(-grid_radius, grid_radius + 1):
                # 计算网格点坐标
                grid_x = node.x + i * self.resolution
                grid_y = node.y + j * self.resolution

                # 检查是否在环境范围内
                if not (self.min_x <= grid_x <= self.max_x and
                        self.min_y <= grid_y <= self.max_y):
                    # 如果超出范围，视为障碍物
                    state.append(1.0)
                else:
                    # 检查是否是障碍物
                    is_obstacle = self.env.check_collision((grid_x, grid_y))
                    state.append(1.0 if is_obstacle else 0.0)

        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def _is_goal(self, node: RLNode) -> bool:
        """
        检查是否到达目标

        参数:
            node: 当前节点

        返回:
            是否到达目标
        """
        dist = np.hypot(node.x - self.goal.x, node.y - self.goal.y)
        return dist <= self.resolution

    def load_model(self, model_path: str) -> None:
        """
        加载预训练模型

        参数:
            model_path: 模型路径
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用未训练的模型（随机策略）")

    def plan(self) -> List[Tuple[float, float]]:
        """
        执行路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        current = self.start
        self.node_list.append(current)

        for _ in range(self.max_steps):
            if self._is_goal(current):
                # 到达目标，返回路径
                return [(x, y) for x, y in zip(current.path_x, current.path_y)]

            # 获取当前状态
            state = self._get_state(current)

            # 使用策略网络预测动作概率
            with torch.no_grad():
                action_probs = self.policy_net(state.unsqueeze(0)).squeeze(0)

            # 选择动作（可以使用贪婪策略或采样）
            # 在实际应用时，可能需要进行探索
            action_idx = torch.argmax(action_probs).item()
            dx, dy = self.actions[action_idx]

            # 计算新位置
            new_x = current.x + dx * self.resolution
            new_y = current.y + dy * self.resolution

            # 检查是否在环境范围内
            if not (self.min_x <= new_x <= self.max_x and
                    self.min_y <= new_y <= self.max_y):
                # 超出边界，尝试其他动作
                action_probs[action_idx] = 0
                if torch.sum(action_probs) > 0:
                    action_probs = action_probs / torch.sum(action_probs)
                    action_idx = torch.argmax(action_probs).item()
                    dx, dy = self.actions[action_idx]
                    new_x = current.x + dx * self.resolution
                    new_y = current.y + dy * self.resolution
                else:
                    # 所有动作都不可行，返回当前路径
                    return [(x, y) for x, y in zip(current.path_x, current.path_y)]

            # 检查是否碰撞
            if self.env.check_collision((new_x, new_y)):
                # 碰撞，尝试其他动作
                action_probs[action_idx] = 0
                if torch.sum(action_probs) > 0:
                    action_probs = action_probs / torch.sum(action_probs)
                    action_idx = torch.argmax(action_probs).item()
                    dx, dy = self.actions[action_idx]
                    new_x = current.x + dx * self.resolution
                    new_y = current.y + dy * self.resolution

                    # 再次检查是否碰撞
                    if self.env.check_collision((new_x, new_y)):
                        # 仍然碰撞，返回当前路径
                        return [(x, y) for x, y in zip(current.path_x, current.path_y)]
                else:
                    # 所有动作都不可行，返回当前路径
                    return [(x, y) for x, y in zip(current.path_x, current.path_y)]

            # 创建新节点
            new_node = RLNode(new_x, new_y, parent=current)
            new_node.path_x = current.path_x.copy()
            new_node.path_y = current.path_y.copy()
            new_node.path_x.append(new_x)
            new_node.path_y.append(new_y)

            # 使用价值网络评估新节点
            with torch.no_grad():
                new_node.value = self.value_net(
                    self._get_state(new_node)).item()

            # 更新当前节点
            current = new_node
            self.node_list.append(current)

        # 达到最大步数，返回当前路径
        return [(x, y) for x, y in zip(current.path_x, current.path_y)]
