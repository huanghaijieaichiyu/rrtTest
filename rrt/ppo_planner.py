#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于PPO的路径规划器

使用近端策略优化（Proximal Policy Optimization）算法进行路径规划。
PPO是一种先进的强化学习算法，它通过限制策略更新的幅度来保证训练的稳定性。
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dataclasses import dataclass

from rrt.rrt_base import Node


@dataclass
class PPONode(Node):
    """PPO路径规划节点"""
    value: float = 0.0  # 节点的价值估计
    log_prob: float = 0.0  # 动作的对数概率

    def __init__(self, x: float, y: float):
        """初始化PPO节点"""
        super().__init__(x, y)
        self.value = 0.0
        self.log_prob = 0.0


class PPONetwork(nn.Module):
    """PPO网络，包含策略网络和价值网络"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        std: float = 0.1
    ):
        """
        初始化PPO网络

        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            std: 动作分布的标准差
        """
        super(PPONetwork, self).__init__()

        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 将输出限制在[-1, 1]范围内
        )

        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 动作分布的标准差（使用log_std以确保std始终为正）
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(std))

    def forward(self, state):
        """前向传播"""
        mean = self.actor(state)
        value = self.critic(state)
        return mean, value

    def get_action(self, state, deterministic: bool = False):
        """
        获取动作

        参数:
            state: 当前状态
            deterministic: 是否使用确定性策略

        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        mean, value = self(state)

        if deterministic:
            return mean, None, value

        # 使用softplus确保标准差为正
        action_std = F.softplus(self.log_std)

        # 创建正态分布
        dist = Normal(mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


class PPOPathPlanner:
    """
    基于PPO的路径规划器
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
        初始化PPO路径规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象
            model_path: 预训练模型路径
            resolution: 网格分辨率
            max_steps: 最大步数
            device: 计算设备
        """
        self.start = PPONode(start[0], start[1])
        self.goal = PPONode(goal[0], goal[1])
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

        # 初始化PPO网络
        state_dim = self._get_state_dim()
        action_dim = 2  # x和y方向的移动
        self.network = PPONetwork(state_dim, action_dim).to(device)

        if model_path:
            self.load_model(model_path)

        # 记录搜索统计信息
        self.node_list = []

        # PPO超参数
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def _get_state_dim(self) -> int:
        """获取状态空间维度"""
        # 基础状态：当前位置(2) + 目标位置(2) = 4
        # 加上局部环境感知（以当前位置为中心的局部网格，例如5x5=25）
        return 4 + 25

    def _get_state(self, node: PPONode) -> torch.Tensor:
        """获取状态表示"""
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

                # 检查是否在环境范围内和是否有障碍物
                if not (self.min_x <= grid_x <= self.max_x and
                        self.min_y <= grid_y <= self.max_y):
                    state.append(1.0)  # 超出范围视为障碍物
                else:
                    is_obstacle = self.env.check_collision((grid_x, grid_y))
                    state.append(1.0 if is_obstacle else 0.0)

        return torch.tensor(state, dtype=torch.float32).to(self.device)

    def _is_goal(self, node: PPONode) -> bool:
        """检查是否到达目标"""
        dist = np.hypot(node.x - self.goal.x, node.y - self.goal.y)
        return dist <= self.resolution

    def _is_valid_position(self, x: float, y: float) -> bool:
        """检查位置是否有效"""
        if not (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y):
            return False
        return not self.env.check_collision((x, y))

    def load_model(self, model_path: str) -> None:
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.network.load_state_dict(checkpoint['network'])
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
                return [(x, y) for x, y in zip(current.path_x, current.path_y)]

            # 获取当前状态
            state = self._get_state(current)

            # 使用PPO网络预测动作
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(
                    state.unsqueeze(0),
                    deterministic=True
                )
                action = action.squeeze(0).cpu().numpy()

            # 将动作转换为位置增量
            dx = action[0] * self.resolution
            dy = action[1] * self.resolution

            # 计算新位置
            new_x = current.x + dx
            new_y = current.y + dy

            # 检查新位置是否有效
            if not self._is_valid_position(new_x, new_y):
                continue

            # 创建新节点
            new_node = PPONode(new_x, new_y)
            new_node.value = value.item()
            new_node.log_prob = (log_prob.item() if log_prob is not None
                                 else 0.0)
            new_node.parent = current
            new_node.path_x = current.path_x + [new_x]
            new_node.path_y = current.path_y + [new_y]

            self.node_list.append(new_node)
            current = new_node

        # 如果达到最大步数仍未找到路径，返回空列表
        return []

    def train(
        self,
        num_episodes: int,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        save_path: Optional[str] = None
    ):
        """
        训练PPO网络

        参数:
            num_episodes: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            save_path: 模型保存路径
        """
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate)

        for episode in range(num_episodes):
            # 收集轨迹
            states = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            masks = []

            current = self.start
            done = False

            while not done:
                state = self._get_state(current)
                states.append(state)

                # 获取动作
                with torch.no_grad():
                    action, log_prob, value = self.network.get_action(
                        state.unsqueeze(0)
                    )
                action = action.squeeze(0)

                # 执行动作
                dx = action[0].item() * self.resolution
                dy = action[1].item() * self.resolution
                new_x = current.x + dx
                new_y = current.y + dy

                # 检查是否有效
                if not self._is_valid_position(new_x, new_y):
                    reward = -1.0
                    done = True
                else:
                    new_node = PPONode(new_x, new_y)
                    dist_to_goal = np.hypot(
                        new_x - self.goal.x,
                        new_y - self.goal.y
                    )
                    reward = -dist_to_goal / 100.0

                    if self._is_goal(new_node):
                        reward += 10.0
                        done = True

                    current = new_node

                # 保存轨迹
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(1.0 - float(done))

            # 计算优势函数
            returns = []
            advantages = []
            R = 0.0
            for r, v, mask in zip(reversed(rewards),
                                  reversed(values),
                                  reversed(masks)):
                R = r + 0.99 * R * mask
                advantage = R - v.item()
                returns.append(R)
                advantages.append(advantage)

            returns = torch.tensor(list(reversed(returns)),
                                   dtype=torch.float32).to(self.device)
            advantages = torch.tensor(list(reversed(advantages)),
                                      dtype=torch.float32).to(self.device)
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8)

            # 将轨迹转换为张量
            states = torch.stack(states)
            actions = torch.stack(actions)
            old_log_probs = torch.cat(log_probs).detach()

            # PPO更新
            for _ in range(10):
                # 计算新的动作分布
                means, values = self.network(states)
                action_std = F.softplus(self.network.log_std)
                dist = Normal(means, action_std)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)

                # 计算比率
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_ratio,
                                    1.0 + self.clip_ratio) * advantages

                # 计算损失
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), returns)
                entropy = dist.entropy().mean()

                loss = (policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy)

                # 更新网络
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            # 打印训练信息
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Average Reward: {sum(rewards)/len(rewards):.3f}")

        # 保存模型
        if save_path:
            torch.save({
                'network': self.network.state_dict()
            }, save_path)
            print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    from simulation.scenario_generator import ScenarioGenerator

    # 创建场景生成器
    generator = ScenarioGenerator(
        width=100.0,
        height=100.0,
        min_obstacle_size=2.0,
        max_obstacle_size=10.0,
        min_gap=5.0
    )

    # 创建训练环境列表
    training_envs = []

    # 添加不同类型的训练场景
    print("生成训练场景...")

    # 随机障碍物场景
    for _ in range(3):
        env = generator.generate_random_scenario(
            num_obstacles=np.random.randint(10, 30),
            density=np.random.uniform(0.2, 0.4)
        )
        training_envs.append(env)

    # 迷宫型场景
    for _ in range(2):
        env = generator.generate_maze_scenario(
            cell_size=np.random.uniform(8.0, 15.0),
            complexity=np.random.uniform(0.3, 0.7)
        )
        training_envs.append(env)

    # 房间型场景
    for _ in range(2):
        env = generator.generate_room_scenario(
            num_rooms=np.random.randint(3, 6),
            min_room_size=15.0,
            max_room_size=30.0
        )
        training_envs.append(env)

    # 走廊型场景
    for _ in range(2):
        env = generator.generate_corridor_scenario(
            corridor_width=np.random.uniform(8.0, 12.0),
            num_turns=np.random.randint(2, 5)
        )
        training_envs.append(env)

    # 混合型场景
    for _ in range(3):
        env = generator.generate_mixed_scenario(
            num_random_obstacles=np.random.randint(5, 15),
            num_rooms=np.random.randint(1, 3),
            corridor_width=np.random.uniform(8.0, 12.0)
        )
        training_envs.append(env)

    print(f"共生成 {len(training_envs)} 个训练场景")

    # 创建并训练PPO规划器
    planner = PPOPathPlanner(
        start=(10, 10),
        goal=(90, 90),
        env=training_envs[0],  # 初始使用第一个环境
        resolution=1.0,
        max_steps=1000
    )

    print("\n开始训练PPO模型...")

    # 训练模型
    num_episodes = 2000  # 增加训练轮数
    episodes_per_env = num_episodes // len(training_envs)

    for i, env in enumerate(training_envs):
        print(f"\n使用场景 {i+1}/{len(training_envs)} 进行训练...")

        # 更新环境
        planner.env = env

        # 在当前环境上训练
        planner.train(
            num_episodes=episodes_per_env,
            batch_size=64,
            learning_rate=3e-4,
            save_path=f"models/ppo_model_env_{i+1}.pth"
        )

    # 保存最终模型
    torch.save({
        'network': planner.network.state_dict()
    }, "models/ppo_model_final.pth")

    print("\n训练完成，最终模型已保存到 models/ppo_model_final.pth")

    # 在每种类型的场景上测试模型
    print("\n在不同类型的场景上测试模型...")

    for i, env in enumerate(training_envs):
        print(f"\n测试场景 {i+1}:")
        planner.env = env
        path = planner.plan()

        if path:
            print(f"找到路径，长度为: {len(path)}个点")
        else:
            print("未找到有效路径")
