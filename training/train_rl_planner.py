#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
强化学习路径规划器训练脚本

实现基于PPO (Proximal Policy Optimization) 算法的路径规划器训练。
"""

from rrt.rl_planner import PolicyNetwork, ValueNetwork
from simulation.environment import Environment
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径，确保能够导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class PPOAgent:
    """基于PPO算法的强化学习代理"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化PPO代理

        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_ratio: PPO剪切比率
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度剪切阈值
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # 创建策略网络和价值网络
        self.policy_net = PolicyNetwork(
            state_dim, hidden_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        # 创建优化器
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

        # 训练统计信息
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': []
        }

    def get_action(self, state: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        """
        根据状态选择动作

        参数:
            state: 状态张量

        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        with torch.no_grad():
            state = state.to(self.device)
            action_probs = self.policy_net(state.unsqueeze(0)).squeeze(0)
            value = self.value_net(state.unsqueeze(0)).squeeze(0)

            # 创建分布
            dist = Categorical(action_probs)

            # 采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        更新策略和价值网络

        参数:
            states: 状态张量
            actions: 动作张量
            old_log_probs: 旧的动作对数概率
            returns: 回报张量
            advantages: 优势张量
            epochs: 每次更新的训练轮数
            batch_size: 批次大小

        返回:
            训练统计信息
        """
        # 计算数据集大小
        dataset_size = states.size(0)

        # 训练统计
        value_losses = []
        policy_losses = []
        entropy_losses = []

        # 打乱索引
        indices = np.arange(dataset_size)

        for _ in range(epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, batch_size):
                # 获取批次索引
                idx = indices[start_idx:start_idx + batch_size]

                # 提取批次数据
                mb_states = states[idx].to(self.device)
                mb_actions = actions[idx].to(self.device)
                mb_old_log_probs = old_log_probs[idx].to(self.device)
                mb_returns = returns[idx].to(self.device)
                mb_advantages = advantages[idx].to(self.device)

                # 计算当前策略的动作概率和状态价值
                mb_action_probs = self.policy_net(mb_states)
                mb_values = self.value_net(mb_states).squeeze(-1)

                # 创建分布
                dist = Categorical(mb_action_probs)

                # 计算动作的对数概率和熵
                mb_new_log_probs = dist.log_prob(mb_actions)
                mb_entropy = dist.entropy().mean()

                # 计算比率
                mb_ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)

                # 计算策略损失（PPO-Clip 目标函数）
                mb_obj = mb_ratio * mb_advantages
                mb_obj_clipped = torch.clamp(
                    mb_ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(mb_obj, mb_obj_clipped).mean()

                # 计算价值损失
                value_loss = ((mb_returns - mb_values) ** 2).mean()

                # 计算总损失
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * mb_entropy
                )

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度剪切
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) +
                    list(self.value_net.parameters()),
                    self.max_grad_norm
                )

                self.optimizer.step()

                # 记录损失
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(mb_entropy.item())

        return {
            'value_loss': np.mean(value_losses),
            'policy_loss': np.mean(policy_losses),
            'entropy_loss': np.mean(entropy_losses)
        }

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        """
        计算广义优势估计（GAE）

        参数:
            rewards: 奖励列表
            values: 价值列表
            dones: 终止状态列表
            next_value: 下一个状态的价值

        返回:
            returns: 回报列表
            advantages: 优势列表
        """
        returns = []
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * \
                next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        return returns, advantages

    def save(self, path: str) -> None:
        """
        保存模型

        参数:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats
        }, path)
        print(f"模型已保存到 {path}")

    def load(self, path: str) -> None:
        """
        加载模型

        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.stats = checkpoint['stats']
        print(f"模型已从 {path} 加载")


class PathPlanningEnv:
    """
    路径规划环境类

    为强化学习代理提供路径规划环境接口。
    """

    def __init__(
        self,
        env: Environment,
        resolution: float = 1.0,
        max_steps: int = 100,
        reward_goal: float = 100.0,
        reward_collision: float = -10.0,
        reward_step: float = -0.1,
        reward_closer: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        初始化路径规划环境

        参数:
            env: 环境对象
            resolution: 网格分辨率
            max_steps: 最大步数
            reward_goal: 到达目标的奖励
            reward_collision: 碰撞的惩罚
            reward_step: 每步的惩罚
            reward_closer: 接近目标的奖励
            seed: 随机种子
        """
        self.env = env
        self.resolution = resolution
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.reward_step = reward_step
        self.reward_closer = reward_closer

        # 计算环境边界
        self.min_x = 0
        self.max_x = env.width
        self.min_y = 0
        self.max_y = env.height

        # 动作空间（8个方向）
        self.actions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        # 当前状态
        self.current_pos = None
        self.goal_pos = None
        self.steps = 0
        self.prev_dist = None

        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def reset(
        self,
        start: Optional[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        重置环境

        参数:
            start: 起点坐标
            goal: 终点坐标

        返回:
            状态表示
        """
        # 随机生成起点和终点（如果未指定）
        if start is None:
            while True:
                x = random.uniform(self.min_x, self.max_x)
                y = random.uniform(self.min_y, self.max_y)
                if not self.env.check_collision((x, y)):
                    start = (x, y)
                    break

        if goal is None:
            while True:
                x = random.uniform(self.min_x, self.max_x)
                y = random.uniform(self.min_y, self.max_y)
                # 确保目标点不是障碍物，且与起点有一定距离
                if (not self.env.check_collision((x, y)) and
                        np.hypot(x - start[0], y - start[1]) > 20.0):
                    goal = (x, y)
                    break

        self.current_pos = start
        self.goal_pos = goal
        self.steps = 0
        self.prev_dist = np.hypot(
            self.goal_pos[0] - self.current_pos[0],
            self.goal_pos[1] - self.current_pos[1]
        )

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步

        参数:
            action: 动作索引

        返回:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.steps += 1

        # 获取动作方向
        dx, dy = self.actions[action]

        # 计算新位置
        new_x = self.current_pos[0] + dx * self.resolution
        new_y = self.current_pos[1] + dy * self.resolution

        # 检查是否在环境边界内
        if not (self.min_x <= new_x <= self.max_x and
                self.min_y <= new_y <= self.max_y):
            # 超出边界，保持原位置并给予惩罚
            reward = self.reward_collision
            done = False
            info = {'status': 'out_of_bounds'}
        else:
            # 检查是否碰撞
            if self.env.check_collision((new_x, new_y)):
                # 碰撞，保持原位置并给予惩罚
                reward = self.reward_collision
                done = False
                info = {'status': 'collision'}
            else:
                # 更新位置
                self.current_pos = (new_x, new_y)

                # 计算到目标的距离
                dist = np.hypot(
                    self.goal_pos[0] - self.current_pos[0],
                    self.goal_pos[1] - self.current_pos[1]
                )

                # 检查是否到达目标
                if dist <= self.resolution:
                    reward = self.reward_goal
                    done = True
                    info = {'status': 'goal_reached'}
                else:
                    # 计算基本步骤惩罚
                    reward = self.reward_step

                    # 添加接近目标的奖励
                    if dist < self.prev_dist:
                        reward += self.reward_closer

                    self.prev_dist = dist

                    # 检查是否达到最大步数
                    if self.steps >= self.max_steps:
                        done = True
                        info = {'status': 'max_steps_reached'}
                    else:
                        done = False
                        info = {'status': 'in_progress'}

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """
        获取状态表示

        返回:
            状态数组
        """
        # 基础状态：当前位置和目标位置（归一化）
        state = [
            self.current_pos[0] / self.max_x,
            self.current_pos[1] / self.max_y,
            self.goal_pos[0] / self.max_x,
            self.goal_pos[1] / self.max_y
        ]

        # 添加局部环境感知（5x5网格）
        grid_size = 5
        grid_radius = grid_size // 2

        for i in range(-grid_radius, grid_radius + 1):
            for j in range(-grid_radius, grid_radius + 1):
                # 计算网格点坐标
                grid_x = self.current_pos[0] + i * self.resolution
                grid_y = self.current_pos[1] + j * self.resolution

                # 检查是否在环境范围内
                if not (self.min_x <= grid_x <= self.max_x and
                        self.min_y <= grid_y <= self.max_y):
                    # 如果超出范围，视为障碍物
                    state.append(1.0)
                else:
                    # 检查是否是障碍物
                    is_obstacle = self.env.check_collision((grid_x, grid_y))
                    state.append(1.0 if is_obstacle else 0.0)

        return np.array(state, dtype=np.float32)


def create_training_env(env_type='complex', seed=42):
    """
    创建训练环境

    参数:
        env_type: 环境类型 ('simple' 或 'complex')
        seed: 随机种子

    返回:
        环境对象
    """
    from examples.pygame_simulation_example import create_complex_environment, create_simple_environment

    if env_type == 'complex':
        env = create_complex_environment(seed=seed)
    else:
        env = create_simple_environment()

    return env


def train(
    env_type: str = 'complex',
    num_episodes: int = 10000,
    max_steps_per_episode: int = 100,
    save_interval: int = 500,
    eval_interval: int = 100,
    save_path: str = 'models/rl_planner',
    seed: Optional[int] = 42,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    训练强化学习路径规划器

    参数:
        env_type: 环境类型 ('simple' 或 'complex')
        num_episodes: 训练的回合数
        max_steps_per_episode: 每个回合的最大步数
        save_interval: 保存模型的间隔回合数
        eval_interval: 评估模型的间隔回合数
        save_path: 模型保存路径
        seed: 随机种子
        device: 计算设备
    """
    print(f"Training RL Path Planner - Device: {device}")

    # 创建环境
    env_obj = create_training_env(env_type=env_type, seed=seed)
    env = PathPlanningEnv(
        env=env_obj,
        resolution=1.0,
        max_steps=max_steps_per_episode,
        reward_goal=100.0,
        reward_collision=-10.0,
        reward_step=-0.1,
        reward_closer=0.2,
        seed=seed
    )

    # 创建代理
    state_dim = 4 + 25  # 状态维度（当前位置+目标位置+5x5局部感知）
    action_dim = 8      # 动作维度（8个方向）

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device=device
    )

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 跟踪训练进度
    rewards_history = []
    success_history = []

    # 训练循环
    for episode in tqdm(range(1, num_episodes + 1)):
        # 重置环境
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # 存储经验
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        # 记录当前回合的奖励和步数
        episode_reward = 0
        success = False

        # 一个回合
        for _ in range(max_steps_per_episode):
            # 选择动作
            action, log_prob, value = agent.get_action(state_tensor)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 记录
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            # 更新状态
            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # 累积奖励
            episode_reward += reward

            # 检查是否成功
            if info['status'] == 'goal_reached':
                success = True

            # 如果回合结束，跳出循环
            if done:
                break

        # 记录回合结果
        rewards_history.append(episode_reward)
        success_history.append(1 if success else 0)

        # 计算最后一个状态的价值
        if done:
            next_value = 0
        else:
            with torch.no_grad():
                next_value = agent.value_net(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(
                        0).to(device)
                ).item()

        # 计算GAE
        returns, advantages = agent.compute_gae(
            rewards, values, dones, next_value)

        # 将列表转换为张量
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.long)
        old_log_probs_tensor = torch.tensor(
            np.array(log_probs), dtype=torch.float32)
        returns_tensor = torch.tensor(np.array(returns), dtype=torch.float32)
        advantages_tensor = torch.tensor(
            np.array(advantages), dtype=torch.float32)

        # 标准化优势
        advantages_tensor = (
            advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # 更新策略和价值网络
        update_stats = agent.update(
            states=states_tensor,
            actions=actions_tensor,
            old_log_probs=old_log_probs_tensor,
            returns=returns_tensor,
            advantages=advantages_tensor,
            epochs=10,
            batch_size=min(64, len(states))
        )

        # 更新训练统计信息
        agent.stats['episode_rewards'].append(episode_reward)
        agent.stats['episode_lengths'].append(len(rewards))
        agent.stats['value_losses'].append(update_stats['value_loss'])
        agent.stats['policy_losses'].append(update_stats['policy_loss'])
        agent.stats['entropy_losses'].append(update_stats['entropy_loss'])

        # 定期保存模型
        if episode % save_interval == 0:
            agent.save(f"{save_path}_episode_{episode}.pt")

            # 绘制训练曲线
            plot_training_curves(agent.stats, save_path)

        # 定期评估
        if episode % eval_interval == 0:
            success_rate = np.mean(success_history[-eval_interval:])
            avg_reward = np.mean(rewards_history[-eval_interval:])
            print(
                f"Episode {episode}: Success Rate = {success_rate:.2f}, Avg Reward = {avg_reward:.2f}")

    # 保存最终模型
    agent.save(f"{save_path}_final.pt")
    print("训练完成！")

    # 绘制最终训练曲线
    plot_training_curves(agent.stats, save_path)


def plot_training_curves(stats, save_path):
    """
    绘制训练曲线

    参数:
        stats: 训练统计信息
        save_path: 保存路径
    """
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 奖励曲线
    smoothed_rewards = np.convolve(
        stats['episode_rewards'], np.ones(100)/100, mode='valid'
    ) if len(stats['episode_rewards']) > 100 else stats['episode_rewards']
    axs[0, 0].plot(smoothed_rewards)
    axs[0, 0].set_title('Smoothed Episode Rewards')
    axs[0, 0].set_xlabel('Episodes')
    axs[0, 0].set_ylabel('Reward')

    # 回合长度曲线
    smoothed_lengths = np.convolve(
        stats['episode_lengths'], np.ones(100)/100, mode='valid'
    ) if len(stats['episode_lengths']) > 100 else stats['episode_lengths']
    axs[0, 1].plot(smoothed_lengths)
    axs[0, 1].set_title('Smoothed Episode Lengths')
    axs[0, 1].set_xlabel('Episodes')
    axs[0, 1].set_ylabel('Length')

    # 策略损失曲线
    axs[1, 0].plot(stats['policy_losses'])
    axs[1, 0].set_title('Policy Loss')
    axs[1, 0].set_xlabel('Updates')
    axs[1, 0].set_ylabel('Loss')

    # 价值损失曲线
    axs[1, 1].plot(stats['value_losses'])
    axs[1, 1].set_title('Value Loss')
    axs[1, 1].set_xlabel('Updates')
    axs[1, 1].set_ylabel('Loss')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}_training_curves.png")
    plt.close()


def evaluate(
    model_path: str,
    env_type: str = 'complex',
    num_episodes: int = 100,
    render: bool = False,
    seed: Optional[int] = 42,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    评估训练好的模型

    参数:
        model_path: 模型路径
        env_type: 环境类型
        num_episodes: 评估的回合数
        render: 是否渲染
        seed: 随机种子
        device: 计算设备
    """
    print(f"Evaluating RL Path Planner - Model: {model_path}")

    # 创建环境
    env_obj = create_training_env(env_type=env_type, seed=seed)
    env = PathPlanningEnv(
        env=env_obj,
        resolution=1.0,
        max_steps=100,
        reward_goal=100.0,
        reward_collision=-10.0,
        reward_step=-0.1,
        reward_closer=0.2,
        seed=seed
    )

    # 创建代理
    state_dim = 4 + 25  # 状态维度（当前位置+目标位置+5x5局部感知）
    action_dim = 8      # 动作维度（8个方向）

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        device=device
    )

    # 加载模型
    agent.load(model_path)

    # 评估指标
    success_count = 0
    rewards = []
    path_lengths = []

    # 评估循环
    for episode in tqdm(range(1, num_episodes + 1)):
        # 重置环境
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # 记录当前回合的奖励和步数
        episode_reward = 0
        steps = 0
        success = False

        # 一个回合
        while True:
            # 选择动作
            action, _, _ = agent.get_action(state_tensor)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 更新状态
            state = next_state
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # 累积奖励
            episode_reward += reward
            steps += 1

            # 检查是否成功
            if info['status'] == 'goal_reached':
                success = True

            # 如果回合结束，跳出循环
            if done:
                break

        # 记录结果
        rewards.append(episode_reward)
        path_lengths.append(steps)
        if success:
            success_count += 1

    # 计算统计信息
    success_rate = success_count / num_episodes
    avg_reward = np.mean(rewards)
    avg_path_length = np.mean(path_lengths)

    print(f"成功率: {success_rate:.2f}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均路径长度: {avg_path_length:.2f}")

    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_path_length': avg_path_length
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='强化学习路径规划器训练脚本')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate'],
                        help='模式：train（训练）或 evaluate（评估）')

    parser.add_argument('--env-type', type=str, default='complex',
                        choices=['simple', 'complex'],
                        help='环境类型')

    parser.add_argument('--episodes', type=int, default=10000,
                        help='训练回合数')

    parser.add_argument('--steps', type=int, default=100,
                        help='每个回合的最大步数')

    parser.add_argument('--save-interval', type=int, default=500,
                        help='保存模型的间隔回合数')

    parser.add_argument('--eval-interval', type=int, default=100,
                        help='评估的间隔回合数')

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
    # 解析命令行参数
    args = parse_args()

    # 设置设备
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    if args.mode == 'train':
        # 训练模型
        train(
            env_type=args.env_type,
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            save_path=args.save_path,
            seed=args.seed,
            device=device
        )
    else:
        # 评估模型
        if args.model_path is None:
            print("请提供要评估的模型路径")
            return

        evaluate(
            model_path=args.model_path,
            env_type=args.env_type,
            num_episodes=100,
            render=False,
            seed=args.seed,
            device=device
        )


if __name__ == "__main__":
    main()
