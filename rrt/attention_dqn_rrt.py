import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple, Optional
from rrt.rrt_star import RRTStar


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性变换
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

        # 应用注意力
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, attention


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.conv2 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, d_model: int, num_heads: int):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # 自注意力
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)

        # 前馈网络
        ff = self.feed_forward(x)
        x = self.norm2(x + ff)
        return x


class PredictionNetwork(nn.Module):
    """预测网络，用于预测障碍物运动和路径可行性"""

    def __init__(self, input_dim: int, hidden_dim: int, sequence_length: int):
        super(PredictionNetwork, self).__init__()
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # 只使用最后一个时间步的输出
        return predictions


class EnhancedDQNetwork(nn.Module):
    """增强的深度Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(EnhancedDQNetwork, self).__init__()

        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )

        # Transformer编码器
        self.transformer = nn.Sequential(
            TransformerBlock(hidden_dim, num_heads=8),
            TransformerBlock(hidden_dim, num_heads=8)
        )

        # 动作价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 优势头
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        # 特征提取
        features = self.feature_net(state)

        # Transformer处理
        features = features.unsqueeze(1)  # 添加序列维度
        features = self.transformer(features)
        features = features.squeeze(1)  # 移除序列维度

        # 计算价值和优势
        value = self.value_head(features)
        advantage = self.advantage_head(features)

        # 组合价值和优势得到Q值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, features


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class AttentionDQNRRT(RRTStar):
    """增强的注意力DQN-RRT算法"""

    def __init__(self, start, goal, env, vehicle_width, vehicle_length,
                 step_size=2.0, max_iterations=1000, rewire_factor=1.5,
                 learning_rate=0.001, gamma=0.99, epsilon=0.1,
                 buffer_capacity=10000, batch_size=64, hidden_dim=256,
                 prediction_horizon=5):
        """
        使用深度强化学习和注意力机制增强的RRT*算法

        参数:
            start: 起点
            goal: 目标点
            env: 环境对象
            vehicle_width: 车辆宽度
            vehicle_length: 车辆长度
            step_size: 每次扩展的步长
            max_iterations: 最大迭代次数
            rewire_factor: 重连接因子
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon: 探索率
            buffer_capacity: 经验回放缓冲区容量
            batch_size: 批量大小
            hidden_dim: 隐藏层维度
            prediction_horizon: 预测时间步长
        """
        # 确保max_iterations是整数
        max_iterations = int(max_iterations)

        # 调用父类初始化
        super().__init__(
            start=start,
            goal=goal,
            env=env,
            step_size=step_size,
            max_iterations=max_iterations,
            vehicle_width=vehicle_width,
            vehicle_length=vehicle_length,
            rewire_factor=rewire_factor
        )

        # 扩展状态维度
        self.state_dim = 16  # 增加状态维度
        self.action_dim = 16  # 增加动作维度以提供更细粒度的控制
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon

        # 网络和优化器
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = EnhancedDQNetwork(
            self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network = EnhancedDQNetwork(
            self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 预测网络
        self.prediction_net = PredictionNetwork(
            input_dim=4,  # 位置和速度
            hidden_dim=64,
            sequence_length=prediction_horizon
        ).to(self.device)

        # 优化器
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=learning_rate)
        self.pred_optimizer = torch.optim.Adam(
            self.prediction_net.parameters(), lr=learning_rate)

        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.prediction_buffer = deque(maxlen=prediction_horizon)

        # 训练参数
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # 历史记录
        self.episode_rewards = []
        self.prediction_errors = []
        self.attention_maps = []

        # 路径历史
        self.path_history = []
        self.obstacle_history = []

    def get_state(self, node: tuple) -> np.ndarray:
        """获取增强的状态表示"""
        x, y = node
        goal_x, goal_y = self.goal

        # 获取最近的k个障碍物
        k = 3
        nearest_obstacles = self.get_k_nearest_obstacles(node, k)

        # 计算到目标的距离和角度
        dist_to_goal = np.hypot(goal_x - x, goal_y - y)
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)

        # 计算路径历史特征
        path_features = self.compute_path_features(node)

        # 构建状态向量
        state = np.array([
            x, y,                          # 当前位置
            goal_x, goal_y,                # 目标位置
            dist_to_goal, angle_to_goal,   # 目标相关特征
            self.vehicle_width, self.vehicle_length,  # 车辆尺寸
            *path_features,                # 路径历史特征
            *(obs[0] for obs in nearest_obstacles),  # 最近障碍物x坐标
            *(obs[1] for obs in nearest_obstacles),  # 最近障碍物y坐标
        ])
        return state

    def get_k_nearest_obstacles(self, node: tuple, k: int) -> List[tuple]:
        """获取k个最近的障碍物"""
        obstacles = []
        for obs in self.env.obstacles:
            dist = np.hypot(node[0] - obs.x, node[1] - obs.y)
            obstacles.append(((obs.x, obs.y), dist))

        # 按距离排序并返回前k个
        obstacles.sort(key=lambda x: x[1])
        return [obs[0] for obs in obstacles[:k]]

    def compute_path_features(self, node: tuple) -> np.ndarray:
        """计算路径相关特征"""
        if not self.path_history:
            return np.zeros(4)

        # 计算与历史路径的关系
        min_dist = float('inf')
        min_angle = 0

        for path_point in self.path_history[-10:]:  # 只使用最近的10个点
            dist = np.hypot(node[0] - path_point[0], node[1] - path_point[1])
            if dist < min_dist:
                min_dist = dist
                min_angle = np.arctan2(
                    path_point[1] - node[1], path_point[0] - node[0])

        # 计算路径曲率
        if len(self.path_history) >= 3:
            p1, p2, p3 = self.path_history[-3:]
            curvature = self.compute_curvature(p1, p2, p3)
        else:
            curvature = 0

        return np.array([min_dist, min_angle, curvature, len(self.path_history)])

    def compute_curvature(self, p1: tuple, p2: tuple, p3: tuple) -> float:
        """计算三点曲率"""
        try:
            # 使用外接圆半径的倒数作为曲率
            a = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            b = np.hypot(p3[0] - p2[0], p3[1] - p2[1])
            c = np.hypot(p3[0] - p1[0], p3[1] - p1[1])

            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))

            if area > 0:
                return 4 * area / (a * b * c)
            return 0
        except:
            return 0

    def predict_obstacle_motion(self, obstacle_history: List[tuple]) -> np.ndarray:
        """预测障碍物运动"""
        if len(obstacle_history) < self.prediction_horizon:
            return np.zeros(2)  # 返回静态位置

        # 准备序列数据
        sequence = torch.FloatTensor(
            obstacle_history[-self.prediction_horizon:]).unsqueeze(0)

        # 预测下一个位置
        with torch.no_grad():
            prediction = self.prediction_net(sequence)

        return prediction.cpu().numpy()[0]

    def get_reward(self, new_node: tuple, collision: bool) -> float:
        """增强的奖励计算"""
        if collision:
            return -100.0

        # 基础奖励
        dist_to_goal = np.hypot(
            new_node[0] - self.goal[0], new_node[1] - self.goal[1])
        dist_reward = -dist_to_goal * 0.1

        # 安全距离奖励
        safety_reward = 0
        for obs in self.get_k_nearest_obstacles(new_node, 3):
            dist = np.hypot(new_node[0] - obs[0], new_node[1] - obs[1])
            safety_reward += min(dist * 0.2, 10.0)

        # 路径平滑度奖励
        smoothness_reward = 0
        if len(self.path_history) >= 3:
            curvature = self.compute_curvature(
                self.path_history[-2],
                self.path_history[-1],
                new_node
            )
            smoothness_reward = -curvature * 5.0  # 惩罚高曲率

        # 预测奖励
        prediction_reward = 0
        for obs_history in self.obstacle_history:
            if len(obs_history) >= self.prediction_horizon:
                predicted_pos = self.predict_obstacle_motion(obs_history)
                future_dist = np.hypot(
                    new_node[0] - predicted_pos[0],
                    new_node[1] - predicted_pos[1]
                )
                prediction_reward += min(future_dist * 0.1, 5.0)

        total_reward = dist_reward + safety_reward + \
            smoothness_reward + prediction_reward
        return total_reward

    def update_prediction_network(self):
        """更新预测网络"""
        if len(self.obstacle_history) < self.prediction_horizon:
            return

        # 准备训练数据
        sequences = []
        targets = []

        for obs_history in self.obstacle_history:
            if len(obs_history) > self.prediction_horizon:
                for i in range(len(obs_history) - self.prediction_horizon):
                    seq = obs_history[i:i+self.prediction_horizon]
                    target = obs_history[i+self.prediction_horizon]
                    sequences.append(seq)
                    targets.append(target)

        if not sequences:
            return

        # 转换为tensor
        sequences = torch.FloatTensor(sequences).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)

        # 训练预测网络
        predictions = self.prediction_net(sequences)
        pred_loss = F.mse_loss(predictions, targets)

        self.pred_optimizer.zero_grad()
        pred_loss.backward()
        self.pred_optimizer.step()

        self.prediction_errors.append(pred_loss.item())

    def extend(self, nearest_node: tuple) -> Optional[tuple]:
        """增强的扩展方法"""
        state = self.get_state(nearest_node)
        action = self.select_action(state)

        # 将离散动作转换为连续方向和步长
        angle = 2 * np.pi * (action % 8) / 8  # 8个方向
        step_scale = 0.5 + (action // 8) * 0.5  # 2个步长选项

        dx = self.step_size * step_scale * np.cos(angle)
        dy = self.step_size * step_scale * np.sin(angle)

        new_node = (nearest_node[0] + dx, nearest_node[1] + dy)

        # 检查碰撞
        collision = self.check_collision(nearest_node, new_node)

        # 更新历史记录
        if not collision:
            self.path_history.append(new_node)

        # 更新障碍物历史
        current_obstacles = self.get_k_nearest_obstacles(new_node, 3)
        for i, obs in enumerate(current_obstacles):
            if i >= len(self.obstacle_history):
                self.obstacle_history.append([obs])
            else:
                self.obstacle_history[i].append(obs)

        # 计算奖励并存储经验
        reward = self.get_reward(new_node, collision)
        next_state = self.get_state(new_node)
        done = collision or self.is_goal_reached(new_node)

        self.replay_buffer.push(state, action, reward, next_state, done)

        # 更新网络
        self.update_network()
        self.update_prediction_network()

        return new_node if not collision else None

    def update_network(self):
        """更新Q网络"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # 采样经验数据
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        current_q_values, features = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + \
                (1 - dones) * self.gamma * max_next_q_values

        # 计算TD误差
        td_error = (target_q_values - current_q_values.squeeze()).abs()

        # 使用Huber损失
        loss = F.huber_loss(current_q_values.squeeze(), target_q_values)

        # 添加L2正则化
        l2_lambda = 0.01
        l2_norm = sum(p.pow(2.0).sum() for p in self.q_network.parameters())
        loss = loss + l2_lambda * l2_norm

        self.q_optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        self.q_optimizer.step()

        # 软更新目标网络
        tau = 0.001
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

        self.episode_rewards.append(rewards[0])  # 记录第一个样本的奖励

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'prediction_network_state_dict': self.prediction_net.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'pred_optimizer_state_dict': self.pred_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'prediction_errors': self.prediction_errors,
            'attention_maps': self.attention_maps
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(
            checkpoint['target_network_state_dict'])
        self.prediction_net.load_state_dict(
            checkpoint['prediction_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.pred_optimizer.load_state_dict(
            checkpoint['pred_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.prediction_errors = checkpoint['prediction_errors']
        self.attention_maps = checkpoint['attention_maps']
