import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Optional
from rrt.rrt_star import RRTStar
from rrt.rrt_base import Node


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
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

        # 应用注意力
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output, attention


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.conv2 = nn.Linear(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
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
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model))

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

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
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

        # 状态编码器
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                                           nn.Dropout(0.1))

        # 特征提取网络
        self.feature_net = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim), nn.LayerNorm(hidden_dim))

        # Transformer编码器
        self.transformer = nn.Sequential(TransformerBlock(hidden_dim, num_heads=8),
                                         TransformerBlock(hidden_dim, num_heads=8), nn.LayerNorm(hidden_dim))

        # 动作价值头
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                                        nn.Linear(hidden_dim, 1))

        # 优势头
        self.advantage_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
                                            nn.Linear(hidden_dim, action_dim))

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state):
        # 状态编码
        encoded_state = self.state_encoder(state)

        # 特征提取
        features = self.feature_net(encoded_state)

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

    def __init__(self,
                 start,
                 goal,
                 env,
                 vehicle_width,
                 vehicle_length,
                 step_size=2.0,
                 max_iterations=1000,
                 rewire_factor=1.5,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon=0.1,
                 buffer_capacity=10000,
                 batch_size=64,
                 hidden_dim=256,
                 prediction_horizon=5):
        """
        使用深度强化学习和注意力机制增强的RRT*算法

        参数:
            start: 起点，可以是(x, y)元组或Node对象
            goal: 终点，可以是(x, y)元组或Node对象
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

        # 将Node对象转换为元组
        start_tuple = (start.x, start.y) if isinstance(start, Node) else start
        goal_tuple = (goal.x, goal.y) if isinstance(goal, Node) else goal

        # 调用父类初始化
        super().__init__(start=start_tuple,
                         goal=goal_tuple,
                         env=env,
                         step_size=step_size,
                         max_iterations=max_iterations,
                         vehicle_width=vehicle_width,
                         vehicle_length=vehicle_length,
                         rewire_factor=rewire_factor)

        # 存储Node对象版本的起点和终点
        self.start_node = Node(start_tuple[0], start_tuple[1]) if not isinstance(start, Node) else start
        self.goal_node = Node(goal_tuple[0], goal_tuple[1]) if not isinstance(goal, Node) else goal

        # 更新状态和动作维度
        self.state_dim = 24  # 更新为实际的状态维度
        self.action_dim = 64  # 8(方向)*4(步长)*2(转向)
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon

        # 网络和优化器
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = EnhancedDQNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network = EnhancedDQNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 预测网络
        self.prediction_net = PredictionNetwork(
            input_dim=4,  # 位置和速度
            hidden_dim=64,
            sequence_length=prediction_horizon).to(self.device)

        # 优化器
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.pred_optimizer = torch.optim.Adam(self.prediction_net.parameters(), lr=learning_rate)

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
        self.path_history: List[Node] = []
        self.obstacle_history: List[List[Node]] = []

    def get_state(self, node: Node) -> np.ndarray:
        """获取增强的状态表示"""
        x, y = node.x, node.y
        goal_x, goal_y = self.goal_node.x, self.goal_node.y

        # 获取最近的k个障碍物
        k = 3
        nearest_obstacles = self.get_k_nearest_obstacles(node, k)

        # 计算到目标的距离和角度
        dist_to_goal = np.hypot(goal_x - x, goal_y - y)
        angle_to_goal = np.arctan2(goal_y - y, goal_x - x)

        # 计算路径历史特征
        path_features = self.compute_path_features(node)

        # 计算障碍物相对位置和距离
        obstacle_features = []
        for obs in nearest_obstacles:
            rel_x = obs.x - x
            rel_y = obs.y - y
            dist = np.hypot(rel_x, rel_y)
            angle = np.arctan2(rel_y, rel_x)
            obstacle_features.extend([rel_x, rel_y, dist, angle])

        # 如果障碍物数量不足，用零填充
        while len(obstacle_features) < 12:  # 3个障碍物，每个4个特征
            obstacle_features.extend([0.0, 0.0, 0.0, 0.0])

        # 构建状态向量
        state = np.array(
            [
                x,
                y,  # 当前位置 (2)
                goal_x,
                goal_y,  # 目标位置 (2)
                dist_to_goal,  # 到目标距离 (1)
                angle_to_goal,  # 到目标角度 (1)
                self.vehicle_width,  # 车辆宽度 (1)
                self.vehicle_length,  # 车辆长度 (1)
                *path_features,  # 路径历史特征 (4)
                *obstacle_features,  # 障碍物特征 (12)
            ],
            dtype=np.float32)

        # 归一化状态
        state = self._normalize_state(state)
        return state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """归一化状态向量"""
        # 位置归一化（基于环境大小）
        state[0] /= self.env.width
        state[1] /= self.env.height
        state[2] /= self.env.width
        state[3] /= self.env.height

        # 距离归一化（基于环境对角线长度）
        max_dist = np.hypot(self.env.width, self.env.height)
        state[4] /= max_dist

        # 角度已经是归一化的（-π到π）
        # 车辆尺寸归一化（基于环境大小）
        state[6] /= self.env.width
        state[7] /= self.env.height

        # 路径特征归一化
        # 距离
        state[8] /= max_dist
        # 角度已经是归一化的
        # 曲率归一化（假设最大曲率为1/min(车长,车宽)）
        max_curvature = 1.0 / min(self.vehicle_length, self.vehicle_width)
        state[10] /= max_curvature
        # 路径长度归一化（基于最大迭代次数）
        state[11] /= self.max_iterations

        # 障碍物特征归一化
        for i in range(12, len(state), 4):
            # 相对位置
            state[i] /= self.env.width
            state[i + 1] /= self.env.height
            # 距离
            state[i + 2] /= max_dist
            # 角度已经是归一化的

        return state

    def get_k_nearest_obstacles(self, node: Node, k: int) -> List[Node]:
        """获取k个最近的障碍物"""
        obstacles = []
        for obs in self.env.obstacles:
            dist = np.hypot(node.x - obs.x, node.y - obs.y)
            obstacles.append((Node(obs.x, obs.y), dist))

        # 按距离排序并返回前k个
        obstacles.sort(key=lambda x: x[1])
        return [obs[0] for obs in obstacles[:k]]

    def compute_path_features(self, node: Node) -> np.ndarray:
        """计算路径相关特征"""
        if not self.path_history:
            return np.zeros(4)

        # 计算与历史路径的关系
        min_dist = float('inf')
        min_angle = 0

        for path_point in self.path_history[-10:]:  # 只使用最近的10个点
            dist = np.hypot(node.x - path_point.x, node.y - path_point.y)
            if dist < min_dist:
                min_dist = dist
                min_angle = np.arctan2(path_point.y - node.y, path_point.x - node.x)

        # 计算路径曲率
        if len(self.path_history) >= 3:
            p1, p2, p3 = self.path_history[-3:]
            curvature = self.compute_curvature(p1, p2, p3)
        else:
            curvature = 0

        return np.array([min_dist, min_angle, curvature, len(self.path_history)])

    def compute_curvature(self, p1: Node, p2: Node, p3: Node) -> float:
        """计算三点曲率"""
        try:
            # 使用外接圆半径的倒数作为曲率
            a = np.hypot(p2.x - p1.x, p2.y - p1.y)
            b = np.hypot(p3.x - p2.x, p3.y - p2.y)
            c = np.hypot(p3.x - p1.x, p3.y - p1.y)

            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))

            if area > 0:
                return 4 * area / (a * b * c)
            return 0
        except Exception as e:
            print(f"计算曲率时出错: {e}")
            return 0

    def predict_obstacle_motion(self, obstacle_history: List[Node]) -> np.ndarray:
        """预测障碍物运动"""
        try:
            if len(obstacle_history) < self.prediction_horizon:
                return np.zeros(2)  # 返回静态位置

            # 准备序列数据，包括位置和速度
            sequence_data = []
            for i in range(len(obstacle_history) - self.prediction_horizon, len(obstacle_history)):
                obs = obstacle_history[i]

                # 计算速度 (如果有前一个点)
                vx, vy = 0.0, 0.0
                if i > 0:
                    prev_obs = obstacle_history[i - 1]
                    vx = obs.x - prev_obs.x
                    vy = obs.y - prev_obs.y

                # 位置和速度: [x, y, vx, vy]
                sequence_data.append([obs.x, obs.y, vx, vy])

            sequence = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)

            # 预测下一个位置
            with torch.no_grad():
                prediction = self.prediction_net(sequence)

            return prediction.cpu().numpy()[0][:2]  # 只返回位置预测，不返回速度预测
        except Exception as e:
            print(f"预测障碍物运动出错: {e}")
            return np.zeros(2)  # 出错时返回静态位置

    def get_reward(self, new_node: Node, collision: bool) -> float:
        """增强的奖励计算，考虑停车动作的合理性"""
        try:
            if collision:
                return -100.0

            # 基础距离奖励
            dist_to_goal = np.hypot(new_node.x - self.goal_node.x, new_node.y - self.goal_node.y)
            dist_reward = -dist_to_goal * 0.1

            # 方向奖励：鼓励车辆朝向目标方向
            if len(self.path_history) >= 2:
                prev_node = self.path_history[-2]
                current_direction = np.arctan2(new_node.y - prev_node.y, new_node.x - prev_node.x)
                goal_direction = np.arctan2(self.goal_node.y - new_node.y, self.goal_node.x - new_node.x)
                angle_diff = abs(current_direction - goal_direction)
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                direction_reward = -angle_diff * 10.0
            else:
                direction_reward = 0.0

            # 安全距离奖励
            safety_reward = 0
            for obs in self.get_k_nearest_obstacles(new_node, 3):
                dist = np.hypot(new_node.x - obs.x, new_node.y - obs.y)
                safety_reward += min(dist * 0.2, 10.0)

            # 路径平滑度奖励
            smoothness_reward = 0
            if len(self.path_history) >= 3:
                curvature = self.compute_curvature(self.path_history[-2], self.path_history[-1], new_node)
                smoothness_reward = -curvature * 5.0  # 惩罚高曲率

            # 预测奖励
            prediction_reward = 0
            # 如果障碍物历史不足，跳过预测奖励
            if self.obstacle_history and all(len(h) >= self.prediction_horizon for h in self.obstacle_history):
                for obs_history in self.obstacle_history:
                    predicted_pos = self.predict_obstacle_motion(obs_history)
                    future_dist = np.hypot(new_node.x - predicted_pos[0], new_node.y - predicted_pos[1])
                    prediction_reward += min(future_dist * 0.1, 5.0)

            total_reward = dist_reward + safety_reward + \
                smoothness_reward + direction_reward + (prediction_reward if prediction_reward != 0 else 0)

            return total_reward
        except Exception as e:
            print(f"计算奖励出错: {e}")
            return -dist_to_goal * 0.1  # 出错时返回简单的距离奖励

    def update_prediction_network(self):
        """更新预测网络"""
        try:
            if len(self.obstacle_history) < self.prediction_horizon:
                return

            # 准备训练数据
            sequences = []
            targets = []

            for obs_history in self.obstacle_history:
                if len(obs_history) > self.prediction_horizon:
                    for i in range(len(obs_history) - self.prediction_horizon):
                        try:
                            # 准备输入序列（包括位置和速度）
                            seq_data = []
                            for j in range(i, i + self.prediction_horizon):
                                obs = obs_history[j]

                                # 计算速度 (如果有前一个点)
                                vx, vy = 0.0, 0.0
                                if j > 0:
                                    prev_obs = obs_history[j - 1]
                                    vx = obs.x - prev_obs.x
                                    vy = obs.y - prev_obs.y

                                seq_data.append([obs.x, obs.y, vx, vy])

                            # 目标：下一个点的位置和速度
                            target_obs = obs_history[i + self.prediction_horizon]

                            # 计算目标速度
                            target_vx, target_vy = 0.0, 0.0
                            if i + self.prediction_horizon > 0:
                                prev_obs = obs_history[i + self.prediction_horizon - 1]
                                target_vx = target_obs.x - prev_obs.x
                                target_vy = target_obs.y - prev_obs.y

                            target = [target_obs.x, target_obs.y, target_vx, target_vy]

                            sequences.append(seq_data)
                            targets.append(target)
                        except Exception as e:
                            print(f"准备训练样本出错: {e}")
                            continue

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
        except Exception as e:
            print(f"更新预测网络出错: {e}")

    def extend(self, nearest_node: Node) -> Optional[Node]:
        """增强的扩展方法"""
        try:
            # 获取状态并选择动作
            state = self.get_state(nearest_node)
            action = self.select_action(state)

            # 将扩展的离散动作转换为连续参数
            # 动作空间: 64 = 8(方向) * 4(步长) * 2(转向)
            direction = action % 8  # 8个方向
            step_scale = ((action // 8) % 4) * 0.25 + 0.25  # 步长比例 [0.25, 0.5, 0.75, 1.0]
            steering = (action // 32) % 2  # 转向选项 [0, 1]

            # 计算角度和步长
            base_angle = 2 * np.pi * direction / 8
            if steering == 1:  # 应用转向调整
                base_angle += np.pi / 16  # 轻微转向调整

            dx = self.step_size * step_scale * np.cos(base_angle)
            dy = self.step_size * step_scale * np.sin(base_angle)

            new_node = Node(nearest_node.x + dx, nearest_node.y + dy)

            # 检查碰撞
            collision = self.check_collision(nearest_node, new_node)

            # 更新历史记录
            if not collision:
                self.path_history.append(new_node)

                # 更新障碍物历史（只在有障碍物时更新）
                try:
                    current_obstacles = self.get_k_nearest_obstacles(new_node, 3)
                    if current_obstacles:  # 确保有障碍物
                        for i, obs in enumerate(current_obstacles):
                            if i >= len(self.obstacle_history):
                                self.obstacle_history.append([obs])
                            else:
                                self.obstacle_history[i].append(obs)
                except Exception as e:
                    print(f"更新障碍物历史出错: {e}")

                # 计算奖励并存储经验
                reward = self.get_reward(new_node, collision)
                next_state = self.get_state(new_node)
                done = collision or self.is_goal_reached(new_node)

                # 存储经验
                self.replay_buffer.push(state, action, reward, next_state, done)

                # 仅当缓冲区有足够样本时更新网络
                if len(self.replay_buffer) >= self.batch_size * 10:  # 确保有足够样本
                    self.update_network()
                    self.update_prediction_network()

            return new_node if not collision else None
        except Exception as e:
            print(f"扩展节点出错: {e}")
            return None

    def update_network(self):
        """更新Q网络"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # 采样经验数据
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # 计算当前Q值
        current_q_values, features = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + \
                (1 - dones) * self.gamma * max_next_q_values

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
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        self.episode_rewards.append(rewards[0].item())  # 记录第一个样本的奖励

    def save_model(self, path: str):
        """保存模型"""
        torch.save(
            {
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
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.prediction_net.load_state_dict(checkpoint['prediction_network_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.pred_optimizer.load_state_dict(checkpoint['pred_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.prediction_errors = checkpoint['prediction_errors']
        self.attention_maps = checkpoint['attention_maps']

    def select_action(self, state):
        """选择动作"""
        # 将状态转换为tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # epsilon-greedy策略
        if random.random() < self.epsilon:
            # 随机探索
            action = random.randint(0, self.action_dim - 1)
        else:
            # 利用当前策略
            with torch.no_grad():
                q_values, _ = self.q_network(state)
                action = q_values.argmax(1).item()

        return action

    def plan(self):
        """规划路径，优化停车动作"""
        # 初始化路径
        self.path_history = [self.start_node]
        current_node = self.start_node

        # 用于跟踪最佳停车姿态
        best_parking_score = float('-inf')
        best_parking_node = None
        best_action = None  # 记录最佳动作

        for _ in range(self.max_iterations):
            # 获取当前状态
            state = self.get_state(current_node)

            # 选择动作
            action = self.select_action(state)

            # 扩展节点
            new_node = self.extend(current_node)

            if new_node is None:
                continue

            # 更新当前节点
            current_node = new_node

            # 检查是否接近目标
            dist_to_goal = np.hypot(current_node.x - self.goal_node.x, current_node.y - self.goal_node.y)

            if dist_to_goal < self.step_size * 2:
                # 计算停车姿态得分
                if len(self.path_history) >= 2:
                    prev_node = self.path_history[-2]
                    current_direction = np.arctan2(current_node.y - prev_node.y, current_node.x - prev_node.x)
                    desired_direction = np.arctan2(self.goal_node.y - self.start_node.y,
                                                   self.goal_node.x - self.start_node.x)
                    angle_diff = abs(current_direction - desired_direction)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                    parking_score = 1.0 - angle_diff / np.pi

                    # 考虑与障碍物的距离
                    min_obstacle_dist = float('inf')
                    for obs in self.get_k_nearest_obstacles(current_node, 3):
                        dist = np.hypot(current_node.x - obs.x, current_node.y - obs.y)
                        min_obstacle_dist = min(min_obstacle_dist, dist)

                    # 综合评分：姿态 + 安全距离 + 动作平滑度
                    action_smoothness = 1.0 - abs(action % 8 - 4) / 4.0  # 评估动作的平滑度
                    total_score = parking_score + min(min_obstacle_dist * 0.1, 1.0) + action_smoothness * 0.5

                    if total_score > best_parking_score:
                        best_parking_score = total_score
                        best_parking_node = current_node
                        best_action = action

            # 检查是否达到目标
            if self.is_goal_reached(current_node):
                # 如果找到了更好的停车姿态，使用它
                if best_parking_node is not None and best_parking_score > 0.8:
                    # 存储最佳动作用于经验回放
                    if best_action is not None:
                        state = self.get_state(self.path_history[-1])
                        next_state = self.get_state(best_parking_node)
                        reward = self.get_reward(best_parking_node, False) * 2  # 额外奖励
                        self.replay_buffer.push(state, best_action, reward, next_state, True)

                    self.path_history.append(best_parking_node)
                    return self.path_history

                # 否则使用当前路径
                self.path_history.append(self.goal_node)
                return self.path_history

        # 如果找到了可接受的停车姿态但未完全到达目标
        if best_parking_node is not None and best_parking_score > 0.7:
            # 存储最佳动作用于经验回放
            if best_action is not None:
                state = self.get_state(self.path_history[-1])
                next_state = self.get_state(best_parking_node)
                reward = self.get_reward(best_parking_node, False) * 1.5  # 额外奖励
                self.replay_buffer.push(state, best_action, reward, next_state, True)

            self.path_history.append(best_parking_node)
            return self.path_history

        return None

    def is_goal_reached(self, node: Node) -> bool:
        """检查是否到达目标，考虑停车要求"""
        if not node:
            return False

        dist = np.hypot(node.x - self.goal_node.x, node.y - self.goal_node.y)

        # 基本距离要求
        if dist > self.step_size * 2:
            return False

        # 检查停车姿态
        if len(self.path_history) >= 2:
            prev_node = self.path_history[-2]
            current_direction = np.arctan2(node.y - prev_node.y, node.x - prev_node.x)
            desired_direction = np.arctan2(self.goal_node.y - self.start_node.y, self.goal_node.x - self.start_node.x)
            angle_diff = abs(current_direction - desired_direction)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            # 要求角度差小于30度
            if angle_diff > np.pi / 6:
                return False

        return True

    def check_collision(self, from_node: Node, to_node: Node) -> bool:
        """检查路径是否碰撞"""
        # 检查节点是否在环境边界内
        if not (0 <= to_node.x <= self.env.width and 0 <= to_node.y <= self.env.height):
            return True

        # 检查与障碍物的碰撞
        for obs in self.env.obstacles:
            # 简化的碰撞检测：检查线段是否与矩形相交
            if obs.type == "rectangle":
                # 获取矩形的边界
                left = obs.x - obs.width / 2
                right = obs.x + obs.width / 2
                bottom = obs.y - obs.height / 2
                top = obs.y + obs.height / 2

                # 检查线段是否与矩形相交
                if self._line_rectangle_intersection((from_node.x, from_node.y), (to_node.x, to_node.y), (left, bottom),
                                                     (right, top)):
                    return True

        return False

    def _line_rectangle_intersection(self, line_start: tuple, line_end: tuple, rect_bl: tuple, rect_tr: tuple) -> bool:
        """检查线段是否与矩形相交"""
        # 线段参数
        x1, y1 = line_start
        x2, y2 = line_end

        # 矩形参数
        left, bottom = rect_bl
        right, top = rect_tr

        # 快速排除：检查线段的包围盒是否与矩形相交
        if max(x1, x2) < left or min(x1, x2) > right or \
           max(y1, y2) < bottom or min(y1, y2) > top:
            return False

        # 检查线段是否完全在矩形内部
        if left <= x1 <= right and bottom <= y1 <= top and \
           left <= x2 <= right and bottom <= y2 <= top:
            return True

        # 检查线段是否与矩形的边相交
        edges = [
            ((left, bottom), (right, bottom)),  # 底边
            ((right, bottom), (right, top)),  # 右边
            ((right, top), (left, top)),  # 顶边
            ((left, top), (left, bottom))  # 左边
        ]

        for edge_start, edge_end in edges:
            if self._line_segments_intersect(line_start, line_end, edge_start, edge_end):
                return True

        return False

    def _line_segments_intersect(self, p1: tuple, p2: tuple, p3: tuple, p4: tuple) -> bool:
        """检查两条线段是否相交"""

        # 计算方向
        def direction(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) - \
                   (p2[1] - p1[1]) * (p3[0] - p1[0])

        # 检查点是否在线段上
        def on_segment(p, q, r):
            return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
                   q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

        d1 = direction(p3, p4, p1)
        d2 = direction(p3, p4, p2)
        d3 = direction(p1, p2, p3)
        d4 = direction(p1, p2, p4)

        # 一般情况下的相交
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        # 特殊情况：线段共线
        if d1 == 0 and on_segment(p3, p1, p4):
            return True
        if d2 == 0 and on_segment(p3, p2, p4):
            return True
        if d3 == 0 and on_segment(p1, p3, p2):
            return True
        if d4 == 0 and on_segment(p1, p4, p2):
            return True

        return False
