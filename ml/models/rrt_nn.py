"""
RRT神经网络模型模块

提供用于增强RRT算法的神经网络模型。
这些模型可以用于：
1. 优化采样策略
2. 预测碰撞可能性
3. 学习启发式函数
4. 端到端路径规划
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional


class SamplingNetwork(nn.Module):
    """
    采样网络

    用于学习RRT算法的采样策略，根据当前状态和目标状态预测下一个采样点。
    """

    def __init__(
        self,
        # [start_x, start_y, goal_x, goal_y, width, height]
        input_dim: int = 6,
        hidden_sizes: List[int] = [128, 64],
        output_dim: int = 2,  # [x, y]
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_prob: float = 0.1
    ):
        """
        初始化采样网络

        参数:
            input_dim: 输入维度
            hidden_sizes: 隐藏层大小列表
            output_dim: 输出维度
            activation: 激活函数
            use_batch_norm: 是否使用批标准化
            dropout_prob: Dropout概率
        """
        super(SamplingNetwork, self).__init__()

        # 保存参数
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        # 创建网络层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))

        # 添加激活函数
        if activation.lower() == 'relu':
            layers.append(nn.ReLU())
        elif activation.lower() == 'tanh':
            layers.append(nn.Tanh())
        elif activation.lower() == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        # 添加Dropout
        layers.append(nn.Dropout(dropout_prob))

        # 添加隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))

            # 添加激活函数
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())

            # 添加Dropout
            layers.append(nn.Dropout(dropout_prob))

        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        # 输出使用Sigmoid激活，将输出限制在[0, 1]范围内，用于表示归一化的坐标
        layers.append(nn.Sigmoid())

        # 创建网络
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, input_dim]

        返回:
            预测的采样点，形状为 [batch_size, output_dim]
        """
        return self.network(x)

    def sample(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        width: float,
        height: float,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        生成采样点

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            width: 环境宽度
            height: 环境高度
            batch_size: 批大小
            device: 设备

        返回:
            采样点张量，形状为 [batch_size, 2]
        """
        # 归一化坐标
        start_x_norm = start[0] / width
        start_y_norm = start[1] / height
        goal_x_norm = goal[0] / width
        goal_y_norm = goal[1] / height

        # 创建输入张量
        x = torch.tensor(
            [start_x_norm, start_y_norm, goal_x_norm, goal_y_norm, 1.0, 1.0],
            dtype=torch.float32
        )

        # 扩展为批大小
        x = x.unsqueeze(0).repeat(batch_size, 1)

        # 移动到指定设备
        if device is not None:
            x = x.to(device)
            self.to(device)

        # 预测采样点
        with torch.no_grad():
            sampled_points_norm = self.forward(x)

        # 反归一化
        sampled_points = torch.zeros_like(sampled_points_norm)
        sampled_points[:, 0] = sampled_points_norm[:, 0] * width
        sampled_points[:, 1] = sampled_points_norm[:, 1] * height

        return sampled_points


class CollisionNet(nn.Module):
    """
    碰撞预测网络

    用于预测线段是否与环境中的障碍物发生碰撞。
    """

    def __init__(
        self,
        input_dim: int = 4,  # [start_x, start_y, end_x, end_y]
        hidden_sizes: List[int] = [128, 64],
        use_batch_norm: bool = True,
        dropout_prob: float = 0.1
    ):
        """
        初始化碰撞预测网络

        参数:
            input_dim: 输入维度
            hidden_sizes: 隐藏层大小列表
            use_batch_norm: 是否使用批标准化
            dropout_prob: Dropout概率
        """
        super(CollisionNet, self).__init__()

        # 保存参数
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.use_batch_norm = use_batch_norm

        # 创建网络层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

        # 添加隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # 输出层，输出碰撞概率
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(nn.Sigmoid())

        # 创建网络
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, input_dim]

        返回:
            碰撞概率，形状为 [batch_size, 1]
        """
        return self.network(x)

    def predict_collision(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        threshold: float = 0.5,
        normalize: bool = True,
        width: float = 100.0,
        height: float = 100.0,
        device: Optional[torch.device] = None
    ) -> bool:
        """
        预测线段是否与障碍物碰撞

        参数:
            start: 线段起点坐标 (x, y)
            end: 线段终点坐标 (x, y)
            threshold: 碰撞概率阈值
            normalize: 是否归一化坐标
            width: 环境宽度，用于归一化
            height: 环境高度，用于归一化
            device: 设备

        返回:
            如果预测碰撞则返回True，否则返回False
        """
        # 归一化坐标
        if normalize:
            start_x_norm = start[0] / width
            start_y_norm = start[1] / height
            end_x_norm = end[0] / width
            end_y_norm = end[1] / height
        else:
            start_x_norm = start[0]
            start_y_norm = start[1]
            end_x_norm = end[0]
            end_y_norm = end[1]

        # 创建输入张量
        x = torch.tensor(
            [start_x_norm, start_y_norm, end_x_norm, end_y_norm],
            dtype=torch.float32
        )

        # 扩展为批大小1
        x = x.unsqueeze(0)

        # 移动到指定设备
        if device is not None:
            x = x.to(device)
            self.to(device)

        # 预测碰撞概率
        with torch.no_grad():
            collision_prob = self.forward(x)

        # 根据阈值判断是否碰撞
        return collision_prob.item() > threshold


class HeuristicNet(nn.Module):
    """
    启发式函数网络

    用于学习RRT*算法中的启发式函数，评估节点的代价。
    """

    def __init__(
        self,
        input_dim: int = 6,  # [node_x, node_y, goal_x, goal_y, width, height]
        hidden_sizes: List[int] = [128, 64],
        use_batch_norm: bool = True,
        dropout_prob: float = 0.1
    ):
        """
        初始化启发式函数网络

        参数:
            input_dim: 输入维度
            hidden_sizes: 隐藏层大小列表
            use_batch_norm: 是否使用批标准化
            dropout_prob: Dropout概率
        """
        super(HeuristicNet, self).__init__()

        # 保存参数
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.use_batch_norm = use_batch_norm

        # 创建网络层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_sizes[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

        # 添加隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # 输出层，输出代价估计
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        # 使用ReLU确保代价非负
        layers.append(nn.ReLU())

        # 创建网络
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, input_dim]

        返回:
            代价估计，形状为 [batch_size, 1]
        """
        return self.network(x)

    def estimate_cost(
        self,
        node_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        normalize: bool = True,
        width: float = 100.0,
        height: float = 100.0,
        device: Optional[torch.device] = None
    ) -> float:
        """
        估计从节点到目标的代价

        参数:
            node_pos: 节点位置 (x, y)
            goal_pos: 目标位置 (x, y)
            normalize: 是否归一化坐标
            width: 环境宽度，用于归一化
            height: 环境高度，用于归一化
            device: 设备

        返回:
            估计的代价值
        """
        # 归一化坐标
        if normalize:
            node_x_norm = node_pos[0] / width
            node_y_norm = node_pos[1] / height
            goal_x_norm = goal_pos[0] / width
            goal_y_norm = goal_pos[1] / height
            width_norm = 1.0
            height_norm = 1.0
        else:
            node_x_norm = node_pos[0]
            node_y_norm = node_pos[1]
            goal_x_norm = goal_pos[0]
            goal_y_norm = goal_pos[1]
            width_norm = width
            height_norm = height

        # 创建输入张量
        x = torch.tensor(
            [node_x_norm, node_y_norm, goal_x_norm,
                goal_y_norm, width_norm, height_norm],
            dtype=torch.float32
        )

        # 扩展为批大小1
        x = x.unsqueeze(0)

        # 移动到指定设备
        if device is not None:
            x = x.to(device)
            self.to(device)

        # 估计代价
        with torch.no_grad():
            cost = self.forward(x)

        return cost.item()


class EndToEndRRTNet(nn.Module):
    """
    端到端RRT网络

    将RRT算法端到端地实现为神经网络，同时结合采样、碰撞检测和代价估计功能。
    """

    def __init__(
        self,
        input_channels: int = 3,  # [起点图层, 终点图层, 障碍物图层]
        hidden_channels: List[int] = [16, 32, 64],
        output_channels: int = 1,  # 路径概率图
        kernel_size: int = 3,
        use_batch_norm: bool = True
    ):
        """
        初始化端到端RRT网络

        参数:
            input_channels: 输入通道数
            hidden_channels: 隐藏层通道数列表
            output_channels: 输出通道数
            kernel_size: 卷积核大小
            use_batch_norm: 是否使用批标准化
        """
        super(EndToEndRRTNet, self).__init__()

        # 保存参数
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

        # 计算padding
        padding = kernel_size // 2

        # 创建编码器（下采样）
        self.encoder = nn.ModuleList()
        in_channels = input_channels

        for out_channels in hidden_channels:
            block = []

            # 卷积层
            block.append(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size, padding=padding)
            )

            # 批标准化
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))

            # 激活函数
            block.append(nn.ReLU())

            # 下采样
            block.append(nn.MaxPool2d(2))

            self.encoder.append(nn.Sequential(*block))
            in_channels = out_channels

        # 创建解码器（上采样）
        self.decoder = nn.ModuleList()
        hidden_channels.reverse()

        for i, out_channels in enumerate(hidden_channels[1:] + [input_channels]):
            block = []

            # 上采样
            block.append(nn.Upsample(scale_factor=2,
                         mode='bilinear', align_corners=True))

            # 卷积层
            in_ch = hidden_channels[i]
            block.append(
                nn.Conv2d(in_ch, out_channels, kernel_size, padding=padding)
            )

            # 批标准化
            if use_batch_norm and i < len(hidden_channels) - 1:
                block.append(nn.BatchNorm2d(out_channels))

            # 激活函数
            if i < len(hidden_channels) - 1:
                block.append(nn.ReLU())

            self.decoder.append(nn.Sequential(*block))

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, input_channels, height, width]

        返回:
            预测的路径概率图，形状为 [batch_size, output_channels, height, width]
        """
        # 编码器前向传播
        encoder_features = []
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            encoder_features.append(x)

        # 解码器前向传播
        encoder_features.reverse()
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)

            # 跳跃连接（除了最后一层）
            if i < len(self.decoder) - 1:
                x = x + encoder_features[i+1]

        # 输出层
        x = self.output_layer(x)

        return x


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
    samples = sampling_net.sample(start, goal, width, height, batch_size=5)
    print(f"生成的采样点:\n{samples}")

    # 测试碰撞预测网络
    collision_net = CollisionNet()
    print(f"\n碰撞预测网络结构:\n{collision_net}")

    # 测试碰撞预测
    is_collision = collision_net.predict_collision(start, goal)
    print(f"碰撞预测结果: {is_collision}")

    # 测试启发式函数网络
    heuristic_net = HeuristicNet()
    print(f"\n启发式函数网络结构:\n{heuristic_net}")

    # 测试代价估计
    cost = heuristic_net.estimate_cost(start, goal)
    print(f"代价估计结果: {cost}")

    # 测试端到端RRT网络
    end_to_end_net = EndToEndRRTNet()
    print(f"\n端到端RRT网络结构:\n{end_to_end_net}")

    # 创建模拟输入
    batch_size = 2
    height, width = 64, 64
    x = torch.randn(batch_size, 3, height, width)

    # 前向传播
    output = end_to_end_net(x)
    print(f"端到端网络输出形状: {output.shape}")
