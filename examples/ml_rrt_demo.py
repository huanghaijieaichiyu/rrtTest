#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习增强RRT算法演示脚本

演示如何使用PyTorch神经网络增强RRT算法的路径规划效果。
"""

from ml.models.rrt_nn import SamplingNetwork, CollisionNet, HeuristicNet
from simulation.visualization import Visualization
from simulation.environment import Environment
from rrt.rrt_star import RRTStar
from rrt.rrt_base import RRT
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目模块


class MLEnhancedRRT(RRTStar):
    """
    机器学习增强的RRT*算法

    使用PyTorch神经网络来增强RRT*算法的性能，包括：
    1. 使用采样网络优化采样策略
    2. 使用碰撞预测网络加速碰撞检测
    3. 使用启发式函数网络优化路径评估
    """

    def __init__(
        self,
        start,
        goal,
        env,
        step_size=1.0,
        max_iterations=1000,
        goal_sample_rate=0.05,
        search_radius=50.0,
        sampling_network=None,
        collision_network=None,
        heuristic_network=None,
        device=None,
        ml_sample_ratio=0.5,  # ML采样的比例
        ml_collision_ratio=0.7,  # ML碰撞检测的比例
        use_ml_heuristic=True  # 是否使用ML启发式函数
    ):
        """
        初始化机器学习增强的RRT*规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象
            step_size: 步长
            max_iterations: 最大迭代次数
            goal_sample_rate: 目标采样率
            search_radius: 搜索半径
            sampling_network: 采样网络
            collision_network: 碰撞预测网络
            heuristic_network: 启发式函数网络
            device: PyTorch设备
            ml_sample_ratio: ML采样的比例
            ml_collision_ratio: ML碰撞检测的比例
            use_ml_heuristic: 是否使用ML启发式函数
        """
        super().__init__(
            start, goal, env, step_size, max_iterations,
            goal_sample_rate, search_radius
        )

        self.sampling_network = sampling_network
        self.collision_network = collision_network
        self.heuristic_network = heuristic_network
        self.device = device or torch.device('cpu')

        self.ml_sample_ratio = ml_sample_ratio
        self.ml_collision_ratio = ml_collision_ratio
        self.use_ml_heuristic = use_ml_heuristic

        # 性能统计
        self.stats = {
            'ml_samples': 0,
            'random_samples': 0,
            'ml_collisions': 0,
            'env_collisions': 0,
            'ml_heuristics': 0,
            'euclidean_heuristics': 0,
        }

    def _random_node(self):
        """生成随机节点，有概率使用ML采样"""
        # 是否使用ML采样
        if self.sampling_network and np.random.random() < self.ml_sample_ratio:
            # 使用ML采样
            self.stats['ml_samples'] += 1
            sample = self.sampling_network.sample(
                start=(self.start.x, self.start.y),
                goal=(self.goal.x, self.goal.y),
                width=self.max_x - self.min_x,
                height=self.max_y - self.min_y,
                device=self.device
            )[0]  # 只取一个样本

            # 将PyTorch张量转换为NumPy
            sample = sample.cpu().numpy()

            # 创建节点
            from rrt.rrt_base import Node
            node = Node(float(sample[0]), float(sample[1]))
            return node
        else:
            # 使用随机采样
            self.stats['random_samples'] += 1
            return super()._random_node()

    def _check_collision(self, node1, node2):
        """检查碰撞，有概率使用ML碰撞预测"""
        # 是否使用ML碰撞检测
        if self.collision_network and np.random.random() < self.ml_collision_ratio:
            # 使用ML碰撞预测
            self.stats['ml_collisions'] += 1

            # 预测是否碰撞
            is_collision = self.collision_network.predict_collision(
                start=(node1.x, node1.y),
                end=(node2.x, node2.y),
                width=self.max_x - self.min_x,
                height=self.max_y - self.min_y,
                device=self.device
            )

            # 如果预测碰撞，再用环境确认一次
            if is_collision:
                return False  # 碰撞，不可通过
            else:
                # 预测无碰撞，使用环境确认
                return super()._check_collision(node1, node2)
        else:
            # 使用环境碰撞检测
            self.stats['env_collisions'] += 1
            return super()._check_collision(node1, node2)

    def _calc_new_cost(self, from_node, to_node):
        """计算代价，可选择使用ML启发式函数"""
        # 基础代价（欧几里得距离）
        base_cost = super()._calc_new_cost(from_node, to_node)

        # 是否使用ML启发式函数
        if self.heuristic_network and self.use_ml_heuristic:
            # 使用ML启发式函数
            self.stats['ml_heuristics'] += 1

            # 估计代价
            ml_cost = self.heuristic_network.estimate_cost(
                node_pos=(to_node.x, to_node.y),
                goal_pos=(self.goal.x, self.goal.y),
                width=self.max_x - self.min_x,
                height=self.max_y - self.min_y,
                device=self.device
            )

            # 组合代价（这里简单地加权平均）
            return 0.7 * base_cost + 0.3 * ml_cost
        else:
            # 使用基础代价
            self.stats['euclidean_heuristics'] += 1
            return base_cost


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    print("机器学习增强RRT算法演示")

    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config/default_config.yaml"
    )
    config = load_config(config_path)

    # 创建环境
    env = Environment(width=100, height=100)

    # 添加障碍物
    # 设置随机种子，以便结果可复现
    np.random.seed(42)

    # 添加一些随机圆形障碍物
    for _ in range(15):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        radius = np.random.uniform(3, 8)
        env.add_obstacle(x, y, obstacle_type="circle", radius=radius)

    # 添加一些随机矩形障碍物
    for _ in range(8):
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        width = np.random.uniform(5, 15)
        height = np.random.uniform(5, 15)
        angle = np.random.uniform(0, 2 * np.pi)
        env.add_obstacle(
            x, y, obstacle_type="rectangle",
            width=width, height=height, angle=angle
        )

    # 设置起点和终点
    start = (10, 10)
    goal = (90, 90)

    # 可视化环境
    vis = Visualization(env)
    vis.plot_path(
        path=[],  # 空路径，只展示环境
        start=start,
        goal=goal,
        title="规划环境"
    )
    plt.show()

    # 创建神经网络模型
    print("创建神经网络模型...")

    # 1. 采样网络
    sampling_net = SamplingNetwork(
        hidden_sizes=[128, 64],
        activation='relu',
        use_batch_norm=True
    ).to(device)

    # 2. 碰撞预测网络
    collision_net = CollisionNet(
        hidden_sizes=[128, 64],
        use_batch_norm=True
    ).to(device)

    # 3. 启发式函数网络
    heuristic_net = HeuristicNet(
        hidden_sizes=[128, 64],
        use_batch_norm=True
    ).to(device)

    # 创建标准RRT*规划器（用于比较）
    print("创建标准RRT*规划器...")
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        env=env,
        step_size=3.0,
        max_iterations=1000,
        goal_sample_rate=0.1,
        search_radius=30.0
    )

    # 创建ML增强的RRT*规划器
    print("创建ML增强的RRT*规划器...")
    ml_rrt = MLEnhancedRRT(
        start=start,
        goal=goal,
        env=env,
        step_size=3.0,
        max_iterations=1000,
        goal_sample_rate=0.1,
        search_radius=30.0,
        sampling_network=sampling_net,
        collision_network=collision_net,
        heuristic_network=heuristic_net,
        device=device,
        ml_sample_ratio=0.7,
        ml_collision_ratio=0.8,
        use_ml_heuristic=True
    )

    # 执行标准RRT*规划
    print("执行标准RRT*路径规划...")
    start_time = time.time()
    rrt_star_path = rrt_star.plan()
    rrt_star_time = time.time() - start_time

    if rrt_star_path:
        print(
            f"标准RRT*找到路径，包含 {len(rrt_star_path)} 个节点，用时 {rrt_star_time:.3f} 秒")
        rrt_star_length = sum(
            np.hypot(
                rrt_star_path[i][0] - rrt_star_path[i-1][0],
                rrt_star_path[i][1] - rrt_star_path[i-1][1]
            )
            for i in range(1, len(rrt_star_path))
        )
        print(f"路径长度: {rrt_star_length:.2f}")
    else:
        print(f"标准RRT*未能找到路径，用时 {rrt_star_time:.3f} 秒")

    # 执行ML增强的RRT*规划
    print("执行ML增强的RRT*路径规划...")
    start_time = time.time()
    ml_rrt_path = ml_rrt.plan()
    ml_rrt_time = time.time() - start_time

    if ml_rrt_path:
        print(f"ML增强RRT*找到路径，包含 {len(ml_rrt_path)} 个节点，用时 {ml_rrt_time:.3f} 秒")
        ml_rrt_length = sum(
            np.hypot(
                ml_rrt_path[i][0] - ml_rrt_path[i-1][0],
                ml_rrt_path[i][1] - ml_rrt_path[i-1][1]
            )
            for i in range(1, len(ml_rrt_path))
        )
        print(f"路径长度: {ml_rrt_length:.2f}")
    else:
        print(f"ML增强RRT*未能找到路径，用时 {ml_rrt_time:.3f} 秒")

    # 打印ML增强RRT*的统计信息
    print("\nML增强RRT*的统计信息:")
    print(f"ML采样次数: {ml_rrt.stats['ml_samples']}")
    print(f"随机采样次数: {ml_rrt.stats['random_samples']}")
    print(f"ML碰撞检测次数: {ml_rrt.stats['ml_collisions']}")
    print(f"环境碰撞检测次数: {ml_rrt.stats['env_collisions']}")
    print(f"ML启发式函数使用次数: {ml_rrt.stats['ml_heuristics']}")
    print(f"欧几里得启发式函数使用次数: {ml_rrt.stats['euclidean_heuristics']}")

    # 比较和可视化结果
    if rrt_star_path and ml_rrt_path:
        print("\n比较结果:")
        print(
            f"标准RRT*: {len(rrt_star_path)} 节点, {rrt_star_time:.3f} 秒, 长度 {rrt_star_length:.2f}")
        print(
            f"ML增强RRT*: {len(ml_rrt_path)} 节点, {ml_rrt_time:.3f} 秒, 长度 {ml_rrt_length:.2f}")

        # 计算性能提升
        time_improvement = (rrt_star_time - ml_rrt_time) / rrt_star_time * 100
        length_improvement = (
            rrt_star_length - ml_rrt_length) / rrt_star_length * 100

        print(f"时间提升: {time_improvement:.2f}%")
        print(f"路径长度提升: {length_improvement:.2f}%")

        # 可视化路径比较
        vis.plot_multi_paths(
            paths=[rrt_star_path, ml_rrt_path],
            labels=["标准RRT*", "ML增强RRT*"],
            colors=["blue", "red"],
            title="路径比较: 标准RRT* vs ML增强RRT*"
        )
        plt.show()

    # 可视化ML增强RRT*的搜索树
    if ml_rrt_path:
        vis.plot_search_tree(
            nodes=ml_rrt.node_list,
            path=ml_rrt_path,
            title="ML增强RRT*搜索树和规划路径"
        )
        plt.show()

        # 创建路径动画
        print("创建ML增强RRT*路径动画...")
        # 忽略警告
        import warnings
        warnings.filterwarnings("ignore")
        anim = vis.animate_path(ml_rrt_path)
        plt.show()

    print("演示完成!")


if __name__ == "__main__":
    main()
