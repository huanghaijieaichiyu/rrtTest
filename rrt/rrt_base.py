#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RRT基础算法实现

包含RRT算法的基本功能：
- 随机采样
- 寻找最近节点
- 路径扩展
- 碰撞检测
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class Node:
    """
    RRT树中的节点类
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.path_x: List[float] = []  # 从根节点到当前节点的路径x坐标
        self.path_y: List[float] = []  # 从根节点到当前节点的路径y坐标
        self.parent: Optional['Node'] = None  # 父节点
        self.cost: float = 0.0  # 从起点到该节点的代价


class RRT:
    """
    基础RRT算法实现
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env,
        step_size: float = 1.0,
        max_iterations: int = 1000,
        goal_sample_rate: float = 0.05,
        search_radius: float = 50.0,
        vehicle_width: float = 1.8,  # 添加车辆宽度参数
        vehicle_length: float = 4.5  # 添加车辆长度参数
    ):
        """
        初始化RRT路径规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            step_size: 每次扩展的步长
            max_iterations: 最大迭代次数
            goal_sample_rate: 采样目标点的概率
            search_radius: 搜索半径，用于确定搜索范围
            vehicle_width: 车辆宽度(米)
            vehicle_length: 车辆长度(米)
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.env = env
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length

        # 初始化树
        self.node_list = [self.start]

        # 路径结果
        self.path = []

        # 计算区域边界
        self.min_x, self.max_x = self._calculate_x_bounds()
        self.min_y, self.max_y = self._calculate_y_bounds()

    def _calculate_x_bounds(self) -> Tuple[float, float]:
        """计算x轴方向的边界"""
        # 可以从环境中获取，这里简化处理
        min_x = min(self.start.x, self.goal.x) - self.search_radius
        max_x = max(self.start.x, self.goal.x) + self.search_radius
        return min_x, max_x

    def _calculate_y_bounds(self) -> Tuple[float, float]:
        """计算y轴方向的边界"""
        # 可以从环境中获取，这里简化处理
        min_y = min(self.start.y, self.goal.y) - self.search_radius
        max_y = max(self.start.y, self.goal.y) + self.search_radius
        return min_y, max_y

    def plan(self) -> List[Tuple[float, float]]:
        """
        执行RRT路径规划

        返回:
            path: 规划得到的路径，由坐标点组成的列表
        """
        for i in range(self.max_iterations):
            # 随机采样
            if np.random.random() < self.goal_sample_rate:
                rnd = Node(self.goal.x, self.goal.y)
            else:
                rnd = self._random_node()

            # 找到最近的节点
            nearest_ind = self._get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]

            # 扩展树
            new_node = self._steer(nearest_node, rnd)

            # 碰撞检测
            if not self._check_collision(new_node, nearest_node):
                continue

            # 添加新节点
            self.node_list.append(new_node)

            # 检查是否到达目标
            if self._is_near_goal(new_node):
                if self._check_segment(new_node, self.goal):
                    self._connect_to_goal(new_node)
                    return self._extract_path()

            # 可视化（如果需要）
            # self._plot_current_tree(i)

        # 达到最大迭代次数但未找到路径
        return []

    def _random_node(self) -> Node:
        """生成随机节点"""
        # 在区域内随机采样
        x = np.random.uniform(0, self.env.width)
        y = np.random.uniform(0, self.env.height)

        # 打印调试信息
        print(f"生成随机节点: ({x:.2f}, {y:.2f})")

        return Node(x, y)

    def _get_nearest_node_index(self, node: Node) -> int:
        """找到距离给定节点最近的节点索引"""
        distances = [
            (node.x - n.x) ** 2 + (node.y - n.y) ** 2
            for n in self.node_list
        ]
        return int(np.argmin(distances))

    def _steer(self, from_node: Node, to_node: Node) -> Node:
        """从一个节点朝向另一个节点扩展一定距离"""
        # 计算方向
        theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)

        # 计算距离
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)

        # 如果距离小于步长，直接使用目标点
        if dist < self.step_size:
            new_x, new_y = to_node.x, to_node.y
        else:
            # 否则在方向上前进一个步长
            new_x = from_node.x + self.step_size * np.cos(theta)
            new_y = from_node.y + self.step_size * np.sin(theta)

        # 创建新节点
        new_node = Node(new_x, new_y)
        new_node.parent = from_node

        # 更新路径
        new_node.path_x = from_node.path_x.copy()
        new_node.path_y = from_node.path_y.copy()
        new_node.path_x.append(new_x)
        new_node.path_y.append(new_y)

        # 更新代价
        new_node.cost = from_node.cost + self.step_size

        print(
            f"扩展节点: 从 ({from_node.x:.2f}, {from_node.y:.2f}) 到 ({new_x:.2f}, {new_y:.2f})")

        return new_node

    def _check_collision(self, node1: Node, node2: Node) -> bool:
        """检查两个节点之间的路径是否无碰撞"""
        # 实际实现中，应该调用环境的碰撞检测功能
        # 这里假设环境有一个check_segment方法
        result = self._check_segment(node1, node2)
        print(
            f"碰撞检测: 从 ({node1.x:.2f}, {node1.y:.2f}) 到 ({node2.x:.2f}, {node2.y:.2f}) - {'无碰撞' if result else '有碰撞'}")
        return result

    def _check_segment(self, node1: Node, node2: Node) -> bool:
        """检查两点之间的线段是否无碰撞，考虑车辆尺寸"""
        # 调用环境的碰撞检测，传入车辆尺寸参数
        print(
            f"DEBUG RRT: 检查线段 ({node1.x:.2f}, {node1.y:.2f}) -> ({node2.x:.2f}, {node2.y:.2f})")
        result = not self.env.check_segment_collision(
            (node1.x, node1.y),
            (node2.x, node2.y),
            self.vehicle_width,
            self.vehicle_length
        )
        print(f"DEBUG RRT: 线段碰撞检测结果: {'无碰撞' if result else '有碰撞'}")
        return result

    def _is_near_goal(self, node: Node) -> bool:
        """检查节点是否靠近目标点"""
        dist = np.hypot(node.x - self.goal.x, node.y - self.goal.y)
        # 增加容忍度，使用更大的阈值
        return dist < self.step_size * 5.0  # 原来是 self.step_size

    def _connect_to_goal(self, node: Node) -> None:
        """将节点连接到目标点"""
        self.goal.parent = node
        self.goal.path_x = node.path_x.copy()
        self.goal.path_y = node.path_y.copy()
        self.goal.path_x.append(self.goal.x)
        self.goal.path_y.append(self.goal.y)
        self.goal.cost = node.cost + np.hypot(
            self.goal.x - node.x,
            self.goal.y - node.y
        )
        self.node_list.append(self.goal)

    def _extract_path(self) -> List[Tuple[float, float]]:
        """提取路径坐标"""
        path = []
        node = self.goal

        while node.parent:
            path.append((node.x, node.y))
            node = node.parent

        path.append((self.start.x, self.start.y))
        path.reverse()

        self.path = path
        return path

    def _plot_current_tree(self, iteration: int) -> None:
        """可视化当前搜索树（用于调试和展示）"""
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None]
        )

        # 绘制障碍物（如果环境提供了这个功能）
        if hasattr(self.env, 'plot_obstacles'):
            self.env.plot_obstacles(plt)

        # 绘制起点和终点
        plt.plot(self.start.x, self.start.y, "go", markersize=10)
        plt.plot(self.goal.x, self.goal.y, "ro", markersize=10)

        # 绘制搜索树
        for node in self.node_list:
            if node.parent:
                plt.plot(
                    [node.x, node.parent.x],
                    [node.y, node.parent.y],
                    "-g"
                )

        # 设置坐标轴范围
        plt.axis((self.min_x, self.max_x, self.min_y, self.max_y))
        plt.grid(True)
        plt.title(f"RRT 搜索树 (迭代: {iteration})")
        plt.pause(0.01)

    def plot_path(self) -> None:
        """绘制最终路径"""
        plt.figure(figsize=(10, 8))

        # 绘制障碍物
        if hasattr(self.env, 'plot_obstacles'):
            self.env.plot_obstacles(plt)

        # 绘制起点和终点
        plt.plot(self.start.x, self.start.y, "go", markersize=10, label="起点")
        plt.plot(self.goal.x, self.goal.y, "ro", markersize=10, label="终点")

        # 绘制搜索树
        for node in self.node_list:
            if node.parent:
                plt.plot(
                    [node.x, node.parent.x],
                    [node.y, node.parent.y],
                    "-g",
                    alpha=0.3
                )

        # 绘制最终路径
        if self.path:
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            plt.plot(path_x, path_y, '-b', linewidth=3, label="规划路径")

        plt.axis((self.min_x, self.max_x, self.min_y, self.max_y))
        plt.grid(True)
        plt.title("RRT路径规划结果")
        plt.legend()
        plt.show()

    def check_line_collision(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> bool:
        """检查线段是否与障碍物碰撞"""
        raise NotImplementedError("子类必须实现此方法")


if __name__ == "__main__":
    # 简单测试代码
    from simulation.environment import Environment

    # 创建一个简单的环境
    env = Environment(width=100, height=100)

    # 添加一些障碍物
    for _ in range(10):
        env.add_obstacle(
            np.random.uniform(10, 90),
            np.random.uniform(10, 90),
            radius=5.0
        )

    # 创建RRT规划器
    rrt = RRT(
        start=(10, 10),
        goal=(90, 90),
        env=env,
        step_size=5.0,
        max_iterations=1000,
        goal_sample_rate=0.1
    )

    # 规划路径
    path = rrt.plan()

    if path:
        print(f"找到路径，包含 {len(path)} 个节点")
        rrt.plot_path()
    else:
        print("未能找到路径")
