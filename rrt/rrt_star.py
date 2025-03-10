#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RRT* (Rapidly-exploring Random Tree Star) 算法实现

RRT*是RRT算法的改进版本，通过重新连接和优化路径代价来实现渐进最优性。
主要改进包括：
1. 在给定半径内寻找最优父节点
2. 重新连接（rewiring）以优化现有路径
"""

from typing import List, Tuple
import numpy as np

from .rrt_base import RRT, Node


class RRTStar(RRT):
    """
    RRT* 算法实现类

    继承自基础RRT类，添加了路径优化功能。
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
        rewire_factor: float = 1.5  # 重连接搜索半径因子
    ):
        """
        初始化RRT*规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            step_size: 每次扩展的步长
            max_iterations: 最大迭代次数
            goal_sample_rate: 采样目标点的概率
            search_radius: 搜索半径
            rewire_factor: 重连接搜索半径因子
        """
        super().__init__(
            start, goal, env, step_size,
            max_iterations, goal_sample_rate, search_radius
        )
        self.rewire_factor = rewire_factor

    def _find_near_nodes(self, node: Node) -> List[Node]:
        """
        找到给定节点附近的所有节点

        参数:
            node: 目标节点

        返回:
            附近节点列表
        """
        n = len(self.node_list)
        r = self.step_size * self.rewire_factor * np.sqrt(np.log(n) / n)
        r = min(r, self.search_radius)  # 限制搜索半径

        dist_list = [
            (node.x - n.x) ** 2 + (node.y - n.y) ** 2
            for n in self.node_list
        ]
        near_inds = [i for i, d in enumerate(dist_list) if d <= r ** 2]
        return [self.node_list[i] for i in near_inds]

    def _choose_parent(self, new_node: Node, near_nodes: List[Node]) -> Node:
        """
        为新节点选择最优父节点

        参数:
            new_node: 新节点
            near_nodes: 附近节点列表

        返回:
            选择的父节点
        """
        if not near_nodes:
            return new_node

        costs = []
        for near_node in near_nodes:
            # 计算从near_node到new_node的方向和距离
            d = np.hypot(new_node.x - near_node.x, new_node.y - near_node.y)

            # 如果距离大于步长，跳过
            if d > self.step_size:
                costs.append(float('inf'))
                continue

            # 检查连接是否可行
            if not self._check_collision(near_node, new_node):
                costs.append(float('inf'))
                continue

            # 计算代价
            costs.append(near_node.cost + d)

        # 找到最小代价的节点
        min_cost = min(costs)
        if min_cost == float('inf'):
            return new_node

        min_ind = costs.index(min_cost)
        min_node = near_nodes[min_ind]

        # 更新新节点
        new_node.parent = min_node
        new_node.cost = min_cost

        return new_node

    def _rewire(self, new_node: Node, near_nodes: List[Node]) -> None:
        """
        尝试通过新节点重新连接附近节点以优化路径

        参数:
            new_node: 新节点
            near_nodes: 附近节点列表
        """
        for near_node in near_nodes:
            # 计算如果通过新节点到达near_node的代价
            d = np.hypot(near_node.x - new_node.x, near_node.y - new_node.y)

            # 如果距离大于步长，跳过
            if d > self.step_size:
                continue

            new_cost = new_node.cost + d

            # 如果新路径代价更低，尝试重新连接
            if new_cost < near_node.cost:
                if self._check_collision(new_node, near_node):
                    # 更新父节点和代价
                    near_node.parent = new_node
                    near_node.cost = new_cost
                    # 更新路径
                    near_node.path_x = new_node.path_x.copy()
                    near_node.path_y = new_node.path_y.copy()
                    near_node.path_x.append(near_node.x)
                    near_node.path_y.append(near_node.y)

    def _calc_new_cost(self, from_node: Node, to_node: Node) -> float:
        """
        计算从一个节点到另一个节点的代价

        参数:
            from_node: 起始节点
            to_node: 目标节点

        返回:
            路径代价
        """
        d = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        return from_node.cost + d

    def plan(self) -> List[Tuple[float, float]]:
        """
        执行RRT*路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        # 增加搜索半径，提高成功率
        self.search_radius = max(
            self.search_radius,
            np.hypot(self.goal.x - self.start.x,
                     self.goal.y - self.start.y) * 0.5
        )

        # 增加重连接因子，提高路径质量
        self.rewire_factor = max(self.rewire_factor, 2.0)

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

            # 如果路径无碰撞
            if self._check_collision(nearest_node, new_node):
                # 找到附近的节点
                near_nodes = self._find_near_nodes(new_node)
                # 选择最优父节点
                new_node = self._choose_parent(new_node, near_nodes)

                if new_node.parent is not None:
                    # 添加新节点
                    self.node_list.append(new_node)
                    # 尝试重新连接
                    self._rewire(new_node, near_nodes)

                    # 检查是否到达目标
                    if self._is_near_goal(new_node):
                        if self._check_segment(new_node, self.goal):
                            self._connect_to_goal(new_node)
                            return self._extract_path()

            # 每隔一定迭代次数，尝试直接连接到目标
            if i % 100 == 0:
                # 找到离目标最近的节点
                closest_ind = self._get_nearest_node_index(self.goal)
                closest_node = self.node_list[closest_ind]

                # 尝试直接连接
                if self._is_near_goal(closest_node) and self._check_segment(closest_node, self.goal):
                    self._connect_to_goal(closest_node)
                    return self._extract_path()

        # 达到最大迭代次数但未找到路径
        # 尝试连接最近的节点到目标
        closest_ind = self._get_nearest_node_index(self.goal)
        closest_node = self.node_list[closest_ind]

        if self._check_segment(closest_node, self.goal):
            self._connect_to_goal(closest_node)
            return self._extract_path()

        return []


if __name__ == "__main__":
    # 测试代码
    from simulation.environment import Environment

    # 创建环境
    env = Environment(width=100, height=100)

    # 添加一些障碍物
    for _ in range(10):
        env.add_obstacle(
            np.random.uniform(10, 90),
            np.random.uniform(10, 90),
            radius=5.0
        )

    # 创建RRT*规划器
    rrt_star = RRTStar(
        start=(10, 10),
        goal=(90, 90),
        env=env,
        step_size=5.0,
        max_iterations=1000,
        goal_sample_rate=0.1
    )

    # 规划路径
    path = rrt_star.plan()

    if path:
        print(f"找到路径，包含 {len(path)} 个节点")
        # 可视化结果
        rrt_star.plot_path()
    else:
        print("未能找到路径")
