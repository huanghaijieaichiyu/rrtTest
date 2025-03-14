#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Theta*算法实现

Theta*是A*算法的变体，它通过考虑任意角度的路径来生成更平滑的路径。
它允许父节点不必是相邻节点，从而可以生成更自然的路径。
"""

from typing import List, Tuple, Set, Optional
import numpy as np
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """Theta*搜索节点"""
    f_score: float = field(compare=True)  # f = g + h
    g_score: float = field(compare=False)  # 起点到当前节点的实际代价
    x: float = field(compare=False)
    y: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)
    path_x: List[float] = field(default_factory=list, compare=False)
    path_y: List[float] = field(default_factory=list, compare=False)


class ThetaStar:
    """
    Theta*算法实现类
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env,
        resolution: float = 1.0,
        diagonal_movement: bool = True,
        weight: float = 1.0,  # 启发式权重
        vehicle_width: float = 2.0,  # 车辆宽度
        vehicle_length: float = 4.0  # 车辆长度
    ):
        """
        初始化Theta*规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            resolution: 搜索分辨率，影响网格大小
            diagonal_movement: 是否允许对角线移动
            weight: 启发式权重
            vehicle_width: 车辆宽度(米)
            vehicle_length: 车辆长度(米)
        """
        self.start = Node(0, 0, start[0], start[1])
        self.goal = Node(0, 0, goal[0], goal[1])
        self.env = env
        self.resolution = resolution
        self.diagonal_movement = diagonal_movement
        self.weight = weight
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length

        # 计算区域边界
        self.min_x = 0
        self.max_x = env.width
        self.min_y = 0
        self.max_y = env.height

        # 将起点加入路径
        self.start.path_x.append(self.start.x)
        self.start.path_y.append(self.start.y)

        # 记录搜索统计信息
        self.nodes_visited = 0
        self.node_list = []

    def _h_cost(self, node: Node) -> float:
        """
        计算启发式代价（节点到目标的估计代价）

        参数:
            node: 当前节点

        返回:
            启发式代价
        """
        return self.weight * np.hypot(
            self.goal.x - node.x,
            self.goal.y - node.y
        )

    def _line_of_sight(self, node1: Node, node2: Node) -> bool:
        """
        检查两个节点之间是否有直线视线（无障碍物）

        参数:
            node1: 起始节点
            node2: 目标节点

        返回:
            是否有直线视线
        """
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x = x1
        y = y1
        n = int(1 + dx + dy)  # Convert to integer for range
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if self.env.check_collision((x, y), self.vehicle_width, self.vehicle_length):
                return False

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return True

    def _get_neighbors(self, node: Node) -> List[Node]:
        """
        获取节点的邻居节点

        参数:
            node: 当前节点

        返回:
            邻居节点列表
        """
        neighbors = []
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1)  # 上下左右
        ]

        if self.diagonal_movement:
            directions.extend([
                (1, 1), (-1, 1), (-1, -1), (1, -1)  # 对角线
            ])

        for dx, dy in directions:
            new_x = node.x + dx * self.resolution
            new_y = node.y + dy * self.resolution

            # 检查是否在环境范围内
            if not (self.min_x <= new_x <= self.max_x and
                    self.min_y <= new_y <= self.max_y):
                continue

            # 检查是否碰撞
            if self.env.check_collision((new_x, new_y), self.vehicle_width, self.vehicle_length):
                continue

            # 创建新节点
            neighbor = Node(
                f_score=0,  # 暂时设为0，后面再计算
                g_score=float('inf'),
                x=new_x,
                y=new_y
            )

            neighbors.append(neighbor)

        return neighbors

    def _update_vertex(self, current: Node, neighbor: Node):
        """
        更新顶点（Theta*的核心操作）

        参数:
            current: 当前节点
            neighbor: 邻居节点
        """
        # 检查是否可以通过current的父节点直接到达neighbor
        if (current.parent is not None and
                self._line_of_sight(current.parent, neighbor)):
            # 计算通过父节点到达neighbor的代价
            new_g_score = (current.parent.g_score +
                           np.hypot(neighbor.x - current.parent.x,
                                    neighbor.y - current.parent.y))

            if new_g_score < neighbor.g_score:
                neighbor.g_score = new_g_score
                neighbor.parent = current.parent
                neighbor.path_x = current.parent.path_x.copy()
                neighbor.path_y = current.parent.path_y.copy()
                neighbor.path_x.append(neighbor.x)
                neighbor.path_y.append(neighbor.y)
                neighbor.f_score = neighbor.g_score + self._h_cost(neighbor)
        else:
            # 如果没有直线视线，使用传统的A*更新方式
            new_g_score = (current.g_score +
                           np.hypot(neighbor.x - current.x,
                                    neighbor.y - current.y))

            if new_g_score < neighbor.g_score:
                neighbor.g_score = new_g_score
                neighbor.parent = current
                neighbor.path_x = current.path_x.copy()
                neighbor.path_y = current.path_y.copy()
                neighbor.path_x.append(neighbor.x)
                neighbor.path_y.append(neighbor.y)
                neighbor.f_score = neighbor.g_score + self._h_cost(neighbor)

    def _is_goal(self, node: Node) -> bool:
        """
        检查节点是否为目标节点

        参数:
            node: 当前节点

        返回:
            是否为目标节点
        """
        dist = np.hypot(node.x - self.goal.x, node.y - self.goal.y)
        return dist <= self.resolution

    def _node_to_grid_key(self, x: float, y: float) -> Tuple[int, int]:
        """
        将连续坐标转换为网格索引

        参数:
            x: x坐标
            y: y坐标

        返回:
            网格索引 (grid_x, grid_y)
        """
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return (grid_x, grid_y)

    def plan(self) -> List[Tuple[float, float]]:
        """
        执行Theta*路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        # 初始化优先队列
        open_set = []
        heapq.heappush(open_set, self.start)

        # 记录已经访问的节点
        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            # 取出f值最小的节点
            current = heapq.heappop(open_set)
            self.nodes_visited += 1
            self.node_list.append(current)

            # 检查是否到达目标
            if self._is_goal(current):
                self.goal.parent = current.parent
                self.goal.path_x = current.path_x
                self.goal.path_y = current.path_y
                return [(x, y) for x, y in zip(current.path_x, current.path_y)]

            # 将当前节点加入closed set
            grid_key = self._node_to_grid_key(current.x, current.y)
            if grid_key in closed_set:
                continue
            closed_set.add(grid_key)

            # 遍历邻居节点
            for neighbor in self._get_neighbors(current):
                neighbor_key = self._node_to_grid_key(neighbor.x, neighbor.y)

                # 如果邻居已经在closed set中，跳过
                if neighbor_key in closed_set:
                    continue

                # 更新邻居节点
                self._update_vertex(current, neighbor)

                # 将邻居加入open set
                heapq.heappush(open_set, neighbor)

        # 如果没有找到路径，返回空列表
        return []

    def post_process_path(
        self,
        path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        后处理路径，移除不必要的中间点

        参数:
            path: 原始路径

        返回:
            优化后的路径
        """
        if len(path) <= 2:
            return path

        # 使用射线投射法优化路径
        optimized_path = [path[0]]
        current_point = path[0]
        i = 1

        while i < len(path):
            # 寻找可以直接到达的最远点
            for j in range(len(path) - 1, i - 1, -1):
                if self._line_of_sight(
                    Node(0, 0, current_point[0], current_point[1]),
                    Node(0, 0, path[j][0], path[j][1])
                ):
                    optimized_path.append(path[j])
                    current_point = path[j]
                    i = j + 1
                    break
            else:
                i += 1

        return optimized_path
