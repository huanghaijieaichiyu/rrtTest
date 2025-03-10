#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A* 算法实现

A*是一种启发式搜索算法，通常用于路径规划。
它结合了Dijkstra算法和最佳优先搜索算法的特点，使用启发式函数估计从当前节点到目标的代价。
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """A*搜索节点"""
    f_score: float = field(compare=True)  # f = g + h
    g_score: float = field(compare=False)  # 起点到当前节点的实际代价
    x: float = field(compare=False)
    y: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)

    # 用于路径存储
    path_x: List[float] = field(default_factory=list, compare=False)
    path_y: List[float] = field(default_factory=list, compare=False)


class AStar:
    """
    A*算法实现类

    用于在给定环境中规划最短路径。
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env,
        resolution: float = 1.0,
        diagonal_movement: bool = True,
        weight: float = 1.0  # 启发式权重，值越大越倾向于贪心搜索
    ):
        """
        初始化A*规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            resolution: 搜索分辨率，影响网格大小
            diagonal_movement: 是否允许对角线移动
            weight: 启发式权重
        """
        self.start = Node(0, 0, start[0], start[1])
        self.goal = Node(0, 0, goal[0], goal[1])
        self.env = env
        self.resolution = resolution
        self.diagonal_movement = diagonal_movement
        self.weight = weight

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
        self.node_list = []  # 用于可视化的节点列表

    def _h_cost(self, node: Node) -> float:
        """
        计算启发式代价（节点到目标的估计代价）

        参数:
            node: 当前节点

        返回:
            启发式代价
        """
        # 使用欧几里得距离作为启发式函数
        return np.hypot(self.goal.x - node.x, self.goal.y - node.y)

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
            if self.env.check_collision((new_x, new_y)):
                continue

            # 检查路径是否碰撞（只有对角线移动需要）
            if dx != 0 and dy != 0:
                if (self.env.check_collision((node.x + dx * self.resolution, node.y)) or
                        self.env.check_collision((node.x, node.y + dy * self.resolution))):
                    continue

            # 计算移动代价
            if dx != 0 and dy != 0:
                move_cost = np.sqrt(2) * self.resolution  # 对角线移动
            else:
                move_cost = self.resolution  # 正交移动

            g_score = node.g_score + move_cost

            # 创建新节点
            neighbor = Node(
                f_score=0,  # 暂时设为0，后面再计算
                g_score=g_score,
                x=new_x,
                y=new_y,
                parent=node,
                path_x=node.path_x.copy(),
                path_y=node.path_y.copy()
            )

            # 添加到路径
            neighbor.path_x.append(new_x)
            neighbor.path_y.append(new_y)

            # 计算f值
            h_cost = self._h_cost(neighbor)
            neighbor.f_score = g_score + self.weight * h_cost

            neighbors.append(neighbor)

        return neighbors

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
        执行A*路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        # 初始化优先队列（open set）
        open_set = []
        heapq.heappush(open_set, self.start)

        # 记录已经访问的节点（closed set）
        closed_set: Set[Tuple[int, int]] = set()

        # 记录当前最优g值
        g_scores: Dict[Tuple[int, int], float] = {
            self._node_to_grid_key(self.start.x, self.start.y): 0
        }

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

                # 检查是否找到更好的路径
                if (neighbor_key not in g_scores or
                        neighbor.g_score < g_scores[neighbor_key]):

                    g_scores[neighbor_key] = neighbor.g_score
                    heapq.heappush(open_set, neighbor)

        # 如果没有找到路径，返回空列表
        return []
