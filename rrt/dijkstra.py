#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dijkstra算法实现

Dijkstra算法是一种用于寻找图形中单源最短路径的算法。
它可以找到一个节点到其他所有节点的最短路径。
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """Dijkstra搜索节点"""
    distance: float = field(compare=True)  # 从起点到当前节点的距离
    x: float = field(compare=False)
    y: float = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)
    path_x: List[float] = field(default_factory=list, compare=False)
    path_y: List[float] = field(default_factory=list, compare=False)


class Dijkstra:
    """
    Dijkstra算法实现类
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env,
        resolution: float = 1.0,
        diagonal_movement: bool = True,
        vehicle_width: float = 2.0,  # 车辆宽度
        vehicle_length: float = 4.0  # 车辆长度
    ):
        """
        初始化Dijkstra规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            resolution: 搜索分辨率，影响网格大小
            diagonal_movement: 是否允许对角线移动
            vehicle_width: 车辆宽度(米)
            vehicle_length: 车辆长度(米)
        """
        self.start = Node(0, start[0], start[1])
        self.goal = Node(float('inf'), goal[0], goal[1])
        self.env = env
        self.resolution = resolution
        self.diagonal_movement = diagonal_movement
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

            # 检查路径是否碰撞（只有对角线移动需要）
            if dx != 0 and dy != 0:
                if (self.env.check_collision((node.x + dx * self.resolution, node.y), self.vehicle_width, self.vehicle_length) or
                        self.env.check_collision((node.x, node.y + dy * self.resolution), self.vehicle_width, self.vehicle_length)):
                    continue

            # 计算移动代价
            if dx != 0 and dy != 0:
                move_cost = np.sqrt(2) * self.resolution  # 对角线移动
            else:
                move_cost = self.resolution  # 正交移动

            distance = node.distance + move_cost

            # 创建新节点
            neighbor = Node(
                distance=distance,
                x=new_x,
                y=new_y,
                parent=node,
                path_x=node.path_x.copy(),
                path_y=node.path_y.copy()
            )

            # 添加到路径
            neighbor.path_x.append(new_x)
            neighbor.path_y.append(new_y)

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
        执行Dijkstra路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        # 初始化优先队列
        open_set = []
        heapq.heappush(open_set, self.start)

        # 记录已经访问的节点
        closed_set: Set[Tuple[int, int]] = set()

        # 记录当前最短距离
        distances: Dict[Tuple[int, int], float] = {
            self._node_to_grid_key(self.start.x, self.start.y): 0
        }

        while open_set:
            # 取出距离最小的节点
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

                # 检查是否找到更短的路径
                if (neighbor_key not in distances or
                        neighbor.distance < distances[neighbor_key]):

                    distances[neighbor_key] = neighbor.distance
                    heapq.heappush(open_set, neighbor)

        # 如果没有找到路径，返回空列表
        return []
