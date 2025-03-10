#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
D* Lite算法实现

D* Lite是一种增量搜索算法，适用于动态环境中的路径规划。
它可以在环境发生变化时高效地重新规划路径。
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import heapq
from dataclasses import dataclass, field


@dataclass
class State:
    """D* Lite状态"""
    x: float
    y: float
    g: float = float('inf')  # 从目标到当前状态的代价
    rhs: float = float('inf')  # 右值，用于更新g值
    key: Tuple[float, float] = field(default=(float('inf'), float('inf')))
    parent: Optional['State'] = None
    path_x: List[float] = field(default_factory=list)
    path_y: List[float] = field(default_factory=list)


class DStarLite:
    """
    D* Lite算法实现类
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        env,
        resolution: float = 1.0,
        diagonal_movement: bool = True
    ):
        """
        初始化D* Lite规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            resolution: 搜索分辨率，影响网格大小
            diagonal_movement: 是否允许对角线移动
        """
        self.resolution = resolution
        self.diagonal_movement = diagonal_movement
        self.env = env

        # 计算区域边界
        self.min_x = 0
        self.max_x = env.width
        self.min_y = 0
        self.max_y = env.height

        # 创建起点和终点状态
        self.start = State(start[0], start[1])
        self.goal = State(goal[0], goal[1])
        self.goal.rhs = 0  # 目标点的rhs值为0

        # 初始化路径
        self.start.path_x = [self.start.x]
        self.start.path_y = [self.start.y]

        # 状态字典
        self.states: Dict[Tuple[int, int], State] = {}
        self.states[self._state_to_key(self.start)] = self.start
        self.states[self._state_to_key(self.goal)] = self.goal

        # 优先队列
        self.queue = []

        # 记录搜索统计信息
        self.nodes_visited = 0
        self.node_list = []

        # 启发式函数的权重
        self.heuristic_weight = 1.0

        # 当前机器人位置（用于动态更新）
        self.current = self.start

        # km值（用于处理动态变化）
        self.km = 0

    def _state_to_key(self, state: State) -> Tuple[int, int]:
        """将状态转换为网格索引"""
        return (int(state.x / self.resolution),
                int(state.y / self.resolution))

    def _calculate_key(self, state: State) -> Tuple[float, float]:
        """计算状态的键值"""
        h = self._heuristic(state, self.current)
        return (min(state.g, state.rhs) + h + self.km,
                min(state.g, state.rhs))

    def _heuristic(self, state1: State, state2: State) -> float:
        """计算启发式值（两点间的欧几里得距离）"""
        return self.heuristic_weight * np.hypot(
            state1.x - state2.x,
            state1.y - state2.y
        )

    def _get_neighbors(self, state: State) -> List[State]:
        """获取邻居状态"""
        neighbors = []
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1)  # 上下左右
        ]

        if self.diagonal_movement:
            directions.extend([
                (1, 1), (-1, 1), (-1, -1), (1, -1)  # 对角线
            ])

        for dx, dy in directions:
            new_x = state.x + dx * self.resolution
            new_y = state.y + dy * self.resolution

            # 检查是否在环境范围内
            if not (self.min_x <= new_x <= self.max_x and
                    self.min_y <= new_y <= self.max_y):
                continue

            # 检查是否碰撞
            if self.env.check_collision((new_x, new_y)):
                continue

            # 创建新状态
            new_state = State(new_x, new_y)
            key = self._state_to_key(new_state)

            # 如果状态已存在，使用已有状态
            if key in self.states:
                neighbors.append(self.states[key])
            else:
                self.states[key] = new_state
                neighbors.append(new_state)

        return neighbors

    def _update_state(self, state: State):
        """更新状态"""
        if state != self.goal:
            # 计算rhs值
            state.rhs = float('inf')
            for neighbor in self._get_neighbors(state):
                cost = self._get_cost(state, neighbor)
                state.rhs = min(state.rhs, neighbor.g + cost)

        # 如果状态在队列中，移除它
        key = self._state_to_key(state)
        for i, (_, s) in enumerate(self.queue):
            if self._state_to_key(s) == key:
                self.queue.pop(i)
                break

        # 如果状态不一致，加入队列
        if state.g != state.rhs:
            state.key = self._calculate_key(state)
            heapq.heappush(self.queue, (state.key, state))

    def _get_cost(self, state1: State, state2: State) -> float:
        """计算两个状态之间的代价"""
        # 对角线移动代价为√2，正交移动代价为1
        dx = abs(state1.x - state2.x)
        dy = abs(state1.y - state2.y)
        return np.sqrt(2) * min(dx, dy) + abs(dx - dy)

    def _compute_shortest_path(self):
        """计算最短路径"""
        while (self.queue and
                (self.queue[0][0] < self._calculate_key(self.current) or
                 self.current.rhs != self.current.g)):

            k_old, state = heapq.heappop(self.queue)
            self.nodes_visited += 1
            self.node_list.append(state)

            k_new = self._calculate_key(state)

            if k_old < k_new:
                state.key = k_new
                heapq.heappush(self.queue, (k_new, state))
            elif state.g > state.rhs:
                state.g = state.rhs
                for neighbor in self._get_neighbors(state):
                    self._update_state(neighbor)
            else:
                state.g = float('inf')
                self._update_state(state)
                for neighbor in self._get_neighbors(state):
                    self._update_state(neighbor)

    def update_environment(self, changed_points: List[Tuple[float, float]]):
        """
        更新环境（处理动态变化）

        参数:
            changed_points: 发生变化的点的列表
        """
        self.km += self._heuristic(self.current, self.start)
        self.start = self.current

        # 更新受影响的状态
        for x, y in changed_points:
            state = State(x, y)
            key = self._state_to_key(state)
            if key in self.states:
                state = self.states[key]
                self._update_state(state)
                for neighbor in self._get_neighbors(state):
                    self._update_state(neighbor)

    def plan(self) -> List[Tuple[float, float]]:
        """
        执行路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        # 初始化目标状态
        self._update_state(self.goal)

        # 计算路径
        self._compute_shortest_path()

        # 提取路径
        path = []
        current = self.current
        while current != self.goal:
            if current.g == float('inf'):
                return []  # 无法找到路径

            path.append((current.x, current.y))

            # 选择最佳后继状态
            min_cost = float('inf')
            next_state = None
            for neighbor in self._get_neighbors(current):
                cost = self._get_cost(current, neighbor) + neighbor.g
                if cost < min_cost:
                    min_cost = cost
                    next_state = neighbor

            if next_state is None:
                return []  # 无法找到路径

            current = next_state

        path.append((self.goal.x, self.goal.y))
        return path
