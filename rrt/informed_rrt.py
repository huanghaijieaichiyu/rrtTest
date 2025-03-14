#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Informed RRT* (Informed Rapidly-exploring Random Tree Star) 算法实现

Informed RRT*是RRT*算法的改进版本，通过使用启发式信息来引导采样，
从而加快收敛速度并提高路径质量。主要改进包括：
1. 使用椭圆采样区域
2. 基于当前最优解限制采样空间
3. 动态调整采样策略
"""

from typing import List, Tuple
import numpy as np

from .rrt_star import RRTStar, Node


class InformedRRTStar(RRTStar):
    """
    Informed RRT*算法实现类

    继承自RRT*类，添加了启发式采样功能。
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
        rewire_factor: float = 1.5,
        focus_factor: float = 1.0,  # 椭圆焦点缩放因子
        vehicle_width: float = 2.0,  # 车辆宽度
        vehicle_length: float = 4.0  # 车辆长度
    ):
        """
        初始化Informed RRT*规划器

        参数:
            start: 起点坐标 (x, y)
            goal: 目标点坐标 (x, y)
            env: 环境对象，用于碰撞检测等
            step_size: 每次扩展的步长
            max_iterations: 最大迭代次数
            goal_sample_rate: 采样目标点的概率
            search_radius: 搜索半径
            rewire_factor: 重连接搜索半径因子
            focus_factor: 椭圆焦点缩放因子，控制采样区域大小
            vehicle_width: 车辆宽度
            vehicle_length: 车辆长度
        """
        super().__init__(
            start, goal, env, step_size,
            max_iterations, goal_sample_rate, search_radius, rewire_factor,
            vehicle_width, vehicle_length
        )
        self.focus_factor = focus_factor
        self.c_best = float('inf')  # 当前最优路径代价
        self.c_min = np.hypot(  # 理论最小路径代价
            self.goal.x - self.start.x,
            self.goal.y - self.start.y
        )

    def _calculate_ellipse_parameters(self) -> Tuple[float, float, float, float, float]:
        """
        计算采样椭圆的参数

        返回:
            center_x: 椭圆中心x坐标
            center_y: 椭圆中心y坐标
            angle: 椭圆旋转角度
            a: 椭圆长轴长度
            b: 椭圆短轴长度
        """
        # 计算椭圆中心
        center_x = (self.start.x + self.goal.x) / 2.0
        center_y = (self.start.y + self.goal.y) / 2.0

        # 计算椭圆旋转角度
        angle = np.arctan2(
            self.goal.y - self.start.y,
            self.goal.x - self.start.x
        )

        # 计算长轴和短轴
        c = self.c_min / 2.0  # 焦点到中心的距离
        if self.c_best < float('inf'):
            a = self.c_best / 2.0  # 长轴长度的一半
            b = np.sqrt(abs(a ** 2 - c ** 2))  # 短轴长度的一半
            # 应用缩放因子
            a *= self.focus_factor
            b *= self.focus_factor
        else:
            # 如果还没有找到路径，使用一个大椭圆
            a = self.search_radius
            b = self.search_radius

        return center_x, center_y, angle, a, b

    def _random_node(self) -> Node:
        """
        生成随机节点，使用椭圆采样

        返回:
            随机节点
        """
        if self.c_best == float('inf'):
            # 如果还没有找到路径，使用普通RRT*的采样方法
            return super()._random_node()

        # 计算椭圆参数
        center_x, center_y, angle, a, b = self._calculate_ellipse_parameters()

        while True:
            # 在单位圆内采样
            r = np.random.uniform(0, 1)
            theta = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # 转换到椭圆坐标系
            x_ellipse = x * a
            y_ellipse = y * b

            # 旋转并平移到实际位置
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            x_final = (x_ellipse * cos_angle -
                       y_ellipse * sin_angle) + center_x
            y_final = (x_ellipse * sin_angle +
                       y_ellipse * cos_angle) + center_y

            # 检查是否在有效范围内
            if (self.min_x <= x_final <= self.max_x and
                    self.min_y <= y_final <= self.max_y):
                return Node(x_final, y_final)

    def _update_best_cost(self) -> None:
        """更新当前最优路径代价"""
        if self.path:
            path_cost = sum(
                np.hypot(
                    self.path[i][0] - self.path[i-1][0],
                    self.path[i][1] - self.path[i-1][1]
                )
                for i in range(1, len(self.path))
            )
            self.c_best = min(self.c_best, path_cost)

    def plan(self) -> List[Tuple[float, float]]:
        """
        执行Informed RRT*路径规划

        返回:
            规划得到的路径，由坐标点组成的列表
        """
        # 先使用RRT*找到初始路径
        path = super().plan()
        if path:
            self._update_best_cost()

        # 继续优化路径
        remaining_iterations = self.max_iterations
        while remaining_iterations > 0:
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

                    # 检查是否到达目标并更新最优路径
                    if self._is_near_goal(new_node):
                        if self._check_segment(new_node, self.goal):
                            self._connect_to_goal(new_node)
                            path = self._extract_path()
                            self._update_best_cost()

            remaining_iterations -= 1

        return self.path if self.path else []


if __name__ == "__main__":
    # 测试代码
    from simulation.environment import Environment
    import matplotlib.pyplot as plt

    # 创建环境
    env = Environment(width=100, height=100)

    # 添加一些障碍物
    for _ in range(10):
        env.add_obstacle(
            np.random.uniform(10, 90),
            np.random.uniform(10, 90),
            radius=5.0
        )

    # 创建Informed RRT*规划器
    informed_rrt = InformedRRTStar(
        start=(10, 10),
        goal=(90, 90),
        env=env,
        step_size=5.0,
        max_iterations=1000,
        goal_sample_rate=0.1,
        focus_factor=1.2,
        vehicle_width=2.0,
        vehicle_length=4.0
    )

    # 规划路径
    path = informed_rrt.plan()

    if path:
        print(f"找到路径，包含 {len(path)} 个节点")
        print(f"路径代价: {informed_rrt.c_best:.2f}")
        print(f"理论最小代价: {informed_rrt.c_min:.2f}")
        # 可视化结果
        informed_rrt.plot_path()
    else:
        print("未能找到路径")
