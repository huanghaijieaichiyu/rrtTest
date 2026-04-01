#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
路径平滑处理模块

包含以下功能:
1. TED (Trajectory Error Detection) - 轨迹误差检测
2. Quintic polynomial interpolation - 五次多项式插值
3. Kalman filter - 卡尔曼滤波
"""

import numpy as np
from typing import List, Tuple
# 导入独立的碰撞检测函数
from simulation.pygame_simulator import check_segment_collision


class PathSmoother:
    """路径平滑处理类"""

    def __init__(self, vehicle_width: float, vehicle_length: float):
        """
        初始化路径平滑器

        参数:
            vehicle_width: 车辆宽度
            vehicle_length: 车辆长度
        """
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length

    def ted_detection(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        轨迹误差检测(TED)

        参数:
            path: 原始路径

        返回:
            经过误差检测后的路径
        """
        if len(path) < 3:
            return path

        # 将路径转换为numpy数组以便计算
        path_array = np.array(path)

        # 计算相邻点之间的向量
        vectors = np.diff(path_array, axis=0)

        # 计算向量的角度（弧度）
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])

        # 计算角度变化
        angle_changes = np.diff(angles)

        # 将角度变化限制在 -pi 到 pi 之间
        angle_changes = np.where(angle_changes > np.pi, angle_changes - 2 * np.pi, angle_changes)
        angle_changes = np.where(angle_changes < -np.pi, angle_changes + 2 * np.pi, angle_changes)

        # 计算曲率（角度变化/路径长度）
        segment_lengths = np.sqrt(np.sum(vectors**2, axis=1))
        valid_lengths = np.maximum(segment_lengths[1:], 1e-6)
        curvatures = np.abs(angle_changes) / valid_lengths
        curvatures = curvatures[np.isfinite(curvatures)]
        if curvatures.size == 0:
            return path

        # 设置曲率阈值（可根据实际情况调整）
        curvature_threshold = np.mean(curvatures) + 2 * np.std(curvatures)

        # 找出曲率正常的点
        good_points = [0]  # 第一个点总是保留
        for i in range(1, len(path) - 1):
            if i - 1 < len(curvatures) and curvatures[i - 1] < curvature_threshold:
                good_points.append(i)
        good_points.append(len(path) - 1)  # 最后一个点总是保留

        # 返回筛选后的路径
        return [path[i] for i in good_points]

    def quintic_polynomial_interpolation(self,
                                         path: List[Tuple[float, float]],
                                         num_points: int = 10) -> List[Tuple[float, float]]:
        """
        五次多项式插值

        参数:
            path: 原始路径
            num_points: 每两个点之间插入的点数

        返回:
            插值后的平滑路径
        """
        if len(path) < 2:
            return path

        # 将路径点分解为x和y坐标
        x_coords = np.array([p[0] for p in path])
        y_coords = np.array([p[1] for p in path])

        # 创建参数t，表示路径上的相对位置
        t = np.linspace(0, len(path) - 1, len(path))

        # 为插值创建更密集的t值
        t_new = np.linspace(0, len(path) - 1, (len(path) - 1) * num_points + 1)

        # 计算五次多项式系数
        x_coeffs = np.polyfit(t, x_coords, 5)
        y_coeffs = np.polyfit(t, y_coords, 5)

        # 使用多项式系数计算新的路径点
        x_new = np.polyval(x_coeffs, t_new)
        y_new = np.polyval(y_coeffs, t_new)

        # 将结果转换回路径点列表
        smoothed_path = list(zip(x_new.tolist(), y_new.tolist()))

        return smoothed_path

    def kalman_filter_smoothing(self,
                                path: List[Tuple[float, float]],
                                process_noise: float = 0.1,
                                measurement_noise: float = 0.1) -> List[Tuple[float, float]]:
        """
        卡尔曼滤波平滑

        参数:
            path: 原始路径
            process_noise: 过程噪声
            measurement_noise: 测量噪声

        返回:
            经过卡尔曼滤波平滑后的路径
        """
        if len(path) < 2:
            return path

        # 将路径转换为numpy数组
        path_array = np.array(path)

        # 状态向量 [x, y, dx, dy]
        state = np.zeros(4)
        state[:2] = path_array[0]

        # 状态转移矩阵
        F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

        # 测量矩阵
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # 过程噪声协方差
        Q = np.eye(4) * process_noise

        # 测量噪声协方差
        R = np.eye(2) * measurement_noise

        # 状态估计协方差
        P = np.eye(4)

        # 存储滤波后的结果
        filtered_path = [path[0]]

        for point in path_array[1:]:
            # 预测步骤
            state = F @ state
            P = F @ P @ F.T + Q

            # 更新步骤
            measurement = point
            y = measurement - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            state = state + K @ y
            P = (np.eye(4) - K @ H) @ P

            # 保存滤波后的位置
            filtered_path.append((state[0], state[1]))

        return filtered_path

    def shortcut_path(self, path: List[Tuple[float, float]], env) -> List[Tuple[float, float]]:
        """
        路径快捷化：尝试连接非相邻节点以缩短路径。

        参数:
            path: 待快捷化的路径
            env: 环境对象，用于碰撞检测

        返回:
            快捷化后的路径
        """
        if len(path) < 3:
            return path

        # # 警告：当前未使用精确的碰撞检测 (注释掉，因为我们现在使用了)
        # print("警告: shortcut_path当前未使用精确碰撞检测。生成的路径可能不安全。")
        # def dummy_collision_check(p1, p2, env, vehicle_length, vehicle_width):
        #     """虚拟碰撞检测，始终返回False。"""
        #     # TODO: 实现或导入一个精确的基于车辆模型的碰撞检测函数
        #     # 例如: return check_vehicle_path_collision(p1, p2, env, vehicle_length, vehicle_width)
        #     return False

        shortcutted_path = [path[0]]  # 结果路径，从起点开始
        current_index = 0

        while current_index < len(path) - 1:
            best_next_index = current_index + 1
            # 从当前点之后的最远点开始尝试连接
            for next_index in range(len(path) - 1, current_index + 1, -1):
                p1 = path[current_index]
                p2 = path[next_index]

                # 使用导入的碰撞检测函数
                is_collision = check_segment_collision(p1, p2, env, self.vehicle_length, self.vehicle_width)

                if not is_collision:
                    # 如果不碰撞，这是当前可达的最远点
                    best_next_index = next_index
                    break  # 找到了最远的无碰撞连接，跳出内层循环

            # 将最佳的下一个节点添加到结果路径
            shortcutted_path.append(path[best_next_index])
            # 更新当前索引
            current_index = best_next_index

        return shortcutted_path

    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        综合平滑处理流程

        1. 先进行轨迹误差检测
        2. 然后进行五次多项式插值
        3. 最后进行卡尔曼滤波平滑

        参数:
            path: 原始路径

        返回:
            经过所有平滑处理后的路径
        """
        path = self.ted_detection(path)
        path = self.quintic_polynomial_interpolation(path)
        path = self.kalman_filter_smoothing(path)
        return path


if __name__ == "__main__":
    # 测试代码
    test_path = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    smoother = PathSmoother(vehicle_width=1.8, vehicle_length=4.5)
    smoothed_path = smoother.smooth_path(test_path)
    print("原始路径:", test_path)
    print("平滑后路径:", smoothed_path)
