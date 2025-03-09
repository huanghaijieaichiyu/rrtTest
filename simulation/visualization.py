"""
可视化工具模块

提供用于可视化路径规划和仿真结果的工具。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.patches as mpatches

from .environment import Environment


class Visualization:
    """
    可视化工具类

    提供用于可视化路径规划和仿真结果的工具。
    """

    def __init__(self, env: Environment = None):
        """
        初始化可视化工具

        参数:
            env: 环境对象
        """
        self.env = env
        self.fig = None
        self.ax = None

    def set_environment(self, env: Environment) -> None:
        """
        设置环境

        参数:
            env: 环境对象
        """
        self.env = env

    def plot_path(
        self,
        path: List[Tuple[float, float]],
        start: Optional[Tuple[float, float]] = None,
        goal: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_grid: bool = True,
        show_environment: bool = True,
        path_color: str = 'blue',
        path_width: float = 2.0,
        title: str = '规划路径',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制路径

        参数:
            path: 路径点列表
            start: 起点坐标，如果为None则使用路径第一个点
            goal: 终点坐标，如果为None则使用路径最后一个点
            figsize: 图形大小
            show_grid: 是否显示网格
            show_environment: 是否显示环境
            path_color: 路径颜色
            path_width: 路径线宽
            title: 图形标题
            save_path: 图形保存路径，如果为None则不保存

        返回:
            matplotlib图形对象
        """
        if not path:
            print("路径为空，无法绘制")
            return None

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # 获取起点和终点
        if start is None and path:
            start = path[0]
        if goal is None and path:
            goal = path[-1]

        # 绘制环境
        if show_environment and self.env:
            # 绘制边界
            self.ax.plot([0, self.env.width, self.env.width, 0, 0],
                         [0, 0, self.env.height, self.env.height, 0], 'k-')

            # 绘制障碍物
            self.env.plot_obstacles(self.ax)

        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            self.ax.plot(path_x, path_y, '-', color=path_color,
                         linewidth=path_width, label='路径')

            # 绘制路径点
            self.ax.scatter(path_x, path_y, color=path_color, s=10)

        # 绘制起点和终点
        if start:
            self.ax.scatter(start[0], start[1],
                            color='green', s=100, label='起点')
        if goal:
            self.ax.scatter(goal[0], goal[1], color='red', s=100, label='终点')

        # 设置坐标轴
        if self.env:
            self.ax.set_xlim(-5, self.env.width + 5)
            self.ax.set_ylim(-5, self.env.height + 5)
        else:
            # 如果没有环境，根据路径设置坐标轴范围
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            x_min, x_max = min(path_x), max(path_x)
            y_min, y_max = min(path_y), max(path_y)
            margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 5)
            self.ax.set_xlim(x_min - margin, x_max + margin)
            self.ax.set_ylim(y_min - margin, y_max + margin)

        self.ax.set_aspect('equal')
        if show_grid:
            self.ax.grid(True)
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return self.fig

    def plot_multi_paths(
        self,
        paths: List[List[Tuple[float, float]]],
        labels: List[str],
        colors: List[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_grid: bool = True,
        show_environment: bool = True,
        title: str = '多路径对比',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制多条路径进行对比

        参数:
            paths: 多条路径列表
            labels: 路径标签列表
            colors: 路径颜色列表
            figsize: 图形大小
            show_grid: 是否显示网格
            show_environment: 是否显示环境
            title: 图形标题
            save_path: 图形保存路径，如果为None则不保存

        返回:
            matplotlib图形对象
        """
        if not paths:
            print("路径为空，无法绘制")
            return None

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # 如果未提供颜色，使用默认颜色循环
        if colors is None:
            colors = plt.cm.tab10.colors[:len(paths)]
        elif len(colors) < len(paths):
            # 如果颜色不够，循环使用
            colors = colors * (len(paths) // len(colors) + 1)
            colors = colors[:len(paths)]

        # 绘制环境
        if show_environment and self.env:
            # 绘制边界
            self.ax.plot([0, self.env.width, self.env.width, 0, 0],
                         [0, 0, self.env.height, self.env.height, 0], 'k-')

            # 绘制障碍物
            self.env.plot_obstacles(self.ax)

        # 绘制所有路径
        for i, path in enumerate(paths):
            if not path:
                continue

            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]

            label = labels[i] if i < len(labels) else f"路径 {i+1}"
            color = colors[i % len(colors)]

            self.ax.plot(path_x, path_y, '-', color=color,
                         linewidth=2.0, label=label)

        # 设置坐标轴
        if self.env:
            self.ax.set_xlim(-5, self.env.width + 5)
            self.ax.set_ylim(-5, self.env.height + 5)
        else:
            # 如果没有环境，根据所有路径设置坐标轴范围
            all_x = [p[0] for path in paths for p in path if path]
            all_y = [p[1] for path in paths for p in path if path]
            if all_x and all_y:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)
                margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 5)
                self.ax.set_xlim(x_min - margin, x_max + margin)
                self.ax.set_ylim(y_min - margin, y_max + margin)

        self.ax.set_aspect('equal')
        if show_grid:
            self.ax.grid(True)
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return self.fig

    def plot_search_tree(
        self,
        nodes: List[Any],
        path: List[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_grid: bool = True,
        show_environment: bool = True,
        tree_color: str = 'green',
        tree_alpha: float = 0.3,
        path_color: str = 'blue',
        title: str = '搜索树',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制搜索树

        参数:
            nodes: 搜索树节点列表
            path: 最终路径
            figsize: 图形大小
            show_grid: 是否显示网格
            show_environment: 是否显示环境
            tree_color: 搜索树颜色
            tree_alpha: 搜索树透明度
            path_color: 路径颜色
            title: 图形标题
            save_path: 图形保存路径，如果为None则不保存

        返回:
            matplotlib图形对象
        """
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # 绘制环境
        if show_environment and self.env:
            # 绘制边界
            self.ax.plot([0, self.env.width, self.env.width, 0, 0],
                         [0, 0, self.env.height, self.env.height, 0], 'k-')

            # 绘制障碍物
            self.env.plot_obstacles(self.ax)

        # 绘制搜索树
        for node in nodes:
            if node.parent:
                self.ax.plot(
                    [node.x, node.parent.x],
                    [node.y, node.parent.y],
                    '-', color=tree_color, alpha=tree_alpha
                )

        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            self.ax.plot(path_x, path_y, '-', color=path_color,
                         linewidth=2.0, label='最终路径')

        # 绘制起点和终点
        if nodes:
            start_node = nodes[0]  # 假设第一个节点是起点
            self.ax.scatter(start_node.x, start_node.y,
                            color='green', s=100, label='起点')

            # 查找终点（路径最后一个点）
            if path:
                goal_x, goal_y = path[-1]
                self.ax.scatter(goal_x, goal_y,
                                color='red', s=100, label='终点')

        # 设置坐标轴
        if self.env:
            self.ax.set_xlim(-5, self.env.width + 5)
            self.ax.set_ylim(-5, self.env.height + 5)
        else:
            # 如果没有环境，根据节点设置坐标轴范围
            node_x = [node.x for node in nodes]
            node_y = [node.y for node in nodes]
            x_min, x_max = min(node_x), max(node_x)
            y_min, y_max = min(node_y), max(node_y)
            margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 5)
            self.ax.set_xlim(x_min - margin, x_max + margin)
            self.ax.set_ylim(y_min - margin, y_max + margin)

        self.ax.set_aspect('equal')
        if show_grid:
            self.ax.grid(True)
        self.ax.set_title(title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return self.fig

    def animate_path(
        self,
        path: List[Tuple[float, float]],
        interval: int = 50,
        figsize: Tuple[int, int] = (10, 8),
        show_environment: bool = True,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        创建路径动画

        参数:
            path: 路径点列表
            interval: 帧间隔（毫秒）
            figsize: 图形大小
            show_environment: 是否显示环境
            save_path: 动画保存路径，如果为None则不保存

        返回:
            动画对象
        """
        if not path:
            print("路径为空，无法创建动画")
            return None

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # 绘制环境
        if show_environment and self.env:
            # 绘制边界
            self.ax.plot([0, self.env.width, self.env.width, 0, 0],
                         [0, 0, self.env.height, self.env.height, 0], 'k-')

            # 绘制障碍物
            self.env.plot_obstacles(self.ax)

        # 绘制路径
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        path_line, = self.ax.plot([], [], '-b', linewidth=2)

        # 绘制车辆（用圆表示）
        vehicle = plt.Circle((path[0][0], path[0][1]),
                             1.0, color='red', fill=True)
        self.ax.add_patch(vehicle)

        # 绘制起点和终点
        self.ax.scatter(path[0][0], path[0][1],
                        color='green', s=100, label='起点')
        self.ax.scatter(path[-1][0], path[-1][1],
                        color='red', s=100, label='终点')

        # 设置坐标轴
        if self.env:
            self.ax.set_xlim(-5, self.env.width + 5)
            self.ax.set_ylim(-5, self.env.height + 5)
        else:
            x_min, x_max = min(path_x), max(path_x)
            y_min, y_max = min(path_y), max(path_y)
            margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 5)
            self.ax.set_xlim(x_min - margin, x_max + margin)
            self.ax.set_ylim(y_min - margin, y_max + margin)

        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('路径动画')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend()

        # 初始化函数
        def init():
            path_line.set_data([], [])
            vehicle.center = (path[0][0], path[0][1])
            return path_line, vehicle

        # 动画更新函数
        def update(frame):
            # 更新已经走过的路径
            path_line.set_data(path_x[:frame+1], path_y[:frame+1])

            # 更新车辆位置
            vehicle.center = (path_x[frame], path_y[frame])

            return path_line, vehicle

        # 创建动画
        anim = FuncAnimation(
            self.fig, update, frames=len(path),
            init_func=init, blit=True, interval=interval
        )

        # 保存动画
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)

        return anim

    def plot_metrics(
        self,
        metrics: Dict[str, List[float]],
        figsize: Tuple[int, int] = (10, 6),
        title: str = '性能指标',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制性能指标

        参数:
            metrics: 指标字典，键为指标名称，值为指标数值列表
            figsize: 图形大小
            title: 图形标题
            save_path: 图形保存路径，如果为None则不保存

        返回:
            matplotlib图形对象
        """
        if not metrics:
            print("指标为空，无法绘制")
            return None

        # 创建图形
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]

        # 绘制每个指标
        for i, (name, values) in enumerate(metrics.items()):
            ax = axes[i]
            x = list(range(1, len(values) + 1))
            ax.plot(x, values, 'o-', linewidth=2)
            ax.set_title(name)
            ax.set_xlabel('迭代')
            ax.set_ylabel('值')
            ax.grid(True)

        plt.tight_layout()
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.9)

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def show(self) -> None:
        """显示图形"""
        plt.show()

    def close(self) -> None:
        """关闭图形"""
        plt.close(self.fig)
        self.fig = None
        self.ax = None


# 简单的用法示例
if __name__ == "__main__":
    from environment import Environment

    # 创建环境
    env = Environment(width=100, height=100)

    # 添加一些障碍物
    env.add_obstacle(20, 20, obstacle_type="circle", radius=5)
    env.add_obstacle(40, 50, obstacle_type="circle", radius=8)
    env.add_obstacle(70, 30, obstacle_type="rectangle", width=10, height=15)

    # 创建一条简单的路径
    path = [(10, 10), (20, 30), (40, 40), (60, 50), (80, 70), (90, 90)]

    # 创建可视化工具
    vis = Visualization(env)

    # 绘制路径
    vis.plot_path(path, title="测试路径")

    # 绘制多条路径
    path2 = [(10, 10), (15, 25), (30, 45), (50, 60), (70, 75), (90, 90)]
    path3 = [(10, 10), (25, 20), (45, 35), (65, 60), (85, 80), (90, 90)]

    vis.plot_multi_paths(
        [path, path2, path3],
        ['路径1', '路径2', '路径3'],
        colors=['blue', 'red', 'purple']
    )

    # 创建路径动画
    anim = vis.animate_path(path)

    # 绘制一些性能指标
    metrics = {
        '路径长度': [100, 95, 90, 88, 85, 84, 83],
        '执行时间(ms)': [150, 140, 135, 120, 118, 115, 110],
        '计算节点数': [200, 180, 160, 150, 140, 130, 120]
    }

    vis.plot_metrics(metrics)

    # 显示图形
    plt.show()
