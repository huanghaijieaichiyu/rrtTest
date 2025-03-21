#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行动态环境中的RRT*路径规划

此程序创建一个城市环境仿真，并使用Pygame可视化RRT*算法在动态环境中的路径规划。
支持跟随移动目标，实时更新路径规划。
"""

import argparse
import numpy as np
import time

from simulation.dynamic_environment import DynamicEnvironment, MovementPattern
from simulation.pygame_urban_simulator import PygameUrbanSimulator


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行动态环境中的RRT*路径规划')
    parser.add_argument('--width', type=int, default=1024, help='窗口宽度 (像素)')
    parser.add_argument('--height', type=int, default=768, help='窗口高度 (像素)')
    parser.add_argument('--env_width', type=float,
                        default=100.0, help='环境宽度 (米)')
    parser.add_argument('--env_height', type=float,
                        default=100.0, help='环境高度 (米)')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    parser.add_argument('--start_x', type=float, default=10.0, help='起点X坐标')
    parser.add_argument('--start_y', type=float, default=10.0, help='起点Y坐标')
    parser.add_argument('--goal_x', type=float, default=90.0, help='终点X坐标')
    parser.add_argument('--goal_y', type=float, default=90.0, help='终点Y坐标')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--follow', action='store_true', help='启用目标跟随')
    parser.add_argument('--auto_replan', action='store_true', help='启用自动重规划')
    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)

    print("======================================")
    print("    动态环境中的RRT*路径规划演示程序    ")
    print("======================================")
    print(f"环境大小: {args.env_width} x {args.env_height}")
    print(f"起点: ({args.start_x}, {args.start_y})")
    print(f"终点: ({args.goal_x}, {args.goal_y})")
    print(f"目标跟随: {'启用' if args.follow else '禁用'}")
    print(f"自动重规划: {'启用' if args.auto_replan else '禁用'}")
    print("--------------------------------------")

    # 创建动态环境
    env = DynamicEnvironment(width=args.env_width, height=args.env_height)

    # 生成城市环境
    print("创建城市环境...")
    env.create_urban_environment()

    # 添加移动目标
    target_x, target_y = args.goal_x, args.goal_y
    pattern_params = {
        'center_x': target_x,
        'center_y': target_y,
        'radius': 20.0,
        'angle': 0.0
    }
    env.add_moving_target(
        x=target_x,
        y=target_y,
        speed=3.0,
        pattern=MovementPattern.CIRCULAR,
        pattern_params=pattern_params
    )
    print(f"添加移动目标: ({target_x}, {target_y})")

    print(f"静态障碍物数量: {len(env.obstacles)}")
    print(f"移动障碍物数量: {len(env.moving_obstacles)}")
    print(f"交通信号灯数量: {len(env.traffic_lights)}")

    # 创建仿真器
    simulator = PygameUrbanSimulator(
        env,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    # 设置起点和终点
    start_point = (args.start_x, args.start_y)
    goal_point = (args.goal_x, args.goal_y)

    # 设置跟随目标和自动重规划
    simulator.follow_target = args.follow
    simulator.auto_replan = args.auto_replan

    print("\n开始仿真...")
    print("提示:")
    print("- 使用ESC退出")
    print("- 使用空格暂停/继续")
    print("- 使用R键重新规划路径")
    print("- 使用F键切换目标跟随")
    print("- 使用A键切换自动重规划")
    print("- 使用鼠标左键设置新的起点")
    print("- 使用鼠标右键设置新的终点")
    print("- 使用鼠标中键添加新的移动目标")
    print("更多控制选项请参考GUI中的控制说明")
    print("--------------------------------------")

    # 运行仿真
    simulator.run(start_point, goal_point)

    print("仿真结束")


if __name__ == "__main__":
    main()
