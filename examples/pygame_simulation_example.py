#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pygame 仿真示例脚本

演示如何使用 Pygame 代替 CarSim 进行路径规划仿真。
"""


########### 添加项目根目录到 Python 路径 ##########
import os
import sys


pythonpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)

#####


import numpy as np
import argparse
from simulation.environment import Environment
from simulation.pygame_simulator import PygameSimulator
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar
from rrt.informed_rrt import InformedRRTStar



# Project imports - these must come after modifying sys.path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Pygame 路径规划仿真')

    parser.add_argument('--start', type=float, nargs=2, default=[10, 10],
                        help='起点坐标，例如："--start 10 10"')

    parser.add_argument('--goal', type=float, nargs=2, default=[90, 90],
                        help='终点坐标，例如："--goal 90 90"')

    parser.add_argument('--map', type=str, default=None,
                        help='地图文件路径')

    parser.add_argument(
        '--config',
        type=str,
        default='config/pygame_config.yaml',
        help='Pygame配置文件路径'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['rrt', 'rrt_star', 'informed_rrt'],
        default='rrt_star',
        help='路径规划算法'
    )

    parser.add_argument('--iterations', type=int, default=1000,
                        help='算法最大迭代次数')

    parser.add_argument('--save-fig', action='store_true',
                        help='保存结果图表')

    parser.add_argument('--save-path', type=str, default=None,
                        help='保存路径的文件路径')

    return parser.parse_args()


def create_environment():
    """创建仿真环境"""
    env = Environment(width=100.0, height=100.0)

    # 添加一些障碍物
    env.add_obstacle(x=30, y=30, obstacle_type="circle", radius=5.0)
    env.add_obstacle(x=50, y=50, obstacle_type="circle", radius=7.0)
    env.add_obstacle(x=70, y=40, obstacle_type="rectangle",
                     width=10.0, height=20.0)
    env.add_obstacle(x=40, y=70, obstacle_type="rectangle",
                     width=15.0, height=10.0)

    return env


def load_environment(map_file):
    """加载环境"""
    import yaml
    import os

    env = Environment(width=100.0, height=100.0)

    if not os.path.exists(map_file):
        print(f"地图文件不存在: {map_file}")
        return create_environment()

    try:
        with open(map_file, 'r', encoding='utf-8') as f:
            map_data = yaml.safe_load(f)

        if 'environment' in map_data:
            env_data = map_data['environment']

            # 设置环境参数
            if 'width' in env_data:
                env.width = env_data['width']
            if 'height' in env_data:
                env.height = env_data['height']

            # 加载障碍物
            if 'obstacles' in env_data:
                for obstacle in env_data['obstacles']:
                    if obstacle['type'] == 'circle':
                        env.add_obstacle(
                            x=obstacle['x'],
                            y=obstacle['y'],
                            obstacle_type="circle",
                            radius=obstacle['radius']
                        )
                    elif obstacle['type'] == 'rectangle':
                        env.add_obstacle(
                            x=obstacle['x'],
                            y=obstacle['y'],
                            obstacle_type="rectangle",
                            width=obstacle['width'],
                            height=obstacle['height']
                        )

        print(f"成功加载地图: {map_file}")
        return env

    except Exception as e:
        print(f"加载地图失败: {e}")
        return create_environment()


def plan_path(env, start, goal, algorithm='rrt_star', max_iterations=1000):
    """规划路径"""
    if algorithm == 'rrt':
        planner = RRT(
            start=start,
            goal=goal,
            env=env,
            max_iterations=max_iterations,
            step_size=5.0
        )
    elif algorithm == 'rrt_star':
        planner = RRTStar(
            start=start,
            goal=goal,
            env=env,
            max_iterations=max_iterations,
            step_size=5.0
        )
    elif algorithm == 'informed_rrt':
        planner = InformedRRTStar(
            start=start,
            goal=goal,
            env=env,
            max_iterations=max_iterations,
            step_size=5.0
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    # 执行规划
    path = planner.plan()

    # 获取搜索树节点（假设算法实现中提供了这个属性）
    nodes = getattr(planner, 'nodes', [])

    return path, nodes


def simulate_path(env, path, nodes=None, config_path=None):
    """使用Pygame仿真路径执行"""
    # 创建仿真器
    simulator = PygameSimulator(config_path)

    # 设置环境
    simulator.set_environment(env)

    # 执行路径
    print("开始路径仿真，按ESC退出，空格键暂停/继续，R键重置")
    simulator.execute_path(path)

    # 获取仿真结果
    results = simulator.get_simulation_results()

    # 可视化结果
    simulator.visualize_results(results)

    return results


def save_results(path, results, save_path):
    """保存结果"""
    import pandas as pd
    import os

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # 转换为DataFrame
    df = pd.DataFrame({
        '时间': results['time'],
        'X位置': results['position_x'],
        'Y位置': results['position_y'],
        '朝向': [np.degrees(h) for h in results['heading']],
        '速度': results['speed'],
        '转向角': [np.degrees(s) for s in results['steer_angle']],
        '加速度': results['acceleration']
    })

    # 保存CSV
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f"结果已保存到: {save_path}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建或加载环境
    if args.map:
        env = load_environment(args.map)
    else:
        env = create_environment()

    # 规划路径
    start = tuple(args.start)
    goal = tuple(args.goal)

    print(f"使用算法 {args.algorithm} 进行路径规划，从 {start} 到 {goal}")
    path, nodes = plan_path(env, start, goal, args.algorithm, args.iterations)

    if not path:
        print("路径规划失败，请尝试增加迭代次数或修改起止点")
        return

    print(f"路径规划成功，路径长度: {len(path)}个点")

    # 仿真路径
    results = simulate_path(env, path, nodes, args.config)

    # 保存结果
    if args.save_path:
        save_results(path, results, args.save_path)


if __name__ == "__main__":
    main()
