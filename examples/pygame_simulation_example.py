#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pygame 仿真示例脚本

演示如何使用 Pygame 代替 CarSim 进行路径规划仿真。
支持多种路径规划算法。
"""

from rrt.informed_rrt import InformedRRTStar
from rrt.rrt_star import RRTStar
from rrt.rrt_base import RRT
from rrt.astar import AStar
from rrt.dijkstra import Dijkstra
from rrt.dstar_lite import DStarLite
from rrt.theta_star import ThetaStar
from rrt.rl_planner import RLPathPlanner
from rrt.ppo_planner import PPOPathPlanner
from simulation.pygame_simulator import PygameSimulator
from simulation.environment import Environment
from simulation.scenario_generator import ScenarioGenerator
import argparse
import numpy as np

# Project imports - these must come after modifying sys.path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Pygame 路径规划仿真')

    parser.add_argument('--start', type=float, nargs=2,
                        default=[10, 10], help='起点坐标，例如："--start 10 10"')

    parser.add_argument('--goal', type=float, nargs=2,
                        default=[70, 70], help='终点坐标，例如："--goal 90 90"')

    parser.add_argument('--map', type=str, default=None, help='地图文件路径')

    parser.add_argument(
        '--config', type=str, default='config/pygame_config.yaml', help='Pygame配置文件路径')

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['rrt', 'rrt_star', 'informed_rrt', 'astar',
                 'dijkstra', 'dstar_lite', 'theta_star', 'rl', 'ppo'],
        default='dijkstra',
        help='路径规划算法')

    parser.add_argument('--iterations', type=int,
                        default=10000, help='算法最大迭代次数')

    parser.add_argument('--save-fig', action='store_true', help='保存结果图表')

    parser.add_argument('--save-path', type=str,
                        default=None, help='保存路径的文件路径')

    parser.add_argument('--model-path', type=str,
                        default=None, help='RL/PPO模型路径')

    return parser.parse_args()


def check_path_feasibility(env, start_pos, goal_pos, algorithm: str):
    """
    检查路径可行性，使用 Dijkstra 算法进行快速验证

    参数:
        env: 环境对象
        start_pos: 起点坐标
        goal_pos: 终点坐标
        algorithm: 使用的规划算法
    """
    import threading
    import time

    # 使用 Dijkstra 进行快速验证
    common_params = {
        'start': start_pos,
        'goal': goal_pos,
        'env': env,
    }

    # 使用较大的分辨率和较少的采样点来加速验证
    test_planner = Dijkstra(
        **common_params,
        resolution=2.0,  # 增大分辨率
        diagonal_movement=True)  # 允许对角线移动以减少路径点

    # 使用线程实现超时机制
    path = []
    path_found = threading.Event()
    planning_error = None

    def planning_thread():
        nonlocal path, planning_error
        try:
            path = test_planner.plan()
            path_found.set()
        except Exception as e:
            planning_error = e
            path_found.set()

    # 启动规划线程
    planner_thread = threading.Thread(target=planning_thread)
    planner_thread.daemon = True  # 设置为守护线程，主线程退出时会被强制结束
    planner_thread.start()

    # 等待规划完成或超时
    if not path_found.wait(timeout=5.0):  # 5秒超时
        print("路径验证超时（>5.0秒）")
        return False

    # 检查是否有错误发生
    if planning_error is not None:
        print(f"验证过程出错: {str(planning_error)}")
        return False

    if not path:
        print("路径验证失败：无法找到可行路径")
        return False

    # 快速验证路径连续性和碰撞
    # 使用较少的采样点进行快速检查
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        if not check_line_collision_free(env, p1, p2, steps=5):  # 减少采样点数
            print("路径验证失败：路径段存在碰撞")
            return False

    print(f"路径可行性验证通过（使用 Dijkstra 算法）")
    return True


def check_line_collision_free(env, start, end, steps=5):
    """
    检查两点之间的线段是否无碰撞，使用较少的采样点加速检查

    参数:
        env: 环境对象
        start: 起点坐标
        end: 终点坐标
        steps: 检查点数量
    """
    # 首先检查起点和终点
    if env.check_collision(start) or env.check_collision(end):
        return False

    # 如果起点和终点距离很近，直接返回
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx * dx + dy * dy < 4.0:  # 距离小于2个单位
        return True

    # 对较长的线段进行采样检查
    for i in range(1, steps):  # 跳过起点和终点，因为已经检查过
        t = i / steps
        x = start[0] + t * dx
        y = start[1] + t * dy
        if env.check_collision((x, y)):
            return False
    return True


def create_environment(start: tuple, goal: tuple, algorithm: str):
    """
    创建仿真环境，使用优化的障碍物生成逻辑

    参数:
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        algorithm: 使用的规划算法
    """
    # 创建场景生成器，使用更合理的参数
    generator = ScenarioGenerator(
        width=100.0,
        height=100.0,
        min_obstacle_size=3.0,  # 增大最小尺寸，减少小障碍物
        max_obstacle_size=12.0,  # 限制最大尺寸
        min_gap=8.0  # 增大最小间隔，确保路径可行性
    )

    max_attempts = 5  # 减少最大重试次数
    safety_margin = 8.0  # 增大安全距离
    min_path_width = 10.0  # 增大最小路径宽度

    def check_start_goal_clearance(env, start, goal, margin):
        """快速检查起点和终点区域是否可行"""
        # 检查起点区域
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            if env.check_collision((start[0] + dx * margin, start[1] + dy * margin)):
                return False
            if env.check_collision((goal[0] + dx * margin, goal[1] + dy * margin)):
                return False
        return True

    def calculate_obstacle_density(env):
        """计算障碍物密度"""
        total_area = env.width * env.height
        obstacle_area = 0
        for obs in env.obstacles:
            if getattr(obs, 'type', '') == 'circle':
                obstacle_area += np.pi * obs.radius**2
            else:  # rectangle
                obstacle_area += obs.width * obs.height
        return obstacle_area / total_area

    for attempt in range(max_attempts):
        print(f"正在生成场景，第 {attempt + 1} 次尝试...")

        # 生成基本场景，减少随机障碍物数量和房间数量
        env = generator.generate_mixed_scenario(
            num_random_obstacles=3,  # 减少随机障碍物数量
            num_rooms=1,  # 减少房间数量
            corridor_width=12.0  # 增大走廊宽度
        )

        # 1. 快速检查起点和终点区域
        if not check_start_goal_clearance(env, start, goal, safety_margin):
            print(f"第 {attempt + 1} 次生成的场景阻挡了起点或终点区域，重新生成...")
            continue

        # 2. 检查障碍物密度
        density = calculate_obstacle_density(env)
        if density > 0.4:  # 降低允许的最大密度
            print(f"第 {attempt + 1} 次生成的场景障碍物密度过高 ({density:.1%})，重新生成...")
            continue

        # 3. 使用 Dijkstra 验证路径可行性
        if not check_path_feasibility(env, start, goal, algorithm):
            print(f"第 {attempt + 1} 次生成的场景验证失败，重新生成...")
            continue

        # 所有验证都通过
        print(f"已生成有效场景（尝试次数：{attempt + 1}），包含:")
        print(f"- {len(env.obstacles)} 个障碍物")
        print(f"- 走廊宽度: {12.0} 单位")
        print(f"- 障碍物密度: {density:.1%}")
        return env

    # 如果多次尝试后仍未成功，返回简化场景
    print("警告：多次尝试后未能生成有效场景，返回简化场景")

    # 创建一个非常简单的场景，只包含少量大型障碍物
    simple_env = generator.generate_random_scenario(
        num_obstacles=4,  # 非常少的障碍物
        density=0.3  # 低密度
    )

    # 验证简化场景
    if check_path_feasibility(simple_env, start, goal, algorithm):
        print("已生成简化场景：")
        print(f"- {len(simple_env.obstacles)} 个障碍物")
        print("- 低密度分布")
        return simple_env

    # 如果简化场景也不可行，返回空场景
    print("警告：简化场景验证失败，返回空场景")
    return Environment(width=100.0, height=100.0)


def load_environment(map_file, start: tuple, goal: tuple):
    """
    加载环境

    参数:
        map_file: 地图文件路径
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
    """
    import yaml
    import os

    env = Environment(width=100.0, height=100.0)

    if not os.path.exists(map_file):
        print(f"地图文件不存在: {map_file}")
        return create_environment(start, goal, 'rrt_star')

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
                        env.add_obstacle(x=obstacle['x'],
                                         y=obstacle['y'],
                                         obstacle_type="circle",
                                         radius=obstacle['radius'])
                    elif obstacle['type'] == 'rectangle':
                        env.add_obstacle(x=obstacle['x'],
                                         y=obstacle['y'],
                                         obstacle_type="rectangle",
                                         width=obstacle['width'],
                                         height=obstacle['height'])

        print(f"成功加载地图: {map_file}")
        return env

    except Exception as e:
        print(f"加载地图失败: {e}")
        return create_environment(start, goal, 'rrt_star')


def plan_path(env, start, goal, algorithm='rrt_star', max_iterations=1000, model_path=None):
    """规划路径"""
    common_params = {
        'start': start,
        'goal': goal,
        'env': env,
    }

    if algorithm == 'rrt':
        planner = RRT(**common_params,
                      max_iterations=max_iterations, step_size=5.0)
    elif algorithm == 'rrt_star':
        planner = RRTStar(
            **common_params, max_iterations=max_iterations, step_size=5.0)
    elif algorithm == 'informed_rrt':
        planner = InformedRRTStar(
            **common_params, max_iterations=max_iterations, step_size=5.0)
    elif algorithm == 'astar':
        planner = AStar(**common_params, resolution=1.0,
                        diagonal_movement=True)
    elif algorithm == 'dijkstra':
        planner = Dijkstra(**common_params, resolution=1.0,
                           diagonal_movement=True)
    elif algorithm == 'dstar_lite':
        planner = DStarLite(**common_params, resolution=1.0,
                            diagonal_movement=True)
    elif algorithm == 'theta_star':
        planner = ThetaStar(**common_params, resolution=1.0,
                            diagonal_movement=True)
    elif algorithm == 'rl':
        planner = RLPathPlanner(
            **common_params, model_path=model_path, max_steps=max_iterations)
    elif algorithm == 'ppo':
        planner = PPOPathPlanner(
            **common_params, model_path=model_path, max_steps=max_iterations)
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    # 执行规划
    path = planner.plan()

    # 获取搜索树节点（假设算法实现中提供了这个属性）
    nodes = getattr(planner, 'nodes', [])

    # 对于Theta*，进行路径后处理
    if algorithm == 'theta_star' and path:
        path = planner.post_process_path(path)

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
    start = tuple(args.start)
    goal = tuple(args.goal)

    if args.map:
        env = load_environment(args.map, start, goal)
    else:
        env = create_environment(start, goal, args.algorithm)

    # 规划路径
    max_planning_attempts = 10  # 最大规划尝试次数
    path = None
    nodes = None

    print(f"使用算法 {args.algorithm} 进行路径规划，从 {start} 到 {goal}")

    for attempt in range(max_planning_attempts):
        path, nodes = plan_path(
            env, start, goal, args.algorithm, args.iterations, args.model_path)

        if path:
            print(f"路径规划成功（尝试次数：{attempt + 1}），路径长度: {len(path)}个点")
            break
        else:
            print(f"第 {attempt + 1} 次路径规划失败，正在重试...")
            # 增加迭代次数，提高成功率
            args.iterations = int(args.iterations * 1.5)

    if not path:
        print("\n警告：多次尝试后仍未找到可行路径！")
        print("建议检查：")
        print("1. 起点和终点的位置是否合理")
        print("2. 场景中的障碍物分布是否过于密集")
        print("3. 尝试增加迭代次数（当前：{}）".format(args.iterations))
        print("4. 考虑使用其他路径规划算法")
        return

    # 仿真路径
    results = simulate_path(env, path, nodes, args.config)

    # 保存结果
    if args.save_path:
        save_results(path, results, args.save_path)


if __name__ == "__main__":
    main()
