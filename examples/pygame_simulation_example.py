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

    parser.add_argument('--start', type=float, nargs=2, default=[10, 10],
                        help='起点坐标，例如："--start 10 10"')

    parser.add_argument('--goal', type=float, nargs=2, default=[70, 70],
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
        choices=[
            'rrt',
            'rrt_star',
            'informed_rrt',
            'astar',
            'dijkstra',
            'dstar_lite',
            'theta_star',
            'rl',
            'ppo'
        ],
        default='rrt_star',
        help='路径规划算法'
    )

    parser.add_argument('--iterations', type=int, default=10000,
                        help='算法最大迭代次数')

    parser.add_argument('--save-fig', action='store_true',
                        help='保存结果图表')

    parser.add_argument('--save-path', type=str, default=None,
                        help='保存路径的文件路径')

    parser.add_argument('--model-path', type=str, default=None,
                        help='RL/PPO模型路径')

    return parser.parse_args()


def check_path_feasibility(env, start_pos, goal_pos, algorithm: str):
    """
    检查路径可行性，使用实际规划算法和A*进行双重验证

    参数:
        env: 环境对象
        start_pos: 起点坐标
        goal_pos: 终点坐标
        algorithm: 使用的规划算法
    """
    # 1. 使用实际规划算法进行验证
    common_params = {
        'start': start_pos,
        'goal': goal_pos,
        'env': env,
    }

    # 根据算法类型创建对应的规划器
    if algorithm in ['rrt', 'rrt_star', 'informed_rrt']:
        test_planner = globals()[{
            'rrt': 'RRT',
            'rrt_star': 'RRTStar',
            'informed_rrt': 'InformedRRTStar'
        }[algorithm]](
            **common_params,
            max_iterations=5000,  # 验证时使用较大的迭代次数
            step_size=5.0
        )
    elif algorithm in ['astar', 'dijkstra', 'dstar_lite', 'theta_star']:
        planner_class = {
            'astar': AStar,
            'dijkstra': Dijkstra,
            'dstar_lite': DStarLite,
            'theta_star': ThetaStar
        }[algorithm]
        test_planner = planner_class(
            **common_params,
            resolution=0.5,  # 验证时使用更小的分辨率
            diagonal_movement=True
        )
    else:  # RL/PPO算法使用A*验证
        test_planner = AStar(
            **common_params,
            resolution=0.5,
            diagonal_movement=True
        )

    primary_path = test_planner.plan()
    if not primary_path:
        print(f"使用 {algorithm} 算法验证失败：无法找到可行路径")
        return False

    # 验证路径连续性
    for i in range(len(primary_path)-1):
        p1 = primary_path[i]
        p2 = primary_path[i+1]
        if not check_line_collision_free(env, p1, p2):
            print(f"使用 {algorithm} 算法验证失败：路径段存在碰撞")
            return False

    # 2. 使用A*进行额外验证（如果主验证不是A*）
    if algorithm != 'astar':
        astar = AStar(
            **common_params,
            resolution=0.5,
            diagonal_movement=True
        )
        astar_path = astar.plan()
        if not astar_path:
            print("A*额外验证失败：无法找到可行路径")
            return False

        # 验证A*路径的连续性
        for i in range(len(astar_path)-1):
            p1 = astar_path[i]
            p2 = astar_path[i+1]
            if not check_line_collision_free(env, p1, p2):
                print("A*额外验证失败：路径段存在碰撞")
                return False

    print(f"路径可行性验证通过（使用 {algorithm} 算法" +
          (" 和 A*额外验证" if algorithm != 'astar' else "") + "）")
    return True


def check_line_collision_free(env, start, end, steps=10):
    """
    检查两点之间的线段是否无碰撞

    参数:
        env: 环境对象
        start: 起点坐标
        end: 终点坐标
        steps: 检查点数量
    """
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if env.check_collision((x, y)):
            return False
    return True


def create_environment(start: tuple, goal: tuple, algorithm: str):
    """
    创建仿真环境

    参数:
        start: 起点坐标 (x, y)
        goal: 终点坐标 (x, y)
        algorithm: 使用的规划算法
    """
    # 创建场景生成器
    generator = ScenarioGenerator(
        width=100.0,
        height=100.0,
        min_obstacle_size=2.0,
        max_obstacle_size=10.0,
        min_gap=5.0
    )

    max_attempts = 10  # 最大重试次数
    safety_margin = 5.0  # 与起点和终点的安全距离
    min_path_width = 8.0  # 最小路径宽度

    def check_corridor_width(env, pos, direction, width):
        """检查指定位置的走廊宽度"""
        perpendicular = (-direction[1], direction[0])  # 垂直方向
        for d in range(int(width/2)):
            test_pos = (
                pos[0] + perpendicular[0] * d,
                pos[1] + perpendicular[1] * d
            )
            if env.check_collision(test_pos):
                return False
            test_pos = (
                pos[0] - perpendicular[0] * d,
                pos[1] - perpendicular[1] * d
            )
            if env.check_collision(test_pos):
                return False
        return True

    def calculate_obstacle_area(obstacle):
        """计算单个障碍物的面积"""
        if getattr(obstacle, 'type', '') == 'circle':
            return np.pi * obstacle.radius**2
        else:  # rectangle
            return obstacle.width * obstacle.height

    for attempt in range(max_attempts):
        # 生成混合场景
        env = generator.generate_mixed_scenario(
            num_random_obstacles=5,
            num_rooms=2,
            corridor_width=10.0
        )

        # 1. 检查起点和终点周围区域
        start_clear = not any(
            env.check_collision((
                start[0] + dx * safety_margin,
                start[1] + dy * safety_margin
            ))
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        )

        goal_clear = not any(
            env.check_collision((
                goal[0] + dx * safety_margin,
                goal[1] + dy * safety_margin
            ))
            for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        )

        if not (start_clear and goal_clear):
            print(f"第 {attempt + 1} 次生成的场景阻挡了起点或终点及其周围区域，重新生成...")
            continue

        # 2. 使用实际规划算法检查路径可行性
        if not check_path_feasibility(env, start, goal, algorithm):
            print(f"第 {attempt + 1} 次生成的场景中使用 {algorithm} 算法无法找到可行路径，重新生成...")
            continue

        # 3. 检查关键路径点的通行宽度
        test_planner = AStar(
            start=start,
            goal=goal,
            env=env,
            resolution=0.5,
            diagonal_movement=True
        )
        reference_path = test_planner.plan()

        if reference_path:
            # 检查路径上的关键点
            path_valid = True
            for i in range(1, len(reference_path)):
                current = reference_path[i]
                prev = reference_path[i-1]
                direction = (
                    current[0] - prev[0],
                    current[1] - prev[1]
                )
                # 归一化方向向量
                length = np.hypot(direction[0], direction[1])
                if length > 0:
                    direction = (direction[0]/length, direction[1]/length)
                    if not check_corridor_width(env, current, direction, min_path_width):
                        path_valid = False
                        break

            if not path_valid:
                print(f"第 {attempt + 1} 次生成的场景在路径上存在过窄通道，重新生成...")
                continue

        # 4. 验证障碍物分布
        total_area = env.width * env.height
        obstacle_area = sum(calculate_obstacle_area(obs)
                            for obs in env.obstacles)
        space_usage = obstacle_area / total_area

        if space_usage > 0.6:  # 障碍物占比不超过60%
            print(f"第 {attempt + 1} 次生成的场景障碍物密度过高 ({space_usage:.1%})，重新生成...")
            continue

        # 所有验证都通过
        print(f"已生成有效的混合场景（尝试次数：{attempt + 1}），包含:")
        print("- 5个随机障碍物")
        print("- 2个房间")
        print("- 走廊系统")
        print(f"起点 {start} 和终点 {goal} 及其周围 {safety_margin} 单位范围内均可达")
        print(f"最小通道宽度: {min_path_width} 单位")
        print(f"场景空间利用率: {space_usage:.1%}")
        print(f"使用 {algorithm} 算法验证路径可行性：通过")
        return env

    # 如果多次尝试后仍未成功，创建一个简化的随机场景
    print("警告：多次尝试后未能生成复杂场景，返回简化的随机场景")

    # 创建一个新的场景生成器，使用更大的障碍物尺寸
    simple_generator = ScenarioGenerator(
        width=100.0,
        height=100.0,
        min_obstacle_size=3.0,  # 增加最小尺寸
        max_obstacle_size=15.0,  # 增加最大尺寸
        min_gap=6.0  # 增加最小间隔
    )

    simple_env = simple_generator.generate_random_scenario(
        num_obstacles=15,  # 增加到8个随机障碍物
        density=0.5  # 略微增加密度
    )

    # 验证简化场景的可行性
    if check_path_feasibility(simple_env, start, goal, algorithm):
        print("已生成简化的随机场景：")
        print("- 8个随机障碍物")
        print("- 障碍物尺寸范围：3.0-15.0")
        print("- 最小间隔：6.0")
        print(f"- 使用 {algorithm} 算法验证路径可行性：通过")
        return simple_env

    # 如果简化场景也不可行，则返回空场景
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
        return create_environment(start, goal, 'rrt_star')


def plan_path(env, start, goal, algorithm='rrt_star', max_iterations=1000,
              model_path=None):
    """规划路径"""
    common_params = {
        'start': start,
        'goal': goal,
        'env': env,
    }

    if algorithm == 'rrt':
        planner = RRT(
            **common_params,
            max_iterations=max_iterations,
            step_size=5.0
        )
    elif algorithm == 'rrt_star':
        planner = RRTStar(
            **common_params,
            max_iterations=max_iterations,
            step_size=5.0
        )
    elif algorithm == 'informed_rrt':
        planner = InformedRRTStar(
            **common_params,
            max_iterations=max_iterations,
            step_size=5.0
        )
    elif algorithm == 'astar':
        planner = AStar(
            **common_params,
            resolution=1.0,
            diagonal_movement=True
        )
    elif algorithm == 'dijkstra':
        planner = Dijkstra(
            **common_params,
            resolution=1.0,
            diagonal_movement=True
        )
    elif algorithm == 'dstar_lite':
        planner = DStarLite(
            **common_params,
            resolution=1.0,
            diagonal_movement=True
        )
    elif algorithm == 'theta_star':
        planner = ThetaStar(
            **common_params,
            resolution=1.0,
            diagonal_movement=True
        )
    elif algorithm == 'rl':
        planner = RLPathPlanner(
            **common_params,
            model_path=model_path,
            max_steps=max_iterations
        )
    elif algorithm == 'ppo':
        planner = PPOPathPlanner(
            **common_params,
            model_path=model_path,
            max_steps=max_iterations
        )
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
            env,
            start,
            goal,
            args.algorithm,
            args.iterations,
            args.model_path
        )

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
