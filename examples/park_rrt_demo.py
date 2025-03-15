import numpy as np
import random
import argparse
import yaml
from typing import List, Tuple, Dict, Any, Optional
from rrt.astar import AStar
from simulation.environment import Environment
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar, TimedRRTStar
from rrt.informed_rrt import InformedRRTStar
from rrt.dijkstra import Dijkstra
from rrt.dstar_lite import DStarLite
from rrt.theta_star import ThetaStar
from simulation.pygame_simulator import (
    PygameSimulator,
    PathFollower,
    ParkingEnvironment,
    check_vehicle_collision,
    check_path_collision,
    get_font
)
from simulation.pygame_simulator import VehicleModel
import math
import pygame


# 加载配置文件


def load_config(config_path: Optional[str] = None) -> Dict:
    """加载配置文件"""
    # 默认配置
    default_config = {
        # 窗口设置
        'window': {
            'width': 1280,
            'height': 677,
            'title': "停车场路径规划仿真器"
        },

        # 仿真参数
        'simulation': {
            'scale': 10.0,  # 像素/米
            'fps': 60,   # 帧率
            'dt': 0.05,  # 仿真时间步长(秒)
            'lookahead': 5.0,  # 路径跟踪前瞻距离
            'simulation_speed': 2.0  # 仿真速度倍率
        },

        # 车辆参数
        'vehicle': {
            'length': 4.5,     # 车辆长度(米)
            'width': 1.8,      # 车辆宽度(米)
            'wheelbase': 2.7,  # 轴距(米)
            'max_speed': 5.0,  # 最大速度(m/s)
            'max_accel': 2.0,   # 最大加速度(m/s^2)
            'max_brake': 4.0,   # 最大制动(m/s^2)
            'max_steer': 0.7854  # 最大转向角(弧度), 约45度
        },

        # 停车场布局参数
        'parking_lot': {
            'geometry': {
                'spot_width': 2.5,   # 停车位宽度(m)
                'spot_length': 5.0,  # 停车位长度(m)
                'lane_width': 8.0,  # 车道宽度(m)
            },
            'layout': {
                'total_columns': 6,  # 停车位列数
                'spots_per_row_top': 14,  # 顶部一排的停车位数量 (23-36)
                'spots_per_row_middle': 12,  # 中间每排的停车位数量 (41-52, 53-64)
                'spots_per_row_bottom': 14,  # 底部一排的停车位数量 (01-14)
                'empty_spots': [26],  # 空白的停车位编号
                'static_ratio': 0.7,  # 静态车辆占用率
            },
            'margin': 5.0,  # 边界margin
            'wall_thickness': 0.5,  # 墙壁厚度
            'entrance_width': 12.0,  # 入口宽度(m)
            'entrance_margin': 15.0  # 入口外的安全距离(m)
        },

        # 路径规划参数
        'path_planning': {
            'default_algorithm': 'rrt_star',
            'algorithms': {
                'rrt': {
                    'step_size': 2.0,
                    'max_iterations': 10000
                },
                'rrt_star': {
                    'step_size': 2.0,
                    'max_iterations': 10000,
                    'rewire_factor': 1.5
                },
                'informed_rrt': {
                    'step_size': 2.0,
                    'max_iterations': 10000,
                    'focus_ratio': 1.0
                },
                'timed_rrt': {
                    'step_size': 2.0,
                    'max_iterations': 10000,
                    'robot_speed': 1.0
                }
            }
        },

        # 控制参数
        'control': {
            'default_method': 'pid',
            'methods': {
                'pid': {
                    'steer': {
                        'kp': 3.0,
                        'ki': 0.01,
                        'kd': 2.0
                    },
                    'speed': {
                        'kp': 5.0,
                        'ki': 2.0,
                        'kd': 0.05
                    }
                },
                'mpc': {
                    'horizon': 10,
                    'dt': 0.1,
                    'weights': {
                        'x': 1.0,
                        'y': 2.0,
                        'heading': 3.0
                    }
                },
                'lqr': {
                    'Q': [1.0, 10.0, 10.0],
                    'R': [0.1]
                }
            }
        }
    }

    # 如果提供了配置文件路径，从文件加载配置
    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                # 加载YAML配置
                user_config = yaml.safe_load(f)
                # 更新默认配置
                update_config(default_config, user_config)
                print(f"已加载配置文件: {config_path}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")

    return default_config

# 递归更新配置字典


def update_config(default_config: Dict, user_config: Dict) -> None:
    """递归更新配置字典"""
    for key, value in user_config.items():
        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
            update_config(default_config[key], value)
        else:
            default_config[key] = value


# 修改RRTStar以支持时间维度


def create_default_scene(width, height, config, env):
    """
    创建默认场景，包括停车场边界和障碍物，并直接添加到环境中

    Parameters:
    -----------
    width: float
        场景宽度
    height: float
        场景高度
    config: dict
        配置参数
    env: ParkingEnvironment
        停车场环境对象
    """
    # 获取配置参数
    parking_config = config.get('parking_lot', {})
    spot_width = parking_config.get('geometry', {}).get('spot_width', 2.5)
    spot_length = parking_config.get('geometry', {}).get('spot_length', 5.0)
    entrance_width = parking_config.get('entrance_width', 12.0)
    wall_thickness = parking_config.get('wall_thickness', 0.5)

    # 获取颜色设置
    wall_color = (80, 80, 80, 255)  # 墙壁颜色 - 深灰色
    spot_color = (220, 220, 220, 50)  # 停车位颜色 - 浅灰色半透明
    car_body_color = (50, 50, 50, 230)  # 车身颜色 - 深灰色

    # 添加边界墙
    # 上边界
    env.add_obstacle(
        x=width/2,
        y=0,
        obstacle_type="rectangle",
        width=width,
        height=wall_thickness,
        angle=0,
        color=wall_color
    )

    # 下边界（左侧）
    env.add_obstacle(
        x=(width-entrance_width)/4,
        y=height,
        obstacle_type="rectangle",
        width=(width-entrance_width)/2,
        height=wall_thickness,
        angle=0,
        color=wall_color
    )

    # 下边界（右侧）
    env.add_obstacle(
        x=width-(width-entrance_width)/4,
        y=height,
        obstacle_type="rectangle",
        width=(width-entrance_width)/2,
        height=wall_thickness,
        angle=0,
        color=wall_color
    )

    # 左边界
    env.add_obstacle(
        x=0,
        y=height/2,
        obstacle_type="rectangle",
        width=wall_thickness,
        height=height,
        angle=0,
        color=wall_color
    )

    # 右边界
    env.add_obstacle(
        x=width,
        y=height/2,
        obstacle_type="rectangle",
        width=wall_thickness,
        height=height,
        angle=0,
        color=wall_color
    )

    # 创建停车位布局
    # 左侧停车区 - 纵向停车位
    left_spots_x = spot_width * 1.5  # 靠近左边界
    for i in range(12):  # 增加停车位数量
        y_pos = (i + 1) * (spot_length + 0.5)  # 从下往上布置停车位
        # 添加停车位标记（不作为障碍物）
        env.add_obstacle(
            x=left_spots_x,
            y=y_pos,
            obstacle_type="rectangle",
            width=spot_width,
            height=spot_length,
            angle=0,  # 纵向停车位
            color=spot_color,
            is_parking_spot=True,
            occupied=(i % 2 == 0),  # 增加占用率，每隔一个停车位放一辆车
            is_filled=False,  # 不填充停车位
            line_width=2  # 增加线宽使标记更明显
        )
        # 如果停车位被占用，添加车辆
        if i % 2 == 0:
            env.add_obstacle(
                x=left_spots_x,
                y=y_pos,
                obstacle_type="rectangle",
                width=spot_width * 0.8,
                height=spot_length * 0.8,
                angle=0,  # 纵向停车
                color=car_body_color
            )

    # 右侧停车区 - 纵向停车位
    right_spots_x = width - spot_width * 1.5
    for i in range(12):  # 增加停车位数量
        y_pos = (i + 1) * (spot_length + 0.5)  # 从下往上布置停车位
        # 添加停车位标记（不作为障碍物）
        env.add_obstacle(
            x=right_spots_x,
            y=y_pos,
            obstacle_type="rectangle",
            width=spot_width,
            height=spot_length,
            angle=0,  # 纵向停车位
            color=spot_color,
            is_parking_spot=True,
            occupied=(i % 2 == 1),  # 错开放置车辆
            is_filled=False,  # 不填充停车位
            line_width=2  # 增加线宽使标记更明显
        )
        # 如果停车位被占用，添加车辆
        if i % 2 == 1:
            env.add_obstacle(
                x=right_spots_x,
                y=y_pos,
                obstacle_type="rectangle",
                width=spot_width * 0.8,
                height=spot_length * 0.8,
                angle=0,  # 纵向停车
                color=car_body_color
            )

    # 中间停车区（双排）- 横向停车位
    middle_left_x = width/2 - spot_length * 0.7  # 调整位置以适应横向停车位
    middle_right_x = width/2 + spot_length * 0.7  # 调整位置以适应横向停车位
    for i in range(10):  # 增加停车位数量
        y_pos = (i + 1) * (spot_width + 1.0)  # 从下往上布置停车位，调整间距
        # 左排停车位 - 横向
        env.add_obstacle(
            x=middle_left_x,
            y=y_pos,
            obstacle_type="rectangle",
            width=spot_length,  # 交换宽度和长度
            height=spot_width,  # 交换宽度和长度
            angle=0,  # 横向停车位
            color=spot_color,
            is_parking_spot=True,
            occupied=(i % 3 == 2),
            is_filled=False,  # 不填充停车位
            line_width=2  # 增加线宽使标记更明显
        )
        if i % 3 == 2:
            env.add_obstacle(
                x=middle_left_x,
                y=y_pos,
                obstacle_type="rectangle",
                width=spot_length * 0.8,  # 交换宽度和长度
                height=spot_width * 0.8,  # 交换宽度和长度
                angle=0,  # 横向停车
                color=car_body_color
            )

        # 右排停车位 - 横向
        env.add_obstacle(
            x=middle_right_x,
            y=y_pos,
            obstacle_type="rectangle",
            width=spot_length,  # 交换宽度和长度
            height=spot_width,  # 交换宽度和长度
            angle=0,  # 横向停车位
            color=spot_color,
            is_parking_spot=True,
            occupied=(i % 3 == 0),
            is_filled=False,  # 不填充停车位
            line_width=2  # 增加线宽使标记更明显
        )
        if i % 3 == 0:
            env.add_obstacle(
                x=middle_right_x,
                y=y_pos,
                obstacle_type="rectangle",
                width=spot_length * 0.8,  # 交换宽度和长度
                height=spot_width * 0.8,  # 交换宽度和长度
                angle=0,  # 横向停车
                color=car_body_color
            )


def create_parking_scenario(use_random_scene=False, config=None):
    """创建停车场场景，包括环境、起点和目标点"""
    print("使用默认停车场场景")

    # 如果没有提供配置，使用默认配置
    if config is None:
        config = load_config()

    # 获取窗口尺寸
    window_width = config['window']['width']
    window_height = config['window']['height']

    # 获取缩放比例
    scale = config['simulation']['scale']

    # 计算实际环境尺寸（米）
    env_width = window_width / scale
    env_height = window_height / scale

    # 创建环境
    env = ParkingEnvironment(env_width, env_height)

    # 创建默认场景并添加到环境中
    create_default_scene(env_width, env_height, config, env)

    # 设置起点在入口中心
    start_x = env_width / 2  # 入口中心的x坐标
    start_y = env_height - 5.0  # 距离下边界5米
    start = (start_x, start_y)

    # 创建可用停车位列表（未占用的停车位）
    parking_spots = []
    spot_width = config['parking_lot']['geometry'].get('spot_width', 2.5)
    spot_length = config['parking_lot']['geometry'].get('spot_length', 5.0)

    # 左侧停车区
    left_spots_x = spot_width * 1.5
    for i in range(12):
        y_pos = (i + 1) * (spot_length + 0.5)  # 从下往上布置停车位
        if i % 2 != 0:  # 不被占用的停车位
            parking_spots.append((left_spots_x, y_pos, 0))  # 添加朝向信息

    # 右侧停车区
    right_spots_x = env_width - spot_width * 1.5
    for i in range(12):
        y_pos = (i + 1) * (spot_length + 0.5)  # 从下往上布置停车位
        if i % 2 != 1:  # 不被占用的停车位
            parking_spots.append((right_spots_x, y_pos, 0))  # 添加朝向信息

    # 中间停车区
    middle_left_x = env_width/2 - spot_length * 0.7  # 调整位置以适应横向停车位
    middle_right_x = env_width/2 + spot_length * 0.7  # 调整位置以适应横向停车位
    for i in range(10):
        y_pos = (i + 1) * (spot_width + 1.0)  # 从下往上布置停车位，调整间距
        if i % 3 != 2:  # 左侧不被占用的停车位
            parking_spots.append((middle_left_x, y_pos, 0))  # 添加朝向信息
        if i % 3 != 0:  # 右侧不被占用的停车位
            parking_spots.append((middle_right_x, y_pos, 0))  # 添加朝向信息

    # 随机选择一个未占用的停车位作为目标
    if parking_spots:
        goal_x, goal_y, goal_orientation = random.choice(parking_spots)
        goal = (goal_x, goal_y)
        print("随机选择了一个未占用的停车位作为目标点")
    else:
        # 如果没有可用停车位，设置一个默认目标
        goal_x = env_width * 0.8
        goal_y = env_height * 0.5
        goal = (goal_x, goal_y)
        goal_orientation = 0  # 默认朝上
        print("警告：没有找到未占用的停车位，使用默认目标点")

    print(f"起点: {start}")
    print(f"目标点: {goal}, 朝向: {goal_orientation}°")

    return env, start, goal, goal_orientation


def get_algorithm_specific_params(algorithm: str, args, ) -> Dict[str, Any]:
    """获取算法特定的参数"""
    base_params = {
        'max_iterations': args.iterations if args.iterations is not None else 10000,
        'step_size': args.step_size if args.step_size is not None else 2.0
    }

    params = {
        'astar': {'resolution': 0.5, 'diagonal_movement': True, 'weight': 1.0},
        'rrt': base_params,
        'rrt_star': {**base_params, 'rewire_factor': 1.5},
        'informed_rrt': {**base_params, 'focus_factor': 1.0},
        'timed_rrt': {**base_params, 'robot_speed': args.robot_speed},
        'dijkstra': {'resolution': 1.0, 'diagonal_movement': True},
        'dstar_lite': {'resolution': 1.0, 'diagonal_movement': True},
        'theta_star': {'resolution': 1.0, 'diagonal_movement': True}
    }

    return params.get(algorithm, {})


def create_planner(algorithm: str, start: tuple, goal: tuple, env: Environment,
                   args, vehicle_width, vehicle_length):
    """创建路径规划器"""
    # 获取车辆尺寸参数
    vehicle_width = vehicle_width  # 车辆宽度
    vehicle_length = vehicle_length  # 车辆长度

    # 基本参数，所有规划器都需要
    common_params = {
        'start': start,
        'goal': goal,
        'env': env,
        'vehicle_width': vehicle_width,  # 所有算法都需要车辆参数
        'vehicle_length': vehicle_length
    }

    # 获取算法特定参数
    algorithm_params = get_algorithm_specific_params(algorithm, args)

    planners = {
        'astar': AStar,
        'rrt': RRT,
        'rrt_star': RRTStar,
        'informed_rrt': InformedRRTStar,
        'timed_rrt': TimedRRTStar,
        'dijkstra': Dijkstra,
        'dstar_lite': DStarLite,
        'theta_star': ThetaStar
    }

    if algorithm not in planners:
        raise ValueError(f"不支持的算法: {algorithm}")

    planner_class = planners[algorithm]

    return planner_class(**common_params, **algorithm_params)


def try_plan_path(
    planner,
    max_retries: int = 10
) -> Optional[List[Tuple[float, float]]]:
    """尝试规划路径，支持多次重试

    参数:
        planner: 路径规划器
        max_retries: 最大重试次数

    返回:
        规划的路径，如果失败则返回None
    """

    for i in range(max_retries):
        print(f"第 {i+1} 次尝试...")
        path = planner.plan()
        if path:
            print(f"DEBUG: 成功规划路径，路径点数: {len(path)}")
            return path
        print(f"第 {i+1} 次尝试失败，继续尝试...")

    print(f"经过 {max_retries} 次尝试后仍未找到可行路径")
    return None


def check_position_valid(env: Environment, pos: tuple, vehicle_width,
                         vehicle_length, margin: float = 5.0) -> bool:
    """检查位置是否有效（使用A*算法验证可达性）

    参数:
        env: 环境对象
        pos: 位置坐标(x, y)
        vehicle_width: 车辆宽度
        vehicle_length: 车辆长度
        margin: 安全边距

    返回:
        位置是否有效
    """
    x, y = pos

    # 基本碰撞检测 - 使用车辆碰撞检测而不是简单的点碰撞检测
    temp_vehicle = VehicleModel(x, y, 0, vehicle_length, vehicle_width)
    collision_info = check_vehicle_collision(temp_vehicle, env)
    if collision_info['collision']:
        return False

    # 使用A*验证从当前位置到四周的可达性
    test_points = [
        (x + margin, y),      # 右
        (x - margin, y),      # 左
        (x, y + margin),      # 上
        (x, y - margin),      # 下
        (x + margin, y + margin),  # 右上
        (x + margin, y - margin),  # 右下
        (x - margin, y + margin),  # 左上
        (x - margin, y - margin)   # 左下
    ]

    args = argparse.Namespace(
        algorithm='dijkstra',
        iterations=500,  # 减少迭代次数以提高速度
        step_size=0.5,  # 减小步长以提高精度
        robot_speed=3.0
    )

    # 检查是否至少有三个方向可达
    reachable_directions = 0
    min_required = 3  # 降低要求，只需要三个方向可达

    for test_point in test_points:
        # 如果已经找到足够的可达方向，提前返回
        if reachable_directions >= min_required:
            return True

        # 确保测试点在环境范围内且不在障碍物内
        if (0 <= test_point[0] <= env.width and
                0 <= test_point[1] <= env.height):

            # 使用车辆碰撞检测而不是简单的点碰撞检测
            test_vehicle = VehicleModel(
                test_point[0], test_point[1], 0, vehicle_length, vehicle_width)
            test_collision = check_vehicle_collision(test_vehicle, env)

            if not test_collision['collision']:
                test_planner = create_planner(
                    'dijkstra', pos, test_point, env, args, vehicle_width, vehicle_length)
                path = test_planner.plan()
                if path:
                    reachable_directions += 1

    return reachable_directions >= min_required


def check_path_feasibility(
    env: Environment,
    start: tuple,
    goal: tuple,
    algorithm: str,
    args,
    vehicle_width,
    vehicle_length
) -> bool:
    """检查路径可行性

    参数:
        env: 环境对象
        start: 起点坐标
        goal: 终点坐标
        algorithm: 使用的规划算法
        args: 算法参数
        vehicle_width: 车辆宽度
        vehicle_length: 车辆长度

    返回:
        路径是否可行
    """
    # 创建规划器进行测试
    test_planner = create_planner(
        algorithm, start, goal, env, args, vehicle_width, vehicle_length)

    # 使用较大的迭代次数进行测试
    test_planner.max_iterations = args.iterations * 2

    # 尝试规划路径
    path = test_planner.plan()
    if not path:
        print("路径规划测试失败：无法找到可行路径")
        return False

    # 检查路径是否与障碍物碰撞
    collision_info = check_path_collision(
        path, env, vehicle_length, vehicle_width)
    if collision_info['collision']:
        print("路径规划测试失败：路径与障碍物碰撞")
        return False

    # 验证路径连续性
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist > args.step_size * 1.5:  # 允许一定的误差
            print(f"路径规划测试失败：路径不连续，在点 {i} 和 {i+1} 之间的距离为 {dist}")
            return False

    return True


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='停车场路径规划仿真')

    parser.add_argument(
        '--config',
        type=str,
        default='config/parking_config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['rrt', 'rrt_star', 'informed_rrt',
                 'timed_rrt', 'dijkstra', 'dstar_lite', 'theta_star', 'astar'],
        default='dijkstra',
        help='路径规划算法'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='最大迭代次数'
    )

    parser.add_argument(
        '--step_size',
        type=float,
        default=2,
        help='步长'
    )

    parser.add_argument(
        '--robot_speed',
        type=float,
        default=4,
        help='机器人速度'
    )

    parser.add_argument(
        '--random_scene',
        action='store_true',
        help='使用随机生成场景（默认使用默认场景）'
    )

    parser.add_argument(
        '--control_method',
        type=str,
        choices=['default', 'pid', 'mpc', 'lqr'],
        default='pid',
        help='车辆控制算法'
    )

    return parser.parse_args()


def main():
    """主函数：创建场景、规划路径并仿真"""
    # 解析命令行参数
    args = parse_args()

    # 加载配置文件
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.algorithm is not None:
        config['path_planning']['default_algorithm'] = args.algorithm
    if args.iterations is not None:
        for alg in config['path_planning']['algorithms']:
            config['path_planning']['algorithms'][alg]['max_iterations'] = args.iterations
    if args.step_size is not None:
        for alg in config['path_planning']['algorithms']:
            if 'step_size' in config['path_planning']['algorithms'][alg]:
                config['path_planning']['algorithms'][alg]['step_size'] = args.step_size
    if args.robot_speed is not None:
        config['path_planning']['algorithms']['timed_rrt']['robot_speed'] = args.robot_speed
    if args.control_method is not None:
        config['control']['default_method'] = args.control_method

    # 创建场景
    env, start, goal, goal_orientation = create_parking_scenario(
        use_random_scene=args.random_scene,
        config=config
    )

    # 创建仿真器并设置环境
    simulator = PygameSimulator({
        'window_width': config['window']['width'],
        'window_height': config['window']['height'],
        'window_title': config['window']['title'],
        'scale': config['simulation']['scale'],
        'fps': config['simulation']['fps'],
        'dt': config['simulation']['dt'],
        'lookahead': config['simulation']['lookahead'],
        'vehicle': {
            'length': config['vehicle']['length'],
            'width': config['vehicle']['width'],
            'wheelbase': config['vehicle']['wheelbase']
        }
    })
    simulator.set_environment(env)

    # 启动交互式规划
    interactive_planning(simulator, env, start, args)


def interactive_planning(simulator, env: ParkingEnvironment, start: Tuple[float, float], args):
    """
    交互式路径规划入口函数

    参数:
        simulator: 仿真器
        env: 停车场环境
        start: 起点坐标
        args: 命令行参数
    """
    # 打印调试信息
    print("\n===== interactive_planning 函数 =====")
    print(
        f"原始simulator参数: width={simulator.width}, height={simulator.height}, scale={simulator.scale}")
    print(
        f"原始simulator参数: offset_x={simulator.offset_x}, offset_y={simulator.offset_y}")
    print(f"环境尺寸: width={env.width}, height={env.height}")

    # 计算合适的偏移值，使环境居中显示
    # 计算环境在屏幕上的像素尺寸
    env_width_px = env.width * simulator.scale
    env_height_px = env.height * simulator.scale

    # 计算居中显示所需的偏移
    offset_x = (simulator.width - env_width_px) / 2
    offset_y = (simulator.height - env_height_px) / 2

    # 确保偏移值不为负
    offset_x = max(0, offset_x)
    offset_y = max(0, offset_y)

    print(f"计算的偏移值: offset_x={offset_x}, offset_y={offset_y}")
    print(f"环境像素尺寸: width={env_width_px}, height={env_height_px}")
    print("================================\n")

    # 从simulator的配置中提取需要的参数，保持完整的配置结构
    planner_config = {
        'window_width': simulator.width,
        'window_height': simulator.height,
        'window_title': simulator.config.get('window', {}).get('title', "停车场路径规划仿真器"),
        'scale': simulator.scale,
        'fps': simulator.config.get('simulation', {}).get('fps', 60),
        'dt': simulator.config.get('simulation', {}).get('dt', 0.05),
        'lookahead': simulator.config.get('simulation', {}).get('lookahead', 5.0),
        'offset_x': offset_x,  # 使用计算的偏移值
        'offset_y': offset_y,  # 使用计算的偏移值
        'vehicle': {
            'length': simulator.vehicle.length,
            'width': simulator.vehicle.width,
            'wheelbase': simulator.vehicle.wheelbase,
            'max_speed': simulator.vehicle.max_speed,
            'max_accel': simulator.vehicle.max_accel,
            'max_brake': simulator.vehicle.max_brake,
            'max_steer': simulator.vehicle.max_steer
        }
    }

    # 创建并运行交互式规划器
    planner = InteractivePlanner(planner_config, env, start, args)
    planner.run()


class InteractivePlanner(PygameSimulator):
    """交互式路径规划器，继承自PygameSimulator"""

    def __init__(self, config, env: ParkingEnvironment, start: Tuple[float, float], args):
        """初始化交互式规划器"""
        # 确保pygame已初始化
        if not pygame.get_init():
            pygame.init()

        # 打印传入的配置信息
        print("\n===== InteractivePlanner 初始化 =====")
        print(
            f"传入配置: window_width={config.get('window_width')}, window_height={config.get('window_height')}")
        print(
            f"传入配置: scale={config.get('scale')}, offset_x={config.get('offset_x')}, offset_y={config.get('offset_y')}")
        print(f"环境尺寸: width={env.width}, height={env.height}")
        print(f"起点坐标: start={start}")

        # 保存原始配置值，以便在父类初始化后恢复
        self.original_width = config['window_width']
        self.original_height = config['window_height']
        self.original_scale = config['scale']
        self.original_offset_x = config.get('offset_x', 0)
        self.original_offset_y = config.get('offset_y', 0)

        # 构造完整的父类配置
        simulator_config = {
            'window': {
                'width': config['window_width'],
                'height': config['window_height'],
                'title': config['window_title']
            },
            'simulation': {
                'scale': config['scale'],
                'fps': config['fps'],
                'dt': config['dt'],
                'lookahead': config['lookahead'],
                'simulation_speed': 1.0
            },
            'vehicle': config['vehicle']
        }

        # 初始化父类
        super().__init__(simulator_config)

        # 打印父类初始化后的参数
        print("\n===== 父类初始化后 =====")
        print(
            f"父类参数: width={self.width}, height={self.height}, scale={self.scale}")
        print(f"父类参数: offset_x={self.offset_x}, offset_y={self.offset_y}")
        print(
            f"计算值: 环境宽度(米)={self.width/self.scale}, 环境高度(米)={self.height/self.scale}")

        # 恢复原始配置值，防止父类覆盖
        self.width = self.original_width
        self.height = self.original_height
        self.scale = self.original_scale
        self.offset_x = self.original_offset_x
        self.offset_y = self.original_offset_y

        # 重新创建屏幕，确保尺寸正确
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(config['window_title'])

        # 设置环境和起点
        self.environment = env
        self.start = start
        self.args = args
        self.goal = None
        self.path = None

        # 从配置中获取参数
        self.dt = config.get('dt', 0.05)

        # 打印最终使用的参数
        print("\n===== 最终参数 =====")
        print(
            f"最终参数: scale={self.scale}, offset_x={self.offset_x}, offset_y={self.offset_y}")
        print(f"屏幕尺寸: width={self.width}, height={self.height}")
        print("================================\n")

        # 状态信息
        self.status_text = "请右键点击选择目标点"
        self.status_color = (0, 0, 255)  # 蓝色
        self.simulating = False
        self.collision_detected = False
        self.collision_info = None

        # 初始化路径跟踪器
        self.follower = PathFollower(
            lookahead=config.get('lookahead', 5.0),
            control_method=args.control_method if args.control_method else "default"
        )

        # 初始化车辆位置和朝向
        self._reset_vehicle()

        # 添加按键提示信息
        self.key_hints = [
            "R: 重置车辆",
            "C: 切换控制方法",
            "P: 切换规划算法",
            "S: 切换转向模式",
            "空格: 暂停/继续",
            "右键: 选择目标点",
            "ESC: 退出"
        ]
        self.hint_color = (50, 50, 50)  # 深灰色
        self.hint_font_size = 20

        # 添加泊车相关属性
        self.parking_spot = None  # 当前选中的停车位
        self.parking_type = None  # 停车类型：parallel或perpendicular
        self.approach_points = []  # 泊车路径的关键点
        self.parking_path = []  # 完整的泊车路径

    def _reset_vehicle(self):
        """重置车辆位置"""
        self.vehicle.x, self.vehicle.y = self.start
        self.vehicle.heading = math.pi * 3 / 2  # 朝下
        self.vehicle.speed = 0.0
        self.vehicle.trajectory = [(self.start[0], self.start[1])]
        self.simulating = False
        self.collision_detected = False
        self.collision_info = None
        self.status_text = "车辆已重置到起点"
        self.status_color = (0, 128, 0)  # 绿色
        print("车辆已重置到起点")

    def _switch_control_method(self):
        """切换控制方法"""
        if not hasattr(self, 'control_methods'):
            self.control_methods = ['default', 'pid', 'mpc', 'lqr']

        try:
            current_index = self.control_methods.index(
                self.follower.control_method)
            next_index = (current_index + 1) % len(self.control_methods)
            self.follower.set_control_method(self.control_methods[next_index])
            self.status_text = f"已切换到{self.control_methods[next_index]}控制方法"
            self.status_color = (0, 128, 0)  # 绿色
            print(f"已切换到{self.control_methods[next_index]}控制方法")
        except ValueError:
            print("当前控制方法未找到，重置为默认方法")
            self.follower.set_control_method('default')

    def _switch_planning_algorithm(self):
        """切换规划算法"""
        algorithms = ['rrt', 'rrt_star', 'informed_rrt', 'timed_rrt',
                      'dijkstra', 'dstar_lite', 'theta_star', 'astar']
        current_index = algorithms.index(self.args.algorithm)
        next_index = (current_index + 1) % len(algorithms)
        self.args.algorithm = algorithms[next_index]
        self.status_text = f"已切换到{self.args.algorithm}规划算法"
        self.status_color = (0, 128, 0)  # 绿色
        print(f"已切换到{self.args.algorithm}规划算法")

        # 如果有目标点，重新规划路径
        if self.goal and not self.simulating:
            self.path = self._plan_path()
            if self.path:
                self.follower.set_path(self.path)

    def _switch_steering_mode(self):
        """切换转向模式"""
        modes = ["normal", "counter", "crab"]
        current_index = modes.index(self.vehicle.steering_mode)
        next_index = (current_index + 1) % len(modes)
        self.vehicle.set_steering_mode(modes[next_index])
        self.status_text = f"已切换到{modes[next_index]}转向模式"
        self.status_color = (0, 128, 0)  # 绿色
        print(f"已切换到{modes[next_index]}转向模式")

    def _plan_path(self):
        """规划路径"""
        if not self.goal:
            return None

        # 识别停车位类型
        self.parking_spot = self.environment.find_parking_spot(self.goal)
        if not self.parking_spot:
            self.status_text = "请选择一个有效的停车位"
            self.status_color = (255, 0, 0)  # 红色
            return None

        try:
            # 根据停车位朝向判断停车类型
            spot_angle = math.degrees(self.parking_spot.angle)
            if abs(spot_angle) < 30 or abs(spot_angle - 180) < 30:
                self.parking_type = 'parallel'  # 平行停车位
            else:
                self.parking_type = 'perpendicular'  # 垂直停车位

            # 生成泊车路径
            if self.parking_type == 'parallel':
                return self._plan_parallel_parking_path()
            else:
                return self._plan_perpendicular_parking_path()
        except Exception as e:
            print(f"规划泊车路径时出错: {e}")
            self.status_text = "规划泊车路径失败"
            self.status_color = (255, 0, 0)  # 红色
            return None

    def _plan_parallel_parking_path(self):
        """规划平行泊车路径"""
        if not self.parking_spot:
            return None

        # 获取停车位信息
        spot = self.parking_spot
        spot_center_x = spot.x
        spot_center_y = spot.y
        spot_angle = spot.angle

        # 计算车辆尺寸和安全边距
        vehicle_length = self.vehicle.length
        vehicle_width = self.vehicle.width  # 用于计算转向半径和避障
        safety_margin = 0.5  # 用于碰撞检测和路径规划的安全边距

        # 使用安全边距调整目标位置
        goal_x = spot_center_x + safety_margin * \
            math.cos(spot_angle + math.pi/2)
        goal_y = spot_center_y + safety_margin * \
            math.sin(spot_angle + math.pi/2)
        goal_heading = spot_angle

        # 2. 倒车起始点（在停车位前方）
        approach_distance = 1.5 * vehicle_length  # 预留足够的操作空间
        approach_x = spot_center_x + approach_distance * math.cos(spot_angle)
        approach_y = spot_center_y + approach_distance * math.sin(spot_angle)
        approach_heading = spot_angle

        # 3. 倒车转向点
        turn_distance = vehicle_length
        # 根据车辆宽度计算转向偏移量
        turn_offset = vehicle_width * 1.2  # 增加20%的余量确保转向空间
        turn_x = approach_x - turn_distance * \
            math.cos(spot_angle) + turn_offset * \
            math.cos(spot_angle + math.pi/2)
        turn_y = approach_y - turn_distance * \
            math.sin(spot_angle) + turn_offset * \
            math.sin(spot_angle + math.pi/2)

        # 保存关键点
        self.approach_points = [
            (approach_x, approach_y, approach_heading),
            (turn_x, turn_y, approach_heading),
            (goal_x, goal_y, goal_heading)
        ]

        # 生成完整路径
        path = []

        # 1. 从当前位置到接近点的路径
        start_to_approach = self._generate_smooth_path(
            (self.vehicle.x, self.vehicle.y),
            (approach_x, approach_y),
            self.vehicle.heading,
            approach_heading,
            10
        )
        path.extend(start_to_approach)

        # 2. 从接近点到转向点的路径
        approach_to_turn = self._generate_smooth_path(
            (approach_x, approach_y),
            (turn_x, turn_y),
            approach_heading,
            approach_heading,
            5
        )
        path.extend(approach_to_turn)

        # 3. 从转向点到目标点的路径（使用贝塞尔曲线生成平滑的转向路径）
        turn_to_goal = self._generate_parking_curve(
            (turn_x, turn_y),
            (goal_x, goal_y),
            approach_heading,
            goal_heading
        )
        path.extend(turn_to_goal)

        # 保存完整路径
        self.parking_path = path

        # 设置路径跟踪器为泊车模式
        self.follower.set_control_method('parking')
        if not self.follower.set_parking_type('parallel'):
            print("设置平行泊车模式失败")
            return None

        return path

    def _plan_perpendicular_parking_path(self):
        """规划垂直泊车路径"""
        if not self.parking_spot:
            return None

        # 获取停车位信息
        spot = self.parking_spot
        spot_center_x = spot.x
        spot_center_y = spot.y
        spot_angle = spot.angle

        # 计算车辆尺寸和安全边距
        vehicle_length = self.vehicle.length
        vehicle_width = self.vehicle.width  # 用于计算入库轨迹
        safety_margin = 0.5  # 用于调整最终停车位置

        # 使用安全边距调整目标位置
        goal_x = spot_center_x + safety_margin * math.cos(spot_angle)
        goal_y = spot_center_y + safety_margin * math.sin(spot_angle)
        goal_heading = spot_angle

        # 2. 倒车起始点（在停车位前方）
        # 根据车辆宽度调整接近距离，确保有足够的转向空间
        approach_distance = 2.0 * vehicle_length + vehicle_width * 0.5
        approach_x = spot_center_x + approach_distance * math.cos(spot_angle)
        approach_y = spot_center_y + approach_distance * math.sin(spot_angle)
        approach_heading = spot_angle

        # 保存关键点
        self.approach_points = [
            (approach_x, approach_y, approach_heading),
            (goal_x, goal_y, goal_heading)
        ]

        # 生成完整路径
        path = []

        # 1. 从当前位置到接近点的路径
        start_to_approach = self._generate_smooth_path(
            (self.vehicle.x, self.vehicle.y),
            (approach_x, approach_y),
            self.vehicle.heading,
            approach_heading,
            10
        )
        path.extend(start_to_approach)

        # 2. 从接近点到目标点的路径
        approach_to_goal = self._generate_parking_curve(
            (approach_x, approach_y),
            (goal_x, goal_y),
            approach_heading,
            goal_heading
        )
        path.extend(approach_to_goal)

        # 保存完整路径
        self.parking_path = path

        # 设置路径跟踪器为泊车模式
        self.follower.set_control_method('parking')
        if not self.follower.set_parking_type('perpendicular'):
            print("设置垂直泊车模式失败")
            return None

        return path

    def _generate_smooth_path(self, start, end, start_heading, end_heading, points):
        """生成平滑路径"""
        path = []
        for i in range(points):
            t = i / (points - 1)
            # 使用简单的线性插值
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            path.append((x, y))
        return path

    def _generate_parking_curve(self, start, end, start_heading, end_heading):
        """生成泊车曲线（使用三次贝塞尔曲线）"""
        # 计算控制点
        distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        control_length = distance * 0.5

        # 起点控制点
        control1_x = start[0] + control_length * math.cos(start_heading)
        control1_y = start[1] + control_length * math.sin(start_heading)

        # 终点控制点
        control2_x = end[0] - control_length * math.cos(end_heading)
        control2_y = end[1] - control_length * math.sin(end_heading)

        # 生成贝塞尔曲线点
        path = []
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            # 三次贝塞尔曲线公式
            x = (1-t)**3 * start[0] + \
                3*(1-t)**2 * t * control1_x + \
                3*(1-t) * t**2 * control2_x + \
                t**3 * end[0]
            y = (1-t)**3 * start[1] + \
                3*(1-t)**2 * t * control1_y + \
                3*(1-t) * t**2 * control2_y + \
                t**3 * end[1]
            path.append((x, y))
        return path

    def _handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # 右键点击
                    # 获取鼠标位置并转换为世界坐标
                    mouse_x, mouse_y = event.pos
                    world_x = (mouse_x - self.offset_x) / self.scale
                    world_y = (mouse_y - self.offset_y) / self.scale

                    # 检查坐标是否在环境范围内
                    if 0 <= world_x <= self.environment.width and 0 <= world_y <= self.environment.height:
                        # 检查是否点击在停车位内
                        spot = self.environment.find_parking_spot(
                            (world_x, world_y))
                        if spot and not spot.occupied:
                            self.goal = (world_x, world_y)
                            print(f"设置目标点: {self.goal}")

                            # 自动切换到泊车控制模式
                            self.follower.set_control_method('parking')
                            print("已切换到泊车控制模式")

                            # 规划泊车路径
                            self.path = self._plan_path()
                            if self.path:
                                self.follower.set_path(self.path)
                                self.simulating = True
                                self.status_text = "开始自动泊车"
                                self.status_color = (0, 128, 0)  # 绿色
                        else:
                            self.status_text = "请选择一个空闲的停车位"
                            self.status_color = (255, 165, 0)  # 橙色
                    else:
                        self.status_text = "点击位置超出环境范围"
                        self.status_color = (255, 0, 0)  # 红色

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # R键重置车辆
                    self._reset_vehicle()
                    return True
                elif event.key == pygame.K_c:  # C键切换控制方法
                    self._switch_control_method()
                    return True
                elif event.key == pygame.K_p:  # P键切换规划算法
                    self._switch_planning_algorithm()
                    return True
                elif event.key == pygame.K_s:  # S键切换转向模式
                    self._switch_steering_mode()
                    return True
                elif event.key == pygame.K_t:  # T键重置仿真
                    self._reset_simulation()
                    return True
                elif event.key == pygame.K_SPACE:  # 空格键暂停/继续
                    self.simulating = not self.simulating
                    self.status_text = "仿真已暂停" if not self.simulating else "仿真已继续"
                    self.status_color = (
                        0, 0, 255) if not self.simulating else (0, 128, 0)
                    return True
                elif event.key == pygame.K_ESCAPE:  # ESC键退出
                    self.running = False
                    return False

        return True

    def update(self):
        """更新仿真状态"""
        if not self.simulating or self.collision_detected:
            return

        # 计算控制输入
        throttle, brake, steer = self.follower.get_control(self.vehicle)

        # 更新车辆状态
        self.vehicle.update(throttle, brake, steer, self.dt)

        # 检查碰撞和安全边界
        if self.environment is not None:
            self.collision_info = check_vehicle_collision(
                self.vehicle, self.environment)
            if self.collision_info['collision']:
                self.collision_detected = True
                self.simulating = False
                self.status_text = "检测到碰撞：与障碍物相撞！按R键重置车辆位置"
                self.status_color = (255, 0, 0)  # 红色
                print(f"检测到碰撞！位置: {self.collision_info['position']}")
                return
            elif self.collision_info['safety_warning']:
                self.status_text = "警告：进入安全边界区域"
                self.status_color = (255, 165, 0)  # 橙色
                print(f"安全警告！位置: {self.collision_info['position']}")

        # 检查是否到达终点
        if self.goal:
            # 计算到目标点的距离和朝向差异
            dx = self.goal[0] - self.vehicle.x
            dy = self.goal[1] - self.vehicle.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)

            # 如果车辆已经非常接近目标点且速度很小，认为已到达
            if distance_to_goal < 0.5 and self.vehicle.speed < 0.1:  # 距离阈值0.5米，速度阈值0.1m/s
                self.simulating = False
                self.status_text = "到达目标点！按T键重新选择目标点"
                self.status_color = (0, 255, 0)  # 绿色
                print("车辆已到达目标点")

    def draw(self):
        """绘制场景"""
        # 清空屏幕
        self.screen.fill((255, 255, 255))  # 白色背景

        # 绘制信息 - 坐标系和边界
        self._draw_info()

        # 绘制环境
        if self.environment is not None:
            self._draw_environment()

        # 绘制目标点
        if self.goal:
            goal_x = self.goal[0] * self.scale + self.offset_x
            goal_y = self.goal[1] * self.scale + self.offset_y
            pygame.draw.circle(self.screen, (255, 0, 0),
                               (int(goal_x), int(goal_y)), 10)

        # 如果有路径，绘制路径
        if self.path:
            for i in range(len(self.path) - 1):
                p1_x = self.path[i][0] * self.scale + self.offset_x
                p1_y = self.path[i][1] * self.scale + self.offset_y
                p2_x = self.path[i+1][0] * self.scale + self.offset_x
                p2_y = self.path[i+1][1] * self.scale + self.offset_y
                pygame.draw.line(self.screen, (0, 0, 255),
                                 (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), 3)

        # 绘制车辆轨迹
        if hasattr(self.vehicle, 'trajectory') and len(self.vehicle.trajectory) > 1:
            for i in range(len(self.vehicle.trajectory) - 1):
                p1_x = self.vehicle.trajectory[i][0] * \
                    self.scale + self.offset_x
                p1_y = self.vehicle.trajectory[i][1] * \
                    self.scale + self.offset_y
                p2_x = self.vehicle.trajectory[i +
                                               1][0] * self.scale + self.offset_x
                p2_y = self.vehicle.trajectory[i +
                                               1][1] * self.scale + self.offset_y
                pygame.draw.line(self.screen, (0, 200, 0),
                                 (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), 2)

        # 绘制车辆
        self._draw_vehicle(self.screen, self.vehicle)

        # 绘制状态信息
        try:
            font = get_font(24)
            if font:
                text_surface = font.render(
                    self.status_text, True, self.status_color)
                text_rect = text_surface.get_rect()
                text_rect.centerx = self.screen.get_rect().centerx
                text_rect.top = 10
                self.screen.blit(text_surface, text_rect)
        except Exception as e:
            print(f"字体渲染错误: {e}")

        # 添加泊车路径绘制
        self._draw_parking_path()

        # 更新显示
        pygame.display.flip()

    def _draw_info(self):
        """绘制信息面板"""
        # 绘制环境边界
        env_width_px = self.environment.width * self.scale
        env_height_px = self.environment.height * self.scale

        # 绘制环境边界矩形 - 红色虚线
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(
                self.offset_x,
                self.offset_y,
                env_width_px,
                env_height_px
            ),
            1  # 减小线宽，使其不那么显眼
        )

        # 绘制原点标记
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(self.offset_x), int(self.offset_y)),
            3  # 减小原点标记大小
        )

        # 绘制车辆物理信息和操作提示
        try:
            font = get_font(16)
            if font:
                # 创建半透明背景
                info_surface = pygame.Surface(
                    (self.width, 100), pygame.SRCALPHA)
                info_surface.fill((240, 240, 240, 200))  # 浅灰色半透明背景
                self.screen.blit(info_surface, (0, self.height - 100))

                # 绘制分隔线
                pygame.draw.line(
                    self.screen,
                    (200, 200, 200),
                    (0, self.height - 100),
                    (self.width, self.height - 100),
                    1
                )

                # 显示车辆物理信息 - 左侧
                vehicle_info = [
                    f"车速: {self.vehicle.speed:.2f} m/s",
                    f"朝向: {math.degrees(self.vehicle.heading):.1f}°",
                    f"位置: ({self.vehicle.x:.1f}, {self.vehicle.y:.1f})"
                ]

                for i, info in enumerate(vehicle_info):
                    text_surface = font.render(info, True, (0, 0, 0))
                    self.screen.blit(
                        text_surface, (20, self.height - 90 + i * 20))

                # 显示控制信息 - 中间
                control_info = [
                    f"控制方法: {self.follower.control_method}",
                    f"规划算法: {self.args.algorithm}",
                    f"状态: {'运行中' if self.simulating else '已停止'}"
                ]

                for i, info in enumerate(control_info):
                    text_surface = font.render(info, True, (0, 0, 0))
                    text_rect = text_surface.get_rect()
                    text_rect.centerx = self.width // 2
                    text_rect.y = self.height - 90 + i * 20
                    self.screen.blit(text_surface, text_rect)

                # 显示操作提示 - 右侧
                key_hints = [
                    "R: 重置车辆  |  T: 重置仿真",
                    "C: 切换控制  |  P: 切换算法  |  S: 切换转向",
                    "空格: 暂停/继续  |  右键: 选择目标  |  ESC: 退出"
                ]

                for i, hint in enumerate(key_hints):
                    text_surface = font.render(hint, True, (0, 0, 0))
                    text_rect = text_surface.get_rect()
                    text_rect.right = self.width - 20
                    text_rect.y = self.height - 90 + i * 20
                    self.screen.blit(text_surface, text_rect)

                # 添加泊车相关信息
                if self.parking_type and self.follower.control_method == 'parking':
                    # 创建泊车信息面板背景
                    parking_info_surface = pygame.Surface(
                        (300, 120), pygame.SRCALPHA)
                    parking_info_surface.fill((240, 240, 240, 180))  # 浅灰色半透明背景
                    self.screen.blit(parking_info_surface,
                                     (self.width - 320, 20))

                    # 绘制泊车信息标题
                    title_font = get_font(18)
                    if title_font:
                        title_text = "自动泊车状态"
                        title_surface = title_font.render(
                            title_text, True, (0, 0, 0))
                        title_rect = title_surface.get_rect()
                        title_rect.centerx = self.width - 170
                        title_rect.y = 25
                        self.screen.blit(title_surface, title_rect)

                    # 绘制泊车类型和阶段
                    parking_type_text = f"停车类型: {'侧方停车' if self.parking_type == 'parallel' else '直角停车'}"
                    parking_phase_text = f"泊车阶段: {self._get_phase_name(self.follower.parking_phase)}"
                    gear_text = f"档位: {'倒车档' if self.follower.reverse_gear else '前进档'}"

                    # 计算到目标点的距离
                    if self.goal:
                        dx = self.goal[0] - self.vehicle.x
                        dy = self.goal[1] - self.vehicle.y
                        distance = math.sqrt(dx*dx + dy*dy)
                        distance_text = f"距目标: {distance:.2f} m"
                    else:
                        distance_text = "距目标: 未知"

                    # 显示泊车信息
                    parking_info = [
                        parking_type_text,
                        parking_phase_text,
                        gear_text,
                        distance_text
                    ]

                    for i, info in enumerate(parking_info):
                        text_surface = font.render(info, True, (0, 0, 0))
                        text_rect = text_surface.get_rect()
                        text_rect.x = self.width - 300
                        text_rect.y = 50 + i * 22
                        self.screen.blit(text_surface, text_rect)

        except Exception as e:
            print(f"信息渲染错误: {e}")

    def _get_phase_name(self, phase):
        """获取泊车阶段的中文名称"""
        phase_names = {
            'approach': '接近阶段',
            'reverse': '倒车入库',
            'adjust': '位置微调'
        }
        return phase_names.get(phase, phase)

    def _draw_vehicle(self, screen, vehicle):
        """绘制车辆"""
        # 获取车辆状态
        x = vehicle.x * self.scale + self.offset_x
        y = vehicle.y * self.scale + self.offset_y
        heading = vehicle.heading
        width = vehicle.width * self.scale
        length = vehicle.length * self.scale

        # 创建车辆表面
        surface = pygame.Surface((length, width), pygame.SRCALPHA)
        pygame.draw.rect(surface, (0, 0, 0), (0, 0, length, width), 2)

        # 旋转表面
        rotated_surface = pygame.transform.rotate(
            surface, -math.degrees(heading))
        rect = rotated_surface.get_rect(center=(int(x), int(y)))

        # 绘制到屏幕
        screen.blit(rotated_surface, rect)

    def _draw_environment(self):
        """绘制环境"""
        if self.environment is not None:
            for obs in self.environment.obstacles:
                # 转换坐标
                x = obs.x * self.scale + self.offset_x
                y = obs.y * self.scale + self.offset_y
                width = obs.width * self.scale
                height = obs.height * self.scale

                if obs.type == "rectangle":
                    # 创建旋转的矩形表面
                    surface = pygame.Surface((width, height), pygame.SRCALPHA)
                    if obs.is_filled:
                        pygame.draw.rect(surface, obs.color,
                                         (0, 0, width, height))
                    else:
                        pygame.draw.rect(surface, obs.color,
                                         (0, 0, width, height), obs.line_width)

                    # 旋转表面
                    rotated_surface = pygame.transform.rotate(
                        surface, -obs.angle)
                    # 获取旋转后的矩形
                    rect = rotated_surface.get_rect(center=(int(x), int(y)))
                    # 绘制到屏幕
                    self.screen.blit(rotated_surface, rect)
                else:  # circle
                    radius = int(obs.radius * self.scale)
                    if obs.is_filled:
                        pygame.draw.circle(
                            self.screen, obs.color, (int(x), int(y)), radius)
                    else:
                        pygame.draw.circle(
                            self.screen, obs.color, (int(x), int(y)), radius, obs.line_width)

    def _draw_parking_path(self):
        """绘制泊车路径"""
        if not self.parking_path:
            return

        # 绘制完整路径
        for i in range(len(self.parking_path) - 1):
            p1_x = self.parking_path[i][0] * self.scale + self.offset_x
            p1_y = self.parking_path[i][1] * self.scale + self.offset_y
            p2_x = self.parking_path[i+1][0] * self.scale + self.offset_x
            p2_y = self.parking_path[i+1][1] * self.scale + self.offset_y
            pygame.draw.line(self.screen, (0, 255, 0),
                             (int(p1_x), int(p1_y)),
                             (int(p2_x), int(p2_y)), 2)

        # 绘制关键点
        for point in self.approach_points:
            x = point[0] * self.scale + self.offset_x
            y = point[1] * self.scale + self.offset_y
            pygame.draw.circle(self.screen, (255, 165, 0),
                               (int(x), int(y)), 5)

    def _reset_simulation(self):
        """重置仿真状态"""
        self._reset_vehicle()
        self.goal = None
        self.path = None
        self.status_text = "仿真已重置，请右键点击选择目标点"
        self.status_color = (0, 0, 255)  # 蓝色
        print("仿真已重置")

    def run(self):
        """运行仿真"""
        if not pygame.get_init():
            pygame.init()

        clock = pygame.time.Clock()
        self.running = True

        while self.running:
            # 处理事件
            if not self._handle_events():
                break

            # 更新仿真状态
            self.update()

            # 绘制场景
            self.draw()

            # 控制帧率
            clock.tick(60)

        # 退出前清理
        pygame.quit()


if __name__ == "__main__":
    main()
