import numpy as np
import random
import argparse
import pygame
import yaml
import traceback
from typing import List, Tuple, Dict, Any, Optional
from rrt.astar import AStar
from simulation.environment import Environment
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar, TimedRRTStar
from rrt.informed_rrt import InformedRRTStar
from rrt.dijkstra import Dijkstra
from rrt.dstar_lite import DStarLite
from rrt.theta_star import ThetaStar
from simulation.pygame_simulator import ParkingEnvironment, PathFollower, PygameSimulator, VehicleModel
from simulation.pygame_simulator import check_vehicle_collision, check_path_collision
import math
from rrt.attention_dqn_rrt import AttentionDQNRRT


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
    # 确保max_iterations始终是整数
    max_iterations = int(
        args.iterations) if args.iterations is not None else 10000

    base_params = {
        'max_iterations': max_iterations,
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
        'theta_star': {'resolution': 1.0, 'diagonal_movement': True},
        'attention_dqn_rrt': {
            **base_params,
            'rewire_factor': 1.5,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 0.1,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'hidden_dim': 128
        }
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
        'theta_star': ThetaStar,
        'attention_dqn_rrt': AttentionDQNRRT
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

    # 增强的碰撞检测 - 使用多个包络线进行更细致的检测
    # 1. 使用标准车辆模型进行基本碰撞检测
    collision_info = check_path_collision(
        path, env, vehicle_length, vehicle_width, steps=20)  # 增加采样点数量

    if collision_info['collision']:
        print("路径规划测试失败：路径与障碍物碰撞")
        return False

    # 2. 使用不同朝向的车辆模型进行额外检测
    # 在路径的关键点处进行更细致的检测
    critical_points = []

    # 添加起点和终点
    critical_points.append((0, path[0]))
    critical_points.append((len(path)-1, path[-1]))

    # 添加转弯点（路径方向变化较大的点）
    for i in range(1, len(path)-1):
        prev_vec = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        next_vec = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])

        # 计算两个向量的夹角
        dot_product = prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1]
        prev_mag = math.sqrt(prev_vec[0]**2 + prev_vec[1]**2)
        next_mag = math.sqrt(next_vec[0]**2 + next_vec[1]**2)

        if prev_mag * next_mag > 0:  # 避免除以零
            cos_angle = dot_product / (prev_mag * next_mag)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # 确保在[-1, 1]范围内
            angle = math.acos(cos_angle)

            # 如果角度变化大于15度，认为是转弯点
            if angle > math.radians(15):
                critical_points.append((i, path[i]))

    # 在关键点处进行多角度碰撞检测
    for idx, point in critical_points:
        # 确定车辆朝向
        if idx == 0:  # 起点
            if len(path) > 1:
                dx = path[1][0] - path[0][0]
                dy = path[1][1] - path[0][1]
                heading = math.atan2(dy, dx)
            else:
                heading = 0  # 默认朝向
        elif idx == len(path) - 1:  # 终点
            dx = path[-1][0] - path[-2][0]
            dy = path[-1][1] - path[-2][1]
            heading = math.atan2(dy, dx)
        else:  # 中间点
            # 使用前后点的方向平均值
            dx1 = path[idx][0] - path[idx-1][0]
            dy1 = path[idx][1] - path[idx-1][1]
            dx2 = path[idx+1][0] - path[idx][0]
            dy2 = path[idx+1][1] - path[idx][1]
            heading1 = math.atan2(dy1, dx1)
            heading2 = math.atan2(dy2, dx2)
            heading = (heading1 + heading2) / 2

        # 创建临时车辆模型
        temp_vehicle = VehicleModel(
            point[0], point[1], heading, vehicle_length, vehicle_width)

        # 基本碰撞检测
        collision = check_vehicle_collision(temp_vehicle, env)
        if collision['collision']:
            print(f"路径规划测试失败：在关键点 {idx} 处检测到碰撞")
            return False

        # 额外检测 - 在不同转向角度下检测
        for steer_angle in [-0.3, -0.15, 0.15, 0.3]:  # 约±17°和±8.5°
            # 模拟车辆在该点以不同转向角行驶
            temp_vehicle.front_steer_angle = steer_angle
            # 更新车轮位置
            temp_vehicle.get_wheel_positions()

            # 检测碰撞
            collision = check_vehicle_collision(temp_vehicle, env)
            if collision['collision']:
                print(
                    f"路径规划测试失败：在关键点 {idx} 处以转向角 {math.degrees(steer_angle):.1f}° 检测到碰撞")
                return False

    # 3. 验证路径连续性
    for i in range(len(path)-1):
        p1 = path[i]
        p2 = path[i+1]
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if dist > args.step_size * 1.5:  # 允许一定的误差
            print(f"路径规划测试失败：路径不连续，在点 {i} 和 {i+1} 之间的距离为 {dist}")
            return False

    # 4. 验证路径曲率是否满足车辆转向约束
    # 创建一个临时车辆模型来获取最大转向角和轴距
    temp_vehicle = VehicleModel(0, 0, 0, vehicle_length, vehicle_width)
    max_steer_angle = temp_vehicle.max_steer  # 使用车辆模型的最大转向角
    wheelbase = temp_vehicle.wheelbase  # 使用车辆模型的轴距
    max_curvature = math.tan(max_steer_angle) / wheelbase

    for i in range(1, len(path)-1):
        # 计算路径曲率
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        x3, y3 = path[i+1]

        # 使用三点法估计曲率
        # 首先计算三点确定的圆的半径
        # 参考: https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2

        # 避免共线点导致的除零错误
        if abs((y2-y1)*(x3-x2) - (y3-y2)*(x2-x1)) < 1e-6:
            continue  # 三点共线，曲率为0

        # 计算三角形的边长
        a = math.sqrt((x2-x3)**2 + (y2-y3)**2)
        b = math.sqrt((x1-x3)**2 + (y1-y3)**2)
        c = math.sqrt((x1-x2)**2 + (y1-y2)**2)

        # 计算半周长
        s = (a + b + c) / 2

        # 计算三角形面积
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))

        # 计算外接圆半径
        if area < 1e-6:
            continue  # 面积太小，可能是三点几乎共线

        radius = (a*b*c) / (4*area)

        # 曲率 = 1/半径
        curvature = 1 / radius if radius > 0 else 0

        # 检查曲率是否超过车辆能力
        if curvature > max_curvature * 1.1:  # 允许10%的误差
            print(
                f"路径规划测试失败：路径曲率在点 {i} 处过大 ({curvature:.4f} > {max_curvature:.4f})")
            return False

    return True


def interactive_planning(simulator, env, start, args):
    """交互式路径规划"""
    # 确保 pygame 已初始化
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()

    # 初始化变量
    goal = None
    path = None
    running = True
    results = None

    try:
        # 加载配置文件
        config = load_config(args.config)

        # 从配置文件中获取窗口尺寸
        screen_width = config['window']['width']
        screen_height = config['window']['height']

        # 设置显示模式
        screen = pygame.display.set_mode((screen_width, screen_height))
        if not screen:
            raise RuntimeError("无法创建pygame显示窗口")

        pygame.display.set_caption(
            "停车场路径规划 - 右键选择未占用停车位，T重选，R重置，E切换算法，S切换转向模式，C切换控制方法")

        # 从配置文件中获取缩放比例
        scale = config['simulation']['scale']

        # 计算偏移量（使场景居中）
        offset_x = (screen_width - env.width * scale) / 2
        offset_y = (screen_height - env.height * scale) / 2

        # 获取支持中文的字体
        def get_font(size=24):
            """获取支持中文的字体"""
            # 尝试加载系统字体
            system_fonts = [
                # Windows 字体
                "SimHei",  # 黑体
                "Microsoft YaHei",  # 微软雅黑
                "SimSun",  # 宋体
                # Linux 字体
                "WenQuanYi Micro Hei",  # 文泉驿微米黑
                "Noto Sans CJK SC",  # Google Noto 字体
                "Droid Sans Fallback",  # Android 默认字体
                # macOS 字体
                "PingFang SC",  # 苹方
                "STHeiti"  # 华文黑体
            ]

            # 尝试按优先级加载字体
            for font_name in system_fonts:
                try:
                    return pygame.font.SysFont(font_name, size)
                except Exception:
                    continue

            # 如果都失败了，使用默认字体
            return pygame.font.Font(None, size)

        # 初始化车辆模型
        vehicle = VehicleModel(start[0], start[1], math.pi * 3 / 2)  # 朝下

        # 从配置文件获取仿真参数
        simulating = False
        simulation_speed = config['simulation'].get(
            'simulation_speed', 2.0)  # 仿真速度倍率
        dt = config['simulation'].get('dt', 0.05)  # 时间步长
        collision_detected = False  # 碰撞检测标志
        collision_info = None  # 碰撞详细信息

        # 状态文本
        status_text = "等待选择目标点"
        status_color = (0, 0, 0)  # 黑色

        # 控制方法列表
        control_methods = ["default", "pid", "mpc", "lqr"]
        current_control_method = args.control_method if args.control_method else "default"

        # 保存simulator的原始属性
        original_width = simulator.width
        original_height = simulator.height
        original_scale = simulator.scale
        original_vehicle = simulator.vehicle
        original_follower = simulator.follower
        original_environment = simulator.environment
        original_dt = simulator.config.get('dt', 0.05)
        original_fps = simulator.config.get('fps', 60)

        # 更新simulator属性以适应当前场景
        simulator.width = screen_width
        simulator.height = screen_height
        simulator.scale = scale
        simulator.screen = screen
        simulator.environment = env
        simulator.vehicle = vehicle
        simulator.follower = PathFollower(
            lookahead=config['simulation'].get('lookahead', 5.0), control_method=current_control_method)
        simulator.config['dt'] = dt
        simulator.config['fps'] = config['simulation'].get('fps', 60)

        # 坐标转换函数（屏幕坐标 -> 环境坐标）
        def screen_to_env(pos):
            print(f"屏幕坐标: {pos}, 偏移: ({offset_x}, {offset_y}), 缩放: {scale}")
            x = (pos[0] - offset_x) / scale
            y = (pos[1] - offset_y) / scale
            env_pos = (x, y)
            print(f"转换后的环境坐标: {env_pos}")
            return env_pos

        # 环境坐标转屏幕坐标
        def env_to_screen(pos):
            x = pos[0] * scale + offset_x
            y = pos[1] * scale + offset_y
            screen_pos = (int(x), int(y))
            return screen_pos

        # 重置车辆位置到起点
        def reset_vehicle():
            nonlocal simulating, collision_detected, collision_info, status_text, status_color
            vehicle.x, vehicle.y = start
            vehicle.heading = math.pi * 3 / 2  # 朝下
            vehicle.speed = 0.0
            vehicle.trajectory = [start]
            simulating = False
            collision_detected = False
            collision_info = None
            status_text = "车辆已重置到起点"
            status_color = (0, 128, 0)  # 绿色
            print("车辆已重置到起点")

        # 模拟车辆沿路径移动
        def simulate_path():
            nonlocal simulating, collision_detected, collision_info, status_text, status_color

            # 如果已经检测到碰撞，不再继续仿真
            if collision_detected:
                return

            # 计算控制输入
            throttle, brake, steer = simulator.follower.get_control(vehicle)

            # 更新车辆状态
            vehicle.update(throttle, brake, steer, dt * simulation_speed)

            # 检查碰撞和安全边界
            collision_info = check_vehicle_collision(vehicle, env)
            if collision_info['collision']:
                collision_detected = True
                simulating = False
                status_text = "检测到碰撞：与障碍物相撞！按R键重置车辆位置"
                status_color = (255, 0, 0)  # 红色
                print(f"检测到碰撞！位置: {collision_info['position']}")
                return
            elif collision_info['safety_warning']:
                status_text = "警告：进入安全边界区域"
                status_color = (255, 165, 0)  # 橙色
                print(f"安全警告！位置: {collision_info['position']}")

            # 检查是否到达终点
            if goal:  # 添加检查以避免None错误
                # 计算到目标点的距离和朝向差异
                dx = goal[0] - vehicle.x
                dy = goal[1] - vehicle.y
                distance_to_goal = math.sqrt(dx*dx + dy*dy)

                # 如果车辆已经非常接近目标点且速度很小，认为已到达
                if distance_to_goal < 0.5 and vehicle.speed < 0.1:  # 距离阈值0.5米，速度阈值0.1m/s
                    simulating = False
                    status_text = "到达目标点！按T键重新选择目标点"
                    status_color = (0, 255, 0)  # 绿色
                    print("车辆已到达目标点")

        # 绘制场景
        def draw_scene():
            screen.fill((255, 255, 255))  # 白色背景

            # 使用自定义绘制函数绘制环境，而不是使用simulator的方法
            # 这样可以确保使用正确的坐标系统和偏移量
            for obs in env.obstacles:
                # 转换坐标
                x = obs.x * scale + offset_x
                y = obs.y * scale + offset_y
                width = obs.width * scale
                height = obs.height * scale

                # 获取填充和线宽属性
                is_filled = getattr(obs, 'is_filled', True)
                line_width = getattr(obs, 'line_width', 1)

                # 检查是否是停车位，使用特殊颜色显示
                if hasattr(obs, 'is_parking_spot') and obs.is_parking_spot:
                    # 根据占用状态设置颜色
                    color = (255, 0, 0, 150) if obs.occupied else (
                        0, 255, 0, 150)  # 红色表示占用，绿色表示空闲

                    if obs.type == "rectangle":
                        # 创建旋转后的矩形
                        rect = pygame.Rect(0, 0, width, height)
                        surface = pygame.Surface(
                            (width, height), pygame.SRCALPHA)

                        # 绘制边框和填充
                        if is_filled:
                            pygame.draw.rect(surface, color, rect)
                        pygame.draw.rect(surface, color, rect, 3)  # 加粗边框

                        # 旋转并绘制
                        if hasattr(obs, 'angle') and obs.angle != 0:
                            rotated_surface = pygame.transform.rotate(
                                surface, -obs.angle)
                            screen.blit(rotated_surface,
                                        rotated_surface.get_rect(center=(x, y)))
                        else:
                            screen.blit(surface, pygame.Rect(
                                x - width/2, y - height/2, width, height))
                    continue

                # 绘制其他障碍物
                if obs.type == "rectangle":
                    # 创建旋转后的矩形
                    rect = pygame.Rect(0, 0, width, height)
                    surface = pygame.Surface((width, height), pygame.SRCALPHA)

                    # 绘制矩形
                    if is_filled:
                        pygame.draw.rect(surface, obs.color, rect)
                    else:
                        pygame.draw.rect(surface, obs.color, rect, line_width)

                    # 旋转并绘制
                    if hasattr(obs, 'angle') and obs.angle != 0:
                        rotated_surface = pygame.transform.rotate(
                            surface, -obs.angle)
                        screen.blit(rotated_surface,
                                    rotated_surface.get_rect(center=(x, y)))
                    else:
                        screen.blit(surface, pygame.Rect(
                            x - width/2, y - height/2, width, height))
                elif obs.type == "circle":
                    if is_filled:
                        pygame.draw.circle(screen, obs.color,
                                           (int(x), int(y)), int(width/2))
                    else:
                        pygame.draw.circle(screen, obs.color,
                                           (int(x), int(y)), int(width/2), line_width)

            # 绘制起点
            start_screen = env_to_screen(start)
            pygame.draw.circle(screen, (0, 255, 0), start_screen, 10)

            # 如果有目标点，绘制目标点
            if goal:
                goal_screen = env_to_screen(goal)
                pygame.draw.circle(screen, (255, 0, 0), goal_screen, 10)

            # 如果有路径，绘制路径
            if path:
                # 手动绘制路径，确保使用正确的坐标系统
                for i in range(len(path) - 1):
                    p1_screen = env_to_screen(path[i])
                    p2_screen = env_to_screen(path[i+1])
                    pygame.draw.line(screen, (0, 0, 255),
                                     p1_screen, p2_screen, 3)

            # 绘制车辆轨迹
            if len(vehicle.trajectory) > 1:
                # 手动绘制轨迹，确保使用正确的坐标系统
                for i in range(len(vehicle.trajectory) - 1):
                    p1_screen = env_to_screen(vehicle.trajectory[i])
                    p2_screen = env_to_screen(vehicle.trajectory[i+1])
                    pygame.draw.line(screen, (0, 200, 0),
                                     p1_screen, p2_screen, 2)

            # 绘制车辆 - 根据碰撞状态设置颜色
            car_color = (255, 0, 0) if collision_detected else \
                (255, 165, 0) if collision_info and collision_info.get('safety_warning') else \
                (0, 128, 0)  # 红色表示碰撞，橙色表示警告，绿色表示正常
            simulator._draw_vehicle(screen, vehicle, scale,
                                    offset_x, offset_y, car_color)

            # 创建半透明背景
            info_surface = pygame.Surface(
                (300, screen_height), pygame.SRCALPHA)
            info_surface.fill((255, 255, 255, 180))  # 白色半透明背景
            screen.blit(info_surface, (screen_width - 310, 0))

            # 显示提示信息
            font = get_font(20)  # 稍微减小字体大小

            # 显示状态文本
            status_surface = font.render(status_text, True, status_color)
            screen.blit(status_surface, (screen_width // 2 - status_surface.get_width() // 2,
                                         screen_height - 30))

            # 创建所有文本
            texts = [
                ("右键点击选择未占用的停车位作为目标点", (0, 0, 0)),
                ("按T键重新选择目标点", (0, 0, 0)),
                (f"当前算法: {args.algorithm}", (0, 0, 0)),
                (f"控制方法: {current_control_method}", (0, 0, 0)),
                (f"转向模式: {vehicle.steering_mode}", (0, 0, 0)),
                ("按E键切换规划算法", (0, 0, 0)),
                ("按C键切换控制方法", (0, 0, 0)),
                ("按S键切换转向模式", (0, 0, 0)),
                ("按R键重置车辆位置", (0, 0, 0)),
                ("碰撞检测: " + ("已触发" if collision_detected else "正常"),
                 (255, 0, 0) if collision_detected else (0, 0, 0)),
                ("绿色边框表示可选择的未占用停车位", (0, 150, 0))
            ]

            # 在右侧显示文本
            y_offset = 20
            x_pos = screen_width - 300
            for text, color in texts:
                text_surface = font.render(text, True, color)
                screen.blit(text_surface, (x_pos, y_offset))
                y_offset += 30

            # 如果发生碰撞，显示碰撞信息
            if collision_detected and collision_info:
                collision_text = "碰撞类型: 障碍物碰撞"
                text_surface = font.render(collision_text, True, (255, 0, 0))
                screen.blit(text_surface, (x_pos, y_offset))

            # 显示车辆状态信息
            if simulating:
                vehicle_info = [
                    f"速度: {vehicle.speed:.2f} m/s",
                    f"加速度: {vehicle.acceleration:.2f} m/s²",
                    f"转向角: {math.degrees(vehicle.front_steer_angle):.1f}°",
                    f"位置: ({vehicle.x:.1f}, {vehicle.y:.1f})"
                ]

                y_offset += 40
                for info in vehicle_info:
                    info_surface = font.render(info, True, (0, 0, 150))
                    screen.blit(info_surface, (x_pos, y_offset))
                    y_offset += 25

            pygame.display.flip()

        # 规划路径函数
        def plan_path_to_goal():
            if not goal:
                return None

            print(
                f"\n使用 {args.algorithm} 算法规划从 {vehicle.x, vehicle.y} 到 {goal} 的路径...")
            planner = create_planner(
                args.algorithm, (vehicle.x, vehicle.y), goal, env, args, vehicle.width, vehicle.length)
            path = try_plan_path(planner)

            # 如果找到路径，检查路径是否有碰撞
            if path:
                collision_points = check_path_collision(
                    path, env, vehicle.length, vehicle.width)
                if collision_points['collision']:
                    print("警告：规划的路径存在碰撞")
                    # 这里可以选择是否继续使用这条路径
                    # 如果需要重新规划，可以返回 None

            return path

        # 主循环
        clock = pygame.time.Clock()

        while running:
            try:
                # 处理pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    # 调试打印鼠标事件信息
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        button_name = "左键" if event.button == 1 else "右键" if event.button == 3 else f"按钮{event.button}"
                        print(f"检测到鼠标{button_name}点击事件，位置: {event.pos}")

                        # 鼠标右键点击选择目标点 - 注意系统差异，有些系统可能用不同的button值
                        # 通常右键是3，但有些系统可能是2或其他值
                        if (event.button == 3 or event.button == 2) and not simulating:
                            goal = screen_to_env(event.pos)
                            print(f"选择目标点: {goal}")
                            status_text = "正在规划路径..."
                            status_color = (0, 0, 0)  # 黑色

                            # 立即更新显示，给用户反馈
                            draw_scene()
                            pygame.display.flip()

                            # 检查目标点是否在未占用的停车位内
                            try:
                                # 鼠标右键在某些系统可能是不同的按钮值
                                parking_spot = env.find_parking_spot(goal)
                                if parking_spot is None:
                                    print(f"在位置 {goal} 没有找到未占用的停车位")
                                    # 查看一下当前环境中的停车位状态
                                    parking_spots_count = 0
                                    available_spots = 0
                                    for obs in env.obstacles:
                                        if hasattr(obs, 'is_parking_spot') and obs.is_parking_spot:
                                            parking_spots_count += 1
                                            if not obs.occupied:
                                                available_spots += 1
                                    print(
                                        f"当前环境中共有 {parking_spots_count} 个停车位，其中 {available_spots} 个未占用")
                                else:
                                    print(
                                        f"找到停车位: ({parking_spot.x}, {parking_spot.y})")
                            except Exception as e:
                                print(f"查找停车位时出错: {e}")
                                traceback.print_exc()
                                parking_spot = None

                            if parking_spot:
                                # 将目标点设置为停车位中心
                                goal = (parking_spot.x, parking_spot.y)
                                print(f"已选择停车位，目标点调整为: {goal}")

                                # 规划路径
                                path = plan_path_to_goal()
                                if not path:
                                    print("无法规划到该目标点的路径，请重新选择")
                                    goal = None
                                    status_text = "无法规划路径，请重新选择目标点"
                                    status_color = (255, 0, 0)  # 红色
                                else:
                                    # 设置路径并开始仿真
                                    simulator.follower.set_path(path)
                                    simulating = True
                                    collision_detected = False
                                    collision_info = None
                                    status_text = "正在仿真..."
                                    status_color = (0, 0, 255)  # 蓝色
                            else:
                                # 即使没有选择停车位，也允许直接选择位置作为目标
                                print(f"目标点不在未占用的停车位内，但仍尝试规划路径到 {goal}")
                                path = plan_path_to_goal()
                                if path:
                                    simulator.follower.set_path(path)
                                    simulating = True
                                    collision_detected = False
                                    collision_info = None
                                    status_text = "正在仿真..."
                                    status_color = (0, 0, 255)  # 蓝色
                                else:
                                    print("无法规划到该目标点的路径，请重新选择")
                                    goal = None
                                    status_text = "无法规划路径，请重新选择目标点"
                                    status_color = (255, 0, 0)  # 红色

                    # 按T键重新选择目标点
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                        goal = None
                        path = None
                        simulating = False
                        collision_detected = False
                        collision_info = None
                        # 重置车辆位置
                        vehicle.x, vehicle.y = start
                        vehicle.heading = math.pi * 3 / 2  # 朝下
                        vehicle.speed = 0.0
                        vehicle.trajectory = [start]
                        status_text = "等待选择目标点"
                        status_color = (0, 0, 0)  # 黑色
                        print("重新选择目标点")

                    # 按R键重置车辆位置
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        reset_vehicle()

                    # 按C键切换控制方法
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                        # 切换控制方法
                        current_control_method = control_methods[
                            (control_methods.index(current_control_method) + 1) % len(control_methods)]
                        simulator.follower.set_control_method(
                            current_control_method)
                        status_text = f"控制方法已切换为: {current_control_method}"
                        status_color = (0, 0, 255)  # 蓝色

                    # 按E键切换规划算法
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                        # 切换规划算法
                        algorithms = ["rrt", "rrt_star", "informed_rrt", "timed_rrt",
                                      "astar", "dijkstra", "dstar_lite", "theta_star", "attention_dqn_rrt"]
                        current_algorithm_index = algorithms.index(
                            args.algorithm) if args.algorithm in algorithms else 0
                        args.algorithm = algorithms[(
                            current_algorithm_index + 1) % len(algorithms)]
                        status_text = f"规划算法已切换为: {args.algorithm}"
                        status_color = (0, 0, 255)  # 蓝色
                        print(f"规划算法已切换为: {args.algorithm}")

                        # 如果有目标点，重新规划路径
                        if goal and not simulating:
                            path = plan_path_to_goal()
                            if path:
                                simulator.follower.set_path(path)

                    # 按S键切换转向模式
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                        # 切换转向模式
                        steering_modes = ["normal", "counter", "crab"]
                        current_mode_index = steering_modes.index(
                            vehicle.steering_mode)
                        new_mode = steering_modes[(
                            current_mode_index + 1) % len(steering_modes)]
                        vehicle.set_steering_mode(new_mode)
                        status_text = f"转向模式已切换为: {new_mode}"
                        status_color = (0, 0, 255)  # 蓝色

                    # 按P键暂停/继续仿真
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        # 暂停仿真
                        simulating = not simulating
                        pause_state = "暂停" if not simulating else "继续"
                        status_text = f"仿真已{pause_state}"
                        status_color = (0, 0, 255)  # 蓝色

                    # 按ESC键退出
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False

                # 更新仿真
                if simulating:
                    simulate_path()

                # 绘制场景
                draw_scene()

                # 控制帧率
                clock.tick(60)

            except pygame.error as e:
                print(f"pygame错误: {e}")
                if "video system not initialized" in str(e):
                    # 尝试重新初始化视频系统
                    try:
                        pygame.display.quit()
                        pygame.display.init()
                        screen = pygame.display.set_mode(
                            (screen_width, screen_height))
                        continue  # 继续主循环
                    except Exception as reinit_error:
                        print(f"重新初始化视频系统失败: {reinit_error}")
                        running = False
                else:
                    running = False

        # 收集仿真结果
        results = {
            'vehicle_trajectory': vehicle.trajectory,
            'path': path,
            'collision_points': [],  # 如果有碰撞点记录可以添加
            'algorithm': args.algorithm,
            'control_method': current_control_method,
            'simulation_time': pygame.time.get_ticks() / 1000.0 if pygame.get_init() else 0  # 转换为秒
        }

        # 在退出主循环后，显示可视化结果
        visualize_results(env, start, goal, path, results)

    except Exception as e:
        print(f"仿真过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # 恢复simulator的原始属性
        simulator.width = original_width
        simulator.height = original_height
        simulator.scale = original_scale
        simulator.vehicle = original_vehicle
        simulator.follower = original_follower
        simulator.environment = original_environment
        simulator.config['dt'] = original_dt
        simulator.config['fps'] = original_fps

        # 确保正确清理 pygame
        try:
            pygame.display.quit()
        except Exception as e:
            print(f"清理pygame display时出错: {e}")
            pass
        try:
            pygame.quit()
        except Exception as e:
            print(f"清理pygame时出错: {e}")
            pass

    return results


def visualize_results(env, start, goal, path, results):
    """使用matplotlib可视化结果"""
    if not results:
        print("没有可视化结果")
        return

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # 绘制环境边界和障碍物
        for obs in env.obstacles:
            if obs.type == "rectangle":
                if hasattr(obs, 'is_parking_spot') and obs.is_parking_spot:
                    color = 'r' if obs.occupied else 'g'
                    alpha = 0.3
                else:
                    color = 'gray'
                    alpha = 0.5

                # 计算矩形的四个角点
                w, h = obs.width/2, obs.height/2
                corners = np.array(
                    [[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]])

                # 如果有角度，进行旋转
                if hasattr(obs, 'angle') and obs.angle != 0:
                    angle_rad = np.radians(obs.angle)
                    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                           [np.sin(angle_rad), np.cos(angle_rad)]])
                    corners = np.dot(corners, rot_matrix.T)

                # 平移到障碍物位置
                corners = corners + np.array([obs.x, obs.y])
                plt.fill(corners[:, 0], corners[:, 1],
                         color=color, alpha=alpha)

        # 绘制规划路径-添加不同算法轨迹对比
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-',
                     label='Planned Path', linewidth=2)

        # 绘制车辆轨迹
        if results['vehicle_trajectory']:
            traj_x = [p[0] for p in results['vehicle_trajectory']]
            traj_y = [p[1] for p in results['vehicle_trajectory']]
            plt.plot(traj_x, traj_y, 'g--',
                     label='Vehicle Trajectory', linewidth=2)
        # 绘制车辆在轨迹上的轮廓线
        if results['vehicle_trajectory']:
            traj_x = [p[0] for p in results['vehicle_trajectory']]
            traj_y = [p[1] for p in results['vehicle_trajectory']]
            plt.plot(traj_x, traj_y, 'g--',
                     label='Vehicle Trajectory', linewidth=2)
        # 绘制起点和终点
        plt.plot(start[0], start[1], 'go', label='Start', markersize=10)
        if goal:
            plt.plot(goal[0], goal[1], 'ro', label='Goal', markersize=10)

        # 设置图表属性
        plt.title(
            f'Path Planning Results\nAlgorithm: {results["algorithm"]}, Control: {results["control_method"]}')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        # 显示图表
        plt.show()
    except Exception as e:
        print(f"可视化结果时出错: {e}")
        traceback.print_exc()


def main():
    """主函数：创建场景、规划路径并仿真"""
    # 初始化 pygame
    pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()

    # 设置环境变量以避免在某些系统上的音频初始化问题
    import os
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    try:
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

    except Exception as e:
        print(f"仿真过程中发生错误: {e}")
        traceback.print_exc()
    finally:
        # 确保正确清理 pygame
        try:
            pygame.display.quit()
        except Exception as e:
            print(f"清理pygame display时出错: {e}")
        try:
            pygame.quit()
        except Exception as e:
            print(f"清理pygame时出错: {e}")


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
        choices=['rrt', 'rrt_star', 'informed_rrt', 'timed_rrt', 'dijkstra',
                 'dstar_lite', 'theta_star', 'astar', 'attention_dqn_rrt'],
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

    args = parser.parse_args()

    # 确保iterations是整数
    args.iterations = int(args.iterations)

    return args


if __name__ == "__main__":
    main()
