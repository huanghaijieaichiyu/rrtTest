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
from simulation.pygame_simulator import ParkingEnvironment, PathFollower, PygameSimulator, VehicleModel
from simulation.pygame_simulator import check_vehicle_collision, check_path_collision
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


def interactive_planning(simulator, env, start, args):
    """交互式路径规划

    参数:
        simulator: 仿真器对象
        env: 环境对象
        start: 起点坐标
        args: 命令行参数
    """
    import pygame

    # 初始化pygame
    pygame.init()

    # 设置窗口大小为1080p
    screen_width = 860
    screen_height = 640
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(
        "停车场路径规划 - 右键选择未占用停车位，T重选，R重置，E切换算法，S切换转向模式，C切换控制方法")

    # 计算缩放比例
    scale_x = screen_width / env.width
    scale_y = screen_height / env.height
    scale = min(scale_x, scale_y) * 0.9

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

    # 初始化车辆模型和路径跟踪器
    vehicle = VehicleModel(start[0], start[1], math.pi * 3 / 2)  # 朝下
    follower = PathFollower(lookahead=5.0, control_method=args.control_method)

    # 仿真参数
    simulating = False
    simulation_speed = 2.0  # 仿真速度倍率
    dt = 0.05  # 时间步长
    collision_detected = False  # 碰撞检测标志
    collision_info = None  # 碰撞详细信息

    # 状态文本
    status_text = "等待选择目标点"
    status_color = (0, 0, 0)  # 黑色

    # 控制方法列表
    control_methods = ["default", "pid", "mpc", "lqr"]
    current_control_method = args.control_method if args.control_method else "default"

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
        throttle, brake, steer = follower.get_control(vehicle)

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

    # 绘制车辆
    def draw_vehicle():
        # 获取车辆四个角的坐标
        corners = vehicle.get_corners()

        # 转换到屏幕坐标
        screen_corners = []
        for x, y in corners:
            sx = x * scale + offset_x
            sy = y * scale + offset_y
            screen_corners.append((int(sx), int(sy)))

        # 根据状态设置不同颜色
        if collision_detected:
            car_color = (255, 0, 0)  # 红色表示碰撞
        elif collision_info and collision_info.get('safety_warning'):
            car_color = (255, 165, 0)  # 橙色表示安全警告
        else:
            car_color = (0, 128, 0)  # 绿色表示正常

        # 绘制车身
        pygame.draw.polygon(screen, car_color, screen_corners)

        # 绘制车窗 (内部区域)
        window_inset = 0.2  # 车窗内缩比例
        half_length = vehicle.length / 2 * (1 - window_inset)
        half_width = vehicle.width / 2 * (1 - window_inset)

        # 车窗在车身坐标系中的位置
        window_local = [
            (half_length, half_width),   # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width),  # 左后
            (-half_length, half_width)   # 右后
        ]

        # 转换到世界坐标系，再转换到屏幕坐标系
        cos_h = math.cos(vehicle.heading)
        sin_h = math.sin(vehicle.heading)

        window_screen = []
        for lx, ly in window_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = wx * scale + offset_x
            sy = wy * scale + offset_y
            window_screen.append((int(sx), int(sy)))

        # 绘制车窗 (深蓝色半透明)
        pygame.draw.polygon(screen, (30, 30, 80, 180), window_screen)

        # 获取车轮位置和角度
        wheels = vehicle.get_wheel_positions()

        # 绘制车轮
        for wheel_x, wheel_y, wheel_angle in wheels:
            # 计算车轮的四个角
            wheel_half_length = vehicle.wheel_length / 2
            wheel_half_width = vehicle.wheel_width / 2

            # 车轮在自身坐标系中的四个角
            wheel_corners_local = [
                (wheel_half_length, wheel_half_width),   # 右前
                (wheel_half_length, -wheel_half_width),  # 左前
                (-wheel_half_length, -wheel_half_width),  # 左后
                (-wheel_half_length, wheel_half_width)   # 右后
            ]

            # 转换到世界坐标系，考虑车轮自身的转向角
            cos_w = math.cos(wheel_angle)
            sin_w = math.sin(wheel_angle)

            wheel_corners_screen = []
            for lx, ly in wheel_corners_local:
                wx = wheel_x + lx * cos_w - ly * sin_w
                wy = wheel_y + lx * sin_w + ly * cos_w
                sx = wx * scale + offset_x
                sy = wy * scale + offset_y
                wheel_corners_screen.append((int(sx), int(sy)))

            # 绘制车轮 (黑色)
            pygame.draw.polygon(screen, (20, 20, 20), wheel_corners_screen)

            # 绘制车轮中心点 (轮毂)
            hub_x = int(wheel_x * scale + offset_x)
            hub_y = int(wheel_y * scale + offset_y)
            pygame.draw.circle(screen, (150, 150, 150), (hub_x, hub_y), 2)

        # 绘制车头方向
        head_x = vehicle.x + math.cos(vehicle.heading) * vehicle.length / 2
        head_y = vehicle.y + math.sin(vehicle.heading) * vehicle.length / 2
        center_screen = (int(vehicle.x * scale + offset_x),
                         int(vehicle.y * scale + offset_y))
        head_screen = (int(head_x * scale + offset_x),
                       int(head_y * scale + offset_y))
        pygame.draw.line(screen, (0, 0, 255), center_screen, head_screen, 2)

        # 绘制车灯
        light_radius = vehicle.width * 0.1
        light_offset_y = vehicle.width * 0.3

        # 前灯位置 (黄色)
        front_light_local = [
            (vehicle.length/2 - light_radius, light_offset_y),  # 右前灯
            (vehicle.length/2 - light_radius, -light_offset_y)  # 左前灯
        ]

        for lx, ly in front_light_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = int(wx * scale + offset_x)
            sy = int(wy * scale + offset_y)
            pygame.draw.circle(screen, (255, 255, 0),
                               (sx, sy), int(light_radius * scale))

        # 后灯位置 (红色)
        rear_light_local = [
            (-vehicle.length/2 + light_radius, light_offset_y),  # 右后灯
            (-vehicle.length/2 + light_radius, -light_offset_y)  # 左后灯
        ]

        for lx, ly in rear_light_local:
            wx = vehicle.x + lx * cos_h - ly * sin_h
            wy = vehicle.y + lx * sin_h + ly * cos_h
            sx = int(wx * scale + offset_x)
            sy = int(wy * scale + offset_y)
            pygame.draw.circle(screen, (255, 0, 0), (sx, sy),
                               int(light_radius * scale))

        # 仅当show_sensors为True时绘制传感器
        if vehicle.show_sensors:
            # 绘制传感器
            sensor_positions = vehicle.get_sensor_positions()

            # 绘制环视摄像头 (黄色)
            for camera in sensor_positions['fisheye_cameras']:
                pos = camera['pos']
                color = camera['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                pygame.draw.circle(screen, color, (sx, sy), 5)
                # 绘制摄像头视野范围指示
                pygame.draw.circle(screen, color, (sx, sy), 15, 1)

            # 绘制前视摄像头 (红色)
            if sensor_positions['front_camera']:
                pos = sensor_positions['front_camera']['pos']
                color = sensor_positions['front_camera']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                pygame.draw.circle(screen, color, (sx, sy), 4)
                # 绘制摄像头视野范围
                view_length = vehicle.length * 0.8
                view_x = pos[0] + math.cos(vehicle.heading) * view_length
                view_y = pos[1] + math.sin(vehicle.heading) * view_length
                view_sx = int(view_x * scale + offset_x)
                view_sy = int(view_y * scale + offset_y)
                pygame.draw.line(screen, color, (sx, sy),
                                 (view_sx, view_sy), 1)

            # 绘制超声波雷达 (紫色)
            for sensor in sensor_positions['ultrasonic']:
                pos = sensor['pos']
                color = sensor['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制超声波雷达点
                pygame.draw.circle(screen, color, (sx, sy), 3)

                # 计算超声波雷达方向 - 从车辆中心指向传感器
                sensor_angle = math.atan2(
                    pos[1] - vehicle.y, pos[0] - vehicle.x)
                # 绘制超声波雷达探测范围
                range_length = 1.0  # 探测范围1米
                range_x = pos[0] + math.cos(sensor_angle) * range_length
                range_y = pos[1] + math.sin(sensor_angle) * range_length
                range_sx = int(range_x * scale + offset_x)
                range_sy = int(range_y * scale + offset_y)
                pygame.draw.line(screen, color, (sx, sy),
                                 (range_sx, range_sy), 1)

            # 绘制IMU (绿色)
            if sensor_positions['imu']:
                pos = sensor_positions['imu']['pos']
                color = sensor_positions['imu']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制IMU为一个小方块
                imu_size = 4
                pygame.draw.rect(screen, color, (sx - imu_size //
                                 2, sy - imu_size//2, imu_size, imu_size))

            # 绘制GPS (浅绿色)
            if sensor_positions['gps']:
                pos = sensor_positions['gps']['pos']
                color = sensor_positions['gps']['color']
                sx = int(pos[0] * scale + offset_x)
                sy = int(pos[1] * scale + offset_y)
                # 绘制GPS为一个十字形
                cross_size = 5
                pygame.draw.line(
                    screen, color, (sx - cross_size, sy), (sx + cross_size, sy), 2)
                pygame.draw.line(
                    screen, color, (sx, sy - cross_size), (sx, sy + cross_size), 2)

    # 绘制函数
    def draw_scene():
        screen.fill((255, 255, 255))  # 白色背景

        # 绘制障碍物
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
                    surface = pygame.Surface((width, height), pygame.SRCALPHA)

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
        start_x = start[0] * scale + offset_x
        start_y = start[1] * scale + offset_y
        pygame.draw.circle(screen, (0, 255, 0),
                           (int(start_x), int(start_y)), 10)

        # 如果有目标点，绘制目标点
        if goal:
            goal_x = goal[0] * scale + offset_x
            goal_y = goal[1] * scale + offset_y
            pygame.draw.circle(screen, (255, 0, 0),
                               (int(goal_x), int(goal_y)), 10)

        # 如果有路径，绘制路径
        if path:
            for i in range(len(path) - 1):
                p1_x = path[i][0] * scale + offset_x
                p1_y = path[i][1] * scale + offset_y
                p2_x = path[i+1][0] * scale + offset_x
                p2_y = path[i+1][1] * scale + offset_y
                pygame.draw.line(screen, (0, 0, 255),
                                 (int(p1_x), int(p1_y)),
                                 (int(p2_x), int(p2_y)), 3)

        # 绘制车辆轨迹
        if len(vehicle.trajectory) > 1:
            for i in range(len(vehicle.trajectory) - 1):
                p1_x = vehicle.trajectory[i][0] * scale + offset_x
                p1_y = vehicle.trajectory[i][1] * scale + offset_y
                p2_x = vehicle.trajectory[i+1][0] * scale + offset_x
                p2_y = vehicle.trajectory[i+1][1] * scale + offset_y
                pygame.draw.line(screen, (0, 200, 0),
                                 (int(p1_x), int(p1_y)),
                                 (int(p2_x), int(p2_y)), 2)

        # 绘制车辆
        draw_vehicle()

        # 创建半透明背景
        info_surface = pygame.Surface((300, screen_height), pygame.SRCALPHA)
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

        pygame.display.flip()

    # 坐标转换函数（屏幕坐标 -> 环境坐标）
    def screen_to_env(pos):
        x = (pos[0] - offset_x) / scale
        y = (pos[1] - offset_y) / scale
        return (x, y)

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

    # 初始化变量
    goal = None
    path = None
    running = True
    clock = pygame.time.Clock()

    # 主循环
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 鼠标右键点击选择目标点
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and not simulating:
                goal = screen_to_env(event.pos)
                print(f"选择目标点: {goal}")
                status_text = "正在规划路径..."
                status_color = (0, 0, 0)  # 黑色
                draw_scene()  # 立即更新显示

                # 检查目标点是否在未占用的停车位内
                parking_spot = env.find_parking_spot(goal)
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
                        follower.set_path(path)
                        simulating = True
                        collision_detected = False
                        collision_info = None
                        status_text = "正在仿真..."
                        status_color = (0, 0, 255)  # 蓝色
                else:
                    print("目标点不在未占用的停车位内，请重新选择")
                    goal = None
                    status_text = "请选择未占用的停车位作为目标点"
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
                    (control_methods.index(current_control_method) +
                     1) % len(control_methods)]
                follower.set_control_method(current_control_method)
                status_text = f"控制方法已切换为: {current_control_method}"
                status_color = (0, 0, 255)  # 蓝色

            # 按E键切换规划算法
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                # 切换规划算法
                algorithms = ["rrt", "rrt_star", "informed_rrt", "timed_rrt",
                              "astar", "dijkstra", "dstar_lite", "theta_star"]
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
                        follower.set_path(path)

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

        # 更新仿真
        if simulating:
            simulate_path()

        # 绘制场景
        draw_scene()

        # 控制帧率
        clock.tick(60)

    pygame.quit()


def draw_scene(screen, env, scale, offset_x, offset_y):
    """绘制场景"""
    for obstacle in env.obstacles:
        # 转换坐标到屏幕坐标系
        screen_x = int(obstacle.x * scale + offset_x)
        screen_y = int(obstacle.y * scale + offset_y)

        # 获取颜色
        color = obstacle.color if hasattr(
            obstacle, 'color') else (100, 100, 100, 200)

        if obstacle.type == "rectangle":
            # 计算矩形的屏幕尺寸
            screen_width = int(obstacle.width * scale)
            screen_height = int(obstacle.height * scale)

            # 创建矩形表面
            rect_surface = pygame.Surface(
                (screen_width, screen_height), pygame.SRCALPHA)

            # 绘制矩形
            is_filled = obstacle.is_filled if hasattr(
                obstacle, 'is_filled') else True
            line_width = obstacle.line_width if hasattr(
                obstacle, 'line_width') else 1

            if is_filled:
                pygame.draw.rect(rect_surface, color,
                                 (0, 0, screen_width, screen_height))
            else:
                pygame.draw.rect(rect_surface, color, (0, 0, screen_width, screen_height),
                                 width=line_width)

            # 旋转矩形
            angle = obstacle.angle if hasattr(obstacle, 'angle') else 0
            rotated_surface = pygame.transform.rotate(rect_surface, angle)

            # 获取旋转后的矩形中心位置
            rot_rect = rotated_surface.get_rect(center=(screen_x, screen_y))

            # 绘制到屏幕
            screen.blit(rotated_surface, rot_rect.topleft)

        elif obstacle.type == "circle":
            # 计算圆的屏幕半径
            screen_radius = int(obstacle.radius * scale)

            is_filled = obstacle.is_filled if hasattr(
                obstacle, 'is_filled') else True
            line_width = obstacle.line_width if hasattr(
                obstacle, 'line_width') else 1

            if is_filled:
                pygame.draw.circle(
                    screen, color, (screen_x, screen_y), screen_radius)
            else:
                pygame.draw.circle(screen, color, (screen_x, screen_y), screen_radius,
                                   width=line_width)


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


if __name__ == "__main__":
    main()
