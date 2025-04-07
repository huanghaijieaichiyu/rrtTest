import numpy as np
import random
import argparse
import pygame
import yaml
import traceback
import time
import os
import math
from typing import List, Tuple, Dict, Any, Optional, Union

# 导入RRT相关规划器
from rrt.astar import AStar
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar, TimedRRTStar
from rrt.informed_rrt import InformedRRTStar
from rrt.dijkstra import Dijkstra
from rrt.dstar_lite import DStarLite
from rrt.theta_star import ThetaStar
from rrt.attention_dqn_rrt import AttentionDQNRRT

# 导入Pygame仿真器组件
from simulation.environment import Environment
from simulation.pygame_simulator import (ParkingEnvironment, PathFollower, PygameSimulator, VehicleModel,
                                         check_vehicle_collision, check_path_collision, get_font, BLACK, WHITE, RED,
                                         GREEN, BLUE, YELLOW, GRAY)

# 可视化
import matplotlib.patches as patches


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
            'fps': 60,  # 帧率
            'dt': 0.05,  # 仿真时间步长(秒)
            'lookahead': 5.0,  # 路径跟踪前瞻距离
            'simulation_speed': 2.0  # 仿真速度倍率
        },

        # 车辆参数
        'vehicle': {
            'length': 4.5,  # 车辆长度(米)
            'width': 1.8,  # 车辆宽度(米)
            'wheelbase': 2.7,  # 轴距(米)
            'max_speed': 5.0,  # 最大速度(m/s)
            'max_accel': 2.0,  # 最大加速度(m/s^2)
            'max_brake': 4.0,  # 最大制动(m/s^2)
            'max_steer': 0.7854  # 最大转向角(弧度), 约45度
        },

        # 停车场布局参数
        'parking_lot': {
            'geometry': {
                'spot_width': 2.5,  # 停车位宽度(m)
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
    env.add_obstacle(x=width / 2,
                     y=0,
                     obstacle_type="rectangle",
                     width=width,
                     height=wall_thickness,
                     angle=0,
                     color=wall_color)

    # 下边界（左侧）
    env.add_obstacle(x=(width - entrance_width) / 4,
                     y=height,
                     obstacle_type="rectangle",
                     width=(width - entrance_width) / 2,
                     height=wall_thickness,
                     angle=0,
                     color=wall_color)

    # 下边界（右侧）
    env.add_obstacle(x=width - (width - entrance_width) / 4,
                     y=height,
                     obstacle_type="rectangle",
                     width=(width - entrance_width) / 2,
                     height=wall_thickness,
                     angle=0,
                     color=wall_color)

    # 左边界
    env.add_obstacle(x=0,
                     y=height / 2,
                     obstacle_type="rectangle",
                     width=wall_thickness,
                     height=height,
                     angle=0,
                     color=wall_color)

    # 右边界
    env.add_obstacle(x=width,
                     y=height / 2,
                     obstacle_type="rectangle",
                     width=wall_thickness,
                     height=height,
                     angle=0,
                     color=wall_color)

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
                color=car_body_color)

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
                color=car_body_color)

    # 中间停车区（双排）- 横向停车位
    middle_left_x = width / 2 - spot_length * 0.7  # 调整位置以适应横向停车位
    middle_right_x = width / 2 + spot_length * 0.7  # 调整位置以适应横向停车位
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
                color=car_body_color)

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
                color=car_body_color)


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
    middle_left_x = env_width / 2 - spot_length * 0.7  # 调整位置以适应横向停车位
    middle_right_x = env_width / 2 + spot_length * 0.7  # 调整位置以适应横向停车位
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


def get_algorithm_specific_params(
    algorithm: str,
    args,
) -> Dict[str, Any]:
    """获取算法特定的参数"""
    # 确保max_iterations始终是整数
    max_iterations = int(args.iterations) if args.iterations is not None else 10000

    base_params = {'max_iterations': max_iterations, 'step_size': args.step_size if args.step_size is not None else 2.0}

    params = {
        'astar': {
            'resolution': 0.5,
            'diagonal_movement': True,
            'weight': 1.0
        },
        'rrt': base_params,
        'rrt_star': {
            **base_params, 'rewire_factor': 1.5
        },
        'informed_rrt': {
            **base_params, 'focus_factor': 1.0
        },
        'timed_rrt': {
            **base_params, 'robot_speed': args.robot_speed
        },
        'dijkstra': {
            'resolution': 1.0,
            'diagonal_movement': True
        },
        'dstar_lite': {
            'resolution': 1.0,
            'diagonal_movement': True
        },
        'theta_star': {
            'resolution': 1.0,
            'diagonal_movement': True
        },
        'attention_dqn_rrt': {
            **base_params, 'rewire_factor': 1.5,
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon': 0.1,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'hidden_dim': 256,
            'prediction_horizon': 5
        }
    }

    # 返回算法特定参数
    result = params.get(algorithm, {})

    # 如果是attention_dqn_rrt且指定了模型路径，添加模型路径
    if algorithm == 'attention_dqn_rrt' and hasattr(args, 'model_path') and args.model_path:
        # 检查模型文件是否存在
        if os.path.exists(args.model_path):
            print(f"将加载预训练模型: {args.model_path}")
            result['model_path'] = args.model_path
        else:
            print(f"警告: 模型文件 {args.model_path} 不存在，将使用默认初始化")

    return result


def optimize_path(path: List[Tuple[float, float]], env: Environment, vehicle_width: float,
                  vehicle_length: float) -> List[Tuple[float, float]]:
    """
    对规划好的路径进行优化，包括去冗余、平滑和快捷化。

    参数:
        path: 原始路径
        env: 环境对象
        vehicle_width: 车辆宽度
        vehicle_length: 车辆长度

    返回:
        优化后的路径
    """
    if not path or len(path) < 3:
        print("路径点过少，跳过优化")
        return path

    original_path = path  # 保存原始路径以备回退

    try:
        # 初始化路径平滑器
        from rrt.path_smoothing import PathSmoother
        smoother = PathSmoother(vehicle_width, vehicle_length)

        # 1. 轨迹误差检测 (TED)，去除冗余点 (Re-enabled)
        print("优化步骤1: 轨迹误差检测...")
        simplified_path = smoother.ted_detection(path)
        if not simplified_path or len(simplified_path) < 2:
            print("TED简化后路径点过少，使用原始路径进行后续优化")
            simplified_path = path  # 回退到上一步的路径
        else:
            print(f"TED后路径点数: {len(simplified_path)}")
            path = simplified_path  # 更新路径

        # 2. 五次多项式插值平滑
        print("优化步骤2: 五次多项式插值...")
        # 调整num_points可以改变平滑度和路径长度的平衡
        interpolated_path = smoother.quintic_polynomial_interpolation(path, num_points=5)
        if not interpolated_path or len(interpolated_path) < 2:
            print("插值失败，使用简化路径进行后续优化")
            interpolated_path = path  # 回退到上一步的路径
        else:
            print(f"插值后路径点数: {len(interpolated_path)}")
            path = interpolated_path  # 更新路径

        # 3. 卡尔曼滤波平滑
        print("优化步骤3: 卡尔曼滤波平滑...")
        kalman_smoothed_path = smoother.kalman_filter_smoothing(path, process_noise=0.1, measurement_noise=0.1)
        if not kalman_smoothed_path or len(kalman_smoothed_path) < 2:
            print("卡尔曼滤波失败，使用插值路径进行后续优化")
            kalman_smoothed_path = path  # 回退到上一步的路径
        else:
            print(f"卡尔曼滤波后路径点数: {len(kalman_smoothed_path)}")
            path = kalman_smoothed_path  # 更新路径

        # 4. 路径快捷化 (Shortcutting)
        print("优化步骤4: 路径快捷化...")
        shortcut_path = smoother.shortcut_path(path, env)  # 假设PathSmoother有shortcut_path方法
        if not shortcut_path or len(shortcut_path) < 2:
            print("快捷化失败，使用卡尔曼滤波路径")
            shortcut_path = path  # 回退到上一步
        else:
            print(f"快捷化后路径点数: {len(shortcut_path)}")
            path = shortcut_path  # 更新路径

        # 5. 检查优化后的路径是否可行 (Re-enabled)
        print("优化步骤5: 最终碰撞检测...")
        collision_info = check_path_collision(path, env, vehicle_length, vehicle_width, steps=len(path) * 2)  # 增加检查密度
        if collision_info['collision']:
            print(f"警告: 优化后的路径在点 {collision_info.get('point', 'N/A')} 附近发生碰撞，将回退到原始路径。")
            return original_path  # 回退到原始路径
        else:
            print("优化后路径碰撞检测通过。")

        # 检查路径长度是否大幅增加 (示例：增加超过50%)
        original_length = sum(
            np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(original_path) - 1))
        optimized_length = sum(
            np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))
        if optimized_length > original_length * 1.5:
            print(f"警告: 优化后路径长度 ({optimized_length:.2f}m) 大幅超过原始路径 ({original_length:.2f}m)。可能需要调整优化参数。")
            # 可选：如果长度增加过多，也回退到原始路径
            # print("路径长度增加过多，回退到原始路径。")
            # return original_path

        print(f"路径优化完成。原始长度: {original_length:.2f}m, 优化后长度: {optimized_length:.2f}m")
        return path

    except ImportError:
        print("错误：需要PathSmoother类来进行路径优化。请确保 'rrt.path_smoothing' 存在。")
        return original_path  # 导入失败则返回原始路径
    except Exception as e:
        print(f"路径优化过程中发生错误: {e}")
        traceback.print_exc()
        return original_path  # 发生任何错误都返回原始路径


def create_planner(algorithm: str, start: tuple, goal: tuple, env: Environment, args, vehicle_width, vehicle_length):
    """创建路径规划器"""
    # 获取车辆尺寸参数
    vehicle_width = vehicle_width  # 车辆宽度
    vehicle_length = vehicle_length  # 车辆长度

    # 获取算法特定参数
    algorithm_params = get_algorithm_specific_params(algorithm, args)

    # 基本参数，所有规划器都需要
    common_params = {
        'start': start,
        'goal': goal,
        'env': env,
        'vehicle_width': vehicle_width,
        'vehicle_length': vehicle_length,
    }

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


def try_plan_path(planner, max_retries: int = 10) -> Optional[List[Tuple[float, float]]]:
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


class ParkingDemoSimulator(PygameSimulator):
    """继承PygameSimulator并实现停车场演示的特定逻辑"""

    def __init__(self,
                 config_input: Optional[Union[str, Dict]] = None,
                 args: Optional[argparse.Namespace] = None,
                 start_pos: Optional[Tuple[float, float]] = None):
        super().__init__(config_input)

        self.args = args if args else argparse.Namespace()  # 命令行参数
        self.start_pos = start_pos if start_pos else (0, 0)
        self.goal_pos = None
        self.path = None
        self.collision_info = None
        self.simulating = False  # 是否正在执行路径
        self.simulation_speed = self.config.get('simulation', {}).get('simulation_speed', 2.0)
        self.dt = self.config.get('simulation', {}).get('dt', 0.05)

        # 初始化车辆到起点
        self.vehicle.x, self.vehicle.y = self.start_pos
        self.vehicle.heading = math.pi * 3 / 2  # 默认朝下
        self.vehicle.trajectory = [self.start_pos]

        # 规划算法列表
        self.planning_algorithms = [
            "rrt", "rrt_star", "informed_rrt", "timed_rrt", "astar", "dijkstra", "dstar_lite", "theta_star",
            "attention_dqn_rrt"
        ]
        # 确保args.algorithm在列表中，如果不在，设置为默认值
        if not hasattr(self.args, 'algorithm') or self.args.algorithm not in self.planning_algorithms:
            self.args.algorithm = self.planning_algorithms[0]

        # 控制方法列表
        self.control_methods = ["default", "pid", "mpc", "lqr"]
        # 确保args.control_method在列表中，如果不在，设置为默认值
        if not hasattr(self.args, 'control_method') or self.args.control_method not in self.control_methods:
            self.args.control_method = self.control_methods[0]
        self.current_control_method = self.args.control_method
        self.follower.set_control_method(self.current_control_method)

        # 转向模式列表
        self.steering_modes = ["normal", "counter", "crab"]

        # 更新按键提示
        self.key_hints = ["右键:选目标", "T:重选", "R:重置车", "E:换算法", "S:换转向", "C:换控制", "P:暂停/继续", "ESC:退出"]
        self.status_text = "等待选择目标点"
        self.status_color = BLACK

        # 计算绘制偏移量 (如果环境尺寸和窗口尺寸不同)
        if self.environment:
            self.offset_x = (self.width - self.environment.width * self.scale) / 2
            self.offset_y = (self.height - self.environment.height * self.scale) / 2
        else:
            self.offset_x = 0
            self.offset_y = 0

        # 确保字体已初始化
        if not pygame.font.get_init():
            pygame.font.init()
        self.font = get_font(18)  # 用于绘制信息的字体

    def _plan_path_to_goal(self):
        """规划路径到目标点"""
        if not self.goal_pos or not self.environment:
            self.status_text = "错误：未设置目标点或环境"
            self.status_color = RED
            return

        self.status_text = f"使用 {self.args.algorithm} 规划中..."
        self.status_color = BLUE
        self.draw()  # 更新屏幕显示状态
        pygame.display.flip()
        print(f"\n使用 {self.args.algorithm} 算法规划从 {self.vehicle.x:.2f, self.vehicle.y:.2f} 到 {self.goal_pos} 的路径...")

        try:
            planner = create_planner(self.args.algorithm, (self.vehicle.x, self.vehicle.y), self.goal_pos,
                                     self.environment, self.args, self.vehicle.width, self.vehicle.length)
            raw_path = try_plan_path(planner)

            if not raw_path:
                print("无法规划到该目标点的路径，请重新选择")
                self.goal_pos = None
                self.path = None
                self.status_text = "无法规划路径，请重新选择目标点"
                self.status_color = RED
                return

            print("路径规划成功，开始优化路径...")
            self.status_text = "路径优化中..."
            self.status_color = BLUE
            self.draw()
            pygame.display.flip()

            optimized_path = optimize_path(raw_path, self.environment, self.vehicle.width, self.vehicle.length)

            print("检查优化后的路径碰撞...")
            collision_check = check_path_collision(optimized_path, self.environment, self.vehicle.length,
                                                   self.vehicle.width)
            if collision_check['collision']:
                print("优化后的路径存在碰撞，使用原始路径")
                self.path = raw_path
            else:
                print("路径优化完成")
                self.path = optimized_path

            # 设置路径并开始仿真
            self.follower.set_path(self.path)
            self.simulating = True
            self.paused = False
            self.collision_detected = False
            self.collision_info = None
            self.status_text = "正在仿真..."
            self.status_color = BLUE

        except Exception as e:
            print(f"路径规划或优化过程中发生错误: {e}")
            traceback.print_exc()
            self.status_text = f"规划/优化错误: {e}"
            self.status_color = RED
            self.path = None
            self.goal_pos = None

    def _switch_planning_algorithm(self):
        """切换规划算法"""
        try:
            current_index = self.planning_algorithms.index(self.args.algorithm)
            next_index = (current_index + 1) % len(self.planning_algorithms)
            self.args.algorithm = self.planning_algorithms[next_index]
            self.status_text = f"规划算法切换为: {self.args.algorithm}"
            self.status_color = BLUE
            print(f"规划算法已切换为: {self.args.algorithm}")

            # 如果有目标点且不在仿真中，重新规划
            if self.goal_pos and not self.simulating:
                self._plan_path_to_goal()
        except ValueError:
            # 如果当前算法不在列表中，重置为第一个
            self.args.algorithm = self.planning_algorithms[0]
            self.status_text = f"算法重置为: {self.args.algorithm}"
            self.status_color = YELLOW

    def _switch_steering_mode(self):
        """切换车辆转向模式"""
        try:
            current_index = self.steering_modes.index(self.vehicle.steering_mode)
            next_index = (current_index + 1) % len(self.steering_modes)
            new_mode = self.steering_modes[next_index]
            self.vehicle.set_steering_mode(new_mode)
            self.status_text = f"转向模式切换为: {new_mode}"
            self.status_color = BLUE
        except ValueError:
            # 如果当前模式不在列表中，重置为 normal
            self.vehicle.set_steering_mode("normal")
            self.status_text = "转向模式重置为: normal"
            self.status_color = YELLOW

    def _reset_simulation(self):
        """重置仿真状态和车辆位置"""
        self.vehicle.x, self.vehicle.y = self.start_pos
        self.vehicle.heading = math.pi * 3 / 2  # 朝下
        self.vehicle.speed = 0.0
        self.vehicle.acceleration = 0.0
        self.vehicle.front_steer_angle = 0.0
        self.vehicle.trajectory = [self.start_pos]

        self.goal_pos = None
        self.path = None
        self.follower.set_path([])  # 清空路径跟随器的路径
        self.simulating = False
        self.paused = False
        self.collision_detected = False
        self.collision_info = None
        self.status_text = "仿真已重置，请选择目标点"
        self.status_color = BLACK
        print("仿真已重置")

    def _update(self):
        """更新仿真状态（车辆移动、碰撞检测、目标到达检测）"""
        if self.simulating and not self.paused and not self.collision_detected:
            # 计算控制输入
            throttle, brake, steer = self.follower.get_control(self.vehicle)

            # 更新车辆状态
            self.vehicle.update(throttle, brake, steer, self.dt * self.simulation_speed)

            # 检查碰撞和安全边界
            self.collision_info = check_vehicle_collision(self.vehicle, self.environment)
            if self.collision_info['collision']:
                self.collision_detected = True
                self.simulating = False
                self.status_text = "检测到碰撞！按R键重置"
                self.status_color = RED
                print(f"检测到碰撞！位置: {self.collision_info['position']}")
                return
            elif self.collision_info['safety_warning']:
                # 只在没有更严重状态时显示警告
                if self.status_color != RED:
                    self.status_text = "警告：进入安全边界"
                    self.status_color = YELLOW
                    # print(f"安全警告！位置: {self.collision_info['position']}") # 避免过多打印

            # 检查是否到达终点
            if self.goal_pos:
                dx = self.goal_pos[0] - self.vehicle.x
                dy = self.goal_pos[1] - self.vehicle.y
                distance_to_goal = math.sqrt(dx * dx + dy * dy)

                if distance_to_goal < 0.5 and abs(self.vehicle.speed) < 0.1:
                    self.simulating = False
                    self.status_text = "到达目标点！按T键重选"
                    self.status_color = GREEN
                    print("车辆已到达目标点")

    def _handle_events(self):
        """处理Pygame事件，包括基类事件和演示特定事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE or event.key == pygame.K_p:  # P键也用于暂停/继续
                    self.paused = not self.paused
                    pause_state = "暂停" if self.paused else "继续"
                    self.status_text = f"仿真已{pause_state}"
                    self.status_color = BLUE if not self.paused else YELLOW
                elif event.key == pygame.K_r:  # 重置
                    self._reset_simulation()
                elif event.key == pygame.K_c:  # 切换控制方法
                    self._switch_control_method()  # 使用基类的方法切换
                    # 更新状态文本
                    self.status_text = f"控制方法切换为: {self.current_control_method}"
                    self.status_color = BLUE
                elif event.key == pygame.K_e:  # 切换规划算法
                    self._switch_planning_algorithm()
                elif event.key == pygame.K_s:  # 切换转向模式
                    self._switch_steering_mode()
                elif event.key == pygame.K_t:  # 重新选择目标点
                    self._reset_simulation()  # 重置包含清空目标和路径
                    self.status_text = "等待选择目标点"
                    self.status_color = BLACK

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 鼠标右键点击选择目标点 (Button 3)
                if event.button == 3 and not self.simulating:
                    screen_pos = event.pos
                    world_pos = self.screen_to_world(screen_pos[0], screen_pos[1])
                    print(f"右键点击屏幕: {screen_pos} -> 世界坐标: {world_pos}")

                    parking_spot = None
                    if isinstance(self.environment, ParkingEnvironment):
                        try:
                            parking_spot = self.environment.find_parking_spot(world_pos)
                        except Exception as e:
                            print(f"查找停车位时出错: {e}")

                    if parking_spot:
                        self.goal_pos = (parking_spot.x, parking_spot.y)
                        print(f"选择停车位，目标点设为: {self.goal_pos}")
                        self._plan_path_to_goal()
                    else:
                        print(f"点击位置不在可用停车位内，将 {world_pos} 设为目标点")
                        # 检查点击位置是否在障碍物内
                        temp_vehicle = VehicleModel(world_pos[0], world_pos[1], 0, self.vehicle.length,
                                                    self.vehicle.width)
                        if check_vehicle_collision(temp_vehicle, self.environment)['collision']:
                            self.status_text = "目标点在障碍物内，请重新选择"
                            self.status_color = RED
                        else:
                            self.goal_pos = world_pos
                            self._plan_path_to_goal()
        return True

    def _draw_parking_environment(self):
        """专门绘制停车场环境，包括停车位特殊显示"""
        if not self.environment:
            return

        # 绘制所有障碍物
        for obs in self.environment.obstacles:
            screen_center_x, screen_center_y = self.world_to_screen(obs.x, obs.y)

            if obs.type == "rectangle":
                width = obs.width * self.scale
                height = obs.height * self.scale
                is_filled = getattr(obs, 'is_filled', True)
                line_width = int(getattr(obs, 'line_width', 1))  # Pygame线宽需为整数
                angle = getattr(obs, 'angle', 0)
                color = getattr(obs, 'color', GRAY)

                # 特殊处理停车位
                if hasattr(obs, 'is_parking_spot') and obs.is_parking_spot:
                    spot_color = RED if obs.occupied else GREEN
                    # 绘制停车位边框（不填充）
                    rect = pygame.Rect(0, 0, width, height)
                    surface = pygame.Surface((width, height), pygame.SRCALPHA)
                    pygame.draw.rect(surface, spot_color + (150, ), rect, 2)  # 半透明边框

                    if angle != 0:
                        rotated_surface = pygame.transform.rotate(surface, -angle)
                        rotated_rect = rotated_surface.get_rect(center=(screen_center_x, screen_center_y))
                        self.screen.blit(rotated_surface, rotated_rect)
                    else:
                        rect.center = (screen_center_x, screen_center_y)
                        self.screen.blit(surface, rect)
                else:
                    # 绘制普通矩形障碍物
                    rect = pygame.Rect(0, 0, width, height)
                    surface = pygame.Surface((width, height), pygame.SRCALPHA)
                    draw_width = 0 if is_filled else max(1, line_width)  # 填充或线宽
                    pygame.draw.rect(surface, color, rect, draw_width)

                    if angle != 0:
                        rotated_surface = pygame.transform.rotate(surface, -angle)
                        rotated_rect = rotated_surface.get_rect(center=(screen_center_x, screen_center_y))
                        self.screen.blit(rotated_surface, rotated_rect)
                    else:
                        rect.center = (screen_center_x, screen_center_y)
                        self.screen.blit(surface, rect)

            elif obs.type == "circle":
                radius = int(obs.radius * self.scale)
                is_filled = getattr(obs, 'is_filled', True)
                line_width = int(getattr(obs, 'line_width', 1))
                color = getattr(obs, 'color', GRAY)
                draw_width = 0 if is_filled else max(1, line_width)
                pygame.draw.circle(self.screen, color, (screen_center_x, screen_center_y), radius, draw_width)
