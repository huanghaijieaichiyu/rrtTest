#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RRT-PyTorch-CarSim项目主入口

提供命令行界面和项目主要功能的入口点。
"""

import os
import sys
import argparse
import time
import yaml
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

# 导入项目模块
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar
from rrt.informed_rrt import InformedRRTStar
from simulation.environment import Environment
from simulation.carsim_interface import CarSimInterface
from simulation.visualization import Visualization
from ml.models.rrt_nn import SamplingNetwork, CollisionNet, HeuristicNet, EndToEndRRTNet


# 设置日志
def setup_logger(log_level: str = 'INFO') -> logging.Logger:
    """设置日志记录器"""
    # 创建日志目录
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 获取日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'无效的日志级别: {log_level}')

    # 配置日志
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                f'logs/rrt_{time.strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('RRT-PyTorch-CarSim')


# 加载配置
def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# 创建RRT规划器
def create_planner(
    algorithm: str,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    env: Environment,
    config: Dict[str, Any]
):
    """
    创建路径规划器

    参数:
        algorithm: 算法类型，'rrt', 'rrt_star', 或 'informed_rrt'
        start: 起点坐标
        goal: 终点坐标
        env: 环境对象
        config: 配置字典

    返回:
        路径规划器对象
    """
    # 获取RRT参数
    rrt_config = config.get('rrt', {})
    step_size = rrt_config.get('step_size', 5.0)
    max_iterations = rrt_config.get('max_iterations', 1000)
    goal_sample_rate = rrt_config.get('goal_sample_rate', 0.1)
    search_radius = rrt_config.get('search_radius', 50.0)

    # 创建规划器
    if algorithm.lower() == 'rrt':
        return RRT(
            start=start,
            goal=goal,
            env=env,
            step_size=step_size,
            max_iterations=max_iterations,
            goal_sample_rate=goal_sample_rate,
            search_radius=search_radius
        )
    elif algorithm.lower() == 'rrt_star':
        return RRTStar(
            start=start,
            goal=goal,
            env=env,
            step_size=step_size,
            max_iterations=max_iterations,
            goal_sample_rate=goal_sample_rate,
            search_radius=search_radius
        )
    elif algorithm.lower() == 'informed_rrt':
        return InformedRRTStar(
            start=start,
            goal=goal,
            env=env,
            step_size=step_size,
            max_iterations=max_iterations,
            goal_sample_rate=goal_sample_rate,
            search_radius=search_radius
        )
    else:
        raise ValueError(f"不支持的算法类型: {algorithm}")


# 执行路径规划
def run_planning(args, config: Dict[str, Any], logger: logging.Logger) -> List[Tuple[float, float]]:
    """
    执行路径规划

    参数:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器

    返回:
        规划路径，点列表
    """
    # 获取环境配置
    env_config = config.get('environment', {})
    width = env_config.get('width', 100.0)
    height = env_config.get('height', 100.0)

    # 创建环境
    if args.map_path:
        logger.info(f"从地图文件加载环境: {args.map_path}")
        env = Environment(map_path=args.map_path)
    elif env_config.get('random_obstacles', False):
        # 创建随机障碍物环境
        logger.info("创建随机障碍物环境")
        env = Environment(width=width, height=height)

        # 添加随机障碍物
        num_obstacles = env_config.get('num_random_obstacles', 10)
        obstacle_ratio = env_config.get('obstacle_type_ratio', 0.7)

        np.random.seed(config.get('project', {}).get('random_seed', 42))
        for _ in range(num_obstacles):
            x = np.random.uniform(width * 0.1, width * 0.9)
            y = np.random.uniform(height * 0.1, height * 0.9)

            if np.random.random() < obstacle_ratio:
                # 创建圆形障碍物
                radius = np.random.uniform(2.0, 10.0)
                env.add_obstacle(x, y, obstacle_type="circle", radius=radius)
            else:
                # 创建矩形障碍物
                w = np.random.uniform(5.0, 15.0)
                h = np.random.uniform(5.0, 15.0)
                angle = np.random.uniform(0, 2 * np.pi)
                env.add_obstacle(
                    x, y, obstacle_type="rectangle",
                    width=w, height=h, angle=angle
                )
    else:
        # 使用默认地图
        map_path = env_config.get('map_path', "data/maps/default_map.yaml")
        logger.info(f"从默认地图文件加载环境: {map_path}")
        env = Environment(map_path=map_path)

    # 获取起点和终点
    if args.start:
        start = tuple(map(float, args.start.split(',')))
    else:
        start = (width * 0.1, height * 0.1)

    if args.goal:
        goal = tuple(map(float, args.goal.split(',')))
    else:
        goal = (width * 0.9, height * 0.9)

    logger.info(f"规划路径从 {start} 到 {goal}")

    # 创建规划器
    algorithm = args.algorithm or config.get(
        'rrt', {}).get('algorithm', 'rrt_star')
    logger.info(f"使用算法: {algorithm}")

    planner = create_planner(algorithm, start, goal, env, config)

    # 执行规划
    logger.info("开始规划路径...")
    start_time = time.time()
    path = planner.plan()
    elapsed_time = time.time() - start_time

    if path:
        logger.info(f"找到路径，包含 {len(path)} 个点，用时 {elapsed_time:.3f} 秒")
    else:
        logger.warning(f"未能找到路径，用时 {elapsed_time:.3f} 秒")

    # 可视化
    if args.visualize or config.get('visualization', {}).get('show_path', True):
        logger.info("可视化规划结果")
        vis = Visualization(env)
        vis.plot_path(path, start=start, goal=goal)

        if args.save_fig or config.get('visualization', {}).get('save_figures', False):
            figures_path = config.get('visualization', {}).get(
                'figures_path', 'results/figures/')
            if not os.path.exists(figures_path):
                os.makedirs(figures_path)

            fig_path = os.path.join(
                figures_path, f"{algorithm}_path_{time.strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(fig_path, dpi=config.get(
                'visualization', {}).get('figure_dpi', 300))
            logger.info(f"图形已保存到: {fig_path}")

        plt.show()

    return path


# 运行CarSim仿真
def run_carsim_simulation(path: List[Tuple[float, float]], args, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    在CarSim中运行仿真

    参数:
        path: 规划路径
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    if not path:
        logger.error("路径为空，无法执行仿真")
        return

    # 创建CarSim接口
    carsim_config_path = args.carsim_config or "config/carsim_config.yaml"
    logger.info(f"使用CarSim配置: {carsim_config_path}")

    try:
        carsim = CarSimInterface(carsim_config_path)

        # 执行路径
        logger.info(f"在CarSim中执行包含 {len(path)} 个点的路径")
        success = carsim.execute_path(path)

        if success:
            logger.info("路径执行成功")

            # 仿真
            sim_duration = config.get('simulation', {}).get(
                'max_steps', 1000) * config.get('simulation', {}).get('step_size', 0.1)
            logger.info(f"启动CarSim仿真，持续时间: {sim_duration} 秒")

            carsim.start_simulation(duration=sim_duration)

            # 获取结果
            results = carsim.get_simulation_results()

            # 可视化结果
            if args.visualize or config.get('visualization', {}).get('show_path', True):
                logger.info("可视化仿真结果")
                carsim.visualize_results(results)

            # 断开连接
            carsim.disconnect()
        else:
            logger.error("路径执行失败")

    except Exception as e:
        logger.error(f"CarSim仿真失败: {e}")


# 运行PyTorch模型
def run_ml_model(args, config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    运行机器学习模型

    参数:
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器
    """
    # 检查CUDA可用性
    use_gpu = config.get('project', {}).get(
        'use_gpu', True) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"使用设备: {device}")

    # 获取模型配置
    model_config = config.get('model', {})
    model_type = args.model_type or model_config.get('type', 'mlp')
    hidden_sizes = model_config.get('hidden_sizes', [128, 64])
    logger.info(f"使用模型类型: {model_type}, 隐藏层大小: {hidden_sizes}")

    # 创建模型
    if model_type.lower() == 'sampling':
        logger.info("创建采样网络")
        model = SamplingNetwork(
            hidden_sizes=hidden_sizes,
            use_batch_norm=model_config.get('use_batch_norm', True),
            dropout_prob=model_config.get('dropout_prob', 0.2)
        )
    elif model_type.lower() == 'collision':
        logger.info("创建碰撞预测网络")
        model = CollisionNet(
            hidden_sizes=hidden_sizes,
            use_batch_norm=model_config.get('use_batch_norm', True),
            dropout_prob=model_config.get('dropout_prob', 0.2)
        )
    elif model_type.lower() == 'heuristic':
        logger.info("创建启发式函数网络")
        model = HeuristicNet(
            hidden_sizes=hidden_sizes,
            use_batch_norm=model_config.get('use_batch_norm', True),
            dropout_prob=model_config.get('dropout_prob', 0.2)
        )
    elif model_type.lower() == 'endtoend':
        logger.info("创建端到端RRT网络")
        model = EndToEndRRTNet(
            hidden_channels=[16, 32, 64],
            use_batch_norm=model_config.get('use_batch_norm', True)
        )
    else:
        logger.error(f"不支持的模型类型: {model_type}")
        return

    # 移动模型到设备
    model = model.to(device)
    logger.info(f"模型结构:\n{model}")

    # 加载预训练模型
    if args.load_model or model_config.get('load_pretrained', False):
        model_path = args.model_path or model_config.get(
            'pretrained_path', "data/model_weights/pretrained.pt")
        if os.path.exists(model_path):
            logger.info(f"加载预训练模型: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            logger.warning(f"预训练模型不存在: {model_path}")

    # 简单测试模型
    logger.info("测试模型...")

    if model_type.lower() == 'sampling':
        # 测试采样网络
        start = (10.0, 10.0)
        goal = (90.0, 90.0)
        width, height = 100.0, 100.0

        samples = model.sample(start, goal, width, height,
                               batch_size=5, device=device)
        logger.info(f"生成的采样点:\n{samples}")

    elif model_type.lower() == 'collision':
        # 测试碰撞预测网络
        start = (10.0, 10.0)
        end = (90.0, 90.0)

        collision = model.predict_collision(start, end, device=device)
        logger.info(f"碰撞预测结果: {collision}")

    elif model_type.lower() == 'heuristic':
        # 测试启发式函数网络
        node_pos = (10.0, 10.0)
        goal_pos = (90.0, 90.0)

        cost = model.estimate_cost(node_pos, goal_pos, device=device)
        logger.info(f"代价估计结果: {cost}")

    elif model_type.lower() == 'endtoend':
        # 测试端到端网络
        batch_size = 2
        height, width = 64, 64
        x = torch.randn(batch_size, 3, height, width, device=device)

        output = model(x)
        logger.info(f"端到端网络输出形状: {output.shape}")


# 创建命令行接口
def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='RRT-PyTorch-CarSim 路径规划和仿真工具')

    # 通用参数
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO',
                                 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日志级别')

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 路径规划命令
    plan_parser = subparsers.add_parser('plan', help='执行路径规划')
    plan_parser.add_argument('--algorithm', type=str,
                             choices=['rrt', 'rrt_star', 'informed_rrt'],
                             help='规划算法类型')
    plan_parser.add_argument('--start', type=str, help='起点坐标，格式: x,y')
    plan_parser.add_argument('--goal', type=str, help='终点坐标，格式: x,y')
    plan_parser.add_argument('--map-path', type=str, help='地图文件路径')
    plan_parser.add_argument(
        '--visualize', action='store_true', help='可视化规划结果')
    plan_parser.add_argument('--save-fig', action='store_true', help='保存图形')
    plan_parser.add_argument('--save-path', action='store_true', help='保存路径')

    # 仿真命令
    sim_parser = subparsers.add_parser('simulate', help='执行仿真')
    sim_parser.add_argument('--path', type=str, help='路径文件，如果不提供则执行规划')
    sim_parser.add_argument('--algorithm', type=str,
                            choices=['rrt', 'rrt_star', 'informed_rrt'],
                            help='规划算法类型（如果需要规划路径）')
    sim_parser.add_argument('--start', type=str, help='起点坐标，格式: x,y')
    sim_parser.add_argument('--goal', type=str, help='终点坐标，格式: x,y')
    sim_parser.add_argument('--map-path', type=str, help='地图文件路径')
    sim_parser.add_argument('--carsim-config', type=str, help='CarSim配置文件路径')
    sim_parser.add_argument('--visualize', action='store_true', help='可视化仿真结果')

    # ML命令
    ml_parser = subparsers.add_parser('ml', help='运行机器学习模型')
    ml_parser.add_argument('--model-type', type=str,
                           choices=['sampling', 'collision',
                                    'heuristic', 'endtoend'],
                           help='模型类型')
    ml_parser.add_argument('--load-model', action='store_true', help='加载预训练模型')
    ml_parser.add_argument('--model-path', type=str, help='模型文件路径')
    ml_parser.add_argument('--train', action='store_true', help='训练模型')
    ml_parser.add_argument('--test', action='store_true', help='测试模型')
    ml_parser.add_argument('--save-model', action='store_true', help='保存模型')

    return parser


# 主函数
def main():
    """程序主入口点"""
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()

    # 设置日志记录器
    logger = setup_logger(args.log_level)
    logger.info("启动 RRT-PyTorch-CarSim")

    # 加载配置
    config_path = args.config
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)

    # 根据命令执行相应功能
    if args.command == 'plan':
        logger.info("执行路径规划")
        path = run_planning(args, config, logger)

        # 保存路径
        if args.save_path and path:
            results_path = config.get('project', {}).get(
                'results_path', 'results/')
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            path_file = os.path.join(
                results_path, f"path_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            with open(path_file, 'w') as f:
                f.write("x,y\n")
                for point in path:
                    f.write(f"{point[0]},{point[1]}\n")

            logger.info(f"路径已保存到: {path_file}")

    elif args.command == 'simulate':
        logger.info("执行仿真")

        # 获取路径
        path = None
        if args.path:
            # 从文件加载路径
            try:
                path = []
                with open(args.path, 'r') as f:
                    # 跳过标题行
                    next(f)
                    for line in f:
                        x, y = line.strip().split(',')
                        path.append((float(x), float(y)))

                logger.info(f"从文件加载路径: {args.path}, 包含 {len(path)} 个点")
            except Exception as e:
                logger.error(f"加载路径文件失败: {e}")

        if not path:
            # 执行规划
            logger.info("未提供路径，执行规划")
            path = run_planning(args, config, logger)

        # 执行仿真
        run_carsim_simulation(path, args, config, logger)

    elif args.command == 'ml':
        logger.info("运行机器学习模型")
        run_ml_model(args, config, logger)

    else:
        logger.warning("未指定命令，请使用 'plan', 'simulate' 或 'ml'")
        parser.print_help()

    logger.info("执行完成")


if __name__ == "__main__":
    main()
