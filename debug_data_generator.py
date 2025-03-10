#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据生成器调试脚本
"""

import os
import sys
import traceback

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from simulation.environment import Environment
    from rrt.rrt_star import RRTStar
    print("成功导入 simulation 和 rrt 模块")
except ImportError as e:
    print(f"导入 simulation 或 rrt 模块失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试环境创建
try:
    env = Environment(width=1000.0, height=500.0)
    print("成功创建环境")

    # 添加一些障碍物
    env.add_obstacle(x=300, y=250, obstacle_type="circle", radius=50)
    env.add_obstacle(x=600, y=300, obstacle_type="rectangle",
                     width=100, height=80)
    print("成功添加障碍物")

    # 设置起点和终点
    start_point = (100, 100)
    goal_point = (900, 400)

    # 测试碰撞检测
    print(f"起点碰撞检测: {env.check_collision(start_point)}")
    print(f"终点碰撞检测: {env.check_collision(goal_point)}")

    # 测试 RRT* 规划
    planner = RRTStar(
        start=start_point,
        goal=goal_point,
        env=env,
        max_iterations=1000
    )
    print("成功创建 RRT* 规划器")

    path = planner.plan()
    if path:
        print(f"成功规划路径，路径点数: {len(path)}")
    else:
        print("路径规划失败")

except Exception as e:
    print(f"测试环境和规划器时出错: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试数据生成器
try:
    from ml.data.data_generator import DataGenerator, obstacle_to_dict
    print("成功导入 DataGenerator")

    generator = DataGenerator(
        env_width=1000.0,
        env_height=500.0,
        grid_size=(64, 64),
        min_obstacles=5,  # 减少障碍物数量以提高成功率
        max_obstacles=10,
        num_samples=100,  # 减少采样点数量以加快测试
        rrt_iterations=2000
    )
    print("成功创建数据生成器")

    # 测试随机环境生成
    test_env = generator.generate_random_environment()
    print(f"成功生成随机环境，障碍物数量: {len(test_env.obstacles)}")

    # 测试随机点生成
    try:
        start, goal = generator.generate_random_points(test_env)
        print(f"成功生成随机起点 {start} 和终点 {goal}")
    except Exception as e:
        print(f"生成随机点时出错: {e}")
        traceback.print_exc()

    # 测试采样点收集
    try:
        valid_samples, invalid_samples = generator.collect_samples(
            test_env, start, goal)
        print(f"成功收集采样点，有效: {len(valid_samples)}，无效: {len(invalid_samples)}")
    except Exception as e:
        print(f"收集采样点时出错: {e}")
        traceback.print_exc()

    # 测试单个样本生成
    print("尝试生成单个训练样本...")
    for i in range(5):  # 尝试 5 次
        try:
            example = generator.generate_example()
            if example is not None:
                print(f"成功生成训练样本 #{i+1}")
                print(f"  环境状态形状: {example.env_state.shape}")
                print(f"  障碍物数量: {len(example.obstacles)}")
                print(f"  路径点数: {len(example.path)}")
                print(f"  有效采样点数: {len(example.valid_samples)}")
                print(f"  无效采样点数: {len(example.invalid_samples)}")
                break
            else:
                print(f"生成样本 #{i+1} 失败，返回 None")
        except Exception as e:
            print(f"生成样本 #{i+1} 时出错: {e}")
            traceback.print_exc()

except ImportError as e:
    print(f"导入 DataGenerator 失败: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"测试数据生成器时出错: {e}")
    traceback.print_exc()
