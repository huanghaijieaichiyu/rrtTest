#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自定义地图创建示例脚本

本脚本展示了如何创建自定义地图并将其用于CarSim仿真。
按照教程中的步骤逐步实现，方便理解和修改。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time

from simulation.environment import Environment
from simulation.visualization import Visualization
from simulation.carsim_interface import CarSimInterface


def create_basic_environment():
    """创建基础环境"""
    env = Environment(
        width=100.0,    # 地图宽度
        height=100.0,   # 地图高度
        grid_size=1.0   # 网格大小
    )
    return env


def add_obstacles(env):
    """添加障碍物"""
    # 添加圆形障碍物
    env.add_obstacle(
        x=30.0,
        y=30.0,
        obstacle_type="circle",
        radius=5.0
    )

    # 添加矩形障碍物
    env.add_obstacle(
        x=70.0,
        y=40.0,
        obstacle_type="rectangle",
        width=10.0,
        height=20.0
    )


def create_road_properties():
    """创建路面属性"""
    return {
        'friction': 0.85,        # 摩擦系数
        'roughness': 0.02,       # 粗糙度
        'banking_angle': 0.0,    # 横向倾角
        'elevation': 0.0,        # 纵向坡度
        'lanes': {
            'number': 2,         # 车道数
            'width': 3.75        # 车道宽度(m)
        }
    }


def add_reference_line_and_markers(env):
    """添加参考线和标记点"""
    # 创建参考线
    reference_line = [
        (0, 0),
        (20, 20),
        (40, 40),
        (60, 40),
        (80, 60),
        (100, 80)
    ]
    env.add_reference_line(reference_line)

    # 添加检查点
    checkpoints = [
        {'x': 20, 'y': 20, 'description': '检查点1'},
        {'x': 60, 'y': 40, 'description': '检查点2'},
        {'x': 100, 'y': 80, 'description': '终点'}
    ]

    for point in checkpoints:
        env.add_marker(
            x=point['x'],
            y=point['y'],
            marker_type='checkpoint',
            description=point['description']
        )

    return reference_line, checkpoints


def save_map_config(env, reference_line, checkpoints, road_properties):
    """保存地图配置"""
    map_config = {
        'environment': {
            'width': env.width,
            'height': env.height,
            'grid_size': env.grid_size,
            'obstacles': env.get_obstacles(),
            'reference_line': reference_line,
            'checkpoints': checkpoints,
            'properties': road_properties
        }
    }

    # 确保目录存在
    map_file = 'data/maps/custom_map.yaml'
    os.makedirs(os.path.dirname(map_file), exist_ok=True)

    # 保存配置
    with open(map_file, 'w', encoding='utf-8') as f:
        yaml.dump(map_config, f, allow_unicode=True)

    return map_file


def convert_to_carsim_format(map_file):
    """转换为CarSim格式"""
    with open(map_file, 'r', encoding='utf-8') as f:
        map_data = yaml.safe_load(f)

    env_data = map_data['environment']
    reference_line = env_data['reference_line']
    properties = env_data['properties']

    road_data = []
    road_data.append("* Road geometry data")
    road_data.append("* Station X Y Z Bank Grade")
    road_data.append("ROAD_DZ")

    for i, point in enumerate(reference_line):
        station = i * 10  # 每10米一个站点
        x, y = point
        z = 0  # 高程
        bank = properties['banking_angle']
        grade = properties['elevation']

        road_data.append(
            f"{station:.1f} {x:.3f} {y:.3f} {z:.3f} {bank:.3f} {grade:.3f}")

    return "\n".join(road_data)


def validate_map(env, vis):
    """验证地图"""
    # 创建测试路径
    test_path = [
        (0, 0),
        (20, 20),
        (40, 40),
        (60, 40),
        (80, 60),
        (100, 80)
    ]

    # 检查碰撞
    is_valid = env.check_path_collision(test_path)
    print(f"路径是否有效: {is_valid}")

    # 可视化
    vis.plot_path(
        path=test_path,
        show_environment=True,
        title="测试路径验证"
    )
    plt.show()

    return test_path


def test_carsim_interface(carsim_road_file, test_path):
    """测试CarSim接口"""
    try:
        # 初始化接口
        carsim = CarSimInterface(config_path="config/carsim_config.yaml")

        # 加载地图
        carsim.load_custom_road(carsim_road_file)

        # 执行路径
        success = carsim.execute_path(test_path)
        print(f"路径执行状态: {'成功' if success else '失败'}")

        # 获取结果
        results = carsim.get_simulation_results()
        carsim.visualize_results(results)

        # 断开连接
        carsim.disconnect()

    except Exception as e:
        print(f"CarSim接口测试失败: {e}")


def main():
    """主函数"""
    try:
        # 1. 创建环境
        env = create_basic_environment()
        vis = Visualization(env)

        # 2. 添加障碍物
        add_obstacles(env)

        # 3. 创建路面属性
        road_properties = create_road_properties()

        # 4. 添加参考线和标记点
        reference_line, checkpoints = add_reference_line_and_markers(env)

        # 5. 保存地图配置
        map_file = save_map_config(
            env, reference_line, checkpoints, road_properties)
        print(f"地图配置已保存到: {map_file}")

        # 6. 转换为CarSim格式
        carsim_road_data = convert_to_carsim_format(map_file)
        carsim_road_file = 'data/maps/carsim_road.pars'

        with open(carsim_road_file, 'w', encoding='utf-8') as f:
            f.write(carsim_road_data)
        print(f"CarSim路面文件已生成: {carsim_road_file}")

        # 7. 验证地图
        test_path = validate_map(env, vis)

        # 8. 测试CarSim接口
        test_carsim_interface(carsim_road_file, test_path)

        print("地图创建和测试完成")

    except Exception as e:
        print(f"程序执行失败: {e}")
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()
