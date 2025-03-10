#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
场景生成器

用于生成不同类型和复杂度的训练场景，包括：
- 随机障碍物场景
- 迷宫型场景
- 房间型场景
- 走廊型场景
- 混合型场景
"""

from typing import List, Optional
import numpy as np
from simulation.environment import Environment


class ScenarioGenerator:
    """场景生成器类"""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        min_obstacle_size: float = 2.0,
        max_obstacle_size: float = 10.0,
        min_gap: float = 5.0  # 障碍物之间的最小间隔
    ):
        """
        初始化场景生成器

        参数:
            width: 场景宽度
            height: 场景高度
            min_obstacle_size: 最小障碍物尺寸
            max_obstacle_size: 最大障碍物尺寸
            min_gap: 障碍物之间的最小间隔
        """
        self.width = width
        self.height = height
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.min_gap = min_gap

    def _is_valid_position(
        self,
        x: float,
        y: float,
        size: float,
        obstacles: List[dict]
    ) -> bool:
        """检查位置是否有效（不与其他障碍物重叠）"""
        for obs in obstacles:
            if obs['type'] == 'circle':
                dist = np.hypot(x - obs['x'], y - obs['y'])
                if dist < (size + obs['radius'] + self.min_gap):
                    return False
            elif obs['type'] == 'rectangle':
                # 简化的矩形碰撞检测
                half_width = size + obs['width']/2 + self.min_gap
                half_height = size + obs['height']/2 + self.min_gap
                if (abs(x - obs['x']) < half_width and
                        abs(y - obs['y']) < half_height):
                    return False
        return True

    def _generate_random_obstacle(
        self,
        obstacles: List[dict],
        max_attempts: int = 100
    ) -> Optional[dict]:
        """生成随机障碍物"""
        for _ in range(max_attempts):
            # 随机选择障碍物类型
            obs_type = np.random.choice(['circle', 'rectangle'])

            # 随机生成位置和大小
            size = np.random.uniform(
                self.min_obstacle_size,
                self.max_obstacle_size
            )
            x = np.random.uniform(size, self.width - size)
            y = np.random.uniform(size, self.height - size)

            if self._is_valid_position(x, y, size, obstacles):
                if obs_type == 'circle':
                    return {
                        'type': 'circle',
                        'x': x,
                        'y': y,
                        'radius': size/2
                    }
                else:
                    width = size * np.random.uniform(0.5, 2.0)
                    height = size * np.random.uniform(0.5, 2.0)
                    return {
                        'type': 'rectangle',
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    }
        return None

    def generate_random_scenario(
        self,
        num_obstacles: int,
        density: float = 0.3
    ) -> Environment:
        """
        生成随机障碍物场景

        参数:
            num_obstacles: 障碍物数量
            density: 障碍物密度 (0-1)

        返回:
            环境对象
        """
        env = Environment(width=self.width, height=self.height)
        obstacles = []

        # 根据密度调整实际障碍物数量
        area = self.width * self.height
        max_area = area * density
        current_area = 0

        for _ in range(num_obstacles):
            if current_area >= max_area:
                break

            obstacle = self._generate_random_obstacle(obstacles)
            if obstacle:
                obstacles.append(obstacle)
                if obstacle['type'] == 'circle':
                    current_area += np.pi * obstacle['radius']**2
                    env.add_obstacle(
                        x=obstacle['x'],
                        y=obstacle['y'],
                        obstacle_type='circle',
                        radius=obstacle['radius']
                    )
                else:
                    current_area += obstacle['width'] * obstacle['height']
                    env.add_obstacle(
                        x=obstacle['x'],
                        y=obstacle['y'],
                        obstacle_type='rectangle',
                        width=obstacle['width'],
                        height=obstacle['height']
                    )

        return env

    def generate_maze_scenario(
        self,
        cell_size: float = 10.0,
        complexity: float = 0.7
    ) -> Environment:
        """
        生成迷宫型场景

        参数:
            cell_size: 迷宫单元格大小
            complexity: 迷宫复杂度 (0-1)

        返回:
            环境对象
        """
        env = Environment(width=self.width, height=self.height)

        # 计算网格大小
        nx = int(self.width / cell_size)
        ny = int(self.height / cell_size)

        # 创建迷宫墙壁
        for i in range(nx):
            for j in range(ny):
                if np.random.random() < complexity:
                    # 随机选择是否添加墙壁
                    wall_type = np.random.choice(['horizontal', 'vertical'])
                    if wall_type == 'horizontal':
                        env.add_obstacle(
                            x=i * cell_size + cell_size/2,
                            y=j * cell_size,
                            obstacle_type='rectangle',
                            width=cell_size,
                            height=self.min_obstacle_size
                        )
                    else:
                        env.add_obstacle(
                            x=i * cell_size,
                            y=j * cell_size + cell_size/2,
                            obstacle_type='rectangle',
                            width=self.min_obstacle_size,
                            height=cell_size
                        )

        return env

    def generate_room_scenario(
        self,
        num_rooms: int = 4,
        min_room_size: float = 20.0,
        max_room_size: float = 40.0
    ) -> Environment:
        """
        生成房间型场景

        参数:
            num_rooms: 房间数量
            min_room_size: 最小房间尺寸
            max_room_size: 最大房间尺寸

        返回:
            环境对象
        """
        env = Environment(width=self.width, height=self.height)
        rooms = []

        # 生成房间
        for _ in range(num_rooms):
            width = np.random.uniform(min_room_size, max_room_size)
            height = np.random.uniform(min_room_size, max_room_size)
            x = np.random.uniform(0, self.width - width)
            y = np.random.uniform(0, self.height - height)

            # 添加房间墙壁
            # 上墙
            env.add_obstacle(
                x=x + width/2,
                y=y,
                obstacle_type='rectangle',
                width=width,
                height=self.min_obstacle_size
            )
            # 下墙
            env.add_obstacle(
                x=x + width/2,
                y=y + height,
                obstacle_type='rectangle',
                width=width,
                height=self.min_obstacle_size
            )
            # 左墙
            env.add_obstacle(
                x=x,
                y=y + height/2,
                obstacle_type='rectangle',
                width=self.min_obstacle_size,
                height=height
            )
            # 右墙
            env.add_obstacle(
                x=x + width,
                y=y + height/2,
                obstacle_type='rectangle',
                width=self.min_obstacle_size,
                height=height
            )

            # 添加门
            door_pos = np.random.choice(['top', 'bottom', 'left', 'right'])
            door_size = 5.0
            if door_pos == 'top':
                env.add_obstacle(
                    x=x + width/2,
                    y=y,
                    obstacle_type='rectangle',
                    width=door_size,
                    height=self.min_obstacle_size
                )
            elif door_pos == 'bottom':
                env.add_obstacle(
                    x=x + width/2,
                    y=y + height,
                    obstacle_type='rectangle',
                    width=door_size,
                    height=self.min_obstacle_size
                )
            elif door_pos == 'left':
                env.add_obstacle(
                    x=x,
                    y=y + height/2,
                    obstacle_type='rectangle',
                    width=self.min_obstacle_size,
                    height=door_size
                )
            else:
                env.add_obstacle(
                    x=x + width,
                    y=y + height/2,
                    obstacle_type='rectangle',
                    width=self.min_obstacle_size,
                    height=door_size
                )

            rooms.append({
                'x': x,
                'y': y,
                'width': width,
                'height': height
            })

        return env

    def generate_corridor_scenario(
        self,
        corridor_width: float = 10.0,
        num_turns: int = 3
    ) -> Environment:
        """
        生成走廊型场景

        参数:
            corridor_width: 走廊宽度
            num_turns: 转弯次数

        返回:
            环境对象
        """
        env = Environment(width=self.width, height=self.height)

        # 生成走廊路径点
        points = [(0.0, self.height/2)]
        current_x = 0.0
        current_y = self.height/2

        for _ in range(num_turns):
            if len(points) % 2 == 1:
                # 垂直移动
                new_y = np.random.uniform(
                    corridor_width,
                    self.height - corridor_width
                )
                points.append((current_x, new_y))
                current_y = new_y
            else:
                # 水平移动
                new_x = min(
                    current_x + np.random.uniform(
                        20.0,
                        self.width/num_turns
                    ),
                    self.width
                )
                points.append((current_x, new_y))
                current_x = new_x

        # 确保最后一个点到达右边界
        points.append((self.width, current_y))

        # 根据路径点生成走廊墙壁
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # 计算走廊方向
            dx = x2 - x1
            dy = y2 - y1

            # 添加走廊墙壁
            if abs(dx) > abs(dy):  # 水平走廊
                env.add_obstacle(
                    x=(x1 + x2)/2,
                    y=y1 - corridor_width/2,
                    obstacle_type='rectangle',
                    width=abs(dx),
                    height=self.min_obstacle_size
                )
                env.add_obstacle(
                    x=(x1 + x2)/2,
                    y=y1 + corridor_width/2,
                    obstacle_type='rectangle',
                    width=abs(dx),
                    height=self.min_obstacle_size
                )
            else:  # 垂直走廊
                env.add_obstacle(
                    x=x1 - corridor_width/2,
                    y=(y1 + y2)/2,
                    obstacle_type='rectangle',
                    width=self.min_obstacle_size,
                    height=abs(dy)
                )
                env.add_obstacle(
                    x=x1 + corridor_width/2,
                    y=(y1 + y2)/2,
                    obstacle_type='rectangle',
                    width=self.min_obstacle_size,
                    height=abs(dy)
                )

        return env

    def generate_mixed_scenario(
        self,
        num_random_obstacles: int = 10,
        num_rooms: int = 2,
        corridor_width: float = 8.0
    ) -> Environment:
        """
        生成混合型场景，包含随机障碍物、房间和走廊

        参数:
            num_random_obstacles: 随机障碍物数量
            num_rooms: 房间数量
            corridor_width: 走廊宽度

        返回:
            环境对象
        """
        # 首先生成基础的走廊场景
        env = self.generate_corridor_scenario(
            corridor_width=corridor_width,
            num_turns=2
        )

        # 添加房间
        room_env = self.generate_room_scenario(
            num_rooms=num_rooms,
            min_room_size=15.0,
            max_room_size=25.0
        )
        for obstacle in room_env.obstacles:
            # 提取障碍物参数并添加到环境中
            params = {
                'x': getattr(obstacle, 'x', 0.0),
                'y': getattr(obstacle, 'y', 0.0),
                'obstacle_type': getattr(obstacle, 'type', 'rectangle')
            }
            if params['obstacle_type'] == 'circle':
                params['radius'] = getattr(obstacle, 'radius', 1.0)
            else:  # rectangle
                params['width'] = getattr(obstacle, 'width', 1.0)
                params['height'] = getattr(obstacle, 'height', 1.0)
            env.add_obstacle(**params)

        # 添加随机障碍物
        random_env = self.generate_random_scenario(
            num_obstacles=num_random_obstacles,
            density=0.1
        )
        for obstacle in random_env.obstacles:
            # 提取障碍物参数并添加到环境中
            params = {
                'x': getattr(obstacle, 'x', 0.0),
                'y': getattr(obstacle, 'y', 0.0),
                'obstacle_type': getattr(obstacle, 'type', 'rectangle')
            }
            if params['obstacle_type'] == 'circle':
                params['radius'] = getattr(obstacle, 'radius', 1.0)
            else:  # rectangle
                params['width'] = getattr(obstacle, 'width', 1.0)
                params['height'] = getattr(obstacle, 'height', 1.0)
            env.add_obstacle(**params)

        return env
