import numpy as np
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional
from rrt.astar import AStar
from simulation.environment import Environment
from rrt.rrt_base import RRT, Node
from rrt.rrt_star import RRTStar
from rrt.informed_rrt import InformedRRTStar
from rrt.dijkstra import Dijkstra
from simulation.pygame_simulator import PygameSimulator
import math
from shapely.geometry import Point, LineString, Polygon

# 定义动态障碍物类


class DynamicObstacle:
    def __init__(self, x0, y0, vx, vy, width, height):
        """初始化动态障碍物
        参数:
            x0, y0: 初始位置
            vx, vy: 速度分量
            width, height: 宽度和高度
        """
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.width = width
        self.height = height

    def get_position_at_time(self, t):
        """计算在时间t时的位置"""
        x = self.x0 + self.vx * t
        y = self.y0 + self.vy * t
        return x, y

# 修改Environment类以支持停车场场景


class ParkingEnvironment(Environment):
    def __init__(self, width, height):
        """初始化停车场环境"""
        super().__init__(width, height)
        self.dynamic_obstacles = []

    def add_dynamic_obstacle(self, x0, y0, vx, vy, width, height):
        """添加动态障碍物"""
        self.dynamic_obstacles.append(
            DynamicObstacle(x0, y0, vx, vy, width, height))

    def check_segment_collision_with_time(self, start, end, start_time, end_time):
        """检查路径段在时间区间内是否发生碰撞"""
        # 检查静态障碍物
        if super().check_segment_collision(start, end):
            return True

        # 检查动态障碍物
        for dyn_obs in self.dynamic_obstacles:
            for t in np.arange(start_time, end_time, 0.1):
                robot_pos = interpolate_position(
                    start, end, start_time, end_time, t)
                obs_pos = dyn_obs.get_position_at_time(t)
                if check_collision(robot_pos, obs_pos, dyn_obs.width, dyn_obs.height):
                    return True
        return False


def interpolate_position(start, end, start_time, end_time, t):
    """插值计算机器人当前位置"""
    ratio = (t - start_time) / (end_time - start_time)
    x = start[0] + ratio * (end[0] - start[0])
    y = start[1] + ratio * (end[1] - start[1])
    return x, y


def check_collision(robot_pos, obs_pos, obs_width, obs_height):
    """简化的碰撞检测：使用圆形近似"""
    robot_radius = 2.5  # 假设机器人为圆形，半径为2.5
    obs_radius = np.hypot(obs_width / 2, obs_height / 2)
    dist = np.hypot(robot_pos[0] - obs_pos[0], robot_pos[1] - obs_pos[1])
    return dist < (robot_radius + obs_radius)

# 修改RRTStar以支持时间维度


class TimedRRTStar(RRTStar):
    def __init__(
        self,
        start,
        goal,
        env,
        max_iterations,
        step_size,
        robot_speed=1.0
    ):
        """初始化时间维度的RRT*规划器"""
        super().__init__(start, goal, env, step_size, max_iterations)
        self.robot_speed = robot_speed
        self.node_list = [self.start]  # Initialize node list with start node

    def _random_node(self) -> Node:
        """生成随机节点"""
        if np.random.random() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y)
        return Node(
            np.random.uniform(self.min_x, self.max_x),
            np.random.uniform(self.min_y, self.max_y)
        )

    def _steer(self, from_node: Node, to_node: Node) -> Node:
        """从一个节点朝向另一个节点扩展"""
        dist = np.hypot(to_node.x - from_node.x, to_node.y - from_node.y)
        if dist > self.step_size:
            theta = np.arctan2(to_node.y - from_node.y,
                               to_node.x - from_node.x)
            new_x = from_node.x + self.step_size * np.cos(theta)
            new_y = from_node.y + self.step_size * np.sin(theta)
            new_node = Node(new_x, new_y)
        else:
            new_node = Node(to_node.x, to_node.y)

        new_node.parent = from_node
        new_node.cost = from_node.cost + dist
        new_node.path_x = from_node.path_x.copy()
        new_node.path_y = from_node.path_y.copy()
        new_node.path_x.append(new_node.x)
        new_node.path_y.append(new_node.y)
        return new_node

    def plan(self) -> List[Tuple[float, float]]:
        """规划路径，考虑时间维度"""
        for _ in range(self.max_iterations):
            # 随机采样
            rnd = self._random_node()

            # 找到最近的节点
            nearest_ind = self._get_nearest_node_index(rnd)
            nearest_node = self.node_list[nearest_ind]

            # 扩展树
            new_node = self._steer(nearest_node, rnd)

            # 计算时间增量
            dist = np.hypot(new_node.x - nearest_node.x,
                            new_node.y - nearest_node.y)
            time_increment = dist / self.robot_speed

            # 检查时间维度的碰撞
            if not self.env.check_segment_collision_with_time(
                (nearest_node.x, nearest_node.y),
                (new_node.x, new_node.y),
                nearest_node.cost,  # 使用cost作为时间
                nearest_node.cost + time_increment
            ):
                # 找到附近的节点
                near_nodes = self._find_near_nodes(new_node)
                # 选择最优父节点
                new_node = self._choose_parent(new_node, near_nodes)

                if new_node.parent is not None:
                    # 添加新节点
                    self.node_list.append(new_node)
                    # 尝试重新连接
                    self._rewire(new_node, near_nodes)

                    # 检查是否到达目标
                    if self._is_near_goal(new_node):
                        if self._check_segment(new_node, self.goal):
                            self._connect_to_goal(new_node)
                            return self._extract_path()

        return []

# 创建停车场场景


def create_default_scene(width, height, margin, car_width, car_length,
                         spot_width, spot_length, lane_width):
    """创建默认停车场场景"""
    env = ParkingEnvironment(width, height)

    # 添加基本边界
    wall_thickness = 0.5
    # 左右边界---预留入口
    env.add_obstacle(
        x=margin/2 - 20, y=height/2,
        obstacle_type="rectangle",
        width=wall_thickness, height=height
    )
    env.add_obstacle(
        x=width-margin/2 - 20, y=height/2,
        obstacle_type="rectangle",
        width=wall_thickness, height=height
    )

    # 优化停车场布局参数
    total_columns = 6  # 减少列数，增加机动空间
    spots_per_column = 8  # 减少每列停车位数量
    static_car_ratio = 0.7  # 降低静态车辆占用率

    # 记录车道中心线，用于后续添加装饰物
    lane_centers = []

    parking_spots = []
    parking_orientations = []
    for col in range(total_columns):
        is_left_oriented = col % 2 == 0
        base_x = margin + (col // 2) * (2 * spot_length + lane_width)

        # 记录车道中心线
        if col % 2 == 1:
            lane_center_x = base_x + spot_length + lane_width/2
            lane_centers.append(lane_center_x)

        if is_left_oriented:
            x = base_x + spot_length / 2
            orientation = 0  # 车头朝右
        else:
            x = base_x + spot_length * 1.5 + lane_width
            orientation = np.pi  # 车头朝左

        # 在每列中留出更多间隔
        for i in range(spots_per_column):
            y = margin + spot_width * (i + 0.5) + i * 1.0  # 增加停车位之间的间隔
            parking_spots.append((x, y))
            parking_orientations.append(orientation)

    # 添加静态车辆
    num_static_cars = int(len(parking_spots) * static_car_ratio)
    static_indices = random.sample(range(len(parking_spots)), num_static_cars)
    static_spots = []
    static_orientations = []

    for idx in static_indices:
        spot = parking_spots[idx]
        orientation = parking_orientations[idx]
        static_spots.append(spot)
        static_orientations.append(orientation)

    # 添加静态车辆，确保车辆在停车位内居中
    for spot, orientation in zip(static_spots, static_orientations):
        env.add_obstacle(
            x=spot[0],
            y=spot[1],
            obstacle_type="rectangle",
            width=car_length,
            height=car_width,
            angle=orientation
        )

    return env, static_spots, parking_spots


def create_parking_scenario(use_random_scene=False):
    """创建停车场仿真场景

    参数:
        use_random_scene: 是否使用随机生成场景，默认False使用默认场景
    """
    # 优化车辆和停车位尺寸
    car_width = 1.8  # 减小车辆宽度
    car_length = 4.5  # 减小车辆长度
    spot_width = 3.0  # 保持停车位宽度
    spot_length = 6.0  # 保持停车位长度
    lane_width = 10.0  # 增加车道宽度
    margin = 8.0  # 增加边界margin
    entrance_width = 12.0  # 增加入口宽度
    entrance_margin = 15.0  # 保持入口外的安全距离

    # 计算停车场总尺寸（考虑到减少的列数）
    total_columns = 4  # 与create_default_scene保持一致
    spots_per_column = 6  # 与create_default_scene保持一致
    width = (spot_length * 2 + lane_width) * \
        (total_columns // 2) + margin * 2
    height = (spot_width * spots_per_column + margin * 2)

    # 如果不使用随机场景，直接返回默认场景
    if not use_random_scene:
        print("\n使用默认停车场场景")
        env, static_spots, parking_spots = create_default_scene(
            width, height, margin, car_width, car_length,
            spot_width, spot_length, lane_width
        )

        # 添加上下边界墙
        wall_thickness = 0.5
        env.add_obstacle(
            x=width/2,
            y=margin/2,
            obstacle_type="rectangle",
            width=width-margin,
            height=wall_thickness
        )
        env.add_obstacle(
            x=width/2,
            y=height-margin/2,
            obstacle_type="rectangle",
            width=width-margin,
            height=wall_thickness
        )

        # 设置默认起点（从下原点进入）
        start = (0, height/2)

        # 从未占用的停车位中选择目标
        available_spots = [
            spot for spot in parking_spots if spot not in static_spots]

        if not available_spots:
            raise ValueError("默认场景中没有可用的停车位")

        goal = random.choice(available_spots)
        goal_orientation = 0.0  # 默认朝向

        return env, start, goal, goal_orientation
    else:
        # 随机场景生成逻辑（包含验证）
        print("\n尝试生成随机停车场场景")

        # 设置入口和出口位置
        entrance_pos = width * 0.2
        exit_pos = width * 0.8

        # 设置起点在入口外的空旷区域
        if random.random() < 0.5:
            start = (entrance_pos + entrance_width/2, -entrance_margin)
        else:
            start = (exit_pos + entrance_width/2, height + entrance_margin)

        # 多次尝试生成有效场景
        max_attempts = 10
        for attempt in range(max_attempts):
            print(f"\n尝试生成场景 #{attempt + 1}")

            # 创建环境
            env = ParkingEnvironment(width, height)

            # 添加边界墙
            wall_thickness = 0.5
            # 左右边界
            env.add_obstacle(
                x=margin/2 - 10,
                y=height/2,
                obstacle_type="rectangle",
                width=wall_thickness,
                height=height
            )
            env.add_obstacle(
                x=width-margin/2 - 10,
                y=height/2,
                obstacle_type="rectangle",
                width=wall_thickness,
                height=height
            )

            # 上边界（分段）
            env.add_obstacle(
                x=(entrance_pos-margin)/2,
                y=margin/2,
                obstacle_type="rectangle",
                width=entrance_pos-margin,
                height=wall_thickness
            )

            middle_section_width = exit_pos - entrance_pos - entrance_width
            env.add_obstacle(
                x=(entrance_pos+entrance_width+exit_pos)/2,
                y=margin/2,
                obstacle_type="rectangle",
                width=middle_section_width,
                height=wall_thickness
            )

            end_section_width = width - exit_pos - entrance_width
            env.add_obstacle(
                x=(width+exit_pos+entrance_width)/2,
                y=margin/2,
                obstacle_type="rectangle",
                width=end_section_width,
                height=wall_thickness
            )

            # 下边界（分段）
            env.add_obstacle(
                x=(entrance_pos-margin)/2,
                y=height-margin/2,
                obstacle_type="rectangle",
                width=entrance_pos-margin,
                height=wall_thickness
            )

            env.add_obstacle(
                x=(entrance_pos+entrance_width+exit_pos)/2,
                y=height-margin/2,
                obstacle_type="rectangle",
                width=middle_section_width,
                height=wall_thickness
            )

            env.add_obstacle(
                x=(width+exit_pos+entrance_width)/2,
                y=height-margin/2,
                obstacle_type="rectangle",
                width=end_section_width,
                height=wall_thickness
            )

            # 生成停车位布局
            parking_spots = []
            parking_orientations = []
            valid_parking_areas = []  # 记录有效的停车区域

            # 创建停车位和有效停车区域
            for col in range(total_columns):
                is_left_oriented = col % 2 == 0
                base_x = margin + (col // 2) * (2 * spot_length + lane_width)

                if is_left_oriented:
                    x = base_x + spot_length / 2
                    orientation = 0  # 车头朝右
                    area_x = base_x
                else:
                    x = base_x + spot_length * 1.5 + lane_width
                    orientation = np.pi  # 车头朝左
                    area_x = base_x + spot_length + lane_width

                # 记录这一列的停车区域
                valid_parking_areas.append({
                    'x': area_x,
                    'width': spot_length,
                    'y_start': margin,
                    'y_end': margin + spot_width * spots_per_column
                })

                for i in range(spots_per_column):
                    y = margin + spot_width * (i + 0.5)
                    # 确保车辆完全在停车位内
                    adjusted_x = x
                    adjusted_y = y
                    parking_spots.append((adjusted_x, adjusted_y))
                    parking_orientations.append(orientation)

            # 随机选择停车位作为静态车辆（占用率70%）
            num_static_cars = int(len(parking_spots) * 0.7)
            static_indices = random.sample(
                range(len(parking_spots)), num_static_cars)
            static_spots = []
            static_orientations = []

            # 只在合法停车位内生成静态车辆
            for idx in static_indices:
                spot = parking_spots[idx]
                orientation = parking_orientations[idx]

                # 验证车辆是否在有效停车区域内
                is_valid = False
                for area in valid_parking_areas:
                    area_right = area['x'] + area['width']
                    x_valid = area['x'] <= spot[0]
                    x_in_area = x_valid and spot[0] <= area_right
                    y_valid = area['y_start'] <= spot[1]
                    y_in_area = y_valid and spot[1] <= area['y_end']
                    if x_in_area and y_in_area:
                        is_valid = True
                        break

                if is_valid:
                    static_spots.append(spot)
                    static_orientations.append(orientation)

            # 添加静态车辆（确保在停车位内）
            for spot, orientation in zip(static_spots, static_orientations):
                env.add_obstacle(
                    x=spot[0],
                    y=spot[1],
                    obstacle_type="rectangle",
                    width=car_length,
                    height=car_width,
                    angle=orientation  # 添加车辆朝向
                )

            # 验证起点位置
            if not check_position_valid(env, start, margin=5.0):
                print("起点位置无效，重新生成场景...")
                continue

            # 选择可用的停车位（必须验证可达性）
            available_spots = []
            available_orientations = []
            for i, spot in enumerate(parking_spots):
                if (spot not in static_spots and
                        check_position_valid(env, spot, margin=3.0)):
                    available_spots.append(spot)
                    available_orientations.append(parking_orientations[i])

            if not available_spots:
                print("没有可用的停车位，重新生成场景...")
                continue

            # 随机选择目标停车位
            goal_index = random.randrange(len(available_spots))
            goal = available_spots[goal_index]
            goal_orientation = available_orientations[goal_index]

            # 验证路径可行性
            args = argparse.Namespace(
                algorithm='rrt_star',
                iterations=5000,
                step_size=2.0,
                robot_speed=3.0
            )
            if check_path_feasibility(env, start, goal, 'rrt_star', args):
                print(f"成功生成有效的随机停车场场景（尝试次数：{attempt + 1}）")
                return env, start, goal, goal_orientation

            print("路径规划测试失败，重新生成场景...")

        # 如果随机生成失败，返回默认场景
        print("\n随机场景生成失败，使用默认停车场场景")
        return create_parking_scenario(use_random_scene=False)


def get_algorithm_specific_params(algorithm: str, args) -> Dict[str, Any]:
    """获取算法特定的参数"""
    base_params = {
        'max_iterations': args.iterations,
        'step_size': args.step_size
    }

    params = {
        'astar': {'resolution': 0.5, 'diagonal_movement': True, 'weight': 1.0},
        'rrt': base_params,
        'rrt_star': {**base_params, 'rewire_factor': 1.5},
        'informed_rrt': {**base_params, 'focus_factor': 1.0},
        'timed_rrt': {**base_params, 'robot_speed': args.robot_speed},
        'dijkstra': {'resolution': 1.0, 'diagonal_movement': True}
    }

    return params.get(algorithm, {})


def create_planner(algorithm: str, start: tuple, goal: tuple, env: Environment, args):
    """创建路径规划器"""
    # 获取车辆尺寸参数
    vehicle_width = 1.8  # 车辆宽度
    vehicle_length = 4.5  # 车辆长度

    # 基本参数，所有规划器都需要
    common_params = {
        'start': start,
        'goal': goal,
        'env': env,
    }

    # RRT系列算法需要的额外参数
    rrt_params = {
        'vehicle_width': vehicle_width,
        'vehicle_length': vehicle_length
    }

    planners = {
        'astar': AStar,
        'rrt': RRT,
        'rrt_star': RRTStar,
        'informed_rrt': InformedRRTStar,
        'timed_rrt': TimedRRTStar,
        'dijkstra': Dijkstra
    }

    if algorithm not in planners:
        raise ValueError(f"不支持的算法: {algorithm}")

    planner_class = planners[algorithm]
    algorithm_params = get_algorithm_specific_params(algorithm, args)

    # 根据算法类型添加额外参数
    if algorithm in ['rrt', 'rrt_star', 'informed_rrt', 'timed_rrt']:
        common_params.update(rrt_params)

    return planner_class(**common_params, **algorithm_params)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='停车场路径规划仿真')

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['rrt', 'rrt_star', 'informed_rrt',
                 'timed_rrt', 'dijkstra', 'astar'],
        default='rrt_star',
        help='路径规划算法'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=10000,
        help='最大迭代次数'
    )

    parser.add_argument(
        '--step_size',
        type=float,
        default=2.0,
        help='步长'
    )

    parser.add_argument(
        '--robot_speed',
        type=float,
        default=3.0,
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
        default='default',
        help='车辆控制算法'
    )

    return parser.parse_args()


def try_plan_path(
    planner,
    max_retries: int = 10
) -> Optional[List[Tuple[float, float]]]:
    """尝试多次规划路径

    参数:
        planner: 路径规划器
        max_retries: 最大重试次数

    返回:
        成功时返回路径，失败时返回None
    """
    for attempt in range(max_retries):
        path = planner.plan()
        if path:
            print(f"在第 {attempt + 1} 次尝试中成功规划路径")
            return path
        print(f"第 {attempt + 1} 次尝试失败，继续尝试...")

    print(f"经过 {max_retries} 次尝试后仍未找到可行路径")
    return None


def check_position_valid(env: Environment, pos: tuple, margin: float = 5.0) -> bool:
    """检查位置是否有效（使用A*算法验证可达性）

    参数:
        env: 环境对象
        pos: 位置坐标 (x, y)
        margin: 安全边距

    返回:
        位置是否有效
    """
    x, y = pos

    # 基本碰撞检测
    if env.check_collision(pos):
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
            0 <= test_point[1] <= env.height and
                not env.check_collision(test_point)):

            test_planner = create_planner(
                'dijkstra', pos, test_point, env, args)
            path = test_planner.plan()
            if path:
                reachable_directions += 1

    return reachable_directions >= min_required


def check_path_feasibility(
    env: Environment,
    start: tuple,
    goal: tuple,
    algorithm: str,
    args
) -> bool:
    """检查路径可行性

    参数:
        env: 环境对象
        start: 起点坐标
        goal: 终点坐标
        algorithm: 使用的规划算法
        args: 算法参数

    返回:
        路径是否可行
    """
    # 创建规划器进行测试
    test_planner = create_planner(algorithm, start, goal, env, args)

    # 使用较大的迭代次数进行测试
    test_planner.max_iterations = args.iterations * 2

    # 尝试规划路径
    path = test_planner.plan()
    if not path:
        print("路径规划测试失败：无法找到可行路径")
        return False

    # 验证路径连续性
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        if env.check_segment_collision(p1, p2):
            print("路径规划测试失败：路径段存在碰撞")
            return False

    print("路径可行性验证通过")
    return True


def main():
    """主函数：创建场景、规划路径并仿真"""
    # 解析命令行参数
    args = parse_args()

    # 创建场景
    env, start, _, _ = create_parking_scenario(
        use_random_scene=args.random_scene
    )

    # 创建仿真器并设置环境
    simulator = PygameSimulator()
    simulator.set_environment(env)

    # 交互式选择目标点并规划路径
    interactive_planning(simulator, env, start, args)


class VehicleModel:
    """简化的车辆动力学模型"""

    def __init__(self, x: float = 0, y: float = 0, heading: float = 0,
                 length: float = 4.5, width: float = 1.8):
        """
        初始化车辆模型

        参数:
            x: 初始x坐标
            y: 初始y坐标
            heading: 初始朝向角度(弧度)
            length: 车辆长度(米)
            width: 车辆宽度(米)
        """
        self.x = x
        self.y = y
        self.heading = heading  # 弧度
        self.length = length
        self.width = width
        self.speed = 0.0  # m/s
        self.acceleration = 0.0  # m/s^2
        self.steer_angle = 0.0  # 转向角(弧度)
        self.wheelbase = 2.7  # 轴距(米)

        # 记录轨迹
        self.trajectory = [(x, y)]

        # 车辆控制参数
        self.max_speed = 5.0  # m/s
        self.max_accel = 2.0   # m/s^2
        self.max_brake = 4.0   # m/s^2
        self.max_steer = math.pi/4  # 最大转向角(弧度)

    def get_corners(self) -> List[Tuple[float, float]]:
        """获取车辆四个角的坐标(用于碰撞检测和渲染)"""
        half_length = self.length / 2
        half_width = self.width / 2

        # 车辆本地坐标系中的四个角
        corners_local = [
            (half_length, half_width),   # 右前
            (half_length, -half_width),  # 左前
            (-half_length, -half_width),  # 左后
            (-half_length, half_width)   # 右后
        ]

        # 转换到世界坐标系
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        corners_world = []
        for lx, ly in corners_local:
            wx = self.x + lx * cos_h - ly * sin_h
            wy = self.y + lx * sin_h + ly * cos_h
            corners_world.append((wx, wy))

        return corners_world

    def update(self, throttle: float, brake: float, steer: float, dt: float) -> None:
        """
        更新车辆状态

        参数:
            throttle: 油门输入 [0, 1]
            brake: 制动输入 [0, 1]
            steer: 转向输入 [-1, 1]
            dt: 时间步长(秒)
        """
        # 计算加速度
        if throttle > 0:
            self.acceleration = throttle * self.max_accel
        else:
            self.acceleration = 0

        if brake > 0:
            self.acceleration -= brake * self.max_brake

        # 更新速度
        self.speed += self.acceleration * dt
        self.speed = max(0, min(self.speed, self.max_speed))  # 限制速度范围

        # 更新转向角
        self.steer_angle = steer * self.max_steer

        # 简化的自行车模型
        if abs(self.speed) > 0.1:  # 当速度足够大时才转向
            turn_radius = self.wheelbase / \
                math.tan(abs(self.steer_angle) + 1e-10)
            angular_velocity = self.speed / turn_radius if self.steer_angle != 0 else 0

            if self.steer_angle < 0:
                angular_velocity = -angular_velocity

            # 更新位置和朝向
            self.heading += angular_velocity * dt
            self.heading = self.heading % (2 * math.pi)  # 规范化到 [0, 2π]

        # 根据当前朝向和速度更新位置
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

        # 记录轨迹
        self.trajectory.append((self.x, self.y))


def check_vehicle_collision(vehicle, env):
    """
    车辆碰撞检测算法，只检查与障碍物的碰撞

    参数:
        vehicle: 车辆模型对象
        env: 环境对象

    返回:
        collision_info: 碰撞信息字典，包含是否碰撞、碰撞位置和碰撞对象
    """
    # 初始化碰撞信息
    collision_info = {
        'collision': False,
        'position': None,
        'obstacle': None,
        'distance': float('inf')
    }

    # 获取车辆四个角的坐标
    corners = vehicle.get_corners()

    # 创建车辆多边形
    vehicle_polygon = Polygon(corners)

    # 检查车辆与每个障碍物的碰撞
    for obstacle in env.obstacles:
        # 根据障碍物类型创建不同的几何形状
        if hasattr(obstacle, 'type') and obstacle.type == 'circle':
            # 圆形障碍物
            obstacle_circle = Point(
                obstacle.x, obstacle.y).buffer(obstacle.radius)
            if vehicle_polygon.intersects(obstacle_circle):
                collision_info['collision'] = True
                collision_info['position'] = (obstacle.x, obstacle.y)
                collision_info['obstacle'] = obstacle
                # 计算碰撞距离（简化为中心点距离）
                dist = np.hypot(vehicle.x - obstacle.x, vehicle.y - obstacle.y)
                if dist < collision_info['distance']:
                    collision_info['distance'] = dist
        else:
            # 矩形障碍物
            x_min = obstacle.x - obstacle.width / 2
            x_max = obstacle.x + obstacle.width / 2
            y_min = obstacle.y - obstacle.height / 2
            y_max = obstacle.y + obstacle.height / 2

            # 考虑障碍物的旋转角度
            if hasattr(obstacle, 'angle') and obstacle.angle != 0:
                # 创建旋转后的矩形
                rect_corners = [
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max)
                ]

                # 旋转矩形的角点
                cos_angle = math.cos(-obstacle.angle)  # 负号是因为pygame和数学坐标系方向相反
                sin_angle = math.sin(-obstacle.angle)
                rotated_corners = []

                for x, y in rect_corners:
                    # 平移到原点
                    tx = x - obstacle.x
                    ty = y - obstacle.y
                    # 旋转
                    rx = tx * cos_angle - ty * sin_angle
                    ry = tx * sin_angle + ty * cos_angle
                    # 平移回原位置
                    rotated_corners.append((rx + obstacle.x, ry + obstacle.y))

                obstacle_polygon = Polygon(rotated_corners)
            else:
                # 不旋转的矩形
                obstacle_polygon = Polygon([
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max)
                ])

            # 检查碰撞
            if vehicle_polygon.intersects(obstacle_polygon):
                collision_info['collision'] = True
                collision_info['position'] = (obstacle.x, obstacle.y)
                collision_info['obstacle'] = obstacle
                # 计算碰撞距离（简化为中心点距离）
                dist = np.hypot(vehicle.x - obstacle.x, vehicle.y - obstacle.y)
                if dist < collision_info['distance']:
                    collision_info['distance'] = dist

    return collision_info


def check_path_collision(path, env, vehicle_length, vehicle_width, steps=10):
    """
    检查路径是否与障碍物碰撞

    参数:
        path: 路径点列表
        env: 环境对象
        vehicle_length: 车辆长度
        vehicle_width: 车辆宽度
        steps: 每段路径的检查步数

    返回:
        collision_points: 碰撞点列表，如果没有碰撞则为空列表
    """
    collision_points = []

    if len(path) < 2:
        return collision_points

    # 创建一个临时车辆模型用于碰撞检测
    temp_vehicle = VehicleModel(0, 0, 0, vehicle_length, vehicle_width)

    # 检查路径上的每个段
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]

        # 计算段的方向
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.sqrt(dx*dx + dy*dy)

        if segment_length < 0.1:  # 忽略非常短的段
            continue

        # 计算段的朝向角度
        heading = math.atan2(dy, dx)

        # 在段上均匀采样点进行检查
        for j in range(steps + 1):
            t = j / steps
            # 插值计算位置
            x = start[0] + t * dx
            y = start[1] + t * dy

            # 更新临时车辆位置和朝向
            temp_vehicle.x = x
            temp_vehicle.y = y
            temp_vehicle.heading = heading

            # 检查碰撞
            collision_info = check_vehicle_collision(temp_vehicle, env)
            if collision_info['collision']:
                collision_points.append((x, y))
                break  # 找到碰撞点后停止检查当前段

    return collision_points


def interactive_planning(simulator, env, start, args):
    """交互式路径规划

    参数:
        simulator: 仿真器对象
        env: 环境对象
        start: 起点坐标
        args: 命令行参数
    """
    import pygame
    import math

    # 初始化pygame
    pygame.init()

    # 设置窗口大小为1080p
    screen_width = 860
    screen_height = 640
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("停车场路径规划 - 右键选择目标点，按T重新选择，按R重置车辆位置")

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
    current_control_method = args.control_method

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

        # 确保 path 存在且有效
        if not path:
            simulating = False
            status_text = "仿真完成，按T键重新选择目标点"
            status_color = (0, 128, 0)  # 绿色
            return

        # 计算控制输入
        throttle, brake, steer = follower.get_control(vehicle)

        # 更新车辆状态
        vehicle.update(throttle, brake, steer, dt * simulation_speed)

        # 检查碰撞
        collision_info = check_vehicle_collision(vehicle, env)
        if collision_info['collision']:
            collision_detected = True
            simulating = False
            status_text = "检测到碰撞：与障碍物相撞！按R键重置车辆位置"
            status_color = (255, 0, 0)  # 红色
            print(f"检测到碰撞！位置: {collision_info['position']}")
            return

        # 检查是否到达终点
        if follower.current_target_idx >= len(follower.path) - 1 and vehicle.speed < 0.1:
            simulating = False
            status_text = "仿真完成，按T键重新选择目标点"
            status_color = (0, 128, 0)  # 绿色
            return

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

        # 绘制车身 - 根据状态设置不同颜色
        if collision_detected:
            car_color = (255, 255, 0)  # 黄色表示碰撞
        else:
            car_color = (255, 0, 0)  # 红色表示正常

        pygame.draw.polygon(screen, car_color, screen_corners)

        # 绘制车头方向
        head_x = vehicle.x + math.cos(vehicle.heading) * vehicle.length / 2
        head_y = vehicle.y + math.sin(vehicle.heading) * vehicle.length / 2
        center_screen = (int(vehicle.x * scale + offset_x),
                         int(vehicle.y * scale + offset_y))
        head_screen = (int(head_x * scale + offset_x),
                       int(head_y * scale + offset_y))
        pygame.draw.line(screen, (0, 0, 255), center_screen, head_screen, 2)

        # 如果发生碰撞，绘制碰撞点
        if collision_detected and collision_info and collision_info['position']:
            cx, cy = collision_info['position']
            collision_screen_pos = (
                int(cx * scale + offset_x), int(cy * scale + offset_y))
            pygame.draw.circle(screen, (255, 0, 0),
                               collision_screen_pos, 5)  # 红色圆点表示碰撞位置

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

            # 创建旋转后的矩形
            rect = pygame.Rect(x - width/2, y - height/2, width, height)
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(surface, (100, 100, 100, 200), surface.get_rect())

            # 旋转并绘制
            if hasattr(obs, 'angle'):
                rotated_surface = pygame.transform.rotate(
                    surface, -obs.angle * 180 / np.pi)
                screen.blit(rotated_surface,
                            rotated_surface.get_rect(center=(x, y)))
            else:
                screen.blit(surface, rect)

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

        # 显示提示信息
        font = get_font(24)

        # 显示状态文本
        status_surface = font.render(status_text, True, status_color)
        screen.blit(status_surface, (screen_width // 2 - status_surface.get_width() // 2,
                                     screen_height - 40))

        # 显示操作提示
        text1 = font.render("右键点击选择目标点", True, (0, 0, 0))
        text2 = font.render("按T键重新选择目标点", True, (0, 0, 0))
        text3 = font.render(f"当前算法: {args.algorithm}", True, (0, 0, 0))
        text4 = font.render(f"控制方法: {current_control_method}", True, (0, 0, 0))
        text5 = font.render("按C键切换控制方法", True, (0, 0, 0))
        text6 = font.render("按R键重置车辆位置", True, (0, 0, 0))
        text7 = font.render("碰撞检测: " + ("已触发" if collision_detected else "正常"),
                            True, (255, 0, 0) if collision_detected else (0, 0, 0))

        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 40))
        screen.blit(text3, (10, 70))
        screen.blit(text4, (10, 100))
        screen.blit(text5, (10, 130))
        screen.blit(text6, (10, 160))
        screen.blit(text7, (10, 190))

        # 如果发生碰撞，显示碰撞信息
        if collision_detected and collision_info:
            collision_text = "碰撞类型: 障碍物碰撞"
            text8 = font.render(collision_text, True, (255, 0, 0))
            screen.blit(text8, (10, 220))

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
            args.algorithm, (vehicle.x, vehicle.y), goal, env, args)
        path = try_plan_path(planner)

        # 如果找到路径，检查路径是否有碰撞
        if path:
            collision_points = check_path_collision(
                path, env, vehicle.length, vehicle.width)
            if collision_points:
                print(f"警告：规划的路径存在 {len(collision_points)} 个碰撞点")
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

                # 检查目标点是否有效
                if env.check_collision(goal):
                    print("目标点在障碍物内，请重新选择")
                    goal = None
                    status_text = "目标点无效，请重新选择"
                    status_color = (255, 0, 0)  # 红色
                else:
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
                print(f"控制方法已切换为: {current_control_method}")

        # 更新仿真
        if simulating:
            simulate_path()

        # 绘制场景
        draw_scene()

        # 控制帧率
        clock.tick(60)

    pygame.quit()


class PathFollower:
    """路径跟踪控制器"""

    def __init__(self, lookahead=5.0, control_method='default'):
        """
        初始化路径跟踪控制器

        参数:
            lookahead: 前瞻距离(米)
            control_method: 控制方法 ('default', 'pid', 'mpc', 'lqr')
        """
        self.path = []
        self.lookahead = lookahead
        self.current_target_idx = 0
        self.control_method = control_method
        self.target_speed = 5.0  # 目标速度(m/s)

        # PID控制参数
        self.pid_params = {
            'kp_steer': 0.7,   # 转向比例系数
            'ki_steer': 0.01,  # 转向积分系数
            'kd_steer': 0.1,   # 转向微分系数
            'kp_speed': 0.5,   # 速度比例系数
            'ki_speed': 0.01,  # 速度积分系数
            'kd_speed': 0.05   # 速度微分系数
        }
        self.steer_error_prev = 0.0
        self.steer_error_sum = 0.0
        self.speed_error_prev = 0.0
        self.speed_error_sum = 0.0

        # MPC控制参数
        self.mpc_params = {
            'horizon': 10,     # 预测步长
            'dt': 0.1,         # 时间步长
            'q_x': 1.0,        # 纵向误差权重
            'q_y': 2.0,        # 横向误差权重
            'q_heading': 3.0,  # 朝向误差权重
            'r_steer': 1.0,    # 转向输入权重
            'r_accel': 0.5     # 加速度输入权重
        }

        # LQR控制参数
        self.lqr_params = {
            'q_y': 1.0,        # 横向误差权重
            'q_heading': 2.0,  # 朝向误差权重
            'q_speed': 0.5,    # 速度误差权重
            'r_steer': 0.1,    # 转向输入权重
            'r_accel': 0.1     # 加速度输入权重
        }

    def set_path(self, path):
        """设置跟踪路径"""
        self.path = path
        self.current_target_idx = 0

    def set_control_method(self, method):
        """设置控制方法"""
        if method in ['default', 'pid', 'mpc', 'lqr']:
            self.control_method = method
        else:
            print(f"不支持的控制方法: {method}，使用默认方法")
            self.control_method = 'default'

    def get_control(self, vehicle):
        """获取控制输入"""
        if not self.path:
            return 0.0, 0.0, 0.0  # 无路径时不动作

        if self.control_method == 'pid':
            return self._pid_control(vehicle)
        elif self.control_method == 'mpc':
            return self._mpc_control(vehicle)
        elif self.control_method == 'lqr':
            return self._lqr_control(vehicle)
        else:
            return self._default_control(vehicle)

    def _default_control(self, vehicle):
        """默认控制方法"""
        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')

        # 向前找到一个在前瞻距离范围内的点
        for i in range(self.current_target_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                target_idx = i

            if dist > self.lookahead:
                break

        # 更新当前目标点索引
        self.current_target_idx = target_idx

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算车辆到目标点的向量
        dx = tx - vehicle.x
        dy = ty - vehicle.y

        # 计算目标点相对于车头的角度
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle.heading

        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # 根据角度误差计算转向
        steer = heading_error / (math.pi/4)  # 假设最大转向角为π/4
        steer = max(-1.0, min(1.0, steer))  # 限制在 [-1, 1] 范围内

        # 简单的速度控制：根据转向角的大小调整速度
        throttle = 0.5 * (1.0 - 0.5 * abs(steer))
        brake = 0.0

        # 如果即将转弯，提前减速
        if abs(steer) > 0.5:
            throttle *= 0.5

        return throttle, brake, steer

    def _pid_control(self, vehicle):
        """PID控制方法"""
        # 动态调整前瞻距离 - 根据车速调整
        dynamic_lookahead = max(3.0, min(self.lookahead, vehicle.speed * 0.8))

        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')
        closest_idx = target_idx

        # 首先找到最近点
        for i in range(self.current_target_idx,
                       min(self.current_target_idx + 30, len(self.path))):
            if i >= len(self.path):
                break

            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 从最近点开始，找到前瞻距离范围内的目标点
        target_idx = closest_idx
        for i in range(closest_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist > dynamic_lookahead:
                target_idx = i
                break

        # 确保目标点不会超出路径范围
        target_idx = min(target_idx, len(self.path) - 1)

        # 更新当前目标点索引，但不要后退
        self.current_target_idx = max(self.current_target_idx, closest_idx)

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算车辆到目标点的向量
        dx = tx - vehicle.x
        dy = ty - vehicle.y

        # 计算目标点相对于车头的角度
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle.heading

        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # PID控制 - 转向
        # 限制积分项，防止积分饱和
        self.steer_error_sum = max(-3.0, min(3.0,
                                             self.steer_error_sum + heading_error))
        steer_error_diff = heading_error - self.steer_error_prev
        self.steer_error_prev = heading_error

        # 计算PID控制输出
        steer = (self.pid_params['kp_steer'] * heading_error +
                 self.pid_params['ki_steer'] * self.steer_error_sum +
                 self.pid_params['kd_steer'] * steer_error_diff)

        # 限制在 [-1, 1] 范围内
        steer = max(-1.0, min(1.0, steer))

        # PID控制 - 速度
        speed_error = self.target_speed - vehicle.speed

        # 限制积分项，防止积分饱和
        self.speed_error_sum = max(-5.0, min(5.0,
                                             self.speed_error_sum + speed_error))
        speed_error_diff = speed_error - self.speed_error_prev
        self.speed_error_prev = speed_error

        # 计算PID控制输出
        throttle_brake = (self.pid_params['kp_speed'] * speed_error +
                          self.pid_params['ki_speed'] * self.speed_error_sum +
                          self.pid_params['kd_speed'] * speed_error_diff)

        # 将输出转换为油门和刹车
        if throttle_brake >= 0:
            throttle = min(1.0, throttle_brake)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -throttle_brake)

        # 如果即将转弯，减速
        if abs(steer) > 0.5:
            throttle *= 0.5
        elif abs(steer) > 0.3:
            throttle *= 0.7

        return throttle, brake, steer

    def _mpc_control(self, vehicle):
        """简化的模型预测控制方法"""
        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')

        # 向前找到一个在前瞻距离范围内的点
        for i in range(self.current_target_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                target_idx = i

            if dist > self.lookahead:
                break

        # 更新当前目标点索引
        self.current_target_idx = target_idx

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算车辆到目标点的向量
        dx = tx - vehicle.x
        dy = ty - vehicle.y

        # 计算目标点相对于车头的角度
        target_angle = math.atan2(dy, dx)
        heading_error = target_angle - vehicle.heading

        # 规范化到 [-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # 简化的MPC控制 - 使用预测模型计算最优控制输入
        # 这里使用简化方法：将状态误差与权重相乘作为控制输入
        steer = (self.mpc_params['q_y'] * math.sin(heading_error) +
                 self.mpc_params['q_heading'] * heading_error) / self.mpc_params['r_steer']

        # 限制在 [-1, 1] 范围内
        steer = max(-1.0, min(1.0, steer))

        # 计算速度误差
        speed_error = self.target_speed - vehicle.speed

        # 计算油门和刹车
        accel_cmd = self.mpc_params['q_x'] * \
            speed_error / self.mpc_params['r_accel']

        if accel_cmd >= 0:
            throttle = min(1.0, accel_cmd)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -accel_cmd)

        # 如果即将转弯，减速
        if abs(steer) > 0.5:
            throttle *= 0.5
        elif abs(steer) > 0.3:
            throttle *= 0.7

        return throttle, brake, steer

    def _lqr_control(self, vehicle):
        """简化的线性二次型调节器控制方法"""
        # 寻找目标点
        target_idx = self.current_target_idx
        min_dist = float('inf')

        # 向前找到一个在前瞻距离范围内的点
        for i in range(self.current_target_idx, len(self.path)):
            tx, ty = self.path[i]
            dist = math.sqrt((tx - vehicle.x)**2 + (ty - vehicle.y)**2)

            if dist < min_dist:
                min_dist = dist
                target_idx = i

            if dist > self.lookahead:
                break

        # 更新当前目标点索引
        self.current_target_idx = target_idx

        # 如果已接近终点，减速
        if target_idx >= len(self.path) - 3:
            return 0.0, 0.3, 0.0  # 轻踩刹车

        # 获取目标点
        tx, ty = self.path[target_idx]

        # 计算参考路径的切线方向（简化版）
        next_idx = min(target_idx + 1, len(self.path) - 1)
        next_x, next_y = self.path[next_idx]
        ref_heading = math.atan2(next_y - ty, next_x - tx)

        # 计算状态误差
        dx = tx - vehicle.x
        dy = ty - vehicle.y
        dheading = ref_heading - vehicle.heading

        # 规范化到 [-π, π]
        while dheading > math.pi:
            dheading -= 2 * math.pi
        while dheading < -math.pi:
            dheading += 2 * math.pi

        # 计算横向误差（车辆坐标系中）
        cos_heading = math.cos(vehicle.heading)
        sin_heading = math.sin(vehicle.heading)
        lateral_error = -dx * sin_heading + dy * cos_heading

        # 简化的LQR控制 - 在实际应用中应求解Riccati方程
        # 这里使用简化方法：将状态误差与权重相乘作为控制输入
        steer = (self.lqr_params['q_y'] * lateral_error +
                 self.lqr_params['q_heading'] * dheading) / self.lqr_params['r_steer']

        # 限制在 [-1, 1] 范围内
        steer = max(-1.0, min(1.0, steer))

        # 计算速度误差
        speed_error = self.target_speed - vehicle.speed

        # 计算油门和刹车
        accel_cmd = self.lqr_params['q_speed'] * \
            speed_error / self.lqr_params['r_accel']

        if accel_cmd >= 0:
            throttle = min(1.0, accel_cmd)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -accel_cmd)

        # 如果即将转弯，减速
        if abs(steer) > 0.5:
            throttle *= 0.5

        return throttle, brake, steer


if __name__ == "__main__":
    main()
