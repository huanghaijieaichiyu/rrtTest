import numpy as np
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional
from simulation.environment import Environment
from rrt.rrt_base import RRT, Node
from rrt.rrt_star import RRTStar
from rrt.informed_rrt import InformedRRTStar
from rrt.dijkstra import Dijkstra
from simulation.pygame_simulator import PygameSimulator

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
    # 左右边界
    env.add_obstacle(
        x=margin/2, y=height/2,
        obstacle_type="rectangle",
        width=wall_thickness, height=height
    )
    env.add_obstacle(
        x=width-margin/2, y=height/2,
        obstacle_type="rectangle",
        width=wall_thickness, height=height
    )

    # 优化停车场布局参数
    total_columns = 4  # 减少列数，增加机动空间
    spots_per_column = 6  # 减少每列停车位数量
    static_car_ratio = 0.4  # 降低静态车辆占用率

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

        # 设置默认起点（从上方入口进入）
        start = (width * 0.2 + entrance_width/2, -entrance_margin/2)

        # 从未占用的停车位中选择目标
    available_spots = [
        spot for spot in parking_spots if spot not in static_spots]

    if not available_spots:
        raise ValueError("默认场景中没有可用的停车位")

    goal = random.choice(available_spots)
    goal_orientation = 0.0  # 默认朝向

    return env, start, goal, goal_orientation

    # 以下是随机场景生成逻辑（包含验证）
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
            x=margin/2,
            y=height/2,
            obstacle_type="rectangle",
            width=wall_thickness,
            height=height
        )
        env.add_obstacle(
            x=width-margin/2,
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
        'rrt': base_params,
        'rrt_star': {**base_params, 'rewire_factor': 1.5},
        'informed_rrt': {**base_params, 'focus_factor': 1.0},
        'timed_rrt': {**base_params, 'robot_speed': args.robot_speed},
        'dijkstra': {'resolution': 1.0, 'diagonal_movement': True}
    }

    return params.get(algorithm, {})


def create_planner(algorithm: str, start: tuple, goal: tuple, env: Environment, args):
    """创建路径规划器"""
    common_params = {
        'start': start,
        'goal': goal,
        'env': env,
    }

    planners = {
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
    return planner_class(**common_params, **algorithm_params)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='停车场路径规划仿真')

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['rrt', 'rrt_star', 'informed_rrt', 'timed_rrt', 'dijkstra'],
        default='timed_rrt',
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
    env, start, goal, goal_orientation = create_parking_scenario(
        use_random_scene=args.random_scene
    )

    # 创建规划器
    planner = create_planner(args.algorithm, start, goal, env, args)

    # 规划路径（包含重试机制）
    print(f"\n使用 {args.algorithm} 算法进行路径规划...")
    path = try_plan_path(planner)

    if path:
        print("\n规划成功！")
        print(f"起点：{start}")
        print(f"目标停车位置：{goal}")
        print(f"停车方向：{goal_orientation:.2f}弧度")
        print(f"路径点数量：{len(path)}")

        # 使用Pygame进行仿真
        simulator = PygameSimulator()
        simulator.set_environment(env)
        simulator.execute_path(path)
    else:
        print("\n路径规划失败。")
        print("建议：")
        print("1. 增加迭代次数")
        print("2. 调整步长")
        print("3. 尝试其他算法")
        print(f"当前算法：{args.algorithm}")
        print("当前参数：")
        print(f"  - 迭代次数：{args.iterations}")
        print(f"  - 步长：{args.step_size}")
        if args.algorithm == 'timed_rrt':
            print(f"  - 机器人速度：{args.robot_speed}")
        elif args.algorithm == 'neural_rrt':
            print(f"  - 模型路径：{args.model_path}")


if __name__ == "__main__":
    main()
