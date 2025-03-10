#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pygame城市环境仿真器

用于可视化动态障碍物环境中的RRT路径规划，并实时显示移动障碍物和交通信号灯。
"""

import pygame
import numpy as np
import time
from typing import Tuple, Optional

from simulation.dynamic_environment import (
    DynamicEnvironment,
    TrafficLightState,
    Pedestrian,
    Vehicle,
    MovementPattern
)
from rrt.rrt_star import RRTStar


class PygameUrbanSimulator:
    """使用Pygame实现的城市环境仿真器"""

    # 颜色定义
    COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'grey': (200, 200, 200),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'building': (150, 150, 150),
        'road': (50, 50, 50),
        'sidewalk': (200, 200, 150),
        'car': (255, 50, 50),
        'pedestrian': (50, 50, 255),
        'path': (0, 200, 0),
        'background': (230, 230, 230),
        'target': (0, 255, 0)
    }

    def __init__(
        self,
        env: DynamicEnvironment,
        width: int = 800,
        height: int = 800,
        fps: int = 30,
        title: str = "城市环境中的RRT*路径规划"
    ):
        """
        初始化Pygame仿真器

        参数:
            env: 动态环境对象
            width: 窗口宽度（像素）
            height: 窗口高度（像素）
            fps: 帧率
            title: 窗口标题
        """
        # 初始化Pygame
        pygame.init()
        pygame.display.set_caption(title)

        self.env = env
        self.screen_width = width
        self.screen_height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.running = False

        # 计算缩放因子
        self.scale_x = width / env.width
        self.scale_y = height / env.height

        # 路径规划相关
        self.path = []
        self.rrt_nodes = []
        self.start_point = None
        self.goal_point = None
        self.respect_red_light = True  # 是否遵守红灯
        self.follow_target = False     # 是否跟随移动目标
        self.replan_interval = 2.0     # 重新规划路径的时间间隔（秒）
        self.last_replan_time = 0.0    # 上次重新规划路径的时间

        # 字体
        pygame.font.init()
        self._init_fonts()

        # 控制参数
        self.show_rrt_tree = True
        self.show_path = True
        self.show_info = True
        self.pause = False
        self.step_mode = False
        self.planning = False
        self.replan = False
        self.auto_replan = False       # 是否自动重新规划路径

        # 性能统计
        self.fps_history = []
        self.last_update_time = time.time()
        self.planning_time = 0

        # 消息显示
        self.message = ""
        self.message_time = 0
        self.message_duration = 3.0    # 消息显示时间（秒）

    def _init_fonts(self):
        """初始化字体"""
        # 尝试加载中文字体
        chinese_fonts = [
            'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi',
            'STHeiti', 'STKaiti', 'STSong', 'STFangsong', 'STXihei', 'STZhongsong',
            'FZShuTi', 'FZYaoti', 'YouYuan', 'LiSu', 'STXingkai', 'STXinwei'
        ]

        # 尝试系统字体
        font_found = False
        for font_name in chinese_fonts:
            try:
                self.font = pygame.font.SysFont(font_name, 14)
                self.big_font = pygame.font.SysFont(font_name, 20, bold=True)
                print(f"使用字体: {font_name}")
                font_found = True
                break
            except:
                continue

        # 如果系统字体都不可用，尝试加载字体文件
        if not font_found:
            try:
                # 尝试加载内置字体
                font_path = pygame.font.match_font('freesansbold')
                if font_path:
                    self.font = pygame.font.Font(font_path, 14)
                    self.big_font = pygame.font.Font(font_path, 20)
                    print(f"使用内置字体: {font_path}")
                else:
                    # 使用默认字体
                    self.font = pygame.font.Font(None, 14)
                    self.big_font = pygame.font.Font(None, 20)
                    print("使用默认字体")
            except Exception as e:
                print(f"字体加载失败: {e}")
                # 最后的备选方案
                self.font = pygame.font.Font(None, 14)
                self.big_font = pygame.font.Font(None, 20)
                print("使用默认字体")

    def to_screen_coords(self, x: float, y: float) -> Tuple[int, int]:
        """将环境坐标转换为屏幕坐标"""
        return (int(x * self.scale_x), self.screen_height - int(y * self.scale_y))

    def from_screen_coords(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """将屏幕坐标转换为环境坐标"""
        return (screen_x / self.scale_x, (self.screen_height - screen_y) / self.scale_y)

    def _draw_circle(
        self,
        x: float,
        y: float,
        radius: float,
        color,
        filled: bool = True,
        width: int = 0
    ):
        """绘制圆形"""
        pos = self.to_screen_coords(x, y)
        scaled_radius = int(radius * self.scale_x)  # 假设x和y的缩放比例一致
        if filled:
            pygame.draw.circle(self.screen, color, pos, scaled_radius)
        else:
            pygame.draw.circle(self.screen, color, pos, scaled_radius, width)

    def _draw_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color,
        filled: bool = True,
        line_width: int = 0,
        angle: float = 0.0
    ):
        """绘制矩形"""
        # 计算缩放后的宽度和高度
        scaled_width = int(width * self.scale_x)
        scaled_height = int(height * self.scale_y)

        # 计算左上角坐标
        left_top = self.to_screen_coords(
            x - width/2,
            y + height/2
        )

        # 创建矩形
        rect = pygame.Rect(left_top[0], left_top[1],
                           scaled_width, scaled_height)

        if angle == 0.0:
            # 没有旋转
            if filled:
                pygame.draw.rect(self.screen, color, rect)
            else:
                pygame.draw.rect(self.screen, color, rect, line_width)
        else:
            # 有旋转，需要创建一个Surface并旋转
            shape_surf = pygame.Surface(
                (scaled_width, scaled_height), pygame.SRCALPHA)
            if filled:
                pygame.draw.rect(shape_surf, color, pygame.Rect(
                    0, 0, scaled_width, scaled_height))
            else:
                pygame.draw.rect(shape_surf, color, pygame.Rect(
                    0, 0, scaled_width, scaled_height), line_width)

            # 旋转Surface
            rotated_surf = pygame.transform.rotate(
                shape_surf, angle * 180 / np.pi)

            # 计算旋转后的中心位置
            center = self.to_screen_coords(x, y)
            rot_rect = rotated_surf.get_rect(center=center)

            # 绘制到屏幕
            self.screen.blit(rotated_surf, rot_rect)

    def _draw_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color,
        width: int = 1
    ):
        """绘制线段"""
        start_pos = self.to_screen_coords(x1, y1)
        end_pos = self.to_screen_coords(x2, y2)
        pygame.draw.line(self.screen, color, start_pos, end_pos, width)

    def draw_environment(self):
        """绘制环境"""
        # 填充背景
        self.screen.fill(self.COLORS['background'])

        # 绘制静态障碍物（建筑物等）
        for obstacle in self.env.obstacles:
            if hasattr(obstacle, 'type'):
                if obstacle.type == 'circle':
                    self._draw_circle(
                        obstacle.x,
                        obstacle.y,
                        obstacle.radius,
                        self.COLORS['building']
                    )
                else:  # 矩形
                    self._draw_rectangle(
                        obstacle.x,
                        obstacle.y,
                        obstacle.width,
                        obstacle.height,
                        self.COLORS['building'],
                        angle=getattr(obstacle, 'angle', 0.0)
                    )

        # 绘制交通信号灯
        for light in self.env.traffic_lights:
            color = self.COLORS['red']
            if light.state == TrafficLightState.GREEN:
                color = self.COLORS['green']
            elif light.state == TrafficLightState.YELLOW:
                color = self.COLORS['yellow']

            self._draw_circle(
                light.x,
                light.y,
                light.radius,
                color
            )

            # 绘制信号灯的影响区域
            if light.state == TrafficLightState.RED and self.respect_red_light:
                self._draw_circle(
                    light.x,
                    light.y,
                    10.0,  # 影响半径
                    color,
                    filled=False,
                    width=1
                )

        # 绘制移动障碍物
        for obstacle in self.env.moving_obstacles:
            # 检查是否是移动目标
            if hasattr(obstacle, 'is_target') and obstacle.is_target:
                # 绘制移动目标
                self._draw_circle(
                    obstacle.x,
                    obstacle.y,
                    obstacle.width / 2,  # 使用宽度的一半作为半径
                    getattr(obstacle, 'color', self.COLORS['target']),
                    filled=True
                )
                # 绘制目标标记
                self._draw_circle(
                    obstacle.x,
                    obstacle.y,
                    obstacle.width / 2 + 1,
                    self.COLORS['white'],
                    filled=False,
                    width=2
                )
                continue

            if isinstance(obstacle, Pedestrian):
                # 绘制行人
                self._draw_rectangle(
                    obstacle.x,
                    obstacle.y,
                    obstacle.width,
                    obstacle.height,
                    self.COLORS['pedestrian'],
                    angle=obstacle.direction
                )
            elif isinstance(obstacle, Vehicle):
                # 绘制车辆
                self._draw_rectangle(
                    obstacle.x,
                    obstacle.y,
                    obstacle.width,
                    obstacle.height,
                    self.COLORS['car'],
                    angle=obstacle.direction
                )
            else:
                # 处理其他类型的移动障碍物
                if hasattr(obstacle, 'width') and hasattr(obstacle, 'height'):
                    # 这是矩形障碍物
                    self._draw_rectangle(
                        obstacle.x,
                        obstacle.y,
                        obstacle.width,
                        obstacle.height,
                        self.COLORS['blue'],
                        angle=getattr(obstacle, 'direction', 0.0)
                    )
                elif hasattr(obstacle, 'radius'):
                    # 这是圆形障碍物
                    self._draw_circle(
                        obstacle.x,
                        obstacle.y,
                        obstacle.radius,
                        self.COLORS['blue']
                    )

        # 绘制RRT树
        if self.show_rrt_tree and self.rrt_nodes:
            for node in self.rrt_nodes:
                if node.parent:
                    self._draw_line(
                        node.x,
                        node.y,
                        node.parent.x,
                        node.parent.y,
                        self.COLORS['cyan'],
                        1
                    )

        # 绘制路径
        if self.show_path and self.path:
            for i in range(len(self.path) - 1):
                self._draw_line(
                    self.path[i][0],
                    self.path[i][1],
                    self.path[i+1][0],
                    self.path[i+1][1],
                    self.COLORS['path'],
                    3
                )

        # 绘制起点和终点
        if self.start_point:
            self._draw_circle(
                self.start_point[0],
                self.start_point[1],
                1.5,
                self.COLORS['green']
            )

        if self.goal_point:
            self._draw_circle(
                self.goal_point[0],
                self.goal_point[1],
                1.5,
                self.COLORS['red']
            )

    def draw_info_panel(self):
        """绘制信息面板"""
        if not self.show_info:
            return

        # 计算平均FPS
        if len(self.fps_history) > 0:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
        else:
            avg_fps = 0

        # 信息文本
        info_texts = [
            f"帧率: {avg_fps:.1f}",
            f"物理时间: {time.time() - self.last_update_time:.1f}秒",
            f"静态障碍物: {len(self.env.obstacles)}",
            f"移动障碍物: {len(self.env.moving_obstacles)}",
            f"信号灯: {len(self.env.traffic_lights)}",
            f"路径长度: {len(self.path)}",
            f"规划时间: {self.planning_time:.3f}秒",
            f"遵守红灯: {'是' if self.respect_red_light else '否'}",
            f"跟随目标: {'是' if self.follow_target else '否'}",
            f"自动重规划: {'是' if self.auto_replan else '否'}"
        ]

        # 控制提示
        control_texts = [
            "ESC键: 退出",
            "空格键: 暂停/继续",
            "R键: 重新规划",
            "T键: 显示/隐藏树",
            "P键: 显示/隐藏路径",
            "I键: 显示/隐藏信息",
            "L键: 切换遵守红灯",
            "F键: 切换跟随目标",
            "A键: 切换自动重规划",
            "S键: 步进模式",
            "鼠标左键: 设置起点",
            "鼠标右键: 设置终点",
            "中键: 添加移动目标"
        ]

        # 绘制背景半透明矩形
        info_surface = pygame.Surface((250, 200), pygame.SRCALPHA)
        info_surface.fill((0, 0, 0, 128))
        self.screen.blit(info_surface, (10, 10))

        # 绘制信息文本
        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, self.COLORS['white'])
            self.screen.blit(text_surface, (20, 20 + i * 20))

        # 绘制控制提示
        control_surface = pygame.Surface((200, 280), pygame.SRCALPHA)
        control_surface.fill((0, 0, 0, 128))
        self.screen.blit(control_surface, (self.screen_width - 210, 10))

        # 绘制控制提示标题
        title_surface = self.big_font.render(
            "控制说明", True, self.COLORS['white'])
        self.screen.blit(title_surface, (self.screen_width - 200, 15))

        # 绘制控制提示文本
        for i, text in enumerate(control_texts):
            text_surface = self.font.render(text, True, self.COLORS['white'])
            self.screen.blit(
                text_surface, (self.screen_width - 200, 40 + i * 20))

        # 绘制状态信息
        status_text = "当前状态: "
        if self.pause:
            status_text += "已暂停"
        elif self.planning:
            status_text += "路径规划中..."
        else:
            status_text += "运行中"

        status_surface = self.big_font.render(
            status_text, True, self.COLORS['yellow'])
        self.screen.blit(status_surface, (self.screen_width // 2 - 60, 10))

        # 显示临时消息
        if self.message and time.time() - self.message_time < self.message_duration:
            message_surface = self.big_font.render(
                self.message, True, self.COLORS['yellow'])
            self.screen.blit(message_surface, (
                self.screen_width // 2 - message_surface.get_width() // 2,
                self.screen_height // 2 - message_surface.get_height() // 2
            ))

    def plan_path(self):
        """使用RRT*规划路径"""
        if not self.start_point or not self.goal_point:
            return

        # 如果跟随移动目标，更新目标点
        if self.follow_target and self.env.moving_target:
            self.goal_point = (self.env.moving_target.x,
                               self.env.moving_target.y)

        print(f"开始规划从 {self.start_point} 到 {self.goal_point} 的路径...")
        self.message = "正在规划路径..."
        self.message_time = time.time()

        # 标记为规划中
        self.planning = True

        # 创建RRT*规划器
        rrt_star = RRTStar(
            start=self.start_point,
            goal=self.goal_point,
            env=self.env,
            step_size=5.0,
            max_iterations=500,
            goal_sample_rate=0.1,
            search_radius=10.0,
            rewire_factor=1.5
        )

        # 规划路径
        start_time = time.time()
        path = rrt_star.plan()
        self.planning_time = time.time() - start_time

        # 保存路径和RRT树
        self.path = path
        self.rrt_nodes = rrt_star.node_list

        # 标记为非规划中
        self.planning = False

        # 输出结果
        if path:
            self.message = f"找到路径! 长度: {len(path)}"
            print(f"成功找到路径! 长度: {len(path)}, 耗时: {self.planning_time:.3f}s")
        else:
            self.message = "未找到路径!"
            print(f"未找到路径! 耗时: {self.planning_time:.3f}s")

        self.message_time = time.time()
        return path

    def handle_events(self):
        """处理Pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.pause = not self.pause
                    self.message = f"仿真{'暂停' if self.pause else '继续'}"
                    self.message_time = time.time()
                elif event.key == pygame.K_r:
                    self.replan = True
                elif event.key == pygame.K_t:
                    self.show_rrt_tree = not self.show_rrt_tree
                    self.message = f"{'显示' if self.show_rrt_tree else '隐藏'}RRT树"
                    self.message_time = time.time()
                elif event.key == pygame.K_p:
                    self.show_path = not self.show_path
                    self.message = f"{'显示' if self.show_path else '隐藏'}路径"
                    self.message_time = time.time()
                elif event.key == pygame.K_i:
                    self.show_info = not self.show_info
                    self.message = f"{'显示' if self.show_info else '隐藏'}信息面板"
                    self.message_time = time.time()
                elif event.key == pygame.K_l:
                    self.respect_red_light = not self.respect_red_light
                    self.message = f"{'遵守' if self.respect_red_light else '忽略'}红灯"
                    self.message_time = time.time()
                elif event.key == pygame.K_f:
                    self.follow_target = not self.follow_target
                    self.message = f"{'跟随' if self.follow_target else '不跟随'}移动目标"
                    self.message_time = time.time()
                    if self.follow_target and self.env.moving_target:
                        self.goal_point = (
                            self.env.moving_target.x, self.env.moving_target.y)
                        self.replan = True
                elif event.key == pygame.K_a:
                    self.auto_replan = not self.auto_replan
                    self.message = f"{'开启' if self.auto_replan else '关闭'}自动重规划"
                    self.message_time = time.time()
                elif event.key == pygame.K_s:
                    self.step_mode = not self.step_mode
                    if self.step_mode:
                        self.pause = True
                    self.message = f"{'开启' if self.step_mode else '关闭'}步进模式"
                    self.message_time = time.time()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 获取鼠标点击位置并转换为环境坐标
                pos = pygame.mouse.get_pos()
                env_pos = self.from_screen_coords(pos[0], pos[1])

                if event.button == 1:  # 左键
                    self.start_point = env_pos
                    print(f"设置起点: {env_pos}")
                    self.message = f"设置起点: ({env_pos[0]:.1f}, {env_pos[1]:.1f})"
                    self.message_time = time.time()
                    self.replan = True
                elif event.button == 3:  # 右键
                    self.goal_point = env_pos
                    print(f"设置终点: {env_pos}")
                    self.message = f"设置终点: ({env_pos[0]:.1f}, {env_pos[1]:.1f})"
                    self.message_time = time.time()
                    self.replan = True
                elif event.button == 2:  # 中键
                    # 添加移动目标
                    pattern_params = {
                        'center_x': env_pos[0],
                        'center_y': env_pos[1],
                        'radius': 20.0,
                        'angle': 0.0
                    }
                    self.env.add_moving_target(
                        x=env_pos[0],
                        y=env_pos[1],
                        speed=3.0,
                        pattern=MovementPattern.CIRCULAR,
                        pattern_params=pattern_params
                    )
                    print(f"添加移动目标: {env_pos}")
                    self.message = f"添加移动目标: ({env_pos[0]:.1f}, {env_pos[1]:.1f})"
                    self.message_time = time.time()

                    # 如果启用了跟随目标，更新目标点并重新规划
                    if self.follow_target:
                        self.goal_point = env_pos
                        self.replan = True

    def run(
        self,
        start_point: Optional[Tuple[float, float]] = None,
        goal_point: Optional[Tuple[float, float]] = None
    ):
        """
        运行仿真

        参数:
            start_point: 起点坐标
            goal_point: 终点坐标
        """
        self.start_point = start_point
        self.goal_point = goal_point
        self.running = True
        self.last_update_time = time.time()
        self.last_replan_time = time.time()

        # 如果指定了起点和终点，先规划一次路径
        if start_point and goal_point:
            self.plan_path()

        # 主循环
        while self.running:
            # 处理事件
            self.handle_events()

            # 如果需要重新规划
            if self.replan:
                self.plan_path()
                self.replan = False
                self.last_replan_time = time.time()

            # 更新环境
            current_time = time.time()
            if not self.pause:
                dt = current_time - self.last_update_time
                self.env.update(dt)
                self.last_update_time = current_time

                # 更新FPS历史
                self.fps_history.append(self.clock.get_fps())
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)

                # 如果启用了跟随目标和自动重规划，检查是否需要重新规划
                if self.follow_target and self.auto_replan and self.env.moving_target:
                    # 如果目标移动了足够远或者时间间隔足够长，重新规划
                    if current_time - self.last_replan_time >= self.replan_interval:
                        target_pos = (self.env.moving_target.x,
                                      self.env.moving_target.y)
                        if self.goal_point != target_pos:
                            self.goal_point = target_pos
                            self.plan_path()
                            self.last_replan_time = current_time

            # 绘制环境
            self.draw_environment()

            # 绘制信息面板
            self.draw_info_panel()

            # 更新屏幕
            pygame.display.flip()

            # 控制帧率
            self.clock.tick(self.fps)

        # 清理
        pygame.quit()


def main():
    """主函数"""
    # 创建动态环境
    env = DynamicEnvironment(width=100.0, height=100.0)

    # 生成城市环境
    print("创建城市环境...")
    env.create_urban_environment()

    # 创建仿真器
    simulator = PygameUrbanSimulator(env, width=1024, height=768)

    # 设置起点和终点
    start_point = (10.0, 10.0)
    goal_point = (90.0, 90.0)

    # 运行仿真
    simulator.run(start_point, goal_point)


if __name__ == "__main__":
    main()
