import argparse
import math
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygame
import yaml

from rrt.astar import AStar
from rrt.attention_dqn_rrt import AttentionDQNRRT
from rrt.dijkstra import Dijkstra
from rrt.dstar_lite import DStarLite
from rrt.informed_rrt import InformedRRTStar
from rrt.path_smoothing import PathSmoother
from rrt.planner_factory import create_planner as build_planner
from rrt.planner_factory import get_algorithm_specific_params as get_factory_algorithm_specific_params
from rrt.rrt_base import RRT
from rrt.rrt_star import RRTStar, TimedRRTStar
from rrt.theta_star import ThetaStar
from simulation.environment import Environment
from simulation.pygame_simulator import (
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    ParkingEnvironment,
    PygameSimulator,
    VehicleModel,
    check_path_collision,
    check_vehicle_collision,
    get_font,
    get_font_resolution,
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件，并补齐兼容字段。"""
    default_config: Dict[str, Any] = {
        "window": {
            "width": 1280,
            "height": 677,
            "title": "停车场路径规划仿真器",
        },
        "simulation": {
            "scale": 10.0,
            "fps": 60,
            "dt": 0.05,
            "lookahead": 5.0,
            "speed_multiplier": 2.0,
        },
        "ui": {
            "fonts": {
                "ui": {
                    "system": [
                        "Noto Sans CJK SC",
                        "WenQuanYi Micro Hei",
                        "Source Han Sans SC",
                        "Droid Sans Fallback",
                        "Microsoft YaHei UI",
                        "Microsoft YaHei",
                        "SimHei",
                        "DengXian",
                        "SimSun",
                    ],
                    "files": ["NotoSansCJKsc-Regular.otf"],
                },
                "title": {
                    "system": [
                        "Noto Sans CJK SC",
                        "Source Han Sans SC",
                        "WenQuanYi Micro Hei",
                        "Microsoft YaHei UI",
                        "Microsoft YaHei",
                        "DengXian",
                    ],
                    "files": ["STKAITI.TTF", "NotoSansCJKsc-Regular.otf"],
                },
                "mono": {
                    "system": [
                        "Sarasa Mono SC",
                        "WenQuanYi Zen Hei Mono",
                        "Noto Sans Mono CJK SC",
                        "Noto Sans CJK SC",
                        "Microsoft YaHei UI",
                        "Consolas",
                    ],
                    "files": ["NotoSansCJKsc-Regular.otf"],
                },
            }
        },
        "vehicle": {
            "length": 4.5,
            "width": 1.8,
            "wheelbase": 2.7,
            "dynamics": {
                "max_speed": 5.0,
                "max_reverse_speed": 2.5,
                "max_accel": 2.0,
                "max_brake": 4.0,
                "max_steer": 0.7854,
                "steer_rate": math.pi,
                "rolling_resistance": 0.2,
                "drag_coefficient": 0.015,
                "creep_speed": 0.35,
                "creep_accel": 0.45,
                "jerk_limit": 6.0,
                "throttle_response": 2.4,
                "brake_response": 4.0,
                "steer_speed_sensitivity": 0.08,
            },
            "render": {
                "body_color": [46, 160, 109, 255],
                "roof_color": [73, 178, 128, 245],
                "window_color": [200, 225, 234, 220],
                "window_shadow": [93, 121, 138, 185],
                "trim_color": [25, 34, 42, 255],
                "wheel_color": [32, 35, 38, 255],
                "wheel_hub_color": [184, 191, 198, 255],
                "shadow_color": [18, 22, 24, 70],
                "headlight_color": [255, 244, 196, 230],
                "taillight_color": [220, 63, 52, 230],
                "brake_color": [255, 76, 76, 255],
                "reverse_color": [146, 228, 255, 255],
                "outline_color": [15, 26, 31, 255],
            },
        },
        "parking": {
            "final_pose": {
                "enabled": True,
                "trigger_distance": 3.2,
                "position_tolerance": 0.25,
                "heading_tolerance_deg": 5.0,
                "stop_speed_tolerance": 0.05,
                "hold_time": 0.35,
                "max_duration": 18.0,
                "max_stagnation_time": 4.0,
                "max_gear_switches": 2,
                "minimum_clearance": 0.2,
                "staging_offset": 1.15,
                "align_heading_deg": 12.0,
                "max_stage_speed": 1.0,
                "max_dock_speed": 0.75,
                "max_adjust_speed": 0.45,
                "path_tail_samples": 6,
            }
        },
        "parking_lot": {
            "geometry": {
                "spot_width": 2.5,
                "spot_length": 5.0,
                "lane_width": 8.0,
            },
            "margin": 5.0,
            "wall_thickness": 0.5,
            "entrance_width": 12.0,
            "entrance_margin": 15.0,
            "layout": [],
            "colors": {
                "wall": [80, 80, 80, 255],
                "parking_spot": [220, 220, 220, 80],
                "parking_spot_border": [234, 234, 228, 220],
                "parking_spot_line": [255, 255, 255, 220],
                "no_parking": [215, 98, 70, 110],
                "car_body": [60, 67, 72, 235],
                "safety_zone": [120, 196, 154, 45],
                "target_spot_glow": [76, 161, 255, 70],
                "target_spot_border": [76, 161, 255, 255],
                "target_pose": [255, 214, 102, 220],
                "panel_bg": [250, 248, 243, 235],
                "panel_border": [210, 204, 194, 255],
                "panel_title": [18, 27, 34, 255],
                "text_muted": [88, 96, 103, 255],
                "success": [46, 160, 109, 255],
                "warning": [214, 146, 52, 255],
                "danger": [194, 76, 76, 255],
            },
        },
        "path_planning": {
            "default_algorithm": "rrt_star",
        },
        "control": {
            "default_method": "pid",
        },
    }

    if config_path:
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                user_config = yaml.safe_load(handle) or {}
            update_config(default_config, user_config)
            print(f"已加载配置文件: {config_path}")
        except Exception as exc:
            print(f"加载配置文件失败: {exc}")
            print("使用默认配置")

    _normalize_config(default_config)
    return default_config


def update_config(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
    """递归更新配置字典。"""
    for key, value in user_config.items():
        if (
            isinstance(value, dict)
            and key in default_config
            and isinstance(default_config[key], dict)
        ):
            update_config(default_config[key], value)
        else:
            default_config[key] = value


def _normalize_config(config: Dict[str, Any]) -> None:
    """兼容旧键名，并补齐默认结构。"""
    simulation = config.setdefault("simulation", {})
    if "speed_multiplier" not in simulation:
        simulation["speed_multiplier"] = simulation.get("simulation_speed", 2.0)

    ui = config.setdefault("ui", {})
    fonts = ui.setdefault("fonts", {})
    for role, bundled_default in (
        ("ui", ["NotoSansCJKsc-Regular.otf"]),
        ("title", ["STKAITI.TTF", "NotoSansCJKsc-Regular.otf"]),
        ("mono", ["NotoSansCJKsc-Regular.otf"]),
    ):
        role_config = fonts.setdefault(role, {})
        if isinstance(role_config, (list, tuple, str)):
            role_config = {"preferred": role_config, "bundled": bundled_default}
            fonts[role] = role_config
        role_config.setdefault("preferred", [])
        role_config.setdefault("bundled", list(bundled_default))
        if role_config.get("system") and not role_config.get("preferred"):
            role_config["preferred"] = list(role_config.get("system", []))
        if role_config.get("files") and not role_config.get("bundled"):
            role_config["bundled"] = list(role_config.get("files", []))
        role_config.setdefault("system", list(role_config.get("preferred", [])))
        role_config.setdefault("files", list(role_config.get("bundled", [])))

    vehicle = config.setdefault("vehicle", {})
    dynamics = vehicle.setdefault("dynamics", {})
    render = vehicle.setdefault("render", {})
    vehicle["wheelbase"] = vehicle.get("wheelbase", vehicle.get("wheel_base", 2.7))

    for field in (
        "max_speed",
        "max_reverse_speed",
        "max_accel",
        "max_brake",
        "max_decel",
        "max_steer",
        "max_steer_angle",
        "steer_rate",
        "rolling_resistance",
        "drag_coefficient",
        "creep_speed",
        "creep_accel",
        "jerk_limit",
        "throttle_response",
        "brake_response",
        "steer_speed_sensitivity",
    ):
        if field in vehicle and field not in dynamics:
            dynamics[field] = vehicle[field]

    if "max_brake" not in dynamics and "max_decel" in dynamics:
        dynamics["max_brake"] = dynamics["max_decel"]
    if "max_steer" not in dynamics and "max_steer_angle" in dynamics:
        dynamics["max_steer"] = dynamics["max_steer_angle"]

    parking = config.setdefault("parking", {})
    final_pose = parking.setdefault("final_pose", {})
    final_pose.setdefault("enabled", True)
    final_pose.setdefault("trigger_distance", 3.2)
    final_pose.setdefault("position_tolerance", 0.25)
    final_pose.setdefault("heading_tolerance_deg", 5.0)
    final_pose.setdefault("stop_speed_tolerance", 0.05)
    final_pose.setdefault("hold_time", 0.35)
    final_pose.setdefault("max_duration", 18.0)
    final_pose.setdefault("max_stagnation_time", 4.0)
    final_pose.setdefault("max_gear_switches", 2)
    final_pose.setdefault("minimum_clearance", 0.2)
    final_pose.setdefault("staging_offset", 1.15)
    final_pose.setdefault("align_heading_deg", 12.0)
    final_pose.setdefault("max_stage_speed", 1.0)
    final_pose.setdefault("max_dock_speed", 0.75)
    final_pose.setdefault("max_adjust_speed", 0.45)
    final_pose.setdefault("path_tail_samples", 6)

    render_defaults = {
        "body_color": [46, 160, 109, 255],
        "roof_color": [73, 178, 128, 245],
        "window_color": [200, 225, 234, 220],
        "window_shadow": [93, 121, 138, 185],
        "trim_color": [25, 34, 42, 255],
        "wheel_color": [32, 35, 38, 255],
        "wheel_hub_color": [184, 191, 198, 255],
        "shadow_color": [18, 22, 24, 70],
        "headlight_color": [255, 244, 196, 230],
        "taillight_color": [220, 63, 52, 230],
        "brake_color": [255, 76, 76, 255],
        "reverse_color": [146, 228, 255, 255],
        "outline_color": [15, 26, 31, 255],
    }
    for key, value in render_defaults.items():
        render.setdefault(key, value)

    parking_lot = config.setdefault("parking_lot", {})
    parking_lot.setdefault("geometry", {})
    parking_lot.setdefault("colors", {})


def build_simulator_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """把嵌套配置拍平成 PygameSimulator 需要的结构。"""
    simulation = config.get("simulation", {})
    vehicle = config.get("vehicle", {})
    dynamics = vehicle.get("dynamics", {})
    render = vehicle.get("render", {})

    return {
        "window_width": config.get("window", {}).get("width", 1280),
        "window_height": config.get("window", {}).get("height", 677),
        "window_title": config.get("window", {}).get("title", "停车场路径规划仿真器"),
        "scale": simulation.get("scale", 10.0),
        "fps": simulation.get("fps", 60),
        "dt": simulation.get("dt", 0.05),
        "lookahead": simulation.get("lookahead", 5.0),
        "speed_multiplier": simulation.get(
            "speed_multiplier",
            simulation.get("simulation_speed", 2.0),
        ),
        "control_method": config.get("control", {}).get("default_method", "pid"),
        "ui": config.get("ui", {}),
        "parking": config.get("parking", {}),
        "vehicle": {
            "length": vehicle.get("length", 4.5),
            "width": vehicle.get("width", 1.8),
            "wheelbase": vehicle.get("wheelbase", vehicle.get("wheel_base", 2.7)),
            "max_speed": dynamics.get("max_speed", vehicle.get("max_speed", 5.0)),
            "max_reverse_speed": dynamics.get(
                "max_reverse_speed",
                vehicle.get("max_reverse_speed", 2.5),
            ),
            "max_accel": dynamics.get("max_accel", vehicle.get("max_accel", 2.0)),
            "max_brake": dynamics.get(
                "max_brake",
                dynamics.get("max_decel", vehicle.get("max_brake", 4.0)),
            ),
            "max_steer": dynamics.get(
                "max_steer",
                dynamics.get("max_steer_angle", vehicle.get("max_steer", 0.7854)),
            ),
            "dynamics": {
                "max_speed": dynamics.get("max_speed", vehicle.get("max_speed", 5.0)),
                "max_reverse_speed": dynamics.get(
                    "max_reverse_speed",
                    vehicle.get("max_reverse_speed", 2.5),
                ),
                "max_accel": dynamics.get("max_accel", vehicle.get("max_accel", 2.0)),
                "max_brake": dynamics.get(
                    "max_brake",
                    dynamics.get("max_decel", vehicle.get("max_brake", 4.0)),
                ),
                "max_steer": dynamics.get(
                    "max_steer",
                    dynamics.get("max_steer_angle", vehicle.get("max_steer", 0.7854)),
                ),
                "steer_rate": dynamics.get("steer_rate", math.pi),
                "rolling_resistance": dynamics.get("rolling_resistance", 0.2),
                "drag_coefficient": dynamics.get("drag_coefficient", 0.015),
                "creep_speed": dynamics.get("creep_speed", 0.35),
                "creep_accel": dynamics.get("creep_accel", 0.45),
                "jerk_limit": dynamics.get("jerk_limit", 6.0),
                "throttle_response": dynamics.get("throttle_response", 2.4),
                "brake_response": dynamics.get("brake_response", 4.0),
                "steer_speed_sensitivity": dynamics.get("steer_speed_sensitivity", 0.08),
            },
            "render": dict(render),
        },
    }


def _color_tuple(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)) and 3 <= len(value) <= 4:
        return tuple(int(component) for component in value)
    return default


def _scene_colors(config: Dict[str, Any]) -> Dict[str, Tuple[int, ...]]:
    colors = config.get("parking_lot", {}).get("colors", {})
    spot_border = _color_tuple(colors.get("parking_spot_border"), (234, 234, 228, 220))
    return {
        "wall": _color_tuple(colors.get("wall"), (80, 80, 80, 255)),
        "parking_spot": _color_tuple(colors.get("parking_spot"), (220, 220, 220, 80)),
        "parking_spot_border": spot_border,
        "parking_spot_line": _color_tuple(colors.get("parking_spot_line"), (255, 255, 255, 220)),
        "occupied_fill": _color_tuple(colors.get("car_body"), (60, 67, 72, 235)),
        "occupied_border": (214, 104, 91, 235),
        "no_parking": _color_tuple(colors.get("no_parking"), (215, 98, 70, 110)),
        "safety_zone": _color_tuple(colors.get("safety_zone"), (120, 196, 154, 45)),
        "occupied_safety": (214, 104, 91, 45),
        "asphalt": (75, 82, 86),
        "lane_line": (233, 231, 224),
        "target_spot_glow": _color_tuple(colors.get("target_spot_glow"), (76, 161, 255, 70)),
        "target_spot_border": _color_tuple(colors.get("target_spot_border"), (76, 161, 255, 255)),
        "target_pose": _color_tuple(colors.get("target_pose"), (255, 214, 102, 220)),
        "panel_bg": _color_tuple(colors.get("panel_bg"), (250, 248, 243, 235)),
        "panel_border": _color_tuple(colors.get("panel_border"), (210, 204, 194, 255)),
        "panel_title": _color_tuple(colors.get("panel_title"), (18, 27, 34, 255)),
        "text_muted": _color_tuple(colors.get("text_muted"), (88, 96, 103, 255)),
        "success": _color_tuple(colors.get("success"), (46, 160, 109, 255)),
        "warning": _color_tuple(colors.get("warning"), (214, 146, 52, 255)),
        "danger": _color_tuple(colors.get("danger"), (194, 76, 76, 255)),
    }


def _apply_spot_state(
    spot_obstacle: Any,
    occupied: bool,
    colors: Dict[str, Tuple[int, ...]],
) -> None:
    """同步停车位占用状态与显示属性。"""
    spot_obstacle.occupied = occupied
    spot_obstacle.is_filled = occupied
    spot_obstacle.color = colors["occupied_fill"] if occupied else colors["parking_spot"]
    spot_obstacle.line_width = 2

    safety_obstacle = getattr(spot_obstacle, "safety_obstacle", None)
    if safety_obstacle is not None:
        safety_obstacle.occupied = occupied
        safety_obstacle.color = (
            colors["occupied_safety"] if occupied else colors["safety_zone"]
        )

    spot_meta = getattr(spot_obstacle, "spot_meta", None)
    if isinstance(spot_meta, dict):
        spot_meta["occupied"] = occupied


def _add_wall(
    env: ParkingEnvironment,
    x: float,
    y: float,
    width: float,
    height: float,
    color: Tuple[int, ...],
) -> None:
    obstacle = env.add_obstacle(
        x=x,
        y=y,
        obstacle_type="rectangle",
        width=width,
        height=height,
        angle=0,
        color=color,
        is_filled=True,
        line_width=1,
    )
    obstacle.scene_role = "wall"


def _add_boundary_walls(
    width: float,
    height: float,
    config: Dict[str, Any],
    env: ParkingEnvironment,
    colors: Dict[str, Tuple[int, ...]],
) -> None:
    parking_config = config.get("parking_lot", {})
    entrance_width = float(parking_config.get("entrance_width", 12.0))
    wall_thickness = float(parking_config.get("wall_thickness", 0.5))

    top_y = wall_thickness / 2
    bottom_y = height - wall_thickness / 2
    left_x = wall_thickness / 2
    right_x = width - wall_thickness / 2
    half_bottom = max((width - entrance_width) / 2, wall_thickness)

    _add_wall(env, width / 2, top_y, width, wall_thickness, colors["wall"])
    _add_wall(env, half_bottom / 2, bottom_y, half_bottom, wall_thickness, colors["wall"])
    _add_wall(
        env,
        width - half_bottom / 2,
        bottom_y,
        half_bottom,
        wall_thickness,
        colors["wall"],
    )
    _add_wall(env, left_x, height / 2, wall_thickness, height, colors["wall"])
    _add_wall(env, right_x, height / 2, wall_thickness, height, colors["wall"])


def _register_parking_spot(
    env: ParkingEnvironment,
    obstacle: Any,
    spot_id: Any,
    orientation: float,
    occupied: bool,
    slot_type: str = "perpendicular",
) -> None:
    if not hasattr(env, "parking_spots_metadata"):
        env.parking_spots_metadata = []

    label = f"{int(spot_id):02d}" if isinstance(spot_id, (int, float)) else str(spot_id)
    meta = {
        "id": spot_id,
        "label": label,
        "position": (obstacle.x, obstacle.y),
        "orientation": float(orientation),
        "occupied": occupied,
        "slot_type": slot_type or "perpendicular",
        "obstacle": obstacle,
    }
    obstacle.spot_id = spot_id
    obstacle.spot_meta = meta
    env.parking_spots_metadata.append(meta)


def _build_scene_from_layout(
    width: float,
    height: float,
    config: Dict[str, Any],
    env: ParkingEnvironment,
    colors: Dict[str, Tuple[int, ...]],
) -> None:
    parking_config = config.get("parking_lot", {})
    geometry = parking_config.get("geometry", {})
    default_spot_width = float(
        parking_config.get("spot_width", geometry.get("spot_width", 2.5))
    )
    default_spot_length = float(
        parking_config.get("spot_height", geometry.get("spot_length", 5.0))
    )

    layout = parking_config.get("layout", [])
    next_spot_id = 1

    for item in layout:
        item_type = item.get("type", "parking_spot")
        position = item.get("position", [0.0, 0.0])
        x = float(position[0])
        y = float(position[1])

        if item_type == "parking_spot":
            occupied = bool(item.get("occupied", item.get("static", False)))
            orientation = float(item.get("orientation", 0.0))
            geometry_angle = float(item.get("geometry_angle", item.get("angle", 0.0)))
            spot_width = float(item.get("width", default_spot_width))
            spot_length = float(item.get("height", default_spot_length))

            spot = env.add_obstacle(
                x=x,
                y=y,
                obstacle_type="rectangle",
                width=spot_width,
                height=spot_length,
                angle=geometry_angle,
                color=colors["parking_spot"],
                is_parking_spot=True,
                occupied=occupied,
                is_filled=occupied,
                line_width=2,
            )
            spot.scene_role = "parking_spot"
            _register_parking_spot(
                env,
                spot,
                item.get("id", next_spot_id),
                orientation,
                occupied,
                item.get("slot_type", "perpendicular"),
            )
            _apply_spot_state(spot, occupied, colors)
            next_spot_id += 1
            continue

        if item_type == "no_parking":
            obstacle = env.add_obstacle(
                x=x,
                y=y,
                obstacle_type="rectangle",
                width=float(item.get("width", 5.0)),
                height=float(item.get("height", 3.0)),
                angle=float(item.get("angle", 0.0)),
                color=colors["no_parking"],
                is_filled=True,
                line_width=1,
            )
            obstacle.scene_role = "no_parking"
            continue

        if item_type == "circle":
            obstacle = env.add_obstacle(
                x=x,
                y=y,
                obstacle_type="circle",
                radius=float(item.get("radius", 1.0)),
                color=_color_tuple(item.get("color"), (120, 120, 120, 220)),
                is_filled=True,
                line_width=1,
            )
            obstacle.scene_role = "obstacle"
            continue

        obstacle = env.add_obstacle(
            x=x,
            y=y,
            obstacle_type="rectangle",
            width=float(item.get("width", 2.0)),
            height=float(item.get("height", 2.0)),
            angle=float(item.get("angle", 0.0)),
            color=_color_tuple(item.get("color"), (120, 120, 120, 220)),
            is_filled=True,
            line_width=1,
        )
        obstacle.scene_role = "obstacle"


def _build_fallback_scene(
    width: float,
    height: float,
    config: Dict[str, Any],
    env: ParkingEnvironment,
    colors: Dict[str, Tuple[int, ...]],
) -> None:
    parking_config = config.get("parking_lot", {})
    geometry = parking_config.get("geometry", {})
    spot_width = float(geometry.get("spot_width", 2.5))
    spot_length = float(geometry.get("spot_length", 5.0))

    spot_id = 1
    left_x = spot_width * 1.6
    right_x = width - spot_width * 1.6
    for index in range(12):
        y_pos = 6.0 + index * (spot_length + 0.6)
        for x_pos, occupied, orientation in (
            (left_x, index % 2 == 0, 270.0),
            (right_x, index % 2 == 1, 90.0),
        ):
            spot = env.add_obstacle(
                x=x_pos,
                y=y_pos,
                obstacle_type="rectangle",
                width=spot_width,
                height=spot_length,
                angle=0.0,
                color=colors["parking_spot"],
                is_parking_spot=True,
                occupied=occupied,
                is_filled=occupied,
                line_width=2,
            )
            spot.scene_role = "parking_spot"
            _register_parking_spot(env, spot, spot_id, orientation, occupied)
            _apply_spot_state(spot, occupied, colors)
            spot_id += 1

    row_y = height * 0.54
    for index in range(10):
        x_pos = 16.0 + index * (spot_length + 0.7)
        occupied = index % 3 == 0
        spot = env.add_obstacle(
            x=x_pos,
            y=row_y,
            obstacle_type="rectangle",
            width=spot_length,
            height=spot_width,
            angle=0.0,
            color=colors["parking_spot"],
            is_parking_spot=True,
            occupied=occupied,
            is_filled=occupied,
            line_width=2,
        )
        spot.scene_role = "parking_spot"
        _register_parking_spot(env, spot, spot_id, 180.0, occupied, "perpendicular")
        _apply_spot_state(spot, occupied, colors)
        spot_id += 1

    no_parking = env.add_obstacle(
        x=width * 0.14,
        y=height * 0.45,
        obstacle_type="rectangle",
        width=9.0,
        height=3.0,
        angle=0.0,
        color=colors["no_parking"],
        is_filled=True,
        line_width=1,
    )
    no_parking.scene_role = "no_parking"


def create_default_scene(
    width: float,
    height: float,
    config: Dict[str, Any],
    env: ParkingEnvironment,
) -> None:
    """创建场景，并把停车位元数据挂到环境对象上。"""
    colors = _scene_colors(config)
    env.scene_colors = colors
    env.parking_spots_metadata = []

    _add_boundary_walls(width, height, config, env, colors)

    layout = config.get("parking_lot", {}).get("layout", [])
    if isinstance(layout, list) and layout:
        _build_scene_from_layout(width, height, config, env, colors)
    else:
        _build_fallback_scene(width, height, config, env, colors)


def _available_spot_metas(env: ParkingEnvironment) -> List[Dict[str, Any]]:
    return [
        meta
        for meta in getattr(env, "parking_spots_metadata", [])
        if not meta.get("occupied", False)
    ]


def find_spot_meta_by_id(
    env: ParkingEnvironment,
    goal_id: Optional[Union[int, str]],
    include_occupied: bool = False,
) -> Optional[Dict[str, Any]]:
    if goal_id is None:
        return None

    for meta in getattr(env, "parking_spots_metadata", []):
        if str(meta.get("id")) != str(goal_id):
            continue
        if meta.get("occupied", False) and not include_occupied:
            return None
        return meta
    return None


def choose_goal_meta(
    env: ParkingEnvironment,
    goal_id: Optional[Union[int, str]] = None,
) -> Optional[Dict[str, Any]]:
    if goal_id is not None:
        return find_spot_meta_by_id(env, goal_id)

    available = _available_spot_metas(env)
    if not available:
        return None
    return random.choice(available)


def create_parking_scenario(
    use_random_scene: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[ParkingEnvironment, Tuple[float, float], Tuple[float, float], float]:
    """创建停车场场景、起点和一个默认目标。"""
    del use_random_scene

    if config is None:
        config = load_config()

    simulation = config.get("simulation", {})
    env_width = config.get("window", {}).get("width", 1280) / simulation.get("scale", 10.0)
    env_height = config.get("window", {}).get("height", 677) / simulation.get("scale", 10.0)

    env = ParkingEnvironment(env_width, env_height)
    create_default_scene(env_width, env_height, config, env)

    start = (env_width / 2, max(2.5, env_height - 5.0))
    goal_meta = choose_goal_meta(env)
    if goal_meta:
        goal = goal_meta["position"]
        goal_orientation = float(goal_meta["orientation"])
    else:
        goal = (env_width * 0.8, env_height * 0.5)
        goal_orientation = 270.0

    print(f"起点: {start}")
    print(f"默认目标点: {goal}, 朝向: {goal_orientation:.1f}°")
    return env, start, goal, goal_orientation


def get_algorithm_specific_params(
    algorithm: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """获取算法特定参数。"""
    return get_factory_algorithm_specific_params(algorithm, args)


def optimize_path(
    path: List[Tuple[float, float]],
    env: Environment,
    vehicle_width: float,
    vehicle_length: float,
) -> List[Tuple[float, float]]:
    """对规划路径做简化、平滑和快捷化，并在失败时回退。"""
    if not path or len(path) < 3:
        print("路径点过少，跳过优化")
        return path

    original_path = list(path)
    smoother = PathSmoother(vehicle_width, vehicle_length)

    try:
        print("优化步骤1: 轨迹误差检测...")
        simplified_path = smoother.ted_detection(path)
        if simplified_path and len(simplified_path) >= 2:
            path = simplified_path
        print(f"TED后路径点数: {len(path)}")

        if len(path) >= 3:
            print("优化步骤2: 五次多项式插值...")
            interpolated_path = smoother.quintic_polynomial_interpolation(path, num_points=5)
            if interpolated_path and len(interpolated_path) >= 2:
                path = interpolated_path
            print(f"插值后路径点数: {len(path)}")
        else:
            print("优化步骤2: 路径点过少，跳过五次多项式插值")

        if len(path) >= 3:
            print("优化步骤3: 卡尔曼滤波平滑...")
            kalman_path = smoother.kalman_filter_smoothing(
                path,
                process_noise=0.1,
                measurement_noise=0.1,
            )
            if kalman_path and len(kalman_path) >= 2:
                path = kalman_path
            print(f"卡尔曼滤波后路径点数: {len(path)}")
        else:
            print("优化步骤3: 路径点过少，跳过卡尔曼滤波")

        print("优化步骤4: 路径快捷化...")
        shortcut_path = smoother.shortcut_path(path, env)
        if shortcut_path and len(shortcut_path) >= 2:
            path = shortcut_path
        print(f"快捷化后路径点数: {len(path)}")

        print("优化步骤5: 最终碰撞检测...")
        collision_info = check_path_collision(
            path,
            env,
            vehicle_length,
            vehicle_width,
            steps=max(len(path) * 2, 10),
        )
        if collision_info["collision"]:
            print(
                "警告: 优化后的路径发生碰撞，将回退到原始路径。"
            )
            return original_path

        print("优化后路径碰撞检测通过。")
        return path
    except Exception as exc:
        print(f"路径优化过程中发生错误: {exc}")
        traceback.print_exc()
        return original_path


def create_planner(
    algorithm: str,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    env: Environment,
    args: argparse.Namespace,
    vehicle_width: float,
    vehicle_length: float,
):
    """创建路径规划器。"""
    return build_planner(
        algorithm=algorithm,
        start=start,
        goal=goal,
        env=env,
        args=args,
        vehicle_width=vehicle_width,
        vehicle_length=vehicle_length,
    )


def try_plan_path(
    planner_factory: Any,
    max_retries: int = 10,
) -> Optional[List[Tuple[float, float]]]:
    """尝试多次规划，直到成功或耗尽重试次数。每次重试都重建 planner 实例。"""
    for retry in range(max_retries):
        print(f"第 {retry + 1} 次尝试...")
        try:
            planner = planner_factory()
            path = planner.plan()
        except Exception as exc:
            print(f"第 {retry + 1} 次尝试发生异常: {exc}")
            traceback.print_exc()
            path = []
        if path:
            print(f"成功规划路径，路径点数: {len(path)}")
            return path
        print(f"第 {retry + 1} 次尝试失败，继续尝试...")
    print(f"经过 {max_retries} 次尝试后仍未找到可行路径")
    return None


def _path_length(path: List[Tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1])
            for index in range(len(path) - 1)
        )
    )


def _run_planning_job(payload: Dict[str, Any], result_queue: Any) -> None:
    """在独立进程中执行规划和路径优化。"""
    request_id = int(payload["request_id"])
    started_at = time.perf_counter()

    try:
        args = argparse.Namespace(**payload["args"])
        start = tuple(payload["start"])
        goal = tuple(payload["goal"])
        env = payload["environment"]
        vehicle_width = float(payload["vehicle_width"])
        vehicle_length = float(payload["vehicle_length"])

        def planner_factory() -> Any:
            return create_planner(
                payload["algorithm"],
                start,  # type: ignore[arg-type]
                goal,  # type: ignore[arg-type]
                env,
                args,
                vehicle_width,
                vehicle_length,
            )

        raw_path = try_plan_path(planner_factory)
        if not raw_path:
            result_queue.put(
                {
                    "request_id": request_id,
                    "success": False,
                    "error": "未找到可行路径",
                    "duration": time.perf_counter() - started_at,
                }
            )
            return

        optimized_path = optimize_path(
            raw_path,
            env,
            vehicle_width,
            vehicle_length,
        )
        collision_check = check_path_collision(
            optimized_path,
            env,
            vehicle_length,
            vehicle_width,
            steps=max(len(optimized_path) * 2, 10),
        )
        if collision_check["collision"]:
            print("优化后的路径存在碰撞，回退到原始路径")
            optimized_path = raw_path

        result_queue.put(
            {
                "request_id": request_id,
                "success": True,
                "raw_path": list(raw_path),
                "path": list(optimized_path),
                "duration": time.perf_counter() - started_at,
                "raw_points": len(raw_path),
                "optimized_points": len(optimized_path),
                "path_length": _path_length(list(optimized_path)),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "request_id": request_id,
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "duration": time.perf_counter() - started_at,
            }
        )


class ParkingDemoSimulator(PygameSimulator):
    """停车场 demo 专用仿真器。"""

    planning_algorithms = [
        "rrt",
        "rrt_star",
        "informed_rrt",
        "timed_rrt",
        "astar",
        "dijkstra",
        "dstar_lite",
        "theta_star",
        "attention_dqn_rrt",
    ]
    control_methods = ["default", "pid", "mpc", "lqr"]
    steering_modes = ["normal", "counter", "crab"]

    def __init__(
        self,
        config_input: Optional[Union[str, Dict[str, Any]]] = None,
        args: Optional[argparse.Namespace] = None,
        start_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(config_input)

        self.args = args or argparse.Namespace()
        self.start_pos = start_pos or (0.0, 0.0)
        self.goal_pos: Optional[Tuple[float, float]] = None
        self.goal_meta: Optional[Dict[str, Any]] = None
        self.goal_orientation: Optional[float] = None
        self.path: Optional[List[Tuple[float, float]]] = None
        self.original_path: Optional[List[Tuple[float, float]]] = None
        self.collision_info: Optional[Dict[str, Any]] = None
        self.simulating = False
        self.show_grid = True
        self.show_spot_ids = True
        self.show_orientation = True
        self.show_safety_overlay = False
        self.simulation_speed = self.config.get(
            "speed_multiplier",
            self.config.get("simulation_speed", 2.0),
        )
        self.dt = self.config.get("dt", 0.05)
        self.last_plan_duration = 0.0
        self.last_path_length = 0.0
        self.last_raw_path_points = 0
        self.last_optimized_path_points = 0
        self.map_rect = pygame.Rect(0, 0, self.width, self.height)
        self.font_info: Dict[str, Dict[str, str]] = {}
        self.latest_terminal_snapshot: Dict[str, Any] = {}
        self._mp_context = mp.get_context("spawn" if os.name == "nt" else "fork")
        self._planning_process: Optional[mp.Process] = None
        self._planning_queue: Any = None
        self._planning_request_id = 0
        self._planning_active_request_id = 0
        self._planning_origin = "manual"
        self._planning_started_at = 0.0

        if not getattr(self.args, "algorithm", None):
            self.args.algorithm = self.config.get("path_planning", {}).get(
                "default_algorithm",
                self.planning_algorithms[0],
            )
        if self.args.algorithm not in self.planning_algorithms:
            self.args.algorithm = self.planning_algorithms[0]

        if not getattr(self.args, "control_method", None):
            self.args.control_method = self.config.get("control", {}).get(
                "default_method",
                self.control_methods[0],
            )
        if self.args.control_method not in self.control_methods:
            self.args.control_method = self.control_methods[0]

        self.current_control_method = self.args.control_method
        self.follower.set_control_method(self.current_control_method)
        self.follower.configure_terminal_parking(
            self.config.get("parking", {}).get("final_pose", {})
        )

        if not pygame.font.get_init():
            pygame.font.init()
        self.font_caption = get_font(13, role="ui")
        self.font_small = get_font(15, role="ui")
        self.font_medium = get_font(18, role="ui")
        self.font_large = get_font(22, role="title")
        self.font_panel_title = get_font(19, role="title")
        self.font_mono = get_font(16, role="mono")
        self.font_info = {
            role: get_font_resolution(role)
            for role in ("ui", "title", "mono")
        }
        for role, info in self.font_info.items():
            name = info.get("name", "unknown")
            source = info.get("source", "unknown")
            print(f"字体解析[{role}]: {name} ({source})")

        self.key_hints = [
            "右键: 选目标",
            "左键: 切换车位占用",
            "E: 换算法",
            "C: 换控制",
            "S: 换转向",
            "G: 网格",
            "I: 编号",
            "O: 安全区",
            "R/T: 重置",
            "P/空格: 暂停",
            "ESC: 退出",
        ]
        self.latest_terminal_snapshot = self.follower.get_status_snapshot()

        self._reset_vehicle_state()
        self.operation_mode = "交互等待选点"
        self.last_plan_origin = "manual"
        self.status_text = "等待选择目标点，右键选择目标后开始规划"
        self.status_color = BLACK

    def set_environment(self, env: Environment) -> None:
        super().set_environment(env)
        map_width = int(round(env.width * self.scale))
        map_height = int(round(env.height * self.scale))
        self.offset_x = max(16.0, (self.width - map_width) / 2)
        self.offset_y = max(16.0, (self.height - map_height) / 2)
        self.map_rect = pygame.Rect(
            int(self.offset_x),
            int(self.offset_y),
            map_width,
            map_height,
        )

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(round(self.offset_x + x * self.scale)),
            int(round(self.offset_y + y * self.scale)),
        )

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        return (
            (screen_x - self.offset_x) / self.scale,
            (screen_y - self.offset_y) / self.scale,
        )

    def _reset_vehicle_state(self) -> None:
        self.vehicle.x, self.vehicle.y = self.start_pos
        self.vehicle.heading = 3 * math.pi / 2
        self.vehicle.speed = 0.0
        self.vehicle.v = 0.0
        self.vehicle.acceleration = 0.0
        self.vehicle.a = 0.0
        self.vehicle.front_steer_angle = 0.0
        self.vehicle.rear_steer_angle = 0.0
        self.vehicle.steer_angle = 0.0
        self.vehicle.reverse = False
        self.vehicle.trajectory = [self.start_pos]

    def _is_planning_active(self) -> bool:
        return self._planning_process is not None or self._planning_queue is not None

    def _dispose_planning_handles(self) -> None:
        planning_process = self._planning_process
        planning_queue = self._planning_queue
        self._planning_process = None
        self._planning_queue = None
        self._planning_active_request_id = 0
        self._planning_started_at = 0.0

        if planning_process is not None:
            try:
                planning_process.join(timeout=0.05)
            except Exception:
                pass
            try:
                planning_process.close()
            except Exception:
                pass

        if planning_queue is not None:
            try:
                planning_queue.close()
            except Exception:
                pass
            try:
                planning_queue.join_thread()
            except Exception:
                pass

    def _cancel_planning_task(self, reason: Optional[str] = None) -> bool:
        if not self._is_planning_active():
            return False

        planning_process = self._planning_process
        was_alive = False
        if planning_process is not None:
            try:
                was_alive = planning_process.is_alive()
            except Exception:
                was_alive = False

        if was_alive:
            try:
                planning_process.terminate()
                planning_process.join(timeout=0.25)
            except Exception:
                pass
            try:
                if planning_process.is_alive():
                    planning_process.kill()
                    planning_process.join(timeout=0.25)
            except Exception:
                pass

        self._dispose_planning_handles()
        if reason:
            print(reason)
        return True

    def _clear_plan(self, keep_goal: bool = True) -> None:
        self.path = None
        self.original_path = None
        self.last_path_length = 0.0
        self.last_raw_path_points = 0
        self.last_optimized_path_points = 0
        self.follower.set_path([])
        self.simulating = False
        self.paused = False
        self.collision_detected = False
        self.collision_info = None
        if not keep_goal:
            self.goal_pos = None
            self.goal_meta = None
            self.goal_orientation = None
            self.follower.clear_goal_pose()
        elif self.goal_orientation is None:
            self.follower.clear_goal_pose()
        self.latest_terminal_snapshot = self.follower.get_status_snapshot()

    def _begin_planning_task(self, origin: str) -> bool:
        self._cancel_planning_task()
        self._clear_plan(keep_goal=True)
        self.last_plan_origin = origin
        self._planning_origin = origin
        self._planning_request_id += 1
        self._planning_active_request_id = self._planning_request_id
        self._planning_started_at = time.perf_counter()

        self.operation_mode = "自动规划中" if origin == "auto" else "交互规划中"
        self.status_text = (
            f"自动规划中: {self.args.algorithm}"
            if origin == "auto"
            else f"使用 {self.args.algorithm} 规划中..."
        )
        self.status_color = BLUE

        payload = {
            "request_id": self._planning_active_request_id,
            "algorithm": self.args.algorithm,
            "start": (self.vehicle.x, self.vehicle.y),
            "goal": self.goal_pos,
            "environment": self.environment,
            "args": vars(self.args).copy(),
            "vehicle_width": self.vehicle.width,
            "vehicle_length": self.vehicle.length,
        }

        try:
            self._planning_queue = self._mp_context.Queue()
            self._planning_process = self._mp_context.Process(
                target=_run_planning_job,
                args=(payload, self._planning_queue),
                daemon=True,
            )
            self._planning_process.start()
            print(
                f"\n开始规划: {(self.vehicle.x, self.vehicle.y)} -> "
                f"{self.goal_pos} ({self.args.algorithm})"
            )
            return True
        except Exception as exc:
            self._dispose_planning_handles()
            self.operation_mode = "交互等待选点"
            self.status_text = f"规划任务启动失败: {exc}"
            self.status_color = RED
            print(f"规划任务启动失败: {exc}")
            traceback.print_exc()
            return False

    def _apply_planning_result(self, result: Dict[str, Any]) -> None:
        request_id = int(result.get("request_id", -1))
        if request_id != self._planning_active_request_id:
            return

        origin = self._planning_origin
        duration = float(result.get("duration", 0.0))
        self._dispose_planning_handles()

        if not result.get("success"):
            error_text = result.get("error", "规划失败")
            tb_text = result.get("traceback")
            if tb_text:
                print(tb_text)
            self.operation_mode = "交互等待选点"
            self.status_text = (
                "自动规划失败，进入手动选点模式"
                if origin == "auto"
                else f"规划失败: {error_text}"
            )
            self.status_color = RED
            self.last_plan_duration = duration
            self._clear_plan(keep_goal=False)
            return

        raw_path = list(result.get("raw_path", []))
        path = list(result.get("path", []))
        if not raw_path or not path:
            self.operation_mode = "交互等待选点"
            self.status_text = "规划结果无效，请重新选择目标点"
            self.status_color = RED
            self.last_plan_duration = duration
            self._clear_plan(keep_goal=False)
            return

        self.path = path
        self.original_path = raw_path
        self.last_plan_duration = duration
        self.last_path_length = float(result.get("path_length", _path_length(path)))
        self.last_raw_path_points = int(result.get("raw_points", len(raw_path)))
        self.last_optimized_path_points = int(result.get("optimized_points", len(path)))

        self.follower.set_path(self.path)
        self.follower.set_control_method(self.current_control_method)
        self.follower.configure_terminal_parking(
            self.config.get("parking", {}).get("final_pose", {})
        )
        if self.goal_pos and self.goal_orientation is not None:
            slot_id = self.goal_meta.get("id") if self.goal_meta else None
            slot_type = self.goal_meta.get("slot_type", "perpendicular") if self.goal_meta else "perpendicular"
            self.follower.set_goal_pose(
                self.goal_pos[0],
                self.goal_pos[1],
                self.goal_orientation,
                slot_id=slot_id,
                slot_type=slot_type,
            )
        else:
            self.follower.clear_goal_pose()
        self.latest_terminal_snapshot = self.follower.get_status_snapshot()
        self.vehicle.steering_mode = "normal"
        self.simulating = True
        self.paused = False
        self.operation_mode = "自动仿真中" if origin == "auto" else "交互仿真中"
        self.status_text = "自动规划完成，开始仿真" if origin == "auto" else "规划完成，开始仿真"
        self.status_color = BLUE
        print(
            f"规划成功，原始点数 {self.last_raw_path_points}，优化后点数 "
            f"{self.last_optimized_path_points}，路径长度 {self.last_path_length:.2f} m"
        )

    def _poll_planning_task(self) -> None:
        if not self._is_planning_active():
            return

        result: Optional[Dict[str, Any]] = None
        if self._planning_queue is not None:
            try:
                result = self._planning_queue.get_nowait()
            except queue.Empty:
                result = None
            except Exception as exc:
                result = {
                    "request_id": self._planning_active_request_id,
                    "success": False,
                    "error": f"读取规划结果失败: {exc}",
                    "duration": time.perf_counter() - self._planning_started_at,
                }

        if result is not None:
            self._apply_planning_result(result)
            return

        if self._planning_process is None:
            return

        try:
            alive = self._planning_process.is_alive()
        except Exception:
            alive = False

        if alive:
            return

        if self._planning_queue is not None:
            try:
                result = self._planning_queue.get(timeout=0.01)
            except Exception:
                result = None

        if result is not None:
            self._apply_planning_result(result)
            return

        exitcode = getattr(self._planning_process, "exitcode", None)
        self._apply_planning_result(
            {
                "request_id": self._planning_active_request_id,
                "success": False,
                "error": f"规划进程异常退出 (exitcode={exitcode})",
                "duration": time.perf_counter() - self._planning_started_at,
            }
        )

    def _set_goal_from_meta(self, meta: Dict[str, Any]) -> None:
        self.goal_meta = meta
        self.goal_pos = meta["position"]
        self.goal_orientation = meta.get("orientation")

    def _switch_planning_algorithm(self) -> None:
        current_index = self.planning_algorithms.index(self.args.algorithm)
        self.args.algorithm = self.planning_algorithms[
            (current_index + 1) % len(self.planning_algorithms)
        ]
        self.status_text = f"规划算法切换为: {self.args.algorithm}"
        self.status_color = BLUE
        if self.goal_pos and not self.simulating:
            self._plan_path_to_goal()

    def _switch_control_method(self) -> None:
        current_index = self.control_methods.index(self.current_control_method)
        self.current_control_method = self.control_methods[
            (current_index + 1) % len(self.control_methods)
        ]
        self.follower.set_control_method(self.current_control_method)
        self.args.control_method = self.current_control_method
        self.status_text = f"控制方法切换为: {self.current_control_method}"
        self.status_color = BLUE

    def _switch_steering_mode(self) -> None:
        if self.simulating:
            self.vehicle.steering_mode = "normal"
            self.status_text = "自动泊车阶段固定 normal 转向，4WS 仅保留手动演示"
            self.status_color = YELLOW
            return
        current_mode = self.vehicle.get_steering_mode()
        current_index = self.steering_modes.index(current_mode)
        next_mode = self.steering_modes[(current_index + 1) % len(self.steering_modes)]
        self.vehicle.set_steering_mode(next_mode)
        self.status_text = f"转向模式切换为: {next_mode}"
        self.status_color = BLUE

    def _reset_simulation(self, clear_goal: bool = True) -> None:
        self._cancel_planning_task(reason="已取消当前规划任务")
        self._reset_vehicle_state()
        self._clear_plan(keep_goal=not clear_goal)
        self.operation_mode = "交互等待选点"
        self.status_text = "仿真已重置，请重新选择目标点" if clear_goal else "车辆已复位"
        self.status_color = BLACK

    def _plan_path_to_goal(self, origin: str = "manual") -> bool:
        if not self.goal_pos or not self.environment:
            self.operation_mode = "交互等待选点"
            self.status_text = "错误：未设置目标点或环境"
            self.status_color = RED
            return False

        started = self._begin_planning_task(origin)
        if started:
            self.draw()
            pygame.display.flip()
        return started

    def _toggle_clicked_spot(self, screen_pos: Tuple[int, int]) -> None:
        if not self.environment or not self.map_rect.collidepoint(screen_pos):
            return

        world_pos = self.screen_to_world(*screen_pos)
        spot = self.environment.find_parking_spot(world_pos, include_occupied=True)
        if not spot:
            self.status_text = "当前位置不是可编辑的停车位"
            self.status_color = YELLOW
            return

        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        new_state = not bool(spot.occupied)
        _apply_spot_state(spot, new_state, colors)

        meta = getattr(spot, "spot_meta", None)
        label = meta.get("label", "未知") if isinstance(meta, dict) else "未知"
        state_text = "占用" if new_state else "空闲"

        if self.goal_meta is meta and new_state:
            self._clear_plan(keep_goal=False)
            self.status_text = f"车位 {label} 设为占用，已清空当前目标"
            self.status_color = YELLOW
            return

        if self.goal_pos and not self.simulating:
            self.status_text = f"车位 {label} 已切换为{state_text}，重新规划中..."
            self.status_color = BLUE
            self._plan_path_to_goal(origin="manual")
            return

        self.status_text = f"车位 {label} 已切换为{state_text}"
        self.status_color = BLUE

    def _update(self) -> None:
        if not self.simulating or self.paused or self.collision_detected:
            return

        self.vehicle.steering_mode = "normal"
        throttle, brake, steer = self.follower.get_control(
            self.vehicle,
            self.dt * self.simulation_speed,
        )
        self.vehicle.update(throttle, brake, steer, self.dt * self.simulation_speed)
        self.collision_info = check_vehicle_collision(self.vehicle, self.environment)
        self.follower.update_terminal_clearance(self.collision_info.get("clearance", float("inf")))
        terminal_snapshot = self.follower.get_status_snapshot()
        self.latest_terminal_snapshot = terminal_snapshot

        if self.collision_info["collision"]:
            self.collision_detected = True
            self.simulating = False
            self.operation_mode = "交互等待选点"
            self.status_text = "检测到碰撞，按 R 重置"
            self.status_color = RED
            return

        if terminal_snapshot.get("failure_reason"):
            self.simulating = False
            self.operation_mode = "交互等待选点"
            self.status_text = f"{terminal_snapshot['failure_reason']}，请重规划或复位"
            self.status_color = RED
            return

        if self.collision_info["safety_warning"]:
            self.status_text = "注意：车辆接近安全边界"
            self.status_color = YELLOW
        elif terminal_snapshot["terminal_active"]:
            self.operation_mode = "终端泊车中"
            self.status_text = terminal_snapshot["status"]
            self.status_color = YELLOW if terminal_snapshot["phase"] == "hold" else BLUE

        if terminal_snapshot["success"]:
            self.simulating = False
            self.operation_mode = "交互等待选点"
            goal_label = self.goal_meta.get("label") if self.goal_meta else "-"
            self.status_text = f"已完成泊车入位: {goal_label}"
            self.status_color = GREEN
            return

        if self.goal_pos and self.goal_orientation is None:
            distance_to_goal = math.hypot(
                self.goal_pos[0] - self.vehicle.x,
                self.goal_pos[1] - self.vehicle.y,
            )
            if (
                distance_to_goal < 0.6
                and abs(self.vehicle.speed) < 0.2
            ) or (
                self.path
                and self.follower.current_target_idx >= len(self.path) - 1
                and distance_to_goal < 1.0
            ):
                self.simulating = False
                self.operation_mode = "交互等待选点"
                self.status_text = "已到达目标点，右键可重新规划"
                self.status_color = GREEN

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_p):
                    self.paused = not self.paused
                    self.status_text = "仿真已暂停" if self.paused else "仿真继续"
                    self.status_color = YELLOW if self.paused else BLUE
                elif event.key == pygame.K_r:
                    self._reset_simulation(clear_goal=False)
                elif event.key == pygame.K_t:
                    self._reset_simulation(clear_goal=True)
                elif event.key == pygame.K_e:
                    self._switch_planning_algorithm()
                elif event.key == pygame.K_c:
                    self._switch_control_method()
                elif event.key == pygame.K_s:
                    self._switch_steering_mode()
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    self.status_text = f"网格显示: {'开' if self.show_grid else '关'}"
                    self.status_color = BLUE
                elif event.key == pygame.K_i:
                    self.show_spot_ids = not self.show_spot_ids
                    self.status_text = f"车位编号显示: {'开' if self.show_spot_ids else '关'}"
                    self.status_color = BLUE
                elif event.key == pygame.K_o:
                    self.show_safety_overlay = not self.show_safety_overlay
                    self.status_text = f"安全区显示: {'开' if self.show_safety_overlay else '关'}"
                    self.status_color = BLUE

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and not self.simulating:
                    self._toggle_clicked_spot(event.pos)
                    continue

                if event.button == 3 and not self.simulating:
                    if not self.map_rect.collidepoint(event.pos):
                        continue

                    world_pos = self.screen_to_world(*event.pos)
                    if not (
                        0.0 <= world_pos[0] <= self.environment.width
                        and 0.0 <= world_pos[1] <= self.environment.height
                    ):
                        continue

                    parking_spot = None
                    if isinstance(self.environment, ParkingEnvironment):
                        parking_spot = self.environment.find_parking_spot(world_pos)

                    if parking_spot is not None:
                        self._set_goal_from_meta(parking_spot.spot_meta)
                    else:
                        temp_vehicle = VehicleModel(
                            world_pos[0],
                            world_pos[1],
                            0.0,
                            self.vehicle.length,
                            self.vehicle.width,
                        )
                        collision_info = check_vehicle_collision(temp_vehicle, self.environment)
                        if collision_info["collision"]:
                            self.status_text = "目标点在障碍物内，请重新选择"
                            self.status_color = RED
                            continue
                        self.goal_meta = None
                        self.goal_pos = world_pos
                        self.goal_orientation = None

                    self._plan_path_to_goal(origin="manual")

        return True

    def _rgba(self, color: Tuple[int, ...], fallback_alpha: int = 255) -> Tuple[int, int, int, int]:
        if len(color) == 4:
            return color  # type: ignore[return-value]
        return (int(color[0]), int(color[1]), int(color[2]), fallback_alpha)

    def _alpha_scale(self, color: Tuple[int, ...], factor: float) -> Tuple[int, int, int, int]:
        rgba = self._rgba(color)
        alpha = max(0, min(255, int(round(rgba[3] * factor))))
        return (rgba[0], rgba[1], rgba[2], alpha)

    def _is_target_spot(self, obstacle: Any) -> bool:
        meta = getattr(obstacle, "spot_meta", None)
        if not isinstance(meta, dict) or not self.goal_meta:
            return False
        return str(meta.get("id")) == str(self.goal_meta.get("id"))

    def _wrap_text(
        self,
        text: str,
        font: pygame.font.Font,
        max_width: int,
        max_lines: int = 2,
    ) -> List[str]:
        if not text:
            return [""]
        raw_lines = text.splitlines() or [text]
        wrapped: List[str] = []
        truncated = False

        for raw_line in raw_lines:
            current = ""
            for char in raw_line:
                candidate = f"{current}{char}"
                if not current or font.size(candidate)[0] <= max_width:
                    current = candidate
                    continue
                wrapped.append(current)
                current = char
                if len(wrapped) >= max_lines:
                    truncated = True
                    break
            if truncated:
                break
            if current or not wrapped:
                wrapped.append(current)
            if len(wrapped) >= max_lines and raw_line != raw_lines[-1]:
                truncated = True
                break

        wrapped = wrapped[: max(1, max_lines)]
        if truncated and wrapped:
            last_line = wrapped[-1].rstrip()
            while last_line and font.size(f"{last_line}...")[0] > max_width:
                last_line = last_line[:-1].rstrip()
            wrapped[-1] = f"{last_line}..." if last_line else "..."
        return wrapped

    def _panel_line_specs(
        self,
        text: str,
        color: Tuple[int, int, int],
        font: Optional[pygame.font.Font] = None,
        max_lines: int = 1,
    ) -> Dict[str, Any]:
        return {
            "text": text,
            "color": color,
            "font": font or self.font_small,
            "max_lines": max_lines,
        }

    def _phase_label(self, phase: str) -> str:
        return {
            "idle": "路径跟踪",
            "stage": "接近目标位姿",
            "dock": "倒车对位",
            "adjust": "姿态精调",
            "hold": "停车保持",
            "failed": "需重规划/复位",
        }.get(phase, phase)

    def _draw_direction_arrow(
        self,
        center: Tuple[float, float],
        angle_deg: float,
        length: float,
        color: Tuple[int, ...],
        width: int = 3,
    ) -> None:
        angle_rad = math.radians(angle_deg)
        start = (
            center[0] - math.cos(angle_rad) * length * 0.25,
            center[1] - math.sin(angle_rad) * length * 0.25,
        )
        end = (
            center[0] + math.cos(angle_rad) * length * 0.75,
            center[1] + math.sin(angle_rad) * length * 0.75,
        )
        pygame.draw.line(
            self.screen,
            self._rgba(color),
            self.world_to_screen(*start),
            self.world_to_screen(*end),
            width,
        )
        head_left = (
            end[0] - math.cos(angle_rad) * length * 0.22 + math.sin(angle_rad) * length * 0.12,
            end[1] - math.sin(angle_rad) * length * 0.22 - math.cos(angle_rad) * length * 0.12,
        )
        head_right = (
            end[0] - math.cos(angle_rad) * length * 0.22 - math.sin(angle_rad) * length * 0.12,
            end[1] - math.sin(angle_rad) * length * 0.22 + math.cos(angle_rad) * length * 0.12,
        )
        pygame.draw.polygon(
            self.screen,
            self._rgba(color),
            [
                self.world_to_screen(*end),
                self.world_to_screen(*head_left),
                self.world_to_screen(*head_right),
            ],
        )

    def _draw_rotated_rect(
        self,
        center: Tuple[float, float],
        width: float,
        height: float,
        angle: float,
        color: Tuple[int, ...],
        fill: bool = True,
        line_width: int = 1,
        border_radius: int = 5,
    ) -> None:
        pixel_width = max(2, int(round(width * self.scale)))
        pixel_height = max(2, int(round(height * self.scale)))
        surface = pygame.Surface((pixel_width + 6, pixel_height + 6), pygame.SRCALPHA)
        rect = pygame.Rect(3, 3, pixel_width, pixel_height)
        draw_width = 0 if fill else max(1, line_width)
        pygame.draw.rect(surface, self._rgba(color), rect, draw_width, border_radius=border_radius)
        rotated = pygame.transform.rotate(surface, -angle)
        rotated_rect = rotated.get_rect(center=self.world_to_screen(*center))
        self.screen.blit(rotated, rotated_rect)

    def _draw_line_in_rect(
        self,
        center: Tuple[float, float],
        angle: float,
        local_start: Tuple[float, float],
        local_end: Tuple[float, float],
        color: Tuple[int, ...],
        width: int = 2,
    ) -> None:
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        def transform(point: Tuple[float, float]) -> Tuple[float, float]:
            px, py = point
            return (
                center[0] + px * cos_angle - py * sin_angle,
                center[1] + px * sin_angle + py * cos_angle,
            )

        pygame.draw.line(
            self.screen,
            self._rgba(color),
            self.world_to_screen(*transform(local_start)),
            self.world_to_screen(*transform(local_end)),
            width,
        )

    def _draw_grid(self) -> None:
        if not self.show_grid or not self.environment:
            return

        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        grid_color = (*colors["lane_line"][:3], 55)
        step = 5.0

        for x in np.arange(0.0, self.environment.width + 1e-6, step):
            pygame.draw.line(
                self.screen,
                grid_color,
                self.world_to_screen(float(x), 0.0),
                self.world_to_screen(float(x), self.environment.height),
                1,
            )
        for y in np.arange(0.0, self.environment.height + 1e-6, step):
            pygame.draw.line(
                self.screen,
                grid_color,
                self.world_to_screen(0.0, float(y)),
                self.world_to_screen(self.environment.width, float(y)),
                1,
            )

    def _draw_parking_spot(self, obstacle: Any) -> None:
        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        occupied = bool(obstacle.occupied)
        is_target = self._is_target_spot(obstacle)
        terminal_active = bool(self.latest_terminal_snapshot.get("terminal_active"))
        fade_factor = 0.42 if terminal_active and not is_target else 1.0
        border_color = colors["occupied_border"] if occupied else colors["parking_spot_border"]
        fill_color = colors["occupied_fill"] if occupied else colors["parking_spot"]
        border_color = self._alpha_scale(border_color, fade_factor)
        fill_color = self._alpha_scale(fill_color, fade_factor)
        line_color = self._alpha_scale(colors["parking_spot_line"], 0.55 if terminal_active and not is_target else 1.0)

        safety_obstacle = getattr(obstacle, "safety_obstacle", None)
        if safety_obstacle is not None and (self.show_safety_overlay or is_target):
            self._draw_rotated_rect(
                (safety_obstacle.x, safety_obstacle.y),
                safety_obstacle.width,
                safety_obstacle.height,
                getattr(safety_obstacle, "angle", 0.0),
                self._alpha_scale(getattr(safety_obstacle, "color", colors["safety_zone"]), 0.65 if is_target else 0.42),
                fill=True,
                line_width=1,
                border_radius=8,
            )

        if is_target:
            self._draw_rotated_rect(
                (obstacle.x, obstacle.y),
                obstacle.width + 0.75,
                obstacle.height + 0.75,
                getattr(obstacle, "angle", 0.0),
                colors["target_spot_glow"],
                fill=True,
                line_width=1,
                border_radius=10,
            )

        self._draw_rotated_rect(
            (obstacle.x, obstacle.y),
            obstacle.width,
            obstacle.height,
            getattr(obstacle, "angle", 0.0),
            fill_color,
            fill=occupied,
            line_width=2,
            border_radius=6,
        )
        self._draw_rotated_rect(
            (obstacle.x, obstacle.y),
            obstacle.width,
            obstacle.height,
            getattr(obstacle, "angle", 0.0),
            border_color,
            fill=False,
            line_width=2,
            border_radius=6,
        )

        line_length = obstacle.width * 0.34
        self._draw_line_in_rect(
            (obstacle.x, obstacle.y),
            getattr(obstacle, "angle", 0.0),
            (-line_length, 0.0),
            (line_length, 0.0),
            line_color,
            width=2,
        )

        meta = getattr(obstacle, "spot_meta", None)
        if isinstance(meta, dict):
            orientation = float(meta.get("orientation", 0.0))
            if self.show_orientation or is_target:
                arrow_color = colors["target_spot_border"] if is_target else line_color
                self._draw_direction_arrow(
                    (obstacle.x, obstacle.y),
                    orientation,
                    min(obstacle.width, obstacle.height) * (0.48 if is_target else 0.36),
                    arrow_color,
                    width=3 if is_target else 2,
                )

            if is_target and self.goal_orientation is not None and self.goal_pos is not None:
                pos_tolerance = float(
                    self.config.get("parking", {}).get("final_pose", {}).get("position_tolerance", 0.25)
                )
                self._draw_rotated_rect(
                    self.goal_pos,
                    self.vehicle.length + pos_tolerance * 2.0,
                    self.vehicle.width + pos_tolerance * 2.0,
                    self.goal_orientation,
                    colors["target_pose"],
                    fill=False,
                    line_width=2,
                    border_radius=10,
                )
                self._draw_rotated_rect(
                    self.goal_pos,
                    self.vehicle.length,
                    self.vehicle.width,
                    self.goal_orientation,
                    self._alpha_scale(colors["target_pose"], 0.35),
                    fill=False,
                    line_width=1,
                    border_radius=8,
                )
                self._draw_rotated_rect(
                    (obstacle.x, obstacle.y),
                    obstacle.width,
                    obstacle.height,
                    getattr(obstacle, "angle", 0.0),
                    colors["target_spot_border"],
                    fill=False,
                    line_width=3,
                    border_radius=8,
                )

        if self.show_spot_ids and isinstance(meta, dict):
            label_surface = self.font_caption.render(
                meta.get("label", ""),
                True,
                colors["panel_bg"][:3] if occupied else colors["panel_title"][:3],
            )
            label_rect = label_surface.get_rect(center=self.world_to_screen(obstacle.x, obstacle.y))
            self.screen.blit(label_surface, label_rect)

    def _draw_no_parking_area(self, obstacle: Any) -> None:
        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        self._draw_rotated_rect(
            (obstacle.x, obstacle.y),
            obstacle.width,
            obstacle.height,
            getattr(obstacle, "angle", 0.0),
            colors["no_parking"],
            fill=True,
            line_width=1,
            border_radius=4,
        )
        for ratio in np.linspace(-0.4, 0.4, 4):
            start = (
                obstacle.x - obstacle.width * 0.45,
                obstacle.y + obstacle.height * ratio,
            )
            end = (
                obstacle.x + obstacle.width * 0.45,
                obstacle.y + obstacle.height * (ratio - 0.35),
            )
            pygame.draw.line(
                self.screen,
                self._rgba((170, 60, 40, 150)),
                self.world_to_screen(*start),
                self.world_to_screen(*end),
                2,
            )

    def _draw_obstacles(self) -> None:
        if not self.environment:
            return

        for index in range(1, len(self.environment.obstacles), 2):
            obstacle = self.environment.obstacles[index]
            scene_role = getattr(obstacle, "scene_role", "obstacle")

            if getattr(obstacle, "is_parking_spot", False):
                self._draw_parking_spot(obstacle)
                continue

            if obstacle.type == "circle":
                pygame.draw.circle(
                    self.screen,
                    self._rgba(getattr(obstacle, "color", (120, 120, 120, 220))),
                    self.world_to_screen(obstacle.x, obstacle.y),
                    int(round(obstacle.radius * self.scale)),
                )
                continue

            if scene_role == "no_parking":
                self._draw_no_parking_area(obstacle)
                continue

            self._draw_rotated_rect(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                getattr(obstacle, "angle", 0.0),
                getattr(obstacle, "color", (120, 120, 120, 220)),
                fill=True,
                line_width=max(1, int(getattr(obstacle, "line_width", 1))),
                border_radius=3 if scene_role == "wall" else 2,
            )

    def _draw_path(self) -> None:
        if not self.path:
            return
        screen_points = [self.world_to_screen(x, y) for x, y in self.path]
        if len(screen_points) > 1:
            pygame.draw.lines(self.screen, (51, 107, 255), False, screen_points, 4)
            pygame.draw.lines(self.screen, (215, 229, 255), False, screen_points, 2)
        for point in screen_points[:: max(1, len(screen_points) // 24)]:
            pygame.draw.circle(self.screen, (21, 63, 181), point, 4)

    def _draw_trajectory(self) -> None:
        if not self.vehicle.trajectory or len(self.vehicle.trajectory) < 2:
            return
        screen_points = [self.world_to_screen(x, y) for x, y in self.vehicle.trajectory]
        pygame.draw.lines(self.screen, (243, 158, 53), False, screen_points, 3)

    def _draw_markers(self) -> None:
        start_screen = self.world_to_screen(*self.start_pos)
        pygame.draw.circle(self.screen, (34, 139, 98), start_screen, 8)
        pygame.draw.circle(self.screen, WHITE, start_screen, 8, 2)

        if not self.goal_pos:
            return

        goal_screen = self.world_to_screen(*self.goal_pos)
        pygame.draw.circle(self.screen, (194, 56, 56), goal_screen, 10)
        pygame.draw.circle(self.screen, WHITE, goal_screen, 10, 2)
        pygame.draw.line(
            self.screen,
            WHITE,
            (goal_screen[0] - 12, goal_screen[1]),
            (goal_screen[0] + 12, goal_screen[1]),
            2,
        )
        pygame.draw.line(
            self.screen,
            WHITE,
            (goal_screen[0], goal_screen[1] - 12),
            (goal_screen[0], goal_screen[1] + 12),
            2,
        )

    def _draw_panel_box(
        self,
        rect: pygame.Rect,
        lines: List[Dict[str, Any]],
        title: Optional[str] = None,
    ) -> None:
        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        panel_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, colors["panel_bg"], panel_surface.get_rect(), border_radius=14)
        pygame.draw.rect(panel_surface, colors["panel_border"], panel_surface.get_rect(), 1, border_radius=14)
        self.screen.blit(panel_surface, rect)

        y = rect.top + 12
        if title:
            title_surface = self.font_panel_title.render(title, True, colors["panel_title"][:3])
            self.screen.blit(title_surface, (rect.left + 14, y))
            y += 28

        content_width = rect.width - 28
        for line in lines:
            text = line.get("text", "")
            color = line.get("color", BLACK)
            font = line.get("font", self.font_small)
            max_lines = int(line.get("max_lines", 1))
            wrapped_lines = self._wrap_text(text, font, content_width, max_lines=max_lines)
            for wrapped_line in wrapped_lines:
                line_surface = font.render(wrapped_line, True, color)
                self.screen.blit(line_surface, (rect.left + 14, y))
                y += font.get_linesize()
            y += 4

    def _draw_hud(self) -> None:
        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        snapshot = self.latest_terminal_snapshot or self.follower.get_status_snapshot()
        errors = snapshot.get("errors", {})
        available = len(_available_spot_metas(self.environment)) if self.environment else 0
        total_spots = len(getattr(self.environment, "parking_spots_metadata", [])) if self.environment else 0
        steering_mode = self.vehicle.get_steering_mode() if hasattr(self.vehicle, "get_steering_mode") else "normal"
        gear_label = self.vehicle.get_gear_label() if hasattr(self.vehicle, "get_gear_label") else ("R" if self.vehicle.reverse else "D")
        steer_deg = math.degrees(getattr(self.vehicle, "front_steer_angle", 0.0))
        path_points = len(self.path or [])
        phase_text = self._phase_label(snapshot.get("phase", "idle"))
        dx = float(errors.get("center_dx", self.goal_pos[0] - self.vehicle.x if self.goal_pos else 0.0))
        dy = float(errors.get("center_dy", self.goal_pos[1] - self.vehicle.y if self.goal_pos else 0.0))
        dyaw = float(errors.get("yaw_error_deg", 0.0))
        clearance = float(snapshot.get("clearance", float("inf")))
        clearance_text = f"{clearance:.2f} m" if math.isfinite(clearance) else "--"
        slot_type = snapshot.get("goal_slot_type") or (self.goal_meta or {}).get("slot_type") or "perpendicular"
        planning_active = self._is_planning_active()
        planning_elapsed = max(0.0, time.perf_counter() - self._planning_started_at) if planning_active else 0.0

        left_lines = [
            self._panel_line_specs(f"模式  {self.operation_mode}", colors["panel_title"][:3], self.font_small, 2),
            self._panel_line_specs(f"算法  {self.args.algorithm}", BLACK, self.font_small),
            self._panel_line_specs(f"控制  {self.current_control_method}", BLACK, self.font_small),
            self._panel_line_specs(f"转向模式  {steering_mode}", BLACK, self.font_small),
            self._panel_line_specs(
                f"规划状态  {'进行中' if planning_active else '空闲'}",
                BLUE if planning_active else BLACK,
                self.font_small,
            ),
            self._panel_line_specs(f"挡位  {gear_label}", BLACK, self.font_mono),
            self._panel_line_specs(f"速度  {self.vehicle.speed:.2f} m/s", BLACK, self.font_mono),
            self._panel_line_specs(f"转角  {steer_deg:+.1f}°", BLACK, self.font_mono),
            self._panel_line_specs(f"路径点  {path_points}", BLACK, self.font_mono),
            self._panel_line_specs(f"空闲车位  {available}/{total_spots}", colors["text_muted"][:3], self.font_small),
        ]
        if planning_active:
            left_lines.append(
                self._panel_line_specs(f"规划计时  {planning_elapsed:.1f} s", BLUE, self.font_mono)
            )
        if self.last_optimized_path_points:
            left_lines.extend(
                [
                    self._panel_line_specs(f"规划耗时  {self.last_plan_duration:.2f} s", BLACK, self.font_mono),
                    self._panel_line_specs(
                        f"优化前后  {self.last_raw_path_points} -> {self.last_optimized_path_points}",
                        BLACK,
                        self.font_mono,
                    ),
                    self._panel_line_specs(f"路径长度  {self.last_path_length:.2f} m", BLACK, self.font_mono),
                ]
            )
        left_panel_height = max(292, 56 + len(left_lines) * 23)
        self._draw_panel_box(pygame.Rect(20, 18, 316, left_panel_height), left_lines, title="运行信息")

        goal_label = "-"
        if self.goal_meta:
            goal_label = f"车位 {self.goal_meta.get('label', self.goal_meta.get('id', '-'))}"
        elif self.goal_pos:
            goal_label = f"({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})"

        task_lines = [
            self._panel_line_specs(f"目标  {goal_label}", colors["panel_title"][:3], self.font_small, 2),
            self._panel_line_specs(f"类型  {slot_type}", colors["text_muted"][:3], self.font_small),
            self._panel_line_specs(f"dx / dy  {dx:+.2f} / {dy:+.2f} m", BLACK, self.font_mono),
            self._panel_line_specs(f"dyaw  {dyaw:+.1f}°", BLACK, self.font_mono),
            self._panel_line_specs(f"阶段  {phase_text}", BLACK, self.font_small, 2),
            self._panel_line_specs(f"安全间隙  {clearance_text}", BLACK, self.font_mono),
            self._panel_line_specs(
                f"暂停  {'是' if self.paused else '否'}    网格/编号  {'开' if self.show_grid else '关'} / {'开' if self.show_spot_ids else '关'}",
                colors["text_muted"][:3],
                self.font_caption,
                2,
            ),
            self._panel_line_specs(
                f"状态  {self.status_text}",
                self.status_color,
                self.font_small,
                3,
            ),
        ]
        self._draw_panel_box(
            pygame.Rect(self.width - 388, 18, 368, 248),
            task_lines,
            title="任务信息",
        )

        hint_rect = pygame.Rect(20, self.height - 78, self.width - 40, 58)
        hint_surface = pygame.Surface((hint_rect.width, hint_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(hint_surface, (28, 33, 36, 188), hint_surface.get_rect(), border_radius=14)
        self.screen.blit(hint_surface, hint_rect)
        hint_text = "  |  ".join(self.key_hints)
        hint_lines = self._wrap_text(hint_text, self.font_caption, hint_rect.width - 140, max_lines=2)
        for index, line in enumerate(hint_lines):
            hint_render = self.font_caption.render(line, True, WHITE)
            hint_box = hint_render.get_rect()
            hint_box.left = hint_rect.left + 18
            hint_box.centery = hint_rect.centery - (8 if len(hint_lines) > 1 and index == 0 else -8 if len(hint_lines) > 1 else 0)
            self.screen.blit(hint_render, hint_box)

        badge_color = colors["warning"][:3] if self.paused else self.status_color
        badge_rect = pygame.Rect(hint_rect.right - 184, hint_rect.top + 11, 164, hint_rect.height - 22)
        pygame.draw.rect(self.screen, (*badge_color, 228), badge_rect, border_radius=12)
        badge_font = self.font_small if len(self.status_text) < 18 else self.font_caption
        badge_lines = self._wrap_text(self.status_text, badge_font, badge_rect.width - 22, max_lines=2)
        for index, line in enumerate(badge_lines):
            badge_surface = badge_font.render(line, True, WHITE)
            badge_box = badge_surface.get_rect(centerx=badge_rect.centerx)
            badge_box.centery = badge_rect.centery - (8 if len(badge_lines) > 1 and index == 0 else -8 if len(badge_lines) > 1 else 0)
            self.screen.blit(badge_surface, badge_box)

    def _draw_planning_overlay(self) -> None:
        if not self._is_planning_active():
            return

        elapsed = max(0.0, time.perf_counter() - self._planning_started_at)
        dot_count = int(elapsed * 3.0) % 4
        label = f"规划中{'.' * dot_count}"
        sub_label = f"{self.args.algorithm} 运行中  {elapsed:.1f} s"

        overlay_rect = pygame.Rect(0, 0, 320, 108)
        overlay_rect.center = (self.width // 2, self.height // 2)

        overlay = pygame.Surface((overlay_rect.width, overlay_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (17, 24, 31, 210), overlay.get_rect(), border_radius=18)
        pygame.draw.rect(overlay, (86, 157, 255, 235), overlay.get_rect(), 2, border_radius=18)

        spinner_center = (56, overlay_rect.height // 2)
        spinner_radius = 18
        pygame.draw.circle(overlay, (90, 104, 118, 180), spinner_center, spinner_radius, 4)
        rotation = (elapsed * 5.5) % (2 * math.pi)
        pygame.draw.arc(
            overlay,
            (255, 214, 102, 255),
            pygame.Rect(
                spinner_center[0] - spinner_radius,
                spinner_center[1] - spinner_radius,
                spinner_radius * 2,
                spinner_radius * 2,
            ),
            rotation,
            rotation + math.radians(250),
            5,
        )

        title_surface = self.font_medium.render(label, True, WHITE)
        title_rect = title_surface.get_rect()
        title_rect.left = 92
        title_rect.top = 26
        overlay.blit(title_surface, title_rect)

        detail_surface = self.font_caption.render(sub_label, True, (208, 218, 226))
        detail_rect = detail_surface.get_rect()
        detail_rect.left = 92
        detail_rect.top = 58
        overlay.blit(detail_surface, detail_rect)

        hint_surface = self.font_caption.render("按 R 立即重置，按 T 清空目标", True, (176, 190, 201))
        hint_rect = hint_surface.get_rect()
        hint_rect.left = 92
        hint_rect.top = 78
        overlay.blit(hint_surface, hint_rect)

        self.screen.blit(overlay, overlay_rect)

    def draw(self) -> None:
        colors = getattr(self.environment, "scene_colors", _scene_colors(load_config(None)))
        self.screen.fill((233, 229, 220))

        if self.environment:
            pygame.draw.rect(self.screen, colors["asphalt"], self.map_rect, border_radius=16)
            pygame.draw.rect(self.screen, (30, 36, 38), self.map_rect, 2, border_radius=16)
            self._draw_grid()
            self._draw_obstacles()

        self._draw_path()
        self._draw_trajectory()
        self._draw_markers()
        self._draw_vehicle(
            self.screen,
            self.vehicle,
            scale=self.scale,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            color=None,
        )
        self._draw_hud()
        self._draw_planning_overlay()

    def _auto_start(self, preferred_goal_meta: Optional[Dict[str, Any]] = None) -> bool:
        """启动时自动选择目标并开始规划。"""
        if not self.environment:
            self.operation_mode = "交互等待选点"
            self.status_text = "环境未初始化，进入手动选点模式"
            self.status_color = RED
            return False

        goal_meta = preferred_goal_meta or choose_goal_meta(self.environment)
        if goal_meta is not None:
            label = goal_meta.get("label", goal_meta.get("id", "-"))
            print(f"启动自动规划，目标车位: {label} @ {goal_meta['position']}")
            self._set_goal_from_meta(goal_meta)
        else:
            fallback_goal = (self.environment.width * 0.8, self.environment.height * 0.5)
            print(f"未找到可用车位，回退到自由目标点: {fallback_goal}")
            self.goal_meta = None
            self.goal_pos = fallback_goal
            self.goal_orientation = None

        started = self._plan_path_to_goal(origin="auto")
        if started:
            print("已启动自动规划任务，等待结果")
            return True

        print("自动规划失败，进入手动选点模式")
        self.operation_mode = "交互等待选点"
        self.status_text = "自动规划失败，进入手动选点模式"
        self.status_color = YELLOW
        return False

    def run(self) -> None:
        self.running = True
        fps = self.config.get("fps", 60)

        try:
            while self.running:
                if not self._handle_events():
                    break
                self._poll_planning_task()
                self._update()
                self.draw()
                pygame.display.flip()
                self.clock.tick(fps)
        finally:
            self._cancel_planning_task(reason="退出仿真，清理后台规划任务")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="改进版停车场路径规划仿真")
    parser.add_argument(
        "--config",
        type=str,
        default="config/parking_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=ParkingDemoSimulator.planning_algorithms,
        help="规划算法",
    )
    parser.add_argument("--iterations", type=int, default=2000, help="最大迭代次数")
    parser.add_argument("--step_size", type=float, default=2.0, help="步长")
    parser.add_argument("--robot_speed", type=float, default=4.0, help="Timed RRT 机器人速度")
    parser.add_argument(
        "--control_method",
        type=str,
        default=None,
        choices=ParkingDemoSimulator.control_methods,
        help="路径跟踪控制方法",
    )
    parser.add_argument("--model_path", type=str, default="", help="Attention DQN 模型路径")
    parser.add_argument("--goal-id", type=str, default=None, help="启动时指定目标车位编号")
    parser.add_argument("--plan-only", action="store_true", help="只做一次规划并退出")
    parser.add_argument("--hide-grid", action="store_true", help="启动时隐藏网格")
    parser.add_argument("--hide-ids", action="store_true", help="启动时隐藏车位编号")
    parser.add_argument("--random_scene", action="store_true", help="保留兼容参数，当前不启用随机地图生成")
    return parser.parse_args()


def plan_once(
    env: ParkingEnvironment,
    start: Tuple[float, float],
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    goal_meta = choose_goal_meta(env, args.goal_id)
    if args.goal_id is not None and goal_meta is None:
        print(f"错误: 未找到可用车位 {args.goal_id}")
        return False

    if goal_meta is not None:
        goal = goal_meta["position"]
        goal_desc = f"车位 {goal_meta['label']}"
    else:
        goal = (env.width * 0.8, env.height * 0.5)
        goal_desc = f"自由点 {goal}"

    simulator_config = build_simulator_config(config)
    vehicle_config = simulator_config["vehicle"]
    def planner_factory() -> Any:
        return create_planner(
            args.algorithm,
            start,
            goal,
            env,
            args,
            vehicle_config["width"],
            vehicle_config["length"],
        )

    path = try_plan_path(planner_factory)
    if not path:
        print(f"规划失败: {start} -> {goal_desc}")
        return False

    optimized = optimize_path(path, env, vehicle_config["width"], vehicle_config["length"])
    print(
        f"规划成功: {start} -> {goal_desc}, 原始点数 {len(path)}, "
        f"优化后点数 {len(optimized)}, 路径长度 {_path_length(optimized):.2f} m"
    )
    return True


def main() -> None:
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    PygameSimulator.configure_rendering_environment()

    args = parse_args()
    config = load_config(args.config)

    if args.algorithm is None:
        args.algorithm = config.get("path_planning", {}).get("default_algorithm", "rrt_star")
    if args.control_method is None:
        args.control_method = config.get("control", {}).get("default_method", "pid")

    simulator_config = build_simulator_config(config)
    env, start, _, _ = create_parking_scenario(
        use_random_scene=args.random_scene,
        config=config,
    )

    if args.plan_only:
        success = plan_once(env, start, config, args)
        raise SystemExit(0 if success else 1)

    pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()

    print("启用 SDL 软件渲染兼容模式，优先避免 EGL/DRI 图形告警影响启动")

    try:
        simulator = ParkingDemoSimulator(simulator_config, args=args, start_pos=start)
    except Exception as exc:
        print(f"图形初始化失败: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc

    simulator.set_environment(env)
    simulator.show_grid = not args.hide_grid
    simulator.show_spot_ids = not args.hide_ids

    preferred_goal_meta = None
    if args.goal_id is not None:
        preferred_goal_meta = choose_goal_meta(env, args.goal_id)
        if preferred_goal_meta is None:
            print(f"警告: 未找到可用车位 {args.goal_id}，将尝试自动选择其他目标")

    simulator._auto_start(preferred_goal_meta)
    if simulator.operation_mode == "交互等待选点":
        print("当前处于交互模式：右键规划到车位或自由点，左键切换车位占用，Esc 退出")

    try:
        simulator.run()
    except Exception as exc:
        print(f"仿真过程中发生错误: {exc}")
        traceback.print_exc()
    finally:
        try:
            pygame.display.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
