#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""统一的 planner 构造与参数适配入口。"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from .astar import AStar
from .attention_dqn_rrt import AttentionDQNRRT
from .dijkstra import Dijkstra
from .dstar_lite import DStarLite
from .informed_rrt import InformedRRTStar
from .ppo_planner import PPOPathPlanner
from .rl_planner import RLPathPlanner
from .rrt_base import RRT
from .rrt_star import RRTStar, TimedRRTStar
from .theta_star import ThetaStar

PLANNER_REGISTRY: Dict[str, Type[Any]] = {
    "astar": AStar,
    "rrt": RRT,
    "rrt_star": RRTStar,
    "informed_rrt": InformedRRTStar,
    "timed_rrt": TimedRRTStar,
    "dijkstra": Dijkstra,
    "dstar_lite": DStarLite,
    "theta_star": ThetaStar,
    "attention_dqn_rrt": AttentionDQNRRT,
    "rl": RLPathPlanner,
    "ppo": PPOPathPlanner,
}


def get_planner_class(algorithm: str) -> Type[Any]:
    """根据算法名获取 planner 类型。"""
    if algorithm not in PLANNER_REGISTRY:
        raise ValueError(f"不支持的算法: {algorithm}")
    return PLANNER_REGISTRY[algorithm]


def _get_arg(args: Optional[Any], name: str, default: Any) -> Any:
    return getattr(args, name, default) if args is not None else default


def get_algorithm_specific_params(
    algorithm: str,
    args: Optional[Any] = None,
) -> Dict[str, Any]:
    """统一返回算法特定参数。"""
    max_iterations = int(_get_arg(args, "iterations", 10000))
    step_size = _get_arg(args, "step_size", 2.0)
    base_params = {
        "max_iterations": max_iterations,
        "step_size": step_size,
    }

    params: Dict[str, Dict[str, Any]] = {
        "astar": {
            "resolution": 0.5,
            "diagonal_movement": True,
            "weight": 1.0,
        },
        "rrt": base_params,
        "rrt_star": {**base_params, "rewire_factor": 1.5},
        "informed_rrt": {**base_params, "focus_factor": 1.0},
        "timed_rrt": {**base_params, "robot_speed": _get_arg(args, "robot_speed", 1.0)},
        "dijkstra": {
            "resolution": 1.0,
            "diagonal_movement": True,
        },
        "dstar_lite": {
            "resolution": 1.0,
            "diagonal_movement": True,
        },
        "theta_star": {
            "resolution": 1.0,
            "diagonal_movement": True,
        },
        "attention_dqn_rrt": {
            **base_params,
            "rewire_factor": 1.5,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1,
            "buffer_capacity": 10000,
            "batch_size": 64,
            "hidden_dim": 256,
            "prediction_horizon": 5,
        },
        "rl": {
            "resolution": 1.0,
            "max_steps": max_iterations,
            "model_path": _get_arg(args, "model_path", None),
        },
        "ppo": {
            "resolution": 1.0,
            "max_steps": max_iterations,
            "model_path": _get_arg(args, "model_path", None),
        },
    }

    result = params.get(algorithm, {}).copy()
    model_path = _get_arg(args, "model_path", None)
    if algorithm == "attention_dqn_rrt" and model_path:
        if os.path.exists(model_path):
            result["model_path"] = model_path
        else:
            print(f"警告: 模型文件 {model_path} 不存在，将使用默认初始化")
    return result


def normalize_planner_kwargs(planner_class: Type[Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """按 planner 签名过滤参数，避免项目侧硬编码每个构造函数。"""
    signature = inspect.signature(planner_class.__init__)
    accepted = set(signature.parameters)
    accepted.discard("self")
    return {key: value for key, value in kwargs.items() if key in accepted}


def create_planner(
    algorithm: str,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    env: Any,
    args: Optional[Any] = None,
    vehicle_width: Optional[float] = None,
    vehicle_length: Optional[float] = None,
    **overrides: Any,
) -> Any:
    """创建并返回统一适配后的 planner 实例。"""
    planner_class = get_planner_class(algorithm)
    params = get_algorithm_specific_params(algorithm, args)
    params.update(overrides)

    common_params = {
        "start": start,
        "goal": goal,
        "env": env,
        "vehicle_width": vehicle_width,
        "vehicle_length": vehicle_length,
    }

    return planner_class(**normalize_planner_kwargs(planner_class, {**common_params, **params}))
