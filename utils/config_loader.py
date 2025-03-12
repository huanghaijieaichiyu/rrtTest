#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置加载器模块

提供配置文件的加载和管理功能
"""

import os
import yaml
from typing import Any, Dict


class ConfigLoader:
    """配置加载器类"""

    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """
        初始化配置加载器

        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"配置文件格式错误: {e}")

    def get_environment_config(self) -> Dict[str, Any]:
        """获取环境配置"""
        return self.config.get('environment', {})

    def get_vehicle_config(self) -> Dict[str, Any]:
        """获取车辆配置"""
        return self.config.get('vehicle', {})

    def get_parking_lot_config(self) -> Dict[str, Any]:
        """获取停车场布局配置"""
        return self.config.get('parking_lot', {})

    def get_path_planning_config(self) -> Dict[str, Any]:
        """获取路径规划配置"""
        return self.config.get('path_planning', {})

    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """获取特定算法的配置"""
        algorithms = self.config.get('path_planning', {}).get('algorithms', {})
        return algorithms.get(algorithm, {})

    def get_path_following_config(self) -> Dict[str, Any]:
        """获取路径跟踪配置"""
        return self.config.get('path_following', {})

    def get_control_method_config(self, method: str) -> Dict[str, Any]:
        """获取特定控制方法的配置"""
        control_methods = self.config.get(
            'path_following', {}).get('control_methods', {})
        return control_methods.get(method, {})

    def get_display_config(self) -> Dict[str, Any]:
        """获取显示配置"""
        return self.config.get('display', {})

    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置"""
        return self.config.get('simulation', {})

    def get_color(self, category: str, subcategory: str = None) -> tuple:
        """
        获取颜色配置

        参数:
            category: 颜色类别
            subcategory: 子类别（可选）

        返回:
            RGB颜色元组
        """
        colors = self.config.get('display', {}).get('colors', {})
        if subcategory:
            color = colors.get(category, {}).get(subcategory, [0, 0, 0])
        else:
            color = colors.get(category, [0, 0, 0])
        return tuple(color)

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        保存配置到文件

        参数:
            config: 配置字典
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置

        参数:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        self.save_config(self.config)
