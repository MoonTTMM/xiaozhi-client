"""
视频处理系统配置文件
提供视频处理相关的配置选项和默认值
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import os


@dataclass
class VideoProcessorConfig:
    """视频处理器配置类"""
    
    # 基础配置
    enabled: bool = True
    continuous_monitoring: bool = True
    auto_start: bool = True
    
    # 视频质量配置
    frame_width: int = 640
    frame_height: int = 480
    frame_rate: int = 30
    quality: int = 80
    
    # 监控配置
    monitoring_interval: float = 1.0  # 监控间隔（秒）
    sensitivity: float = 0.7  # 监控敏感度 (0.0-1.0)
    
    # 事件阈值配置
    motion_threshold: float = 0.8
    color_change_threshold: float = 0.7
    brightness_threshold: float = 0.7
    object_detection_threshold: float = 0.6
    
    # 会话触发配置
    auto_trigger_session: bool = True
    trigger_cooldown: float = 5.0  # 触发冷却时间（秒）
    max_triggers_per_hour: int = 10
    
    # 存储配置
    save_events: bool = True
    event_history_size: int = 100
    storage_path: str = "video_events"
    
    # 分析器配置
    analyzers: Dict[str, bool] = field(default_factory=lambda: {
        "motion": True,
        "color": True,
        "brightness": True,
        "object": False
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'enabled': self.enabled,
            'continuous_monitoring': self.continuous_monitoring,
            'auto_start': self.auto_start,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'frame_rate': self.frame_rate,
            'quality': self.quality,
            'monitoring_interval': self.monitoring_interval,
            'sensitivity': self.sensitivity,
            'motion_threshold': self.motion_threshold,
            'color_change_threshold': self.color_change_threshold,
            'brightness_threshold': self.brightness_threshold,
            'object_detection_threshold': self.object_detection_threshold,
            'auto_trigger_session': self.auto_trigger_session,
            'trigger_cooldown': self.trigger_cooldown,
            'max_triggers_per_hour': self.max_triggers_per_hour,
            'save_events': self.save_events,
            'event_history_size': self.event_history_size,
            'storage_path': self.storage_path,
            'analyzers': self.analyzers.copy()
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key == 'analyzers' and isinstance(value, dict):
                    self.analyzers.update(value)
                else:
                    setattr(self, key, value)
    
    def save_to_file(self, filepath: str) -> None:
        """保存配置到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'VideoProcessorConfig':
        """从文件加载配置"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = cls()
            config.update_from_dict(config_dict)
            return config
        return cls()
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        if not 0 <= self.sensitivity <= 1:
            return False
        if not 0 <= self.motion_threshold <= 1:
            return False
        if not 0 <= self.color_change_threshold <= 1:
            return False
        if not 0 <= self.brightness_threshold <= 1:
            return False
        if not 0 <= self.object_detection_threshold <= 1:
            return False
        if self.frame_width <= 0 or self.frame_height <= 0:
            return False
        if self.frame_rate <= 0:
            return False
        if self.quality < 0 or self.quality > 100:
            return False
        return True
    
    def get_threshold(self, event_type: str) -> float:
        """获取指定事件类型的阈值"""
        threshold_map = {
            'motion': self.motion_threshold,
            'color_change': self.color_change_threshold,
            'brightness': self.brightness_threshold,
            'object_detection': self.object_detection_threshold
        }
        return threshold_map.get(event_type, 0.5)


# 默认配置实例
DEFAULT_CONFIG = VideoProcessorConfig()
