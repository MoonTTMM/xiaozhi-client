"""
视频处理模块

提供可扩展的视频分析系统，支持多种视频分析方法。
"""

from .video_manager import VideoManager, VideoManagerConfig
from .camera_manager import CameraManager
from .frame_processor import FrameProcessor
from .base_analyzer import BaseVideoAnalyzer
from .video_analyzer import (
    MotionDetector,
    ColorAnalyzer,
    BrightnessAnalyzer,
    ObjectCounter
)
from .video_config import VideoProcessorConfig, DEFAULT_CONFIG

__all__ = [
    'VideoManager',
    'VideoManagerConfig',
    'CameraManager', 
    'FrameProcessor',
    'BaseVideoAnalyzer',
    'MotionDetector',
    'ColorAnalyzer',
    'BrightnessAnalyzer',
    'ObjectCounter',
    'VideoProcessorConfig',
    'DEFAULT_CONFIG'
]

__version__ = "1.0.0"
