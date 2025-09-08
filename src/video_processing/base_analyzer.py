"""
视频分析器基类

提供所有视频分析器的基础功能和接口。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
    analyzer_name: str
    analysis_type: str
    timestamp: float
    data: Dict[str, Any]
    processing_time: float
    frame_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVideoAnalyzer(ABC):
    """
    视频分析器基类
    
    所有视频分析器都应该继承此类并实现必要的方法。
    """
    
    def __init__(self, name: str, enabled: bool = True):
        """
        初始化分析器
        
        Args:
            name: 分析器名称
            enabled: 是否启用
        """
        self.name = name
        self.enabled = enabled
        self.analysis_type = self.get_analysis_type()
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "failed_frames": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_processed_time": 0.0,
            "errors": []
        }
        
        # 配置参数
        self.config = {}
        
        # 状态
        self.is_running = False
        self.last_frame_time = 0.0
        
        logger.info(f"初始化分析器: {self.name} ({self.analysis_type})")
    
    @abstractmethod
    async def analyze_frame(self, frame) -> Optional[Dict[str, Any]]:
        """
        分析单帧图像
        
        Args:
            frame: 输入帧（numpy数组）
            
        Returns:
            分析结果字典，如果分析失败返回None
        """
        pass
    
    @abstractmethod
    def get_analysis_type(self) -> str:
        """
        获取分析类型标识
        
        Returns:
            分析类型字符串
        """
        pass
    
    def get_name(self) -> str:
        """获取分析器名称"""
        return self.name
    
    def is_enabled(self) -> bool:
        """检查分析器是否启用"""
        return self.enabled
    
    def enable(self):
        """启用分析器"""
        self.enabled = True
        logger.info(f"启用分析器: {self.name}")
    
    def disable(self):
        """禁用分析器"""
        self.enabled = False
        logger.info(f"禁用分析器: {self.name}")
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新配置参数
        
        Args:
            config: 新的配置字典
        """
        self.config.update(config)
        logger.info(f"更新分析器配置: {self.name}, 配置: {config}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    async def analyze_with_pipeline(self, frame, frame_info: Optional[Dict[str, Any]] = None) -> Optional[AnalysisResult]:
        """
        使用分析管道处理帧
        
        Args:
            frame: 输入帧
            frame_info: 帧信息（可选）
            
        Returns:
            分析结果对象
        """
        if not self.enabled:
            return None
        
        start_time = time.time()
        frame_info = frame_info or {}
        
        try:
            # 执行分析
            result_data = await self.analyze_frame(frame)
            
            if result_data is not None:
                # 分析成功
                processing_time = time.time() - start_time
                
                # 更新统计信息
                self._update_stats(True, processing_time)
                
                # 创建结果对象
                result = AnalysisResult(
                    analyzer_name=self.name,
                    analysis_type=self.analysis_type,
                    timestamp=start_time,
                    data=result_data,
                    processing_time=processing_time,
                    frame_info=frame_info
                )
                
                return result
            else:
                # 分析失败
                self._update_stats(False, 0.0)
                return None
                
        except Exception as e:
            # 处理异常
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time, str(e))
            logger.error(f"分析器 {self.name} 处理帧时发生错误: {e}")
            return None
    
    def _update_stats(self, success: bool, processing_time: float, error: Optional[str] = None):
        """更新统计信息"""
        self.stats["total_frames"] += 1
        
        if success:
            self.stats["processed_frames"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["processed_frames"]
            )
            self.stats["last_processed_time"] = time.time()
        else:
            self.stats["failed_frames"] += 1
            if error:
                self.stats["errors"].append({
                    "timestamp": time.time(),
                    "error": error
                })
                # 保持错误列表在合理范围内
                if len(self.stats["errors"]) > 100:
                    self.stats["errors"] = self.stats["errors"][-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_frames"] > 0:
            stats["success_rate"] = stats["processed_frames"] / stats["total_frames"]
        else:
            stats["success_rate"] = 0.0
        
        # 计算错误率
        if stats["total_frames"] > 0:
            stats["error_rate"] = stats["failed_frames"] / stats["total_frames"]
        else:
            stats["error_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "failed_frames": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_processed_time": 0.0,
            "errors": []
        }
        logger.info(f"重置分析器统计: {self.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取分析器状态"""
        return {
            "name": self.name,
            "type": self.analysis_type,
            "enabled": self.enabled,
            "running": self.is_running,
            "last_frame_time": self.last_frame_time,
            "stats": self.get_stats()
        }
    
    async def start(self):
        """启动分析器"""
        self.is_running = True
        logger.info(f"启动分析器: {self.name}")
    
    async def stop(self):
        """停止分析器"""
        self.is_running = False
        logger.info(f"停止分析器: {self.name}")
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self):
        return self.__str__()
