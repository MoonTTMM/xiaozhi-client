"""
视频管理器

整个视频处理系统的核心协调器，管理摄像头、帧处理器和分析器。
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from src.utils.logging_config import get_logger
from .camera_manager import CameraManager, CameraConfig
from .frame_processor import FrameProcessor, ProcessingConfig
from .base_analyzer import BaseVideoAnalyzer
from .video_analyzer import (
    MotionDetector,
    ColorAnalyzer,
    BrightnessAnalyzer,
    ObjectCounter
)
from .video_config import VideoProcessorConfig, DEFAULT_CONFIG

logger = get_logger(__name__)


@dataclass
class VideoManagerConfig:
    """视频管理器配置"""
    camera_config: CameraConfig = None
    processing_config: ProcessingConfig = None
    auto_start: bool = True
    error_callback: Optional[Callable] = None
    # 持续监控相关配置
    continuous_monitoring: bool = True
    monitoring_sensitivity: float = 0.5
    event_thresholds: Dict[str, float] = None
    auto_session_trigger: bool = True
    # 文本输出配置
    text_callback: Optional[Callable] = None


class VideoManager:
    """
    视频管理器
    
    整个视频处理系统的核心协调器，负责：
    1. 管理摄像头设备
    2. 协调帧处理器
    3. 管理分析器
    4. 处理分析结果
    5. 提供系统状态监控
    """
    
    def __init__(self, config: Optional[VideoManagerConfig] = None):
        """
        初始化视频管理器
        
        Args:
            config: 管理器配置，如果为None则使用默认配置
        """
        self.config = config or VideoManagerConfig()
        
        # 组件
        self.camera_manager: Optional[CameraManager] = None
        self.frame_processor: Optional[FrameProcessor] = None
        
        # 状态
        self.is_initialized = False
        self.is_running = False
        self.is_paused = False
        
        # 统计信息
        self.stats = {
            "start_time": 0.0,
            "total_runtime": 0.0,
            "frames_processed": 0,
            "results_generated": 0,
            "errors_count": 0,
            "last_frame_time": 0.0
        }
        
        # 回调函数
        self.error_callbacks: List[Callable] = []
        self.text_callback: Optional[Callable] = None
        
        # 内部状态
        self._processing_task: Optional[asyncio.Task] = None
        self._last_stats_update = time.time()
        self._last_event_time = 0.0  # 初始化为0，确保第一次可以发送
        self._events_detected = 0
        
        # 设置默认事件阈值
        if self.config.event_thresholds is None:
            self.config.event_thresholds = {
                "motion": 0.8,
                "color_change": 0.7,
                "brightness_change": 0.7,
                "object_detected": 0.6,
                "fall_detection": 0.3
            }
        
        logger.info("初始化视频管理器")
    
    async def initialize(self) -> bool:
        """
        初始化视频管理器
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("正在初始化视频管理器...")
            
            # 创建摄像头管理器
            camera_config = self.config.camera_config or CameraConfig()
            self.camera_manager = CameraManager(camera_config)
            
            # 创建帧处理器
            processing_config = self.config.processing_config or ProcessingConfig()
            self.frame_processor = FrameProcessor(processing_config)
            
            # 设置文本回调
            if self.config.text_callback:
                self.text_callback = self.config.text_callback
            
            # 添加默认分析器
            await self._add_default_analyzers()
            
            # 设置回调函数
            if self.config.error_callback:
                self.add_error_callback(self.config.error_callback)
            
            
            self.is_initialized = True
            logger.info("视频管理器初始化完成")
            
            # 自动启动
            if self.config.auto_start:
                await self.start()
            
            return True
            
        except Exception as e:
            logger.error(f"初始化视频管理器失败: {e}")
            return False
    
    async def _add_default_analyzers(self):
        """添加默认分析器"""
        try:
            # 运动检测器
            motion_detector = MotionDetector()
            self.frame_processor.add_analyzer(motion_detector)
            
            # # 颜色分析器
            # color_analyzer = ColorAnalyzer()
            # self.frame_processor.add_analyzer(color_analyzer)
            
            # # 亮度分析器
            # brightness_analyzer = BrightnessAnalyzer()
            # self.frame_processor.add_analyzer(brightness_analyzer)
            
            # # 对象计数器
            # object_counter = ObjectCounter()
            # self.frame_processor.add_analyzer(object_counter)
            
            logger.info("已添加默认分析器")
            
        except Exception as e:
            logger.error(f"添加默认分析器失败: {e}")
    
    async def start(self) -> bool:
        """
        启动视频管理器
        
        Returns:
            启动是否成功
        """
        if not self.is_initialized:
            logger.error("视频管理器未初始化")
            return False
        
        try:
            logger.info("正在启动视频管理器...")
            
            # 连接摄像头
            if not await self.camera_manager.connect():
                raise RuntimeError("无法连接摄像头")
            
            # 启动摄像头流
            await self.camera_manager.start_streaming()
            
            # 启动帧处理器
            await self.frame_processor.start()
            
            # 启动处理循环
            self._processing_task = asyncio.create_task(self._processing_loop())
            
            self.is_running = True
            self.stats["start_time"] = time.time()
            self.stats["last_frame_time"] = time.time()
            
            logger.info("视频管理器启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动视频管理器失败: {e}")
            await self._notify_error_callbacks(str(e))
            return False
    
    async def stop(self):
        """停止视频管理器"""
        try:
            logger.info("正在停止视频管理器...")
            
            self.is_running = False
            
            # 停止处理循环
            if self._processing_task and not self._processing_task.done():
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass
            
            # 停止帧处理器
            if self.frame_processor:
                await self.frame_processor.stop()
            
            # 停止摄像头
            if self.camera_manager:
                await self.camera_manager.stop_streaming()
                await self.camera_manager.disconnect()
            
            # 更新统计信息
            if self.stats["start_time"] > 0:
                self.stats["total_runtime"] += time.time() - self.stats["start_time"]
                self.stats["start_time"] = 0
            
            logger.info("视频管理器已停止")
            
        except Exception as e:
            logger.error(f"停止视频管理器时发生错误: {e}")
    
    async def pause(self):
        """暂停视频处理"""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            logger.info("视频处理已暂停")
    
    async def resume(self):
        """恢复视频处理"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            logger.info("视频处理已恢复")
    
    async def _processing_loop(self):
        """主处理循环"""
        try:
            while self.is_running:
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue
                
                # 获取帧
                frame = await self.camera_manager.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # 处理帧
                results = await self.frame_processor.process_frame(frame)
                
                # 更新统计信息
                self._update_stats(len(results))
                
                # 如果有文本回调，生成文本结果并发送
                if results and self.text_callback:
                    await self._process_text_output(results)
                
                # 控制帧率
                target_fps = self.frame_processor.config.target_fps
                if target_fps > 0:
                    await asyncio.sleep(1.0 / target_fps)
                
        except asyncio.CancelledError:
            logger.info("处理循环被取消")
        except Exception as e:
            logger.error(f"处理循环发生错误: {e}")
            await self._notify_error_callbacks(str(e))
    
    def _update_stats(self, results_count: int):
        """更新统计信息"""
        current_time = time.time()
        
        self.stats["frames_processed"] += 1
        self.stats["results_generated"] += results_count
        self.stats["last_frame_time"] = current_time
        
        # 每秒更新一次统计
        if current_time - self._last_stats_update >= 1.0:
            self._last_stats_update = current_time
            
            # 更新运行时间
            if self.stats["start_time"] > 0:
                self.stats["total_runtime"] = current_time - self.stats["start_time"]
    
    def add_analyzer(self, analyzer: BaseVideoAnalyzer):
        """添加分析器"""
        if self.frame_processor:
            self.frame_processor.add_analyzer(analyzer)
            logger.info(f"添加分析器: {analyzer.name}")
    
    def remove_analyzer(self, analyzer: BaseVideoAnalyzer):
        """移除分析器"""
        if self.frame_processor:
            self.frame_processor.remove_analyzer(analyzer)
            logger.info(f"移除分析器: {analyzer.name}")
    
    def get_analyzers(self) -> List[BaseVideoAnalyzer]:
        """获取所有分析器"""
        if self.frame_processor:
            return self.frame_processor.get_analyzers()
        return []
    
    def get_analyzer(self, name: str) -> Optional[BaseVideoAnalyzer]:
        """根据名称获取分析器"""
        if self.frame_processor:
            return self.frame_processor.get_analyzer(name)
        return None
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调函数"""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
            logger.info(f"添加错误回调函数: {callback}")
    
    def remove_error_callback(self, callback: Callable):
        """移除错误回调函数"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
            logger.info(f"移除错误回调函数: {callback}")
    
    async def _notify_error_callbacks(self, error: str):
        """通知错误回调函数"""
        self.stats["errors_count"] += 1
        
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"错误回调函数执行错误: {e}")
    
    async def _process_text_output(self, results: List[Any]):
        """处理文本输出 - 基于事件条件判断是否发送"""
        try:
            if not self.text_callback or not results:
                return
            
            # 检查冷却时间，防止频繁发送
            current_time = time.time()
            if current_time - self._last_event_time < 2.0:  # 2秒冷却时间
                return
            
            # 检查是否有事件超过阈值
            significant_events = []
            text_results = []
            
            for result in results:
                if hasattr(result, 'analyzer') and hasattr(result.analyzer, 'get_text_result'):
                    # 检查事件是否超过阈值
                    if self._is_significant_event(result):
                        significant_events.append(result)
                        text = result.analyzer.get_text_result(result.data)
                        if text:
                            text_results.append(text)
            
            # 只有当有显著事件时才发送
            if significant_events and text_results:
                # 组合所有文本结果
                combined_text = " | ".join(text_results)
                
                # 更新最后事件时间
                self._last_event_time = current_time
                self._events_detected += 1
                
                logger.info(f"检测到显著事件，发送文本: {combined_text}")
                
                # 调用文本回调函数
                if asyncio.iscoroutinefunction(self.text_callback):
                    await self.text_callback(combined_text)
                else:
                    self.text_callback(combined_text)
            
        except Exception as e:
            logger.error(f"处理文本输出时出错: {e}")
    
    def _is_significant_event(self, result) -> bool:
        """判断是否为显著事件（超过阈值）"""
        try:
            if not hasattr(result, 'analyzer') or not hasattr(result, 'data'):
                return False
            
            analyzer_name = result.analyzer.name
            data = result.data
            
            # 根据分析器类型检查不同的阈值
            if analyzer_name == "motion_detector":
                # 运动检测：检查运动强度、运动区域数量或摔倒检测
                motion_intensity = data.get('motion_intensity', 0)
                motion_regions_count = data.get('motion_regions_count', 0)
                fall_detected = data.get('fall_detected', False)
                fall_confidence = data.get('fall_confidence', 0.0)
                threshold = self.config.event_thresholds.get('motion', 0.8)
                
                # 调试信息
                logger.debug(f"运动检测判断: 强度={motion_intensity}, 区域数={motion_regions_count}, 摔倒={fall_detected}, 置信度={fall_confidence}, 阈值={threshold}")
                
                # 摔倒检测优先级最高
                if fall_detected:
                    logger.info(f"🚨 检测到摔倒事件！置信度: {fall_confidence:.2f}")
                    return True
                
                # 运动强度超过阈值或检测到多个运动区域
                result = motion_intensity > threshold or motion_regions_count >= 2
                logger.debug(f"运动检测结果: {result}")
                return result
                
            elif analyzer_name == "color_analyzer":
                # 颜色分析：检查颜色变化程度
                color_change = data.get('color_change_intensity', 0)
                threshold = self.config.event_thresholds.get('color_change', 0.7)
                return color_change > threshold
                
            elif analyzer_name == "brightness_analyzer":
                # 亮度分析：检查亮度变化
                brightness_change = data.get('brightness_change', 0)
                threshold = self.config.event_thresholds.get('brightness_change', 0.7)
                return brightness_change > threshold
                
            elif analyzer_name == "object_counter":
                # 对象检测：检查对象数量变化
                object_count = data.get('object_count', 0)
                threshold = self.config.event_thresholds.get('object_detected', 0.6)
                return object_count > 0 and object_count >= threshold
                
            else:
                # 未知分析器，默认不发送
                return False
                
        except Exception as e:
            logger.error(f"判断事件显著性时出错: {e}")
            return False
    
    def set_text_callback(self, callback: Optional[Callable]):
        """设置文本回调函数"""
        self.text_callback = callback
        if callback:
            logger.info(f"设置文本回调函数: {callback}")
        else:
            logger.info("清除文本回调函数")
    
    def set_event_thresholds(self, thresholds: Dict[str, float]):
        """设置事件阈值"""
        if self.config.event_thresholds is None:
            self.config.event_thresholds = {}
        self.config.event_thresholds.update(thresholds)
        logger.info(f"更新事件阈值: {thresholds}")
    
    def set_event_threshold(self, event_type: str, threshold: float):
        """设置单个事件类型的阈值"""
        if 0 <= threshold <= 1:
            if self.config.event_thresholds is None:
                self.config.event_thresholds = {}
            self.config.event_thresholds[event_type] = threshold
            logger.info(f"设置 {event_type} 事件阈值为: {threshold}")
        else:
            logger.warning(f"阈值必须在0-1之间: {threshold}")
    
    def get_event_thresholds(self) -> Dict[str, float]:
        """获取当前事件阈值"""
        return self.config.event_thresholds.copy()
    
    def get_events_detected_count(self) -> int:
        """获取检测到的事件数量"""
        return self._events_detected
    
    def reset_events_count(self):
        """重置事件计数"""
        self._events_detected = 0
        logger.info("重置事件计数")
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "paused": self.is_paused,
            "stats": self.stats.copy(),
            "event_thresholds": self.get_event_thresholds(),
            "events_detected": self._events_detected,
            "last_event_time": self._last_event_time
        }
        
        # 添加摄像头状态
        if self.camera_manager:
            status["camera"] = self.camera_manager.get_status()
        
        # 添加帧处理器状态
        if self.frame_processor:
            status["status"] = self.frame_processor.get_status()
        
        return status
    
    def get_config(self) -> VideoManagerConfig:
        """获取当前配置"""
        return self.config
    
    def update_config(self, config: VideoManagerConfig):
        """更新配置"""
        self.config = config
        logger.info(f"更新视频管理器配置: {config}")
        
        # 更新摄像头配置
        if self.camera_manager and config.camera_config:
            self.camera_manager.update_config(config.camera_config)
        
        # 更新帧处理器配置
        if self.frame_processor and config.processing_config:
            self.frame_processor.update_config(config.processing_config)
    
    def get_cached_results(self, limit: Optional[int] = None) -> List[Any]:
        """获取缓存的分析结果"""
        if self.frame_processor:
            return self.frame_processor.get_cached_results(limit)
        return []
    
    def clear_cache(self):
        """清空缓存"""
        if self.frame_processor:
            self.frame_processor.clear_cache()
    
    def enable_continuous_monitoring(self, enabled: bool = True):
        """启用或禁用持续监控模式"""
        if enabled:
            self.is_paused = False
            logger.info("持续监控模式已启用")
        else:
            self.is_paused = True
            logger.info("持续监控模式已暂停")
    
    def set_monitoring_sensitivity(self, sensitivity: float):
        """设置监控敏感度 (0.0 - 1.0)"""
        if 0.0 <= sensitivity <= 1.0:
            self.config.monitoring_sensitivity = sensitivity
            logger.info(f"监控敏感度设置为: {sensitivity}")
        else:
            logger.warning(f"无效的敏感度值: {sensitivity}，应在0.0-1.0之间")
    
    def set_event_thresholds(self, thresholds: Dict[str, float]):
        """设置事件触发阈值"""
        for event_type, threshold in thresholds.items():
            if 0.0 <= threshold <= 1.0:
                self.config.event_thresholds[event_type] = 0.5
                logger.info(f"事件 {event_type} 阈值设置为: {threshold}")
            else:
                logger.warning(f"事件 {event_type} 阈值无效: {threshold}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "continuous_monitoring": not self.is_paused,
            "sensitivity": getattr(self.config, 'monitoring_sensitivity', 0.5),
            "event_thresholds": getattr(self.config, 'event_thresholds', {}),
            "last_event_time": getattr(self, '_last_event_time', 0.0),
            "events_detected": getattr(self, '_events_detected', 0)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "start_time": 0.0,
            "total_runtime": 0.0,
            "frames_processed": 0,
            "results_generated": 0,
            "errors_count": 0,
            "last_frame_time": 0.0
        }
        
        if self.frame_processor:
            self.frame_processor.reset_stats()
        
        logger.info("重置视频管理器统计")
    
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()
