"""
帧处理器

负责协调多个视频分析器，处理视频帧并管理分析结果。
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from src.utils.logging_config import get_logger
from .base_analyzer import BaseVideoAnalyzer, AnalysisResult

logger = get_logger(__name__)


@dataclass
class ProcessingConfig:
    """处理配置"""
    target_fps: int = 10
    frame_skip: int = 0  # 跳过的帧数
    max_concurrent_analyzers: int = 4
    result_buffer_size: int = 100
    enable_parallel_processing: bool = True
    timeout_per_frame: float = 5.0


class FrameProcessor:
    """
    帧处理器
    
    负责协调多个视频分析器，处理视频帧并管理分析结果。
    支持并行处理、帧跳过、结果缓存等功能。
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        初始化帧处理器
        
        Args:
            config: 处理配置，如果为None则使用默认配置
        """
        self.config = config or ProcessingConfig()
        self.analyzers: List[BaseVideoAnalyzer] = []
        self.is_processing = False
        
        # 结果管理
        self.result_buffer: List[AnalysisResult] = []
        self.result_callbacks: List[Callable] = []
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "skipped_frames": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_frame_time": 0.0,
            "analyzer_stats": {},
            "errors": []
        }
        
        # 内部状态
        self._frame_counter = 0
        self._last_stats_update = time.time()
        self._processing_tasks = set()
        
        logger.info(f"初始化帧处理器: 目标FPS={self.config.target_fps}")
    
    def add_analyzer(self, analyzer: BaseVideoAnalyzer):
        """
        添加分析器
        
        Args:
            analyzer: 要添加的分析器
        """
        if analyzer not in self.analyzers:
            self.analyzers.append(analyzer)
            logger.info(f"添加分析器: {analyzer.name}")
            
            # 初始化统计信息
            self.stats["analyzer_stats"][analyzer.name] = analyzer.get_stats()
    
    def remove_analyzer(self, analyzer: BaseVideoAnalyzer):
        """
        移除分析器
        
        Args:
            analyzer: 要移除的分析器
        """
        if analyzer in self.analyzers:
            self.analyzers.remove(analyzer)
            logger.info(f"移除分析器: {analyzer.name}")
            
            # 清理统计信息
            if analyzer.name in self.stats["analyzer_stats"]:
                del self.stats["analyzer_stats"][analyzer.name]
    
    def get_analyzers(self) -> List[BaseVideoAnalyzer]:
        """获取所有分析器"""
        return self.analyzers.copy()
    
    def get_analyzer(self, name: str) -> Optional[BaseVideoAnalyzer]:
        """
        根据名称获取分析器
        
        Args:
            name: 分析器名称
            
        Returns:
            分析器对象，如果不存在返回None
        """
        for analyzer in self.analyzers:
            if analyzer.name == name:
                return analyzer
        return None
    
    async def process_frame(self, frame, frame_info: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            frame_info: 帧信息（可选）
            
        Returns:
            分析结果列表
        """
        if not self.is_processing or not self.analyzers:
            return []
        
        # 检查是否需要跳过帧
        if self.config.frame_skip > 0 and self._frame_counter % (self.config.frame_skip + 1) != 0:
            self.stats["skipped_frames"] += 1
            self._frame_counter += 1
            return []
        
        start_time = time.time()
        self._frame_counter += 1
        
        try:
            # 获取启用的分析器
            enabled_analyzers = [a for a in self.analyzers if a.is_enabled()]
            
            if not enabled_analyzers:
                return []
            
            # 处理帧
            if self.config.enable_parallel_processing:
                results = await self._process_frame_parallel(frame, enabled_analyzers, frame_info)
            else:
                results = await self._process_frame_sequential(frame, enabled_analyzers, frame_info)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time, len(results))
            
            # 缓存结果
            self._cache_results(results)
            
            # 调用回调函数
            await self._notify_result_callbacks(results)
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time, 0, str(e))
            logger.error(f"处理帧时发生错误: {e}")
            return []
    
    def add_result_callback(self, callback: Callable):
        """添加结果回调函数"""
        if callback not in self.result_callbacks:
            self.result_callbacks.append(callback)
            logger.info(f"添加结果回调函数: {callback}")
    
    def remove_result_callback(self, callback: Callable):
        """移除结果回调函数"""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
            logger.info(f"移除结果回调函数: {callback}")
    
    async def _notify_result_callbacks(self, results: List[AnalysisResult]):
        """通知结果回调函数"""
        for callback in self.result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(results)
                else:
                    callback(results)
            except Exception as e:
                logger.error(f"结果回调函数执行错误: {e}")
    
    async def start(self):
        """启动处理器"""
        self.is_processing = True
        
        # 启动所有分析器
        for analyzer in self.analyzers:
            await analyzer.start()
        
        logger.info("帧处理器已启动")
    
    async def stop(self):
        """停止处理器"""
        self.is_processing = False
        
        # 停止所有分析器
        for analyzer in self.analyzers:
            await analyzer.stop()
        
        # 取消所有处理任务
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()
        
        logger.info("帧处理器已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算成功率
        if stats["total_frames"] > 0:
            stats["success_rate"] = stats["processed_frames"] / stats["total_frames"]
            stats["skip_rate"] = stats["skipped_frames"] / stats["total_frames"]
        else:
            stats["success_rate"] = 0.0
            stats["skip_rate"] = 0.0
        
        # 计算错误率
        if stats["total_frames"] > 0:
            stats["error_rate"] = len(stats["errors"]) / stats["total_frames"]
        else:
            stats["error_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "skipped_frames": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "last_frame_time": 0.0,
            "analyzer_stats": {},
            "errors": []
        }
        
        # 重置分析器统计
        for analyzer in self.analyzers:
            analyzer.reset_stats()
            self.stats["analyzer_stats"][analyzer.name] = analyzer.get_stats()
        
        logger.info("重置帧处理器统计")
    
    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        return {
            "processing": self.is_processing,
            "analyzers_count": len(self.analyzers),
            "enabled_analyzers_count": len([a for a in self.analyzers if a.is_enabled()]),
            "cached_results_count": len(self.result_buffer),
            "active_tasks_count": len(self._processing_tasks),
            "stats": self.get_stats()
        }
    
    def update_config(self, config: ProcessingConfig):
        """更新配置"""
        self.config = config
        logger.info(f"更新帧处理器配置: {config}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop()
    
    def _update_stats(self, success: bool, processing_time: float, result_count: int, error: Optional[str] = None):
        """更新统计信息"""
        self.stats["total_frames"] += 1
        
        if success:
            self.stats["processed_frames"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["processed_frames"]
            )
            self.stats["last_frame_time"] = time.time()
        else:
            if error:
                self.stats["errors"].append({
                    "timestamp": time.time(),
                    "error": error
                })
                # 保持错误列表在合理范围内
                if len(self.stats["errors"]) > 100:
                    self.stats["errors"] = self.stats["errors"][-50:]
        
        # 更新分析器统计信息
        for analyzer in self.analyzers:
            if analyzer.name in self.stats["analyzer_stats"]:
                self.stats["analyzer_stats"][analyzer.name] = analyzer.get_stats()
    
    def _cache_results(self, results: List[AnalysisResult]):
        """缓存分析结果"""
        self.result_buffer.extend(results)
        
        # 保持缓冲区大小
        if len(self.result_buffer) > self.config.result_buffer_size:
            self.result_buffer = self.result_buffer[-self.config.result_buffer_size:]
    
    def get_cached_results(self, limit: Optional[int] = None) -> List[AnalysisResult]:
        """
        获取缓存的分析结果
        
        Args:
            limit: 限制返回的结果数量，如果为None则返回所有结果
            
        Returns:
            分析结果列表
        """
        if limit is None:
            return self.result_buffer.copy()
        else:
            return self.result_buffer[-limit:]
    
    def clear_cache(self):
        """清空缓存"""
        self.result_buffer.clear()
        logger.info("清空结果缓存")
    
    async def _process_frame_parallel(self, frame, analyzers: List[BaseVideoAnalyzer], frame_info: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """并行处理帧"""
        tasks = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_analyzers)
        
        async def process_with_semaphore(analyzer):
            async with semaphore:
                return await analyzer.analyze_with_pipeline(frame, frame_info)
        
        # 创建所有分析任务
        for analyzer in analyzers:
            task = asyncio.create_task(process_with_semaphore(analyzer))
            tasks.append(task)
            self._processing_tasks.add(task)
            task.add_done_callback(self._processing_tasks.discard)
        
        # 等待所有任务完成
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_per_frame
            )
        except asyncio.TimeoutError:
            logger.warning("帧处理超时")
            # 取消未完成的任务
            for task in tasks:
                if not task.done():
                    task.cancel()
            return []
        
        # 处理结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"分析器 {analyzers[i].name} 处理失败: {result}")
            elif result is not None:
                valid_results.append(result)
        
        return valid_results
    
    async def _process_frame_sequential(self, frame, analyzers: List[BaseVideoAnalyzer], frame_info: Optional[Dict[str, Any]] = None) -> List[AnalysisResult]:
        """顺序处理帧"""
        results = []
        
        for analyzer in analyzers:
            try:
                result = await analyzer.analyze_with_pipeline(frame, frame_info)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"分析器 {analyzer.name} 处理失败: {e}")
        
        return results
