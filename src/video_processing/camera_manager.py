"""
摄像头管理器

负责管理摄像头设备的连接、配置和帧获取。
"""

import asyncio
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1
    auto_exposure: bool = True
    exposure_time: Optional[float] = None
    gain: Optional[float] = None
    white_balance: Optional[str] = None
    codec: str = "MJPG"
    timeout: float = 5.0


class CameraManager:
    """
    摄像头管理器
    
    负责管理摄像头设备的连接、配置和帧获取。
    支持多种摄像头类型和配置选项。
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        初始化摄像头管理器
        
        Args:
            config: 摄像头配置，如果为None则使用默认配置
        """
        self.config = config or CameraConfig()
        self.camera = None
        self.is_connected = False
        self.is_streaming = False
        
        # 状态信息
        self.status = {
            "connected": False,
            "streaming": False,
            "frame_count": 0,
            "fps": 0.0,
            "last_frame_time": 0.0,
            "errors": []
        }
        
        # 回调函数
        self.frame_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # 内部状态
        self._last_fps_update = time.time()
        self._frame_times = []
        
        logger.info(f"初始化摄像头管理器: 设备ID={self.config.device_id}")
    
    async def connect(self) -> bool:
        """
        连接到摄像头
        
        Returns:
            连接是否成功
        """
        try:
            logger.info(f"正在连接摄像头: 设备ID={self.config.device_id}")
            
            # 创建摄像头对象
            self.camera = cv2.VideoCapture(self.config.device_id)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"无法打开摄像头设备 {self.config.device_id}")
            
            # 设置摄像头参数
            await self._configure_camera()
            
            # 测试读取一帧
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("无法从摄像头读取帧")
            
            self.is_connected = True
            self.status["connected"] = True
            self.status["last_frame_time"] = time.time()
            
            logger.info(f"摄像头连接成功: 设备ID={self.config.device_id}")
            return True
            
        except Exception as e:
            error_msg = f"连接摄像头失败: {e}"
            logger.error(error_msg)
            self._add_error(error_msg)
            return False
    
    async def disconnect(self):
        """断开摄像头连接"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            self.is_connected = False
            self.is_streaming = False
            self.status["connected"] = False
            self.status["streaming"] = False
            
            logger.info("摄像头已断开连接")
            
        except Exception as e:
            logger.error(f"断开摄像头连接时发生错误: {e}")
    
    async def _configure_camera(self):
        """配置摄像头参数"""
        try:
            # 设置分辨率
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # 设置帧率
            self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 设置缓冲区大小
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # 设置自动曝光
            if self.config.auto_exposure:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 自动曝光
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 手动曝光
                if self.config.exposure_time is not None:
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure_time)
            
            # 设置增益
            if self.config.gain is not None:
                self.camera.set(cv2.CAP_PROP_GAIN, self.config.gain)
            
            # 设置白平衡
            if self.config.white_balance is not None:
                if self.config.white_balance == "auto":
                    self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
                else:
                    self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)
                    # 这里可以添加具体的白平衡设置
            
            # 设置编解码器
            if self.config.codec:
                fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            logger.info(f"摄像头配置完成: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            
        except Exception as e:
            logger.warning(f"配置摄像头参数时发生警告: {e}")
    
    async def start_streaming(self):
        """开始视频流"""
        if not self.is_connected:
            logger.error("摄像头未连接，无法开始视频流")
            return
        
        self.is_streaming = True
        self.status["streaming"] = True
        logger.info("开始视频流")
    
    async def stop_streaming(self):
        """停止视频流"""
        self.is_streaming = False
        self.status["streaming"] = False
        logger.info("停止视频流")
    
    async def get_frame(self) -> Optional[np.ndarray]:
        """
        获取一帧图像
        
        Returns:
            图像帧（numpy数组），如果获取失败返回None
        """
        if not self.is_connected or not self.is_streaming:
            return None
        
        try:
            # 设置超时
            start_time = time.time()
            while time.time() - start_time < self.config.timeout:
                ret, frame = self.camera.read()
                if ret:
                    # 更新统计信息
                    self._update_frame_stats()
                    
                    # 调用回调函数
                    await self._notify_frame_callbacks(frame)
                    
                    return frame
                
                await asyncio.sleep(0.001)  # 短暂等待
            
            # 超时
            logger.warning("获取帧超时")
            return None
            
        except Exception as e:
            error_msg = f"获取帧时发生错误: {e}"
            logger.error(error_msg)
            self._add_error(error_msg)
            return None
    
    def _update_frame_stats(self):
        """更新帧统计信息"""
        current_time = time.time()
        
        # 更新帧计数
        self.status["frame_count"] += 1
        
        # 更新FPS
        self._frame_times.append(current_time)
        
        # 保持最近100帧的时间记录
        if len(self._frame_times) > 100:
            self._frame_times = self._frame_times[-100:]
        
        # 计算FPS（每秒更新一次）
        if current_time - self._last_fps_update >= 1.0:
            if len(self._frame_times) >= 2:
                time_diff = self._frame_times[-1] - self._frame_times[0]
                if time_diff > 0:
                    self.status["fps"] = (len(self._frame_times) - 1) / time_diff
            
            self._last_fps_update = current_time
        
        self.status["last_frame_time"] = current_time
    
    def _add_error(self, error_msg: str):
        """添加错误信息"""
        self.status["errors"].append({
            "timestamp": time.time(),
            "error": error_msg
        })
        
        # 保持错误列表在合理范围内
        if len(self.status["errors"]) > 100:
            self.status["errors"] = self.status["errors"][-50:]
    
    async def _notify_frame_callbacks(self, frame: np.ndarray):
        """通知帧回调函数"""
        for callback in self.frame_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(frame)
                else:
                    callback(frame)
            except Exception as e:
                logger.error(f"帧回调函数执行错误: {e}")
    
    async def _notify_error_callbacks(self, error: str):
        """通知错误回调函数"""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"错误回调函数执行错误: {e}")
    
    def add_frame_callback(self, callback: Callable):
        """添加帧回调函数"""
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)
            logger.info(f"添加帧回调函数: {callback}")
    
    def remove_frame_callback(self, callback: Callable):
        """移除帧回调函数"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
            logger.info(f"移除帧回调函数: {callback}")
    
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
    
    def get_status(self) -> Dict[str, Any]:
        """获取摄像头状态"""
        return self.status.copy()
    
    def get_config(self) -> CameraConfig:
        """获取当前配置"""
        return self.config
    
    def update_config(self, config: CameraConfig):
        """更新配置"""
        self.config = config
        logger.info(f"更新摄像头配置: {config}")
        
        # 如果已连接，重新配置
        if self.is_connected:
            asyncio.create_task(self._configure_camera())
    
    def is_available(self) -> bool:
        """检查摄像头是否可用"""
        try:
            temp_camera = cv2.VideoCapture(self.config.device_id)
            if temp_camera.isOpened():
                temp_camera.release()
                return True
            return False
        except:
            return False
    
    def get_available_cameras(self) -> List[int]:
        """获取可用的摄像头设备ID列表"""
        available_cameras = []
        
        for i in range(10):  # 检查前10个设备ID
            try:
                temp_camera = cv2.VideoCapture(i)
                if temp_camera.isOpened():
                    available_cameras.append(i)
                    temp_camera.release()
            except:
                continue
        
        return available_cameras
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
    
    def __del__(self):
        """析构函数"""
        if self.camera is not None:
            self.camera.release()
