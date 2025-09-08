"""
具体视频分析器实现

提供多种预定义的视频分析方法，包括运动检测、颜色分析、亮度分析、对象计数等。
直接返回文本格式的分析结果，适用于向服务端发送。
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from .base_analyzer import BaseVideoAnalyzer

# 运动检测器
class MotionDetector(BaseVideoAnalyzer):
    """运动检测器"""
    
    def __init__(self, threshold: float = 25.0, min_area: int = 500, enable_fall_detection: bool = True):
        """
        初始化运动检测器
        
        Args:
            threshold: 运动检测阈值
            min_area: 最小运动区域面积
            enable_fall_detection: 是否启用摔倒检测
        """
        super().__init__("motion_detector")
        
        self.threshold = threshold
        self.min_area = min_area
        self.enable_fall_detection = enable_fall_detection
        
        # 背景减法器
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=threshold,
            detectShadows=False
        )
        
        # 形态学操作核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 摔倒检测相关变量
        self.fall_detection_history = []  # 存储历史帧的运动信息
        self.max_history_frames = 15  # 保留最近15帧的历史
        self.fall_detection_threshold = 0.3  # 摔倒检测阈值（降低阈值提高敏感度）
        self.last_fall_time = 0  # 上次检测到摔倒的时间
        self.fall_cooldown = 5.0  # 摔倒检测冷却时间（秒）
        self.min_motion_for_fall = 0.01  # 摔倒检测的最小运动强度要求（降低阈值）
        
        # 配置参数
        self.config.update({
            "threshold": threshold,
            "min_area": min_area,
            "enable_fall_detection": enable_fall_detection,
            "fall_detection_threshold": self.fall_detection_threshold,
            "fall_cooldown": self.fall_cooldown,
            "min_motion_for_fall": self.min_motion_for_fall
        })
    
    async def analyze_frame(self, frame) -> Optional[Dict[str, Any]]:
        """分析帧中的运动"""
        try:
            # 应用背景减法
            fg_mask = self.background_subtractor.apply(frame)
            
            # 形态学操作
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤小区域
            motion_regions = []
            total_motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area)
                    })
                    total_motion_area += area
            
            # 计算运动强度
            frame_area = frame.shape[0] * frame.shape[1]
            motion_intensity = total_motion_area / frame_area if frame_area > 0 else 0
            
            # 基础运动检测结果
            motion_result = {
                "motion_detected": len(motion_regions) > 0,
                "motion_regions_count": len(motion_regions),
                "motion_regions": motion_regions,
                "total_motion_area": int(total_motion_area),
                "motion_intensity": float(motion_intensity),
                "threshold": self.threshold,
                "min_area": self.min_area
            }
            
            # 摔倒检测
            if self.enable_fall_detection:
                fall_detection_result = self._detect_fall(frame, motion_regions, motion_intensity)
                motion_result.update(fall_detection_result)
            
            return motion_result
            
        except Exception as e:
            self._add_error(str(e))
            return None
    
    def _detect_fall(self, frame, motion_regions: List[Dict], motion_intensity: float) -> Dict[str, Any]:
        """
        检测摔倒事件
        
        Args:
            frame: 当前帧
            motion_regions: 运动区域列表
            motion_intensity: 运动强度
            
        Returns:
            包含摔倒检测结果的字典
        """
        current_time = time.time()
        
        # 检查冷却时间
        if current_time - self.last_fall_time < self.fall_cooldown:
            return {
                "fall_detected": False,
                "fall_confidence": 0.0,
                "fall_reason": "cooldown"
            }
        
        # 创建当前帧的运动信息
        current_motion_info = {
            "timestamp": current_time,
            "motion_regions": motion_regions,
            "motion_intensity": motion_intensity,
            "frame_height": frame.shape[0],
            "frame_width": frame.shape[1]
        }
        
        # 添加到历史记录
        self.fall_detection_history.append(current_motion_info)
        
        # 保持历史记录长度
        if len(self.fall_detection_history) > self.max_history_frames:
            self.fall_detection_history.pop(0)
        
        # 需要至少3帧历史才能进行摔倒检测
        if len(self.fall_detection_history) < 3:
            return {
                "fall_detected": False,
                "fall_confidence": 0.0,
                "fall_reason": "insufficient_history"
            }
        
        # 检查是否有足够的运动强度（降低要求）
        if motion_intensity < self.min_motion_for_fall and len(motion_regions) == 0:
            return {
                "fall_detected": False,
                "fall_confidence": 0.0,
                "fall_reason": "insufficient_motion"
            }
        
        # 分析运动模式
        fall_confidence = self._analyze_fall_pattern()
        
        # 判断是否摔倒
        fall_detected = fall_confidence > self.fall_detection_threshold
        
        if fall_detected:
            self.last_fall_time = current_time
            return {
                "fall_detected": True,
                "fall_confidence": fall_confidence,
                "fall_reason": "pattern_analysis"
            }
        else:
            return {
                "fall_detected": False,
                "fall_confidence": fall_confidence,
                "fall_reason": "below_threshold"
            }
    
    def _analyze_fall_pattern(self) -> float:
        """
        分析运动模式，判断是否可能发生摔倒
        
        Returns:
            摔倒置信度 (0.0-1.0)
        """
        if len(self.fall_detection_history) < 3:
            return 0.0
        
        confidence = 0.0
        
        # 1. 分析运动强度变化模式
        motion_intensities = [info["motion_intensity"] for info in self.fall_detection_history]
        
        # 检查是否有突然的运动强度增加（摔倒时的快速运动）
        if len(motion_intensities) >= 3:
            # 分析最近3帧的运动强度变化
            recent_intensities = motion_intensities[-3:]
            max_intensity = max(recent_intensities)
            min_intensity = min(recent_intensities)
            intensity_range = max_intensity - min_intensity
            
            # 如果运动强度变化范围较大，可能是摔倒
            if intensity_range > 0.05:  # 运动强度变化超过5%
                confidence += 0.3
        
        # 2. 分析运动区域的位置变化
        if len(self.fall_detection_history) >= 3:
            # 检查运动区域是否从高处移动到低处（垂直下降）
            vertical_movement_score = self._analyze_vertical_movement()
            confidence += vertical_movement_score * 0.4
        
        # 3. 分析运动区域的形状变化
        if len(self.fall_detection_history) >= 2:
            shape_change_score = self._analyze_shape_change()
            confidence += shape_change_score * 0.3
        
        return min(confidence, 1.0)
    
    def _analyze_vertical_movement(self) -> float:
        """
        分析垂直运动模式
        
        Returns:
            垂直运动得分 (0.0-1.0)
        """
        if len(self.fall_detection_history) < 3:
            return 0.0
        
        # 计算运动区域的中心点位置变化
        center_positions = []
        
        for info in self.fall_detection_history[-3:]:
            if info["motion_regions"]:
                # 找到最大的运动区域
                largest_region = max(info["motion_regions"], key=lambda r: r["area"])
                center_y = largest_region["y"] + largest_region["height"] // 2
                center_positions.append(center_y)
            else:
                center_positions.append(None)
        
        # 过滤掉None值
        valid_positions = [pos for pos in center_positions if pos is not None]
        
        if len(valid_positions) < 2:
            return 0.0
        
        # 检查是否向下移动
        frame_height = self.fall_detection_history[-1]["frame_height"]
        normalized_positions = [pos / frame_height for pos in valid_positions]
        
        # 计算位置变化趋势
        if len(normalized_positions) >= 2:
            # 计算位置变化的线性趋势
            start_pos = normalized_positions[0]
            end_pos = normalized_positions[-1]
            position_change = end_pos - start_pos
            
            # 如果向下移动超过阈值，认为是摔倒
            if position_change > 0.1:  # 向下移动超过10%的帧高度
                return min(position_change * 2.0, 1.0)  # 放大变化量，最大为1.0
        
        return 0.0
    
    def _analyze_shape_change(self) -> float:
        """
        分析运动区域形状变化
        
        Returns:
            形状变化得分 (0.0-1.0)
        """
        if len(self.fall_detection_history) < 2:
            return 0.0
        
        # 比较最近两帧的运动区域形状
        current_info = self.fall_detection_history[-1]
        previous_info = self.fall_detection_history[-2]
        
        if not current_info["motion_regions"] or not previous_info["motion_regions"]:
            return 0.0
        
        # 找到最大的运动区域
        current_largest = max(current_info["motion_regions"], key=lambda r: r["area"])
        previous_largest = max(previous_info["motion_regions"], key=lambda r: r["area"])
        
        # 计算宽高比变化
        current_ratio = current_largest["width"] / current_largest["height"] if current_largest["height"] > 0 else 1
        previous_ratio = previous_largest["width"] / previous_largest["height"] if previous_largest["height"] > 0 else 1
        
        # 摔倒时，人体从站立（高瘦）变为躺下（宽扁），宽高比会发生变化
        ratio_change = abs(current_ratio - previous_ratio)
        
        # 如果宽高比变化较大，可能是摔倒
        if ratio_change > 0.3:
            return min(ratio_change, 1.0)
        
        return 0.0

    def get_analysis_type(self) -> str:
        return "motion_detection"
    
    def get_text_result(self, analysis_data: Dict[str, Any]) -> str:
        """返回文本格式的分析结果"""
        if not analysis_data:
            return "运动检测: 无数据"
        
        motion_detected = analysis_data.get("motion_detected", False)
        regions_count = analysis_data.get("motion_regions_count", 0)
        intensity = analysis_data.get("motion_intensity", 0)
        
        # 检查摔倒检测结果
        fall_detected = analysis_data.get("fall_detected", False)
        fall_confidence = analysis_data.get("fall_confidence", 0.0)
        
        # 构建基础运动检测文本
        if motion_detected:
            motion_text = f"运动检测: 检测到{regions_count}个运动区域，强度{intensity:.2f}"
        else:
            motion_text = "运动检测: 无运动"
        
        # 如果有摔倒检测，添加摔倒信息
        if fall_detected:
            return f"🚨 摔倒检测: 检测到摔倒事件！置信度{fall_confidence:.2f} | {motion_text}"
        elif self.enable_fall_detection and fall_confidence > 0:
            return f"⚠️ 运动检测: 摔倒风险{fall_confidence:.2f} | {motion_text}"
        else:
            return motion_text


# 颜色分析器
class ColorAnalyzer(BaseVideoAnalyzer):
    """颜色分析器"""
    
    def __init__(self, target_colors: Optional[List[str]] = None):
        """
        初始化颜色分析器
        
        Args:
            target_colors: 目标颜色列表，支持的颜色：red, green, blue, yellow, orange, purple, pink, brown, gray, white, black
        """
        super().__init__("color_analyzer")
        
        self.target_colors = target_colors or ["red", "green", "blue"]
        
        # 颜色范围定义 (HSV格式)
        self.color_ranges = {
            "red": [
                ((0, 50, 50), (10, 255, 255)),      # 红色范围1
                ((170, 50, 50), (180, 255, 255))    # 红色范围2
            ],
            "green": [((35, 50, 50), (85, 255, 255))],
            "blue": [((100, 50, 50), (130, 255, 255))],
            "yellow": [((20, 50, 50), (35, 255, 255))],
            "orange": [((10, 50, 50), (20, 255, 255))],
            "purple": [((130, 50, 50), (170, 255, 255))],
            "pink": [((140, 50, 50), (170, 255, 255))],
            "brown": [((10, 100, 20), (20, 255, 200))],
            "gray": [((0, 0, 40), (180, 30, 230))],
            "white": [((0, 0, 200), (180, 30, 255))],
            "black": [((0, 0, 0), (180, 255, 30))]
        }
        
        # 配置参数
        self.config.update({
            "target_colors": self.target_colors
        })
    
    async def analyze_frame(self, frame) -> Optional[Dict[str, Any]]:
        """分析帧中的颜色"""
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            color_analysis = {}
            total_pixels = frame.shape[0] * frame.shape[1]
            
            for color_name in self.target_colors:
                if color_name in self.color_ranges:
                    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                    
                    # 合并多个颜色范围
                    for lower, upper in self.color_ranges[color_name]:
                        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                        color_mask = cv2.bitwise_or(color_mask, mask)
                    
                    # 计算颜色像素数量
                    color_pixels = cv2.countNonZero(color_mask)
                    color_percentage = (color_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                    
                    color_analysis[color_name] = {
                        "pixel_count": int(color_pixels),
                        "percentage": float(color_percentage),
                        "dominant": color_percentage > 10.0  # 超过10%认为是主要颜色
                    }
            
            # 找出主要颜色
            dominant_colors = [
                color for color, data in color_analysis.items() 
                if data["dominant"]
            ]
            
            return {
                "color_analysis": color_analysis,
                "dominant_colors": dominant_colors,
                "total_pixels": int(total_pixels),
                "target_colors": self.target_colors
            }
            
        except Exception as e:
            self._add_error(str(e))
            return None
    
    def get_analysis_type(self) -> str:
        return "color_analysis"
    
    def get_text_result(self, analysis_data: Dict[str, Any]) -> str:
        """返回文本格式的分析结果"""
        if not analysis_data:
            return "颜色分析: 无数据"
        
        dominant_colors = analysis_data.get("dominant_colors", [])
        color_analysis = analysis_data.get("color_analysis", {})
        
        if dominant_colors:
            color_info = []
            for color in dominant_colors:
                if color in color_analysis:
                    percentage = color_analysis[color]["percentage"]
                    color_info.append(f"{color}({percentage:.1f}%)")
            return f"颜色分析: 主要颜色 {', '.join(color_info)}"
        else:
            return "颜色分析: 无明显主色调"


# 亮度分析器
class BrightnessAnalyzer(BaseVideoAnalyzer):
    """亮度分析器"""
    
    def __init__(self, brightness_threshold: float = 0.5):
        """
        初始化亮度分析器
        
        Args:
            brightness_threshold: 亮度阈值，用于判断过亮或过暗
        """
        super().__init__("brightness_analyzer")
        
        self.brightness_threshold = brightness_threshold
        
        # 配置参数
        self.config.update({
            "brightness_threshold": brightness_threshold
        })
    
    async def analyze_frame(self, frame) -> Optional[Dict[str, Any]]:
        """分析帧的亮度"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算平均亮度
            mean_brightness = np.mean(gray)
            
            # 计算亮度分布
            brightness_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # 计算亮度标准差
            std_brightness = np.std(gray)
            
            # 判断亮度状态
            if mean_brightness < 64:
                brightness_status = "dark"
            elif mean_brightness > 192:
                brightness_status = "bright"
            else:
                brightness_status = "normal"
            
            # 计算过暗和过亮的像素比例
            dark_pixels = np.sum(gray < 64)
            bright_pixels = np.sum(gray > 192)
            total_pixels = gray.size
            
            dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
            bright_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0
            
            return {
                "mean_brightness": float(mean_brightness),
                "std_brightness": float(std_brightness),
                "brightness_status": brightness_status,
                "dark_pixels_ratio": float(dark_ratio),
                "bright_pixels_ratio": float(bright_ratio),
                "dark_pixels_count": int(dark_pixels),
                "bright_pixels_count": int(bright_pixels),
                "total_pixels": int(total_pixels),
                "threshold": self.brightness_threshold
            }
            
        except Exception as e:
            self._add_error(str(e))
            return None
    
    def get_analysis_type(self) -> str:
        return "brightness_analysis"
    
    def get_text_result(self, analysis_data: Dict[str, Any]) -> str:
        """返回文本格式的分析结果"""
        if not analysis_data:
            return "亮度分析: 无数据"
        
        brightness_status = analysis_data.get("brightness_status", "unknown")
        mean_brightness = analysis_data.get("mean_brightness", 0)
        
        return f"亮度分析: {brightness_status} (平均亮度: {mean_brightness:.1f})"


# 对象计数器
class ObjectCounter(BaseVideoAnalyzer):
    """对象计数器"""
    
    def __init__(self, min_area: int = 100, max_area: int = 10000):
        """
        初始化对象计数器
        
        Args:
            min_area: 最小对象面积
            max_area: 最大对象面积
        """
        super().__init__("object_counter")
        
        self.min_area = min_area
        self.max_area = max_area
        
        # 配置参数
        self.config.update({
            "min_area": min_area,
            "max_area": max_area
        })
    
    async def analyze_frame(self, frame) -> Optional[Dict[str, Any]]:
        """计算帧中的对象数量"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 应用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 二值化
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤对象
            valid_objects = []
            total_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area <= area <= self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    valid_objects.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area)
                    })
                    total_area += area
            
            # 计算对象密度
            frame_area = frame.shape[0] * frame.shape[1]
            object_density = len(valid_objects) / frame_area if frame_area > 0 else 0
            
            return {
                "object_count": len(valid_objects),
                "objects": valid_objects,
                "total_object_area": int(total_area),
                "object_density": float(object_density),
                "min_area": self.min_area,
                "max_area": self.max_area
            }
            
        except Exception as e:
            self._add_error(str(e))
            return None
    
    def get_analysis_type(self) -> str:
        return "object_counting"
    
    def get_text_result(self, analysis_data: Dict[str, Any]) -> str:
        """返回文本格式的分析结果"""
        if not analysis_data:
            return "对象计数: 无数据"
        
        object_count = analysis_data.get("object_count", 0)
        density = analysis_data.get("object_density", 0)
        
        return f"对象计数: 检测到{object_count}个对象 (密度: {density:.4f})"
