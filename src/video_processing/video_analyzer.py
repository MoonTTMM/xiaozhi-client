"""
具体视频分析器实现

提供多种预定义的视频分析方法，包括运动检测、颜色分析、亮度分析、对象计数等。
直接返回文本格式的分析结果，适用于向服务端发送。
"""

import cv2
import numpy as np
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from .base_analyzer import BaseVideoAnalyzer
from src.utils.logging_config import get_logger

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
        
        # 初始化日志记录器
        self.logger = get_logger(f"{__name__}.MotionDetector")
        self.fall_logger = get_logger(f"{__name__}.FallDetection")
        
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
        
        # 摔倒检测相关变量 - 优化版本
        self.fall_detection_history = []  # 存储历史帧的运动信息
        self.max_history_frames = 20  # 增加历史帧数到20帧
        self.fall_detection_threshold = 0.4  # 提高阈值减少误报
        self.last_fall_time = 0  # 上次检测到摔倒的时间
        self.fall_cooldown = 3.0  # 减少冷却时间到3秒
        self.min_motion_for_fall = 0.02  # 提高最小运动要求
        
        # 新增：跌倒检测状态机
        self.fall_state = "normal"  # normal, potential_fall, falling, fallen
        self.fall_state_timer = 0
        self.fall_state_duration = 0
        
        # 新增：自适应阈值
        self.adaptive_threshold = self.fall_detection_threshold
        self.threshold_adjustment_rate = 0.1
        
        # 新增：跌倒特征检测
        self.vertical_velocity_history = []  # 垂直速度历史
        self.acceleration_history = []  # 加速度历史
        self.contour_analysis_history = []  # 轮廓分析历史
        
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
        检测摔倒事件 - 优化版本
        
        Args:
            frame: 当前帧
            motion_regions: 运动区域列表
            motion_intensity: 运动强度
            
        Returns:
            包含摔倒检测结果的字典
        """
        current_time = time.time()
        
        # 记录检测开始
        self.fall_logger.debug(f"开始跌倒检测 - 时间: {current_time:.3f}, 运动区域数: {len(motion_regions)}, 运动强度: {motion_intensity:.4f}")
        
        # 检查冷却时间
        if current_time - self.last_fall_time < self.fall_cooldown:
            remaining_cooldown = self.fall_cooldown - (current_time - self.last_fall_time)
            self.fall_logger.debug(f"冷却时间未结束 - 剩余: {remaining_cooldown:.2f}秒")
            return {
                "fall_detected": False,
                "fall_confidence": 0.0,
                "fall_reason": "cooldown",
                "fall_state": self.fall_state
            }
        
        # 创建当前帧的运动信息
        current_motion_info = {
            "timestamp": current_time,
            "motion_regions": motion_regions,
            "motion_intensity": motion_intensity,
            "frame_height": frame.shape[0],
            "frame_width": frame.shape[1]
        }
        
        # 计算垂直速度和加速度
        self._update_motion_metrics(current_motion_info)
        
        # 添加到历史记录
        self.fall_detection_history.append(current_motion_info)
        
        # 保持历史记录长度
        if len(self.fall_detection_history) > self.max_history_frames:
            self.fall_detection_history.pop(0)
        
        # 需要至少5帧历史才能进行摔倒检测
        if len(self.fall_detection_history) < 5:
            self.fall_logger.debug(f"历史帧数不足 - 当前: {len(self.fall_detection_history)}, 需要: 5")
            return {
                "fall_detected": False,
                "fall_confidence": 0.0,
                "fall_reason": "insufficient_history",
                "fall_state": self.fall_state
            }
        
        # 检查是否有足够的运动强度
        if motion_intensity < self.min_motion_for_fall and len(motion_regions) == 0:
            self.fall_logger.debug(f"运动强度不足 - 当前: {motion_intensity:.4f}, 最小要求: {self.min_motion_for_fall}")
            return {
                "fall_detected": False,
                "fall_confidence": 0.0,
                "fall_reason": "insufficient_motion",
                "fall_state": self.fall_state
            }
        
        # 更新跌倒状态机
        self._update_fall_state_machine()
        
        # 多维度分析运动模式
        fall_confidence = self._analyze_fall_pattern_advanced()
        
        # 自适应阈值调整
        self._adjust_adaptive_threshold(fall_confidence)
        
        # 判断是否摔倒
        fall_detected = fall_confidence > self.adaptive_threshold
        
        # 记录检测结果
        self.fall_logger.info(f"跌倒检测结果 - 置信度: {fall_confidence:.4f}, 阈值: {self.adaptive_threshold:.4f}, 状态: {self.fall_state}, 检测到: {fall_detected}")
        
        if fall_detected and self.fall_state == "fallen":
            self.last_fall_time = current_time
            self.fall_logger.warning(f"检测到跌倒事件！置信度: {fall_confidence:.4f}, 状态: {self.fall_state}, 持续时间: {self.fall_state_duration:.2f}秒")
            return {
                "fall_detected": True,
                "fall_confidence": fall_confidence,
                "fall_reason": "advanced_pattern_analysis",
                "fall_state": self.fall_state,
                "fall_duration": self.fall_state_duration
            }
        else:
            reason = "below_threshold" if not fall_detected else f"state_{self.fall_state}"
            self.fall_logger.debug(f"未检测到跌倒 - 原因: {reason}, 置信度: {fall_confidence:.4f}")
            return {
                "fall_detected": False,
                "fall_confidence": fall_confidence,
                "fall_reason": reason,
                "fall_state": self.fall_state,
                "fall_duration": self.fall_state_duration
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
    
    def _update_motion_metrics(self, current_motion_info: Dict[str, Any]):
        """更新运动指标（速度、加速度等）"""
        if len(self.fall_detection_history) > 0:
            # 计算垂直速度
            previous_info = self.fall_detection_history[-1]
            time_diff = current_motion_info["timestamp"] - previous_info["timestamp"]
            
            if time_diff > 0:
                # 计算运动区域中心点的垂直位置变化
                current_center_y = self._get_motion_center_y(current_motion_info["motion_regions"])
                previous_center_y = self._get_motion_center_y(previous_info["motion_regions"])
                
                if current_center_y is not None and previous_center_y is not None:
                    vertical_velocity = (current_center_y - previous_center_y) / time_diff
                    self.vertical_velocity_history.append(vertical_velocity)
                    
                    # 保持历史长度
                    if len(self.vertical_velocity_history) > 10:
                        self.vertical_velocity_history.pop(0)
                    
                    # 计算加速度
                    if len(self.vertical_velocity_history) >= 2:
                        acceleration = (self.vertical_velocity_history[-1] - self.vertical_velocity_history[-2]) / time_diff
                        self.acceleration_history.append(acceleration)
                        
                        if len(self.acceleration_history) > 10:
                            self.acceleration_history.pop(0)
    
    def _get_motion_center_y(self, motion_regions: List[Dict]) -> Optional[float]:
        """获取运动区域的垂直中心点"""
        if not motion_regions:
            return None
        
        # 找到最大的运动区域
        largest_region = max(motion_regions, key=lambda r: r["area"])
        return largest_region["y"] + largest_region["height"] / 2
    
    def _update_fall_state_machine(self):
        """更新跌倒状态机"""
        current_time = time.time()
        old_state = self.fall_state
        
        # 计算当前状态的特征
        vertical_velocity = self._get_average_vertical_velocity()
        acceleration = self._get_average_acceleration()
        motion_intensity = self._get_recent_motion_intensity()
        
        # 记录状态机输入参数
        self.fall_logger.debug(f"状态机更新 - 垂直速度: {vertical_velocity:.2f}, 加速度: {acceleration:.2f}, 运动强度: {motion_intensity:.4f}")
        
        # 状态转换逻辑
        if self.fall_state == "normal":
            # 检测到快速向下运动，进入潜在跌倒状态
            if vertical_velocity > 50 and acceleration < -100:  # 快速向下运动
                self.fall_state = "potential_fall"
                self.fall_state_timer = current_time
                self.fall_logger.info(f"状态转换: {old_state} -> {self.fall_state} (快速向下运动检测)")
        elif self.fall_state == "potential_fall":
            # 持续快速向下运动，进入跌倒中状态
            if vertical_velocity > 30 and acceleration < -50:
                self.fall_state = "falling"
                self.fall_state_timer = current_time
                self.fall_logger.info(f"状态转换: {old_state} -> {self.fall_state} (持续快速向下运动)")
            # 运动停止，回到正常状态
            elif motion_intensity < 0.01:
                self.fall_state = "normal"
                self.fall_logger.info(f"状态转换: {old_state} -> {self.fall_state} (运动停止)")
        elif self.fall_state == "falling":
            # 运动停止且位置稳定，进入已跌倒状态
            if motion_intensity < 0.02 and abs(vertical_velocity) < 10:
                self.fall_state = "fallen"
                self.fall_state_timer = current_time
                self.fall_logger.warning(f"状态转换: {old_state} -> {self.fall_state} (运动停止且位置稳定)")
            # 运动恢复，回到正常状态
            elif motion_intensity > 0.05:
                self.fall_state = "normal"
                self.fall_logger.info(f"状态转换: {old_state} -> {self.fall_state} (运动恢复)")
        elif self.fall_state == "fallen":
            # 检测到明显运动，回到正常状态
            if motion_intensity > 0.05:
                self.fall_state = "normal"
                self.fall_logger.info(f"状态转换: {old_state} -> {self.fall_state} (检测到明显运动)")
        
        # 更新状态持续时间
        self.fall_state_duration = current_time - self.fall_state_timer
    
    def _get_average_vertical_velocity(self) -> float:
        """获取平均垂直速度"""
        if not self.vertical_velocity_history:
            return 0.0
        return sum(self.vertical_velocity_history) / len(self.vertical_velocity_history)
    
    def _get_average_acceleration(self) -> float:
        """获取平均加速度"""
        if not self.acceleration_history:
            return 0.0
        return sum(self.acceleration_history) / len(self.acceleration_history)
    
    def _get_recent_motion_intensity(self) -> float:
        """获取最近的运动强度"""
        if len(self.fall_detection_history) < 3:
            return 0.0
        
        recent_intensities = [info["motion_intensity"] for info in self.fall_detection_history[-3:]]
        return sum(recent_intensities) / len(recent_intensities)
    
    def _analyze_fall_pattern_advanced(self) -> float:
        """
        高级跌倒模式分析 - 多维度特征融合
        
        Returns:
            跌倒置信度 (0.0-1.0)
        """
        if len(self.fall_detection_history) < 5:
            self.fall_logger.debug("历史帧数不足，无法进行高级模式分析")
            return 0.0
        
        confidence = 0.0
        analysis_details = {}
        
        # 1. 垂直运动分析 (权重: 0.3)
        vertical_score = self._analyze_vertical_movement_advanced()
        confidence += vertical_score * 0.3
        analysis_details["vertical_score"] = vertical_score
        
        # 2. 速度加速度分析 (权重: 0.25)
        velocity_score = self._analyze_velocity_pattern()
        confidence += velocity_score * 0.25
        analysis_details["velocity_score"] = velocity_score
        
        # 3. 形状变化分析 (权重: 0.2)
        shape_score = self._analyze_shape_change_advanced()
        confidence += shape_score * 0.2
        analysis_details["shape_score"] = shape_score
        
        # 4. 时序模式分析 (权重: 0.15)
        temporal_score = self._analyze_temporal_pattern()
        confidence += temporal_score * 0.15
        analysis_details["temporal_score"] = temporal_score
        
        # 5. 状态机分析 (权重: 0.1)
        state_score = self._analyze_state_machine_confidence()
        confidence += state_score * 0.1
        analysis_details["state_score"] = state_score
        
        # 记录详细分析结果
        self.fall_logger.debug(f"跌倒模式分析详情: {json.dumps(analysis_details, indent=2)}")
        self.fall_logger.debug(f"综合置信度: {confidence:.4f}")
        
        return min(confidence, 1.0)
    
    def _analyze_vertical_movement_advanced(self) -> float:
        """高级垂直运动分析"""
        if len(self.fall_detection_history) < 5:
            return 0.0
        
        # 分析最近5帧的垂直位置变化
        recent_frames = self.fall_detection_history[-5:]
        positions = []
        
        for frame_info in recent_frames:
            center_y = self._get_motion_center_y(frame_info["motion_regions"])
            if center_y is not None:
                # 标准化位置 (0-1)
                normalized_y = center_y / frame_info["frame_height"]
                positions.append(normalized_y)
        
        if len(positions) < 3:
            return 0.0
        
        # 计算位置变化趋势
        start_pos = positions[0]
        end_pos = positions[-1]
        position_change = end_pos - start_pos
        
        # 计算变化的一致性
        changes = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        consistent_downward = all(change > 0 for change in changes)  # 持续向下
        
        # 计算变化幅度
        change_magnitude = abs(position_change)
        
        # 综合评分
        score = 0.0
        if consistent_downward and change_magnitude > 0.1:  # 持续向下且变化明显
            score = min(change_magnitude * 2.0, 1.0)
        elif change_magnitude > 0.2:  # 变化很大
            score = min(change_magnitude, 1.0)
        
        return score
    
    def _analyze_velocity_pattern(self) -> float:
        """分析速度模式"""
        if len(self.vertical_velocity_history) < 3:
            return 0.0
        
        # 检查是否有快速向下的速度
        recent_velocities = self.vertical_velocity_history[-3:]
        max_velocity = max(recent_velocities)
        
        # 检查速度变化的一致性
        velocity_changes = [recent_velocities[i+1] - recent_velocities[i] 
                          for i in range(len(recent_velocities)-1)]
        
        # 如果速度持续增加（向下），可能是跌倒
        consistent_acceleration = all(change > 0 for change in velocity_changes)
        
        score = 0.0
        if max_velocity > 50:  # 快速向下运动
            score += 0.5
        if consistent_acceleration:
            score += 0.3
        if max_velocity > 100:  # 非常快的运动
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_shape_change_advanced(self) -> float:
        """高级形状变化分析"""
        if len(self.fall_detection_history) < 3:
            return 0.0
        
        # 分析最近3帧的形状变化
        recent_frames = self.fall_detection_history[-3:]
        shape_ratios = []
        
        for frame_info in recent_frames:
            if frame_info["motion_regions"]:
                largest_region = max(frame_info["motion_regions"], key=lambda r: r["area"])
                ratio = largest_region["width"] / largest_region["height"] if largest_region["height"] > 0 else 1
                shape_ratios.append(ratio)
        
        if len(shape_ratios) < 2:
            return 0.0
        
        # 计算宽高比变化
        ratio_changes = [abs(shape_ratios[i+1] - shape_ratios[i]) 
                        for i in range(len(shape_ratios)-1)]
        
        # 如果宽高比变化很大，可能是从站立变为躺下
        max_change = max(ratio_changes) if ratio_changes else 0
        
        score = 0.0
        if max_change > 0.5:  # 形状变化很大
            score = min(max_change, 1.0)
        
        return score
    
    def _analyze_temporal_pattern(self) -> float:
        """分析时序模式"""
        if len(self.fall_detection_history) < 5:
            return 0.0
        
        # 分析运动强度的时序变化
        intensities = [info["motion_intensity"] for info in self.fall_detection_history[-5:]]
        
        # 检查是否有运动强度突然增加然后减少的模式（跌倒特征）
        if len(intensities) >= 3:
            # 找到运动强度的峰值
            peak_index = intensities.index(max(intensities))
            
            # 检查峰值前后的变化
            if peak_index > 0 and peak_index < len(intensities) - 1:
                before_peak = intensities[peak_index - 1]
                at_peak = intensities[peak_index]
                after_peak = intensities[peak_index + 1]
                
                # 如果峰值明显且之后下降，可能是跌倒
                if at_peak > before_peak * 1.5 and after_peak < at_peak * 0.7:
                    return min((at_peak - before_peak) * 2.0, 1.0)
        
        return 0.0
    
    def _analyze_state_machine_confidence(self) -> float:
        """基于状态机的置信度分析"""
        if self.fall_state == "fallen":
            return 0.8
        elif self.fall_state == "falling":
            return 0.6
        elif self.fall_state == "potential_fall":
            return 0.3
        else:
            return 0.0
    
    def _adjust_adaptive_threshold(self, confidence: float):
        """自适应调整阈值"""
        old_threshold = self.adaptive_threshold
        
        # 如果连续多次高置信度但未检测到跌倒，降低阈值
        # 如果连续多次误报，提高阈值
        if confidence > 0.7 and self.fall_state == "normal":
            self.adaptive_threshold = max(
                self.adaptive_threshold - self.threshold_adjustment_rate * 0.1,
                self.fall_detection_threshold * 0.5
            )
            if abs(self.adaptive_threshold - old_threshold) > 0.001:
                self.fall_logger.debug(f"降低阈值: {old_threshold:.4f} -> {self.adaptive_threshold:.4f} (高置信度但未跌倒)")
        elif confidence < 0.2 and self.fall_state != "normal":
            self.adaptive_threshold = min(
                self.adaptive_threshold + self.threshold_adjustment_rate * 0.1,
                self.fall_detection_threshold * 1.5
            )
            if abs(self.adaptive_threshold - old_threshold) > 0.001:
                self.fall_logger.debug(f"提高阈值: {old_threshold:.4f} -> {self.adaptive_threshold:.4f} (低置信度但状态异常)")

    def get_analysis_type(self) -> str:
        return "motion_detection"
    
    def get_text_result(self, analysis_data: Dict[str, Any]) -> str:
        """返回文本格式的分析结果 - 优化版本"""
        if not analysis_data:
            return "运动检测: 无数据"
        
        motion_detected = analysis_data.get("motion_detected", False)
        regions_count = analysis_data.get("motion_regions_count", 0)
        intensity = analysis_data.get("motion_intensity", 0)
        
        # 检查摔倒检测结果
        fall_detected = analysis_data.get("fall_detected", False)
        fall_confidence = analysis_data.get("fall_confidence", 0.0)
        fall_state = analysis_data.get("fall_state", "normal")
        fall_duration = analysis_data.get("fall_duration", 0.0)
        
        # 构建基础运动检测文本
        if motion_detected:
            motion_text = f"运动检测: 检测到{regions_count}个运动区域，强度{intensity:.2f}"
        else:
            motion_text = "运动检测: 无运动"
        
        # 状态映射
        state_emojis = {
            "normal": "✅",
            "potential_fall": "⚠️",
            "falling": "🔻",
            "fallen": "🚨"
        }
        
        state_names = {
            "normal": "正常",
            "potential_fall": "潜在跌倒",
            "falling": "跌倒中",
            "fallen": "已跌倒"
        }
        
        emoji = state_emojis.get(fall_state, "❓")
        state_name = state_names.get(fall_state, fall_state)
        
        # 如果有摔倒检测，添加摔倒信息
        if fall_detected:
            duration_text = f"持续{fall_duration:.1f}秒" if fall_duration > 0 else ""
            return f"{emoji} 摔倒检测: 检测到摔倒事件！置信度{fall_confidence:.2f} ({state_name}) {duration_text} | {motion_text}"
        elif self.enable_fall_detection and fall_confidence > 0:
            return f"{emoji} 运动检测: 摔倒风险{fall_confidence:.2f} ({state_name}) | {motion_text}"
        else:
            return f"{emoji} {motion_text}"


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
