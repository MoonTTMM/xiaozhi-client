# 视频处理模块

## 概述

视频处理模块是一个可扩展的视频分析系统，提供多种预定义的视频分析方法，包括运动检测、颜色分析、亮度分析、对象计数等。系统采用异步架构，支持并行处理，能够高效地处理实时视频流。

## 架构设计

### 核心组件

1. **VideoManager** - 主协调器，管理整个视频处理流程
2. **CameraManager** - 摄像头设备管理，支持多种摄像头类型和配置
3. **FrameProcessor** - 帧处理器，协调多个分析器，支持并行处理
4. **BaseVideoAnalyzer** - 分析器基类，提供可扩展的分析接口

### 内置分析器

- **MotionDetector** - 运动检测器，使用背景减法算法
- **ColorAnalyzer** - 颜色分析器，支持多种颜色检测
- **BrightnessAnalyzer** - 亮度分析器，分析图像亮度和对比度
- **ObjectCounter** - 对象计数器，统计图像中的对象数量

## 快速开始

### 持续监控模式

```python
import asyncio
from src.video_processing import VideoManager, VideoManagerConfig

async def main():
    # 创建视频管理器，启用持续监控
    config = VideoManagerConfig(
        continuous_monitoring=True,
        monitoring_sensitivity=0.7,
        auto_session_trigger=True
    )
    video_manager = VideoManager(config)
    
    # 初始化并启动持续监控
    await video_manager.initialize()
    
    # 设置监控敏感度和阈值
    video_manager.set_monitoring_sensitivity(0.8)
    video_manager.set_event_thresholds({
        "motion": 0.9,
        "color_change": 0.8,
        "brightness": 0.8
    })
    
    # 添加会话触发回调
    def on_session_trigger(result):
        print(f"检测到事件，触发会话: {result}")
    
    video_manager.add_session_trigger_callback(on_session_trigger)
    
    # 持续监控运行
    await asyncio.sleep(3600)  # 运行1小时
    
    # 停止
    await video_manager.stop()

# 运行
asyncio.run(main())
```

### 基本使用

```python
import asyncio
from src.video_processing import VideoManager, VideoManagerConfig

async def main():
    # 创建视频管理器
    config = VideoManagerConfig()
    video_manager = VideoManager(config)
    
    # 初始化并启动
    await video_manager.initialize()
    
    # 添加结果回调
    def on_result(results):
        for result in results:
            print(f"分析结果: {result.analyzer_name} - {result.data}")
    
    video_manager.add_result_callback(on_result)
    
    # 运行一段时间
    await asyncio.sleep(30)
    
    # 停止
    await video_manager.stop()

# 运行
asyncio.run(main())
```

### 自定义分析器

```python
from src.video_processing import BaseVideoAnalyzer

class CustomAnalyzer(BaseVideoAnalyzer):
    def __init__(self):
        super().__init__("custom_analyzer")
    
    async def analyze_frame(self, frame):
        # 实现你的分析逻辑
        return {"result": "your_analysis"}
    
    def get_analysis_type(self):
        return "custom_analysis"

# 添加到系统
video_manager.add_analyzer(CustomAnalyzer())
```

## 配置选项

### 视频处理器配置

```python
from src.video_processing import VideoProcessorConfig, DEFAULT_CONFIG

# 使用默认配置
config = DEFAULT_CONFIG

# 自定义配置
config = VideoProcessorConfig(
    continuous_monitoring=True,        # 启用持续监控
    sensitivity=0.8,                  # 监控敏感度
    motion_threshold=0.9,             # 运动检测阈值
    color_change_threshold=0.8,       # 颜色变化阈值
    brightness_threshold=0.8,         # 亮度变化阈值
    auto_trigger_session=True,        # 自动触发会话
    trigger_cooldown=3.0,            # 触发冷却时间
    max_triggers_per_hour=15         # 每小时最大触发次数
)

# 保存配置到文件
config.save_to_file("video_config.json")

# 从文件加载配置
config = VideoProcessorConfig.load_from_file("video_config.json")
```

### 摄像头配置

```python
from src.video_processing import CameraConfig

camera_config = CameraConfig(
    device_id=0,           # 摄像头设备ID
    width=640,             # 分辨率宽度
    height=480,            # 分辨率高度
    fps=30,                # 目标帧率
    buffer_size=1,         # 缓冲区大小
    auto_exposure=True,    # 自动曝光
    timeout=5.0            # 超时时间
)
```

### 处理配置

```python
from src.video_processing import ProcessingConfig

processing_config = ProcessingConfig(
    target_fps=10,                     # 目标处理帧率
    frame_skip=0,                      # 跳过的帧数
    max_concurrent_analyzers=4,        # 最大并发分析器数
    result_buffer_size=100,            # 结果缓冲区大小
    enable_parallel_processing=True,   # 启用并行处理
    timeout_per_frame=5.0              # 每帧处理超时
)
```

## 性能优化

### 帧率控制

- 设置合适的 `target_fps` 避免过度处理
- 使用 `frame_skip` 跳过不需要的帧
- 调整 `max_concurrent_analyzers` 平衡性能和资源

### 内存管理

- 设置合理的 `result_buffer_size`
- 定期调用 `clear_cache()` 清理缓存
- 监控内存使用情况

## 错误处理

系统提供完善的错误处理机制：

- 自动重连摄像头
- 分析器异常隔离
- 详细的错误日志
- 错误回调通知

## 扩展开发

### 创建新的分析器

1. 继承 `BaseVideoAnalyzer` 类
2. 实现 `analyze_frame` 方法
3. 实现 `get_analysis_type` 方法
4. 添加配置参数和统计信息

### 集成到现有系统

```python
# 在 Application 类中集成
from src.video_processing import VideoManager

class Application:
    def __init__(self):
        # ... 其他初始化代码 ...
        self.video_manager = None
    
    async def open_audio_channel(self):
        # ... 音频通道打开后 ...
        
        # 初始化视频处理
        if not self.video_manager:
            self.video_manager = VideoManager()
            await self.video_manager.initialize()
            
            # 设置结果回调
            self.video_manager.add_result_callback(self._on_video_result)
    
    def _on_video_result(self, results):
        # 处理视频分析结果
        for result in results:
            # 发送到后端
            self.send_video_analysis_result(result)
```

## 注意事项

1. **依赖要求**: 需要安装 OpenCV (`opencv-python`)
2. **摄像头权限**: 确保应用有访问摄像头的权限
3. **性能考虑**: 在低性能设备上适当降低处理帧率
4. **错误恢复**: 系统会自动处理大部分错误，但建议实现错误回调

## 故障排除

### 常见问题

1. **摄像头无法连接**
   - 检查设备ID是否正确
   - 确认摄像头未被其他程序占用
   - 检查系统权限

2. **处理性能低**
   - 降低目标帧率
   - 减少并发分析器数量
   - 启用帧跳过

3. **内存占用高**
   - 减小结果缓冲区大小
   - 定期清理缓存
   - 检查分析器内存泄漏

## 版本历史

- **v1.0.0** - 初始版本，包含基础功能
