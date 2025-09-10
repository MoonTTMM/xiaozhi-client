#!/usr/bin/env python3
"""
跌倒检测日志分析工具

用于分析跌倒检测的日志记录，评估检测过程和结果
"""

import re
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

class FallDetectionLogAnalyzer:
    """跌倒检测日志分析器"""
    
    def __init__(self, log_file_path: str):
        """
        初始化日志分析器
        
        Args:
            log_file_path: 日志文件路径
        """
        self.log_file_path = log_file_path
        self.log_entries = []
        self.fall_events = []
        self.state_transitions = []
        self.confidence_data = []
        self.threshold_changes = []
        
    def parse_logs(self):
        """解析日志文件"""
        print("正在解析日志文件...")
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # 解析日志行
                    entry = self._parse_log_line(line, line_num)
                    if entry:
                        self.log_entries.append(entry)
                        
                        # 分类不同类型的日志
                        if entry['type'] == 'fall_detection':
                            self.fall_events.append(entry)
                        elif entry['type'] == 'state_transition':
                            self.state_transitions.append(entry)
                        elif entry['type'] == 'confidence_analysis':
                            self.confidence_data.append(entry)
                        elif entry['type'] == 'threshold_adjustment':
                            self.threshold_changes.append(entry)
                            
                except Exception as e:
                    print(f"解析第 {line_num} 行时出错: {e}")
                    continue
        
        print(f"解析完成 - 总日志条目: {len(self.log_entries)}")
        print(f"跌倒检测事件: {len(self.fall_events)}")
        print(f"状态转换: {len(self.state_transitions)}")
        print(f"置信度数据: {len(self.confidence_data)}")
        print(f"阈值调整: {len(self.threshold_changes)}")
    
    def _parse_log_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """解析单行日志"""
        # 匹配时间戳和日志级别
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})'
        level_pattern = r'(DEBUG|INFO|WARNING|ERROR)'
        
        timestamp_match = re.search(timestamp_pattern, line)
        level_match = re.search(level_pattern, line)
        
        if not timestamp_match or not level_match:
            return None
        
        timestamp = timestamp_match.group(1)
        level = level_match.group(1)
        
        # 解析不同类型的日志
        entry = {
            'line_num': line_num,
            'timestamp': timestamp,
            'level': level,
            'raw_line': line.strip()
        }
        
        # 跌倒检测结果
        if '跌倒检测结果' in line:
            entry['type'] = 'fall_detection'
            entry.update(self._parse_fall_detection_result(line))
        # 状态转换
        elif '状态转换:' in line:
            entry['type'] = 'state_transition'
            entry.update(self._parse_state_transition(line))
        # 置信度分析
        elif '跌倒模式分析详情' in line or '综合置信度' in line:
            entry['type'] = 'confidence_analysis'
            entry.update(self._parse_confidence_analysis(line))
        # 阈值调整
        elif '降低阈值:' in line or '提高阈值:' in line:
            entry['type'] = 'threshold_adjustment'
            entry.update(self._parse_threshold_adjustment(line))
        # 跌倒事件
        elif '检测到跌倒事件' in line:
            entry['type'] = 'fall_event'
            entry.update(self._parse_fall_event(line))
        else:
            entry['type'] = 'other'
        
        return entry
    
    def _parse_fall_detection_result(self, line: str) -> Dict[str, Any]:
        """解析跌倒检测结果"""
        # 提取置信度、阈值、状态等信息
        confidence_match = re.search(r'置信度: ([\d.]+)', line)
        threshold_match = re.search(r'阈值: ([\d.]+)', line)
        state_match = re.search(r'状态: (\w+)', line)
        detected_match = re.search(r'检测到: (True|False)', line)
        
        result = {}
        if confidence_match:
            result['confidence'] = float(confidence_match.group(1))
        if threshold_match:
            result['threshold'] = float(threshold_match.group(1))
        if state_match:
            result['state'] = state_match.group(1)
        if detected_match:
            result['detected'] = detected_match.group(1) == 'True'
        
        return result
    
    def _parse_state_transition(self, line: str) -> Dict[str, Any]:
        """解析状态转换"""
        # 提取状态转换信息
        transition_match = re.search(r'状态转换: (\w+) -> (\w+)', line)
        reason_match = re.search(r'\(([^)]+)\)', line)
        
        result = {}
        if transition_match:
            result['from_state'] = transition_match.group(1)
            result['to_state'] = transition_match.group(2)
        if reason_match:
            result['reason'] = reason_match.group(1)
        
        return result
    
    def _parse_confidence_analysis(self, line: str) -> Dict[str, Any]:
        """解析置信度分析"""
        if '跌倒模式分析详情' in line:
            # 解析JSON格式的分析详情
            json_match = re.search(r'跌倒模式分析详情: (.+)', line)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group(1))
                    return {'analysis_details': analysis_data}
                except json.JSONDecodeError:
                    pass
        elif '综合置信度' in line:
            confidence_match = re.search(r'综合置信度: ([\d.]+)', line)
            if confidence_match:
                return {'overall_confidence': float(confidence_match.group(1))}
        
        return {}
    
    def _parse_threshold_adjustment(self, line: str) -> Dict[str, Any]:
        """解析阈值调整"""
        # 提取阈值变化信息
        old_threshold_match = re.search(r'(\w+阈值): ([\d.]+) -> ([\d.]+)', line)
        reason_match = re.search(r'\(([^)]+)\)', line)
        
        result = {}
        if old_threshold_match:
            result['adjustment_type'] = old_threshold_match.group(1)
            result['old_threshold'] = float(old_threshold_match.group(2))
            result['new_threshold'] = float(old_threshold_match.group(3))
        if reason_match:
            result['reason'] = reason_match.group(1)
        
        return result
    
    def _parse_fall_event(self, line: str) -> Dict[str, Any]:
        """解析跌倒事件"""
        # 提取跌倒事件信息
        confidence_match = re.search(r'置信度: ([\d.]+)', line)
        state_match = re.search(r'状态: (\w+)', line)
        duration_match = re.search(r'持续时间: ([\d.]+)秒', line)
        
        result = {}
        if confidence_match:
            result['confidence'] = float(confidence_match.group(1))
        if state_match:
            result['state'] = state_match.group(1)
        if duration_match:
            result['duration'] = float(duration_match.group(1))
        
        return result
    
    def analyze_detection_performance(self):
        """分析检测性能"""
        print("\n=== 跌倒检测性能分析 ===")
        
        # 统计检测结果
        total_detections = len(self.fall_events)
        fall_detected_count = sum(1 for event in self.fall_events if event.get('detected', False))
        
        print(f"总检测次数: {total_detections}")
        print(f"检测到跌倒次数: {fall_detected_count}")
        print(f"检测率: {fall_detected_count/total_detections*100:.2f}%" if total_detections > 0 else "检测率: 0%")
        
        # 置信度分析
        if self.confidence_data:
            confidences = [event.get('overall_confidence', 0) for event in self.confidence_data if 'overall_confidence' in event]
            if confidences:
                print(f"平均置信度: {np.mean(confidences):.4f}")
                print(f"置信度范围: {min(confidences):.4f} - {max(confidences):.4f}")
                print(f"置信度标准差: {np.std(confidences):.4f}")
        
        # 状态分析
        self._analyze_state_patterns()
        
        # 阈值分析
        self._analyze_threshold_changes()
    
    def _analyze_state_patterns(self):
        """分析状态模式"""
        print("\n=== 状态模式分析 ===")
        
        if not self.state_transitions:
            print("无状态转换记录")
            return
        
        # 统计状态转换
        transition_counts = Counter()
        for transition in self.state_transitions:
            from_state = transition.get('from_state', 'unknown')
            to_state = transition.get('to_state', 'unknown')
            transition_counts[f"{from_state} -> {to_state}"] += 1
        
        print("状态转换统计:")
        for transition, count in transition_counts.most_common():
            print(f"  {transition}: {count} 次")
        
        # 分析状态持续时间
        state_durations = defaultdict(list)
        current_state = None
        state_start_time = None
        
        for entry in sorted(self.log_entries, key=lambda x: x['timestamp']):
            if entry['type'] == 'state_transition':
                if current_state and state_start_time:
                    # 计算前一个状态的持续时间
                    try:
                        start_dt = datetime.strptime(state_start_time, '%Y-%m-%d %H:%M:%S,%f')
                        end_dt = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
                        duration = (end_dt - start_dt).total_seconds()
                        state_durations[current_state].append(duration)
                    except ValueError:
                        pass
                
                current_state = entry.get('to_state')
                state_start_time = entry['timestamp']
        
        print("\n状态平均持续时间:")
        for state, durations in state_durations.items():
            if durations:
                avg_duration = np.mean(durations)
                print(f"  {state}: {avg_duration:.2f} 秒 (共 {len(durations)} 次)")
    
    def _analyze_threshold_changes(self):
        """分析阈值变化"""
        print("\n=== 阈值调整分析 ===")
        
        if not self.threshold_changes:
            print("无阈值调整记录")
            return
        
        # 统计阈值调整类型
        adjustment_types = Counter()
        for change in self.threshold_changes:
            adj_type = change.get('adjustment_type', 'unknown')
            adjustment_types[adj_type] += 1
        
        print("阈值调整统计:")
        for adj_type, count in adjustment_types.items():
            print(f"  {adj_type}: {count} 次")
        
        # 分析阈值变化趋势
        if self.threshold_changes:
            old_thresholds = [change.get('old_threshold', 0) for change in self.threshold_changes]
            new_thresholds = [change.get('new_threshold', 0) for change in self.threshold_changes]
            
            print(f"阈值变化范围: {min(old_thresholds + new_thresholds):.4f} - {max(old_thresholds + new_thresholds):.4f}")
            print(f"平均阈值变化: {np.mean([abs(n-o) for n, o in zip(new_thresholds, old_thresholds)]):.4f}")
    
    def generate_performance_report(self, output_file: str = "fall_detection_report.txt"):
        """生成性能报告"""
        print(f"\n正在生成性能报告: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("跌倒检测性能报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计
            f.write("基本统计:\n")
            f.write(f"  总检测次数: {len(self.fall_events)}\n")
            f.write(f"  检测到跌倒次数: {sum(1 for event in self.fall_events if event.get('detected', False))}\n")
            f.write(f"  状态转换次数: {len(self.state_transitions)}\n")
            f.write(f"  阈值调整次数: {len(self.threshold_changes)}\n\n")
            
            # 详细分析
            f.write("详细分析:\n")
            f.write("-" * 30 + "\n")
            
            # 置信度分析
            if self.confidence_data:
                confidences = [event.get('overall_confidence', 0) for event in self.confidence_data if 'overall_confidence' in event]
                if confidences:
                    f.write(f"置信度统计:\n")
                    f.write(f"  平均置信度: {np.mean(confidences):.4f}\n")
                    f.write(f"  置信度范围: {min(confidences):.4f} - {max(confidences):.4f}\n")
                    f.write(f"  置信度标准差: {np.std(confidences):.4f}\n\n")
            
            # 状态转换统计
            if self.state_transitions:
                f.write("状态转换统计:\n")
                transition_counts = Counter()
                for transition in self.state_transitions:
                    from_state = transition.get('from_state', 'unknown')
                    to_state = transition.get('to_state', 'unknown')
                    transition_counts[f"{from_state} -> {to_state}"] += 1
                
                for transition, count in transition_counts.most_common():
                    f.write(f"  {transition}: {count} 次\n")
                f.write("\n")
        
        print(f"性能报告已生成: {output_file}")
    
    def plot_confidence_trend(self, output_file: str = "confidence_trend.png"):
        """绘制置信度趋势图"""
        if not self.confidence_data:
            print("无置信度数据，无法生成趋势图")
            return
        
        confidences = [event.get('overall_confidence', 0) for event in self.confidence_data if 'overall_confidence' in event]
        if not confidences:
            print("无有效置信度数据")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(confidences, marker='o', markersize=3)
        plt.title('跌倒检测置信度趋势')
        plt.xlabel('检测次数')
        plt.ylabel('置信度')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 添加阈值线
        if self.threshold_changes:
            # 使用最后一个阈值作为参考
            last_threshold = self.threshold_changes[-1].get('new_threshold', 0.4)
            plt.axhline(y=last_threshold, color='r', linestyle='--', alpha=0.7, label=f'阈值: {last_threshold:.3f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"置信度趋势图已生成: {output_file}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始跌倒检测日志分析...")
        
        # 解析日志
        self.parse_logs()
        
        # 分析性能
        self.analyze_detection_performance()
        
        # 生成报告
        self.generate_performance_report()
        
        # 生成图表
        self.plot_confidence_trend()
        
        print("\n分析完成！")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='跌倒检测日志分析工具')
    parser.add_argument('log_file', help='日志文件路径')
    parser.add_argument('--output-dir', default='.', help='输出目录')
    
    args = parser.parse_args()
    
    analyzer = FallDetectionLogAnalyzer(args.log_file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
