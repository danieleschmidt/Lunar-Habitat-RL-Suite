"""
Advanced Performance Monitoring and Bottleneck Identification - Generation 3
Real-time performance analysis and optimization for NASA space mission operations.
"""

import asyncio
import threading
import time
import psutil
import numpy as np
import torch
import gc
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
import cProfile
import pstats
from io import StringIO
import traceback
import sys
import os
from pathlib import Path
import weakref

from ..utils.logging import get_logger

logger = get_logger("advanced_monitoring")


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    GPU_BOUND = "gpu_bound"
    NETWORK_BOUND = "network_bound"
    LOCK_CONTENTION = "lock_contention"
    ALGORITHM_INEFFICIENT = "algorithm_inefficient"
    RESOURCE_LEAK = "resource_leak"


class PerformanceIssue(NamedTuple):
    """Represents a detected performance issue."""
    type: BottleneckType
    severity: float  # 0.0 to 1.0
    description: str
    location: str  # Function/module where issue occurs
    timestamp: float
    metrics: Dict[str, Any]
    suggestions: List[str]


@dataclass
class ProfileReport:
    """Detailed profiling report."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    avg_time_per_call: float
    hot_spots: List[str]
    memory_usage: float
    cpu_usage: float


@dataclass
class SystemSnapshot:
    """Complete system performance snapshot."""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    gpu_utilization: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    open_files: int = 0
    gc_collections: int = 0


class AdvancedProfiler:
    """Advanced profiler with hotspot detection and call graph analysis."""
    
    def __init__(self, enable_memory_profiling: bool = True):
        self.enable_memory_profiling = enable_memory_profiling
        self.active_profiles = {}
        self.profile_results = deque(maxlen=1000)
        self.function_stats = defaultdict(lambda: defaultdict(list))
        self.call_graphs = {}
        self.lock = threading.RLock()
        
        # Memory tracking
        if enable_memory_profiling:
            try:
                import tracemalloc
                tracemalloc.start()
                self.memory_tracking_enabled = True
            except ImportError:
                logger.warning("tracemalloc not available, memory profiling disabled")
                self.memory_tracking_enabled = False
        else:
            self.memory_tracking_enabled = False
    
    def profile_function(self, func_name: Optional[str] = None, 
                        include_memory: bool = True,
                        include_callgraph: bool = False):
        """Decorator for profiling individual functions."""
        def decorator(func: Callable) -> Callable:
            profile_name = func_name or f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                return self.profile_execution(
                    func, profile_name, args, kwargs,
                    include_memory=include_memory,
                    include_callgraph=include_callgraph
                )
            
            return wrapper
        return decorator
    
    def profile_execution(self, func: Callable, name: str, args: tuple, kwargs: dict,
                         include_memory: bool = True,
                         include_callgraph: bool = False) -> Any:
        """Profile a single function execution."""
        start_time = time.perf_counter()
        start_memory = 0
        
        if include_memory and self.memory_tracking_enabled:
            import tracemalloc
            start_memory = tracemalloc.get_traced_memory()[0]
        
        # Create profiler
        profiler = cProfile.Profile()
        
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            execution_time = time.perf_counter() - start_time
            memory_delta = 0
            
            if include_memory and self.memory_tracking_enabled:
                import tracemalloc
                end_memory = tracemalloc.get_traced_memory()[0]
                memory_delta = (end_memory - start_memory) / (1024 * 1024)  # MB
            
            # Extract profile stats
            stats_buffer = StringIO()
            ps = pstats.Stats(profiler, stream=stats_buffer)
            ps.sort_stats('cumulative')
            ps.print_stats(20)
            
            profile_data = {
                'name': name,
                'execution_time': execution_time,
                'memory_delta_mb': memory_delta,
                'stats_text': stats_buffer.getvalue(),
                'timestamp': time.time(),
                'call_count': 1
            }
            
            # Store profile data
            with self.lock:
                self.profile_results.append(profile_data)
                self.function_stats[name]['execution_times'].append(execution_time)
                self.function_stats[name]['memory_deltas'].append(memory_delta)
            
            return result
            
        except Exception as e:
            profiler.disable()
            logger.error(f"Error in profile_execution for {name}: {e}")
            raise
    
    def start_system_profiling(self, name: str):
        """Start system-wide profiling session."""
        with self.lock:
            if name in self.active_profiles:
                logger.warning(f"Profile {name} already active")
                return
            
            profiler = cProfile.Profile()
            profiler.enable()
            
            self.active_profiles[name] = {
                'profiler': profiler,
                'start_time': time.perf_counter(),
                'start_memory': 0
            }
            
            if self.memory_tracking_enabled:
                import tracemalloc
                self.active_profiles[name]['start_memory'] = tracemalloc.get_traced_memory()[0]
    
    def stop_system_profiling(self, name: str) -> Optional[ProfileReport]:
        """Stop system profiling and generate report."""
        with self.lock:
            if name not in self.active_profiles:
                logger.warning(f"No active profile named {name}")
                return None
            
            profile_session = self.active_profiles.pop(name)
            profiler = profile_session['profiler']
            profiler.disable()
            
            execution_time = time.perf_counter() - profile_session['start_time']
            memory_delta = 0
            
            if self.memory_tracking_enabled:
                import tracemalloc
                end_memory = tracemalloc.get_traced_memory()[0]
                memory_delta = (end_memory - profile_session['start_memory']) / (1024 * 1024)
            
            # Generate detailed stats
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # Extract top functions
            stats_list = []
            for func_info, (call_count, _, cumtime, _, callers) in stats.stats.items():
                if cumtime > 0.001:  # Filter out very short functions
                    filename, line_num, func_name = func_info
                    stats_list.append({
                        'function': f"{filename}:{line_num}({func_name})",
                        'cumulative_time': cumtime,
                        'call_count': call_count,
                        'avg_time': cumtime / call_count if call_count > 0 else 0
                    })
            
            # Sort by cumulative time
            stats_list.sort(key=lambda x: x['cumulative_time'], reverse=True)
            hot_spots = [item['function'] for item in stats_list[:10]]
            
            report = ProfileReport(
                function_name=name,
                total_time=execution_time,
                cumulative_time=sum(item['cumulative_time'] for item in stats_list),
                call_count=sum(item['call_count'] for item in stats_list),
                avg_time_per_call=execution_time / max(1, sum(item['call_count'] for item in stats_list)),
                hot_spots=hot_spots,
                memory_usage=memory_delta,
                cpu_usage=0.0  # Would need separate CPU monitoring
            )
            
            return report
    
    def get_function_analytics(self, function_name: str) -> Dict[str, Any]:
        """Get analytics for a specific function."""
        with self.lock:
            if function_name not in self.function_stats:
                return {}
            
            stats = self.function_stats[function_name]
            execution_times = stats['execution_times']
            memory_deltas = stats['memory_deltas']
            
            return {
                'total_calls': len(execution_times),
                'avg_execution_time': np.mean(execution_times) if execution_times else 0,
                'std_execution_time': np.std(execution_times) if execution_times else 0,
                'min_execution_time': np.min(execution_times) if execution_times else 0,
                'max_execution_time': np.max(execution_times) if execution_times else 0,
                'p95_execution_time': np.percentile(execution_times, 95) if execution_times else 0,
                'avg_memory_delta': np.mean(memory_deltas) if memory_deltas else 0,
                'total_memory_allocated': np.sum(np.maximum(memory_deltas, 0)) if memory_deltas else 0,
                'memory_leak_indicator': np.sum(memory_deltas) if memory_deltas else 0
            }
    
    def identify_performance_issues(self) -> List[PerformanceIssue]:
        """Identify performance issues from profiling data."""
        issues = []
        
        with self.lock:
            # Analyze function statistics
            for func_name, stats in self.function_stats.items():
                analytics = self.get_function_analytics(func_name)
                
                if not analytics:
                    continue
                
                # Check for slow functions
                if analytics['avg_execution_time'] > 1.0:  # > 1 second
                    issues.append(PerformanceIssue(
                        type=BottleneckType.CPU_INTENSIVE,
                        severity=min(1.0, analytics['avg_execution_time'] / 10.0),
                        description=f"Slow function execution: {analytics['avg_execution_time']:.3f}s average",
                        location=func_name,
                        timestamp=time.time(),
                        metrics=analytics,
                        suggestions=[
                            "Profile function internals for optimization opportunities",
                            "Consider algorithm improvements or caching",
                            "Look for unnecessary computations or loops"
                        ]
                    ))
                
                # Check for memory leaks
                if analytics['memory_leak_indicator'] > 10.0:  # > 10MB consistent growth
                    issues.append(PerformanceIssue(
                        type=BottleneckType.RESOURCE_LEAK,
                        severity=min(1.0, analytics['memory_leak_indicator'] / 100.0),
                        description=f"Potential memory leak: {analytics['memory_leak_indicator']:.1f}MB net growth",
                        location=func_name,
                        timestamp=time.time(),
                        metrics=analytics,
                        suggestions=[
                            "Check for unreleased resources",
                            "Verify proper cleanup in exception handlers",
                            "Review object lifecycle management"
                        ]
                    ))
                
                # Check for high variance (inconsistent performance)
                if (analytics['std_execution_time'] > analytics['avg_execution_time'] * 0.5 and
                    analytics['avg_execution_time'] > 0.1):
                    issues.append(PerformanceIssue(
                        type=BottleneckType.ALGORITHM_INEFFICIENT,
                        severity=analytics['std_execution_time'] / analytics['avg_execution_time'],
                        description=f"Inconsistent performance: {analytics['std_execution_time']:.3f}s std dev",
                        location=func_name,
                        timestamp=time.time(),
                        metrics=analytics,
                        suggestions=[
                            "Profile with different input sizes",
                            "Check for input-dependent performance",
                            "Consider more stable algorithms"
                        ]
                    ))
        
        return sorted(issues, key=lambda x: x.severity, reverse=True)


class SystemMonitor:
    """Comprehensive system monitoring and bottleneck detection."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.snapshots = deque(maxlen=3600)  # Keep 1 hour of data
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        
        # Bottleneck detection thresholds
        self.thresholds = {
            'cpu_high': 85.0,
            'memory_high': 90.0,
            'gpu_high': 95.0,
            'io_wait_high': 20.0,
            'disk_usage_high': 90.0,
            'network_latency_high': 100.0
        }
        
        # Performance baselines
        self.baselines = {}
        self.anomaly_detector = AnomalyDetector()
        
        # Initialize baseline I/O counters
        self._last_io_counters = self._get_io_counters()
        self._last_network_counters = self._get_network_counters()
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                snapshot = self._collect_system_snapshot()
                if snapshot:
                    self.snapshots.append(snapshot)
                    
                    # Detect anomalies
                    anomalies = self.anomaly_detector.detect_anomalies(snapshot)
                    if anomalies:
                        self._handle_anomalies(anomalies)
                    
                    # Update baselines
                    self._update_baselines(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _collect_system_snapshot(self) -> Optional[SystemSnapshot]:
        """Collect comprehensive system snapshot."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # I/O metrics
            io_counters = self._get_io_counters()
            network_counters = self._get_network_counters()
            
            disk_io_read = 0.0
            disk_io_write = 0.0
            network_recv = 0.0
            network_sent = 0.0
            
            if io_counters and self._last_io_counters:
                disk_io_read = (io_counters.read_bytes - self._last_io_counters.read_bytes) / (1024 * 1024)
                disk_io_write = (io_counters.write_bytes - self._last_io_counters.write_bytes) / (1024 * 1024)
                self._last_io_counters = io_counters
            
            if network_counters and self._last_network_counters:
                network_recv = (network_counters.bytes_recv - self._last_network_counters.bytes_recv) / (1024 * 1024)
                network_sent = (network_counters.bytes_sent - self._last_network_counters.bytes_sent) / (1024 * 1024)
                self._last_network_counters = network_counters
            
            # GPU metrics
            gpu_utilization = None
            gpu_memory = None
            try:
                if torch.cuda.is_available():
                    gpu_utilization = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            except:
                pass
            
            # System information
            process_count = len(psutil.pids())
            thread_count = process.num_threads()
            open_files = len(process.open_files()) if hasattr(process, 'open_files') else 0
            
            # Garbage collection stats
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats) if gc_stats else 0
            
            return SystemSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_used_gb=memory.used / (1024**3),
                memory_percent=memory.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_gb=gpu_memory,
                disk_io_read_mb=disk_io_read,
                disk_io_write_mb=disk_io_write,
                network_io_recv_mb=network_recv,
                network_io_sent_mb=network_sent,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                gc_collections=gc_collections
            )
            
        except Exception as e:
            logger.error(f"Error collecting system snapshot: {e}")
            return None
    
    def _get_io_counters(self):
        """Get I/O counters safely."""
        try:
            return psutil.Process().io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return None
    
    def _get_network_counters(self):
        """Get network counters safely."""
        try:
            return psutil.net_io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def _update_baselines(self, snapshot: SystemSnapshot):
        """Update performance baselines."""
        if 'cpu_usage' not in self.baselines:
            self.baselines = {
                'cpu_usage': [],
                'memory_percent': [],
                'gpu_utilization': [],
                'disk_io': [],
                'network_io': []
            }
        
        self.baselines['cpu_usage'].append(snapshot.cpu_percent)
        self.baselines['memory_percent'].append(snapshot.memory_percent)
        
        if snapshot.gpu_utilization is not None:
            self.baselines['gpu_utilization'].append(snapshot.gpu_utilization)
        
        self.baselines['disk_io'].append(snapshot.disk_io_read_mb + snapshot.disk_io_write_mb)
        self.baselines['network_io'].append(snapshot.network_io_recv_mb + snapshot.network_io_sent_mb)
        
        # Keep rolling window
        for key in self.baselines:
            if len(self.baselines[key]) > 1000:
                self.baselines[key] = self.baselines[key][-500:]
    
    def detect_bottlenecks(self) -> List[PerformanceIssue]:
        """Detect system bottlenecks from monitoring data."""
        if len(self.snapshots) < 5:
            return []
        
        issues = []
        recent_snapshots = list(self.snapshots)[-10:]  # Last 10 snapshots
        
        # CPU bottleneck detection
        avg_cpu = np.mean([s.cpu_percent for s in recent_snapshots])
        if avg_cpu > self.thresholds['cpu_high']:
            issues.append(PerformanceIssue(
                type=BottleneckType.CPU_INTENSIVE,
                severity=min(1.0, avg_cpu / 100.0),
                description=f"High CPU usage: {avg_cpu:.1f}% average",
                location="system",
                timestamp=time.time(),
                metrics={'avg_cpu_percent': avg_cpu},
                suggestions=[
                    "Profile CPU-intensive functions",
                    "Consider parallel processing",
                    "Optimize hot code paths"
                ]
            ))
        
        # Memory bottleneck detection
        avg_memory = np.mean([s.memory_percent for s in recent_snapshots])
        if avg_memory > self.thresholds['memory_high']:
            issues.append(PerformanceIssue(
                type=BottleneckType.MEMORY_BOUND,
                severity=min(1.0, avg_memory / 100.0),
                description=f"High memory usage: {avg_memory:.1f}% average",
                location="system",
                timestamp=time.time(),
                metrics={'avg_memory_percent': avg_memory},
                suggestions=[
                    "Check for memory leaks",
                    "Optimize data structures",
                    "Implement memory pooling"
                ]
            ))
        
        # GPU bottleneck detection
        gpu_snapshots = [s for s in recent_snapshots if s.gpu_utilization is not None]
        if gpu_snapshots:
            avg_gpu = np.mean([s.gpu_utilization for s in gpu_snapshots])
            if avg_gpu > self.thresholds['gpu_high']:
                issues.append(PerformanceIssue(
                    type=BottleneckType.GPU_BOUND,
                    severity=min(1.0, avg_gpu / 100.0),
                    description=f"High GPU utilization: {avg_gpu:.1f}% average",
                    location="system",
                    timestamp=time.time(),
                    metrics={'avg_gpu_utilization': avg_gpu},
                    suggestions=[
                        "Optimize GPU kernel launches",
                        "Check for memory transfers",
                        "Consider mixed precision training"
                    ]
                ))
        
        # I/O bottleneck detection
        avg_disk_io = np.mean([s.disk_io_read_mb + s.disk_io_write_mb for s in recent_snapshots])
        if avg_disk_io > 100.0:  # > 100 MB/s
            issues.append(PerformanceIssue(
                type=BottleneckType.IO_BOUND,
                severity=min(1.0, avg_disk_io / 500.0),  # Normalize by 500 MB/s
                description=f"High disk I/O: {avg_disk_io:.1f} MB/s average",
                location="system",
                timestamp=time.time(),
                metrics={'avg_disk_io_mb': avg_disk_io},
                suggestions=[
                    "Optimize file I/O patterns",
                    "Consider caching frequently accessed data",
                    "Use async I/O where possible"
                ]
            ))
        
        return sorted(issues, key=lambda x: x.severity, reverse=True)
    
    def _handle_anomalies(self, anomalies: List[str]):
        """Handle detected anomalies."""
        for anomaly in anomalies:
            logger.warning(f"System anomaly detected: {anomaly}")
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add callback for system alerts."""
        self.alert_callbacks.append(callback)
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report."""
        if not self.snapshots:
            return {}
        
        recent_snapshots = list(self.snapshots)[-60:]  # Last minute
        
        return {
            'timestamp': time.time(),
            'monitoring_duration_hours': len(self.snapshots) * self.monitoring_interval / 3600,
            'current_snapshot': recent_snapshots[-1]._asdict() if recent_snapshots else {},
            'averages': {
                'cpu_percent': np.mean([s.cpu_percent for s in recent_snapshots]),
                'memory_percent': np.mean([s.memory_percent for s in recent_snapshots]),
                'gpu_utilization': (
                    np.mean([s.gpu_utilization for s in recent_snapshots if s.gpu_utilization is not None])
                    if any(s.gpu_utilization is not None for s in recent_snapshots) else None
                ),
                'disk_io_mb': np.mean([s.disk_io_read_mb + s.disk_io_write_mb for s in recent_snapshots]),
                'network_io_mb': np.mean([s.network_io_recv_mb + s.network_io_sent_mb for s in recent_snapshots])
            },
            'peaks': {
                'max_cpu_percent': np.max([s.cpu_percent for s in recent_snapshots]),
                'max_memory_percent': np.max([s.memory_percent for s in recent_snapshots]),
                'max_gpu_utilization': (
                    np.max([s.gpu_utilization for s in recent_snapshots if s.gpu_utilization is not None])
                    if any(s.gpu_utilization is not None for s in recent_snapshots) else None
                )
            },
            'bottlenecks': [issue._asdict() for issue in self.detect_bottlenecks()],
            'thresholds': self.thresholds
        }


class AnomalyDetector:
    """Simple anomaly detection for system metrics."""
    
    def __init__(self, window_size: int = 50, threshold_std: float = 3.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
    
    def detect_anomalies(self, snapshot: SystemSnapshot) -> List[str]:
        """Detect anomalies in system snapshot."""
        anomalies = []
        
        # Track metrics
        self.metric_history['cpu_percent'].append(snapshot.cpu_percent)
        self.metric_history['memory_percent'].append(snapshot.memory_percent)
        
        if snapshot.gpu_utilization is not None:
            self.metric_history['gpu_utilization'].append(snapshot.gpu_utilization)
        
        # Detect anomalies (simplified Z-score approach)
        for metric_name, values in self.metric_history.items():
            if len(values) >= self.window_size:
                current_value = values[-1]
                historical_values = list(values)[:-1]  # Exclude current value
                
                mean_val = np.mean(historical_values)
                std_val = np.std(historical_values)
                
                if std_val > 0:
                    z_score = abs((current_value - mean_val) / std_val)
                    if z_score > self.threshold_std:
                        anomalies.append(f"{metric_name}_anomaly: {current_value:.1f} (Z-score: {z_score:.1f})")
        
        return anomalies


class PerformanceAnalyzer:
    """Main performance analyzer combining profiling and monitoring."""
    
    def __init__(self):
        self.profiler = AdvancedProfiler(enable_memory_profiling=True)
        self.monitor = SystemMonitor(monitoring_interval=1.0)
        self.analysis_results = deque(maxlen=100)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        logger.info("Performance analyzer initialized")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        # Get profiling issues
        profiling_issues = self.profiler.identify_performance_issues()
        
        # Get system bottlenecks
        system_issues = self.monitor.detect_bottlenecks()
        
        # Combine and prioritize issues
        all_issues = profiling_issues + system_issues
        critical_issues = [issue for issue in all_issues if issue.severity > 0.7]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        # System report
        system_report = self.monitor.get_system_report()
        
        analysis = {
            'timestamp': time.time(),
            'total_issues': len(all_issues),
            'critical_issues': len(critical_issues),
            'performance_issues': [issue._asdict() for issue in all_issues],
            'recommendations': recommendations,
            'system_report': system_report,
            'profiling_summary': {
                'active_profiles': len(self.profiler.active_profiles),
                'total_profile_results': len(self.profiler.profile_results),
                'monitored_functions': len(self.profiler.function_stats)
            }
        }
        
        self.analysis_results.append(analysis)
        
        return analysis
    
    def _generate_recommendations(self, issues: List[PerformanceIssue]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Count issue types
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue.type] += 1
        
        # Generate type-specific recommendations
        if issue_counts[BottleneckType.CPU_INTENSIVE] > 0:
            recommendations.append("Consider implementing parallel processing or async operations")
            recommendations.append("Profile CPU-intensive functions for optimization opportunities")
        
        if issue_counts[BottleneckType.MEMORY_BOUND] > 0:
            recommendations.append("Implement memory pooling and object reuse")
            recommendations.append("Review data structures for memory efficiency")
        
        if issue_counts[BottleneckType.GPU_BOUND] > 0:
            recommendations.append("Optimize GPU kernel launches and memory transfers")
            recommendations.append("Consider mixed precision training to reduce memory usage")
        
        if issue_counts[BottleneckType.IO_BOUND] > 0:
            recommendations.append("Implement async I/O and caching strategies")
            recommendations.append("Optimize file access patterns and batch operations")
        
        if issue_counts[BottleneckType.RESOURCE_LEAK] > 0:
            recommendations.append("Review resource cleanup and exception handling")
            recommendations.append("Implement automated resource management")
        
        # Add general recommendations
        if len(issues) > 5:
            recommendations.append("Consider implementing auto-scaling to handle load spikes")
            recommendations.append("Review system architecture for bottlenecks")
        
        return list(set(recommendations))  # Remove duplicates
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_analyses = [
            analysis for analysis in self.analysis_results
            if analysis['timestamp'] >= cutoff_time
        ]
        
        if not recent_analyses:
            return {}
        
        # Calculate trends
        issue_counts = [analysis['total_issues'] for analysis in recent_analyses]
        critical_counts = [analysis['critical_issues'] for analysis in recent_analyses]
        timestamps = [analysis['timestamp'] for analysis in recent_analyses]
        
        return {
            'time_period_hours': hours,
            'analyses_count': len(recent_analyses),
            'avg_issues_per_analysis': np.mean(issue_counts) if issue_counts else 0,
            'avg_critical_per_analysis': np.mean(critical_counts) if critical_counts else 0,
            'issue_trend': self._calculate_trend(timestamps, issue_counts),
            'critical_trend': self._calculate_trend(timestamps, critical_counts),
            'most_common_issue_types': self._get_most_common_issues(recent_analyses)
        }
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate trend slope (positive = increasing, negative = decreasing)."""
        if len(timestamps) < 2:
            return 0.0
        
        try:
            # Simple linear regression
            return np.polyfit(timestamps, values, 1)[0]
        except:
            return 0.0
    
    def _get_most_common_issues(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get most common issue types across analyses."""
        issue_type_counts = defaultdict(int)
        
        for analysis in analyses:
            for issue in analysis['performance_issues']:
                issue_type_counts[issue['type']] += 1
        
        return dict(sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True))
    
    def cleanup(self):
        """Cleanup analyzer resources."""
        self.monitor.stop_monitoring()
        logger.info("Performance analyzer cleanup completed")


def demo_advanced_monitoring():
    """Demonstrate advanced monitoring capabilities."""
    print("🔍 Advanced Performance Monitoring Demo")
    print("=" * 50)
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer()
    
    # Demo function profiling
    @analyzer.profiler.profile_function("cpu_intensive_demo", include_memory=True)
    def cpu_intensive_task(n: int) -> float:
        """Simulate CPU-intensive task."""
        total = 0.0
        for i in range(n * 1000):
            total += np.sin(i) * np.cos(i)
        return total
    
    @analyzer.profiler.profile_function("memory_intensive_demo", include_memory=True)
    def memory_intensive_task(size: int) -> List[np.ndarray]:
        """Simulate memory-intensive task."""
        arrays = []
        for i in range(size):
            array = np.random.random((100, 100))
            arrays.append(array)
        return arrays
    
    print("📊 Running profiled functions...")
    
    # Run tasks to generate profiling data
    for i in range(5):
        result1 = cpu_intensive_task(1000 + i * 500)
        result2 = memory_intensive_task(10 + i * 2)
        time.sleep(0.5)
    
    print("⏱️  Waiting for monitoring data...")
    time.sleep(5)  # Let monitoring collect data
    
    # Perform analysis
    analysis = analyzer.analyze_performance()
    
    print(f"\n📈 Performance Analysis Results:")
    print(f"  Total issues found: {analysis['total_issues']}")
    print(f"  Critical issues: {analysis['critical_issues']}")
    print(f"  Monitored functions: {analysis['profiling_summary']['monitored_functions']}")
    
    # Show performance issues
    if analysis['performance_issues']:
        print(f"\n🚨 Top Performance Issues:")
        for i, issue in enumerate(analysis['performance_issues'][:3], 1):
            print(f"  {i}. {issue['type']} - {issue['description']}")
            print(f"     Severity: {issue['severity']:.2f}")
            print(f"     Location: {issue['location']}")
    
    # Show recommendations
    if analysis['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    # Show system stats
    system_report = analysis['system_report']
    if 'averages' in system_report:
        averages = system_report['averages']
        print(f"\n📊 System Performance:")
        print(f"  Average CPU: {averages.get('cpu_percent', 0):.1f}%")
        print(f"  Average Memory: {averages.get('memory_percent', 0):.1f}%")
        if averages.get('gpu_utilization') is not None:
            print(f"  Average GPU: {averages['gpu_utilization']:.1f}%")
    
    # Get function analytics
    cpu_analytics = analyzer.profiler.get_function_analytics("cpu_intensive_demo")
    if cpu_analytics:
        print(f"\n⚡ CPU Task Analytics:")
        print(f"  Total calls: {cpu_analytics['total_calls']}")
        print(f"  Average time: {cpu_analytics['avg_execution_time']:.3f}s")
        print(f"  Memory impact: {cpu_analytics['avg_memory_delta']:.1f}MB")
    
    memory_analytics = analyzer.profiler.get_function_analytics("memory_intensive_demo")
    if memory_analytics:
        print(f"\n💾 Memory Task Analytics:")
        print(f"  Total calls: {memory_analytics['total_calls']}")
        print(f"  Average memory: {memory_analytics['avg_memory_delta']:.1f}MB")
        print(f"  Memory leak indicator: {memory_analytics['memory_leak_indicator']:.1f}MB")
    
    # Cleanup
    analyzer.cleanup()
    
    print(f"\n✅ Advanced monitoring demo completed!")


if __name__ == "__main__":
    demo_advanced_monitoring()