"""Performance monitoring and optimization utilities."""

import time
import psutil
import threading
import gc
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import cProfile
import pstats
from io import StringIO
import functools

from ..utils.logging import get_logger

logger = get_logger("performance")


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class ResourceUsage:
    """System resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0


class PerformanceProfiler:
    """Advanced performance profiler with detailed metrics."""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.metrics = defaultdict(list)
        self.profilers = {}
        self.timers = {}
        self._lock = threading.RLock()
        
    def start_profiling(self, name: str):
        """Start profiling a section of code."""
        if not self.enable_profiling:
            return
            
        with self._lock:
            profiler = cProfile.Profile()
            profiler.enable()
            self.profilers[name] = {
                'profiler': profiler,
                'start_time': time.time()
            }
    
    def stop_profiling(self, name: str) -> Optional[Dict[str, Any]]:
        """Stop profiling and return results."""
        if not self.enable_profiling or name not in self.profilers:
            return None
            
        with self._lock:
            profile_data = self.profilers.pop(name)
            profiler = profile_data['profiler']
            start_time = profile_data['start_time']
            
            profiler.disable()
            
            # Capture stats
            stats_buffer = StringIO()
            ps = pstats.Stats(profiler, stream=stats_buffer)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            duration = time.time() - start_time
            
            results = {
                'name': name,
                'duration': duration,
                'stats': stats_buffer.getvalue(),
                'timestamp': time.time()
            }
            
            self.record_metric(f"{name}_duration", duration, "seconds")
            
            return results
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile function execution."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                self.start_profiling(name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop_profiling(name)
            
            return wrapper
        return decorator
    
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, name)
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
            # Keep only recent metrics (sliding window)
            if len(self.metrics[name]) > 10000:
                self.metrics[name] = self.metrics[name][-5000:]
    
    def get_metric_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            metrics = self.metrics[name]
            
            # Filter by time window if specified
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        with self._lock:
            return {name: self.get_metric_stats(name) for name in self.metrics.keys()}
    
    def clear_metrics(self):
        """Clear all recorded metrics."""
        with self._lock:
            self.metrics.clear()


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.profiler.record_metric(self.operation_name, duration, "seconds")
        
        if exc_type:
            self.profiler.record_metric(f"{self.operation_name}_errors", 1, "count")


class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.resource_history = deque(maxlen=history_size)
        self.monitoring = False
        self.monitor_thread = None
        self._lock = threading.RLock()
        
        # Initialize baseline measurements
        self._initial_io_counters = self._get_io_counters()
        self._initial_network_counters = self._get_network_counters()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                usage = self._collect_resource_usage()
                with self._lock:
                    self.resource_history.append(usage)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        process = psutil.Process()
        
        # CPU and memory
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # Disk I/O
        io_counters = self._get_io_counters()
        disk_io_read_mb = 0.0
        disk_io_write_mb = 0.0
        
        if io_counters and self._initial_io_counters:
            disk_io_read_mb = (io_counters.read_bytes - self._initial_io_counters.read_bytes) / 1024 / 1024
            disk_io_write_mb = (io_counters.write_bytes - self._initial_io_counters.write_bytes) / 1024 / 1024
        
        # Network I/O
        network_counters = self._get_network_counters()
        network_sent_mb = 0.0
        network_recv_mb = 0.0
        
        if network_counters and self._initial_network_counters:
            network_sent_mb = (network_counters.bytes_sent - self._initial_network_counters.bytes_sent) / 1024 / 1024
            network_recv_mb = (network_counters.bytes_recv - self._initial_network_counters.bytes_recv) / 1024 / 1024
        
        # GPU usage (if available)
        gpu_memory_mb, gpu_utilization = self._get_gpu_usage()
        
        return ResourceUsage(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization
        )
    
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
    
    def _get_gpu_usage(self) -> Tuple[float, float]:
        """Get GPU usage if available."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                # GPU utilization is harder to get, use memory as proxy
                gpu_utilization = gpu_memory / (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024) * 100
                return gpu_memory, gpu_utilization
        except ImportError:
            pass
        
        return 0.0, 0.0
    
    def get_current_usage(self) -> Optional[ResourceUsage]:
        """Get most recent resource usage."""
        with self._lock:
            return self.resource_history[-1] if self.resource_history else None
    
    def get_usage_stats(self, window_seconds: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Get resource usage statistics."""
        with self._lock:
            if not self.resource_history:
                return {}
            
            history = list(self.resource_history)
            
            # Filter by time window
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                history = [usage for usage in history if usage.timestamp >= cutoff_time]
            
            if not history:
                return {}
            
            stats = {}
            
            # CPU stats
            cpu_values = [usage.cpu_percent for usage in history]
            stats['cpu'] = {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            }
            
            # Memory stats
            memory_values = [usage.memory_mb for usage in history]
            stats['memory'] = {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'current': memory_values[-1] if memory_values else 0
            }
            
            # GPU stats
            if any(usage.gpu_memory_mb > 0 for usage in history):
                gpu_memory_values = [usage.gpu_memory_mb for usage in history]
                gpu_util_values = [usage.gpu_utilization for usage in history]
                
                stats['gpu'] = {
                    'memory_mean': np.mean(gpu_memory_values),
                    'memory_max': np.max(gpu_memory_values),
                    'utilization_mean': np.mean(gpu_util_values),
                    'utilization_max': np.max(gpu_util_values)
                }
            
            return stats
    
    def detect_resource_issues(self) -> List[str]:
        """Detect potential resource issues."""
        issues = []
        current_usage = self.get_current_usage()
        
        if not current_usage:
            return issues
        
        # High CPU usage
        if current_usage.cpu_percent > 90:
            issues.append(f"High CPU usage: {current_usage.cpu_percent:.1f}%")
        
        # High memory usage
        if current_usage.memory_percent > 85:
            issues.append(f"High memory usage: {current_usage.memory_percent:.1f}%")
        
        # High GPU memory usage
        if current_usage.gpu_memory_mb > 0:
            try:
                import torch
                if torch.cuda.is_available():
                    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    gpu_memory_percent = (current_usage.gpu_memory_mb / total_gpu_memory) * 100
                    if gpu_memory_percent > 90:
                        issues.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
            except ImportError:
                pass
        
        return issues


class OptimizationManager:
    """Manages performance optimizations and adaptive scaling."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.monitor = ResourceMonitor()
        self.optimizations = {}
        self.auto_optimize = True
        self._optimization_lock = threading.RLock()
        
        # Optimization thresholds
        self.cpu_threshold_high = 85.0
        self.memory_threshold_high = 80.0
        self.latency_threshold_ms = 100.0
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitor.stop_monitoring()
    
    def register_optimization(self, name: str, optimization_func: Callable[[Dict[str, Any]], bool]):
        """Register an optimization function."""
        with self._optimization_lock:
            self.optimizations[name] = optimization_func
    
    def apply_optimizations(self, force: bool = False) -> Dict[str, bool]:
        """Apply registered optimizations based on current performance."""
        if not self.auto_optimize and not force:
            return {}
        
        # Collect performance data
        resource_stats = self.monitor.get_usage_stats(window_seconds=300)  # 5 minutes
        metric_stats = self.profiler.get_all_metrics()
        
        optimization_results = {}
        
        with self._optimization_lock:
            for name, optimization_func in self.optimizations.items():
                try:
                    context = {
                        'resource_stats': resource_stats,
                        'metric_stats': metric_stats,
                        'thresholds': {
                            'cpu_high': self.cpu_threshold_high,
                            'memory_high': self.memory_threshold_high,
                            'latency_ms': self.latency_threshold_ms
                        }
                    }
                    
                    result = optimization_func(context)
                    optimization_results[name] = result
                    
                    if result:
                        logger.info(f"Applied optimization: {name}")
                    
                except Exception as e:
                    logger.error(f"Error applying optimization {name}: {e}")
                    optimization_results[name] = False
        
        return optimization_results
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on current performance."""
        suggestions = []
        
        # Check resource usage
        current_usage = self.monitor.get_current_usage()
        if current_usage:
            if current_usage.cpu_percent > self.cpu_threshold_high:
                suggestions.append("Consider reducing simulation fidelity or enabling parallel processing")
            
            if current_usage.memory_percent > self.memory_threshold_high:
                suggestions.append("Consider enabling caching limits or garbage collection optimizations")
            
            if current_usage.gpu_memory_mb > 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                        gpu_usage_percent = (current_usage.gpu_memory_mb / total_gpu_memory) * 100
                        if gpu_usage_percent > 85:
                            suggestions.append("Consider reducing batch size or enabling mixed precision training")
                except ImportError:
                    pass
        
        # Check metric performance
        metric_stats = self.profiler.get_all_metrics()
        
        for metric_name, stats in metric_stats.items():
            if 'duration' in metric_name.lower():
                if stats.get('mean', 0) > self.latency_threshold_ms / 1000:
                    suggestions.append(f"High latency detected in {metric_name} - consider optimization")
        
        return suggestions
    
    def optimize_garbage_collection(self) -> bool:
        """Optimize Python garbage collection."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Adjust GC thresholds for better performance
            gc.set_threshold(700, 10, 10)  # More aggressive collection
            
            logger.info(f"Garbage collection optimization: collected {collected} objects")
            return True
        except Exception as e:
            logger.error(f"Error optimizing garbage collection: {e}")
            return False
    
    def optimize_numpy_threads(self, num_threads: Optional[int] = None) -> bool:
        """Optimize NumPy threading."""
        try:
            import os
            
            if num_threads is None:
                # Use half of available cores for NumPy operations
                num_threads = max(1, psutil.cpu_count() // 2)
            
            os.environ['OMP_NUM_THREADS'] = str(num_threads)
            os.environ['MKL_NUM_THREADS'] = str(num_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
            
            logger.info(f"Optimized NumPy threading: {num_threads} threads")
            return True
        except Exception as e:
            logger.error(f"Error optimizing NumPy threading: {e}")
            return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'timestamp': time.time(),
            'resource_usage': self.monitor.get_usage_stats(window_seconds=3600),  # 1 hour
            'performance_metrics': self.profiler.get_all_metrics(),
            'resource_issues': self.monitor.detect_resource_issues(),
            'optimization_suggestions': self.suggest_optimizations(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        }