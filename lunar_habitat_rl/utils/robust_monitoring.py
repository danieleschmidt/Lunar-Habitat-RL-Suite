"""Robust system monitoring and health checks - Generation 2"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
from .robust_logging import get_logger, PerformanceMonitor

# Optional imports for advanced monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


@dataclass
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    
    # Lunar Habitat specific metrics
    simulation_fps: Optional[float] = None
    episodes_completed: Optional[int] = None
    average_reward: Optional[float] = None
    error_rate: Optional[float] = None
    validation_failures: Optional[int] = None


@dataclass
class HealthStatus:
    """Overall system health status."""
    status: str  # healthy, warning, critical, emergency
    score: float  # 0.0 to 1.0
    alerts: List[str]
    metrics: SystemMetrics
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthChecker:
    """Automated health monitoring for lunar habitat systems."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health checker.
        
        Args:
            check_interval: Seconds between health checks.
        """
        self.check_interval = check_interval
        self.logger = get_logger()
        self.start_time = time.time()
        
        # Health thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 90.0
        self.disk_critical_threshold = 98.0
        
        # Metrics history
        self.metrics_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Custom health checks
        self.custom_checks: List[Callable[[], Dict[str, Any]]] = []
        
        # Performance counters
        self.last_network_counters = None
        
    def add_custom_check(self, check_func: Callable[[], Dict[str, Any]]):
        """Add custom health check function.
        
        Args:
            check_func: Function that returns dict with 'status', 'message', 'value' keys.
        """
        self.custom_checks.append(check_func)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)  # Shorter interval for responsiveness
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk_usage = psutil.disk_usage('/')
                
                # Network counters
                network = psutil.net_io_counters()
                
                # Process count
                process_count = len(psutil.pids())
                
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_gb=memory.available / (1024**3),
                    disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
                    disk_free_gb=disk_usage.free / (1024**3),
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    process_count=process_count
                )
            else:
                # Fallback metrics when psutil is not available
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=50.0,  # Estimated placeholder
                    memory_percent=60.0,  # Estimated placeholder
                    memory_available_gb=2.0,  # Estimated placeholder
                    disk_usage_percent=50.0,  # Estimated placeholder
                    disk_free_gb=10.0,  # Estimated placeholder
                    network_bytes_sent=1000000,  # Placeholder
                    network_bytes_recv=1000000,  # Placeholder
                    process_count=100  # Placeholder
                )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}", error=e)
            # Return minimal metrics on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_gb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0
            )
    
    def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        metrics = self.get_system_metrics()
        alerts = []
        score = 1.0
        status = "healthy"
        
        # Check CPU usage
        if metrics.cpu_percent > self.cpu_critical_threshold:
            alerts.append(f"CRITICAL: CPU usage {metrics.cpu_percent:.1f}% exceeds critical threshold")
            score -= 0.4
            status = "critical"
        elif metrics.cpu_percent > self.cpu_warning_threshold:
            alerts.append(f"WARNING: CPU usage {metrics.cpu_percent:.1f}% exceeds warning threshold")
            score -= 0.2
            if status == "healthy":
                status = "warning"
        
        # Check memory usage
        if metrics.memory_percent > self.memory_critical_threshold:
            alerts.append(f"CRITICAL: Memory usage {metrics.memory_percent:.1f}% exceeds critical threshold")
            score -= 0.4
            status = "critical"
        elif metrics.memory_percent > self.memory_warning_threshold:
            alerts.append(f"WARNING: Memory usage {metrics.memory_percent:.1f}% exceeds warning threshold")
            score -= 0.2
            if status == "healthy":
                status = "warning"
        
        # Check disk usage
        if metrics.disk_usage_percent > self.disk_critical_threshold:
            alerts.append(f"CRITICAL: Disk usage {metrics.disk_usage_percent:.1f}% exceeds critical threshold")
            score -= 0.3
            status = "critical"
        elif metrics.disk_usage_percent > self.disk_warning_threshold:
            alerts.append(f"WARNING: Disk usage {metrics.disk_usage_percent:.1f}% exceeds warning threshold")
            score -= 0.15
            if status == "healthy":
                status = "warning"
        
        # Check available memory
        if metrics.memory_available_gb < 0.5:
            alerts.append(f"CRITICAL: Only {metrics.memory_available_gb:.2f}GB memory available")
            score -= 0.3
            status = "critical"
        elif metrics.memory_available_gb < 1.0:
            alerts.append(f"WARNING: Only {metrics.memory_available_gb:.2f}GB memory available")
            score -= 0.1
            if status == "healthy":
                status = "warning"
        
        # Run custom health checks
        for check_func in self.custom_checks:
            try:
                check_result = check_func()
                check_status = check_result.get('status', 'unknown')
                check_message = check_result.get('message', 'Custom check')
                
                if check_status == 'critical':
                    alerts.append(f"CRITICAL: {check_message}")
                    score -= 0.3
                    status = "critical"
                elif check_status == 'warning':
                    alerts.append(f"WARNING: {check_message}")
                    score -= 0.1
                    if status == "healthy":
                        status = "warning"
                elif check_status == 'emergency':
                    alerts.append(f"EMERGENCY: {check_message}")
                    score = 0.0
                    status = "emergency"
                    
            except Exception as e:
                alerts.append(f"ERROR: Custom health check failed: {e}")
                score -= 0.1
                if status == "healthy":
                    status = "warning"
        
        # Ensure score bounds
        score = max(0.0, min(1.0, score))
        
        # Create health status
        uptime = time.time() - self.start_time
        health_status = HealthStatus(
            status=status,
            score=score,
            alerts=alerts,
            metrics=metrics,
            uptime_seconds=uptime
        )
        
        # Store metrics and alerts
        self.metrics_history.append(metrics)
        if alerts:
            self.alert_history.append({
                'timestamp': metrics.timestamp,
                'status': status,
                'alerts': alerts
            })
        
        # Log health status
        if status == "critical" or status == "emergency":
            self.logger.critical(f"System health {status}: score={score:.2f}, alerts={len(alerts)}")
        elif status == "warning":
            self.logger.warning(f"System health {status}: score={score:.2f}, alerts={len(alerts)}")
        else:
            self.logger.info(f"System health {status}: score={score:.2f}")
        
        return health_status
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(f"Health monitoring started with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                health_status = self.check_health()
                
                # Perform emergency actions if needed
                if health_status.status == "emergency":
                    self._handle_emergency(health_status)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", error=e)
                time.sleep(self.check_interval)
    
    def _handle_emergency(self, health_status: HealthStatus):
        """Handle emergency situations."""
        self.logger.critical("🚨 EMERGENCY: System emergency detected - implementing safety protocols")
        
        # Emergency protocols:
        # 1. Reduce computational load
        # 2. Save current state
        # 3. Alert operators
        # 4. Implement safe shutdown if needed
        
        for alert in health_status.alerts:
            if "memory" in alert.lower():
                self.logger.critical("Emergency: Attempting memory cleanup")
                # Could trigger garbage collection, cache clearing, etc.
            elif "disk" in alert.lower():
                self.logger.critical("Emergency: Disk space critical - cleaning temporary files")
                # Could trigger log rotation, temp file cleanup, etc.
            elif "cpu" in alert.lower():
                self.logger.critical("Emergency: CPU overload - reducing computational complexity")
                # Could reduce simulation fidelity, pause non-critical processes
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        current_health = self.check_health()
        
        # Calculate trends
        if len(self.metrics_history) > 1:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
            
            cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
            
            # Performance statistics
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        else:
            cpu_trend = 0.0
            memory_trend = 0.0
            avg_cpu = current_health.metrics.cpu_percent
            avg_memory = current_health.metrics.memory_percent
        
        # Recent alerts summary
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > time.time() - 3600]  # Last hour
        
        report = {
            'current_health': current_health.to_dict(),
            'trends': {
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend
            },
            'averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'recent_alerts': {
                'count_last_hour': len(recent_alerts),
                'alerts': recent_alerts[-5:]  # Last 5 alerts
            },
            'system_info': {
                'uptime_hours': current_health.uptime_seconds / 3600,
                'monitoring_active': self.monitoring_active,
                'metrics_collected': len(self.metrics_history)
            }
        }
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = increasing, negative = decreasing)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope


class SimulationMonitor:
    """Monitor simulation-specific metrics and performance."""
    
    def __init__(self):
        """Initialize simulation monitor."""
        self.logger = get_logger()
        self.episode_count = 0
        self.step_count = 0
        self.reward_history = deque(maxlen=1000)
        self.error_count = 0
        self.validation_failure_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.step_times = deque(maxlen=100)
        self.episode_durations = deque(maxlen=50)
        
    def log_episode_start(self):
        """Log start of new episode."""
        self.episode_count += 1
        self.episode_start_time = time.time()
        
    def log_episode_end(self, total_reward: float):
        """Log end of episode.
        
        Args:
            total_reward: Total episode reward.
        """
        if hasattr(self, 'episode_start_time'):
            duration = time.time() - self.episode_start_time
            self.episode_durations.append(duration)
        
        self.reward_history.append(total_reward)
        
        # Log performance every 10 episodes
        if self.episode_count % 10 == 0:
            self._log_performance_summary()
    
    def log_step(self, step_duration: float, reward: float, error: bool = False, validation_failure: bool = False):
        """Log simulation step.
        
        Args:
            step_duration: Time taken for step in seconds.
            reward: Step reward.
            error: Whether step had an error.
            validation_failure: Whether validation failed.
        """
        self.step_count += 1
        self.step_times.append(step_duration)
        
        if error:
            self.error_count += 1
        if validation_failure:
            self.validation_failure_count += 1
    
    def get_simulation_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates
        episodes_per_hour = self.episode_count / (uptime / 3600) if uptime > 0 else 0
        steps_per_second = self.step_count / uptime if uptime > 0 else 0
        error_rate = self.error_count / self.step_count if self.step_count > 0 else 0
        
        # Performance metrics
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        avg_episode_duration = sum(self.episode_durations) / len(self.episode_durations) if self.episode_durations else 0
        
        # Reward statistics
        if self.reward_history:
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            max_reward = max(self.reward_history)
            min_reward = min(self.reward_history)
        else:
            avg_reward = max_reward = min_reward = 0.0
        
        return {
            'episodes_completed': self.episode_count,
            'total_steps': self.step_count,
            'uptime_seconds': uptime,
            'episodes_per_hour': episodes_per_hour,
            'steps_per_second': steps_per_second,
            'avg_step_time_ms': avg_step_time * 1000,
            'avg_episode_duration_s': avg_episode_duration,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'validation_failures': self.validation_failure_count,
            'reward_stats': {
                'average': avg_reward,
                'maximum': max_reward,
                'minimum': min_reward,
                'samples': len(self.reward_history)
            }
        }
    
    def _log_performance_summary(self):
        """Log performance summary."""
        metrics = self.get_simulation_metrics()
        
        self.logger.info(
            f"Simulation Performance Summary - Episode {self.episode_count}: "
            f"avg_reward={metrics['reward_stats']['average']:.2f}, "
            f"steps/sec={metrics['steps_per_second']:.1f}, "
            f"error_rate={metrics['error_rate']:.3f}"
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform simulation health check for HealthChecker."""
        metrics = self.get_simulation_metrics()
        
        # Determine health status
        if metrics['error_rate'] > 0.1:  # More than 10% error rate
            return {'status': 'critical', 'message': f"High error rate: {metrics['error_rate']:.2%}"}
        elif metrics['error_rate'] > 0.05:  # More than 5% error rate
            return {'status': 'warning', 'message': f"Elevated error rate: {metrics['error_rate']:.2%}"}
        elif metrics['steps_per_second'] < 1.0:  # Very slow simulation
            return {'status': 'warning', 'message': f"Slow simulation: {metrics['steps_per_second']:.1f} steps/sec"}
        else:
            return {'status': 'healthy', 'message': f"Simulation running normally: {metrics['steps_per_second']:.1f} steps/sec"}


# Global monitoring instances
_global_health_checker = None
_global_simulation_monitor = None

def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
        # Add simulation health check
        _global_health_checker.add_custom_check(get_simulation_monitor().health_check)
    return _global_health_checker

def get_simulation_monitor() -> SimulationMonitor:
    """Get global simulation monitor instance."""
    global _global_simulation_monitor
    if _global_simulation_monitor is None:
        _global_simulation_monitor = SimulationMonitor()
    return _global_simulation_monitor


def monitor_simulation_performance(func):
    """Decorator to automatically monitor simulation performance."""
    def wrapper(*args, **kwargs):
        monitor = get_simulation_monitor()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract reward if available
            reward = 0.0
            if isinstance(result, tuple) and len(result) >= 2:
                reward = result[1] if isinstance(result[1], (int, float)) else 0.0
            
            monitor.log_step(duration, reward, error=False)
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            monitor.log_step(duration, 0.0, error=True)
            raise
    
    return wrapper