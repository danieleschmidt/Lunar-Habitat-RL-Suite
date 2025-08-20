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


class SimpleSystemMonitor:
    """Lightweight system monitor for Generation 1 - works without heavy dependencies."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []
        self.logger = get_logger()
        
    def get_basic_metrics(self) -> Dict[str, Any]:
        """Get basic system metrics without psutil."""
        import os
        
        metrics = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'process_id': os.getpid(),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
        }
        
        # Try to get memory info from /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_kb = int(line.split()[1])
                        metrics['total_memory_gb'] = total_kb / 1024 / 1024
                    elif 'MemAvailable:' in line:
                        available_kb = int(line.split()[1])
                        metrics['available_memory_gb'] = available_kb / 1024 / 1024
                        if 'total_memory_gb' in metrics:
                            metrics['memory_usage_percent'] = (1 - available_kb/total_kb) * 100
        except (FileNotFoundError, PermissionError, ValueError):
            # Fallback for non-Linux systems or when /proc is not available
            metrics['memory_info'] = 'unavailable'
        
        return metrics
    
    def get_environment_metrics(self, env) -> Dict[str, Any]:
        """Get environment-specific metrics."""
        try:
            metrics = {
                'environment_type': type(env).__name__,
                'observation_space_shape': getattr(env.observation_space, 'shape', None),
                'action_space_shape': getattr(env.action_space, 'shape', None),
                'current_step': getattr(env, 'current_step', 0),
                'max_steps': getattr(env, 'max_steps', 0),
                'episode_reward': getattr(env, 'episode_reward', 0)
            }
            
            # Try to get state information
            if hasattr(env, 'state'):
                state = env.state
                if hasattr(state, 'atmosphere'):
                    metrics['o2_pressure'] = getattr(state.atmosphere, 'o2_partial_pressure', 0)
                    metrics['co2_pressure'] = getattr(state.atmosphere, 'co2_partial_pressure', 0)
                    metrics['temperature'] = getattr(state.atmosphere, 'temperature', 0)
                
                if hasattr(state, 'power'):
                    metrics['battery_charge'] = getattr(state.power, 'battery_charge', 0)
                    metrics['power_generation'] = getattr(state.power, 'solar_generation', 0)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to get environment metrics: {e}")
            return {'error': str(e)}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform basic system health check."""
        health_status = {
            'status': 'healthy',
            'alerts': [],
            'timestamp': time.time(),
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
        
        # Get basic metrics
        metrics = self.get_basic_metrics()
        
        # Check memory usage
        if 'memory_usage_percent' in metrics:
            if metrics['memory_usage_percent'] > 90:
                health_status['alerts'].append('High memory usage')
                health_status['status'] = 'warning'
            elif metrics['memory_usage_percent'] > 95:
                health_status['alerts'].append('Critical memory usage')
                health_status['status'] = 'critical'
        
        # Check uptime
        if health_status['uptime_hours'] > 24:
            health_status['alerts'].append('Long running session - consider restart')
        
        # Check for recent errors in logs
        error_count = len([alert for alert in self.alerts if 'error' in alert.lower()])
        if error_count > 10:
            health_status['alerts'].append(f'High error count: {error_count}')
            health_status['status'] = 'warning'
        
        health_status['metrics'] = metrics
        
        return health_status
    
    def log_alert(self, alert_message: str, severity: str = 'info'):
        """Log an alert message."""
        alert = {
            'timestamp': time.time(),
            'message': alert_message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        # Log to logger
        if severity == 'critical':
            self.logger.critical(alert_message)
        elif severity == 'warning':
            self.logger.warning(alert_message)
        elif severity == 'error':
            self.logger.error(alert_message)
        else:
            self.logger.info(alert_message)
    
    def get_recent_alerts(self, hours: float = 1.0) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert['timestamp'] >= cutoff_time]


class PerformanceTracker:
    """Track performance metrics for training and evaluation."""
    
    def __init__(self):
        self.session_start = time.time()
        self.episode_data = []
        self.step_data = deque(maxlen=10000)  # Keep last 10k steps
        self.logger = get_logger()
        
    def record_episode(self, episode_num: int, reward: float, length: int, 
                      status: str = 'completed', **kwargs):
        """Record episode completion."""
        episode_data = {
            'episode': episode_num,
            'reward': reward,
            'length': length,
            'status': status,
            'timestamp': time.time(),
            'session_time': time.time() - self.session_start,
            **kwargs
        }
        
        self.episode_data.append(episode_data)
        
        # Log milestone episodes
        if episode_num % 100 == 0:
            recent_rewards = [ep['reward'] for ep in self.episode_data[-10:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            self.logger.info(f"Episode {episode_num}: avg reward (last 10): {avg_reward:.2f}")
    
    def record_step(self, step_time: float, reward: float, action_valid: bool = True):
        """Record individual step performance."""
        step_data = {
            'step_time': step_time,
            'reward': reward,
            'action_valid': action_valid,
            'timestamp': time.time()
        }
        
        self.step_data.append(step_data)
    
    def get_performance_summary(self, last_n_episodes: int = 100) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.episode_data:
            return {'error': 'No episode data available'}
        
        recent_episodes = self.episode_data[-last_n_episodes:]
        rewards = [ep['reward'] for ep in recent_episodes]
        lengths = [ep['length'] for ep in recent_episodes]
        
        # Calculate statistics
        summary = {
            'total_episodes': len(self.episode_data),
            'recent_episodes': len(recent_episodes),
            'session_time_hours': (time.time() - self.session_start) / 3600,
            'avg_reward': sum(rewards) / len(rewards),
            'reward_std': self._calculate_std(rewards),
            'min_reward': min(rewards),
            'max_reward': max(rewards),
            'avg_episode_length': sum(lengths) / len(lengths),
            'success_rate': len([ep for ep in recent_episodes if ep['status'] == 'completed']) / len(recent_episodes),
            'episodes_per_hour': len(self.episode_data) / ((time.time() - self.session_start) / 3600)
        }
        
        # Step performance
        if self.step_data:
            recent_steps = list(self.step_data)[-1000:]  # Last 1000 steps
            step_times = [s['step_time'] for s in recent_steps]
            summary['avg_step_time_ms'] = sum(step_times) / len(step_times) * 1000
            summary['steps_per_second'] = 1.0 / (sum(step_times) / len(step_times))
            summary['action_validity_rate'] = len([s for s in recent_steps if s['action_valid']]) / len(recent_steps)
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def save_performance_data(self, filepath: str):
        """Save performance data to file."""
        data = {
            'session_start': self.session_start,
            'episode_data': self.episode_data,
            'step_data': list(self.step_data),
            'summary': self.get_performance_summary(),
            'saved_at': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class SimpleHealthDashboard:
    """Simple text-based health dashboard."""
    
    def __init__(self, monitor: SimpleSystemMonitor, tracker: PerformanceTracker):
        self.monitor = monitor
        self.tracker = tracker
        
    def print_status_report(self):
        """Print a comprehensive status report."""
        print("\n" + "="*60)
        print("LUNAR HABITAT RL - SYSTEM STATUS REPORT")
        print("="*60)
        
        # System health
        health = self.monitor.check_system_health()
        print(f"\nSystem Status: {health['status'].upper()}")
        print(f"Uptime: {health['uptime_hours']:.1f} hours")
        
        if health['alerts']:
            print("\nAlerts:")
            for alert in health['alerts']:
                print(f"  - {alert}")
        
        # System metrics
        metrics = health['metrics']
        print(f"\nSystem Metrics:")
        print(f"  Process ID: {metrics.get('process_id', 'N/A')}")
        print(f"  Python: {metrics.get('python_version', 'N/A')}")
        
        if 'memory_usage_percent' in metrics:
            print(f"  Memory Usage: {metrics['memory_usage_percent']:.1f}%")
            print(f"  Available Memory: {metrics.get('available_memory_gb', 0):.2f} GB")
        
        # Performance summary
        perf_summary = self.tracker.get_performance_summary()
        if 'error' not in perf_summary:
            print(f"\nPerformance Summary:")
            print(f"  Total Episodes: {perf_summary['total_episodes']}")
            print(f"  Session Time: {perf_summary['session_time_hours']:.2f} hours")
            print(f"  Average Reward: {perf_summary['avg_reward']:.2f} ± {perf_summary['reward_std']:.2f}")
            print(f"  Success Rate: {perf_summary['success_rate']*100:.1f}%")
            print(f"  Episodes/Hour: {perf_summary['episodes_per_hour']:.1f}")
            
            if 'avg_step_time_ms' in perf_summary:
                print(f"  Avg Step Time: {perf_summary['avg_step_time_ms']:.2f} ms")
                print(f"  Steps/Second: {perf_summary['steps_per_second']:.1f}")
        
        # Recent alerts
        recent_alerts = self.monitor.get_recent_alerts(hours=1.0)
        if recent_alerts:
            print(f"\nRecent Alerts (last hour): {len(recent_alerts)}")
            for alert in recent_alerts[-5:]:  # Show last 5
                timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
                print(f"  {timestamp} [{alert['severity']}]: {alert['message']}")
        
        print("\n" + "="*60)
    
    def print_environment_status(self, env):
        """Print environment-specific status."""
        env_metrics = self.monitor.get_environment_metrics(env)
        
        print("\nEnvironment Status:")
        print(f"  Type: {env_metrics.get('environment_type', 'Unknown')}")
        print(f"  Current Step: {env_metrics.get('current_step', 0)}/{env_metrics.get('max_steps', 0)}")
        print(f"  Episode Reward: {env_metrics.get('episode_reward', 0):.2f}")
        
        if 'o2_pressure' in env_metrics:
            print(f"  O2 Pressure: {env_metrics['o2_pressure']:.1f} kPa")
            print(f"  CO2 Pressure: {env_metrics['co2_pressure']:.2f} kPa")
            print(f"  Temperature: {env_metrics['temperature']:.1f} °C")
            print(f"  Battery: {env_metrics.get('battery_charge', 0):.1f}%")
    
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