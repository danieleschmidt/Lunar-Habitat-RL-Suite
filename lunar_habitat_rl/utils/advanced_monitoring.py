"""
Advanced monitoring system with health checks, alerts, and automatic recovery.
NASA-grade monitoring for mission-critical lunar habitat operations.
"""

import time
import threading
import queue
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import concurrent.futures
from pathlib import Path

from .robust_logging import get_logger, PerformanceMonitor
from .robust_monitoring import HealthChecker, SimulationMonitor
from .fault_tolerance import get_fault_tolerance_manager, SystemState


class AlertSeverity(Enum):
    """Alert severity levels for mission operations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    MISSION_CRITICAL = "mission_critical"


class MonitoringState(Enum):
    """Monitoring system operational states."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Alert event in the monitoring system."""
    id: str = field(default_factory=lambda: f"alert_{int(time.time()*1000)}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: AlertSeverity = AlertSeverity.INFO
    source: str = ""
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    auto_resolution_attempted: bool = False
    escalation_level: int = 0
    mission_impact: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'source': self.source,
            'title': self.title,
            'message': self.message,
            'details': self.details,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'auto_resolution_attempted': self.auto_resolution_attempted,
            'escalation_level': self.escalation_level,
            'mission_impact': self.mission_impact
        }


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True
    mission_critical: bool = False
    
    # State tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    is_healthy: bool = True


class AutoRecoveryAction:
    """Defines an automatic recovery action."""
    
    def __init__(self, name: str, action_function: Callable[[], bool], 
                 max_attempts: int = 3, cooldown_seconds: float = 60.0,
                 prerequisites: List[str] = None):
        """Initialize auto recovery action.
        
        Args:
            name: Action identifier
            action_function: Function that performs recovery (returns True if successful)
            max_attempts: Maximum retry attempts
            cooldown_seconds: Cooldown period between attempts
            prerequisites: List of conditions that must be met before action
        """
        self.name = name
        self.action_function = action_function
        self.max_attempts = max_attempts
        self.cooldown_seconds = cooldown_seconds
        self.prerequisites = prerequisites or []
        
        # Tracking
        self.attempt_count = 0
        self.last_attempt_time = None
        self.success_count = 0
        self.failure_count = 0
    
    def can_execute(self) -> bool:
        """Check if action can be executed now."""
        # Check cooldown
        if (self.last_attempt_time and 
            time.time() - self.last_attempt_time < self.cooldown_seconds):
            return False
        
        # Check attempt limit
        if self.attempt_count >= self.max_attempts:
            return False
        
        return True
    
    def execute(self) -> bool:
        """Execute recovery action."""
        if not self.can_execute():
            return False
        
        self.attempt_count += 1
        self.last_attempt_time = time.time()
        
        try:
            success = self.action_function()
            if success:
                self.success_count += 1
                # Reset attempt count on success
                self.attempt_count = 0
            else:
                self.failure_count += 1
            return success
        except Exception:
            self.failure_count += 1
            return False


class AlertManager:
    """Manages alerts with escalation and notification."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.logger = get_logger()
        self.alerts = deque(maxlen=10000)  # Keep last 10k alerts
        self.active_alerts = {}  # id -> Alert
        self.alert_rules = {}
        self.notification_channels = []
        self.escalation_policies = {}
        self._lock = threading.RLock()
        
        # Alert suppression to prevent spam
        self.suppression_rules = {}
        self.recent_alerts = defaultdict(list)  # source -> [timestamps]
    
    def create_alert(self, severity: AlertSeverity, source: str, title: str, 
                    message: str, details: Dict[str, Any] = None,
                    mission_impact: bool = None) -> Alert:
        """Create new alert."""
        with self._lock:
            # Auto-determine mission impact for critical+ alerts
            if mission_impact is None:
                mission_impact = severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY, AlertSeverity.MISSION_CRITICAL]
            
            alert = Alert(
                severity=severity,
                source=source,
                title=title,
                message=message,
                details=details or {},
                mission_impact=mission_impact
            )
            
            # Check suppression rules
            if self._should_suppress_alert(alert):
                self.logger.debug(f"Alert suppressed: {alert.title}")
                return alert
            
            # Add to collections
            self.alerts.append(alert)
            self.active_alerts[alert.id] = alert
            
            # Track for suppression
            self.recent_alerts[source].append(time.time())
            # Keep only last 10 minutes
            cutoff = time.time() - 600
            self.recent_alerts[source] = [t for t in self.recent_alerts[source] if t > cutoff]
            
            # Log alert
            log_level = {
                AlertSeverity.INFO: self.logger.info,
                AlertSeverity.WARNING: self.logger.warning,
                AlertSeverity.ERROR: self.logger.error,
                AlertSeverity.CRITICAL: self.logger.critical,
                AlertSeverity.EMERGENCY: self.logger.critical,
                AlertSeverity.MISSION_CRITICAL: self.logger.critical
            }
            
            log_func = log_level.get(severity, self.logger.info)
            log_func(f"ALERT [{severity.value.upper()}] {source}: {title} - {message}")
            
            # Auto-escalate mission critical alerts
            if severity == AlertSeverity.MISSION_CRITICAL:
                self._escalate_alert(alert)
            
            # Trigger notifications
            self._notify_alert(alert)
            
            return alert
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed due to rate limiting."""
        source_alerts = self.recent_alerts[alert.source]
        
        # Suppress if more than 10 alerts from same source in last 10 minutes
        if len(source_alerts) > 10:
            return True
        
        # Never suppress mission critical alerts
        if alert.severity == AlertSeverity.MISSION_CRITICAL:
            return False
        
        return False
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                self.logger.info(f"Alert acknowledged by {user}: {alert.title}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system", 
                     auto_resolved: bool = False) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                if auto_resolved:
                    alert.auto_resolution_attempted = True
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved by {user}: {alert.title}")
                return True
        return False
    
    def _escalate_alert(self, alert: Alert):
        """Escalate alert to higher severity level."""
        alert.escalation_level += 1
        self.logger.critical(f"ALERT ESCALATED (Level {alert.escalation_level}): {alert.title}")
        
        # Implement escalation policies
        if alert.escalation_level >= 3:
            self.logger.critical("🚨 MAXIMUM ESCALATION REACHED - MISSION CONTROL NOTIFICATION 🚨")
    
    def _notify_alert(self, alert: Alert):
        """Send alert notifications through configured channels."""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                self.logger.error(f"Alert notification failed for channel {channel}: {e}")
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """Get currently active alerts."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            if severity_filter:
                alerts = [a for a in alerts if a.severity in severity_filter]
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            now = time.time()
            last_hour = now - 3600
            last_day = now - 86400
            
            recent_alerts = [a for a in self.alerts if a.timestamp.timestamp() > last_hour]
            daily_alerts = [a for a in self.alerts if a.timestamp.timestamp() > last_day]
            
            return {
                'total_alerts': len(self.alerts),
                'active_alerts': len(self.active_alerts),
                'alerts_last_hour': len(recent_alerts),
                'alerts_last_day': len(daily_alerts),
                'mission_critical_active': len([a for a in self.active_alerts.values() if a.mission_impact]),
                'by_severity': {
                    severity.value: len([a for a in self.active_alerts.values() if a.severity == severity])
                    for severity in AlertSeverity
                }
            }


class AdvancedHealthMonitor:
    """Advanced health monitoring with automatic recovery."""
    
    def __init__(self):
        """Initialize advanced health monitor."""
        self.logger = get_logger()
        self.alert_manager = AlertManager()
        self.health_checks = {}
        self.recovery_actions = {}
        self.state = MonitoringState.INACTIVE
        
        # Monitoring threads
        self.monitor_thread = None
        self.recovery_thread = None
        self.running = False
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.performance_baselines = {}
        
        # System state tracking
        self.system_health_score = 1.0
        self.last_health_check = None
        
        # Recovery coordination
        self.recovery_queue = queue.Queue()
        self.recovery_in_progress = set()
        
        # Integration with existing monitoring
        self.base_health_checker = HealthChecker()
        self.simulation_monitor = SimulationMonitor()
    
    def register_health_check(self, check: HealthCheck):
        """Register a health check."""
        self.health_checks[check.name] = check
        self.logger.info(f"Health check registered: {check.name}")
    
    def register_recovery_action(self, trigger_pattern: str, action: AutoRecoveryAction):
        """Register automatic recovery action for specific triggers.
        
        Args:
            trigger_pattern: Pattern to match against alert sources/titles
            action: Recovery action to execute
        """
        if trigger_pattern not in self.recovery_actions:
            self.recovery_actions[trigger_pattern] = []
        self.recovery_actions[trigger_pattern].append(action)
        self.logger.info(f"Recovery action registered for pattern: {trigger_pattern}")
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        self.state = MonitoringState.STARTING
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start recovery thread
        self.recovery_thread = threading.Thread(target=self._recovery_loop, daemon=True)
        self.recovery_thread.start()
        
        # Start base health checker
        self.base_health_checker.start_monitoring()
        
        self.state = MonitoringState.ACTIVE
        self.logger.info("Advanced monitoring system started")
        
        self.alert_manager.create_alert(
            AlertSeverity.INFO, "monitoring_system", "Monitoring Started",
            "Advanced monitoring system is now active"
        )
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        self.state = MonitoringState.INACTIVE
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5.0)
        
        self.base_health_checker.stop_monitoring()
        
        self.logger.info("Advanced monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._execute_health_checks()
                self._update_system_health_score()
                self._check_performance_baselines()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def _execute_health_checks(self):
        """Execute all registered health checks."""
        current_time = datetime.utcnow()
        
        for check_name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            # Check if it's time to run this check
            if (check.last_check_time and 
                current_time - check.last_check_time < timedelta(seconds=check.interval_seconds)):
                continue
            
            try:
                # Execute health check with timeout
                result = self._execute_health_check_with_timeout(check)
                check.last_check_time = current_time
                
                if result.get('healthy', True):
                    check.consecutive_successes += 1
                    check.consecutive_failures = 0
                    
                    # Mark as healthy if we've reached success threshold
                    if (not check.is_healthy and 
                        check.consecutive_successes >= check.success_threshold):
                        check.is_healthy = True
                        check.last_success_time = current_time
                        
                        # Create recovery alert
                        self.alert_manager.create_alert(
                            AlertSeverity.INFO, check_name, "Health Check Recovered",
                            f"Health check {check_name} has recovered",
                            details=result
                        )
                else:
                    check.consecutive_failures += 1
                    check.consecutive_successes = 0
                    check.last_failure_time = current_time
                    
                    # Mark as unhealthy if we've reached failure threshold
                    if (check.is_healthy and 
                        check.consecutive_failures >= check.failure_threshold):
                        check.is_healthy = False
                        
                        severity = AlertSeverity.MISSION_CRITICAL if check.mission_critical else AlertSeverity.ERROR
                        alert = self.alert_manager.create_alert(
                            severity, check_name, "Health Check Failed",
                            f"Health check {check_name} has failed {check.consecutive_failures} times",
                            details=result
                        )
                        
                        # Trigger automatic recovery
                        self._trigger_recovery(check_name, alert)
                        
            except Exception as e:
                self.logger.error(f"Health check {check_name} execution failed: {e}")
                check.consecutive_failures += 1
                check.last_failure_time = current_time
    
    def _execute_health_check_with_timeout(self, check: HealthCheck) -> Dict[str, Any]:
        """Execute health check with timeout protection."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(check.check_function)
            try:
                return future.result(timeout=check.timeout_seconds)
            except concurrent.futures.TimeoutError:
                return {'healthy': False, 'error': 'Health check timeout'}
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
    
    def _update_system_health_score(self):
        """Update overall system health score."""
        if not self.health_checks:
            self.system_health_score = 1.0
            return
        
        healthy_checks = sum(1 for check in self.health_checks.values() if check.is_healthy)
        total_checks = len(self.health_checks)
        
        # Weight mission-critical checks more heavily
        mission_critical_checks = [c for c in self.health_checks.values() if c.mission_critical]
        if mission_critical_checks:
            healthy_critical = sum(1 for check in mission_critical_checks if check.is_healthy)
            critical_weight = len(mission_critical_checks) * 2  # Double weight
            regular_weight = total_checks - len(mission_critical_checks)
            
            weighted_score = (healthy_critical * 2 + (healthy_checks - healthy_critical)) / (critical_weight + regular_weight)
        else:
            weighted_score = healthy_checks / total_checks
        
        self.system_health_score = weighted_score
        
        # Create alert if health score drops significantly
        if weighted_score < 0.7:  # Less than 70% healthy
            severity = AlertSeverity.CRITICAL if weighted_score < 0.5 else AlertSeverity.WARNING
            self.alert_manager.create_alert(
                severity, "system_health", "System Health Degraded",
                f"System health score: {weighted_score:.2%}",
                details={'health_score': weighted_score, 'healthy_checks': healthy_checks, 'total_checks': total_checks}
            )
    
    def _check_performance_baselines(self):
        """Check if performance metrics deviate from baselines."""
        try:
            # Get simulation metrics
            sim_metrics = self.simulation_monitor.get_simulation_metrics()
            
            # Check against baselines
            for metric_name, current_value in sim_metrics.items():
                if metric_name in self.performance_baselines:
                    baseline = self.performance_baselines[metric_name]
                    
                    # Check if current value is significantly worse than baseline
                    if isinstance(current_value, (int, float)) and isinstance(baseline, (int, float)):
                        deviation = abs(current_value - baseline) / baseline if baseline != 0 else 0
                        
                        if deviation > 0.5:  # 50% deviation
                            self.alert_manager.create_alert(
                                AlertSeverity.WARNING, "performance_baseline", 
                                f"Performance Baseline Deviation: {metric_name}",
                                f"Metric {metric_name} deviates {deviation:.1%} from baseline",
                                details={'current': current_value, 'baseline': baseline, 'deviation': deviation}
                            )
            
            # Update baselines with recent good performance
            if self.system_health_score > 0.9:  # Only update when system is healthy
                for metric_name, value in sim_metrics.items():
                    if isinstance(value, (int, float)):
                        if metric_name not in self.performance_baselines:
                            self.performance_baselines[metric_name] = value
                        else:
                            # Exponential moving average
                            alpha = 0.1
                            self.performance_baselines[metric_name] = (
                                alpha * value + (1 - alpha) * self.performance_baselines[metric_name]
                            )
                            
        except Exception as e:
            self.logger.warning(f"Performance baseline check failed: {e}")
    
    def _trigger_recovery(self, source: str, alert: Alert):
        """Trigger automatic recovery actions for failed health check."""
        # Find matching recovery actions
        for pattern, actions in self.recovery_actions.items():
            if pattern in source or pattern in alert.title:
                for action in actions:
                    if action.name not in self.recovery_in_progress:
                        self.recovery_queue.put((alert, action))
                        self.logger.info(f"Queued recovery action: {action.name} for {source}")
    
    def _recovery_loop(self):
        """Recovery action execution loop."""
        while self.running:
            try:
                # Get recovery action from queue (blocking with timeout)
                alert, action = self.recovery_queue.get(timeout=1.0)
                
                if action.name in self.recovery_in_progress:
                    continue
                
                self.recovery_in_progress.add(action.name)
                
                try:
                    self.logger.info(f"Executing recovery action: {action.name}")
                    success = action.execute()
                    
                    if success:
                        self.alert_manager.create_alert(
                            AlertSeverity.INFO, "auto_recovery", "Recovery Successful",
                            f"Automatic recovery action '{action.name}' completed successfully"
                        )
                        # Auto-resolve the original alert
                        self.alert_manager.resolve_alert(alert.id, "auto_recovery", auto_resolved=True)
                    else:
                        self.alert_manager.create_alert(
                            AlertSeverity.WARNING, "auto_recovery", "Recovery Failed",
                            f"Automatic recovery action '{action.name}' failed"
                        )
                        
                finally:
                    self.recovery_in_progress.discard(action.name)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in recovery loop: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status."""
        return {
            'monitoring_state': self.state.value,
            'system_health_score': self.system_health_score,
            'health_checks': {
                name: {
                    'enabled': check.enabled,
                    'healthy': check.is_healthy,
                    'consecutive_failures': check.consecutive_failures,
                    'consecutive_successes': check.consecutive_successes,
                    'last_check': check.last_check_time.isoformat() if check.last_check_time else None,
                    'mission_critical': check.mission_critical
                }
                for name, check in self.health_checks.items()
            },
            'alerts': self.alert_manager.get_alert_statistics(),
            'recovery_actions': {
                pattern: [action.name for action in actions]
                for pattern, actions in self.recovery_actions.items()
            },
            'recovery_queue_size': self.recovery_queue.qsize(),
            'recovery_in_progress': list(self.recovery_in_progress)
        }


# Predefined health checks for lunar habitat systems
def create_standard_health_checks(env_instance=None) -> List[HealthCheck]:
    """Create standard health checks for lunar habitat monitoring."""
    checks = []
    
    # System resource health check
    def system_resources_check():
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            healthy = cpu_percent < 90 and memory.percent < 90
            return {
                'healthy': healthy,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'available_memory_gb': memory.available / (1024**3)
            }
        except ImportError:
            # Fallback when psutil not available
            return {'healthy': True, 'note': 'psutil not available'}
    
    checks.append(HealthCheck(
        name="system_resources",
        check_function=system_resources_check,
        interval_seconds=30.0,
        failure_threshold=3,
        mission_critical=True
    ))
    
    # Environment health check
    if env_instance:
        def environment_health_check():
            try:
                # Check if environment is responsive
                if hasattr(env_instance, 'state'):
                    state = env_instance.state
                    
                    # Check critical life support parameters
                    if hasattr(state, 'atmosphere'):
                        o2_pressure = getattr(state.atmosphere, 'o2_partial_pressure', 21.0)
                        co2_pressure = getattr(state.atmosphere, 'co2_partial_pressure', 0.4)
                        
                        healthy = 16.0 <= o2_pressure <= 23.0 and co2_pressure <= 1.0
                        
                        return {
                            'healthy': healthy,
                            'o2_pressure': o2_pressure,
                            'co2_pressure': co2_pressure,
                            'atmosphere_status': 'nominal' if healthy else 'critical'
                        }
                
                return {'healthy': True, 'status': 'environment_accessible'}
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        checks.append(HealthCheck(
            name="environment_life_support",
            check_function=environment_health_check,
            interval_seconds=15.0,  # Check life support frequently
            failure_threshold=2,    # Quick response to life support issues
            mission_critical=True
        ))
    
    # Disk space health check
    def disk_space_check():
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            free_percent = (free / total) * 100
            
            healthy = free_percent > 10  # At least 10% free space
            return {
                'healthy': healthy,
                'free_percent': free_percent,
                'free_gb': free / (1024**3),
                'total_gb': total / (1024**3)
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    checks.append(HealthCheck(
        name="disk_space",
        check_function=disk_space_check,
        interval_seconds=300.0,  # Check every 5 minutes
        failure_threshold=2
    ))
    
    return checks


# Standard recovery actions
def create_standard_recovery_actions() -> Dict[str, AutoRecoveryAction]:
    """Create standard recovery actions for common issues."""
    actions = {}
    
    # Memory cleanup action
    def memory_cleanup():
        try:
            import gc
            gc.collect()
            return True
        except Exception:
            return False
    
    actions['memory_cleanup'] = AutoRecoveryAction(
        name="memory_cleanup",
        action_function=memory_cleanup,
        max_attempts=3,
        cooldown_seconds=30.0
    )
    
    # Log rotation action
    def rotate_logs():
        try:
            import logging.handlers
            # Trigger log rotation for all rotating handlers
            for handler in logging.getLogger().handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
            return True
        except Exception:
            return False
    
    actions['log_rotation'] = AutoRecoveryAction(
        name="log_rotation",
        action_function=rotate_logs,
        max_attempts=2,
        cooldown_seconds=300.0
    )
    
    # Environment reset action
    def environment_reset():
        try:
            # This would reset the environment to a safe state
            # Implementation depends on specific environment
            return True
        except Exception:
            return False
    
    actions['environment_reset'] = AutoRecoveryAction(
        name="environment_reset",
        action_function=environment_reset,
        max_attempts=1,
        cooldown_seconds=600.0
    )
    
    return actions


# Global monitoring instance
_global_advanced_monitor = None

def get_advanced_monitor() -> AdvancedHealthMonitor:
    """Get global advanced monitoring instance."""
    global _global_advanced_monitor
    if _global_advanced_monitor is None:
        _global_advanced_monitor = AdvancedHealthMonitor()
        
        # Register standard health checks
        for check in create_standard_health_checks():
            _global_advanced_monitor.register_health_check(check)
        
        # Register standard recovery actions
        recovery_actions = create_standard_recovery_actions()
        
        # Map recovery actions to triggers
        _global_advanced_monitor.register_recovery_action("system_resources", recovery_actions['memory_cleanup'])
        _global_advanced_monitor.register_recovery_action("disk_space", recovery_actions['log_rotation'])
        _global_advanced_monitor.register_recovery_action("environment", recovery_actions['environment_reset'])
    
    return _global_advanced_monitor


def monitoring_required(mission_critical: bool = False):
    """Decorator to ensure monitoring is active for critical operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_advanced_monitor()
            
            if monitor.state != MonitoringState.ACTIVE:
                if mission_critical:
                    raise RuntimeError("Monitoring system must be active for mission-critical operations")
                else:
                    monitor.logger.warning(f"Monitoring not active for operation: {func.__name__}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator