"""Auto-scaling capabilities for dynamic resource management."""

import time
import threading
import psutil
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta

from ..utils.logging import get_logger
from ..utils.monitoring import HealthMonitor


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    queue_size: int
    response_time: float
    error_rate: float
    timestamp: datetime


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    metric: str
    threshold: float
    action: str  # 'scale_up', 'scale_down'
    cooldown_seconds: int
    min_instances: int
    max_instances: int


class AutoScaler:
    """Intelligent auto-scaling system for habitat simulations."""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: int = 8,
                 target_cpu_usage: float = 70.0,
                 target_memory_usage: float = 80.0,
                 scale_cooldown: int = 60,
                 monitoring_interval: int = 10):
        """
        Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of worker processes
            max_workers: Maximum number of worker processes
            target_cpu_usage: Target CPU usage percentage
            target_memory_usage: Target memory usage percentage
            scale_cooldown: Cooldown period between scaling actions
            monitoring_interval: Seconds between monitoring checks
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_usage = target_cpu_usage
        self.target_memory_usage = target_memory_usage
        self.scale_cooldown = scale_cooldown
        self.monitoring_interval = monitoring_interval
        
        self.logger = get_logger("autoscaler")
        self.current_workers = min_workers
        self.last_scale_time = datetime.now()
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=100)
        self.scaling_history = deque(maxlen=50)
        
        # Auto-scaling rules
        self.scaling_rules = [
            ScalingRule(
                name="high_cpu_scale_up",
                metric="cpu_usage",
                threshold=target_cpu_usage + 15.0,  # 85% default
                action="scale_up",
                cooldown_seconds=scale_cooldown,
                min_instances=min_workers,
                max_instances=max_workers
            ),
            ScalingRule(
                name="low_cpu_scale_down", 
                metric="cpu_usage",
                threshold=target_cpu_usage - 20.0,  # 50% default
                action="scale_down",
                cooldown_seconds=scale_cooldown * 2,  # Longer cooldown for scale down
                min_instances=min_workers,
                max_instances=max_workers
            ),
            ScalingRule(
                name="high_memory_scale_up",
                metric="memory_usage", 
                threshold=target_memory_usage + 10.0,  # 90% default
                action="scale_up",
                cooldown_seconds=scale_cooldown,
                min_instances=min_workers,
                max_instances=max_workers
            ),
            ScalingRule(
                name="queue_backlog_scale_up",
                metric="queue_size",
                threshold=10.0,  # 10+ items in queue
                action="scale_up",
                cooldown_seconds=30,  # Shorter cooldown for queue backlog
                min_instances=min_workers,
                max_instances=max_workers
            )
        ]
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Worker management callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        self.logger.info(f"AutoScaler initialized: {min_workers}-{max_workers} workers, "
                        f"CPU target: {target_cpu_usage}%, Memory target: {target_memory_usage}%")
    
    def set_scale_callbacks(self, 
                           scale_up_callback: Callable[[int], bool],
                           scale_down_callback: Callable[[int], bool]):
        """
        Set callbacks for scaling actions.
        
        Args:
            scale_up_callback: Function to call when scaling up (new_worker_count) -> success
            scale_down_callback: Function to call when scaling down (new_worker_count) -> success
        """
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self.monitoring_active:
            self.logger.warning("Auto-scaling monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaling monitoring stopped")
    
    def update_metrics(self, 
                      cpu_usage: float,
                      memory_usage: float, 
                      queue_size: int = 0,
                      response_time: float = 0.0,
                      error_rate: float = 0.0,
                      gpu_usage: Optional[float] = None):
        """
        Update current metrics for scaling decisions.
        
        Args:
            cpu_usage: Current CPU usage percentage
            memory_usage: Current memory usage percentage
            queue_size: Current task queue size
            response_time: Average response time in seconds
            error_rate: Error rate (0.0 to 1.0)
            gpu_usage: Optional GPU usage percentage
        """
        metrics = ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            queue_size=queue_size,
            response_time=response_time,
            error_rate=error_rate,
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        
        # Trigger scaling decision
        self._evaluate_scaling_rules()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("Auto-scaling monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # GPU metrics (if available)
                gpu_percent = None
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_percent = np.mean([gpu.load * 100 for gpu in gpus])
                except ImportError:
                    pass
                
                # Update metrics
                self.update_metrics(
                    cpu_usage=cpu_percent,
                    memory_usage=memory_percent,
                    gpu_usage=gpu_percent
                )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling monitoring loop: {e}")
                time.sleep(10)  # Wait longer on error
        
        self.logger.info("Auto-scaling monitoring loop stopped")
    
    def _evaluate_scaling_rules(self):
        """Evaluate scaling rules against current metrics."""
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # Check cooldown period
        time_since_last_scale = datetime.now() - self.last_scale_time
        
        for rule in self.scaling_rules:
            if time_since_last_scale.total_seconds() < rule.cooldown_seconds:
                continue  # Still in cooldown
            
            # Get metric value
            metric_value = getattr(current_metrics, rule.metric, 0)
            
            # Check if rule applies
            should_trigger = False
            if rule.action == "scale_up":
                should_trigger = (metric_value > rule.threshold and 
                                 self.current_workers < rule.max_instances)
            elif rule.action == "scale_down":
                should_trigger = (metric_value < rule.threshold and 
                                 self.current_workers > rule.min_instances)
            
            if should_trigger:
                self._trigger_scaling_action(rule, metric_value)
                break  # Only apply one rule per evaluation
    
    def _trigger_scaling_action(self, rule: ScalingRule, metric_value: float):
        """Trigger a scaling action based on a rule."""
        old_worker_count = self.current_workers
        
        if rule.action == "scale_up":
            new_worker_count = min(self.current_workers + 1, rule.max_instances)
        else:  # scale_down
            new_worker_count = max(self.current_workers - 1, rule.min_instances)
        
        if new_worker_count == old_worker_count:
            return  # No change needed
        
        # Execute scaling action
        success = False
        if rule.action == "scale_up" and self.scale_up_callback:
            success = self.scale_up_callback(new_worker_count)
        elif rule.action == "scale_down" and self.scale_down_callback:
            success = self.scale_down_callback(new_worker_count)
        
        if success:
            self.current_workers = new_worker_count
            self.last_scale_time = datetime.now()
            
            # Record scaling action
            scaling_event = {
                'timestamp': datetime.now(),
                'rule': rule.name,
                'action': rule.action,
                'old_workers': old_worker_count,
                'new_workers': new_worker_count,
                'trigger_metric': rule.metric,
                'trigger_value': metric_value,
                'threshold': rule.threshold
            }
            self.scaling_history.append(scaling_event)
            
            self.logger.info(f"Auto-scaling: {rule.action} triggered by {rule.name} "
                           f"({rule.metric}={metric_value:.1f} vs {rule.threshold:.1f}). "
                           f"Workers: {old_worker_count} -> {new_worker_count}")
        else:
            self.logger.warning(f"Failed to execute scaling action: {rule.action}")
    
    def get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 metrics
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'last_scale_time': self.last_scale_time.isoformat(),
            'total_scaling_events': len(self.scaling_history),
            'average_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'average_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'recent_queue_size': recent_metrics[-1].queue_size if recent_metrics else 0,
            'monitoring_active': self.monitoring_active
        }
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling history for the specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.scaling_history 
            if event['timestamp'] >= cutoff
        ]
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule by name."""
        self.scaling_rules = [r for r in self.scaling_rules if r.name != rule_name]
        self.logger.info(f"Removed scaling rule: {rule_name}")
    
    def predict_scaling_need(self, look_ahead_minutes: int = 10) -> Optional[str]:
        """
        Predict if scaling will be needed in the near future.
        
        Args:
            look_ahead_minutes: Minutes to predict ahead
            
        Returns:
            Predicted action ('scale_up', 'scale_down', None)
        """
        if len(self.metrics_history) < 5:
            return None  # Not enough data
        
        # Analyze trends in recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        
        # CPU trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]  # Slope
        
        # Memory trend
        memory_values = [m.memory_usage for m in recent_metrics]
        memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
        
        # Predict future values
        future_cpu = cpu_values[-1] + cpu_trend * look_ahead_minutes
        future_memory = memory_values[-1] + memory_trend * look_ahead_minutes
        
        # Check if predicted values would trigger scaling
        if (future_cpu > self.target_cpu_usage + 15 or 
            future_memory > self.target_memory_usage + 10):
            return "scale_up"
        elif (future_cpu < self.target_cpu_usage - 20 and 
              future_memory < self.target_memory_usage - 20 and
              self.current_workers > self.min_workers):
            return "scale_down"
        
        return None
    
    def export_scaling_report(self) -> Dict[str, Any]:
        """Export comprehensive scaling report."""
        return {
            'report_generated': datetime.now().isoformat(),
            'configuration': {
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'target_cpu_usage': self.target_cpu_usage,
                'target_memory_usage': self.target_memory_usage,
                'scale_cooldown': self.scale_cooldown
            },
            'current_state': {
                'current_workers': self.current_workers,
                'monitoring_active': self.monitoring_active,
                'last_scale_time': self.last_scale_time.isoformat()
            },
            'statistics': self.get_scaling_statistics(),
            'scaling_history': [
                {
                    'timestamp': event['timestamp'].isoformat(),
                    'rule': event['rule'],
                    'action': event['action'],
                    'old_workers': event['old_workers'],
                    'new_workers': event['new_workers'],
                    'trigger_metric': event['trigger_metric'],
                    'trigger_value': event['trigger_value']
                }
                for event in self.scaling_history
            ],
            'scaling_rules': [
                {
                    'name': rule.name,
                    'metric': rule.metric,
                    'threshold': rule.threshold,
                    'action': rule.action,
                    'cooldown_seconds': rule.cooldown_seconds
                }
                for rule in self.scaling_rules
            ]
        }