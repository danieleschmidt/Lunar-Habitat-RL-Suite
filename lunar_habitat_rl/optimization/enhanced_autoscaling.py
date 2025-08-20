"""
Enhanced Auto-scaling and Load Balancing System - Generation 3
Advanced intelligent resource allocation for NASA space missions.
"""

import asyncio
import threading
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import pickle
import weakref
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger("enhanced_autoscaling")


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"


class ScalingStrategy(Enum):
    """Scaling strategies for different scenarios."""
    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predictions
    HYBRID = "hybrid"  # Combine reactive and predictive
    ADAPTIVE = "adaptive"  # Learn optimal scaling patterns


@dataclass
class ResourceMetrics:
    """Advanced resource metrics for scaling decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage_gb: float
    memory_percent: float
    gpu_usage: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    queue_depth: int = 0
    response_latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    active_connections: int = 0


@dataclass
class ScalingEvent:
    """Record of a scaling action."""
    timestamp: float
    action: str  # scale_up, scale_down, rebalance
    resource_type: ResourceType
    old_capacity: int
    new_capacity: int
    trigger_reason: str
    metrics_snapshot: ResourceMetrics
    success: bool
    execution_time_ms: float


@dataclass
class LoadBalancingPolicy:
    """Load balancing policy configuration."""
    algorithm: str = "weighted_least_connections"  # round_robin, least_connections, weighted_least_connections, resource_aware
    health_check_interval: float = 5.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    sticky_sessions: bool = False
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: float = 0.5  # 50% error rate


class PredictiveScaler:
    """Predictive scaling using time series analysis and machine learning."""
    
    def __init__(self, prediction_window_minutes: int = 30):
        self.prediction_window_minutes = prediction_window_minutes
        self.metrics_history = deque(maxlen=10000)  # Store more history for ML
        self.model_weights = {}  # Simple linear regression weights
        self.seasonal_patterns = {}
        self.trend_coefficients = {}
        
        # Feature engineering
        self.feature_extractors = {
            'time_of_day': lambda ts: datetime.fromtimestamp(ts).hour,
            'day_of_week': lambda ts: datetime.fromtimestamp(ts).weekday(),
            'minute_of_hour': lambda ts: datetime.fromtimestamp(ts).minute,
        }
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add metrics for prediction model training."""
        self.metrics_history.append(metrics)
        
        # Online learning - update model periodically
        if len(self.metrics_history) > 100 and len(self.metrics_history) % 50 == 0:
            self._update_prediction_model()
    
    def predict_load(self, minutes_ahead: int = 10) -> Dict[str, float]:
        """Predict resource load N minutes ahead."""
        if len(self.metrics_history) < 10:
            return {}
        
        current_time = time.time()
        prediction_time = current_time + (minutes_ahead * 60)
        
        # Extract features for prediction time
        features = self._extract_features(prediction_time)
        
        predictions = {}
        
        # Predict CPU usage
        cpu_prediction = self._predict_metric('cpu_usage', features, minutes_ahead)
        predictions['cpu_usage'] = max(0, min(100, cpu_prediction))
        
        # Predict memory usage
        memory_prediction = self._predict_metric('memory_percent', features, minutes_ahead)
        predictions['memory_percent'] = max(0, min(100, memory_prediction))
        
        # Predict throughput
        throughput_prediction = self._predict_metric('throughput', features, minutes_ahead)
        predictions['throughput'] = max(0, throughput_prediction)
        
        return predictions
    
    def _extract_features(self, timestamp: float) -> np.ndarray:
        """Extract features for prediction."""
        features = []
        
        # Time-based features
        dt = datetime.fromtimestamp(timestamp)
        features.extend([
            dt.hour / 24.0,  # Normalized hour
            dt.weekday() / 7.0,  # Normalized day of week
            dt.minute / 60.0,  # Normalized minute
            (dt.hour * 60 + dt.minute) / 1440.0  # Normalized time of day
        ])
        
        # Recent trend features
        if len(self.metrics_history) >= 5:
            recent_cpu = [m.cpu_usage for m in list(self.metrics_history)[-5:]]
            recent_memory = [m.memory_percent for m in list(self.metrics_history)[-5:]]
            
            features.extend([
                np.mean(recent_cpu),
                np.std(recent_cpu),
                np.mean(recent_memory),
                np.std(recent_memory)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def _predict_metric(self, metric_name: str, features: np.ndarray, minutes_ahead: int) -> float:
        """Predict a specific metric value."""
        if metric_name not in self.model_weights:
            return 0.0
        
        weights = self.model_weights[metric_name]
        
        if len(weights) != len(features):
            return 0.0
        
        # Simple linear prediction
        base_prediction = np.dot(features, weights)
        
        # Apply trend
        if metric_name in self.trend_coefficients:
            trend = self.trend_coefficients[metric_name] * minutes_ahead
            base_prediction += trend
        
        return base_prediction
    
    def _update_prediction_model(self):
        """Update the prediction model with recent data."""
        if len(self.metrics_history) < 20:
            return
        
        # Prepare training data
        metrics_list = list(self.metrics_history)
        
        # Train simple linear regression for each metric
        for metric_name in ['cpu_usage', 'memory_percent', 'throughput']:
            try:
                X = []  # Features
                y = []  # Target values
                
                for i in range(10, len(metrics_list)):
                    features = self._extract_features(metrics_list[i].timestamp)
                    target = getattr(metrics_list[i], metric_name, 0)
                    
                    X.append(features)
                    y.append(target)
                
                if len(X) >= 10:
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Simple least squares regression
                    weights = np.linalg.lstsq(X, y, rcond=None)[0]
                    self.model_weights[metric_name] = weights
                    
                    # Calculate trend
                    if len(y) >= 5:
                        time_indices = np.arange(len(y))
                        trend = np.polyfit(time_indices, y, 1)[0]
                        self.trend_coefficients[metric_name] = trend
                
            except Exception as e:
                logger.warning(f"Error updating prediction model for {metric_name}: {e}")


class IntelligentLoadBalancer:
    """Advanced load balancer with multiple algorithms and health monitoring."""
    
    def __init__(self, policy: LoadBalancingPolicy):
        self.policy = policy
        self.nodes = {}  # node_id -> node_info
        self.node_stats = defaultdict(lambda: defaultdict(list))
        self.health_status = {}  # node_id -> healthy/unhealthy
        self.circuit_breakers = {}  # node_id -> circuit_breaker_state
        
        # Connection tracking
        self.active_connections = defaultdict(int)
        self.connection_history = deque(maxlen=1000)
        
        # Performance tracking
        self.routing_stats = defaultdict(int)
        self.response_times = defaultdict(lambda: deque(maxlen=100))
        
        self.lock = threading.RLock()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def register_node(self, node_id: str, capacity: int, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Register a new node for load balancing."""
        with self.lock:
            self.nodes[node_id] = {
                'capacity': capacity,
                'current_load': 0,
                'metadata': metadata or {},
                'registered_at': time.time(),
                'last_health_check': time.time()
            }
            self.health_status[node_id] = True
            self.circuit_breakers[node_id] = {'state': 'closed', 'failures': 0, 'last_failure': 0}
            
        logger.info(f"Registered load balancing node: {node_id} (capacity: {capacity})")
    
    def unregister_node(self, node_id: str):
        """Unregister a node from load balancing."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                del self.health_status[node_id]
                del self.circuit_breakers[node_id]
                
        logger.info(f"Unregistered load balancing node: {node_id}")
    
    def route_request(self, request_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Route a request to the best available node."""
        with self.lock:
            healthy_nodes = [nid for nid, healthy in self.health_status.items() if healthy]
            
            if not healthy_nodes:
                return None
            
            # Apply load balancing algorithm
            if self.policy.algorithm == "round_robin":
                selected_node = self._round_robin_selection(healthy_nodes)
            elif self.policy.algorithm == "least_connections":
                selected_node = self._least_connections_selection(healthy_nodes)
            elif self.policy.algorithm == "weighted_least_connections":
                selected_node = self._weighted_least_connections_selection(healthy_nodes)
            elif self.policy.algorithm == "resource_aware":
                selected_node = self._resource_aware_selection(healthy_nodes, request_metadata)
            else:
                selected_node = healthy_nodes[0]  # Fallback
            
            if selected_node:
                # Update connection tracking
                self.active_connections[selected_node] += 1
                self.routing_stats[selected_node] += 1
                self.connection_history.append({
                    'timestamp': time.time(),
                    'node': selected_node,
                    'algorithm': self.policy.algorithm
                })
            
            return selected_node
    
    def _round_robin_selection(self, nodes: List[str]) -> str:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        
        return selected
    
    def _least_connections_selection(self, nodes: List[str]) -> str:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: self.active_connections[n])
    
    def _weighted_least_connections_selection(self, nodes: List[str]) -> str:
        """Weighted least connections based on node capacity."""
        def connection_ratio(node_id):
            capacity = self.nodes[node_id]['capacity']
            connections = self.active_connections[node_id]
            return connections / max(capacity, 1)
        
        return min(nodes, key=connection_ratio)
    
    def _resource_aware_selection(self, nodes: List[str], 
                                request_metadata: Optional[Dict[str, Any]]) -> str:
        """Resource-aware selection based on current node load."""
        node_scores = {}
        
        for node_id in nodes:
            node_info = self.nodes[node_id]
            
            # Base score (lower is better)
            load_ratio = self.active_connections[node_id] / max(node_info['capacity'], 1)
            
            # Recent response time factor
            recent_response_times = list(self.response_times[node_id])
            avg_response_time = np.mean(recent_response_times) if recent_response_times else 0
            
            # Combine factors
            score = load_ratio * 0.7 + (avg_response_time / 1000.0) * 0.3  # Normalize response time
            node_scores[node_id] = score
        
        return min(node_scores, key=node_scores.get)
    
    def report_response_time(self, node_id: str, response_time_ms: float):
        """Report response time for a node."""
        with self.lock:
            self.response_times[node_id].append(response_time_ms)
    
    def report_request_completion(self, node_id: str, success: bool):
        """Report request completion status."""
        with self.lock:
            if node_id in self.active_connections:
                self.active_connections[node_id] = max(0, self.active_connections[node_id] - 1)
            
            # Update circuit breaker
            circuit_breaker = self.circuit_breakers.get(node_id, {})
            
            if success:
                circuit_breaker['failures'] = max(0, circuit_breaker['failures'] - 1)
                if (circuit_breaker['state'] == 'open' and 
                    circuit_breaker['failures'] <= self.policy.recovery_threshold):
                    circuit_breaker['state'] = 'closed'
                    logger.info(f"Circuit breaker closed for node {node_id}")
            else:
                circuit_breaker['failures'] += 1
                circuit_breaker['last_failure'] = time.time()
                
                failure_rate = circuit_breaker['failures'] / max(self.routing_stats[node_id], 1)
                if (self.policy.enable_circuit_breaker and 
                    failure_rate > self.policy.circuit_breaker_threshold):
                    circuit_breaker['state'] = 'open'
                    self.health_status[node_id] = False
                    logger.warning(f"Circuit breaker opened for node {node_id}")
            
            self.circuit_breakers[node_id] = circuit_breaker
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        def health_check_loop():
            while True:
                try:
                    with self.lock:
                        for node_id in list(self.nodes.keys()):
                            # Simple health check based on recent activity and circuit breaker
                            circuit_breaker = self.circuit_breakers.get(node_id, {})
                            
                            if circuit_breaker.get('state') == 'open':
                                # Check if circuit breaker should be half-opened for testing
                                if (time.time() - circuit_breaker.get('last_failure', 0) > 30):
                                    circuit_breaker['state'] = 'half-open'
                                    self.health_status[node_id] = True
                                    logger.info(f"Circuit breaker half-opened for node {node_id}")
                    
                    time.sleep(self.policy.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(10)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across nodes."""
        with self.lock:
            return {
                node_id: {
                    'active_connections': self.active_connections[node_id],
                    'total_requests': self.routing_stats[node_id],
                    'capacity': self.nodes[node_id]['capacity'],
                    'load_ratio': self.active_connections[node_id] / max(self.nodes[node_id]['capacity'], 1),
                    'healthy': self.health_status[node_id],
                    'circuit_breaker': self.circuit_breakers.get(node_id, {}).get('state', 'closed')
                }
                for node_id in self.nodes.keys()
            }


class EnhancedAutoScaler:
    """Enhanced auto-scaler with predictive capabilities and intelligent load balancing."""
    
    def __init__(self, 
                 min_instances: int = 1,
                 max_instances: int = 20,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID,
                 prediction_window_minutes: int = 15,
                 scale_up_cooldown: float = 60.0,
                 scale_down_cooldown: float = 300.0):
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scaling_strategy = scaling_strategy
        self.prediction_window_minutes = prediction_window_minutes
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        
        # Current state
        self.current_instances = min_instances
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        
        # Components
        self.predictive_scaler = PredictiveScaler(prediction_window_minutes)
        self.load_balancer = IntelligentLoadBalancer(LoadBalancingPolicy())
        
        # Metrics and events
        self.metrics_history = deque(maxlen=10000)
        self.scaling_events = deque(maxlen=1000)
        self.performance_baseline = {}
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Adaptive learning
        self.scaling_efficiency_history = deque(maxlen=100)
        self.optimal_thresholds = {
            'cpu_scale_up': 80.0,
            'cpu_scale_down': 30.0,
            'memory_scale_up': 85.0,
            'memory_scale_down': 40.0,
            'latency_scale_up': 200.0,  # ms
            'queue_scale_up': 20
        }
        
        logger.info(f"Enhanced auto-scaler initialized: {min_instances}-{max_instances} instances")
    
    def set_scaling_callbacks(self, 
                            scale_up_callback: Callable[[int], bool],
                            scale_down_callback: Callable[[int], bool]):
        """Set callbacks for scaling operations."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def start_monitoring(self, interval: float = 10.0):
        """Start continuous monitoring and scaling."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Collect current metrics
                    metrics = self._collect_system_metrics()
                    if metrics:
                        self.add_metrics(metrics)
                        
                        # Evaluate scaling decision
                        self._evaluate_scaling_decision(metrics)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(30)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Enhanced auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Enhanced auto-scaler monitoring stopped")
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add resource metrics for analysis."""
        self.metrics_history.append(metrics)
        self.predictive_scaler.add_metrics(metrics)
        
        # Update performance baseline
        self._update_performance_baseline(metrics)
    
    def _collect_system_metrics(self) -> Optional[ResourceMetrics]:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Network and disk I/O
            net_io = psutil.net_io_counters()
            disk_io = psutil.disk_io_counters()
            
            # GPU metrics if available
            gpu_usage = None
            gpu_memory = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.utilization()
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            except ImportError:
                pass
            
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_percent,
                memory_usage_gb=memory.used / (1024**3),
                memory_percent=memory.percent,
                gpu_usage=gpu_usage,
                gpu_memory_gb=gpu_memory,
                network_io_mbps=0.0,  # Would need delta calculation
                disk_io_mbps=0.0,     # Would need delta calculation
                queue_depth=0,        # Would be provided by application
                response_latency_ms=0.0,  # Would be provided by application
                throughput=0.0        # Would be provided by application
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _evaluate_scaling_decision(self, current_metrics: ResourceMetrics):
        """Evaluate whether scaling is needed."""
        current_time = time.time()
        
        # Check cooldown periods
        can_scale_up = (current_time - self.last_scale_up) > self.scale_up_cooldown
        can_scale_down = (current_time - self.last_scale_down) > self.scale_down_cooldown
        
        if not (can_scale_up or can_scale_down):
            return
        
        scaling_decision = None
        
        if self.scaling_strategy in [ScalingStrategy.REACTIVE, ScalingStrategy.HYBRID]:
            scaling_decision = self._reactive_scaling_decision(current_metrics)
        
        if (self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID] and 
            not scaling_decision):
            predictive_decision = self._predictive_scaling_decision()
            if predictive_decision:
                scaling_decision = predictive_decision
        
        if self.scaling_strategy == ScalingStrategy.ADAPTIVE:
            scaling_decision = self._adaptive_scaling_decision(current_metrics)
        
        # Execute scaling decision
        if scaling_decision:
            self._execute_scaling_decision(scaling_decision, current_metrics)
    
    def _reactive_scaling_decision(self, metrics: ResourceMetrics) -> Optional[str]:
        """Make reactive scaling decision based on current metrics."""
        
        # Scale up conditions
        if (metrics.cpu_usage > self.optimal_thresholds['cpu_scale_up'] or
            metrics.memory_percent > self.optimal_thresholds['memory_scale_up'] or
            metrics.response_latency_ms > self.optimal_thresholds['latency_scale_up'] or
            metrics.queue_depth > self.optimal_thresholds['queue_scale_up']):
            
            if self.current_instances < self.max_instances:
                return "scale_up"
        
        # Scale down conditions
        elif (metrics.cpu_usage < self.optimal_thresholds['cpu_scale_down'] and
              metrics.memory_percent < self.optimal_thresholds['memory_scale_down'] and
              metrics.response_latency_ms < 50.0 and  # Good latency
              metrics.queue_depth < 2):
            
            if self.current_instances > self.min_instances:
                return "scale_down"
        
        return None
    
    def _predictive_scaling_decision(self) -> Optional[str]:
        """Make predictive scaling decision based on forecasted load."""
        predictions = self.predictive_scaler.predict_load(self.prediction_window_minutes)
        
        if not predictions:
            return None
        
        # Check if predicted load would trigger scaling
        predicted_cpu = predictions.get('cpu_usage', 0)
        predicted_memory = predictions.get('memory_percent', 0)
        
        if (predicted_cpu > self.optimal_thresholds['cpu_scale_up'] * 0.9 or  # Scale up earlier
            predicted_memory > self.optimal_thresholds['memory_scale_up'] * 0.9):
            
            if self.current_instances < self.max_instances:
                logger.info(f"Predictive scaling: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%")
                return "scale_up"
        
        elif (predicted_cpu < self.optimal_thresholds['cpu_scale_down'] * 1.2 and  # Scale down later
              predicted_memory < self.optimal_thresholds['memory_scale_down'] * 1.2):
            
            if self.current_instances > self.min_instances:
                return "scale_down"
        
        return None
    
    def _adaptive_scaling_decision(self, metrics: ResourceMetrics) -> Optional[str]:
        """Adaptive scaling that learns from past scaling efficiency."""
        # This would implement reinforcement learning or other ML approaches
        # For now, use adaptive thresholds based on past performance
        
        if len(self.scaling_efficiency_history) > 10:
            # Adjust thresholds based on scaling efficiency
            recent_efficiency = list(self.scaling_efficiency_history)[-10:]
            avg_efficiency = np.mean([e['efficiency'] for e in recent_efficiency])
            
            if avg_efficiency < 0.5:  # Low efficiency, make scaling more conservative
                self.optimal_thresholds['cpu_scale_up'] += 2.0
                self.optimal_thresholds['memory_scale_up'] += 2.0
            elif avg_efficiency > 0.8:  # High efficiency, make scaling more aggressive
                self.optimal_thresholds['cpu_scale_up'] -= 1.0
                self.optimal_thresholds['memory_scale_up'] -= 1.0
        
        # Use reactive decision with adaptive thresholds
        return self._reactive_scaling_decision(metrics)
    
    def _execute_scaling_decision(self, decision: str, metrics: ResourceMetrics):
        """Execute scaling decision."""
        old_instances = self.current_instances
        
        if decision == "scale_up":
            new_instances = min(self.max_instances, self.current_instances + 1)
        else:  # scale_down
            new_instances = max(self.min_instances, self.current_instances - 1)
        
        if new_instances == old_instances:
            return
        
        # Execute scaling action
        start_time = time.time()
        success = False
        
        try:
            if decision == "scale_up" and self.scale_up_callback:
                success = self.scale_up_callback(new_instances)
            elif decision == "scale_down" and self.scale_down_callback:
                success = self.scale_down_callback(new_instances)
            
            if success:
                self.current_instances = new_instances
                if decision == "scale_up":
                    self.last_scale_up = time.time()
                else:
                    self.last_scale_down = time.time()
                
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # Record scaling event
                event = ScalingEvent(
                    timestamp=time.time(),
                    action=decision,
                    resource_type=ResourceType.CPU,  # Primary resource type
                    old_capacity=old_instances,
                    new_capacity=new_instances,
                    trigger_reason=f"CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_percent:.1f}%",
                    metrics_snapshot=metrics,
                    success=True,
                    execution_time_ms=execution_time
                )
                
                self.scaling_events.append(event)
                
                logger.info(f"Scaling {decision}: {old_instances} -> {new_instances} instances "
                          f"(CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_percent:.1f}%)")
                
                # Track scaling efficiency (simplified)
                self._track_scaling_efficiency(event)
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            
            # Record failed event
            event = ScalingEvent(
                timestamp=time.time(),
                action=decision,
                resource_type=ResourceType.CPU,
                old_capacity=old_instances,
                new_capacity=new_instances,
                trigger_reason=f"Failed: {str(e)}",
                metrics_snapshot=metrics,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            self.scaling_events.append(event)
    
    def _update_performance_baseline(self, metrics: ResourceMetrics):
        """Update performance baseline for adaptive scaling."""
        if not self.performance_baseline:
            self.performance_baseline = {
                'cpu_usage': [],
                'memory_percent': [],
                'response_latency': []
            }
        
        self.performance_baseline['cpu_usage'].append(metrics.cpu_usage)
        self.performance_baseline['memory_percent'].append(metrics.memory_percent)
        self.performance_baseline['response_latency'].append(metrics.response_latency_ms)
        
        # Keep rolling window
        for key in self.performance_baseline:
            if len(self.performance_baseline[key]) > 1000:
                self.performance_baseline[key] = self.performance_baseline[key][-500:]
    
    def _track_scaling_efficiency(self, scaling_event: ScalingEvent):
        """Track efficiency of scaling operations."""
        # This is a simplified efficiency calculation
        # In practice, you'd measure actual performance improvement
        
        if len(self.metrics_history) < 10:
            return
        
        # Compare metrics before and after scaling (simplified)
        recent_metrics = list(self.metrics_history)[-5:]
        avg_cpu_before = np.mean([m.cpu_usage for m in recent_metrics])
        
        # Estimate efficiency based on how well the scaling addressed the issue
        if scaling_event.action == "scale_up":
            if avg_cpu_before > 80:
                efficiency = 0.8  # Good scaling decision
            else:
                efficiency = 0.3  # Maybe unnecessary scaling
        else:  # scale_down
            if avg_cpu_before < 40:
                efficiency = 0.8  # Good scaling decision
            else:
                efficiency = 0.3  # Maybe premature scaling
        
        self.scaling_efficiency_history.append({
            'timestamp': scaling_event.timestamp,
            'action': scaling_event.action,
            'efficiency': efficiency
        })
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling statistics."""
        current_time = time.time()
        
        recent_events = [e for e in self.scaling_events 
                        if current_time - e.timestamp < 3600]  # Last hour
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'scaling_strategy': self.scaling_strategy.value,
            'monitoring_active': self.monitoring_active,
            'recent_scaling_events': len(recent_events),
            'successful_scalings': sum(1 for e in recent_events if e.success),
            'last_scale_up': self.last_scale_up,
            'last_scale_down': self.last_scale_down,
            'current_thresholds': self.optimal_thresholds,
            'metrics_collected': len(self.metrics_history),
            'load_balancer_stats': self.load_balancer.get_load_distribution(),
            'avg_scaling_efficiency': (
                np.mean([e['efficiency'] for e in self.scaling_efficiency_history])
                if self.scaling_efficiency_history else 0.0
            )
        }
    
    def export_scaling_report(self) -> Dict[str, Any]:
        """Export comprehensive scaling report."""
        return {
            'report_timestamp': time.time(),
            'configuration': {
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'scaling_strategy': self.scaling_strategy.value,
                'prediction_window_minutes': self.prediction_window_minutes,
                'cooldown_scale_up': self.scale_up_cooldown,
                'cooldown_scale_down': self.scale_down_cooldown
            },
            'current_state': self.get_comprehensive_stats(),
            'scaling_events': [
                {
                    'timestamp': event.timestamp,
                    'action': event.action,
                    'resource_type': event.resource_type.value,
                    'old_capacity': event.old_capacity,
                    'new_capacity': event.new_capacity,
                    'success': event.success,
                    'execution_time_ms': event.execution_time_ms,
                    'trigger_reason': event.trigger_reason
                }
                for event in list(self.scaling_events)
            ],
            'performance_analysis': {
                'total_events': len(self.scaling_events),
                'success_rate': (
                    sum(1 for e in self.scaling_events if e.success) / 
                    max(len(self.scaling_events), 1)
                ),
                'avg_execution_time_ms': (
                    np.mean([e.execution_time_ms for e in self.scaling_events])
                    if self.scaling_events else 0
                ),
                'scaling_efficiency_trend': list(self.scaling_efficiency_history)
            }
        }


def demo_enhanced_autoscaling():
    """Demonstrate enhanced auto-scaling capabilities."""
    print("🚀 Enhanced Auto-scaling Demo")
    print("=" * 50)
    
    # Create auto-scaler
    scaler = EnhancedAutoScaler(
        min_instances=2,
        max_instances=10,
        scaling_strategy=ScalingStrategy.HYBRID,
        prediction_window_minutes=10
    )
    
    # Mock scaling callbacks
    def scale_up_callback(new_instances: int) -> bool:
        print(f"📈 Scaling up to {new_instances} instances")
        return True
    
    def scale_down_callback(new_instances: int) -> bool:
        print(f"📉 Scaling down to {new_instances} instances")
        return True
    
    scaler.set_scaling_callbacks(scale_up_callback, scale_down_callback)
    
    # Simulate metrics over time
    print("\n🔄 Simulating load patterns...")
    
    for i in range(20):
        # Simulate varying load
        if i < 5:
            cpu_usage = 20 + i * 5  # Gradual increase
        elif i < 10:
            cpu_usage = 85 + np.random.normal(0, 5)  # High load
        elif i < 15:
            cpu_usage = 60 + np.random.normal(0, 10)  # Variable load
        else:
            cpu_usage = 25 + np.random.normal(0, 5)  # Low load
        
        memory_percent = cpu_usage * 0.8 + np.random.normal(0, 5)
        
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_usage=max(0, min(100, cpu_usage)),
            memory_usage_gb=8.0,
            memory_percent=max(0, min(100, memory_percent)),
            queue_depth=max(0, int((cpu_usage - 50) / 10)) if cpu_usage > 50 else 0,
            response_latency_ms=max(10, cpu_usage * 2),
            throughput=max(10, 100 - cpu_usage / 2)
        )
        
        scaler.add_metrics(metrics)
        
        # Evaluate scaling (normally done by monitoring thread)
        scaler._evaluate_scaling_decision(metrics)
        
        time.sleep(0.1)  # Simulate time passage
    
    # Get final stats
    stats = scaler.get_comprehensive_stats()
    print(f"\n📊 Final Statistics:")
    print(f"  Current instances: {stats['current_instances']}")
    print(f"  Successful scalings: {stats['successful_scalings']}")
    print(f"  Average efficiency: {stats['avg_scaling_efficiency']:.2f}")
    print(f"  Metrics collected: {stats['metrics_collected']}")
    
    # Generate report
    report = scaler.export_scaling_report()
    print(f"\n📋 Scaling Events: {len(report['scaling_events'])}")
    print(f"  Success rate: {report['performance_analysis']['success_rate']:.1%}")
    print(f"  Avg execution time: {report['performance_analysis']['avg_execution_time_ms']:.1f} ms")
    
    print(f"\n✅ Enhanced auto-scaling demo completed!")


if __name__ == "__main__":
    demo_enhanced_autoscaling()