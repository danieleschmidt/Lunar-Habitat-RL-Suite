"""Real-time monitoring and alerting system."""

import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
from dataclasses import dataclass, asdict

from .logging import get_logger, SafetyLogger
from .exceptions import SafetyError


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    timestamp: datetime
    level: str  # 'info', 'warning', 'error', 'critical'
    system: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class HealthMonitor:
    """Real-time health monitoring for habitat systems."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize health monitor.
        
        Args:
            update_interval: Seconds between monitoring updates
        """
        self.update_interval = update_interval
        self.logger = get_logger("monitoring")
        self.safety_logger = SafetyLogger("monitoring")
        
        # System health tracking
        self.health_metrics = {
            'atmosphere': {'status': 'unknown', 'last_update': None, 'metrics': {}},
            'power': {'status': 'unknown', 'last_update': None, 'metrics': {}},
            'thermal': {'status': 'unknown', 'last_update': None, 'metrics': {}},
            'water': {'status': 'unknown', 'last_update': None, 'metrics': {}},
            'crew': {'status': 'unknown', 'last_update': None, 'metrics': {}}
        }
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_handlers = defaultdict(list)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Thresholds for alerts
        self.alert_thresholds = {
            'atmosphere': {
                'o2_pressure': {'critical': 16.0, 'warning': 18.0},
                'co2_pressure': {'critical': 1.0, 'warning': 0.8},
                'total_pressure': {'critical': 50.0, 'warning': 60.0},
                'temperature': {'critical_low': 10.0, 'warning_low': 15.0, 
                               'critical_high': 35.0, 'warning_high': 30.0}
            },
            'power': {
                'battery_charge': {'critical': 10.0, 'warning': 20.0},
                'grid_stability': {'critical': 0.8, 'warning': 0.9}
            },
            'water': {
                'potable_water': {'critical': 50.0, 'warning': 100.0}
            },
            'crew': {
                'health': {'critical': 0.3, 'warning': 0.5},
                'stress': {'critical': 0.8, 'warning': 0.7}
            }
        }
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        self.logger.info("Health monitoring stopped")
    
    def update_metrics(self, system: str, metrics: Dict[str, Any]):
        """
        Update metrics for a system.
        
        Args:
            system: System name ('atmosphere', 'power', 'thermal', 'water', 'crew')
            metrics: Dictionary of metric values
        """
        try:
            self.metrics_queue.put((system, metrics, datetime.utcnow()), timeout=1.0)
        except queue.Full:
            self.logger.warning("Metrics queue full, dropping update")
    
    def add_alert_handler(self, system: str, handler: Callable[[Alert], None]):
        """
        Add a handler for system alerts.
        
        Args:
            system: System to monitor ('all' for all systems)
            handler: Function to call when alerts occur
        """
        self.alert_handlers[system].append(handler)
    
    def get_system_health(self, system: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current health status.
        
        Args:
            system: Specific system or None for all systems
            
        Returns:
            Dictionary of health information
        """
        if system:
            return self.health_metrics.get(system, {})
        else:
            return self.health_metrics.copy()
    
    def get_active_alerts(self, level: Optional[str] = None) -> List[Alert]:
        """
        Get active alerts.
        
        Args:
            level: Filter by alert level
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of historical alerts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff]
    
    def resolve_alert(self, alert_id: str, reason: str = ""):
        """
        Manually resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            reason: Reason for resolution
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            alert.details['resolution_reason'] = reason
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved: {reason}")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        self.logger.info("Monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Process queued metrics updates
                self._process_metrics_updates()
                
                # Check system health
                self._check_system_health()
                
                # Update performance tracking
                self._update_performance_tracking()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(5.0)  # Wait longer on error
        
        self.logger.info("Monitoring loop stopped")
    
    def _process_metrics_updates(self):
        """Process all pending metrics updates."""
        processed = 0
        
        while not self.metrics_queue.empty() and processed < 50:  # Limit batch size
            try:
                system, metrics, timestamp = self.metrics_queue.get_nowait()
                
                # Update health metrics
                self.health_metrics[system]['metrics'] = metrics
                self.health_metrics[system]['last_update'] = timestamp
                
                # Store in performance history
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.performance_history[f"{system}.{metric_name}"].append({
                            'timestamp': timestamp,
                            'value': value
                        })
                
                processed += 1
                
            except queue.Empty:
                break
    
    def _check_system_health(self):
        """Check all systems for health issues and generate alerts."""
        for system_name, system_data in self.health_metrics.items():
            if not system_data['metrics']:
                continue
            
            try:
                self._check_atmosphere_health(system_data['metrics']) if system_name == 'atmosphere' else None
                self._check_power_health(system_data['metrics']) if system_name == 'power' else None
                self._check_water_health(system_data['metrics']) if system_name == 'water' else None
                self._check_crew_health(system_data['metrics']) if system_name == 'crew' else None
                
                # Update system status
                self.health_metrics[system_name]['status'] = 'healthy'
                
            except Exception as e:
                self.logger.error(f"Error checking {system_name} health: {e}")
                self.health_metrics[system_name]['status'] = 'error'
    
    def _check_atmosphere_health(self, metrics: Dict[str, Any]):
        """Check atmosphere system health."""
        thresholds = self.alert_thresholds['atmosphere']
        
        # Check O2 levels
        if 'o2_partial_pressure' in metrics:
            o2 = metrics['o2_partial_pressure']
            if o2 < thresholds['o2_pressure']['critical']:
                self._create_alert('atmosphere', 'critical', 
                                 f"Critical oxygen depletion: {o2:.1f} kPa",
                                 {'o2_pressure': o2, 'threshold': thresholds['o2_pressure']['critical']})
            elif o2 < thresholds['o2_pressure']['warning']:
                self._create_alert('atmosphere', 'warning',
                                 f"Low oxygen warning: {o2:.1f} kPa", 
                                 {'o2_pressure': o2, 'threshold': thresholds['o2_pressure']['warning']})
        
        # Check CO2 levels
        if 'co2_partial_pressure' in metrics:
            co2 = metrics['co2_partial_pressure']
            if co2 > thresholds['co2_pressure']['critical']:
                self._create_alert('atmosphere', 'critical',
                                 f"Critical CO2 buildup: {co2:.1f} kPa",
                                 {'co2_pressure': co2, 'threshold': thresholds['co2_pressure']['critical']})
            elif co2 > thresholds['co2_pressure']['warning']:
                self._create_alert('atmosphere', 'warning',
                                 f"High CO2 warning: {co2:.1f} kPa",
                                 {'co2_pressure': co2, 'threshold': thresholds['co2_pressure']['warning']})
        
        # Check pressure
        if 'total_pressure' in metrics:
            pressure = metrics['total_pressure']
            if pressure < thresholds['total_pressure']['critical']:
                self._create_alert('atmosphere', 'critical',
                                 f"Critical pressure loss: {pressure:.1f} kPa",
                                 {'pressure': pressure, 'threshold': thresholds['total_pressure']['critical']})
            elif pressure < thresholds['total_pressure']['warning']:
                self._create_alert('atmosphere', 'warning',
                                 f"Low pressure warning: {pressure:.1f} kPa",
                                 {'pressure': pressure, 'threshold': thresholds['total_pressure']['warning']})
    
    def _check_power_health(self, metrics: Dict[str, Any]):
        """Check power system health."""
        thresholds = self.alert_thresholds['power']
        
        # Check battery levels
        if 'battery_charge' in metrics:
            battery = metrics['battery_charge']
            if battery < thresholds['battery_charge']['critical']:
                self._create_alert('power', 'critical',
                                 f"Critical battery level: {battery:.1f}%",
                                 {'battery_charge': battery, 'threshold': thresholds['battery_charge']['critical']})
            elif battery < thresholds['battery_charge']['warning']:
                self._create_alert('power', 'warning',
                                 f"Low battery warning: {battery:.1f}%",
                                 {'battery_charge': battery, 'threshold': thresholds['battery_charge']['warning']})
        
        # Check grid stability
        if 'grid_stability' in metrics:
            stability = metrics['grid_stability']
            if stability < thresholds['grid_stability']['critical']:
                self._create_alert('power', 'critical',
                                 f"Critical power instability: {stability:.2f}",
                                 {'grid_stability': stability, 'threshold': thresholds['grid_stability']['critical']})
    
    def _check_water_health(self, metrics: Dict[str, Any]):
        """Check water system health."""
        thresholds = self.alert_thresholds['water']
        
        if 'potable_water' in metrics:
            water = metrics['potable_water']
            if water < thresholds['potable_water']['critical']:
                self._create_alert('water', 'critical',
                                 f"Critical water shortage: {water:.0f} liters",
                                 {'potable_water': water, 'threshold': thresholds['potable_water']['critical']})
            elif water < thresholds['potable_water']['warning']:
                self._create_alert('water', 'warning',
                                 f"Low water warning: {water:.0f} liters",
                                 {'potable_water': water, 'threshold': thresholds['potable_water']['warning']})
    
    def _check_crew_health(self, metrics: Dict[str, Any]):
        """Check crew health status."""
        thresholds = self.alert_thresholds['crew']
        
        if 'crew_health' in metrics:
            health_levels = metrics['crew_health']
            if isinstance(health_levels, list):
                for i, health in enumerate(health_levels):
                    if health < thresholds['health']['critical']:
                        self._create_alert('crew', 'critical',
                                         f"Critical health - Crew member {i+1}: {health:.2f}",
                                         {'crew_member': i+1, 'health': health, 'threshold': thresholds['health']['critical']})
                    elif health < thresholds['health']['warning']:
                        self._create_alert('crew', 'warning',
                                         f"Low health - Crew member {i+1}: {health:.2f}",
                                         {'crew_member': i+1, 'health': health, 'threshold': thresholds['health']['warning']})
    
    def _create_alert(self, system: str, level: str, message: str, details: Dict[str, Any]):
        """Create a new alert."""
        # Create unique alert ID based on system, level, and key details
        alert_key = f"{system}_{level}_{details.get('crew_member', '')}"
        
        # Check if similar alert already exists
        if alert_key in self.active_alerts:
            # Update existing alert
            self.active_alerts[alert_key].details.update(details)
            return
        
        # Create new alert
        alert = Alert(
            id=alert_key,
            timestamp=datetime.utcnow(),
            level=level,
            system=system,
            message=message,
            details=details
        )
        
        self.active_alerts[alert_key] = alert
        
        # Log the alert
        self.logger.warning(f"ALERT [{level.upper()}] {system}: {message}", extra=details)
        
        # Log safety events for critical alerts
        if level == 'critical':
            self.safety_logger.log_safety_event(
                event_type=f"{system}_alert",
                severity=level,
                details={
                    'message': message,
                    'metrics': details
                }
            )
        
        # Notify alert handlers
        for handler in self.alert_handlers.get(system, []):
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        for handler in self.alert_handlers.get('all', []):
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in global alert handler: {e}")
    
    def _update_performance_tracking(self):
        """Update performance tracking and trends."""
        now = datetime.utcnow()
        
        for metric_name, history in self.performance_history.items():
            if not history:
                continue
            
            # Calculate recent trends
            recent_values = [point['value'] for point in list(history)[-10:]]  # Last 10 points
            if len(recent_values) >= 2:
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                
                # Store trend information
                if not hasattr(self, 'trends'):
                    self.trends = {}
                
                self.trends[metric_name] = {
                    'trend': trend,
                    'last_value': recent_values[-1],
                    'updated': now
                }
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts from history."""
        cutoff = datetime.utcnow() - timedelta(days=7)  # Keep 7 days of history
        
        # Remove old alerts from history
        while self.alert_history and self.alert_history[0].timestamp < cutoff:
            self.alert_history.popleft()
        
        # Auto-resolve stale active alerts (if metrics haven't updated in a while)
        stale_cutoff = datetime.utcnow() - timedelta(minutes=10)
        stale_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            system_last_update = self.health_metrics[alert.system]['last_update']
            if system_last_update and system_last_update < stale_cutoff:
                stale_alerts.append(alert_id)
        
        for alert_id in stale_alerts:
            self.resolve_alert(alert_id, "Auto-resolved due to stale metrics")
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        return {
            'monitoring_active': self.monitoring_active,
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_24h': len(self.get_alert_history(24)),
            'metrics_queue_size': self.metrics_queue.qsize(),
            'systems_monitored': list(self.health_metrics.keys()),
            'performance_metrics_tracked': len(self.performance_history),
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
    
    def export_monitoring_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Export comprehensive monitoring report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Comprehensive monitoring report
        """
        return {
            'report_generated': datetime.utcnow().isoformat(),
            'time_period_hours': hours,
            'system_health': self.get_system_health(),
            'active_alerts': [asdict(alert) for alert in self.get_active_alerts()],
            'alert_history': [asdict(alert) for alert in self.get_alert_history(hours)],
            'performance_trends': getattr(self, 'trends', {}),
            'monitoring_statistics': self.get_monitoring_statistics()
        }