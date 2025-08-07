"""
Real-time Monitoring and Visualization System

This module provides comprehensive real-time monitoring capabilities
for lunar habitat RL systems, including:
- Live performance dashboards
- System health monitoring
- Alert systems for critical conditions
- Data logging and analysis
- Interactive visualization tools

Generation 2 Enhancement: Production-ready monitoring system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
import logging
from pathlib import Path
from threading import Thread, Event
from queue import Queue, Empty
import warnings

# Suppress matplotlib warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring system."""
    
    # Display settings
    update_interval: float = 1.0  # seconds
    history_length: int = 1000    # number of data points to keep
    figure_size: Tuple[int, int] = (16, 12)
    
    # Alert thresholds
    critical_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'o2_pressure': 19.0,      # kPa (critical low)
        'co2_pressure': 1.0,      # kPa (critical high)
        'temperature': 15.0,      # °C (critical low)
        'temperature_high': 35.0, # °C (critical high)
        'power_level': 20.0,      # % (critical low)
        'crew_health': 0.7        # 0-1 scale (critical low)
    })
    
    warning_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'o2_pressure': 20.0,
        'co2_pressure': 0.8,
        'temperature': 18.0,
        'temperature_high': 30.0,
        'power_level': 30.0,
        'crew_health': 0.8
    })
    
    # Data logging
    log_to_file: bool = True
    log_directory: str = "monitoring_logs"
    log_format: str = "csv"  # "csv", "json", "hdf5"
    
    # Advanced features
    enable_alerts: bool = True
    enable_predictions: bool = True
    enable_anomaly_detection: bool = True
    
    # Visualization settings
    color_scheme: str = "dark"  # "dark", "light", "colorblind"
    show_grid: bool = True
    smooth_plots: bool = True


class AlertSystem:
    """Alert system for critical conditions."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks = []
    
    def add_callback(self, callback: Callable[[str, str, Dict], None]):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def check_thresholds(self, data: Dict[str, float], timestamp: float):
        """Check data against alert thresholds."""
        if not self.config.enable_alerts:
            return
        
        current_alerts = set()
        
        # Check critical thresholds
        for metric, threshold in self.config.critical_thresholds.items():
            if metric in data:
                value = data[metric]
                
                # Handle both high and low thresholds
                if metric.endswith('_high'):
                    base_metric = metric.replace('_high', '')
                    if base_metric in data and data[base_metric] > threshold:
                        alert_id = f"critical_{base_metric}_high"
                        current_alerts.add(alert_id)
                        self._trigger_alert(alert_id, "CRITICAL", {
                            'metric': base_metric,
                            'value': data[base_metric],
                            'threshold': threshold,
                            'message': f"{base_metric} critically high: {data[base_metric]:.2f} > {threshold:.2f}"
                        }, timestamp)
                else:
                    if value < threshold:
                        alert_id = f"critical_{metric}_low"
                        current_alerts.add(alert_id)
                        self._trigger_alert(alert_id, "CRITICAL", {
                            'metric': metric,
                            'value': value,
                            'threshold': threshold,
                            'message': f"{metric} critically low: {value:.2f} < {threshold:.2f}"
                        }, timestamp)
        
        # Check warning thresholds
        for metric, threshold in self.config.warning_thresholds.items():
            if metric in data:
                value = data[metric]
                
                if metric.endswith('_high'):
                    base_metric = metric.replace('_high', '')
                    if base_metric in data and data[base_metric] > threshold:
                        alert_id = f"warning_{base_metric}_high"
                        current_alerts.add(alert_id)
                        self._trigger_alert(alert_id, "WARNING", {
                            'metric': base_metric,
                            'value': data[base_metric],
                            'threshold': threshold,
                            'message': f"{base_metric} high: {data[base_metric]:.2f} > {threshold:.2f}"
                        }, timestamp)
                else:
                    if value < threshold:
                        alert_id = f"warning_{metric}_low"
                        current_alerts.add(alert_id)
                        self._trigger_alert(alert_id, "WARNING", {
                            'metric': metric,
                            'value': value,
                            'threshold': threshold,
                            'message': f"{metric} low: {value:.2f} < {threshold:.2f}"
                        }, timestamp)
        
        # Clear resolved alerts
        resolved_alerts = set(self.active_alerts.keys()) - current_alerts
        for alert_id in resolved_alerts:
            self._resolve_alert(alert_id, timestamp)
    
    def _trigger_alert(self, alert_id: str, severity: str, details: Dict, timestamp: float):
        """Trigger an alert."""
        if alert_id not in self.active_alerts:
            alert_data = {
                'id': alert_id,
                'severity': severity,
                'details': details,
                'triggered_at': timestamp,
                'acknowledged': False
            }
            
            self.active_alerts[alert_id] = alert_data
            self.alert_history.append(alert_data.copy())
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_id, severity, details)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
            
            logger.warning(f"ALERT [{severity}] {alert_id}: {details['message']}")
    
    def _resolve_alert(self, alert_id: str, timestamp: float):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert_data = self.active_alerts[alert_id]
            alert_data['resolved_at'] = timestamp
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> Dict[str, Dict]:
        """Get currently active alerts."""
        return self.active_alerts.copy()
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['acknowledged'] = True


class AnomalyDetector:
    """Simple anomaly detection system."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.data_history = {}
        self.anomaly_scores = {}
    
    def update(self, data: Dict[str, float]) -> Dict[str, float]:
        """Update anomaly detection with new data."""
        anomalies = {}
        
        for metric, value in data.items():
            if metric not in self.data_history:
                self.data_history[metric] = []
                self.anomaly_scores[metric] = []
            
            self.data_history[metric].append(value)
            
            # Keep only recent data
            if len(self.data_history[metric]) > self.window_size:
                self.data_history[metric].pop(0)
                self.anomaly_scores[metric].pop(0)
            
            # Compute anomaly score using simple z-score
            if len(self.data_history[metric]) >= 10:  # Need enough data
                recent_data = np.array(self.data_history[metric])
                mean_val = np.mean(recent_data[:-1])  # Exclude current point
                std_val = np.std(recent_data[:-1])
                
                if std_val > 1e-6:  # Avoid division by zero
                    z_score = abs(value - mean_val) / std_val
                    anomalies[metric] = z_score
                    self.anomaly_scores[metric].append(z_score)
                else:
                    anomalies[metric] = 0.0
                    self.anomaly_scores[metric].append(0.0)
            else:
                anomalies[metric] = 0.0
                self.anomaly_scores[metric].append(0.0)
        
        return anomalies


class DataLogger:
    """Data logging system for monitoring data."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.log_queue = Queue()
        self.logging_thread = None
        self.stop_event = Event()
        
        if config.log_to_file:
            self.log_dir = Path(config.log_directory)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._start_logging_thread()
    
    def _start_logging_thread(self):
        """Start background logging thread."""
        self.logging_thread = Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()
    
    def _logging_worker(self):
        """Background worker for file logging."""
        log_buffer = []
        last_flush_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                data = self.log_queue.get(timeout=1.0)
                log_buffer.append(data)
                
                # Flush buffer periodically or when full
                current_time = time.time()
                if (len(log_buffer) >= 100 or 
                    current_time - last_flush_time > 30):
                    self._flush_buffer(log_buffer)
                    log_buffer.clear()
                    last_flush_time = current_time
                
            except Empty:
                # Flush any remaining data
                if log_buffer:
                    self._flush_buffer(log_buffer)
                    log_buffer.clear()
                    last_flush_time = time.time()
        
        # Final flush when stopping
        if log_buffer:
            self._flush_buffer(log_buffer)
    
    def _flush_buffer(self, buffer: List[Dict]):
        """Flush log buffer to file."""
        if not buffer:
            return
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if self.config.log_format == "csv":
                filename = self.log_dir / f"monitoring_data_{timestamp}.csv"
                df = pd.DataFrame(buffer)
                
                # Append to existing file or create new one
                if filename.exists():
                    df.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df.to_csv(filename, index=False)
                    
            elif self.config.log_format == "json":
                filename = self.log_dir / f"monitoring_data_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(buffer, f, indent=2)
            
            logger.debug(f"Logged {len(buffer)} data points to {filename}")
            
        except Exception as e:
            logger.error(f"Error logging data: {e}")
    
    def log_data(self, data: Dict[str, Any]):
        """Add data to logging queue."""
        if self.config.log_to_file and not self.stop_event.is_set():
            # Add timestamp
            log_entry = data.copy()
            log_entry['timestamp'] = time.time()
            log_entry['datetime'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                self.log_queue.put_nowait(log_entry)
            except:
                logger.warning("Log queue full, dropping data point")
    
    def stop(self):
        """Stop logging system."""
        if self.logging_thread:
            self.stop_event.set()
            self.logging_thread.join(timeout=5.0)


class RealTimeMonitor:
    """
    Real-time monitoring system for lunar habitat RL.
    
    Provides comprehensive monitoring capabilities including:
    - Live data visualization
    - Alert system
    - Anomaly detection
    - Data logging
    - Performance metrics
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config if config else MonitoringConfig()
        
        # Initialize components
        self.alert_system = AlertSystem(self.config)
        self.anomaly_detector = AnomalyDetector()
        self.data_logger = DataLogger(self.config)
        
        # Data storage
        self.data_history = {}
        self.anomaly_history = {}
        self.performance_history = {}
        
        # Visualization setup
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.animation = None
        
        # Threading
        self.update_thread = None
        self.stop_monitoring = Event()
        
        # Data source
        self.data_source = None
        
        # Setup alert callbacks
        self.alert_system.add_callback(self._handle_alert)
        
        logger.info("Real-time monitor initialized")
    
    def set_data_source(self, data_source: Callable[[], Dict[str, float]]):
        """Set data source function that returns monitoring data."""
        self.data_source = data_source
    
    def _handle_alert(self, alert_id: str, severity: str, details: Dict):
        """Handle alert notifications."""
        # This could be extended to send notifications, emails, etc.
        print(f"\n🚨 [{severity}] {alert_id}")
        print(f"   {details['message']}\n")
    
    def start_monitoring(self, blocking: bool = False):
        """Start real-time monitoring."""
        if self.data_source is None:
            raise ValueError("Data source must be set before starting monitoring")
        
        # Initialize visualization
        self._setup_visualization()
        
        if blocking:
            # Run monitoring in main thread
            self._monitoring_loop()
        else:
            # Start monitoring thread
            self.update_thread = Thread(target=self._monitoring_loop, daemon=True)
            self.update_thread.start()
            
            # Start matplotlib animation
            if self.fig:
                self.animation = animation.FuncAnimation(
                    self.fig, self._update_plots, interval=int(self.config.update_interval * 1000),
                    blit=False, cache_frame_data=False
                )
                
                plt.show()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_update = time.time()
        
        while not self.stop_monitoring.is_set():
            current_time = time.time()
            
            if current_time - last_update >= self.config.update_interval:
                try:
                    # Get new data
                    data = self.data_source()
                    
                    if data:
                        self._process_data(data, current_time)
                    
                    last_update = current_time
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(0.1)  # Small sleep to prevent busy waiting
    
    def _process_data(self, data: Dict[str, float], timestamp: float):
        """Process incoming monitoring data."""
        # Add timestamp
        data_with_time = data.copy()
        data_with_time['timestamp'] = timestamp
        
        # Update data history
        for key, value in data.items():
            if key not in self.data_history:
                self.data_history[key] = {'time': [], 'values': []}
            
            self.data_history[key]['time'].append(timestamp)
            self.data_history[key]['values'].append(value)
            
            # Keep only recent data
            max_length = self.config.history_length
            if len(self.data_history[key]['time']) > max_length:
                self.data_history[key]['time'] = self.data_history[key]['time'][-max_length:]
                self.data_history[key]['values'] = self.data_history[key]['values'][-max_length:]
        
        # Anomaly detection
        if self.config.enable_anomaly_detection:
            anomalies = self.anomaly_detector.update(data)
            
            for key, score in anomalies.items():
                if key not in self.anomaly_history:
                    self.anomaly_history[key] = {'time': [], 'scores': []}
                
                self.anomaly_history[key]['time'].append(timestamp)
                self.anomaly_history[key]['scores'].append(score)
                
                # Keep only recent data
                max_length = self.config.history_length
                if len(self.anomaly_history[key]['time']) > max_length:
                    self.anomaly_history[key]['time'] = self.anomaly_history[key]['time'][-max_length:]
                    self.anomaly_history[key]['scores'] = self.anomaly_history[key]['scores'][-max_length:]
        
        # Check alerts
        self.alert_system.check_thresholds(data, timestamp)
        
        # Log data
        self.data_logger.log_data(data_with_time)
        
        # Update performance metrics
        self._update_performance_metrics(data, timestamp)
    
    def _update_performance_metrics(self, data: Dict[str, float], timestamp: float):
        """Update system performance metrics."""
        # Calculate derived metrics
        metrics = {}
        
        # System efficiency (example calculation)
        if 'power_generation' in data and 'power_consumption' in data:
            metrics['power_efficiency'] = (data['power_generation'] - data['power_consumption']) / max(data['power_generation'], 1e-6)
        
        # Atmospheric stability
        if 'o2_pressure' in data and 'co2_pressure' in data:
            target_o2 = 21.0  # kPa
            target_co2 = 0.4  # kPa
            o2_error = abs(data['o2_pressure'] - target_o2) / target_o2
            co2_error = abs(data['co2_pressure'] - target_co2) / target_co2
            metrics['atmosphere_stability'] = 1.0 - (o2_error + co2_error) / 2
        
        # Overall system health
        health_indicators = []
        for key, value in data.items():
            if 'health' in key.lower() or 'score' in key.lower():
                health_indicators.append(value)
        
        if health_indicators:
            metrics['overall_health'] = np.mean(health_indicators)
        
        # Store performance metrics
        for key, value in metrics.items():
            if key not in self.performance_history:
                self.performance_history[key] = {'time': [], 'values': []}
            
            self.performance_history[key]['time'].append(timestamp)
            self.performance_history[key]['values'].append(value)
    
    def _setup_visualization(self):
        """Setup matplotlib visualization."""
        # Set style based on config
        if self.config.color_scheme == "dark":
            plt.style.use('dark_background')
            colors = ['#00ff41', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b']
        elif self.config.color_scheme == "colorblind":
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        else:
            plt.style.use('default')
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Create figure
        self.fig = plt.figure(figsize=self.config.figure_size)
        self.fig.suptitle('Lunar Habitat Real-Time Monitor', fontsize=16, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Define subplot configurations
        subplots = [
            ('atmosphere', gs[0, 0], 'Atmospheric Conditions', ['o2_pressure', 'co2_pressure', 'total_pressure']),
            ('thermal', gs[0, 1], 'Thermal Systems', ['temperature', 'heating_power', 'cooling_power']),
            ('power', gs[0, 2], 'Power Systems', ['power_generation', 'power_consumption', 'battery_level']),
            ('crew', gs[1, 0], 'Crew Health', ['crew_health', 'crew_stress', 'crew_productivity']),
            ('resources', gs[1, 1], 'Resource Levels', ['water_level', 'food_level', 'waste_level']),
            ('performance', gs[1, 2], 'System Performance', ['power_efficiency', 'atmosphere_stability', 'overall_health']),
            ('alerts', gs[2, :], 'Active Alerts', []),
            ('anomalies', gs[3, :], 'Anomaly Scores', [])
        ]
        
        # Create subplots
        for plot_id, grid_pos, title, metrics in subplots:
            if plot_id == 'alerts':
                # Special handling for alerts table
                ax = self.fig.add_subplot(grid_pos)
                ax.set_title(title, fontweight='bold')
                ax.axis('off')
                self.axes[plot_id] = ax
            elif plot_id == 'anomalies':
                # Anomaly scores plot
                ax = self.fig.add_subplot(grid_pos)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Anomaly Score')
                if self.config.show_grid:
                    ax.grid(True, alpha=0.3)
                self.axes[plot_id] = ax
                self.lines[plot_id] = {}
            else:
                # Regular time series plots
                ax = self.fig.add_subplot(grid_pos)
                ax.set_title(title, fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Value')
                if self.config.show_grid:
                    ax.grid(True, alpha=0.3)
                
                self.axes[plot_id] = ax
                self.lines[plot_id] = {}
                
                # Initialize lines for each metric
                for i, metric in enumerate(metrics):
                    color = colors[i % len(colors)]
                    line, = ax.plot([], [], color=color, label=metric, linewidth=2)
                    self.lines[plot_id][metric] = line
                
                if metrics:  # Only add legend if there are metrics
                    ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
    
    def _update_plots(self, frame):
        """Update matplotlib plots with new data."""
        if not self.data_history:
            return []
        
        updated_objects = []
        
        # Update time series plots
        for plot_id, ax in self.axes.items():
            if plot_id in ['alerts', 'anomalies']:
                continue  # Handle separately
            
            if plot_id in self.lines:
                for metric, line in self.lines[plot_id].items():
                    if metric in self.data_history:
                        history = self.data_history[metric]
                        if history['time'] and history['values']:
                            # Convert absolute time to relative time
                            if history['time']:
                                current_time = time.time()
                                relative_times = [current_time - t for t in reversed(history['time'][-100:])]
                                values = list(reversed(history['values'][-100:]))
                                
                                # Smooth data if enabled
                                if self.config.smooth_plots and len(values) > 5:
                                    window_size = min(5, len(values) // 3)
                                    values = pd.Series(values).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').tolist()
                                
                                line.set_data(relative_times, values)
                                updated_objects.append(line)
                
                # Adjust axis limits
                if any(metric in self.data_history for metric in self.lines[plot_id].keys()):
                    ax.relim()
                    ax.autoscale_view()
        
        # Update anomaly scores plot
        if 'anomalies' in self.axes and self.anomaly_history:
            ax = self.axes['anomalies']
            ax.clear()
            ax.set_title('Anomaly Scores', fontweight='bold')
            ax.set_xlabel('Time (s ago)')
            ax.set_ylabel('Z-Score')
            
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#6c5ce7']
            color_idx = 0
            
            current_time = time.time()
            
            for metric, history in list(self.anomaly_history.items())[:6]:  # Limit to 6 metrics
                if history['time'] and history['scores']:
                    relative_times = [current_time - t for t in reversed(history['time'][-100:])]
                    scores = list(reversed(history['scores'][-100:]))
                    
                    ax.plot(relative_times, scores, 
                           color=colors[color_idx % len(colors)], 
                           label=metric, linewidth=1.5, alpha=0.8)
                    color_idx += 1
            
            ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Warning (2σ)')
            ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Critical (3σ)')
            
            if self.config.show_grid:
                ax.grid(True, alpha=0.3)
            
            if self.anomaly_history:
                ax.legend(loc='upper right', fontsize=8)
        
        # Update alerts display
        if 'alerts' in self.axes:
            ax = self.axes['alerts']
            ax.clear()
            ax.set_title('Active Alerts', fontweight='bold')
            ax.axis('off')
            
            active_alerts = self.alert_system.get_active_alerts()
            
            if active_alerts:
                alert_text = ""
                for alert_id, alert_data in active_alerts.items():
                    severity_color = 'red' if alert_data['severity'] == 'CRITICAL' else 'orange'
                    ack_status = '✓' if alert_data['acknowledged'] else '!'
                    
                    elapsed = time.time() - alert_data['triggered_at']
                    elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                    
                    alert_text += f"{ack_status} [{alert_data['severity']}] {alert_data['details']['message']} ({elapsed_str})\n"
                
                ax.text(0.05, 0.95, alert_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No active alerts', transform=ax.transAxes, 
                       fontsize=12, ha='center', va='center', color='green',
                       fontweight='bold')
        
        return updated_objects
    
    def stop_monitoring(self):
        """Stop monitoring system."""
        logger.info("Stopping real-time monitor...")
        
        self.stop_monitoring.set()
        
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
        
        if self.animation:
            self.animation.event_source.stop()
        
        self.data_logger.stop()
        
        logger.info("Real-time monitor stopped")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        current_time = time.time()
        
        # Get latest values for each metric
        latest_data = {}
        for metric, history in self.data_history.items():
            if history['values']:
                latest_data[metric] = history['values'][-1]
        
        # Get active alerts
        active_alerts = self.alert_system.get_active_alerts()
        
        # Get anomaly scores
        latest_anomalies = {}
        for metric, history in self.anomaly_history.items():
            if history['scores']:
                latest_anomalies[metric] = history['scores'][-1]
        
        # Get performance metrics
        latest_performance = {}
        for metric, history in self.performance_history.items():
            if history['values']:
                latest_performance[metric] = history['values'][-1]
        
        status = {
            'timestamp': current_time,
            'data': latest_data,
            'alerts': {
                'active_count': len(active_alerts),
                'critical_count': sum(1 for a in active_alerts.values() if a['severity'] == 'CRITICAL'),
                'warning_count': sum(1 for a in active_alerts.values() if a['severity'] == 'WARNING'),
                'details': active_alerts
            },
            'anomalies': latest_anomalies,
            'performance': latest_performance,
            'system_health': self._compute_system_health(latest_data, active_alerts, latest_anomalies)
        }
        
        return status
    
    def _compute_system_health(self, data: Dict, alerts: Dict, anomalies: Dict) -> Dict[str, float]:
        """Compute overall system health metrics."""
        health = {}
        
        # Critical systems health
        critical_systems = ['o2_pressure', 'power_level', 'temperature']
        critical_health_scores = []
        
        for system in critical_systems:
            if system in data:
                # Simple health score based on how close to optimal the value is
                value = data[system]
                
                if system == 'o2_pressure':
                    optimal = 21.0
                    tolerance = 2.0
                elif system == 'power_level':
                    optimal = 80.0
                    tolerance = 20.0
                elif system == 'temperature':
                    optimal = 22.0
                    tolerance = 5.0
                else:
                    optimal = value
                    tolerance = 1.0
                
                deviation = abs(value - optimal) / tolerance
                health_score = max(0.0, 1.0 - deviation)
                critical_health_scores.append(health_score)
        
        if critical_health_scores:
            health['critical_systems'] = np.mean(critical_health_scores)
        
        # Alert penalty
        alert_penalty = len(alerts) * 0.1  # 10% penalty per alert
        critical_penalty = sum(1 for a in alerts.values() if a['severity'] == 'CRITICAL') * 0.2
        
        health['alert_score'] = max(0.0, 1.0 - alert_penalty - critical_penalty)
        
        # Anomaly score (inverse of average anomaly score)
        if anomalies:
            avg_anomaly = np.mean(list(anomalies.values()))
            health['stability_score'] = max(0.0, 1.0 - avg_anomaly / 10.0)
        else:
            health['stability_score'] = 1.0
        
        # Overall health
        health_scores = [score for score in health.values() if isinstance(score, float)]
        if health_scores:
            health['overall'] = np.mean(health_scores)
        else:
            health['overall'] = 0.5
        
        return health


# Mock data source for demonstration
class MockDataSource:
    """Mock data source for testing the monitoring system."""
    
    def __init__(self):
        self.time_start = time.time()
        self.base_values = {
            'o2_pressure': 21.0,
            'co2_pressure': 0.4,
            'total_pressure': 101.3,
            'temperature': 22.0,
            'power_generation': 10.0,
            'power_consumption': 8.0,
            'battery_level': 75.0,
            'crew_health': 0.95,
            'crew_stress': 0.2,
            'crew_productivity': 0.9,
            'water_level': 85.0,
            'food_level': 60.0,
            'waste_level': 20.0
        }
        
        # Simulate some trends and events
        self.trend_phases = {}
        self.event_time = None
    
    def get_data(self) -> Dict[str, float]:
        """Generate mock monitoring data."""
        current_time = time.time()
        elapsed = current_time - self.time_start
        
        data = {}
        
        for metric, base_value in self.base_values.items():
            # Add some realistic variation
            noise = np.random.normal(0, base_value * 0.02)  # 2% noise
            
            # Add sinusoidal variation
            period = 300  # 5 minute cycle
            phase = 2 * np.pi * elapsed / period
            sine_variation = base_value * 0.1 * np.sin(phase + hash(metric) % 100)
            
            # Add occasional events
            event_factor = 1.0
            if elapsed > 120 and elapsed < 180:  # Simulate event between 2-3 minutes
                if metric in ['o2_pressure', 'power_generation']:
                    event_factor = 0.8  # Temporary drop
                elif metric in ['co2_pressure', 'crew_stress']:
                    event_factor = 1.5  # Temporary increase
            
            value = (base_value + noise + sine_variation) * event_factor
            
            # Ensure reasonable bounds
            if 'pressure' in metric:
                value = max(0, value)
            elif 'level' in metric or 'health' in metric or 'productivity' in metric:
                value = max(0, min(100, value)) if 'level' in metric else max(0, min(1, value))
            elif metric == 'crew_stress':
                value = max(0, min(1, value))
            
            data[metric] = value
        
        return data


# Factory function for easy instantiation
def create_real_time_monitor(**kwargs) -> RealTimeMonitor:
    """Create real-time monitor with custom configuration."""
    config = MonitoringConfig(**kwargs)
    return RealTimeMonitor(config)


if __name__ == "__main__":
    # Demonstration of real-time monitoring system
    print("Real-Time Monitoring System - Generation 2")
    print("=" * 50)
    print("Features:")
    print("1. Live data visualization")
    print("2. Alert system with thresholds")
    print("3. Anomaly detection")
    print("4. Data logging")
    print("5. Performance metrics")
    print("6. System health monitoring")
    print("\nStarting demo with mock data source...")
    
    # Create monitor with custom config
    config = MonitoringConfig(
        update_interval=0.5,  # Faster updates for demo
        color_scheme="dark",
        enable_alerts=True,
        enable_anomaly_detection=True,
        log_to_file=True,
        log_directory="demo_monitoring_logs"
    )
    
    monitor = create_real_time_monitor(
        update_interval=config.update_interval,
        color_scheme=config.color_scheme,
        enable_alerts=config.enable_alerts
    )
    
    # Set up mock data source
    mock_source = MockDataSource()
    monitor.set_data_source(mock_source.get_data)
    
    print("Monitor configured. Close the plot window to stop monitoring.")
    
    try:
        # Start monitoring (blocking)
        monitor.start_monitoring(blocking=True)
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"Error: {e}")
        monitor.stop_monitoring()