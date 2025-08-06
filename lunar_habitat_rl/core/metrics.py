"""Performance metrics and tracking for mission evaluation."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import time
from collections import defaultdict, deque


@dataclass
class MissionMetrics:
    """Core metrics for evaluating mission success."""
    
    # Primary success metrics
    survival_time: float = 0.0  # sols
    crew_health_avg: float = 0.0  # 0-1
    resource_efficiency: float = 0.0  # 0-1
    emergency_response_time: float = 0.0  # seconds
    power_stability: float = 0.0  # 0-1 uptime
    
    # Secondary performance metrics
    atmosphere_quality: float = 0.0  # 0-1
    thermal_stability: float = 0.0  # 0-1
    water_conservation: float = 0.0  # 0-1
    equipment_wear: float = 0.0  # 0-1
    crew_productivity: float = 0.0  # 0-1
    
    # Safety and risk metrics
    safety_violations: int = 0
    critical_alarms: int = 0
    emergency_activations: int = 0
    system_failures: int = 0
    
    # Mission-specific scores
    overall_score: float = 0.0  # Weighted combination
    nasa_readiness_level: int = 0  # 1-6 TRL scale
    
    def compute_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted overall mission score."""
        if weights is None:
            weights = {
                'survival': 0.3,
                'crew_health': 0.25,
                'resource_efficiency': 0.2, 
                'power_stability': 0.15,
                'safety': 0.1
            }
        
        safety_score = max(0.0, 1.0 - (self.safety_violations * 0.1 + 
                                      self.critical_alarms * 0.05 +
                                      self.emergency_activations * 0.03))
        
        normalized_survival = min(1.0, self.survival_time / 30.0)  # 30-day target
        
        score = (
            weights['survival'] * normalized_survival +
            weights['crew_health'] * self.crew_health_avg +
            weights['resource_efficiency'] * self.resource_efficiency +
            weights['power_stability'] * self.power_stability +
            weights['safety'] * safety_score
        )
        
        self.overall_score = score
        return score
    
    def get_nasa_trl(self) -> int:
        """Determine NASA Technology Readiness Level based on performance."""
        score = self.overall_score
        
        if score >= 0.95:
            return 6  # System/subsystem model or prototype demonstration
        elif score >= 0.85:
            return 5  # Component and/or breadboard validation in relevant environment
        elif score >= 0.7:
            return 4  # Component and/or breadboard validation in laboratory environment
        elif score >= 0.5:
            return 3  # Analytical and experimental critical function proof-of-concept
        elif score >= 0.3:
            return 2  # Technology concept and/or application formulated
        else:
            return 1  # Basic principles observed and reported
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary for logging."""
        return {
            'survival_time': self.survival_time,
            'crew_health_avg': self.crew_health_avg,
            'resource_efficiency': self.resource_efficiency,
            'emergency_response_time': self.emergency_response_time,
            'power_stability': self.power_stability,
            'atmosphere_quality': self.atmosphere_quality,
            'thermal_stability': self.thermal_stability,
            'water_conservation': self.water_conservation,
            'equipment_wear': self.equipment_wear,
            'crew_productivity': self.crew_productivity,
            'safety_violations': self.safety_violations,
            'critical_alarms': self.critical_alarms,
            'emergency_activations': self.emergency_activations,
            'system_failures': self.system_failures,
            'overall_score': self.overall_score,
            'nasa_readiness_level': self.nasa_readiness_level
        }


class PerformanceTracker:
    """Real-time performance tracking and metrics computation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all tracking state."""
        self.start_time = time.time()
        self.step_count = 0
        self.episode_count = 0
        
        # Rolling windows for metrics
        self.rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.survival_times = deque(maxlen=self.window_size)
        
        # System state tracking
        self.power_outages = 0
        self.atmosphere_violations = 0
        self.thermal_violations = 0
        self.water_shortages = 0
        self.crew_health_events = 0
        
        # Performance counters
        self.emergency_events = []
        self.system_failures = []
        self.safety_violations = []
        
        # Real-time metrics
        self.current_metrics = MissionMetrics()
        
    def update_step(self, 
                   reward: float,
                   state: Dict[str, Any], 
                   action: Dict[str, Any],
                   info: Dict[str, Any]):
        """Update tracking with single step information."""
        self.step_count += 1
        self.rewards.append(reward)
        
        # Check for violations and events
        self._check_safety_violations(state, info)
        self._update_real_time_metrics(state, action, info)
        
    def update_episode(self, episode_length: int, final_state: Dict[str, Any], info: Dict[str, Any]):
        """Update tracking at episode completion."""
        self.episode_count += 1
        self.episode_lengths.append(episode_length)
        
        # Extract final mission metrics
        survival_time = info.get('survival_time', 0.0)
        self.survival_times.append(survival_time)
        
        # Compute episode metrics
        metrics = self._compute_episode_metrics(final_state, info)
        return metrics
        
    def _check_safety_violations(self, state: Dict[str, Any], info: Dict[str, Any]):
        """Check for safety violations in current state."""
        violations = []
        
        # Atmosphere safety checks
        if state.get('o2_partial_pressure', 21.3) < 16.0:
            violations.append('low_oxygen')
            self.atmosphere_violations += 1
            
        if state.get('co2_partial_pressure', 0.4) > 1.0:
            violations.append('high_co2')
            self.atmosphere_violations += 1
            
        if state.get('total_pressure', 101.3) < 50.0:
            violations.append('low_pressure')
            self.atmosphere_violations += 1
            
        # Power safety checks
        if state.get('battery_charge', 75.0) < 10.0 and state.get('solar_generation', 0.0) < 1.0:
            violations.append('power_critical')
            self.power_outages += 1
            
        # Thermal safety checks  
        avg_temp = np.mean(state.get('internal_temp_zones', [22.5]))
        if avg_temp < 15.0 or avg_temp > 30.0:
            violations.append('thermal_extreme')
            self.thermal_violations += 1
            
        # Water safety checks
        if state.get('potable_water', 850.0) < 100.0:
            violations.append('water_shortage')
            self.water_shortages += 1
            
        # Crew health checks
        crew_health = state.get('crew_health', [0.95, 0.98, 0.92, 0.96])
        if any(h < 0.5 for h in crew_health):
            violations.append('crew_health_critical')
            self.crew_health_events += 1
            
        if violations:
            self.safety_violations.extend(violations)
            
    def _update_real_time_metrics(self, state: Dict[str, Any], action: Dict[str, Any], info: Dict[str, Any]):
        """Update real-time performance metrics."""
        # Update current metrics with latest state
        self.current_metrics.crew_health_avg = np.mean(state.get('crew_health', [0.95]))
        self.current_metrics.power_stability = state.get('grid_stability', 0.98)
        self.current_metrics.atmosphere_quality = state.get('air_quality_index', 0.95)
        
        # Update counters
        self.current_metrics.safety_violations = len(self.safety_violations)
        self.current_metrics.critical_alarms = (self.power_outages + 
                                               self.atmosphere_violations + 
                                               self.thermal_violations)
        
    def _compute_episode_metrics(self, final_state: Dict[str, Any], info: Dict[str, Any]) -> MissionMetrics:
        """Compute comprehensive metrics for completed episode."""
        metrics = MissionMetrics()
        
        # Basic metrics
        metrics.survival_time = info.get('survival_time', 0.0)
        metrics.crew_health_avg = np.mean(final_state.get('crew_health', [0.95]))
        
        # Resource efficiency (based on usage vs. optimal)
        power_used = info.get('total_power_used', 1000.0)
        power_optimal = info.get('optimal_power_usage', 800.0) 
        metrics.resource_efficiency = min(1.0, power_optimal / power_used if power_used > 0 else 1.0)
        
        # System stability
        uptime = info.get('system_uptime', 1.0)
        metrics.power_stability = uptime
        
        # Secondary metrics
        metrics.atmosphere_quality = np.mean([
            final_state.get('air_quality_index', 0.95)
        ])
        
        thermal_zones = final_state.get('internal_temp_zones', [22.5, 23.1, 22.8, 21.9])
        thermal_variance = np.var(thermal_zones)
        metrics.thermal_stability = max(0.0, 1.0 - thermal_variance / 10.0)  # Normalize variance
        
        water_efficiency = final_state.get('recycling_efficiency', 0.93)
        metrics.water_conservation = water_efficiency
        
        # Safety metrics
        metrics.safety_violations = len(set(self.safety_violations))  # Unique violations
        metrics.critical_alarms = self.current_metrics.critical_alarms
        metrics.emergency_activations = len(self.emergency_events)
        metrics.system_failures = len(self.system_failures)
        
        # Compute overall score and TRL
        metrics.compute_overall_score()
        metrics.nasa_readiness_level = metrics.get_nasa_trl()
        
        return metrics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = {
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'avg_survival_time': np.mean(self.survival_times) if self.survival_times else 0.0,
            'total_safety_violations': len(self.safety_violations),
            'power_outage_rate': self.power_outages / max(1, self.step_count) * 1000,  # per 1000 steps
            'atmosphere_violation_rate': self.atmosphere_violations / max(1, self.step_count) * 1000,
            'current_metrics': self.current_metrics.to_dict()
        }
        
        return stats
    
    def export_metrics(self, filepath: str):
        """Export comprehensive metrics to file."""
        import json
        
        data = {
            'statistics': self.get_statistics(),
            'safety_violations': self.safety_violations,
            'emergency_events': self.emergency_events,
            'system_failures': self.system_failures,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)