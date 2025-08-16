"""Robust input validation and safety checks - Generation 2"""

import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from .robust_logging import get_logger, log_exception


@dataclass
class ValidationError(Exception):
    """Custom exception for validation errors."""
    parameter: str
    value: Any
    expected: str
    severity: str = "error"  # error, warning, critical
    
    def __str__(self):
        return f"Validation {self.severity} for {self.parameter}: got {self.value}, expected {self.expected}"


@dataclass
class SafetyLimits:
    """NASA-grade safety limits for lunar habitat systems."""
    
    # Atmosphere safety limits
    o2_min: float = 16.0  # kPa - minimum for crew survival
    o2_max: float = 30.0  # kPa - fire risk limit
    o2_nominal: float = 21.3  # kPa - Earth sea level
    
    co2_max: float = 0.5  # kPa - immediate danger to life and health
    co2_warning: float = 0.4  # kPa - NASA recommendation
    
    pressure_min: float = 70.0  # kPa - minimum viable pressure
    pressure_max: float = 130.0  # kPa - structural limit
    
    # Temperature safety limits (Celsius)
    temp_min: float = 15.0  # Minimum survivable
    temp_max: float = 35.0  # Maximum survivable
    temp_nominal: float = 22.5  # Comfort zone
    temp_tolerance: float = 3.0  # +/- comfort range
    
    # Power safety limits
    battery_critical: float = 10.0  # % - emergency power only
    battery_warning: float = 20.0  # % - start conservation
    battery_max: float = 100.0  # % - maximum charge
    
    power_load_max: float = 20.0  # kW - maximum system load
    
    # Water safety limits
    water_critical: float = 50.0  # liters - 1 day emergency supply
    water_warning: float = 200.0  # liters - 7 day supply
    water_max: float = 2000.0  # liters - storage limit
    
    # Crew health limits
    health_critical: float = 0.3  # Below this requires immediate attention
    health_warning: float = 0.7  # Below this requires monitoring
    stress_critical: float = 0.8  # Above this affects performance
    
    # Environmental limits
    dust_critical: float = 0.7  # Blocks critical systems
    degradation_critical: float = 0.3  # Major system failure risk


class RobustValidator:
    """NASA-grade validation system for all habitat inputs and states."""
    
    def __init__(self, safety_limits: Optional[SafetyLimits] = None):
        """Initialize validator with safety limits.
        
        Args:
            safety_limits: Custom safety limits, uses default NASA limits if None.
        """
        self.limits = safety_limits or SafetyLimits()
        self.logger = get_logger()
        self.validation_history = []
        
    @log_exception
    def validate_observation(self, observation: List[float], crew_size: int = 4) -> Dict[str, Any]:
        """Validate environment observation for safety and correctness.
        
        Args:
            observation: State observation array.
            crew_size: Number of crew members.
            
        Returns:
            Validation result with warnings and errors.
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'critical_errors': [],
            'safety_status': 'nominal'
        }
        
        if len(observation) < 30:
            error = ValidationError('observation', len(observation), 'at least 30 dimensions', 'critical')
            result['critical_errors'].append(str(error))
            result['valid'] = False
            result['safety_status'] = 'critical'
            return result
        
        # Validate atmosphere
        atmosphere_result = self._validate_atmosphere(observation[:7])
        result['warnings'].extend(atmosphere_result['warnings'])
        result['errors'].extend(atmosphere_result['errors'])
        result['critical_errors'].extend(atmosphere_result['critical_errors'])
        
        # Validate power
        power_result = self._validate_power(observation[7:13])
        result['warnings'].extend(power_result['warnings'])
        result['errors'].extend(power_result['errors'])
        result['critical_errors'].extend(power_result['critical_errors'])
        
        # Validate thermal
        thermal_result = self._validate_thermal(observation[13:21])
        result['warnings'].extend(thermal_result['warnings'])
        result['errors'].extend(thermal_result['errors'])
        
        # Validate water
        water_result = self._validate_water(observation[21:26])
        result['warnings'].extend(water_result['warnings'])
        result['errors'].extend(water_result['errors'])
        result['critical_errors'].extend(water_result['critical_errors'])
        
        # Validate crew (depends on crew_size)
        crew_dims = crew_size * 3
        if len(observation) >= 26 + crew_dims:
            crew_result = self._validate_crew(observation[26:26+crew_dims], crew_size)
            result['warnings'].extend(crew_result['warnings'])
            result['errors'].extend(crew_result['errors'])
            result['critical_errors'].extend(crew_result['critical_errors'])
        
        # Determine overall safety status
        if result['critical_errors']:
            result['valid'] = False
            result['safety_status'] = 'critical'
        elif result['errors']:
            result['safety_status'] = 'warning'
        elif result['warnings']:
            result['safety_status'] = 'caution'
        
        # Log validation result
        self._log_validation_result('observation', result)
        
        return result
    
    def _validate_atmosphere(self, atmosphere_data: List[float]) -> Dict[str, List[str]]:
        """Validate atmospheric conditions."""
        result = {'warnings': [], 'errors': [], 'critical_errors': []}
        
        if len(atmosphere_data) < 7:
            result['critical_errors'].append("Incomplete atmosphere data")
            return result
        
        o2_pressure, co2_pressure, n2_pressure, total_pressure, humidity, temperature, air_quality = atmosphere_data
        
        # O2 validation - critical for crew survival
        if o2_pressure < self.limits.o2_min:
            result['critical_errors'].append(f"O2 pressure {o2_pressure:.1f} kPa below survival minimum {self.limits.o2_min}")
        elif o2_pressure > self.limits.o2_max:
            result['critical_errors'].append(f"O2 pressure {o2_pressure:.1f} kPa above fire risk limit {self.limits.o2_max}")
        elif abs(o2_pressure - self.limits.o2_nominal) > 3.0:
            result['warnings'].append(f"O2 pressure {o2_pressure:.1f} kPa deviating from nominal {self.limits.o2_nominal}")
        
        # CO2 validation - critical for crew health
        if co2_pressure > self.limits.co2_max:
            result['critical_errors'].append(f"CO2 pressure {co2_pressure:.2f} kPa exceeds safety limit {self.limits.co2_max}")
        elif co2_pressure > self.limits.co2_warning:
            result['warnings'].append(f"CO2 pressure {co2_pressure:.2f} kPa above warning level {self.limits.co2_warning}")
        
        # Total pressure validation
        if total_pressure < self.limits.pressure_min:
            result['critical_errors'].append(f"Total pressure {total_pressure:.1f} kPa below minimum viable {self.limits.pressure_min}")
        elif total_pressure > self.limits.pressure_max:
            result['critical_errors'].append(f"Total pressure {total_pressure:.1f} kPa above structural limit {self.limits.pressure_max}")
        
        # Humidity validation
        if humidity < 20 or humidity > 70:
            result['warnings'].append(f"Humidity {humidity:.1f}% outside comfort range 20-70%")
        
        # Air quality validation
        if air_quality < 0.5:
            result['errors'].append(f"Air quality index {air_quality:.2f} indicates contamination")
        elif air_quality < 0.8:
            result['warnings'].append(f"Air quality index {air_quality:.2f} below optimal")
        
        return result
    
    def _validate_power(self, power_data: List[float]) -> Dict[str, List[str]]:
        """Validate power system conditions."""
        result = {'warnings': [], 'errors': [], 'critical_errors': []}
        
        if len(power_data) < 6:
            result['critical_errors'].append("Incomplete power data")
            return result
        
        solar_gen, battery_charge, fuel_cell_cap, total_load, emergency_reserve, grid_stability = power_data
        
        # Battery validation - critical for system operation
        if battery_charge < self.limits.battery_critical:
            result['critical_errors'].append(f"Battery charge {battery_charge:.1f}% below critical level {self.limits.battery_critical}%")
        elif battery_charge < self.limits.battery_warning:
            result['warnings'].append(f"Battery charge {battery_charge:.1f}% below warning level {self.limits.battery_warning}%")
        
        # Power load validation
        if total_load > self.limits.power_load_max:
            result['critical_errors'].append(f"Power load {total_load:.1f} kW exceeds maximum {self.limits.power_load_max} kW")
        elif total_load > solar_gen + 2.0:  # 2kW battery draw tolerance
            result['warnings'].append(f"Power consumption {total_load:.1f} kW exceeds generation {solar_gen:.1f} kW")
        
        # Grid stability validation
        if grid_stability < 0.9:
            result['errors'].append(f"Grid stability {grid_stability:.2f} indicates power quality issues")
        
        # Emergency reserve validation
        if emergency_reserve < 50:
            result['warnings'].append(f"Emergency power reserve {emergency_reserve:.1f}% below recommended 50%")
        
        return result
    
    def _validate_thermal(self, thermal_data: List[float]) -> Dict[str, List[str]]:
        """Validate thermal system conditions."""
        result = {'warnings': [], 'errors': [], 'critical_errors': []}
        
        if len(thermal_data) < 8:
            result['errors'].append("Incomplete thermal data")
            return result
        
        # Internal temperatures (4 zones)
        zone_temps = thermal_data[:4]
        external_temp = thermal_data[4]
        radiator_temps = thermal_data[5:7]
        heat_pump_efficiency = thermal_data[7]
        
        # Validate zone temperatures
        for i, temp in enumerate(zone_temps):
            if temp < self.limits.temp_min:
                result['critical_errors'].append(f"Zone {i+1} temperature {temp:.1f}°C below survival minimum {self.limits.temp_min}°C")
            elif temp > self.limits.temp_max:
                result['critical_errors'].append(f"Zone {i+1} temperature {temp:.1f}°C above survival maximum {self.limits.temp_max}°C")
            elif abs(temp - self.limits.temp_nominal) > self.limits.temp_tolerance + 2:
                result['warnings'].append(f"Zone {i+1} temperature {temp:.1f}°C outside comfort range")
        
        # Heat pump efficiency validation
        if heat_pump_efficiency < 1.5:
            result['warnings'].append(f"Heat pump efficiency {heat_pump_efficiency:.1f} COP below optimal")
        
        return result
    
    def _validate_water(self, water_data: List[float]) -> Dict[str, List[str]]:
        """Validate water system conditions."""
        result = {'warnings': [], 'errors': [], 'critical_errors': []}
        
        if len(water_data) < 5:
            result['critical_errors'].append("Incomplete water data")
            return result
        
        potable_water, grey_water, black_water, recycling_eff, filter_status = water_data
        
        # Potable water validation - critical for crew survival
        if potable_water < self.limits.water_critical:
            result['critical_errors'].append(f"Potable water {potable_water:.1f}L below critical level {self.limits.water_critical}L")
        elif potable_water < self.limits.water_warning:
            result['warnings'].append(f"Potable water {potable_water:.1f}L below warning level {self.limits.water_warning}L")
        
        # Recycling efficiency validation
        if recycling_eff < 0.8:
            result['errors'].append(f"Water recycling efficiency {recycling_eff:.2f} below minimum viable 0.8")
        elif recycling_eff < 0.9:
            result['warnings'].append(f"Water recycling efficiency {recycling_eff:.2f} below optimal 0.9")
        
        # Filter status validation
        if filter_status < 0.5:
            result['errors'].append(f"Water filter status {filter_status:.2f} indicates replacement needed")
        elif filter_status < 0.8:
            result['warnings'].append(f"Water filter status {filter_status:.2f} approaching replacement threshold")
        
        return result
    
    def _validate_crew(self, crew_data: List[float], crew_size: int) -> Dict[str, List[str]]:
        """Validate crew health and status."""
        result = {'warnings': [], 'errors': [], 'critical_errors': []}
        
        expected_dims = crew_size * 3
        if len(crew_data) < expected_dims:
            result['errors'].append(f"Incomplete crew data: got {len(crew_data)}, expected {expected_dims}")
            return result
        
        # Parse crew data: health, stress, productivity
        health_data = crew_data[:crew_size]
        stress_data = crew_data[crew_size:2*crew_size]
        productivity_data = crew_data[2*crew_size:3*crew_size]
        
        # Validate crew health
        for i, health in enumerate(health_data):
            if health < self.limits.health_critical:
                result['critical_errors'].append(f"Crew member {i+1} health {health:.2f} critically low")
            elif health < self.limits.health_warning:
                result['warnings'].append(f"Crew member {i+1} health {health:.2f} below warning threshold")
        
        # Validate crew stress
        for i, stress in enumerate(stress_data):
            if stress > self.limits.stress_critical:
                result['errors'].append(f"Crew member {i+1} stress {stress:.2f} critically high")
            elif stress > 0.6:
                result['warnings'].append(f"Crew member {i+1} stress {stress:.2f} elevated")
        
        # Validate productivity
        avg_productivity = sum(productivity_data) / len(productivity_data)
        if avg_productivity < 0.5:
            result['errors'].append(f"Average crew productivity {avg_productivity:.2f} critically low")
        elif avg_productivity < 0.7:
            result['warnings'].append(f"Average crew productivity {avg_productivity:.2f} below optimal")
        
        return result
    
    @log_exception
    def validate_action(self, action: List[float], expected_dims: int = 26) -> Dict[str, Any]:
        """Validate control action for safety and feasibility.
        
        Args:
            action: Control action array.
            expected_dims: Expected number of action dimensions.
            
        Returns:
            Validation result.
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'sanitized_action': None
        }
        
        # Dimension validation
        if len(action) != expected_dims:
            result['errors'].append(f"Action has {len(action)} dimensions, expected {expected_dims}")
            result['valid'] = False
            return result
        
        # Range validation and sanitization
        sanitized_action = []
        for i, value in enumerate(action):
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                result['errors'].append(f"Action[{i}] has invalid value: {value}")
                sanitized_action.append(0.5)  # Safe default
            elif value < 0 or value > 1:
                # Special case for solar panel angle (action 12) which can be [-90, 90]
                if i == 12 and -90 <= value <= 90:
                    sanitized_action.append(value)
                else:
                    result['warnings'].append(f"Action[{i}] value {value} clamped to [0,1]")
                    sanitized_action.append(max(0, min(1, value)))
            else:
                sanitized_action.append(value)
        
        result['sanitized_action'] = sanitized_action
        
        # Safety checks for specific actions
        self._validate_action_safety(sanitized_action, result)
        
        # Log validation
        self._log_validation_result('action', result)
        
        return result
    
    def _validate_action_safety(self, action: List[float], result: Dict[str, Any]):
        """Additional safety checks for specific action combinations."""
        
        # Check for dangerous action combinations
        if len(action) >= 6:
            o2_gen = action[0]
            co2_scrub = action[1]
            
            # Warn about low life support activity
            if o2_gen < 0.1 and co2_scrub < 0.1:
                result['warnings'].append("Both O2 generation and CO2 scrubbing very low - risk of atmosphere degradation")
        
        if len(action) >= 15:
            # Check thermal control
            heating_zones = action[15:19] if len(action) >= 19 else action[15:]
            if all(h < 0.1 for h in heating_zones):
                result['warnings'].append("All heating zones very low - risk of freezing")
            elif all(h > 0.9 for h in heating_zones):
                result['warnings'].append("All heating zones very high - risk of overheating")
        
        if len(action) >= 11:
            # Check power management
            battery_charge_rate = action[6]
            fuel_cell_activation = action[11] if len(action) > 11 else 0
            
            if battery_charge_rate > 0.9 and fuel_cell_activation > 0.5:
                result['warnings'].append("High battery charging with fuel cell active - inefficient power usage")
    
    def _log_validation_result(self, validation_type: str, result: Dict[str, Any]):
        """Log validation results for monitoring."""
        self.validation_history.append({
            'type': validation_type,
            'timestamp': time.time(),
            'valid': result['valid'],
            'warning_count': len(result.get('warnings', [])),
            'error_count': len(result.get('errors', [])),
            'critical_count': len(result.get('critical_errors', []))
        })
        
        # Keep only last 1000 validation records
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        # Log critical issues
        if result.get('critical_errors'):
            self.logger.critical(f"Validation failed for {validation_type}: {result['critical_errors']}")
        elif result.get('errors'):
            self.logger.error(f"Validation errors for {validation_type}: {result['errors']}")
        elif result.get('warnings'):
            self.logger.warning(f"Validation warnings for {validation_type}: {result['warnings']}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for monitoring."""
        if not self.validation_history:
            return {}
        
        total = len(self.validation_history)
        valid_count = sum(1 for v in self.validation_history if v['valid'])
        warning_count = sum(v['warning_count'] for v in self.validation_history)
        error_count = sum(v['error_count'] for v in self.validation_history)
        critical_count = sum(v['critical_count'] for v in self.validation_history)
        
        return {
            'total_validations': total,
            'success_rate': valid_count / total,
            'total_warnings': warning_count,
            'total_errors': error_count,
            'total_critical': critical_count,
            'avg_warnings_per_validation': warning_count / total,
            'avg_errors_per_validation': error_count / total
        }


# Global validator instance
_global_validator = None

def get_validator() -> RobustValidator:
    """Get global validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = RobustValidator()
    return _global_validator


def validate_and_sanitize_observation(observation: List[float], crew_size: int = 4) -> Tuple[bool, List[float], Dict[str, Any]]:
    """Validate and sanitize observation data.
    
    Returns:
        Tuple of (is_valid, sanitized_observation, validation_result)
    """
    validator = get_validator()
    result = validator.validate_observation(observation, crew_size)
    
    # Basic sanitization for invalid observations
    if not result['valid']:
        # Return safe default observation
        safe_obs = ([21.3, 0.3, 79.0, 101.3, 45.0, 22.5, 0.95] +  # atmosphere
                   [8.5, 75.0, 90.0, 6.2, 100.0, 0.98] +  # power
                   [22.5, 23.1, 22.8, 21.9, -45.0, 15.0, 16.2, 3.2] +  # thermal
                   [850.0, 120.0, 45.0, 0.93, 0.87] + # water
                   [0.95] * crew_size + [0.3] * crew_size + [0.9] * crew_size + # crew
                   [127.5, 0.3, 0.15, 0.05])  # environment
        return False, safe_obs[:len(observation)], result
    
    return True, observation, result


def validate_and_sanitize_action(action: List[float], expected_dims: int = 26) -> Tuple[bool, List[float], Dict[str, Any]]:
    """Validate and sanitize action data.
    
    Returns:
        Tuple of (is_valid, sanitized_action, validation_result)
    """
    validator = get_validator()
    result = validator.validate_action(action, expected_dims)
    
    sanitized_action = result.get('sanitized_action', action)
    return result['valid'], sanitized_action, result