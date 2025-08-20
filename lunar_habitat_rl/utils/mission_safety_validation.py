"""
NASA-grade mission safety validation and type checking system.
Implements comprehensive safety constraints for lunar habitat operations.
"""

import time
import json
import re
import functools
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import inspect
from collections import defaultdict

from .robust_logging import get_logger
from .exceptions import ValidationError
# Conditional import to avoid circular dependencies
try:
    from .audit_logging import get_audit_logger, AuditEventType, AuditLevel
except ImportError:
    get_audit_logger = None
    AuditEventType = None
    AuditLevel = None


class SafetyLevel(Enum):
    """Safety criticality levels for space mission operations."""
    INFORMATIONAL = "informational"
    ADVISORY = "advisory"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    MISSION_ABORT = "mission_abort"


class ParameterType(Enum):
    """Types of parameters in the lunar habitat system."""
    LIFE_SUPPORT = "life_support"
    POWER_SYSTEM = "power_system"
    THERMAL_CONTROL = "thermal_control"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    STRUCTURAL = "structural"
    CONSUMABLES = "consumables"
    WASTE_MANAGEMENT = "waste_management"
    CREW_HEALTH = "crew_health"
    EMERGENCY_SYSTEM = "emergency_system"


@dataclass
class SafetyConstraint:
    """Defines a safety constraint for mission parameters."""
    name: str
    parameter_type: ParameterType
    safety_level: SafetyLevel
    
    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    forbidden_values: Optional[List[Any]] = None
    
    # Pattern constraints
    pattern: Optional[str] = None
    format_validator: Optional[Callable[[Any], bool]] = None
    
    # Cross-parameter constraints
    dependencies: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    
    # Temporal constraints
    rate_limit: Optional[float] = None  # Max change per second
    stability_required: bool = False    # Must be stable for some time
    
    # Actions
    violation_action: Optional[Callable[[], None]] = None
    emergency_action: Optional[Callable[[], None]] = None
    
    # Metadata
    description: str = ""
    justification: str = ""
    reference_document: str = ""
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """Validate value against this constraint.
        
        Args:
            value: Value to validate
            context: Additional context for validation
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        context = context or {}
        
        # Type validation
        if not self._validate_type(value):
            violations.append(f"Invalid type for {self.name}: expected numeric, got {type(value)}")
        
        # Range validation
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                violations.append(f"{self.name} below minimum: {value} < {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                violations.append(f"{self.name} above maximum: {value} > {self.max_value}")
        
        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            violations.append(f"{self.name} not in allowed values: {value} not in {self.allowed_values}")
        
        # Forbidden values validation
        if self.forbidden_values is not None and value in self.forbidden_values:
            violations.append(f"{self.name} is forbidden value: {value}")
        
        # Pattern validation
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                violations.append(f"{self.name} does not match required pattern: {self.pattern}")
        
        # Custom format validation
        if self.format_validator:
            try:
                if not self.format_validator(value):
                    violations.append(f"{self.name} failed custom format validation")
            except Exception as e:
                violations.append(f"{self.name} format validation error: {e}")
        
        # Cross-parameter validation
        if context:
            violations.extend(self._validate_dependencies(value, context))
            violations.extend(self._validate_incompatibilities(value, context))
        
        return len(violations) == 0, violations
    
    def _validate_type(self, value: Any) -> bool:
        """Validate value type."""
        if self.min_value is not None or self.max_value is not None:
            return isinstance(value, (int, float))
        return True
    
    def _validate_dependencies(self, value: Any, context: Dict[str, Any]) -> List[str]:
        """Validate parameter dependencies."""
        violations = []
        for dep in self.dependencies:
            if dep not in context:
                violations.append(f"{self.name} requires {dep} to be set")
        return violations
    
    def _validate_incompatibilities(self, value: Any, context: Dict[str, Any]) -> List[str]:
        """Validate parameter incompatibilities."""
        violations = []
        for incomp in self.incompatible_with:
            if incomp in context:
                violations.append(f"{self.name} is incompatible with {incomp}")
        return violations


class MissionSafetyValidator:
    """NASA-grade safety validation system for lunar habitat operations."""
    
    def __init__(self):
        """Initialize mission safety validator."""
        self.logger = get_logger()
        self.audit_logger = get_audit_logger() if get_audit_logger else None
        
        # Constraint registry
        self.constraints: Dict[str, SafetyConstraint] = {}
        self.parameter_history: Dict[str, List[Tuple[datetime, Any]]] = defaultdict(list)
        
        # Validation statistics
        self.validation_count = 0
        self.violation_count = 0
        self.last_violation_time = None
        
        # Emergency state tracking
        self.emergency_state = False
        self.mission_abort_triggered = False
        
        # Initialize standard lunar habitat constraints
        self._initialize_standard_constraints()
    
    def _initialize_standard_constraints(self):
        """Initialize standard safety constraints for lunar habitat."""
        
        # Life Support - Oxygen Pressure
        self.register_constraint(SafetyConstraint(
            name="o2_pressure",
            parameter_type=ParameterType.LIFE_SUPPORT,
            safety_level=SafetyLevel.MISSION_ABORT,
            min_value=14.0,    # kPa - absolute minimum for crew survival
            max_value=25.0,    # kPa - maximum safe level
            description="Oxygen partial pressure in habitat atmosphere",
            justification="Critical for crew survival - below 14 kPa causes hypoxia",
            reference_document="NASA-STD-3001 Volume 2",
            emergency_action=self._emergency_o2_action
        ))
        
        # Life Support - CO2 Pressure
        self.register_constraint(SafetyConstraint(
            name="co2_pressure",
            parameter_type=ParameterType.LIFE_SUPPORT,
            safety_level=SafetyLevel.CRITICAL,
            min_value=0.0,     # kPa
            max_value=1.0,     # kPa - NASA limit for long-term exposure
            description="Carbon dioxide partial pressure in habitat atmosphere",
            justification="Elevated CO2 causes cognitive impairment and health issues",
            reference_document="NASA-STD-3001 Volume 2",
            emergency_action=self._emergency_co2_action
        ))
        
        # Life Support - Total Pressure
        self.register_constraint(SafetyConstraint(
            name="total_pressure",
            parameter_type=ParameterType.LIFE_SUPPORT,
            safety_level=SafetyLevel.CRITICAL,
            min_value=70.0,    # kPa - minimum for pressure suit operation
            max_value=110.0,   # kPa - maximum structural limit
            description="Total atmospheric pressure in habitat",
            justification="Required for crew comfort and structural integrity",
            reference_document="NASA-STD-3001 Volume 2"
        ))
        
        # Thermal Control - Temperature
        self.register_constraint(SafetyConstraint(
            name="temperature",
            parameter_type=ParameterType.THERMAL_CONTROL,
            safety_level=SafetyLevel.WARNING,
            min_value=15.0,    # °C - minimum comfortable temperature
            max_value=30.0,    # °C - maximum comfortable temperature
            description="Habitat internal temperature",
            justification="Temperature range for crew comfort and equipment operation",
            reference_document="NASA-STD-3001 Volume 2"
        ))
        
        # Thermal Control - Critical Temperature Limits
        self.register_constraint(SafetyConstraint(
            name="critical_temperature",
            parameter_type=ParameterType.THERMAL_CONTROL,
            safety_level=SafetyLevel.MISSION_ABORT,
            min_value=-10.0,   # °C - absolute minimum
            max_value=50.0,    # °C - absolute maximum
            description="Critical temperature limits for habitat",
            justification="Extreme temperatures threaten crew survival and mission equipment",
            reference_document="NASA-STD-3001 Volume 2",
            emergency_action=self._emergency_thermal_action
        ))
        
        # Power System - Battery Charge
        self.register_constraint(SafetyConstraint(
            name="battery_charge",
            parameter_type=ParameterType.POWER_SYSTEM,
            safety_level=SafetyLevel.CRITICAL,
            min_value=10.0,    # % - minimum for emergency operations
            max_value=100.0,   # % - maximum charge
            description="Main battery charge level",
            justification="Battery power required for life support during solar eclipse",
            reference_document="Lunar Power Management Plan"
        ))
        
        # Power System - Solar Array Output
        self.register_constraint(SafetyConstraint(
            name="solar_generation",
            parameter_type=ParameterType.POWER_SYSTEM,
            safety_level=SafetyLevel.WARNING,
            min_value=0.0,     # kW
            max_value=50.0,    # kW - maximum array output
            description="Solar array power generation",
            justification="Primary power source for habitat operations",
            reference_document="Lunar Power Management Plan"
        ))
        
        # Consumables - Water Level
        self.register_constraint(SafetyConstraint(
            name="water_level",
            parameter_type=ParameterType.CONSUMABLES,
            safety_level=SafetyLevel.CRITICAL,
            min_value=50.0,    # liters - 7-day emergency supply
            max_value=2000.0,  # liters - tank capacity
            description="Potable water storage level",
            justification="Water required for crew survival and suit cooling",
            reference_document="Lunar Life Support Requirements"
        ))
        
        # Structural - Hull Pressure
        self.register_constraint(SafetyConstraint(
            name="hull_pressure_integrity",
            parameter_type=ParameterType.STRUCTURAL,
            safety_level=SafetyLevel.MISSION_ABORT,
            min_value=0.95,    # Integrity factor (0-1)
            max_value=1.0,     # Perfect integrity
            description="Pressure hull structural integrity",
            justification="Hull breach threatens immediate crew survival",
            reference_document="Lunar Habitat Structural Requirements",
            emergency_action=self._emergency_hull_breach_action
        ))
        
        # Communication - Signal Strength
        self.register_constraint(SafetyConstraint(
            name="earth_comm_signal",
            parameter_type=ParameterType.COMMUNICATION,
            safety_level=SafetyLevel.WARNING,
            min_value=-120.0,  # dBm - minimum for voice communication
            max_value=-30.0,   # dBm - maximum input level
            description="Earth communication signal strength",
            justification="Communication required for mission control coordination",
            reference_document="Lunar Communication Requirements"
        ))
        
        # Emergency System - Response Time
        self.register_constraint(SafetyConstraint(
            name="emergency_response_time",
            parameter_type=ParameterType.EMERGENCY_SYSTEM,
            safety_level=SafetyLevel.CRITICAL,
            min_value=0.0,     # seconds
            max_value=30.0,    # seconds - maximum acceptable response time
            description="Emergency system response time",
            justification="Rapid response required for crew safety",
            reference_document="Emergency Response Procedures"
        ))
        
        # Crew Health - Heart Rate
        self.register_constraint(SafetyConstraint(
            name="crew_heart_rate",
            parameter_type=ParameterType.CREW_HEALTH,
            safety_level=SafetyLevel.WARNING,
            min_value=50.0,    # bpm - resting minimum
            max_value=180.0,   # bpm - exercise maximum
            description="Crew member heart rate",
            justification="Heart rate indicates crew health and stress level",
            reference_document="Crew Health Monitoring Protocol"
        ))
    
    def register_constraint(self, constraint: SafetyConstraint):
        """Register a safety constraint."""
        self.constraints[constraint.name] = constraint
        self.logger.info(f"Safety constraint registered: {constraint.name} ({constraint.safety_level.value})")
        
        if self.audit_logger:
            self.audit_logger.log_system_event(
                action="constraint_registered",
                component="safety_validator",
                description=f"Safety constraint registered: {constraint.name}",
                details={
                    'constraint_name': constraint.name,
                    'parameter_type': constraint.parameter_type.value,
                    'safety_level': constraint.safety_level.value,
                    'min_value': constraint.min_value,
                    'max_value': constraint.max_value
                }
            )
    
    def validate_parameter(self, name: str, value: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a single parameter against safety constraints.
        
        Args:
            name: Parameter name
            value: Parameter value
            context: Additional validation context
            
        Returns:
            Validation result dictionary
        """
        self.validation_count += 1
        start_time = time.time()
        
        result = {
            'parameter': name,
            'value': value,
            'valid': True,
            'violations': [],
            'safety_level': SafetyLevel.INFORMATIONAL,
            'timestamp': datetime.utcnow(),
            'validation_time_ms': 0
        }
        
        try:
            # Record parameter history
            self.parameter_history[name].append((datetime.utcnow(), value))
            # Keep only last 1000 values
            if len(self.parameter_history[name]) > 1000:
                self.parameter_history[name] = self.parameter_history[name][-1000:]
            
            # Check if constraint exists
            if name in self.constraints:
                constraint = self.constraints[name]
                is_valid, violations = constraint.validate(value, context)
                
                result.update({
                    'valid': is_valid,
                    'violations': violations,
                    'safety_level': constraint.safety_level,
                    'parameter_type': constraint.parameter_type.value
                })
                
                if not is_valid:
                    self.violation_count += 1
                    self.last_violation_time = datetime.utcnow()
                    
                    # Log violation
                    self._log_safety_violation(constraint, value, violations)
                    
                    # Execute violation actions
                    self._execute_violation_actions(constraint, value, violations)
            else:
                # Parameter not constrained - log for awareness
                self.logger.debug(f"Unconstrained parameter validated: {name} = {value}")
            
            # Validate temporal constraints (rate limiting, stability)
            self._validate_temporal_constraints(name, value, result)
            
        except Exception as e:
            result.update({
                'valid': False,
                'violations': [f"Validation error: {str(e)}"],
                'safety_level': SafetyLevel.CRITICAL
            })
            self.logger.error(f"Parameter validation failed for {name}: {e}")
        
        finally:
            result['validation_time_ms'] = (time.time() - start_time) * 1000
        
        return result
    
    def validate_system_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete system state against all safety constraints.
        
        Args:
            state: Complete system state dictionary
            
        Returns:
            Comprehensive validation result
        """
        start_time = time.time()
        
        validation_results = []
        overall_valid = True
        highest_safety_level = SafetyLevel.INFORMATIONAL
        
        # Validate each parameter
        for param_name, value in state.items():
            result = self.validate_parameter(param_name, value, state)
            validation_results.append(result)
            
            if not result['valid']:
                overall_valid = False
            
            # Track highest safety level
            param_safety_level = result['safety_level']
            if self._safety_level_priority(param_safety_level) > self._safety_level_priority(highest_safety_level):
                highest_safety_level = param_safety_level
        
        # Cross-parameter validation
        cross_validation_results = self._validate_cross_parameter_constraints(state)
        validation_results.extend(cross_validation_results)
        
        # Update overall results based on cross-validation
        for result in cross_validation_results:
            if not result['valid']:
                overall_valid = False
            if self._safety_level_priority(result['safety_level']) > self._safety_level_priority(highest_safety_level):
                highest_safety_level = result['safety_level']
        
        # Calculate overall system safety score
        safety_score = self._calculate_safety_score(validation_results)
        
        comprehensive_result = {
            'timestamp': datetime.utcnow(),
            'overall_valid': overall_valid,
            'safety_score': safety_score,
            'highest_safety_level': highest_safety_level.value,
            'total_parameters': len(state),
            'validated_parameters': len([r for r in validation_results if 'parameter' in r]),
            'violations': sum(len(r['violations']) for r in validation_results),
            'validation_time_ms': (time.time() - start_time) * 1000,
            'parameter_results': validation_results,
            'emergency_state': self.emergency_state,
            'mission_abort_triggered': self.mission_abort_triggered
        }
        
        # Log system validation
        if self.audit_logger:
            self.audit_logger.log_system_event(
                action="system_state_validation",
                component="safety_validator",
                description=f"System state validation: {'PASS' if overall_valid else 'FAIL'}",
                details={
                    'safety_score': safety_score,
                    'highest_safety_level': highest_safety_level.value,
                    'total_violations': comprehensive_result['violations'],
                    'validation_time_ms': comprehensive_result['validation_time_ms']
                },
                level=AuditLevel.WARNING if not overall_valid else AuditLevel.INFO,
                success=overall_valid
            )
        
        # Trigger emergency protocols if needed
        if highest_safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.MISSION_ABORT]:
            self._trigger_emergency_protocols(highest_safety_level, comprehensive_result)
        
        return comprehensive_result
    
    def _validate_temporal_constraints(self, name: str, value: Any, result: Dict[str, Any]):
        """Validate temporal constraints like rate limits and stability."""
        if name not in self.constraints:
            return
        
        constraint = self.constraints[name]
        history = self.parameter_history[name]
        
        if len(history) < 2:
            return  # Need at least 2 values for temporal validation
        
        # Rate limit validation
        if constraint.rate_limit and isinstance(value, (int, float)):
            prev_time, prev_value = history[-2]
            current_time = history[-1][0]
            
            if isinstance(prev_value, (int, float)):
                time_diff = (current_time - prev_time).total_seconds()
                if time_diff > 0:
                    rate = abs(value - prev_value) / time_diff
                    if rate > constraint.rate_limit:
                        result['violations'].append(
                            f"{name} rate of change exceeds limit: {rate:.2f} > {constraint.rate_limit}"
                        )
                        result['valid'] = False
        
        # Stability validation (if required)
        if constraint.stability_required and len(history) >= 5:
            recent_values = [v for _, v in history[-5:]]
            if isinstance(value, (int, float)) and all(isinstance(v, (int, float)) for v in recent_values):
                # Check if values are stable (within 5% of mean)
                mean_value = sum(recent_values) / len(recent_values)
                max_deviation = max(abs(v - mean_value) for v in recent_values)
                if mean_value != 0 and (max_deviation / mean_value) > 0.05:
                    result['violations'].append(f"{name} lacks required stability")
                    result['valid'] = False
    
    def _validate_cross_parameter_constraints(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate constraints that depend on multiple parameters."""
        cross_results = []
        
        # O2/CO2 balance check
        if 'o2_pressure' in state and 'co2_pressure' in state:
            o2 = state['o2_pressure']
            co2 = state['co2_pressure']
            
            if isinstance(o2, (int, float)) and isinstance(co2, (int, float)):
                # Check if O2/CO2 ratio is safe
                if o2 > 0 and (co2 / o2) > 0.1:  # CO2 should be < 10% of O2
                    cross_results.append({
                        'parameter': 'o2_co2_balance',
                        'value': f"O2:{o2}, CO2:{co2}",
                        'valid': False,
                        'violations': [f"Dangerous O2/CO2 ratio: CO2/O2 = {co2/o2:.3f}"],
                        'safety_level': SafetyLevel.CRITICAL,
                        'parameter_type': 'life_support'
                    })
        
        # Power generation vs consumption check
        if 'solar_generation' in state and 'total_power_load' in state:
            generation = state['solar_generation']
            load = state['total_power_load']
            
            if isinstance(generation, (int, float)) and isinstance(load, (int, float)):
                if generation < load * 0.8:  # Generation should be at least 80% of load
                    cross_results.append({
                        'parameter': 'power_balance',
                        'value': f"Gen:{generation}, Load:{load}",
                        'valid': False,
                        'violations': [f"Insufficient power generation: {generation} < {load * 0.8}"],
                        'safety_level': SafetyLevel.WARNING,
                        'parameter_type': 'power_system'
                    })
        
        # Temperature vs pressure correlation
        if 'temperature' in state and 'total_pressure' in state:
            temp = state['temperature']
            pressure = state['total_pressure']
            
            if isinstance(temp, (int, float)) and isinstance(pressure, (int, float)):
                # Check for unrealistic temperature-pressure combinations
                # Using simplified ideal gas relationship check
                if temp < 0 and pressure > 90:  # High pressure with freezing temp
                    cross_results.append({
                        'parameter': 'temp_pressure_correlation',
                        'value': f"T:{temp}, P:{pressure}",
                        'valid': False,
                        'violations': [f"Unrealistic temperature-pressure combination"],
                        'safety_level': SafetyLevel.WARNING,
                        'parameter_type': 'thermal_control'
                    })
        
        return cross_results
    
    def _calculate_safety_score(self, validation_results: List[Dict[str, Any]]) -> float:
        """Calculate overall safety score (0.0 to 1.0)."""
        if not validation_results:
            return 1.0
        
        total_weight = 0
        weighted_score = 0
        
        safety_weights = {
            SafetyLevel.MISSION_ABORT: 10.0,
            SafetyLevel.EMERGENCY: 8.0,
            SafetyLevel.CRITICAL: 6.0,
            SafetyLevel.WARNING: 4.0,
            SafetyLevel.CAUTION: 2.0,
            SafetyLevel.ADVISORY: 1.5,
            SafetyLevel.INFORMATIONAL: 1.0
        }
        
        for result in validation_results:
            safety_level = result.get('safety_level', SafetyLevel.INFORMATIONAL)
            if isinstance(safety_level, str):
                safety_level = SafetyLevel(safety_level)
            
            weight = safety_weights.get(safety_level, 1.0)
            score = 1.0 if result['valid'] else 0.0
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 1.0
    
    def _safety_level_priority(self, level: SafetyLevel) -> int:
        """Get numeric priority for safety level."""
        priorities = {
            SafetyLevel.INFORMATIONAL: 0,
            SafetyLevel.ADVISORY: 1,
            SafetyLevel.CAUTION: 2,
            SafetyLevel.WARNING: 3,
            SafetyLevel.CRITICAL: 4,
            SafetyLevel.EMERGENCY: 5,
            SafetyLevel.MISSION_ABORT: 6
        }
        return priorities.get(level, 0)
    
    def _log_safety_violation(self, constraint: SafetyConstraint, value: Any, violations: List[str]):
        """Log safety constraint violation."""
        violation_details = {
            'constraint_name': constraint.name,
            'parameter_type': constraint.parameter_type.value,
            'safety_level': constraint.safety_level.value,
            'value': value,
            'violations': violations,
            'min_value': constraint.min_value,
            'max_value': constraint.max_value
        }
        
        # Log safety violation if audit logger is available
        if self.audit_logger and AuditLevel:
            # Determine audit level based on safety level
            audit_level = {
                SafetyLevel.MISSION_ABORT: AuditLevel.EMERGENCY,
                SafetyLevel.EMERGENCY: AuditLevel.EMERGENCY,
                SafetyLevel.CRITICAL: AuditLevel.CRITICAL,
                SafetyLevel.WARNING: AuditLevel.ERROR,
                SafetyLevel.CAUTION: AuditLevel.WARNING,
                SafetyLevel.ADVISORY: AuditLevel.WARNING,
                SafetyLevel.INFORMATIONAL: AuditLevel.INFO
            }.get(constraint.safety_level, AuditLevel.WARNING)
            
            self.audit_logger.log_system_event(
                action="safety_violation",
                component="safety_validator",
                description=f"Safety violation: {constraint.name} = {value}",
                details=violation_details,
                level=audit_level,
                success=False
            )
    
    def _execute_violation_actions(self, constraint: SafetyConstraint, value: Any, violations: List[str]):
        """Execute actions for safety violations."""
        try:
            # Execute constraint-specific violation action
            if constraint.violation_action:
                constraint.violation_action()
                self.logger.info(f"Executed violation action for {constraint.name}")
            
            # Execute emergency action for severe violations
            if constraint.safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.MISSION_ABORT]:
                if constraint.emergency_action:
                    constraint.emergency_action()
                    self.logger.critical(f"Executed emergency action for {constraint.name}")
                
                # Set emergency state
                if constraint.safety_level == SafetyLevel.MISSION_ABORT:
                    self.mission_abort_triggered = True
                    self.logger.critical("🚨 MISSION ABORT TRIGGERED 🚨")
                elif constraint.safety_level == SafetyLevel.EMERGENCY:
                    self.emergency_state = True
                    self.logger.critical("🚨 EMERGENCY STATE ACTIVATED 🚨")
            
        except Exception as e:
            self.logger.error(f"Error executing violation actions for {constraint.name}: {e}")
    
    def _trigger_emergency_protocols(self, safety_level: SafetyLevel, validation_result: Dict[str, Any]):
        """Trigger emergency protocols based on safety level."""
        if self.audit_logger:
            if safety_level == SafetyLevel.MISSION_ABORT:
                self.audit_logger.log_emergency_event(
                    action="mission_abort_protocol",
                    description="Mission abort protocol triggered due to safety violations",
                    details=validation_result
                )
                # In a real system, this would trigger actual abort procedures
                
            elif safety_level == SafetyLevel.EMERGENCY:
                self.audit_logger.log_emergency_event(
                    action="emergency_protocol",
                    description="Emergency protocol triggered due to safety violations",
                    details=validation_result
                )
    
    # Emergency action methods
    def _emergency_o2_action(self):
        """Emergency action for O2 pressure violation."""
        self.logger.critical("EMERGENCY: Oxygen pressure critical - activating emergency O2 supply")
        # In real system: Activate backup O2, alert crew, prepare for EVA
    
    def _emergency_co2_action(self):
        """Emergency action for CO2 pressure violation."""
        self.logger.critical("EMERGENCY: CO2 pressure critical - activating CO2 scrubbers")
        # In real system: Increase CO2 scrubber capacity, check ventilation
    
    def _emergency_thermal_action(self):
        """Emergency action for critical temperature violation."""
        self.logger.critical("EMERGENCY: Critical temperature - activating thermal protection")
        # In real system: Adjust thermal control, protect equipment
    
    def _emergency_hull_breach_action(self):
        """Emergency action for hull breach."""
        self.logger.critical("EMERGENCY: Hull breach detected - initiating emergency protocols")
        # In real system: Seal breach, prepare emergency shelter, alert crew
    
    def get_constraint_status(self) -> Dict[str, Any]:
        """Get status of all safety constraints."""
        return {
            'total_constraints': len(self.constraints),
            'constraints_by_type': {
                param_type.value: len([c for c in self.constraints.values() if c.parameter_type == param_type])
                for param_type in ParameterType
            },
            'constraints_by_safety_level': {
                level.value: len([c for c in self.constraints.values() if c.safety_level == level])
                for level in SafetyLevel
            },
            'validation_statistics': {
                'total_validations': self.validation_count,
                'total_violations': self.violation_count,
                'violation_rate': self.violation_count / max(self.validation_count, 1),
                'last_violation': self.last_violation_time.isoformat() if self.last_violation_time else None
            },
            'system_state': {
                'emergency_state': self.emergency_state,
                'mission_abort_triggered': self.mission_abort_triggered
            }
        }


# Type checking decorators and functions
def mission_critical_validated(constraints: Dict[str, Any] = None):
    """Decorator to validate function parameters against mission safety constraints.
    
    Args:
        constraints: Dictionary of parameter constraints
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = get_mission_validator()
            
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            for param_name, value in bound_args.arguments.items():
                if constraints and param_name in constraints:
                    constraint_def = constraints[param_name]
                    # Create temporary constraint for validation
                    temp_constraint = SafetyConstraint(
                        name=param_name,
                        parameter_type=ParameterType.LIFE_SUPPORT,  # Default
                        safety_level=SafetyLevel.CRITICAL,
                        **constraint_def
                    )
                    
                    is_valid, violations = temp_constraint.validate(value)
                    if not is_valid:
                        raise ValidationError(f"Parameter {param_name} validation failed: {violations}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_life_support_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Validate life support parameters for immediate safety assessment.
    
    Args:
        parameters: Dictionary of life support parameters
        
    Returns:
        Validation results
    """
    validator = get_mission_validator()
    
    # Filter to only life support parameters
    life_support_params = {
        name: value for name, value in parameters.items()
        if name in ['o2_pressure', 'co2_pressure', 'total_pressure', 'temperature', 'humidity']
    }
    
    return validator.validate_system_state(life_support_params)


def emergency_safety_check(state: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Perform emergency safety check for immediate go/no-go decision.
    
    Args:
        state: Current system state
        
    Returns:
        Tuple of (is_safe, critical_violations)
    """
    validator = get_mission_validator()
    result = validator.validate_system_state(state)
    
    # Extract only mission-critical violations
    critical_violations = []
    for param_result in result['parameter_results']:
        if param_result.get('safety_level') in ['mission_abort', 'emergency', 'critical']:
            if not param_result['valid']:
                critical_violations.extend(param_result['violations'])
    
    is_safe = len(critical_violations) == 0 and not validator.mission_abort_triggered
    
    return is_safe, critical_violations


# Global mission safety validator instance
_global_mission_validator = None

def get_mission_validator() -> MissionSafetyValidator:
    """Get global mission safety validator instance."""
    global _global_mission_validator
    if _global_mission_validator is None:
        _global_mission_validator = MissionSafetyValidator()
    return _global_mission_validator