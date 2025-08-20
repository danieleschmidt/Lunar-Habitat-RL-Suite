"""
NASA-grade fault tolerance and error handling system for lunar habitat operations.
Implements mission-critical safety protocols with automatic recovery mechanisms.
"""

import time
import threading
import queue
import traceback
import functools
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import asyncio
import concurrent.futures

from .robust_logging import get_logger
from .exceptions import ValidationError


class FailureType(Enum):
    """Classification of system failures for appropriate response."""
    TRANSIENT = "transient"           # Temporary issues (network, resource)
    PERMANENT = "permanent"           # Persistent issues requiring intervention
    CATASTROPHIC = "catastrophic"     # Mission-critical failures
    RECOVERABLE = "recoverable"       # Can be automatically recovered
    DEGRADED = "degraded"            # Partial functionality


class SystemState(Enum):
    """Overall system operational state."""
    NOMINAL = "nominal"               # All systems operating normally
    DEGRADED = "degraded"            # Reduced functionality but operational
    EMERGENCY = "emergency"          # Emergency protocols active
    SAFING = "safing"               # Entering safe mode
    SAFE_MODE = "safe_mode"         # Minimal critical operations only
    SHUTDOWN = "shutdown"           # System shutdown in progress


@dataclass
class FailureEvent:
    """Record of a system failure event."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: str = ""
    failure_type: FailureType = FailureType.TRANSIENT
    error_message: str = ""
    exception: Optional[Exception] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    impact_severity: int = 1  # 1-5 scale
    mission_critical: bool = False


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker pattern."""
    name: str
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    total_requests: int = 0
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3   # for half_open -> closed transition


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, success_threshold: int = 3):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds to wait before trying again
            success_threshold: Successes needed to close from half-open
        """
        self.state = CircuitBreakerState(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold
        )
        self.logger = get_logger()
        self._lock = threading.RLock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        with self._lock:
            if self._should_attempt_call():
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except Exception as e:
                    self._on_failure(e)
                    raise
            else:
                raise RuntimeError(f"Circuit breaker {self.state.name} is OPEN")
    
    def _should_attempt_call(self) -> bool:
        """Determine if call should be attempted based on circuit state."""
        if self.state.state == "closed":
            return True
        elif self.state.state == "open":
            # Check if recovery timeout has elapsed
            if (self.state.last_failure_time and 
                datetime.utcnow() - self.state.last_failure_time > 
                timedelta(seconds=self.state.recovery_timeout)):
                self.state.state = "half_open"
                self.state.success_count = 0
                self.logger.info(f"Circuit breaker {self.state.name} moving to HALF_OPEN")
                return True
            return False
        elif self.state.state == "half_open":
            return True
        return False
    
    def _on_success(self):
        """Handle successful operation."""
        self.state.total_requests += 1
        
        if self.state.state == "half_open":
            self.state.success_count += 1
            if self.state.success_count >= self.state.success_threshold:
                self.state.state = "closed"
                self.state.failure_count = 0
                self.logger.info(f"Circuit breaker {self.state.name} CLOSED after recovery")
        elif self.state.state == "closed":
            # Reset failure count on success
            if self.state.failure_count > 0:
                self.state.failure_count = max(0, self.state.failure_count - 1)
    
    def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        self.state.total_requests += 1
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.utcnow()
        
        if self.state.state == "closed":
            if self.state.failure_count >= self.state.failure_threshold:
                self.state.state = "open"
                self.logger.error(f"Circuit breaker {self.state.name} OPENED after {self.state.failure_count} failures")
        elif self.state.state == "half_open":
            self.state.state = "open"
            self.logger.warning(f"Circuit breaker {self.state.name} returned to OPEN from HALF_OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.state.name,
            "state": self.state.state,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count,
            "total_requests": self.state.total_requests,
            "last_failure": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
            "failure_rate": self.state.failure_count / max(1, self.state.total_requests)
        }


class RetryManager:
    """Advanced retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        """Initialize retry manager.
        
        Args:
            max_attempts: Maximum retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = get_logger()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    self.logger.info(f"Operation succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts:
                    self.logger.error(f"Operation failed after {self.max_attempts} attempts: {e}")
                    break
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = min(self.base_delay * (self.exponential_base ** (attempt - 1)), self.max_delay)
        
        if self.jitter:
            import random
            # Add ±25% jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)


class GracefulDegradation:
    """Implements graceful degradation patterns for system resilience."""
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self.degradation_levels = {}
        self.fallback_functions = {}
        self.performance_thresholds = {}
        self.logger = get_logger()
    
    def register_fallback(self, component: str, primary_func: Callable, 
                         fallback_func: Callable, performance_threshold: float = 0.5):
        """Register fallback function for a component.
        
        Args:
            component: Component identifier
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            performance_threshold: Performance threshold for fallback activation
        """
        self.fallback_functions[component] = {
            'primary': primary_func,
            'fallback': fallback_func,
            'threshold': performance_threshold
        }
        self.degradation_levels[component] = 1.0  # Full performance
    
    def execute_with_fallback(self, component: str, *args, **kwargs) -> Tuple[Any, str]:
        """Execute component function with fallback capability.
        
        Returns:
            Tuple of (result, execution_mode) where mode is 'primary' or 'fallback'
        """
        if component not in self.fallback_functions:
            raise ValueError(f"No fallback registered for component: {component}")
        
        funcs = self.fallback_functions[component]
        current_performance = self.degradation_levels.get(component, 1.0)
        
        # Try primary function first if performance is above threshold
        if current_performance >= funcs['threshold']:
            try:
                result = funcs['primary'](*args, **kwargs)
                # Success - potentially improve performance level
                self.degradation_levels[component] = min(1.0, current_performance + 0.1)
                return result, 'primary'
            except Exception as e:
                self.logger.warning(f"Primary function failed for {component}: {e}")
                # Degrade performance and fall back
                self.degradation_levels[component] = max(0.1, current_performance - 0.2)
        
        # Use fallback function
        try:
            result = funcs['fallback'](*args, **kwargs)
            self.logger.info(f"Using fallback mode for {component}")
            return result, 'fallback'
        except Exception as e:
            self.logger.error(f"Both primary and fallback failed for {component}: {e}")
            raise
    
    def get_system_health(self) -> Dict[str, float]:
        """Get current degradation levels for all components."""
        return self.degradation_levels.copy()


class MissionCriticalSafetySystem:
    """NASA-level safety system for mission-critical operations."""
    
    def __init__(self):
        """Initialize mission critical safety system."""
        self.system_state = SystemState.NOMINAL
        self.failure_history = deque(maxlen=1000)
        self.safety_constraints = {}
        self.emergency_protocols = {}
        self.watchdog_timers = {}
        self.logger = get_logger()
        self._state_lock = threading.RLock()
        
        # Critical system parameters for lunar habitat
        self.critical_parameters = {
            'o2_pressure': {'min': 16.0, 'max': 23.0, 'critical_min': 14.0},  # kPa
            'co2_pressure': {'min': 0.0, 'max': 1.0, 'critical_max': 1.5},   # kPa
            'temperature': {'min': 18.0, 'max': 26.0, 'critical_min': 10.0, 'critical_max': 35.0},  # °C
            'battery_charge': {'min': 20.0, 'max': 100.0, 'critical_min': 5.0},  # %
            'water_level': {'min': 50.0, 'max': 1000.0, 'critical_min': 10.0},  # liters
            'pressure_hull': {'min': 95.0, 'max': 105.0, 'critical_min': 90.0, 'critical_max': 110.0}  # kPa
        }
    
    def register_safety_constraint(self, parameter: str, constraint_func: Callable[[Any], bool],
                                 violation_action: Callable[[], None]):
        """Register safety constraint with violation action.
        
        Args:
            parameter: Parameter name to monitor
            constraint_func: Function to check constraint (returns True if valid)
            violation_action: Action to take on constraint violation
        """
        self.safety_constraints[parameter] = {
            'check': constraint_func,
            'action': violation_action
        }
    
    def check_safety_constraints(self, system_state: Dict[str, Any]) -> List[str]:
        """Check all safety constraints against current system state.
        
        Args:
            system_state: Current system state dictionary
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        with self._state_lock:
            # Check registered constraints
            for param, constraint in self.safety_constraints.items():
                if param in system_state:
                    try:
                        if not constraint['check'](system_state[param]):
                            violations.append(f"Safety constraint violated: {param}")
                            self.logger.critical(f"SAFETY VIOLATION: {param} = {system_state[param]}")
                            # Execute violation action
                            constraint['action']()
                    except Exception as e:
                        violations.append(f"Safety check failed for {param}: {e}")
                        self.logger.error(f"Safety check error for {param}: {e}")
            
            # Check critical parameters
            for param, limits in self.critical_parameters.items():
                if param in system_state:
                    value = system_state[param]
                    
                    # Check critical limits
                    if 'critical_min' in limits and value < limits['critical_min']:
                        violations.append(f"CRITICAL: {param} below critical minimum: {value} < {limits['critical_min']}")
                        self._initiate_emergency_protocol(f"critical_low_{param}")
                    elif 'critical_max' in limits and value > limits['critical_max']:
                        violations.append(f"CRITICAL: {param} above critical maximum: {value} > {limits['critical_max']}")
                        self._initiate_emergency_protocol(f"critical_high_{param}")
                    # Check normal operating limits
                    elif value < limits.get('min', float('-inf')) or value > limits.get('max', float('inf')):
                        violations.append(f"WARNING: {param} outside normal range: {value}")
        
        return violations
    
    def record_failure(self, component: str, failure_type: FailureType, 
                      error_message: str, exception: Optional[Exception] = None,
                      mission_critical: bool = False):
        """Record system failure event.
        
        Args:
            component: Component that failed
            failure_type: Type of failure
            error_message: Description of failure
            exception: Exception object if available
            mission_critical: Whether failure is mission critical
        """
        failure_event = FailureEvent(
            component=component,
            failure_type=failure_type,
            error_message=error_message,
            exception=exception,
            mission_critical=mission_critical,
            impact_severity=5 if mission_critical else 3
        )
        
        with self._state_lock:
            self.failure_history.append(failure_event)
            
            # Update system state based on failure
            if mission_critical or failure_type == FailureType.CATASTROPHIC:
                self._transition_to_state(SystemState.EMERGENCY)
                self.logger.critical(f"MISSION CRITICAL FAILURE: {component} - {error_message}")
            elif failure_type == FailureType.PERMANENT:
                if self.system_state == SystemState.NOMINAL:
                    self._transition_to_state(SystemState.DEGRADED)
                self.logger.error(f"PERMANENT FAILURE: {component} - {error_message}")
            else:
                self.logger.warning(f"FAILURE: {component} - {error_message}")
    
    def _transition_to_state(self, new_state: SystemState):
        """Transition system to new operational state."""
        old_state = self.system_state
        self.system_state = new_state
        
        self.logger.critical(f"SYSTEM STATE TRANSITION: {old_state.value} -> {new_state.value}")
        
        # Execute state-specific protocols
        if new_state == SystemState.EMERGENCY:
            self._execute_emergency_protocols()
        elif new_state == SystemState.SAFING:
            self._execute_safing_procedures()
        elif new_state == SystemState.SAFE_MODE:
            self._execute_safe_mode_procedures()
    
    def _initiate_emergency_protocol(self, trigger: str):
        """Initiate emergency protocol for specific trigger."""
        if trigger in self.emergency_protocols:
            try:
                self.emergency_protocols[trigger]()
                self.logger.critical(f"Emergency protocol executed: {trigger}")
            except Exception as e:
                self.logger.critical(f"Emergency protocol failed: {trigger} - {e}")
        else:
            self.logger.critical(f"No emergency protocol defined for: {trigger}")
        
        # Always transition to emergency state
        self._transition_to_state(SystemState.EMERGENCY)
    
    def _execute_emergency_protocols(self):
        """Execute all emergency protocols."""
        self.logger.critical("🚨 EXECUTING EMERGENCY PROTOCOLS 🚨")
        
        # 1. Secure life support systems
        self.logger.critical("Emergency: Securing life support systems")
        
        # 2. Save critical data
        self.logger.critical("Emergency: Saving critical system state")
        
        # 3. Alert mission control
        self.logger.critical("Emergency: Alerting mission control")
        
        # 4. Prepare for safe mode if needed
        self.logger.critical("Emergency: Preparing for potential safe mode transition")
    
    def _execute_safing_procedures(self):
        """Execute safing procedures to prepare for safe mode."""
        self.logger.critical("Executing safing procedures")
        # Implementation would depend on specific system requirements
        pass
    
    def _execute_safe_mode_procedures(self):
        """Execute safe mode procedures - minimal critical operations only."""
        self.logger.critical("Entering SAFE MODE - minimal operations only")
        # Implementation would maintain only essential life support
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._state_lock:
            recent_failures = [f for f in list(self.failure_history)[-10:]]
            critical_failures = [f for f in recent_failures if f.mission_critical]
            
            return {
                'system_state': self.system_state.value,
                'total_failures': len(self.failure_history),
                'recent_failures': len(recent_failures),
                'critical_failures': len(critical_failures),
                'constraints_registered': len(self.safety_constraints),
                'emergency_protocols': list(self.emergency_protocols.keys()),
                'last_failure': recent_failures[-1].timestamp.isoformat() if recent_failures else None
            }


class FaultTolerantOperationManager:
    """High-level manager coordinating all fault tolerance mechanisms."""
    
    def __init__(self):
        """Initialize fault tolerant operation manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.degradation_manager = GracefulDegradation()
        self.safety_system = MissionCriticalSafetySystem()
        self.logger = get_logger()
        
        # Performance monitoring
        self.operation_metrics = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_duration': 0.0,
            'last_success': None,
            'last_failure': None
        })
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, **kwargs)
        return self.circuit_breakers[name]
    
    def get_retry_manager(self, name: str, **kwargs) -> RetryManager:
        """Get or create retry manager for operation."""
        if name not in self.retry_managers:
            self.retry_managers[name] = RetryManager(**kwargs)
        return self.retry_managers[name]
    
    def fault_tolerant_execute(self, operation_name: str, func: Callable, 
                              *args, use_circuit_breaker: bool = True,
                              use_retry: bool = True, use_fallback: bool = False,
                              **kwargs) -> Any:
        """Execute operation with comprehensive fault tolerance.
        
        Args:
            operation_name: Name of operation for tracking
            func: Function to execute
            use_circuit_breaker: Enable circuit breaker protection
            use_retry: Enable retry mechanism
            use_fallback: Use graceful degradation if available
            
        Returns:
            Operation result
        """
        start_time = time.time()
        metrics = self.operation_metrics[operation_name]
        metrics['total_calls'] += 1
        
        try:
            # Apply circuit breaker if enabled
            if use_circuit_breaker:
                cb = self.get_circuit_breaker(operation_name)
                func = cb.call
            
            # Apply retry mechanism if enabled
            if use_retry:
                rm = self.get_retry_manager(operation_name)
                result = rm.execute_with_retry(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            metrics['successful_calls'] += 1
            metrics['avg_duration'] = (metrics['avg_duration'] * (metrics['total_calls'] - 1) + duration) / metrics['total_calls']
            metrics['last_success'] = datetime.utcnow()
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            metrics['failed_calls'] += 1
            metrics['last_failure'] = datetime.utcnow()
            
            self.safety_system.record_failure(
                component=operation_name,
                failure_type=FailureType.TRANSIENT,  # Default assumption
                error_message=str(e),
                exception=e
            )
            
            # Try fallback if available and enabled
            if use_fallback:
                try:
                    result, mode = self.degradation_manager.execute_with_fallback(
                        operation_name, *args, **kwargs
                    )
                    self.logger.warning(f"Fallback successful for {operation_name} (mode: {mode})")
                    return result
                except Exception as fallback_error:
                    self.logger.error(f"Both primary and fallback failed for {operation_name}: {fallback_error}")
            
            raise
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance system status."""
        return {
            'circuit_breakers': {name: cb.get_state() for name, cb in self.circuit_breakers.items()},
            'operation_metrics': dict(self.operation_metrics),
            'degradation_levels': self.degradation_manager.get_system_health(),
            'safety_system': self.safety_system.get_system_status(),
            'total_operations': len(self.operation_metrics)
        }


# Global instance for easy access
_global_fault_tolerance_manager = None

def get_fault_tolerance_manager() -> FaultTolerantOperationManager:
    """Get global fault tolerance manager instance."""
    global _global_fault_tolerance_manager
    if _global_fault_tolerance_manager is None:
        _global_fault_tolerance_manager = FaultTolerantOperationManager()
    return _global_fault_tolerance_manager


def fault_tolerant(operation_name: str = None, use_circuit_breaker: bool = True,
                  use_retry: bool = True, use_fallback: bool = False):
    """Decorator to make any function fault tolerant.
    
    Args:
        operation_name: Name for tracking (defaults to function name)
        use_circuit_breaker: Enable circuit breaker protection
        use_retry: Enable retry mechanism
        use_fallback: Use graceful degradation if available
    """
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_fault_tolerance_manager()
            return manager.fault_tolerant_execute(
                operation_name, func, *args,
                use_circuit_breaker=use_circuit_breaker,
                use_retry=use_retry,
                use_fallback=use_fallback,
                **kwargs
            )
        return wrapper
    return decorator


def mission_critical_constraint(parameter: str, min_val: float = None, 
                               max_val: float = None, emergency_action: Callable = None):
    """Decorator to add mission-critical constraint checking to functions.
    
    Args:
        parameter: Parameter name to monitor in return value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        emergency_action: Function to call on constraint violation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract parameter value from result
            if isinstance(result, dict) and parameter in result:
                value = result[parameter]
                
                # Check constraints
                violation = False
                if min_val is not None and value < min_val:
                    violation = True
                    message = f"{parameter} below minimum: {value} < {min_val}"
                elif max_val is not None and value > max_val:
                    violation = True
                    message = f"{parameter} above maximum: {value} > {max_val}"
                
                if violation:
                    safety_system = get_fault_tolerance_manager().safety_system
                    safety_system.record_failure(
                        component=func.__name__,
                        failure_type=FailureType.CATASTROPHIC,
                        error_message=message,
                        mission_critical=True
                    )
                    
                    if emergency_action:
                        emergency_action()
                    
                    raise ValidationError(f"MISSION CRITICAL CONSTRAINT VIOLATION: {message}")
            
            return result
        return wrapper
    return decorator