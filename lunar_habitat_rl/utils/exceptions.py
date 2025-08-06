"""Custom exceptions for the Lunar Habitat RL Suite."""

from typing import Optional, Dict, Any


class LunarHabitatError(Exception):
    """Base exception for all Lunar Habitat RL errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            base_msg += f" Details: {self.details}"
        return base_msg


class ConfigurationError(LunarHabitatError):
    """Raised when there are configuration validation errors."""
    
    def __init__(self, message: str, config_field: Optional[str] = None, invalid_value: Optional[Any] = None):
        details = {}
        if config_field:
            details['config_field'] = config_field
        if invalid_value is not None:
            details['invalid_value'] = invalid_value
            
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details=details
        )


class PhysicsError(LunarHabitatError):
    """Raised when physics simulation encounters errors."""
    
    def __init__(self, message: str, simulator: Optional[str] = None, physics_state: Optional[Dict[str, Any]] = None):
        details = {}
        if simulator:
            details['simulator'] = simulator
        if physics_state:
            details['physics_state'] = physics_state
            
        super().__init__(
            message=message,
            error_code="PHYSICS_ERROR",
            details=details
        )


class SafetyError(LunarHabitatError):
    """Raised when safety violations are detected."""
    
    def __init__(self, message: str, safety_system: Optional[str] = None, severity: Optional[str] = None):
        details = {}
        if safety_system:
            details['safety_system'] = safety_system
        if severity:
            details['severity'] = severity
            
        super().__init__(
            message=message,
            error_code="SAFETY_VIOLATION",
            details=details
        )


class ValidationError(LunarHabitatError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, expected_type: Optional[str] = None):
        details = {}
        if field_name:
            details['field_name'] = field_name
        if expected_type:
            details['expected_type'] = expected_type
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class SimulationError(LunarHabitatError):
    """Raised when simulation encounters unrecoverable errors."""
    
    def __init__(self, message: str, simulation_step: Optional[int] = None, system_state: Optional[Dict[str, Any]] = None):
        details = {}
        if simulation_step is not None:
            details['simulation_step'] = simulation_step
        if system_state:
            details['system_state'] = system_state
            
        super().__init__(
            message=message,
            error_code="SIMULATION_ERROR",
            details=details
        )


class ResourceError(LunarHabitatError):
    """Raised when resource limits are exceeded."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, current_usage: Optional[float] = None):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if current_usage is not None:
            details['current_usage'] = current_usage
            
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details=details
        )