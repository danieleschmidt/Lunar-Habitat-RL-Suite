"""Utility functions and helpers for the Lunar Habitat RL Suite - Generation 2."""

# Import Generation 2 robust implementations
try:
    from .robust_logging import get_logger, PerformanceMonitor, log_exception, log_performance
    from .robust_validation import get_validator, validate_and_sanitize_action, validate_and_sanitize_observation
    from .robust_monitoring import get_health_checker, get_simulation_monitor, monitor_simulation_performance
    ROBUST_MODE = True
except ImportError:
    # Fallback to basic implementations
    try:
        from .logging import setup_logging, get_logger
        from .validation import validate_config, validate_action, validate_state
        ROBUST_MODE = False
    except ImportError:
        # Minimal fallback
        def get_logger(*args, **kwargs):
            import logging
            return logging.getLogger(__name__)
        ROBUST_MODE = False

# Always available imports
try:
    from .exceptions import LunarHabitatError, ConfigurationError, PhysicsError, SafetyError, ValidationError, EnvironmentError
except ImportError:
    # Define minimal exceptions
    class LunarHabitatError(Exception): pass
    class ConfigurationError(LunarHabitatError): pass
    class PhysicsError(LunarHabitatError): pass
    class SafetyError(LunarHabitatError): pass
    class ValidationError(LunarHabitatError): pass
    class EnvironmentError(LunarHabitatError): pass

# Generation 2 exports
if ROBUST_MODE:
    __all__ = [
        "get_logger",
        "PerformanceMonitor", 
        "log_exception",
        "log_performance",
        "get_validator",
        "validate_and_sanitize_action",
        "validate_and_sanitize_observation",
        "get_health_checker",
        "get_simulation_monitor",
        "monitor_simulation_performance",
        "LunarHabitatError",
        "ConfigurationError", 
        "PhysicsError",
        "SafetyError",
        "ValidationError",
        "EnvironmentError",
        "ROBUST_MODE",
    ]
else:
    # Fallback exports
    __all__ = [
        "get_logger",
        "LunarHabitatError",
        "ConfigurationError", 
        "PhysicsError",
        "SafetyError",
        "ValidationError",
        "EnvironmentError",
        "ROBUST_MODE",
    ]