"""Robust logging system with multiple handlers and safety measures - Generation 2"""

import os
import sys
import logging
import logging.handlers
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path


class SecurityAwareFormatter(logging.Formatter):
    """Custom formatter that sanitizes sensitive information from logs."""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'auth', 'credential',
        'api_key', 'access_token', 'private_key', 'cert'
    ]
    
    def format(self, record):
        # Sanitize sensitive information
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            msg = record.msg.lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in msg:
                    record.msg = "[REDACTED - SENSITIVE INFORMATION]"
                    break
        
        # Add NASA mission context
        if not hasattr(record, 'mission_context'):
            record.mission_context = "LUNAR_HABITAT_RL"
        
        return super().format(record)


class RobustLogger:
    """Production-grade logging system for lunar habitat RL suite."""
    
    def __init__(self, name: str = "lunar_habitat_rl", log_dir: str = "logs"):
        """Initialize robust logging system.
        
        Args:
            name: Logger name.
            log_dir: Directory for log files.
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handlers()
        self._setup_json_handler()
        self._setup_error_handler()
        
        # Performance tracking
        self.performance_data = []
        
    def _setup_console_handler(self):
        """Setup colored console output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Color formatting for console
        console_format = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
        console_formatter = SecurityAwareFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Setup rotating file handlers."""
        # Main log file with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_format = "%(asctime)s [%(levelname)8s] %(name)s [%(filename)s:%(lineno)d] %(message)s"
        file_formatter = SecurityAwareFormatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_json_handler(self):
        """Setup structured JSON logging for analysis."""
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_structured.jsonl",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        json_handler.setLevel(logging.INFO)
        
        class JSONFormatter(SecurityAwareFormatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'message': record.getMessage(),
                    'mission_context': getattr(record, 'mission_context', 'LUNAR_HABITAT_RL')
                }
                
                # Add extra fields if present
                if hasattr(record, 'performance_metrics'):
                    log_entry['performance'] = record.performance_metrics
                if hasattr(record, 'error_details'):
                    log_entry['error'] = record.error_details
                
                return json.dumps(log_entry)
        
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
    
    def _setup_error_handler(self):
        """Setup dedicated error handler for critical issues."""
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        error_handler.setLevel(logging.ERROR)
        
        error_format = "%(asctime)s [CRITICAL] %(name)s:\n%(message)s\n" + "="*80 + "\n"
        error_formatter = SecurityAwareFormatter(error_format)
        error_handler.setFormatter(error_formatter)
        
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional performance metrics."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details."""
        extra = kwargs.copy()
        if error:
            extra['error_details'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message for mission-critical failures."""
        extra = kwargs.copy()
        if error:
            extra['error_details'] = {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            }
        self.logger.critical(f"🚨 MISSION CRITICAL: {message}", extra=extra)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        perf_data = {
            'operation': operation,
            'duration_ms': duration * 1000,
            'timestamp': time.time(),
            **metrics
        }
        
        self.performance_data.append(perf_data)
        
        # Keep only last 1000 performance records in memory
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]
        
        self.info(f"Performance: {operation} completed in {duration*1000:.2f}ms",
                 performance_metrics=perf_data)
    
    def log_simulation_state(self, step: int, state: Dict[str, Any], reward: float, status: str):
        """Log simulation state for debugging and analysis."""
        state_summary = {
            'step': step,
            'reward': reward,
            'status': status,
            'o2_pressure': state.get('atmosphere', {}).get('o2_partial_pressure', 0),
            'co2_pressure': state.get('atmosphere', {}).get('co2_partial_pressure', 0),
            'battery_charge': state.get('power', {}).get('battery_charge', 0),
            'avg_temp': state.get('thermal', {}).get('avg_temperature', 0),
            'water_level': state.get('water', {}).get('potable_water', 0)
        }
        
        self.info(f"Simulation step {step}: status={status}, reward={reward:.3f}",
                 simulation_state=state_summary)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_data:
            return {}
        
        durations = [p['duration_ms'] for p in self.performance_data]
        operations = {}
        
        for perf in self.performance_data:
            op = perf['operation']
            if op not in operations:
                operations[op] = []
            operations[op].append(perf['duration_ms'])
        
        summary = {
            'total_operations': len(self.performance_data),
            'avg_duration_ms': sum(durations) / len(durations),
            'max_duration_ms': max(durations),
            'min_duration_ms': min(durations),
            'operations': {}
        }
        
        for op, times in operations.items():
            summary['operations'][op] = {
                'count': len(times),
                'avg_ms': sum(times) / len(times),
                'max_ms': max(times),
                'min_ms': min(times)
            }
        
        return summary


class PerformanceMonitor:
    """Context manager for automatic performance monitoring."""
    
    def __init__(self, logger: RobustLogger, operation: str, **metrics):
        """Initialize performance monitor.
        
        Args:
            logger: Logger instance.
            operation: Name of operation being monitored.
            **metrics: Additional metrics to track.
        """
        self.logger = logger
        self.operation = operation
        self.metrics = metrics
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            if exc_type:
                self.metrics['error'] = str(exc_val)
                self.logger.error(f"Performance: {self.operation} failed after {duration*1000:.2f}ms",
                                error=exc_val, performance_metrics=self.metrics)
            else:
                self.logger.log_performance(self.operation, duration, **self.metrics)


# Global logger instance
_global_logger = None

def get_logger(name: str = "lunar_habitat_rl") -> RobustLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = RobustLogger(name)
    return _global_logger


def log_exception(func):
    """Decorator to automatically log exceptions."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}", error=e)
            raise
    return wrapper


def log_performance(operation_name: str):
    """Decorator to automatically log performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            with PerformanceMonitor(logger, operation_name or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator