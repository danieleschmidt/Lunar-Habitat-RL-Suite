"""Logging configuration and utilities."""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class SafetyLogger:
    """Specialized logger for safety-critical events."""
    
    def __init__(self, name: str = "safety"):
        self.logger = logging.getLogger(f"lunar_habitat_rl.{name}")
        self.safety_events = []
        
    def log_safety_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Log a safety-critical event."""
        safety_event = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        self.safety_events.append(safety_event)
        
        # Log with appropriate level based on severity
        if severity == 'critical':
            self.logger.critical("SAFETY EVENT", extra=safety_event)
        elif severity == 'high':
            self.logger.error("SAFETY EVENT", extra=safety_event)
        elif severity == 'medium':
            self.logger.warning("SAFETY EVENT", extra=safety_event)
        else:
            self.logger.info("SAFETY EVENT", extra=safety_event)
    
    def get_safety_events(self, since: Optional[datetime] = None) -> list:
        """Get safety events since specified time."""
        if since is None:
            return self.safety_events.copy()
        
        return [
            event for event in self.safety_events 
            if datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')) >= since
        ]


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 structured: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for the Lunar Habitat RL Suite.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Specific log file path (overrides log_dir)
        log_dir: Directory for log files (default: ./logs)
        structured: Use structured JSON logging format
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup log files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create log directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if log_dir is None:
            log_dir = "./logs"
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file = log_dir_path / "lunar_habitat_rl.log"
    
    # Configure root logger
    root_logger = logging.getLogger("lunar_habitat_rl")
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if structured:
        formatter = StructuredFormatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = formatter
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create specialized loggers
    loggers = {
        'main': root_logger,
        'physics': logging.getLogger("lunar_habitat_rl.physics"),
        'environment': logging.getLogger("lunar_habitat_rl.environment"),
        'algorithms': logging.getLogger("lunar_habitat_rl.algorithms"),
        'evaluation': logging.getLogger("lunar_habitat_rl.evaluation"),
        'safety': logging.getLogger("lunar_habitat_rl.safety")
    }
    
    # Configure safety logger with separate file
    safety_log_file = Path(log_file).parent / "safety.log"
    safety_handler = logging.handlers.RotatingFileHandler(
        filename=safety_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count * 2,  # Keep more safety logs
        encoding='utf-8'
    )
    safety_handler.setLevel(logging.INFO)
    safety_handler.setFormatter(formatter)
    loggers['safety'].addHandler(safety_handler)
    
    # Set individual logger levels
    loggers['physics'].setLevel(numeric_level)
    loggers['environment'].setLevel(numeric_level)
    loggers['algorithms'].setLevel(numeric_level)
    loggers['evaluation'].setLevel(numeric_level)
    loggers['safety'].setLevel(logging.INFO)  # Always capture safety events
    
    # Log initial setup message
    root_logger.info(f"Logging initialized - Level: {level}, File: {log_file}, Structured: {structured}")
    
    return loggers


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified component."""
    return logging.getLogger(f"lunar_habitat_rl.{name}")


class AuditLogger:
    """Specialized logger for audit trails and compliance."""
    
    def __init__(self, log_file: Optional[str] = None):
        if log_file is None:
            log_file = "./logs/audit.log"
        
        self.logger = logging.getLogger("lunar_habitat_rl.audit")
        self.logger.setLevel(logging.INFO)
        
        # Ensure audit logs are always written to file
        if not self.logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=10,
                encoding='utf-8'
            )
            
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_action(self, action_type: str, user: str, details: Dict[str, Any]):
        """Log an auditable action."""
        audit_entry = {
            'action_type': action_type,
            'user': user,
            'session_id': details.get('session_id'),
            'ip_address': details.get('ip_address'),
            'details': details
        }
        
        self.logger.info("AUDIT", extra=audit_entry)
    
    def log_access(self, resource: str, user: str, success: bool, reason: Optional[str] = None):
        """Log resource access attempts."""
        access_entry = {
            'resource': resource,
            'user': user,
            'access_granted': success,
            'reason': reason
        }
        
        self.logger.info("ACCESS", extra=access_entry)
    
    def log_configuration_change(self, config_field: str, old_value: Any, new_value: Any, user: str):
        """Log configuration changes."""
        config_entry = {
            'config_field': config_field,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'user': user
        }
        
        self.logger.info("CONFIG_CHANGE", extra=config_entry)


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self):
        self.logger = get_logger("performance")
        self.metrics = {}
        
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self.logger, operation_name, self.metrics)
    
    def log_metric(self, metric_name: str, value: float, unit: str, tags: Optional[Dict[str, str]] = None):
        """Log a performance metric."""
        metric_entry = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'tags': tags or {}
        }
        
        self.logger.info("METRIC", extra=metric_entry)
        
        # Store for later analysis
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append({
            'timestamp': datetime.utcnow(),
            'value': value,
            'tags': tags
        })
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            values_only = [v['value'] for v in values]
            if values_only:
                summary[metric_name] = {
                    'count': len(values_only),
                    'mean': sum(values_only) / len(values_only),
                    'min': min(values_only),
                    'max': max(values_only),
                    'latest': values_only[-1]
                }
        
        return summary


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation_name: str, metrics_store: Dict):
        self.logger = logger
        self.operation_name = operation_name
        self.metrics_store = metrics_store
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        timing_entry = {
            'operation': self.operation_name,
            'duration_seconds': duration,
            'success': exc_type is None
        }
        
        if exc_type:
            timing_entry['error'] = str(exc_val)
        
        self.logger.info("OPERATION_TIMING", extra=timing_entry)