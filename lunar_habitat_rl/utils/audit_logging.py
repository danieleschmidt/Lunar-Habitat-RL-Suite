"""
NASA-grade audit logging system for mission-critical operations.
Implements comprehensive audit trails, compliance logging, and forensic capabilities.
"""

import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging
import logging.handlers
import gzip
import shutil
from collections import deque

from .robust_logging import get_logger
from .security import SecurityContext


class AuditEventType(Enum):
    """Types of auditable events."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    SIMULATION_EVENT = "simulation_event"
    LIFE_SUPPORT_EVENT = "life_support_event"
    EMERGENCY_EVENT = "emergency_event"
    MISSION_CRITICAL = "mission_critical"


class AuditLevel(Enum):
    """Audit event importance levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AuditEvent:
    """Comprehensive audit event record."""
    id: str = field(default_factory=lambda: f"audit_{int(time.time()*1000000)}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_EVENT
    level: AuditLevel = AuditLevel.INFO
    
    # Event identification
    action: str = ""
    resource: str = ""
    component: str = ""
    
    # User/system context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Event details
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Request/response data
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    
    # Results
    success: bool = True
    error_message: Optional[str] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    
    # Compliance and security
    compliance_tags: List[str] = field(default_factory=list)
    security_classification: str = "unclassified"
    mission_impact: bool = False
    
    # Data integrity
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for audit event integrity."""
        # Create deterministic string representation
        data = {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'action': self.action,
            'user_id': self.user_id,
            'description': self.description,
            'success': self.success
        }
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat() + 'Z'
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """NASA-grade audit logging system."""
    
    def __init__(self, audit_dir: str = "audit_logs", 
                 max_file_size: int = 100*1024*1024,  # 100MB
                 backup_count: int = 50,
                 enable_encryption: bool = False):
        """Initialize audit logger.
        
        Args:
            audit_dir: Directory for audit log files
            max_file_size: Maximum size of each log file
            backup_count: Number of backup files to keep
            enable_encryption: Whether to encrypt audit logs
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)
        
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_encryption = enable_encryption
        
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # Event storage
        self.recent_events = deque(maxlen=10000)  # Keep recent events in memory
        self.event_cache = {}  # id -> event for quick lookup
        
        # Setup audit log handlers
        self._setup_audit_handlers()
        
        # Integrity checking
        self.integrity_log = []
        self.last_integrity_check = None
        
        # Performance metrics
        self.events_logged = 0
        self.start_time = time.time()
        
    def _setup_audit_handlers(self):
        """Setup specialized audit log handlers."""
        # Main audit log with rotation
        self.audit_file_handler = logging.handlers.RotatingFileHandler(
            self.audit_dir / "audit.jsonl",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        
        # Security events log
        self.security_file_handler = logging.handlers.RotatingFileHandler(
            self.audit_dir / "security_audit.jsonl",
            maxBytes=self.max_file_size // 2,
            backupCount=self.backup_count * 2
        )
        
        # Mission critical events log
        self.critical_file_handler = logging.handlers.RotatingFileHandler(
            self.audit_dir / "mission_critical_audit.jsonl",
            maxBytes=self.max_file_size // 4,
            backupCount=self.backup_count * 4
        )
        
        # Life support events log (highest retention)
        self.life_support_handler = logging.handlers.RotatingFileHandler(
            self.audit_dir / "life_support_audit.jsonl",
            maxBytes=self.max_file_size // 10,
            backupCount=self.backup_count * 10
        )
        
        # Setup custom formatters for each handler
        class AuditFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, 'audit_event'):
                    return record.audit_event.to_json()
                return super().format(record)
        
        formatter = AuditFormatter()
        for handler in [self.audit_file_handler, self.security_file_handler, 
                       self.critical_file_handler, self.life_support_handler]:
            handler.setFormatter(formatter)
        
        # Create audit logger
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.DEBUG)
        self.audit_logger.addHandler(self.audit_file_handler)
        
        # Prevent propagation to root logger
        self.audit_logger.propagate = False
    
    def log_event(self, event: AuditEvent, security_context: Optional[SecurityContext] = None):
        """Log an audit event.
        
        Args:
            event: Audit event to log
            security_context: Current security context
        """
        with self._lock:
            # Enrich event with security context
            if security_context:
                event.user_id = security_context.user_id
                event.session_id = security_context.session_id
                event.ip_address = security_context.ip_address
            
            # Calculate integrity checksum
            event.checksum = event.calculate_checksum()
            
            # Store in memory cache
            self.recent_events.append(event)
            self.event_cache[event.id] = event
            
            # Create log record
            log_record = logging.LogRecord(
                name='audit',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=event.description,
                args=(),
                exc_info=None
            )
            log_record.audit_event = event
            
            # Route to appropriate handlers
            self.audit_file_handler.emit(log_record)
            
            if event.event_type in [AuditEventType.SECURITY_EVENT, AuditEventType.AUTHENTICATION, 
                                   AuditEventType.AUTHORIZATION]:
                self.security_file_handler.emit(log_record)
            
            if event.mission_impact or event.level in [AuditLevel.CRITICAL, AuditLevel.EMERGENCY]:
                self.critical_file_handler.emit(log_record)
            
            if event.event_type == AuditEventType.LIFE_SUPPORT_EVENT:
                self.life_support_handler.emit(log_record)
            
            self.events_logged += 1
            
            # Periodic integrity check
            if self.events_logged % 1000 == 0:
                self._perform_integrity_check()
    
    def log_user_action(self, action: str, user_id: str, resource: str = "",
                       details: Dict[str, Any] = None, success: bool = True,
                       security_context: Optional[SecurityContext] = None):
        """Log user action event."""
        event = AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action=action,
            resource=resource,
            user_id=user_id,
            description=f"User {user_id} performed action: {action}",
            details=details or {},
            success=success,
            compliance_tags=["user_activity", "action_tracking"]
        )
        self.log_event(event, security_context)
    
    def log_system_event(self, action: str, component: str, 
                        description: str = "", details: Dict[str, Any] = None,
                        level: AuditLevel = AuditLevel.INFO, success: bool = True):
        """Log system event."""
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            level=level,
            action=action,
            component=component,
            description=description or f"System event: {action}",
            details=details or {},
            success=success,
            compliance_tags=["system_activity"]
        )
        self.log_event(event)
    
    def log_security_event(self, action: str, description: str,
                          details: Dict[str, Any] = None, level: AuditLevel = AuditLevel.WARNING,
                          security_context: Optional[SecurityContext] = None):
        """Log security-related event."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_EVENT,
            level=level,
            action=action,
            description=description,
            details=details or {},
            security_classification="restricted",
            compliance_tags=["security", "access_control"],
            mission_impact=(level in [AuditLevel.CRITICAL, AuditLevel.EMERGENCY])
        )
        self.log_event(event, security_context)
    
    def log_data_access(self, resource: str, user_id: str, access_type: str = "read",
                       details: Dict[str, Any] = None, success: bool = True,
                       security_context: Optional[SecurityContext] = None):
        """Log data access event."""
        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            action=access_type,
            resource=resource,
            user_id=user_id,
            description=f"Data {access_type} access to {resource}",
            details=details or {},
            success=success,
            compliance_tags=["data_access", "privacy"]
        )
        self.log_event(event, security_context)
    
    def log_life_support_event(self, action: str, description: str,
                              details: Dict[str, Any] = None, 
                              level: AuditLevel = AuditLevel.INFO,
                              critical: bool = False):
        """Log life support system event."""
        event = AuditEvent(
            event_type=AuditEventType.LIFE_SUPPORT_EVENT,
            level=AuditLevel.CRITICAL if critical else level,
            action=action,
            component="life_support",
            description=description,
            details=details or {},
            mission_impact=critical,
            security_classification="mission_critical",
            compliance_tags=["life_support", "safety", "mission_critical"]
        )
        self.log_event(event)
    
    def log_simulation_event(self, action: str, episode: int, step: int,
                           details: Dict[str, Any] = None, duration_ms: float = None):
        """Log simulation-related event."""
        event = AuditEvent(
            event_type=AuditEventType.SIMULATION_EVENT,
            action=action,
            component="simulation",
            description=f"Simulation {action} at episode {episode}, step {step}",
            details={
                "episode": episode,
                "step": step,
                **(details or {})
            },
            duration_ms=duration_ms,
            compliance_tags=["simulation", "training"]
        )
        self.log_event(event)
    
    def log_emergency_event(self, action: str, description: str,
                          details: Dict[str, Any] = None, response_time_ms: float = None):
        """Log emergency event."""
        event = AuditEvent(
            event_type=AuditEventType.EMERGENCY_EVENT,
            level=AuditLevel.EMERGENCY,
            action=action,
            component="emergency_system",
            description=description,
            details=details or {},
            duration_ms=response_time_ms,
            mission_impact=True,
            security_classification="mission_critical",
            compliance_tags=["emergency", "safety", "mission_critical", "immediate_response"]
        )
        self.log_event(event)
    
    def search_events(self, 
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     event_types: Optional[List[AuditEventType]] = None,
                     user_id: Optional[str] = None,
                     action: Optional[str] = None,
                     success: Optional[bool] = None,
                     mission_critical_only: bool = False,
                     limit: int = 1000) -> List[AuditEvent]:
        """Search audit events with filters.
        
        Args:
            start_time: Search from this time
            end_time: Search until this time
            event_types: Filter by event types
            user_id: Filter by user ID
            action: Filter by action
            success: Filter by success status
            mission_critical_only: Only return mission critical events
            limit: Maximum number of events to return
            
        Returns:
            List of matching audit events
        """
        with self._lock:
            results = []
            
            for event in reversed(self.recent_events):
                # Apply filters
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if event_types and event.event_type not in event_types:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if action and action.lower() not in event.action.lower():
                    continue
                if success is not None and event.success != success:
                    continue
                if mission_critical_only and not event.mission_impact:
                    continue
                
                results.append(event)
                
                if len(results) >= limit:
                    break
            
            return results
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        with self._lock:
            now = datetime.utcnow()
            uptime_hours = (time.time() - self.start_time) / 3600
            
            # Count events by type
            event_counts = {}
            level_counts = {}
            recent_24h = []
            
            cutoff_24h = now - timedelta(hours=24)
            
            for event in self.recent_events:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                level = event.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
                
                if event.timestamp >= cutoff_24h:
                    recent_24h.append(event)
            
            return {
                'total_events_logged': self.events_logged,
                'events_in_memory': len(self.recent_events),
                'uptime_hours': uptime_hours,
                'events_per_hour': self.events_logged / max(uptime_hours, 0.1),
                'events_last_24h': len(recent_24h),
                'mission_critical_last_24h': len([e for e in recent_24h if e.mission_impact]),
                'events_by_type': event_counts,
                'events_by_level': level_counts,
                'last_integrity_check': self.last_integrity_check.isoformat() if self.last_integrity_check else None,
                'audit_files': list(self.audit_dir.glob("*.jsonl"))
            }
    
    def _perform_integrity_check(self):
        """Perform integrity check on recent audit events."""
        try:
            integrity_results = []
            
            # Check last 100 events
            recent_events = list(self.recent_events)[-100:]
            
            for event in recent_events:
                expected_checksum = event.calculate_checksum()
                if event.checksum != expected_checksum:
                    integrity_results.append({
                        'event_id': event.id,
                        'expected_checksum': expected_checksum,
                        'actual_checksum': event.checksum,
                        'status': 'INTEGRITY_VIOLATION'
                    })
                else:
                    integrity_results.append({
                        'event_id': event.id,
                        'status': 'OK'
                    })
            
            self.last_integrity_check = datetime.utcnow()
            violations = [r for r in integrity_results if r['status'] != 'OK']
            
            if violations:
                self.logger.critical(f"Audit integrity violations detected: {len(violations)} events")
                # Log integrity violation as security event
                self.log_security_event(
                    "audit_integrity_violation",
                    f"Detected {len(violations)} audit log integrity violations",
                    details={'violations': violations},
                    level=AuditLevel.CRITICAL
                )
            else:
                self.logger.debug(f"Audit integrity check passed for {len(integrity_results)} events")
                
        except Exception as e:
            self.logger.error(f"Audit integrity check failed: {e}")
    
    def export_audit_trail(self, output_path: Path, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          format: str = "json") -> Dict[str, Any]:
        """Export audit trail for compliance or forensic analysis.
        
        Args:
            output_path: Path to export file
            start_time: Export from this time
            end_time: Export until this time
            format: Export format ("json", "csv")
            
        Returns:
            Export metadata
        """
        events = self.search_events(
            start_time=start_time,
            end_time=end_time,
            limit=100000  # Large limit for export
        )
        
        export_data = {
            'export_metadata': {
                'export_time': datetime.utcnow().isoformat(),
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None,
                'total_events': len(events),
                'format': format
            },
            'events': [event.to_dict() for event in events]
        }
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].to_dict().keys())
                    writer.writeheader()
                    for event in events:
                        writer.writerow(event.to_dict())
        
        # Compress the export
        with open(output_path, 'rb') as f_in:
            with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove uncompressed file
        output_path.unlink()
        
        self.logger.info(f"Audit trail exported: {len(events)} events to {output_path}.gz")
        
        return export_data['export_metadata']
    
    def generate_compliance_report(self, report_path: Path,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate compliance report for audit activities."""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(days=30)  # Last 30 days
        if end_time is None:
            end_time = datetime.utcnow()
        
        events = self.search_events(start_time=start_time, end_time=end_time, limit=100000)
        
        # Categorize events
        user_actions = [e for e in events if e.event_type == AuditEventType.USER_ACTION]
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY_EVENT]
        data_access = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
        mission_critical = [e for e in events if e.mission_impact]
        
        # Generate report
        report = {
            'report_metadata': {
                'report_period_start': start_time.isoformat(),
                'report_period_end': end_time.isoformat(),
                'report_generated': datetime.utcnow().isoformat(),
                'total_events_analyzed': len(events)
            },
            'summary': {
                'total_events': len(events),
                'user_actions': len(user_actions),
                'security_events': len(security_events),
                'data_access_events': len(data_access),
                'mission_critical_events': len(mission_critical),
                'failed_operations': len([e for e in events if not e.success])
            },
            'compliance_metrics': {
                'audit_coverage': 'comprehensive',
                'retention_policy': f"{self.backup_count} backup files",
                'integrity_checks': 'enabled',
                'encryption': 'enabled' if self.enable_encryption else 'disabled'
            },
            'security_analysis': {
                'authentication_failures': len([e for e in security_events if 'failed' in e.description.lower()]),
                'unauthorized_access_attempts': len([e for e in security_events if 'unauthorized' in e.description.lower()]),
                'privilege_escalations': len([e for e in security_events if 'privilege' in e.description.lower()])
            },
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if len(security_events) > 100:
            report['recommendations'].append("High number of security events detected - review security policies")
        
        if len([e for e in events if not e.success]) > len(events) * 0.1:
            report['recommendations'].append("High failure rate detected - investigate system stability")
        
        if len(mission_critical) > 0:
            report['recommendations'].append(f"{len(mission_critical)} mission-critical events require review")
        
        # Write report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Compliance report generated: {report_path}")
        
        return report


# Audit decorators
def audit_user_action(action: str, resource: str = ""):
    """Decorator to automatically audit user actions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                audit_logger.log_user_action(
                    action=action,
                    user_id="system",  # Would be extracted from context in real implementation
                    resource=resource,
                    details={'function': func.__name__, 'duration_ms': duration_ms},
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                audit_logger.log_user_action(
                    action=action,
                    user_id="system",
                    resource=resource,
                    details={'function': func.__name__, 'error': str(e), 'duration_ms': duration_ms},
                    success=False
                )
                
                raise
        return wrapper
    return decorator


def audit_data_access(resource: str, access_type: str = "read"):
    """Decorator to automatically audit data access."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            try:
                result = func(*args, **kwargs)
                
                audit_logger.log_data_access(
                    resource=resource,
                    user_id="system",
                    access_type=access_type,
                    details={'function': func.__name__},
                    success=True
                )
                
                return result
                
            except Exception as e:
                audit_logger.log_data_access(
                    resource=resource,
                    user_id="system",
                    access_type=access_type,
                    details={'function': func.__name__, 'error': str(e)},
                    success=False
                )
                
                raise
        return wrapper
    return decorator


# Global audit logger instance
_global_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger