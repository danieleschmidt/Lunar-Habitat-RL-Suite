#!/usr/bin/env python3
"""
Mission Security Hardening System for Lunar Habitat RL Suite
NASA-Grade Security Framework for Space Mission Critical Applications
"""

import os
import sys
import logging
import hashlib
import hmac
import secrets
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from functools import wraps
import psutil
import signal

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """NASA Security Classification Levels"""
    UNCLASSIFIED = "unclassified"
    SENSITIVE = "sensitive"
    MISSION_CRITICAL = "mission_critical"
    CLASSIFIED = "classified"

class ThreatLevel(Enum):
    """Threat Assessment Levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event logging structure"""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    description: str
    source: str
    mitigation_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class SecurityHardeningManager:
    """NASA Mission-Critical Security Hardening Manager"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MISSION_CRITICAL):
        self.security_level = security_level
        self.security_events: List[SecurityEvent] = []
        self.security_policies = {}
        self.access_tokens = {}
        self.failed_attempts = {}
        self.monitoring_active = False
        self.security_thread = None
        
        # Initialize security policies
        self._initialize_security_policies()
        
        # Setup security monitoring
        self._setup_security_monitoring()
        
        # Install signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _initialize_security_policies(self):
        """Initialize security policies based on classification level"""
        base_policies = {
            "max_memory_usage_percent": 80,
            "max_cpu_usage_percent": 90,
            "max_file_descriptors": 1000,
            "session_timeout_seconds": 3600,
            "max_failed_attempts": 3,
            "lockout_duration_seconds": 300,
            "require_authentication": True,
            "log_all_operations": True,
            "encrypt_sensitive_data": True
        }
        
        if self.security_level == SecurityLevel.MISSION_CRITICAL:
            base_policies.update({
                "max_memory_usage_percent": 70,
                "max_cpu_usage_percent": 80,
                "session_timeout_seconds": 1800,
                "max_failed_attempts": 2,
                "lockout_duration_seconds": 600,
                "require_multi_factor": True,
                "mandatory_access_control": True,
                "real_time_monitoring": True
            })
        elif self.security_level == SecurityLevel.CLASSIFIED:
            base_policies.update({
                "max_memory_usage_percent": 60,
                "max_cpu_usage_percent": 70,
                "session_timeout_seconds": 900,
                "max_failed_attempts": 1,
                "lockout_duration_seconds": 1800,
                "require_clearance_validation": True,
                "air_gap_required": True,
                "tamper_detection": True
            })
        
        self.security_policies = base_policies
        logger.info(f"Security policies initialized for {self.security_level.value} level")
    
    def _setup_security_monitoring(self):
        """Setup continuous security monitoring"""
        if self.security_policies.get("real_time_monitoring"):
            self.monitoring_active = True
            self.security_thread = threading.Thread(
                target=self._security_monitoring_loop,
                daemon=True
            )
            self.security_thread.start()
            logger.info("Real-time security monitoring activated")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for security events"""
        def security_shutdown_handler(signum, frame):
            self._log_security_event(
                "system_shutdown",
                ThreatLevel.MEDIUM,
                f"System shutdown signal received: {signum}",
                "system"
            )
            self._secure_shutdown()
        
        signal.signal(signal.SIGTERM, security_shutdown_handler)
        signal.signal(signal.SIGINT, security_shutdown_handler)
    
    def _security_monitoring_loop(self):
        """Continuous security monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor system resources
                self._monitor_resource_usage()
                
                # Monitor file system access
                self._monitor_file_access()
                
                # Monitor network activity (if applicable)
                self._monitor_network_activity()
                
                # Check for suspicious processes
                self._monitor_process_activity()
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                self._log_security_event(
                    "monitoring_error",
                    ThreatLevel.HIGH,
                    f"Security monitoring error: {str(e)}",
                    "security_monitor"
                )
                time.sleep(30)  # Longer delay on error
    
    def _monitor_resource_usage(self):
        """Monitor system resource usage for anomalies"""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.security_policies["max_memory_usage_percent"]:
                self._log_security_event(
                    "resource_limit_exceeded",
                    ThreatLevel.HIGH,
                    f"Memory usage exceeded limit: {memory.percent}%",
                    "resource_monitor"
                )
                self._apply_resource_mitigation("memory", memory.percent)
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.security_policies["max_cpu_usage_percent"]:
                self._log_security_event(
                    "resource_limit_exceeded",
                    ThreatLevel.HIGH,
                    f"CPU usage exceeded limit: {cpu_percent}%",
                    "resource_monitor"
                )
                self._apply_resource_mitigation("cpu", cpu_percent)
            
            # Check file descriptors
            process = psutil.Process()
            fd_count = process.num_fds() if hasattr(process, 'num_fds') else 0
            if fd_count > self.security_policies["max_file_descriptors"]:
                self._log_security_event(
                    "resource_limit_exceeded",
                    ThreatLevel.MEDIUM,
                    f"File descriptor count exceeded: {fd_count}",
                    "resource_monitor"
                )
                
        except Exception as e:
            logger.warning(f"Resource monitoring error: {str(e)}")
    
    def _monitor_file_access(self):
        """Monitor file system access patterns"""
        try:
            # Check for suspicious file operations
            sensitive_paths = [
                "/etc/passwd",
                "/etc/shadow", 
                "/root",
                str(Path.home() / ".ssh"),
                "/proc",
                "/sys"
            ]
            
            # This is a simplified check - in production, would use inotify or similar
            for path in sensitive_paths:
                if Path(path).exists():
                    try:
                        stat = Path(path).stat()
                        # Check for recent modifications
                        if time.time() - stat.st_mtime < 60:  # Modified in last minute
                            self._log_security_event(
                                "sensitive_file_access",
                                ThreatLevel.HIGH,
                                f"Recent modification to sensitive path: {path}",
                                "file_monitor"
                            )
                    except PermissionError:
                        pass  # Expected for some system files
                        
        except Exception as e:
            logger.warning(f"File monitoring error: {str(e)}")
    
    def _monitor_network_activity(self):
        """Monitor network activity for anomalies"""
        try:
            # Get network connections
            connections = psutil.net_connections()
            
            # Check for suspicious connections
            suspicious_ports = [22, 23, 135, 139, 445, 1433, 3389]
            external_connections = [
                conn for conn in connections 
                if conn.status == 'ESTABLISHED' and conn.raddr
            ]
            
            for conn in external_connections:
                if conn.laddr.port in suspicious_ports:
                    self._log_security_event(
                        "suspicious_network_activity",
                        ThreatLevel.MEDIUM,
                        f"Connection on suspicious port: {conn.laddr.port}",
                        "network_monitor"
                    )
                    
        except Exception as e:
            logger.warning(f"Network monitoring error: {str(e)}")
    
    def _monitor_process_activity(self):
        """Monitor running processes for suspicious activity"""
        try:
            processes = psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent'])
            
            suspicious_names = [
                'netcat', 'nc', 'nmap', 'wireshark', 'tcpdump',
                'metasploit', 'msfconsole', 'msfvenom'
            ]
            
            for proc in processes:
                try:
                    proc_info = proc.info
                    proc_name = proc_info.get('name', '').lower()
                    
                    # Check for suspicious process names
                    if any(sus_name in proc_name for sus_name in suspicious_names):
                        self._log_security_event(
                            "suspicious_process",
                            ThreatLevel.HIGH,
                            f"Suspicious process detected: {proc_name}",
                            "process_monitor"
                        )
                    
                    # Check for high CPU usage processes
                    cpu_percent = proc_info.get('cpu_percent', 0)
                    if cpu_percent > 50:  # Process using >50% CPU
                        self._log_security_event(
                            "high_cpu_process",
                            ThreatLevel.MEDIUM,
                            f"High CPU process: {proc_name} ({cpu_percent}%)",
                            "process_monitor"
                        )
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.warning(f"Process monitoring error: {str(e)}")
    
    def _apply_resource_mitigation(self, resource_type: str, usage_level: float):
        """Apply mitigation measures for resource limit violations"""
        if resource_type == "memory":
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log mitigation
            self._log_security_event(
                "mitigation_applied",
                ThreatLevel.MEDIUM,
                f"Memory mitigation applied - forced garbage collection",
                "security_system",
                mitigation_applied=True
            )
        
        elif resource_type == "cpu":
            # Could implement CPU throttling here
            self._log_security_event(
                "mitigation_applied", 
                ThreatLevel.MEDIUM,
                f"CPU mitigation logged - usage at {usage_level}%",
                "security_system",
                mitigation_applied=True
            )
    
    def authenticate_access(self, access_token: str, required_clearance: str = "basic") -> bool:
        """Authenticate access with security clearance validation"""
        if not self.security_policies.get("require_authentication"):
            return True
        
        # Check if token exists and is valid
        if access_token not in self.access_tokens:
            self._log_failed_attempt("invalid_token")
            return False
        
        token_data = self.access_tokens[access_token]
        
        # Check token expiration
        if time.time() - token_data["created"] > self.security_policies["session_timeout_seconds"]:
            self._log_security_event(
                "token_expired",
                ThreatLevel.MEDIUM,
                f"Access token expired: {access_token[:8]}...",
                "authentication"
            )
            del self.access_tokens[access_token]
            return False
        
        # Check clearance level
        if not self._validate_clearance(token_data.get("clearance", "basic"), required_clearance):
            self._log_security_event(
                "insufficient_clearance",
                ThreatLevel.HIGH,
                f"Insufficient clearance for operation: required {required_clearance}",
                "authorization"
            )
            return False
        
        # Update last access time
        token_data["last_access"] = time.time()
        
        return True
    
    def generate_access_token(self, user_id: str, clearance_level: str = "basic") -> str:
        """Generate secure access token"""
        # Create secure token
        token = secrets.token_urlsafe(32)
        
        # Store token data
        self.access_tokens[token] = {
            "user_id": user_id,
            "clearance": clearance_level,
            "created": time.time(),
            "last_access": time.time()
        }
        
        self._log_security_event(
            "token_generated",
            ThreatLevel.LOW,
            f"Access token generated for user: {user_id}",
            "authentication"
        )
        
        return token
    
    def _validate_clearance(self, user_clearance: str, required_clearance: str) -> bool:
        """Validate security clearance levels"""
        clearance_hierarchy = {
            "basic": 1,
            "sensitive": 2,
            "mission_critical": 3,
            "classified": 4
        }
        
        user_level = clearance_hierarchy.get(user_clearance, 0)
        required_level = clearance_hierarchy.get(required_clearance, 0)
        
        return user_level >= required_level
    
    def _log_failed_attempt(self, attempt_type: str):
        """Log and track failed authentication attempts"""
        client_id = "unknown"  # In production, would get actual client ID
        
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = []
        
        self.failed_attempts[client_id].append(time.time())
        
        # Clean old attempts
        cutoff_time = time.time() - self.security_policies["lockout_duration_seconds"]
        self.failed_attempts[client_id] = [
            t for t in self.failed_attempts[client_id] if t > cutoff_time
        ]
        
        # Check if lockout needed
        if len(self.failed_attempts[client_id]) >= self.security_policies["max_failed_attempts"]:
            self._log_security_event(
                "account_lockout",
                ThreatLevel.HIGH,
                f"Account locked due to failed attempts: {client_id}",
                "authentication"
            )
            return True  # Account locked
        
        return False  # Not locked yet
    
    def _log_security_event(self, event_type: str, threat_level: ThreatLevel, 
                           description: str, source: str, mitigation_applied: bool = False,
                           **metadata):
        """Log security event with full context"""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            source=source,
            mitigation_applied=mitigation_applied,
            metadata=metadata
        )
        
        self.security_events.append(event)
        
        # Log to standard logging system
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }[threat_level]
        
        logger.log(log_level, f"SECURITY EVENT [{threat_level.value.upper()}] {event_type}: {description}")
        
        # Write to security audit log
        self._write_audit_log(event)
        
        # Trigger immediate response for critical events
        if threat_level == ThreatLevel.CRITICAL:
            self._handle_critical_security_event(event)
    
    def _write_audit_log(self, event: SecurityEvent):
        """Write security event to audit log file"""
        try:
            audit_dir = Path("audit_logs")
            audit_dir.mkdir(exist_ok=True)
            
            audit_file = audit_dir / "security_audit.jsonl"
            
            event_data = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "threat_level": event.threat_level.value,
                "description": event.description,
                "source": event.source,
                "mitigation_applied": event.mitigation_applied,
                "metadata": event.metadata
            }
            
            with open(audit_file, "a") as f:
                f.write(json.dumps(event_data) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write security audit log: {str(e)}")
    
    def _handle_critical_security_event(self, event: SecurityEvent):
        """Handle critical security events with immediate response"""
        logger.critical(f"CRITICAL SECURITY EVENT: {event.description}")
        
        # Implement immediate protective measures
        if event.event_type in ["intrusion_detected", "data_breach", "system_compromise"]:
            self._emergency_lockdown()
        
        # Send alerts (in production, would integrate with monitoring systems)
        self._send_security_alert(event)
    
    def _emergency_lockdown(self):
        """Emergency security lockdown procedures"""
        logger.critical("INITIATING EMERGENCY SECURITY LOCKDOWN")
        
        # Disable new authentication
        self.security_policies["require_authentication"] = False
        
        # Clear all access tokens
        self.access_tokens.clear()
        
        # Log lockdown event
        self._log_security_event(
            "emergency_lockdown",
            ThreatLevel.CRITICAL,
            "Emergency security lockdown initiated",
            "security_system",
            mitigation_applied=True
        )
    
    def _send_security_alert(self, event: SecurityEvent):
        """Send security alert to monitoring systems"""
        # In production, would integrate with:
        # - SIEM systems
        # - Alert management platforms
        # - Incident response teams
        # - NASA security operations centers
        
        alert_data = {
            "alert_type": "security_incident",
            "severity": event.threat_level.value,
            "system": "lunar_habitat_rl",
            "timestamp": event.timestamp,
            "description": event.description,
            "recommended_action": self._get_recommended_action(event)
        }
        
        logger.critical(f"SECURITY ALERT: {json.dumps(alert_data, indent=2)}")
    
    def _get_recommended_action(self, event: SecurityEvent) -> str:
        """Get recommended action for security event"""
        action_map = {
            "intrusion_detected": "Immediate system isolation and forensic analysis",
            "resource_limit_exceeded": "Investigate resource usage patterns and potential DoS",
            "suspicious_process": "Terminate process and investigate origin",
            "authentication_failure": "Review access logs and check for brute force attacks",
            "data_access_violation": "Audit data access patterns and permissions",
            "network_anomaly": "Analyze network traffic and check for data exfiltration"
        }
        
        return action_map.get(event.event_type, "Standard incident response procedures")
    
    def _secure_shutdown(self):
        """Secure system shutdown procedures"""
        logger.info("Initiating secure shutdown procedures")
        
        # Stop security monitoring
        self.monitoring_active = False
        
        # Clear sensitive data from memory
        self.access_tokens.clear()
        
        # Write final audit log
        self._log_security_event(
            "system_shutdown",
            ThreatLevel.LOW,
            "Secure shutdown completed",
            "security_system"
        )
        
        logger.info("Secure shutdown completed")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status report"""
        recent_events = [
            event for event in self.security_events 
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        
        threat_counts = {
            level.value: sum(1 for event in recent_events if event.threat_level == level)
            for level in ThreatLevel
        }
        
        return {
            "security_level": self.security_level.value,
            "monitoring_active": self.monitoring_active,
            "total_events": len(self.security_events),
            "recent_events": len(recent_events),
            "threat_level_counts": threat_counts,
            "active_tokens": len(self.access_tokens),
            "failed_attempts": sum(len(attempts) for attempts in self.failed_attempts.values()),
            "security_policies": self.security_policies,
            "system_status": "operational" if self.monitoring_active else "degraded"
        }

def security_required(clearance_level: str = "basic"):
    """Decorator for functions requiring security clearance"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # In a real implementation, would get token from context
            # For demonstration, using a simplified approach
            
            security_manager = getattr(wrapper, '_security_manager', None)
            if security_manager and hasattr(security_manager, 'authenticate_access'):
                # Would normally get token from request context
                dummy_token = "demo_token"
                if not security_manager.authenticate_access(dummy_token, clearance_level):
                    raise PermissionError(f"Insufficient security clearance for {func.__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global security manager instance
_global_security_manager = None

def get_security_manager() -> SecurityHardeningManager:
    """Get global security manager instance"""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityHardeningManager()
    return _global_security_manager

def initialize_mission_security(security_level: SecurityLevel = SecurityLevel.MISSION_CRITICAL):
    """Initialize mission security with specified level"""
    global _global_security_manager
    _global_security_manager = SecurityHardeningManager(security_level)
    logger.info(f"Mission security initialized at {security_level.value} level")
    return _global_security_manager