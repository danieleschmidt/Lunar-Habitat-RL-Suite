"""Security utilities for safe operation and data protection."""

import hashlib
import hmac
import secrets
import os
import stat
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .exceptions import ValidationError
from .logging import AuditLogger


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    ip_address: str
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if security context has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has required permission."""
        return permission in self.permissions or "admin" in self.permissions


class SecurityManager:
    """Centralized security management for the application."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self._generate_secret_key()
        self.audit_logger = AuditLogger()
        self.active_sessions = {}
        self.failed_attempts = {}
        self.security_events = []
        
        # Rate limiting configuration
        self.max_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        
    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key."""
        return secrets.token_hex(32)
    
    def create_session(self, user_id: str, ip_address: str, permissions: Optional[List[str]] = None) -> str:
        """
        Create authenticated session.
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            permissions: List of granted permissions
            
        Returns:
            Session token
        """
        
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            permissions=permissions or [],
            expires_at=expires_at
        )
        
        self.active_sessions[session_id] = context
        
        self.audit_logger.log_action(
            action_type="session_created",
            user=user_id,
            details={
                "session_id": session_id,
                "ip_address": ip_address,
                "permissions": permissions or [],
                "expires_at": expires_at.isoformat()
            }
        )
        
        return session_id
    
    def validate_session(self, session_id: str, required_permission: Optional[str] = None) -> SecurityContext:
        """
        Validate session and check permissions.
        
        Args:
            session_id: Session identifier
            required_permission: Required permission for operation
            
        Returns:
            Security context if valid
            
        Raises:
            ValidationError: If session is invalid or lacks permission
        """
        
        if session_id not in self.active_sessions:
            self._log_security_event("invalid_session", {"session_id": session_id})
            raise ValidationError("Invalid session")
        
        context = self.active_sessions[session_id]
        
        if context.is_expired():
            self._cleanup_session(session_id)
            self._log_security_event("expired_session", {"session_id": session_id})
            raise ValidationError("Session expired")
        
        if required_permission and not context.has_permission(required_permission):
            self._log_security_event("permission_denied", {
                "session_id": session_id,
                "user_id": context.user_id,
                "required_permission": required_permission,
                "user_permissions": context.permissions
            })
            raise ValidationError(f"Insufficient permissions: {required_permission} required")
        
        return context
    
    def _cleanup_session(self, session_id: str):
        """Remove session from active sessions."""
        if session_id in self.active_sessions:
            context = self.active_sessions.pop(session_id)
            self.audit_logger.log_action(
                action_type="session_destroyed",
                user=context.user_id,
                details={"session_id": session_id, "reason": "cleanup"}
            )
    
    def destroy_session(self, session_id: str):
        """Explicitly destroy session."""
        self._cleanup_session(session_id)
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-relevant events."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "details": details
        }
        
        self.security_events.append(event)
        
        self.audit_logger.log_action(
            action_type="security_event",
            user="system",
            details=event
        )
    
    def check_rate_limit(self, identifier: str, ip_address: str) -> bool:
        """
        Check if identifier/IP is rate limited.
        
        Args:
            identifier: User identifier or other key
            ip_address: Client IP address
            
        Returns:
            True if request is allowed, False if rate limited
        """
        
        current_time = datetime.utcnow()
        key = f"{identifier}:{ip_address}"
        
        # Clean old attempts
        if key in self.failed_attempts:
            attempts = self.failed_attempts[key]
            # Remove attempts older than lockout duration
            attempts['times'] = [
                t for t in attempts['times'] 
                if current_time - t < self.lockout_duration
            ]
            
            if len(attempts['times']) >= self.max_attempts:
                self._log_security_event("rate_limit_exceeded", {
                    "identifier": identifier,
                    "ip_address": ip_address,
                    "attempt_count": len(attempts['times'])
                })
                return False
        
        return True
    
    def record_failed_attempt(self, identifier: str, ip_address: str):
        """Record failed authentication attempt."""
        key = f"{identifier}:{ip_address}"
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = {'times': []}
        
        self.failed_attempts[key]['times'].append(datetime.utcnow())
        
        self._log_security_event("failed_attempt", {
            "identifier": identifier,
            "ip_address": ip_address
        })
    
    def clear_failed_attempts(self, identifier: str, ip_address: str):
        """Clear failed attempts after successful authentication."""
        key = f"{identifier}:{ip_address}"
        if key in self.failed_attempts:
            del self.failed_attempts[key]


def sanitize_input(data: Any, max_string_length: int = 1000) -> Any:
    """
    Recursively sanitize input data to prevent injection attacks.
    
    Args:
        data: Data to sanitize
        max_string_length: Maximum allowed string length
        
    Returns:
        Sanitized data
        
    Raises:
        ValidationError: If data contains dangerous content
    """
    
    if isinstance(data, str):
        return _sanitize_string(data, max_string_length)
    elif isinstance(data, dict):
        return {key: sanitize_input(value, max_string_length) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item, max_string_length) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_input(item, max_string_length) for item in data)
    else:
        return data


def _sanitize_string(text: str, max_length: int) -> str:
    """Sanitize string input."""
    
    if len(text) > max_length:
        raise ValidationError(f"String length {len(text)} exceeds maximum {max_length}")
    
    # Remove null bytes
    sanitized = text.replace('\x00', '')
    
    # Check for SQL injection patterns
    sql_patterns = [
        r"(?i)(union\s+select)",
        r"(?i)(drop\s+table)", 
        r"(?i)(delete\s+from)",
        r"(?i)(insert\s+into)",
        r"(?i)(update\s+.+set)",
        r"(?i)(\'\s*or\s*\')",
        r"(?i)(\'\s*;)",
    ]
    
    import re
    for pattern in sql_patterns:
        if re.search(pattern, sanitized):
            raise ValidationError("Input contains potential SQL injection patterns")
    
    # Check for script injection
    script_patterns = [
        r"(?i)(<script)",
        r"(?i)(javascript:)",
        r"(?i)(on\w+\s*=)",
        r"(?i)(eval\s*\()",
        r"(?i)(exec\s*\()",
    ]
    
    for pattern in script_patterns:
        if re.search(pattern, sanitized):
            raise ValidationError("Input contains potential script injection patterns")
    
    return sanitized


def check_file_permissions(file_path: Path, required_permissions: str = "r") -> bool:
    """
    Check if file has appropriate permissions for safe access.
    
    Args:
        file_path: Path to file
        required_permissions: Required permissions ('r', 'w', 'rw')
        
    Returns:
        True if permissions are appropriate and safe
    """
    
    if not file_path.exists():
        return False
    
    try:
        file_stat = file_path.stat()
    except OSError:
        return False
    
    # Check if file is owned by current user or root
    current_uid = os.getuid() if hasattr(os, 'getuid') else None
    if current_uid is not None and file_stat.st_uid not in [current_uid, 0]:
        return False
    
    # Check permissions
    mode = file_stat.st_mode
    
    if 'r' in required_permissions:
        if not (mode & stat.S_IRUSR):  # User read permission
            return False
    
    if 'w' in required_permissions:
        if not (mode & stat.S_IWUSR):  # User write permission
            return False
    
    # Security check: ensure file is not world-writable
    if mode & stat.S_IWOTH:
        return False
    
    # Security check: ensure file is not group-writable unless necessary
    if mode & stat.S_IWGRP and 'w' not in required_permissions:
        return False
    
    return True


def secure_file_write(file_path: Path, content: str, mode: int = 0o600) -> None:
    """
    Securely write content to file with appropriate permissions.
    
    Args:
        file_path: Path to write to
        content: Content to write
        mode: File permissions mode (default: owner read/write only)
        
    Raises:
        ValidationError: If write operation is unsafe
    """
    
    # Ensure parent directory exists and is secure
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Check parent directory permissions
    if not check_file_permissions(parent_dir, "rw"):
        raise ValidationError(f"Unsafe parent directory permissions: {parent_dir}")
    
    # Write with temporary file for atomicity
    temp_file = file_path.with_suffix(file_path.suffix + '.tmp')
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        # Set secure permissions
        temp_file.chmod(mode)
        
        # Atomic move
        temp_file.replace(file_path)
        
    except Exception as e:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise ValidationError(f"Secure file write failed: {str(e)}") from e


def audit_log(action: str, user: str, details: Optional[Dict[str, Any]] = None, 
              security_context: Optional[SecurityContext] = None) -> None:
    """
    Log auditable action with security context.
    
    Args:
        action: Action being performed
        user: User performing action
        details: Additional action details
        security_context: Current security context
    """
    
    audit_details = details or {}
    
    if security_context:
        audit_details.update({
            "session_id": security_context.session_id,
            "ip_address": security_context.ip_address,
            "permissions": security_context.permissions
        })
    
    # Use global audit logger
    audit_logger = AuditLogger()
    audit_logger.log_action(action, user, audit_details)


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of file for integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
        
    Returns:
        Hex-encoded hash string
    """
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def verify_hmac_signature(data: bytes, signature: str, secret_key: str, algorithm: str = "sha256") -> bool:
    """
    Verify HMAC signature for data integrity and authenticity.
    
    Args:
        data: Data to verify
        signature: HMAC signature to check
        secret_key: Secret key for HMAC
        algorithm: HMAC algorithm
        
    Returns:
        True if signature is valid
    """
    
    try:
        expected_signature = hmac.new(
            secret_key.encode('utf-8'),
            data,
            getattr(hashlib, algorithm)
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False


class ConfigSecurityValidator:
    """Validator for configuration security settings."""
    
    def __init__(self):
        self.security_warnings = []
        self.security_errors = []
    
    def validate_config_security(self, config_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Validate configuration for security issues.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Tuple of (warnings, errors)
        """
        
        self.security_warnings.clear()
        self.security_errors.clear()
        
        # Check for hardcoded secrets
        self._check_hardcoded_secrets(config_dict)
        
        # Check file paths for security
        self._check_file_paths(config_dict)
        
        # Check network settings
        self._check_network_settings(config_dict)
        
        # Check logging configuration
        self._check_logging_config(config_dict)
        
        return self.security_warnings.copy(), self.security_errors.copy()
    
    def _check_hardcoded_secrets(self, config: Dict[str, Any], path: str = ""):
        """Check for hardcoded secrets in configuration."""
        
        secret_keys = ['password', 'secret', 'key', 'token', 'api_key', 'private_key']
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self._check_hardcoded_secrets(value, current_path)
            elif isinstance(value, str):
                # Check if key suggests a secret
                if any(secret in key.lower() for secret in secret_keys):
                    if len(value) > 0 and not value.startswith('${'):  # Not an env var
                        self.security_errors.append(
                            f"Hardcoded secret detected at {current_path}"
                        )
    
    def _check_file_paths(self, config: Dict[str, Any], path: str = ""):
        """Check file paths for security issues."""
        
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self._check_file_paths(value, current_path)
            elif isinstance(value, str) and ('path' in key.lower() or 'file' in key.lower()):
                # Check for path traversal
                if '..' in value:
                    self.security_errors.append(
                        f"Potential path traversal in {current_path}: {value}"
                    )
                
                # Check for absolute paths outside safe directories
                if os.path.isabs(value):
                    safe_prefixes = ['/tmp', '/var/log', './logs', './data']
                    if not any(value.startswith(prefix) for prefix in safe_prefixes):
                        self.security_warnings.append(
                            f"Absolute path outside safe directories: {current_path} = {value}"
                        )
    
    def _check_network_settings(self, config: Dict[str, Any]):
        """Check network configuration for security."""
        
        # Check for binding to all interfaces
        if 'host' in config and config['host'] in ['0.0.0.0', '::']:
            self.security_warnings.append(
                "Service configured to bind to all interfaces (0.0.0.0)"
            )
        
        # Check for insecure protocols
        if 'protocol' in config and config['protocol'] in ['http', 'ftp', 'telnet']:
            self.security_warnings.append(
                f"Insecure protocol configured: {config['protocol']}"
            )
    
    def _check_logging_config(self, config: Dict[str, Any]):
        """Check logging configuration for security."""
        
        logging_config = config.get('logging', {})
        
        # Check if logging is disabled
        if logging_config.get('enabled', True) is False:
            self.security_warnings.append("Logging is disabled")
        
        # Check log level
        log_level = logging_config.get('level', 'INFO')
        if log_level.upper() == 'DEBUG':
            self.security_warnings.append(
                "Debug logging enabled - may expose sensitive information"
            )


def create_secure_temp_file(prefix: str = "lunar_habitat_", suffix: str = ".tmp") -> Path:
    """
    Create temporary file with secure permissions.
    
    Args:
        prefix: Filename prefix
        suffix: Filename suffix
        
    Returns:
        Path to created temporary file
    """
    
    import tempfile
    
    # Create temporary file with secure permissions
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    
    try:
        # Set secure permissions (owner read/write only)
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    
    return Path(temp_path)