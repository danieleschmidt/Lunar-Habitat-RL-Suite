#!/usr/bin/env python3
"""
Secure Input Validation System for Lunar Habitat RL Suite
NASA Mission-Critical Security Validation Framework
"""

import re
import json
import logging
from typing import Any, Dict, List, Union, Optional, Type, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import hashlib
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Security validation levels"""
    PERMISSIVE = "permissive"      # Basic validation
    STRICT = "strict"              # Enhanced validation  
    MISSION_CRITICAL = "mission_critical"  # NASA mission standards

class ValidationError(Exception):
    """Custom validation error with detailed context"""
    def __init__(self, message: str, field: str = None, value: Any = None, level: ValidationLevel = None):
        self.message = message
        self.field = field
        self.value = str(value)[:100] if value is not None else None  # Truncate for security
        self.level = level
        super().__init__(self.message)

@dataclass
class ValidationResult:
    """Validation result with security context"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[str]
    metadata: Dict[str, Any]
    validation_time: float
    security_score: float  # 0-100 security confidence score

class SecureInputValidator:
    """NASA Mission-Critical Input Validation System"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MISSION_CRITICAL):
        self.validation_level = validation_level
        self.validation_cache = {}
        self.suspicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'execfile\s*\(',
            r'reload\s*\(',
            r'vars\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'dir\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
            r'callable\s*\(',
            r'isinstance\s*\(',
            r'issubclass\s*\(',
            r'super\s*\(',
            r'property\s*\(',
            r'staticmethod\s*\(',
            r'classmethod\s*\(',
            r'type\s*\(',
            r'object\s*\(',
            r'basestring\s*\(',
            r'unicode\s*\(',
            r'buffer\s*\(',
            r'slice\s*\(',
            r'xrange\s*\(',
            r'help\s*\(',
            r'copyright\s*\(',
            r'credits\s*\(',
            r'license\s*\(',
            r'quit\s*\(',
            r'exit\s*\(',
        ]
        self.max_string_length = 10000
        self.max_list_length = 1000
        self.max_dict_size = 1000
        
    def validate_environment_input(self, env_input: Any) -> ValidationResult:
        """Validate environment inputs for space mission safety"""
        start_time = time.time()
        errors = []
        warnings = []
        security_score = 100.0
        
        try:
            # Validate input type
            if env_input is None:
                errors.append(ValidationError(
                    "Environment input cannot be None for space missions",
                    field="env_input",
                    value=env_input,
                    level=self.validation_level
                ))
                security_score -= 50
            
            # Validate configuration inputs
            if isinstance(env_input, dict):
                config_result = self._validate_config_dict(env_input)
                errors.extend(config_result.errors)
                warnings.extend(config_result.warnings)
                security_score = min(security_score, config_result.security_score)
            
            # Validate array inputs (observations, actions)
            elif isinstance(env_input, (list, tuple, np.ndarray)):
                array_result = self._validate_array_input(env_input)
                errors.extend(array_result.errors)
                warnings.extend(array_result.warnings)
                security_score = min(security_score, array_result.security_score)
            
            # Validate string inputs
            elif isinstance(env_input, str):
                string_result = self._validate_string_input(env_input)
                errors.extend(string_result.errors)
                warnings.extend(string_result.warnings)
                security_score = min(security_score, string_result.security_score)
            
            # Validate numeric inputs
            elif isinstance(env_input, (int, float, np.number)):
                numeric_result = self._validate_numeric_input(env_input)
                errors.extend(numeric_result.errors)
                warnings.extend(numeric_result.warnings)
                security_score = min(security_score, numeric_result.security_score)
            
            else:
                if self.validation_level == ValidationLevel.MISSION_CRITICAL:
                    errors.append(ValidationError(
                        f"Input type {type(env_input)} not approved for NASA missions",
                        field="type_validation",
                        value=type(env_input).__name__,
                        level=self.validation_level
                    ))
                    security_score -= 30
                else:
                    warnings.append(f"Unusual input type: {type(env_input)}")
                    security_score -= 10
                    
        except Exception as e:
            errors.append(ValidationError(
                f"Validation system error: {str(e)}",
                field="system_validation",
                level=self.validation_level
            ))
            security_score = 0
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "input_type": type(env_input).__name__,
                "validation_level": self.validation_level.value,
                "input_size": self._get_input_size(env_input)
            },
            validation_time=validation_time,
            security_score=max(0, security_score)
        )
    
    def _validate_config_dict(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary inputs"""
        errors = []
        warnings = []
        security_score = 100.0
        
        # Check dictionary size
        if len(config) > self.max_dict_size:
            errors.append(ValidationError(
                f"Configuration dictionary too large: {len(config)} > {self.max_dict_size}",
                field="config_size",
                value=len(config),
                level=self.validation_level
            ))
            security_score -= 30
        
        # Validate each key-value pair
        for key, value in config.items():
            # Validate keys
            if not isinstance(key, str):
                errors.append(ValidationError(
                    f"Configuration key must be string, got {type(key)}",
                    field=f"config_key_{key}",
                    value=key,
                    level=self.validation_level
                ))
                security_score -= 20
                continue
                
            # Check for suspicious key patterns
            if self._contains_suspicious_pattern(key):
                errors.append(ValidationError(
                    f"Suspicious pattern detected in configuration key: {key}",
                    field="config_key_security",
                    value=key,
                    level=self.validation_level
                ))
                security_score -= 40
            
            # Validate nested values recursively
            if isinstance(value, dict):
                nested_result = self._validate_config_dict(value)
                errors.extend(nested_result.errors)
                warnings.extend(nested_result.warnings)
                security_score = min(security_score, nested_result.security_score)
            elif isinstance(value, str):
                string_result = self._validate_string_input(value)
                errors.extend(string_result.errors)
                warnings.extend(string_result.warnings)
                security_score = min(security_score, string_result.security_score)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"dict_size": len(config)},
            validation_time=0,
            security_score=max(0, security_score)
        )
    
    def _validate_array_input(self, array_input: Union[List, tuple, np.ndarray]) -> ValidationResult:
        """Validate array inputs (observations, actions, etc.)"""
        errors = []
        warnings = []
        security_score = 100.0
        
        try:
            # Convert to list for unified processing
            if isinstance(array_input, np.ndarray):
                if array_input.size > self.max_list_length:
                    errors.append(ValidationError(
                        f"Array too large: {array_input.size} > {self.max_list_length}",
                        field="array_size",
                        value=array_input.size,
                        level=self.validation_level
                    ))
                    security_score -= 30
                
                # Check for invalid values
                if np.any(np.isnan(array_input)):
                    errors.append(ValidationError(
                        "Array contains NaN values",
                        field="array_nan",
                        level=self.validation_level
                    ))
                    security_score -= 25
                
                if np.any(np.isinf(array_input)):
                    errors.append(ValidationError(
                        "Array contains infinite values",
                        field="array_inf",
                        level=self.validation_level
                    ))
                    security_score -= 25
                    
            else:
                # Handle list/tuple
                if len(array_input) > self.max_list_length:
                    errors.append(ValidationError(
                        f"List/tuple too large: {len(array_input)} > {self.max_list_length}",
                        field="list_size",
                        value=len(array_input),
                        level=self.validation_level
                    ))
                    security_score -= 30
                
                # Validate each element
                for i, element in enumerate(array_input):
                    if isinstance(element, str):
                        if self._contains_suspicious_pattern(element):
                            errors.append(ValidationError(
                                f"Suspicious pattern in array element {i}",
                                field=f"array_element_{i}",
                                value=element,
                                level=self.validation_level
                            ))
                            security_score -= 20
                            
        except Exception as e:
            errors.append(ValidationError(
                f"Array validation error: {str(e)}",
                field="array_validation",
                level=self.validation_level
            ))
            security_score = 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"array_type": type(array_input).__name__},
            validation_time=0,
            security_score=max(0, security_score)
        )
    
    def _validate_string_input(self, string_input: str) -> ValidationResult:
        """Validate string inputs for security threats"""
        errors = []
        warnings = []
        security_score = 100.0
        
        # Check string length
        if len(string_input) > self.max_string_length:
            errors.append(ValidationError(
                f"String too long: {len(string_input)} > {self.max_string_length}",
                field="string_length",
                value=len(string_input),
                level=self.validation_level
            ))
            security_score -= 30
        
        # Check for suspicious patterns
        if self._contains_suspicious_pattern(string_input):
            errors.append(ValidationError(
                "Suspicious code pattern detected in string",
                field="string_security",
                value=string_input[:100],  # Truncate for logging
                level=self.validation_level
            ))
            security_score -= 50
        
        # Check for control characters
        if any(ord(c) < 32 and c not in '\t\n\r' for c in string_input):
            warnings.append("String contains control characters")
            security_score -= 10
        
        # Check encoding
        try:
            string_input.encode('utf-8')
        except UnicodeEncodeError:
            errors.append(ValidationError(
                "String contains invalid Unicode characters",
                field="string_encoding",
                level=self.validation_level
            ))
            security_score -= 20
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"string_length": len(string_input)},
            validation_time=0,
            security_score=max(0, security_score)
        )
    
    def _validate_numeric_input(self, numeric_input: Union[int, float, np.number]) -> ValidationResult:
        """Validate numeric inputs for space mission constraints"""
        errors = []
        warnings = []
        security_score = 100.0
        
        try:
            # Check for NaN/Inf
            if isinstance(numeric_input, (float, np.floating)):
                if np.isnan(numeric_input):
                    errors.append(ValidationError(
                        "Numeric input is NaN",
                        field="numeric_nan",
                        value=numeric_input,
                        level=self.validation_level
                    ))
                    security_score -= 40
                
                if np.isinf(numeric_input):
                    errors.append(ValidationError(
                        "Numeric input is infinite",
                        field="numeric_inf",
                        value=numeric_input,
                        level=self.validation_level
                    ))
                    security_score -= 40
            
            # Check for extreme values (space mission constraints)
            if abs(float(numeric_input)) > 1e10:
                warnings.append(f"Extremely large numeric value: {numeric_input}")
                security_score -= 5
            
            # Check for negative values where inappropriate
            if self.validation_level == ValidationLevel.MISSION_CRITICAL:
                if isinstance(numeric_input, (int, float, np.number)) and numeric_input < 0:
                    # Some values should never be negative in space missions
                    if "pressure" in str(numeric_input) or "temperature" in str(numeric_input):
                        warnings.append("Negative value for physical quantity")
                        security_score -= 10
                        
        except Exception as e:
            errors.append(ValidationError(
                f"Numeric validation error: {str(e)}",
                field="numeric_validation",
                level=self.validation_level
            ))
            security_score = 0
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"numeric_type": type(numeric_input).__name__},
            validation_time=0,
            security_score=max(0, security_score)
        )
    
    def _contains_suspicious_pattern(self, text: str) -> bool:
        """Check if text contains suspicious code patterns"""
        text_lower = text.lower()
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Additional checks for common attack vectors
        dangerous_keywords = [
            'import', 'exec', 'eval', 'compile', 'open', 'file',
            'subprocess', 'os.system', 'os.popen', '__builtins__',
            'reload', 'input', 'raw_input'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in text_lower:
                return True
                
        return False
    
    def _get_input_size(self, input_data: Any) -> int:
        """Get size of input data"""
        try:
            if isinstance(input_data, str):
                return len(input_data)
            elif isinstance(input_data, (list, tuple)):
                return len(input_data)
            elif isinstance(input_data, dict):
                return len(input_data)
            elif isinstance(input_data, np.ndarray):
                return input_data.size
            else:
                return 1
        except:
            return 0

class MissionCriticalInputFilter:
    """Mission-critical input filtering for NASA standards"""
    
    def __init__(self):
        self.validator = SecureInputValidator(ValidationLevel.MISSION_CRITICAL)
        self.audit_log = []
    
    def filter_and_validate(self, input_data: Any, context: str = "unknown") -> Any:
        """Filter and validate input data, raising exceptions for unsafe inputs"""
        
        # Perform validation
        result = self.validator.validate_environment_input(input_data)
        
        # Log the validation attempt
        self.audit_log.append({
            "timestamp": time.time(),
            "context": context,
            "input_type": type(input_data).__name__,
            "validation_result": result.is_valid,
            "security_score": result.security_score,
            "errors": len(result.errors),
            "warnings": len(result.warnings)
        })
        
        # Raise exception for invalid inputs
        if not result.is_valid:
            error_messages = [str(error) for error in result.errors]
            raise ValidationError(
                f"Input validation failed for {context}: {'; '.join(error_messages)}",
                field=context,
                value=input_data
            )
        
        # Log warnings for review
        if result.warnings:
            logger.warning(f"Input validation warnings for {context}: {'; '.join(result.warnings)}")
        
        # Return validated input
        return input_data
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of validation audit trail"""
        if not self.audit_log:
            return {"total_validations": 0}
        
        total_validations = len(self.audit_log)
        failed_validations = sum(1 for entry in self.audit_log if not entry["validation_result"])
        avg_security_score = sum(entry["security_score"] for entry in self.audit_log) / total_validations
        
        return {
            "total_validations": total_validations,
            "failed_validations": failed_validations,
            "success_rate": (total_validations - failed_validations) / total_validations * 100,
            "average_security_score": avg_security_score,
            "recent_validations": self.audit_log[-10:] if len(self.audit_log) > 10 else self.audit_log
        }

# Global mission-critical input filter instance
mission_input_filter = MissionCriticalInputFilter()

def validate_mission_input(input_data: Any, context: str = "mission_input") -> Any:
    """
    Convenient function for mission-critical input validation
    
    Args:
        input_data: The input data to validate
        context: Context description for audit logging
        
    Returns:
        The validated input data
        
    Raises:
        ValidationError: If input fails validation
    """
    return mission_input_filter.filter_and_validate(input_data, context)

def get_validation_audit() -> Dict[str, Any]:
    """Get validation audit summary"""
    return mission_input_filter.get_audit_summary()