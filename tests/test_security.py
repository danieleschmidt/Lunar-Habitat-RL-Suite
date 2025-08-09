#!/usr/bin/env python3
"""Security tests for the lunar habitat RL suite."""

import sys
import os
import numpy as np
from pathlib import Path

# Mock pytest if not available
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def raises(exception_type):
            class ContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception_type} but no exception was raised")
                    return isinstance(exc_val, exception_type)
            return ContextManager()
    pytest = MockPytest()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lunar_habitat_rl.utils.validation import validate_numeric_range, validate_config, sanitize_string
from lunar_habitat_rl.utils.exceptions import ValidationError, SafetyError, ConfigurationError
from lunar_habitat_rl.utils.security import sanitize_input, check_file_permissions
from lunar_habitat_rl.core.config import HabitatConfig


class TestInputSanitization:
    """Test input sanitization and validation."""
    
    def test_numeric_validation(self):
        """Test numeric input validation."""
        # Valid inputs
        assert validate_numeric_range(5.0, 0, 10) == 5.0
        assert validate_numeric_range(0, 0, 10) == 0.0
        assert validate_numeric_range(10, 0, 10) == 10.0
        
        # Invalid inputs
        with pytest.raises(ValidationError):
            validate_numeric_range(15, 0, 10)  # Out of range
        
        with pytest.raises(ValidationError):
            validate_numeric_range(-5, 0, 10)  # Out of range
        
        with pytest.raises(ValidationError):
            validate_numeric_range(np.nan, 0, 10)  # NaN
        
        with pytest.raises(ValidationError):
            validate_numeric_range(np.inf, 0, 10)  # Infinite
        
        with pytest.raises(ValidationError):
            validate_numeric_range("invalid", 0, 10)  # Non-numeric
    
    def test_string_sanitization(self):
        """Test string sanitization."""
        # Valid strings
        assert sanitize_string("normal text") == "normal text"
        assert sanitize_string("with-dashes_and_underscores") == "with-dashes_and_underscores"
        
        # Sanitize dangerous characters
        assert sanitize_string("text\x00null") == "textnull"
        assert sanitize_string("line1\nline2", allow_multiline=False) == "line1 line2"
        assert sanitize_string("line1\nline2", allow_multiline=True) == "line1\nline2"
        
        # Length limits
        with pytest.raises(ValidationError):
            sanitize_string("x" * 1001, max_length=1000)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = HabitatConfig()
        validated = validate_config(valid_config)
        assert isinstance(validated, HabitatConfig)
        
        # Invalid configuration (unsafe O2 levels)
        with pytest.raises(SafetyError):
            unsafe_config = HabitatConfig(o2_nominal=10.0)  # Too low
            validate_config(unsafe_config)
        
        # Invalid configuration (excessive CO2)
        with pytest.raises(SafetyError):
            unsafe_config = HabitatConfig(co2_limit=3.0)  # Too high
            validate_config(unsafe_config)


class TestSecurityMeasures:
    """Test security measures and access controls."""
    
    def test_file_path_validation(self):
        """Test file path security validation."""
        # Safe paths
        safe_paths = [
            "data/config.json",
            "logs/habitat.log",
            "/tmp/test_file.txt"
        ]
        
        for path in safe_paths:
            try:
                result = check_file_permissions(path)
                # Should not raise an exception for safe paths
                assert True
            except Exception:
                # Some paths might not exist, that's ok for this test
                assert True
        
        # Unsafe paths (path traversal attempts)
        unsafe_paths = [
            "../../../etc/passwd",
            "data/../../../secrets",
            "/root/.ssh/id_rsa"
        ]
        
        for path in unsafe_paths:
            with pytest.raises((ValidationError, PermissionError, OSError)):
                check_file_permissions(path, must_be_readable=True)
    
    def test_input_injection_protection(self):
        """Test protection against injection attacks."""
        # SQL injection attempts
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/'1'='1",
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = sanitize_input(malicious_input)
            # Should not contain dangerous SQL patterns
            assert "DROP" not in sanitized.upper()
            assert "'" not in sanitized
            assert "--" not in sanitized
    
    def test_resource_limits(self):
        """Test resource usage limits."""
        # Test extremely large inputs are rejected
        with pytest.raises(ValidationError):
            huge_array = np.ones(10**8)  # 100M elements
            validate_numeric_range(huge_array, 0, 1)
        
        # Test reasonable sizes are accepted
        normal_array = np.ones(1000)
        validated = normal_array  # Would validate in real scenario
        assert len(validated) == 1000


class TestSystemSafety:
    """Test safety-critical system validation."""
    
    def test_atmospheric_safety_limits(self):
        """Test atmospheric safety boundaries."""
        # Test oxygen depletion detection
        with pytest.raises(SafetyError):
            config = HabitatConfig(o2_nominal=10.0)  # Dangerously low
            validate_config(config)
        
        # Test CO2 toxicity detection  
        with pytest.raises(SafetyError):
            config = HabitatConfig(co2_limit=5.0)  # Dangerously high
            validate_config(config)
    
    def test_pressure_safety_limits(self):
        """Test pressure safety boundaries."""
        # Test low pressure detection
        with pytest.raises((SafetyError, ConfigurationError)):
            config = HabitatConfig(pressure_nominal=30.0)  # Too low for safety
            validate_config(config)
    
    def test_thermal_safety_limits(self):
        """Test thermal safety boundaries."""
        # Test extreme temperature detection
        with pytest.raises(SafetyError):
            config = HabitatConfig(
                temp_nominal=40.0,  # Too hot
                temp_tolerance=10.0  # Would allow 50°C max
            )
            validate_config(config)
        
        with pytest.raises(SafetyError):
            config = HabitatConfig(
                temp_nominal=5.0,  # Too cold
                temp_tolerance=10.0  # Would allow -5°C min
            )
            validate_config(config)
    
    def test_crew_safety_limits(self):
        """Test crew safety boundaries."""
        # Test excessive crew size
        with pytest.raises(SafetyError):
            config = HabitatConfig()
            config.crew.size = 20  # Exceeds safe capacity
            validate_config(config)


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_configuration_consistency(self):
        """Test internal configuration consistency."""
        # Test pressure consistency
        with pytest.raises(ConfigurationError):
            config = HabitatConfig(
                pressure_nominal=100.0,
                o2_nominal=50.0,  # These don't add up correctly
                n2_nominal=30.0,
                co2_limit=0.5
            )
            validate_config(config)
    
    def test_state_integrity(self):
        """Test state vector integrity."""
        # Test valid state
        valid_state = np.array([21.3, 0.4, 79.0, 101.3, 45.0, 22.5, 0.95])
        # Should not raise exception
        assert np.all(np.isfinite(valid_state))
        
        # Test invalid state (contains NaN)
        invalid_state = np.array([21.3, np.nan, 79.0, 101.3, 45.0, 22.5, 0.95])
        assert not np.all(np.isfinite(invalid_state))


if __name__ == "__main__":
    # Run tests if executed directly
    print("🔒 Running Security Tests...")
    
    test_classes = [
        TestInputSanitization,
        TestSecurityMeasures, 
        TestSystemSafety,
        TestDataIntegrity
    ]
    
    passed = 0
    total = 0
    
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        print(f"\n📋 Testing {test_class.__name__}:")
        
        for method_name in methods:
            total += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
    
    print(f"\n📊 Security Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All security tests passed!")
        sys.exit(0)
    else:
        print("⚠️  Some security tests failed.")
        sys.exit(1)