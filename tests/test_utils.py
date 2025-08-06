"""Tests for utility functions and helpers."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from lunar_habitat_rl.utils import (
    setup_logging, get_logger, validate_config, validate_action, validate_state,
    sanitize_input, check_file_permissions, audit_log,
    LunarHabitatError, ConfigurationError, PhysicsError, SafetyError
)
from lunar_habitat_rl.utils.validation import validate_numeric_range, validate_array
from lunar_habitat_rl.utils.security import SecurityManager, ConfigSecurityValidator
from lunar_habitat_rl.core.config import HabitatConfig
from tests import TEST_CONFIG, create_temp_directory, cleanup_temp_directory


class TestExceptions:
    """Test custom exception classes."""
    
    def test_lunar_habitat_error(self):
        """Test base exception class."""
        error = LunarHabitatError("Test error", error_code="TEST_001", details={"key": "value"})
        
        assert str(error) == "[TEST_001] Test error Details: {'key': 'value'}"
        assert error.error_code == "TEST_001"
        assert error.details == {"key": "value"}
    
    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError("Invalid config", config_field="volume", invalid_value=-1)
        
        assert "CONFIG_ERROR" in str(error)
        assert error.details['config_field'] == "volume"
        assert error.details['invalid_value'] == -1
    
    def test_physics_error(self):
        """Test physics error."""
        error = PhysicsError("Simulation failed", simulator="thermal", physics_state={"temp": 1000})
        
        assert "PHYSICS_ERROR" in str(error)
        assert error.details['simulator'] == "thermal"
        assert error.details['physics_state'] == {"temp": 1000}
    
    def test_safety_error(self):
        """Test safety error."""
        error = SafetyError("Oxygen depleted", safety_system="life_support", severity="critical")
        
        assert "SAFETY_VIOLATION" in str(error)
        assert error.details['safety_system'] == "life_support"
        assert error.details['severity'] == "critical"


class TestLogging:
    """Test logging functionality."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        temp_dir = create_temp_directory()
        
        try:
            loggers = setup_logging(
                level="DEBUG",
                log_dir=str(temp_dir),
                structured=True
            )
            
            assert 'main' in loggers
            assert 'physics' in loggers
            assert 'safety' in loggers
            
            # Test logging
            logger = loggers['main']
            logger.info("Test message")
            
            # Check log file was created
            log_files = list(temp_dir.glob("*.log"))
            assert len(log_files) > 0
            
        finally:
            cleanup_temp_directory(temp_dir)
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger("test_component")
        
        assert logger.name == "lunar_habitat_rl.test_component"
        
        # Test logging
        logger.info("Test message")  # Should not raise exception


class TestValidation:
    """Test validation utilities."""
    
    def test_validate_numeric_range(self):
        """Test numeric range validation."""
        # Valid values
        assert validate_numeric_range(5.0, min_val=0.0, max_val=10.0) == 5.0
        assert validate_numeric_range(0, min_val=0.0, max_val=10.0) == 0.0
        
        # Invalid values
        with pytest.raises(Exception):  # ValidationError
            validate_numeric_range(-1.0, min_val=0.0, max_val=10.0)
        
        with pytest.raises(Exception):
            validate_numeric_range(15.0, min_val=0.0, max_val=10.0)
        
        with pytest.raises(Exception):
            validate_numeric_range("not_a_number", min_val=0.0, max_val=10.0)
        
        with pytest.raises(Exception):
            validate_numeric_range(np.inf, min_val=0.0, max_val=10.0)
        
        with pytest.raises(Exception):
            validate_numeric_range(np.nan, min_val=0.0, max_val=10.0)
    
    def test_validate_array(self):
        """Test array validation."""
        # Valid arrays
        arr = validate_array([1, 2, 3], expected_shape=(3,), dtype=float)
        assert arr.shape == (3,)
        assert arr.dtype == float
        
        # Valid numpy array
        np_arr = np.array([[1, 2], [3, 4]])
        validated = validate_array(np_arr, expected_shape=(2, 2))
        assert validated.shape == (2, 2)
        
        # Invalid shape
        with pytest.raises(Exception):
            validate_array([1, 2, 3], expected_shape=(2,))
        
        # Invalid values
        with pytest.raises(Exception):
            validate_array([1, np.inf, 3], min_val=0.0, max_val=10.0)
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid configuration dict
        config_dict = {
            "volume": 200.0,
            "pressure_nominal": 101.3,
            "o2_nominal": 21.3,
            "n2_nominal": 79.0,
            "co2_limit": 0.4
        }
        
        validated_config = validate_config(config_dict)
        assert isinstance(validated_config, HabitatConfig)
        assert validated_config.volume == 200.0
        
        # Valid HabitatConfig object
        original_config = HabitatConfig(volume=150.0)
        validated_config2 = validate_config(original_config)
        assert isinstance(validated_config2, HabitatConfig)
        assert validated_config2.volume == 150.0
        
        # Invalid configuration
        invalid_config = {"volume": -100.0}  # Negative volume
        with pytest.raises(Exception):  # ConfigurationError
            validate_config(invalid_config)
    
    def test_validate_action(self):
        """Test action validation."""
        # Valid action
        valid_action = np.array([0.5, 0.7, 0.3, 0.8, 0.6])
        validated = validate_action(valid_action)
        assert isinstance(validated, np.ndarray)
        assert validated.dtype == np.float32
        
        # Invalid action - out of range
        with pytest.raises(Exception):
            validate_action(np.array([1.5, 0.5, 0.3]))  # Value > 1.0
        
        # Invalid action - NaN
        with pytest.raises(Exception):
            validate_action(np.array([np.nan, 0.5, 0.3]))
        
        # Invalid action - not convertible to array
        with pytest.raises(Exception):
            validate_action("not_an_array")
    
    def test_validate_state(self):
        """Test state validation."""
        # Valid state
        valid_state = np.random.random(48)  # Typical state size
        validated = validate_state(valid_state, expected_size=48)
        assert isinstance(validated, np.ndarray)
        assert len(validated) == 48
        
        # Invalid state size
        with pytest.raises(Exception):
            validate_state(np.random.random(10), expected_size=48)
        
        # Invalid state values
        invalid_state = np.array([1.0, np.inf, 3.0])
        with pytest.raises(Exception):
            validate_state(invalid_state)


class TestSecurity:
    """Test security utilities."""
    
    def test_sanitize_input(self):
        """Test input sanitization.""" 
        # Safe input
        safe_data = {"name": "test", "value": 42}
        sanitized = sanitize_input(safe_data)
        assert sanitized == safe_data
        
        # String sanitization
        long_string = "a" * 2000  # Exceeds default max length
        with pytest.raises(Exception):
            sanitize_input(long_string)
        
        # Nested data sanitization
        nested_data = {"level1": {"level2": ["item1", "item2"]}}
        sanitized_nested = sanitize_input(nested_data)
        assert sanitized_nested == nested_data
    
    def test_security_manager(self):
        """Test security manager functionality."""
        security_manager = SecurityManager()
        
        # Test session creation
        session_id = security_manager.create_session(
            user_id="test_user",
            ip_address="127.0.0.1",
            permissions=["read", "write"]
        )
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # Test session validation
        context = security_manager.validate_session(session_id)
        assert context.user_id == "test_user"
        assert context.ip_address == "127.0.0.1"
        assert "read" in context.permissions
        
        # Test permission checking
        context_with_permission = security_manager.validate_session(session_id, required_permission="read")
        assert context_with_permission is not None
        
        # Test insufficient permissions
        with pytest.raises(Exception):
            security_manager.validate_session(session_id, required_permission="admin")
        
        # Test session cleanup
        security_manager.destroy_session(session_id)
        with pytest.raises(Exception):
            security_manager.validate_session(session_id)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        security_manager = SecurityManager()
        security_manager.max_attempts = 3
        
        # Test normal rate limiting
        identifier = "test_user"
        ip_address = "127.0.0.1"
        
        # First few attempts should be allowed
        for _ in range(2):
            assert security_manager.check_rate_limit(identifier, ip_address) is True
            security_manager.record_failed_attempt(identifier, ip_address)
        
        # After max attempts, should be rate limited
        security_manager.record_failed_attempt(identifier, ip_address)
        assert security_manager.check_rate_limit(identifier, ip_address) is False
        
        # Clear attempts
        security_manager.clear_failed_attempts(identifier, ip_address)
        assert security_manager.check_rate_limit(identifier, ip_address) is True
    
    def test_config_security_validator(self):
        """Test configuration security validation."""
        validator = ConfigSecurityValidator()
        
        # Test secure configuration
        secure_config = {
            "database_url": "${DB_URL}",  # Environment variable
            "log_level": "INFO",
            "host": "localhost"
        }
        
        warnings, errors = validator.validate_config_security(secure_config)
        assert len(errors) == 0  # Should have no security errors
        
        # Test insecure configuration
        insecure_config = {
            "password": "hardcoded_password",  # Hardcoded secret
            "api_key": "abc123",
            "host": "0.0.0.0",  # Binds to all interfaces
            "protocol": "http"  # Insecure protocol
        }
        
        warnings, errors = validator.validate_config_security(insecure_config)
        assert len(errors) > 0  # Should detect hardcoded secrets
        assert len(warnings) > 0  # Should warn about insecure settings
    
    def test_file_permissions(self):
        """Test file permission checking."""
        temp_dir = create_temp_directory()
        
        try:
            # Create test file
            test_file = temp_dir / "test_file.txt"
            test_file.write_text("test content")
            
            # Test readable file
            assert check_file_permissions(test_file, required_permissions="r") is True
            
            # Test writable file
            assert check_file_permissions(test_file, required_permissions="w") is True
            
            # Test non-existent file
            non_existent = temp_dir / "non_existent.txt"
            assert check_file_permissions(non_existent, required_permissions="r") is False
            
        finally:
            cleanup_temp_directory(temp_dir)
    
    def test_audit_logging(self):
        """Test audit logging functionality."""
        temp_dir = create_temp_directory()
        
        try:
            # Mock audit logger to write to temp directory
            with patch('lunar_habitat_rl.utils.security.AuditLogger') as mock_audit_logger:
                mock_instance = Mock()
                mock_audit_logger.return_value = mock_instance
                
                # Test audit logging
                audit_log("test_action", "test_user", {"key": "value"})
                
                # Verify audit logger was called
                mock_instance.log_action.assert_called_once_with(
                    "test_action", "test_user", {"key": "value"}
                )
                
        finally:
            cleanup_temp_directory(temp_dir)


class TestCaching:
    """Test caching utilities."""
    
    def test_lru_cache(self):
        """Test LRU cache functionality."""
        from lunar_habitat_rl.optimization.caching import LRUCache
        
        cache = LRUCache(max_size=3, max_memory_mb=1)
        
        # Test basic operations
        assert cache.put("key1", "value1") is True
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test LRU eviction
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        cache.put("key4", "value4")  # Should evict key1 (least recently used)
        
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key4") == "value4"  # Should be present
        
        # Test cache stats
        stats = cache.stats()
        assert 'size' in stats
        assert 'memory_usage_mb' in stats
        assert stats['size'] <= 3  # Should not exceed max size
    
    def test_simulation_cache(self):
        """Test simulation-specific caching."""
        from lunar_habitat_rl.optimization.caching import SimulationCache
        
        cache = SimulationCache(memory_cache_size=10, persistent_cache=False)
        
        # Test caching simulation results
        initial_state = {"temperature": 22.0, "pressure": 101.3}
        parameters = {"heating_power": 0.5, "fan_speed": 0.8}
        result = {"final_temperature": 23.5, "energy_used": 1000.0}
        
        # Cache result
        success = cache.put_simulation_result(
            simulator_type="thermal",
            initial_state=initial_state,
            parameters=parameters,
            timestep=60.0,
            duration=3600.0,
            result=result
        )
        assert success is True
        
        # Retrieve result
        cached_result = cache.get_simulation_result(
            simulator_type="thermal",
            initial_state=initial_state,
            parameters=parameters,
            timestep=60.0,
            duration=3600.0
        )
        
        assert cached_result == result
        
        # Test cache hit rate
        hit_rate = cache.get_hit_rate()
        assert 0.0 <= hit_rate <= 1.0


class TestPerformanceMonitoring:
    """Test performance monitoring utilities."""
    
    def test_performance_profiler(self):
        """Test performance profiler."""
        from lunar_habitat_rl.optimization.performance import PerformanceProfiler
        
        profiler = PerformanceProfiler(enable_profiling=True)
        
        # Test operation timing
        with profiler.time_operation("test_operation"):
            # Simulate some work
            sum(range(1000))
        
        # Check metric was recorded
        stats = profiler.get_metric_stats("test_operation")
        assert 'count' in stats
        assert 'mean' in stats
        assert stats['count'] >= 1
        
        # Test function profiling decorator
        @profiler.profile_function("test_function")
        def test_function():
            return sum(range(100))
        
        result = test_function()
        assert result == sum(range(100))  # Function should work normally
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        from lunar_habitat_rl.optimization.performance import ResourceMonitor
        
        monitor = ResourceMonitor(monitoring_interval=0.1, history_size=10)
        
        # Test current usage
        current_usage = monitor._collect_resource_usage()
        assert current_usage is not None
        assert current_usage.cpu_percent >= 0
        assert current_usage.memory_mb > 0
        
        # Test monitoring start/stop
        monitor.start_monitoring()
        import time
        time.sleep(0.2)  # Let it collect some data
        monitor.stop_monitoring()
        
        # Check if data was collected
        usage_stats = monitor.get_usage_stats()
        if usage_stats:  # May be empty if monitoring was very brief
            assert 'cpu' in usage_stats or len(monitor.resource_history) > 0


@pytest.mark.integration
class TestIntegration:
    """Integration tests for utility components."""
    
    def test_logging_with_security(self):
        """Test integration of logging and security."""
        temp_dir = create_temp_directory()
        
        try:
            # Setup logging
            loggers = setup_logging(log_dir=str(temp_dir))
            logger = loggers['main']
            
            # Test security logging
            security_manager = SecurityManager()
            session_id = security_manager.create_session(
                user_id="test_user",
                ip_address="127.0.0.1"
            )
            
            # Should create audit entries
            assert session_id is not None
            
        finally:
            cleanup_temp_directory(temp_dir)
    
    def test_validation_with_config(self):
        """Test validation integration with configuration."""
        # Create configuration
        config = HabitatConfig(volume=200.0, crew={"size": 4})
        
        # Validate it
        validated_config = validate_config(config)
        assert isinstance(validated_config, HabitatConfig)
        assert validated_config.volume == 200.0
        assert validated_config.crew.size == 4