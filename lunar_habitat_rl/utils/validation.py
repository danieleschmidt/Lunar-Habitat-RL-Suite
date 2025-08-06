"""Input validation and sanitization utilities."""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import re
from pathlib import Path

from .exceptions import ValidationError, ConfigurationError, SafetyError
from ..core.config import HabitatConfig


class ValidationRules:
    """Validation rules and constraints for different input types."""
    
    # Safety-critical parameter ranges
    SAFETY_LIMITS = {
        'o2_partial_pressure': {'min': 16.0, 'max': 50.0, 'unit': 'kPa'},
        'co2_partial_pressure': {'min': 0.0, 'max': 2.0, 'unit': 'kPa'},
        'total_pressure': {'min': 50.0, 'max': 150.0, 'unit': 'kPa'},
        'temperature': {'min': 10.0, 'max': 35.0, 'unit': '°C'},
        'humidity': {'min': 30.0, 'max': 70.0, 'unit': '%'},
        'crew_health': {'min': 0.3, 'max': 1.0, 'unit': 'fraction'},
        'battery_charge': {'min': 0.0, 'max': 100.0, 'unit': '%'},
        'water_reserves': {'min': 50.0, 'max': 10000.0, 'unit': 'liters'}
    }
    
    # Action validation ranges
    ACTION_LIMITS = {
        'o2_generation_rate': {'min': 0.0, 'max': 1.0, 'type': 'continuous'},
        'co2_scrubber_power': {'min': 0.0, 'max': 1.0, 'type': 'continuous'},
        'heating_power': {'min': 0.0, 'max': 1.0, 'type': 'continuous'},
        'solar_panel_angle': {'min': -90.0, 'max': 90.0, 'type': 'continuous'},
        'fan_speed': {'min': 0.0, 'max': 1.0, 'type': 'continuous'},
        'filter_mode': {'min': 0, 'max': 3, 'type': 'discrete'}
    }
    
    # File and path validation patterns
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    SAFE_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9._/-]+$')


def validate_numeric_range(value: Union[float, int], 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          field_name: str = "value") -> float:
    """
    Validate numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value  
        field_name: Name of field for error messages
        
    Returns:
        Validated value as float
        
    Raises:
        ValidationError: If value is out of range or invalid type
    """
    
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Field '{field_name}' must be numeric, got {type(value).__name__}",
            field_name=field_name,
            expected_type="float"
        ) from e
    
    if not np.isfinite(numeric_value):
        raise ValidationError(
            f"Field '{field_name}' must be finite (not NaN or infinite)",
            field_name=field_name
        )
    
    if min_val is not None and numeric_value < min_val:
        raise ValidationError(
            f"Field '{field_name}' value {numeric_value} is below minimum {min_val}",
            field_name=field_name
        )
    
    if max_val is not None and numeric_value > max_val:
        raise ValidationError(
            f"Field '{field_name}' value {numeric_value} exceeds maximum {max_val}",
            field_name=field_name
        )
    
    return numeric_value


def validate_array(array: Any, 
                  expected_shape: Optional[tuple] = None,
                  dtype: Optional[type] = None,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None,
                  field_name: str = "array") -> np.ndarray:
    """
    Validate numpy array meets specifications.
    
    Args:
        array: Array to validate
        expected_shape: Expected shape tuple
        dtype: Expected data type
        min_val: Minimum value for all elements
        max_val: Maximum value for all elements
        field_name: Name of field for error messages
        
    Returns:
        Validated numpy array
        
    Raises:
        ValidationError: If array doesn't meet specifications
    """
    
    try:
        validated_array = np.asarray(array)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Field '{field_name}' cannot be converted to numpy array",
            field_name=field_name
        ) from e
    
    if expected_shape is not None and validated_array.shape != expected_shape:
        raise ValidationError(
            f"Field '{field_name}' has shape {validated_array.shape}, expected {expected_shape}",
            field_name=field_name
        )
    
    if dtype is not None:
        try:
            validated_array = validated_array.astype(dtype)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Field '{field_name}' cannot be converted to type {dtype.__name__}",
                field_name=field_name
            ) from e
    
    # Check for non-finite values
    if np.issubdtype(validated_array.dtype, np.floating):
        if not np.all(np.isfinite(validated_array)):
            raise ValidationError(
                f"Field '{field_name}' contains non-finite values (NaN or inf)",
                field_name=field_name
            )
    
    # Check value ranges
    if min_val is not None and np.any(validated_array < min_val):
        raise ValidationError(
            f"Field '{field_name}' contains values below minimum {min_val}",
            field_name=field_name
        )
    
    if max_val is not None and np.any(validated_array > max_val):
        raise ValidationError(
            f"Field '{field_name}' contains values above maximum {max_val}",
            field_name=field_name
        )
    
    return validated_array


def validate_config(config: Any) -> HabitatConfig:
    """
    Comprehensive validation of habitat configuration.
    
    Args:
        config: Configuration to validate (dict or HabitatConfig)
        
    Returns:
        Validated HabitatConfig instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    
    try:
        if isinstance(config, dict):
            # Convert dict to HabitatConfig (Pydantic will validate)
            validated_config = HabitatConfig(**config)
        elif isinstance(config, HabitatConfig):
            # Re-validate existing config
            validated_config = HabitatConfig(**config.dict())
        else:
            raise ConfigurationError(
                f"Configuration must be dict or HabitatConfig, got {type(config).__name__}"
            )
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}") from e
    
    # Additional safety-critical validation
    _validate_safety_critical_config(validated_config)
    
    # Validate configuration consistency
    _validate_config_consistency(validated_config)
    
    return validated_config


def _validate_safety_critical_config(config: HabitatConfig):
    """Validate safety-critical configuration parameters."""
    
    # Atmosphere safety checks
    if config.o2_nominal < ValidationRules.SAFETY_LIMITS['o2_partial_pressure']['min']:
        raise SafetyError(
            f"O2 nominal pressure {config.o2_nominal} kPa is below safety minimum",
            safety_system="atmosphere",
            severity="critical"
        )
    
    if config.co2_limit > ValidationRules.SAFETY_LIMITS['co2_partial_pressure']['max']:
        raise SafetyError(
            f"CO2 limit {config.co2_limit} kPa exceeds safety maximum",
            safety_system="atmosphere",
            severity="critical"
        )
    
    # Temperature safety checks
    temp_min = config.temp_nominal - config.temp_tolerance
    temp_max = config.temp_nominal + config.temp_tolerance
    
    if temp_min < ValidationRules.SAFETY_LIMITS['temperature']['min']:
        raise SafetyError(
            f"Minimum temperature {temp_min}°C is below safety limit",
            safety_system="thermal",
            severity="high"
        )
    
    if temp_max > ValidationRules.SAFETY_LIMITS['temperature']['max']:
        raise SafetyError(
            f"Maximum temperature {temp_max}°C exceeds safety limit",
            safety_system="thermal", 
            severity="high"
        )
    
    # Power system safety checks
    if config.emergency_power_reserve < 24.0:
        raise SafetyError(
            f"Emergency power reserve {config.emergency_power_reserve}h is insufficient",
            safety_system="power",
            severity="high"
        )
    
    # Crew safety checks
    if config.crew.size > 12:
        raise SafetyError(
            f"Crew size {config.crew.size} exceeds maximum safe capacity",
            safety_system="life_support",
            severity="medium"
        )


def _validate_config_consistency(config: HabitatConfig):
    """Validate internal consistency of configuration."""
    
    # Check pressure consistency
    partial_pressure_sum = config.o2_nominal + config.n2_nominal + config.co2_limit
    if abs(config.pressure_nominal - partial_pressure_sum) > 5.0:
        raise ConfigurationError(
            f"Total pressure {config.pressure_nominal} kPa doesn't match sum of partial pressures {partial_pressure_sum} kPa",
            config_field="pressure_nominal"
        )
    
    # Check power system consistency
    if config.fuel_cell_capacity > config.solar_capacity * 2:
        raise ConfigurationError(
            "Fuel cell capacity should not exceed twice solar capacity for balanced system",
            config_field="fuel_cell_capacity"
        )
    
    # Check water system consistency
    min_water_per_person = 50.0  # liters per person minimum
    if config.water_storage < config.crew.size * min_water_per_person:
        raise ConfigurationError(
            f"Water storage {config.water_storage}L insufficient for {config.crew.size} crew members",
            config_field="water_storage"
        )


def validate_action(action: Any, action_space_config: Optional[Dict] = None) -> np.ndarray:
    """
    Validate action array meets specifications and safety constraints.
    
    Args:
        action: Action to validate
        action_space_config: Optional action space configuration
        
    Returns:
        Validated action array
        
    Raises:
        ValidationError: If action is invalid
        SafetyError: If action violates safety constraints
    """
    
    # Convert to numpy array
    try:
        action_array = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            "Action must be convertible to numpy array",
            field_name="action"
        ) from e
    
    # Check for non-finite values
    if not np.all(np.isfinite(action_array)):
        raise ValidationError(
            "Action contains non-finite values (NaN or inf)",
            field_name="action"
        )
    
    # Validate action ranges
    _validate_action_ranges(action_array)
    
    # Safety constraint validation
    _validate_action_safety(action_array)
    
    return action_array


def _validate_action_ranges(action: np.ndarray):
    """Validate action values are within expected ranges."""
    
    # Most actions should be in [0, 1] range
    normal_actions = action[:-1]  # Exclude solar panel angle
    if np.any(normal_actions < 0.0) or np.any(normal_actions > 1.0):
        out_of_range = np.where((normal_actions < 0.0) | (normal_actions > 1.0))[0]
        raise ValidationError(
            f"Actions at indices {out_of_range.tolist()} are outside [0,1] range",
            field_name="action"
        )
    
    # Solar panel angle should be in [-90, 90] degrees
    if len(action) > 15:  # Assuming solar angle is around index 15
        solar_angle = action[15]
        if solar_angle < -90.0 or solar_angle > 90.0:
            raise ValidationError(
                f"Solar panel angle {solar_angle} is outside [-90, 90] degree range",
                field_name="solar_panel_angle"
            )


def _validate_action_safety(action: np.ndarray):
    """Validate actions don't violate safety constraints."""
    
    # Check for simultaneous conflicting actions
    if len(action) >= 6:
        o2_gen = action[0]
        co2_scrub = action[1] 
        
        # Don't generate too much O2 while not scrubbing CO2
        if o2_gen > 0.8 and co2_scrub < 0.2:
            raise SafetyError(
                "High O2 generation with low CO2 scrubbing may cause pressure buildup",
                safety_system="atmosphere",
                severity="medium"
            )
    
    # Check for extreme heating/cooling conflicts
    if len(action) >= 20:
        heating_actions = action[10:14]  # Approximate heating zone actions
        if np.any(heating_actions > 0.95):
            raise SafetyError(
                "Extreme heating settings may cause overtemperature conditions",
                safety_system="thermal",
                severity="medium"
            )


def validate_state(state: Any, expected_size: Optional[int] = None) -> np.ndarray:
    """
    Validate environment state meets specifications.
    
    Args:
        state: State to validate
        expected_size: Expected state vector size
        
    Returns:
        Validated state array
        
    Raises:
        ValidationError: If state is invalid
        SafetyError: If state indicates safety violations
    """
    
    # Convert to numpy array
    try:
        state_array = np.asarray(state, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            "State must be convertible to numpy array",
            field_name="state"
        ) from e
    
    if expected_size is not None and len(state_array) != expected_size:
        raise ValidationError(
            f"State size {len(state_array)} doesn't match expected size {expected_size}",
            field_name="state"
        )
    
    # Check for non-finite values
    if not np.all(np.isfinite(state_array)):
        nan_indices = np.where(~np.isfinite(state_array))[0]
        raise ValidationError(
            f"State contains non-finite values at indices {nan_indices.tolist()}",
            field_name="state"
        )
    
    # Safety validation of state values
    _validate_state_safety(state_array)
    
    return state_array


def _validate_state_safety(state: np.ndarray):
    """Validate state doesn't indicate dangerous conditions."""
    
    # Approximate state indices (would be more precise with actual state structure)
    if len(state) >= 7:
        o2_pressure = state[0]  # kPa
        co2_pressure = state[1]  # kPa
        total_pressure = state[3]  # kPa
        temperature = state[5]  # °C
        
        # Critical atmosphere checks
        if o2_pressure < 16.0:
            raise SafetyError(
                f"Oxygen pressure {o2_pressure} kPa is critically low",
                safety_system="atmosphere",
                severity="critical"
            )
        
        if co2_pressure > 1.0:
            raise SafetyError(
                f"CO2 pressure {co2_pressure} kPa is dangerously high",
                safety_system="atmosphere",
                severity="critical"
            )
        
        if total_pressure < 50.0:
            raise SafetyError(
                f"Total pressure {total_pressure} kPa is critically low",
                safety_system="atmosphere",
                severity="critical"
            )
        
        if temperature < 10.0 or temperature > 35.0:
            raise SafetyError(
                f"Temperature {temperature}°C is outside safe range",
                safety_system="thermal",
                severity="high"
            )


def validate_file_path(file_path: str, 
                      must_exist: bool = False,
                      must_be_readable: bool = False,
                      must_be_writable: bool = False,
                      allowed_extensions: Optional[List[str]] = None) -> Path:
    """
    Validate file path for security and accessibility.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must already exist
        must_be_readable: Whether file must be readable
        must_be_writable: Whether file must be writable
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid or unsafe
    """
    
    try:
        path = Path(file_path).resolve()
    except (TypeError, ValueError, OSError) as e:
        raise ValidationError(f"Invalid file path: {file_path}") from e
    
    # Security checks
    if not ValidationRules.SAFE_PATH_PATTERN.match(str(path)):
        raise ValidationError(f"File path contains unsafe characters: {file_path}")
    
    # Path traversal protection
    if ".." in path.parts:
        raise ValidationError(f"Path traversal not allowed: {file_path}")
    
    # Extension validation
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValidationError(
            f"File extension {path.suffix} not in allowed extensions {allowed_extensions}"
        )
    
    # Existence checks
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    # Permission checks
    if must_be_readable and not path.is_file():
        raise ValidationError(f"Path is not a readable file: {file_path}")
    
    if must_be_readable and not os.access(path, os.R_OK):
        raise ValidationError(f"File is not readable: {file_path}")
    
    if must_be_writable and not os.access(path.parent, os.W_OK):
        raise ValidationError(f"Directory is not writable: {path.parent}")
    
    return path


def sanitize_string(text: str, 
                   max_length: int = 1000,
                   allow_multiline: bool = False,
                   allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize string input for safety.
    
    Args:
        text: Text to sanitize
        max_length: Maximum allowed length
        allow_multiline: Whether to allow newline characters
        allowed_chars: Custom regex pattern for allowed characters
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If string is invalid
    """
    
    if not isinstance(text, str):
        raise ValidationError(f"Expected string, got {type(text).__name__}")
    
    if len(text) > max_length:
        raise ValidationError(f"String length {len(text)} exceeds maximum {max_length}")
    
    # Remove null bytes and control characters
    sanitized = text.replace('\x00', '')
    
    if not allow_multiline:
        sanitized = sanitized.replace('\n', ' ').replace('\r', ' ')
    
    # Custom character validation
    if allowed_chars:
        if not re.match(allowed_chars, sanitized):
            raise ValidationError(f"String contains disallowed characters")
    
    return sanitized.strip()


class InputValidator:
    """Centralized input validation with configurable rules."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_errors = []
    
    def validate(self, data: Dict[str, Any], rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate dictionary data against rules.
        
        Args:
            data: Data to validate
            rules: Validation rules per field
            
        Returns:
            Validated data dictionary
            
        Raises:
            ValidationError: If validation fails in strict mode
        """
        
        validated_data = {}
        self.validation_errors.clear()
        
        for field_name, field_rules in rules.items():
            if field_name in data:
                try:
                    validated_data[field_name] = self._validate_field(
                        data[field_name], field_rules, field_name
                    )
                except ValidationError as e:
                    self.validation_errors.append(e)
                    if self.strict_mode:
                        raise
            elif field_rules.get('required', False):
                error = ValidationError(f"Required field '{field_name}' is missing")
                self.validation_errors.append(error)
                if self.strict_mode:
                    raise error
        
        return validated_data
    
    def _validate_field(self, value: Any, rules: Dict[str, Any], field_name: str) -> Any:
        """Validate single field against its rules."""
        
        field_type = rules.get('type')
        
        if field_type == 'numeric':
            return validate_numeric_range(
                value,
                min_val=rules.get('min'),
                max_val=rules.get('max'),
                field_name=field_name
            )
        elif field_type == 'array':
            return validate_array(
                value,
                expected_shape=rules.get('shape'),
                dtype=rules.get('dtype'),
                min_val=rules.get('min'),
                max_val=rules.get('max'),
                field_name=field_name
            )
        elif field_type == 'string':
            return sanitize_string(
                value,
                max_length=rules.get('max_length', 1000),
                allow_multiline=rules.get('multiline', False),
                allowed_chars=rules.get('pattern')
            )
        elif field_type == 'path':
            return validate_file_path(
                value,
                must_exist=rules.get('must_exist', False),
                allowed_extensions=rules.get('extensions')
            )
        else:
            return value  # No validation for unknown types
    
    def get_validation_errors(self) -> List[ValidationError]:
        """Get list of validation errors from last validation."""
        return self.validation_errors.copy()