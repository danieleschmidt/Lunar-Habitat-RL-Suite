"""Test suite for the Lunar Habitat RL Suite."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Test configuration
TEST_CONFIG = {
    'default_timeout': 30.0,
    'physics_simulation_tolerance': 1e-3,
    'environment_step_tolerance': 1e-6,
    'random_seed': 42
}

# Test fixtures directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_DIR.mkdir(exist_ok=True)


def create_temp_directory() -> Path:
    """Create temporary directory for tests."""
    return Path(tempfile.mkdtemp(prefix="lunar_habitat_test_"))


def cleanup_temp_directory(temp_dir: Path):
    """Clean up temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def assert_array_close(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert arrays are close with better error messages."""
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        print(f"Array comparison failed:")
        print(f"Expected shape: {expected.shape}, Actual shape: {actual.shape}")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Difference: {actual - expected}")
        raise e


def assert_dict_close(actual: Dict[str, float], expected: Dict[str, float], rtol: float = 1e-5):
    """Assert dictionary values are close."""
    assert set(actual.keys()) == set(expected.keys()), f"Keys don't match: {actual.keys()} vs {expected.keys()}"
    
    for key in expected:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=rtol,
            err_msg=f"Value mismatch for key '{key}': {actual[key]} vs {expected[key]}"
        )


class MockPhysicsSimulator:
    """Mock physics simulator for testing."""
    
    def __init__(self):
        self.step_count = 0
        self.current_temperature = 22.0
        
    def step(self, dt: float, **kwargs) -> Dict[str, Any]:
        """Mock simulation step."""
        self.step_count += 1
        
        # Simple temperature evolution
        self.current_temperature += np.random.normal(0, 0.1)
        
        return {
            'zone_temperatures': [self.current_temperature] * 4,
            'radiator_temperatures': [15.0, 16.0],
            'heat_pump_cop': 3.0,
            'total_heat_loss': 1000.0,
            'average_temperature': self.current_temperature,
            'temperature_variance': 0.1
        }