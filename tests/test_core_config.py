"""Tests for core configuration classes."""

import pytest
import numpy as np
from pydantic import ValidationError

from lunar_habitat_rl.core.config import HabitatConfig, CrewConfig, PhysicsConfig, ScenarioConfig
from lunar_habitat_rl.utils.exceptions import ConfigurationError


class TestCrewConfig:
    """Test crew configuration."""
    
    def test_default_crew_config(self):
        """Test default crew configuration."""
        crew_config = CrewConfig()
        
        assert crew_config.size == 4
        assert 0.0 <= crew_config.health_variance <= 1.0
        assert 0.0 <= crew_config.stress_sensitivity <= 1.0
        assert 0.0 <= crew_config.productivity_base <= 1.0
        assert crew_config.metabolic_rate >= 50.0
    
    def test_crew_config_validation(self):
        """Test crew configuration validation."""
        # Valid configurations
        valid_config = CrewConfig(size=6, health_variance=0.1, stress_sensitivity=0.5)
        assert valid_config.size == 6
        
        # Invalid crew size
        with pytest.raises(ValidationError):
            CrewConfig(size=0)
        
        with pytest.raises(ValidationError):
            CrewConfig(size=15)
        
        # Invalid variance values
        with pytest.raises(ValidationError):
            CrewConfig(health_variance=-0.1)
        
        with pytest.raises(ValidationError):
            CrewConfig(health_variance=1.5)


class TestPhysicsConfig:
    """Test physics configuration."""
    
    def test_default_physics_config(self):
        """Test default physics configuration."""
        physics_config = PhysicsConfig()
        
        assert physics_config.thermal_enabled is True
        assert physics_config.fluid_enabled is True
        assert physics_config.chemistry_enabled is True
        assert physics_config.thermal_timestep > 0
        assert physics_config.thermal_solver in ["finite_difference", "finite_element", "spectral"]
    
    def test_physics_config_validation(self):
        """Test physics configuration validation."""
        # Valid configuration
        valid_config = PhysicsConfig(
            thermal_solver="finite_element",
            thermal_timestep=30.0
        )
        assert valid_config.thermal_solver == "finite_element"
        
        # Invalid solver
        with pytest.raises(ValidationError):
            PhysicsConfig(thermal_solver="invalid_solver")
        
        # Invalid timestep
        with pytest.raises(ValidationError):
            PhysicsConfig(thermal_timestep=0.0)


class TestScenarioConfig:
    """Test scenario configuration."""
    
    def test_default_scenario_config(self):
        """Test default scenario configuration.""" 
        scenario_config = ScenarioConfig(name="test_scenario")
        
        assert scenario_config.name == "test_scenario"
        assert scenario_config.duration_days >= 1
        assert scenario_config.duration_days <= 1000
        assert scenario_config.difficulty in ["easy", "nominal", "hard", "extreme"]
        assert isinstance(scenario_config.events, list)
    
    def test_scenario_config_validation(self):
        """Test scenario configuration validation."""
        # Valid configuration
        valid_config = ScenarioConfig(
            name="test",
            duration_days=60,
            difficulty="hard"
        )
        assert valid_config.duration_days == 60
        assert valid_config.difficulty == "hard"
        
        # Invalid duration
        with pytest.raises(ValidationError):
            ScenarioConfig(name="test", duration_days=0)
        
        with pytest.raises(ValidationError):
            ScenarioConfig(name="test", duration_days=2000)
        
        # Invalid difficulty
        with pytest.raises(ValidationError):
            ScenarioConfig(name="test", difficulty="impossible")


class TestHabitatConfig:
    """Test main habitat configuration."""
    
    def test_default_habitat_config(self):
        """Test default habitat configuration."""
        config = HabitatConfig()
        
        # Basic parameters
        assert config.volume > 0
        assert 50.0 <= config.pressure_nominal <= 120.0
        
        # Atmosphere composition
        assert 16.0 <= config.o2_nominal <= 30.0
        assert 0.1 <= config.co2_limit <= 1.0
        assert 60.0 <= config.n2_nominal <= 85.0
        
        # Thermal parameters
        assert 18.0 <= config.temp_nominal <= 28.0
        assert 1.0 <= config.temp_tolerance <= 5.0
        
        # Power system
        assert config.solar_capacity > 0
        assert config.battery_capacity > 0
        assert config.fuel_cell_capacity >= 0
        
        # Sub-configurations
        assert isinstance(config.crew, CrewConfig)
        assert isinstance(config.physics, PhysicsConfig)
        assert isinstance(config.scenario, ScenarioConfig)
    
    def test_habitat_config_validation(self):
        """Test habitat configuration validation."""
        # Valid configuration
        valid_config = HabitatConfig(
            volume=150.0,
            o2_nominal=20.0,
            n2_nominal=80.0,
            pressure_nominal=100.0
        )
        assert valid_config.volume == 150.0
        
        # Test pressure consistency validation
        # This should pass (close to sum of partial pressures)
        consistent_config = HabitatConfig(
            o2_nominal=21.0,
            n2_nominal=79.0,
            pressure_nominal=100.4,  # Close to 21 + 79 + 0.4 (default CO2)
            co2_limit=0.4
        )
        assert abs(consistent_config.pressure_nominal - 100.4) < 0.1
    
    def test_habitat_config_from_preset(self):
        """Test loading configuration from presets."""
        # NASA reference preset
        nasa_config = HabitatConfig.from_preset("nasa_reference")
        assert nasa_config.name == "nasa_reference_habitat"
        
        # Apollo derived preset
        apollo_config = HabitatConfig.from_preset("apollo_derived")
        assert apollo_config.volume == 150.0
        assert apollo_config.pressure_nominal == 68.9  # 10 psi
        
        # Mars analog preset
        mars_config = HabitatConfig.from_preset("mars_analog")
        assert mars_config.volume == 300.0
        
        # Invalid preset
        with pytest.raises(ValueError):
            HabitatConfig.from_preset("invalid_preset")
    
    def test_habitat_config_serialization(self):
        """Test configuration serialization."""
        config = HabitatConfig(volume=200.0)
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['volume'] == 200.0
        assert 'crew' in config_dict
        assert 'physics' in config_dict
        assert 'scenario' in config_dict
        
        # Test round-trip
        new_config = HabitatConfig(**config_dict)
        assert new_config.volume == config.volume
        assert new_config.crew.size == config.crew.size
    
    def test_habitat_config_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimum viable configuration
        min_config = HabitatConfig(
            volume=10.0,  # Minimum volume
            pressure_nominal=50.0,  # Minimum pressure
            o2_nominal=16.0,  # Minimum O2
            n2_nominal=33.0,  # Adjusted to maintain pressure balance
            temp_nominal=18.0,  # Minimum temperature
            temp_tolerance=1.0  # Minimum tolerance
        )
        assert min_config.volume == 10.0
        
        # Maximum configuration
        max_config = HabitatConfig(
            volume=1000.0,
            pressure_nominal=120.0,
            o2_nominal=30.0,
            n2_nominal=85.0,
            temp_nominal=28.0,
            temp_tolerance=5.0
        )
        assert max_config.volume == 1000.0
    
    @pytest.mark.parametrize("preset_name", ["nasa_reference", "apollo_derived", "mars_analog"])
    def test_all_presets_load(self, preset_name):
        """Test that all presets load successfully."""
        config = HabitatConfig.from_preset(preset_name)
        assert config is not None
        assert isinstance(config, HabitatConfig)
        
        # Validate loaded configuration
        assert config.volume > 0
        assert config.pressure_nominal > 0
        assert config.o2_nominal > 0
    
    def test_config_modification(self):
        """Test configuration modification after creation."""
        config = HabitatConfig()
        original_volume = config.volume
        
        # Modify configuration
        config.volume = 250.0
        assert config.volume == 250.0
        assert config.volume != original_volume
        
        # Modify sub-configurations
        original_crew_size = config.crew.size
        config.crew.size = 6
        assert config.crew.size == 6
        assert config.crew.size != original_crew_size