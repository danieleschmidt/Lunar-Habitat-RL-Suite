"""
Sim-to-Real Domain Randomization for Lunar Habitat RL

This module provides comprehensive domain randomization and sim-to-real transfer
capabilities to train robust agents that can transfer from simulation to real
lunar habitat deployment.

Features:
- Physics parameter randomization (thermal, atmospheric, mechanical properties)
- Sensor noise and actuator delay modeling
- Environmental condition randomization
- System degradation and failure modeling
- Distribution shift adaptation
- Reality gap analysis and bridging
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import json
import random
from collections import defaultdict
import copy

from .core.state import HabitatState
from .core.config import HabitatConfig
from .utils.logging import get_logger
from .utils.exceptions import SimToRealError, DomainRandomizationError
from .utils.validation import validate_randomization_config

logger = get_logger(__name__)


@dataclass
class RandomizationRange:
    """Range specification for parameter randomization."""
    min_value: float
    max_value: float
    distribution: str = "uniform"  # uniform, normal, log_uniform
    mean: Optional[float] = None
    std: Optional[float] = None
    
    def sample(self, rng: Optional[np.random.Generator] = None) -> float:
        """Sample a value from the specified range and distribution."""
        if rng is None:
            rng = np.random.default_rng()
        
        if self.distribution == "uniform":
            return rng.uniform(self.min_value, self.max_value)
        elif self.distribution == "normal":
            mean = self.mean if self.mean is not None else (self.min_value + self.max_value) / 2
            std = self.std if self.std is not None else (self.max_value - self.min_value) / 6
            value = rng.normal(mean, std)
            return np.clip(value, self.min_value, self.max_value)
        elif self.distribution == "log_uniform":
            log_min = np.log(max(self.min_value, 1e-8))
            log_max = np.log(max(self.max_value, 1e-8))
            log_value = rng.uniform(log_min, log_max)
            return np.exp(log_value)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization parameters."""
    # Physics parameter randomization
    thermal_conductivity_range: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2, "uniform")
    )
    thermal_capacity_range: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.9, 1.1, "uniform")
    )
    air_density_range: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.95, 1.05, "uniform")
    )
    
    # Environmental randomization
    external_temp_variance: float = 10.0  # ±10°C variance
    solar_flux_variation: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.8, 1.2, "uniform")
    )
    dust_accumulation_rate: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.5, 2.0, "log_uniform")
    )
    
    # Sensor noise modeling
    temperature_sensor_noise: float = 0.5  # ±0.5°C
    pressure_sensor_noise: float = 0.1  # ±0.1 kPa
    flow_sensor_noise: float = 0.02  # ±2%
    power_sensor_noise: float = 0.01  # ±1%
    
    # Actuator modeling
    actuator_delay_range: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.1, 0.5, "uniform")  # 0.1-0.5 seconds
    )
    actuator_noise_range: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.01, 0.05, "uniform")  # 1-5% noise
    )
    actuator_deadband: float = 0.02  # 2% deadband
    
    # System degradation
    equipment_degradation_rate: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.999, 1.0, "uniform")  # 0.1% per day max
    )
    filter_clogging_rate: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.0001, 0.001, "log_uniform")
    )
    
    # Communication and networking
    communication_latency: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.01, 0.1, "uniform")  # 10-100ms
    )
    packet_loss_rate: float = 0.001  # 0.1% packet loss
    
    # Calibration drift
    sensor_calibration_drift: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(-0.02, 0.02, "normal")  # ±2% drift
    )
    
    # Enable/disable randomization components
    randomize_physics: bool = True
    randomize_sensors: bool = True
    randomize_actuators: bool = True
    randomize_environment: bool = True
    randomize_degradation: bool = True


class ParameterRandomizer:
    """Randomizes simulation parameters for domain randomization."""
    
    def __init__(self, config: DomainRandomizationConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.base_parameters = {}
        self.current_parameters = {}
        
        logger.info("Initialized parameter randomizer")
    
    def set_base_parameters(self, parameters: Dict[str, Any]):
        """Set base parameter values to randomize from."""
        self.base_parameters = copy.deepcopy(parameters)
        self.current_parameters = copy.deepcopy(parameters)
    
    def randomize_parameters(self) -> Dict[str, Any]:
        """Generate randomized parameters."""
        randomized = copy.deepcopy(self.base_parameters)
        
        if self.config.randomize_physics:
            randomized = self._randomize_physics_parameters(randomized)
        
        if self.config.randomize_environment:
            randomized = self._randomize_environmental_parameters(randomized)
        
        if self.config.randomize_degradation:
            randomized = self._randomize_degradation_parameters(randomized)
        
        self.current_parameters = randomized
        return randomized
    
    def _randomize_physics_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Randomize physics-related parameters."""
        
        # Thermal properties
        if "thermal_conductivity" in params:
            multiplier = self.config.thermal_conductivity_range.sample(self.rng)
            params["thermal_conductivity"] *= multiplier
        
        if "thermal_capacity" in params:
            multiplier = self.config.thermal_capacity_range.sample(self.rng)
            params["thermal_capacity"] *= multiplier
        
        # Atmospheric properties
        if "air_density" in params:
            multiplier = self.config.air_density_range.sample(self.rng)
            params["air_density"] *= multiplier
        
        return params
    
    def _randomize_environmental_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Randomize environmental conditions."""
        
        # Solar flux variation
        if "solar_flux" in params:
            multiplier = self.config.solar_flux_variation.sample(self.rng)
            params["solar_flux"] *= multiplier
        
        # Dust accumulation
        if "dust_rate" in params:
            multiplier = self.config.dust_accumulation_rate.sample(self.rng)
            params["dust_rate"] *= multiplier
        
        # External temperature variance
        if "external_temperature" in params:
            variance = self.rng.normal(0, self.config.external_temp_variance)
            params["external_temperature"] += variance
        
        return params
    
    def _randomize_degradation_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Randomize system degradation parameters."""
        
        # Equipment degradation
        if "equipment_efficiency" in params:
            degradation_rate = self.config.equipment_degradation_rate.sample(self.rng)
            params["equipment_efficiency"] *= degradation_rate
        
        # Filter clogging
        if "filter_efficiency" in params:
            clogging_rate = self.config.filter_clogging_rate.sample(self.rng)
            params["filter_efficiency"] *= (1.0 - clogging_rate)
        
        return params


class SensorNoiseModel:
    """Models realistic sensor noise and calibration drift."""
    
    def __init__(self, config: DomainRandomizationConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Sensor calibration drift (persistent per episode)
        self.calibration_drift = {}
        self.sensor_histories = defaultdict(list)
        
        logger.info("Initialized sensor noise model")
    
    def reset_calibration_drift(self):
        """Reset calibration drift for new episode."""
        self.calibration_drift = {
            "temperature": self.config.sensor_calibration_drift.sample(self.rng),
            "pressure": self.config.sensor_calibration_drift.sample(self.rng),
            "flow": self.config.sensor_calibration_drift.sample(self.rng),
            "power": self.config.sensor_calibration_drift.sample(self.rng)
        }
    
    def add_sensor_noise(self, sensor_readings: Dict[str, float]) -> Dict[str, float]:
        """Add realistic noise to sensor readings."""
        noisy_readings = {}
        
        for sensor_type, true_value in sensor_readings.items():
            # Base noise
            if sensor_type == "temperature":
                noise_std = self.config.temperature_sensor_noise
            elif sensor_type == "pressure":
                noise_std = self.config.pressure_sensor_noise
            elif sensor_type == "flow":
                noise_std = abs(true_value) * self.config.flow_sensor_noise
            elif sensor_type == "power":
                noise_std = abs(true_value) * self.config.power_sensor_noise
            else:
                noise_std = 0.01 * abs(true_value)  # Default 1% noise
            
            # Add noise
            noise = self.rng.normal(0, noise_std)
            
            # Add calibration drift
            drift = self.calibration_drift.get(sensor_type, 0.0)
            
            # Combine true value, noise, and drift
            noisy_value = true_value + noise + (true_value * drift)
            
            # Add quantization noise (ADC effects)
            if sensor_type in ["temperature", "pressure"]:
                quantization_step = noise_std * 0.1  # 10% of noise std
                noisy_value = np.round(noisy_value / quantization_step) * quantization_step
            
            noisy_readings[sensor_type] = noisy_value
            
            # Store history for potential filtering/processing
            self.sensor_histories[sensor_type].append(noisy_value)
            if len(self.sensor_histories[sensor_type]) > 100:  # Keep last 100 readings
                self.sensor_histories[sensor_type].pop(0)
        
        return noisy_readings
    
    def get_sensor_health(self, sensor_type: str) -> float:
        """Estimate sensor health based on reading history."""
        if sensor_type not in self.sensor_histories:
            return 1.0
        
        history = self.sensor_histories[sensor_type]
        if len(history) < 10:
            return 1.0
        
        # Simple health metric based on variance
        variance = np.var(history[-10:])
        expected_variance = 0.01  # Expected variance for healthy sensor
        health = max(0.1, min(1.0, expected_variance / (variance + 1e-8)))
        
        return health


class ActuatorModel:
    """Models realistic actuator dynamics, delays, and limitations."""
    
    def __init__(self, config: DomainRandomizationConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Actuator state tracking
        self.actuator_states = {}
        self.command_histories = defaultdict(list)
        self.delay_buffers = defaultdict(deque)
        
        # Per-actuator characteristics (sampled once per episode)
        self.actuator_delays = {}
        self.actuator_noise_levels = {}
        
        logger.info("Initialized actuator model")
    
    def reset_actuator_characteristics(self, actuator_ids: List[str]):
        """Reset actuator characteristics for new episode."""
        for actuator_id in actuator_ids:
            self.actuator_delays[actuator_id] = self.config.actuator_delay_range.sample(self.rng)
            self.actuator_noise_levels[actuator_id] = self.config.actuator_noise_range.sample(self.rng)
            
            # Initialize state
            self.actuator_states[actuator_id] = 0.0
            self.delay_buffers[actuator_id] = deque(maxlen=10)  # Buffer for delays
    
    def apply_actuator_dynamics(
        self, 
        commands: Dict[str, float], 
        dt: float = 1.0
    ) -> Dict[str, float]:
        """Apply realistic actuator dynamics to commands."""
        
        actual_outputs = {}
        
        for actuator_id, command in commands.items():
            if actuator_id not in self.actuator_states:
                # Initialize if new actuator
                self.actuator_delays[actuator_id] = self.config.actuator_delay_range.sample(self.rng)
                self.actuator_noise_levels[actuator_id] = self.config.actuator_noise_range.sample(self.rng)
                self.actuator_states[actuator_id] = command
                self.delay_buffers[actuator_id] = deque(maxlen=10)
            
            # Apply deadband
            current_state = self.actuator_states[actuator_id]
            command_change = abs(command - current_state)
            if command_change < self.config.actuator_deadband:
                command = current_state  # No change if within deadband
            
            # Add delay by buffering commands
            delay_steps = max(1, int(self.actuator_delays[actuator_id] / dt))
            self.delay_buffers[actuator_id].append(command)
            
            if len(self.delay_buffers[actuator_id]) >= delay_steps:
                delayed_command = self.delay_buffers[actuator_id].popleft()
            else:
                delayed_command = current_state  # Use current state until buffer fills
            
            # Apply first-order dynamics (low-pass filter)
            time_constant = 0.5  # 0.5 second time constant
            alpha = dt / (time_constant + dt)
            filtered_command = alpha * delayed_command + (1 - alpha) * current_state
            
            # Add actuator noise
            noise_level = self.actuator_noise_levels[actuator_id]
            noise = self.rng.normal(0, noise_level * abs(filtered_command))
            noisy_output = filtered_command + noise
            
            # Apply physical limits
            noisy_output = np.clip(noisy_output, -1.0, 1.0)
            
            # Update state
            self.actuator_states[actuator_id] = noisy_output
            actual_outputs[actuator_id] = noisy_output
            
            # Store command history
            self.command_histories[actuator_id].append(command)
            if len(self.command_histories[actuator_id]) > 100:
                self.command_histories[actuator_id].pop(0)
        
        return actual_outputs
    
    def get_actuator_health(self, actuator_id: str) -> float:
        """Estimate actuator health based on command tracking performance."""
        if actuator_id not in self.command_histories or len(self.command_histories[actuator_id]) < 10:
            return 1.0
        
        commands = np.array(self.command_histories[actuator_id][-10:])
        actual_states = [self.actuator_states[actuator_id]] * len(commands)  # Simplified
        
        # Health based on tracking error
        tracking_error = np.mean(np.abs(commands - actual_states))
        health = max(0.1, 1.0 - tracking_error)
        
        return health


class DomainRandomizationWrapper:
    """Wrapper that applies domain randomization to environments."""
    
    def __init__(
        self, 
        base_env: Any, 
        config: DomainRandomizationConfig = None,
        seed: Optional[int] = None
    ):
        self.base_env = base_env
        self.config = config or DomainRandomizationConfig()
        self.seed = seed
        
        # Initialize components
        self.parameter_randomizer = ParameterRandomizer(self.config, seed)
        self.sensor_noise_model = SensorNoiseModel(self.config, seed)
        self.actuator_model = ActuatorModel(self.config, seed)
        
        # Track randomization statistics
        self.randomization_stats = defaultdict(list)
        self.episode_count = 0
        
        logger.info("Initialized domain randomization wrapper")
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with randomized parameters."""
        
        # Extract base environment parameters
        if hasattr(self.base_env, 'get_parameters'):
            base_params = self.base_env.get_parameters()
        else:
            # Default parameters if not available
            base_params = {
                "thermal_conductivity": 1.0,
                "thermal_capacity": 1.0,
                "air_density": 1.2,
                "solar_flux": 1361.0,
                "dust_rate": 0.01,
                "equipment_efficiency": 1.0,
                "filter_efficiency": 0.95
            }
        
        # Set base parameters and randomize
        self.parameter_randomizer.set_base_parameters(base_params)
        randomized_params = self.parameter_randomizer.randomize_parameters()
        
        # Apply randomized parameters to environment
        if hasattr(self.base_env, 'set_parameters'):
            self.base_env.set_parameters(randomized_params)
        
        # Reset sensor calibration drift
        self.sensor_noise_model.reset_calibration_drift()
        
        # Reset actuator characteristics
        actuator_ids = ["heater", "fan", "pump", "valve"]  # Default actuator types
        if hasattr(self.base_env, 'get_actuator_ids'):
            actuator_ids = self.base_env.get_actuator_ids()
        self.actuator_model.reset_actuator_characteristics(actuator_ids)
        
        # Reset base environment
        obs, info = self.base_env.reset(**kwargs)
        
        # Add noise to initial observation
        if hasattr(self, '_extract_sensor_readings'):
            sensor_readings = self._extract_sensor_readings(obs)
            noisy_readings = self.sensor_noise_model.add_sensor_noise(sensor_readings)
            obs = self._reconstruct_observation(obs, noisy_readings)
        
        # Add randomization info
        info["domain_randomization"] = {
            "parameters": randomized_params,
            "episode": self.episode_count
        }
        
        self.episode_count += 1
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute step with actuator dynamics and sensor noise."""
        
        # Convert action to actuator commands
        actuator_commands = self._action_to_actuator_commands(action)
        
        # Apply actuator dynamics
        actual_commands = self.actuator_model.apply_actuator_dynamics(actuator_commands)
        
        # Convert back to action format
        modified_action = self._actuator_commands_to_action(actual_commands, action.shape)
        
        # Execute in base environment
        obs, reward, terminated, truncated, info = self.base_env.step(modified_action)
        
        # Add sensor noise to observations
        if hasattr(self, '_extract_sensor_readings'):
            sensor_readings = self._extract_sensor_readings(obs)
            noisy_readings = self.sensor_noise_model.add_sensor_noise(sensor_readings)
            obs = self._reconstruct_observation(obs, noisy_readings)
        
        # Add system health information
        info["system_health"] = {
            "sensors": {
                sensor_type: self.sensor_noise_model.get_sensor_health(sensor_type)
                for sensor_type in ["temperature", "pressure", "flow", "power"]
            },
            "actuators": {
                actuator_id: self.actuator_model.get_actuator_health(actuator_id)
                for actuator_id in actuator_commands.keys()
            }
        }
        
        return obs, reward, terminated, truncated, info
    
    def _extract_sensor_readings(self, obs: np.ndarray) -> Dict[str, float]:
        """Extract sensor readings from observation vector."""
        # This is environment-specific and would need to be customized
        # For now, assume standard habitat observation structure
        return {
            "temperature": obs[0] if len(obs) > 0 else 22.0,
            "pressure": obs[1] if len(obs) > 1 else 101.3,
            "flow": obs[2] if len(obs) > 2 else 1.0,
            "power": obs[3] if len(obs) > 3 else 1000.0
        }
    
    def _reconstruct_observation(self, original_obs: np.ndarray, noisy_readings: Dict[str, float]) -> np.ndarray:
        """Reconstruct observation vector with noisy sensor readings."""
        # This is environment-specific and would need to be customized
        modified_obs = original_obs.copy()
        
        if len(modified_obs) > 0:
            modified_obs[0] = noisy_readings.get("temperature", modified_obs[0])
        if len(modified_obs) > 1:
            modified_obs[1] = noisy_readings.get("pressure", modified_obs[1])
        if len(modified_obs) > 2:
            modified_obs[2] = noisy_readings.get("flow", modified_obs[2])
        if len(modified_obs) > 3:
            modified_obs[3] = noisy_readings.get("power", modified_obs[3])
        
        return modified_obs
    
    def _action_to_actuator_commands(self, action: np.ndarray) -> Dict[str, float]:
        """Convert action vector to actuator command dictionary."""
        # This is environment-specific
        commands = {}
        
        if len(action) > 0:
            commands["heater"] = float(action[0])
        if len(action) > 1:
            commands["fan"] = float(action[1])
        if len(action) > 2:
            commands["pump"] = float(action[2])
        if len(action) > 3:
            commands["valve"] = float(action[3])
        
        return commands
    
    def _actuator_commands_to_action(self, commands: Dict[str, float], original_shape: Tuple) -> np.ndarray:
        """Convert actuator commands back to action vector."""
        action = np.zeros(original_shape)
        
        if "heater" in commands and len(action) > 0:
            action[0] = commands["heater"]
        if "fan" in commands and len(action) > 1:
            action[1] = commands["fan"]
        if "pump" in commands and len(action) > 2:
            action[2] = commands["pump"]
        if "valve" in commands and len(action) > 3:
            action[3] = commands["valve"]
        
        return action
    
    def get_randomization_stats(self) -> Dict[str, Any]:
        """Get statistics about domain randomization."""
        return {
            "episodes_completed": self.episode_count,
            "parameter_ranges": {
                "thermal_conductivity": self.config.thermal_conductivity_range.__dict__,
                "solar_flux_variation": self.config.solar_flux_variation.__dict__,
                "actuator_delay": self.config.actuator_delay_range.__dict__
            },
            "current_parameters": self.parameter_randomizer.current_parameters
        }


class RealityGapAnalyzer:
    """Analyzes and quantifies the reality gap between simulation and real data."""
    
    def __init__(self):
        self.sim_data = []
        self.real_data = []
        self.gap_metrics = {}
        
        logger.info("Initialized reality gap analyzer")
    
    def add_simulation_data(self, data: Dict[str, Any]):
        """Add simulation trajectory data."""
        self.sim_data.append(data)
    
    def add_real_data(self, data: Dict[str, Any]):
        """Add real system trajectory data."""
        self.real_data.append(data)
    
    def analyze_reality_gap(self) -> Dict[str, float]:
        """Analyze the reality gap between simulation and real data."""
        
        if not self.sim_data or not self.real_data:
            logger.warning("Insufficient data for reality gap analysis")
            return {}
        
        gap_metrics = {}
        
        # Compare state distributions
        gap_metrics["state_distribution_gap"] = self._compare_state_distributions()
        
        # Compare action effectiveness
        gap_metrics["action_effectiveness_gap"] = self._compare_action_effectiveness()
        
        # Compare system dynamics
        gap_metrics["dynamics_gap"] = self._compare_dynamics()
        
        # Compare sensor characteristics
        gap_metrics["sensor_gap"] = self._compare_sensor_characteristics()
        
        self.gap_metrics = gap_metrics
        
        logger.info(f"Reality gap analysis complete. Overall gap: {np.mean(list(gap_metrics.values())):.3f}")
        
        return gap_metrics
    
    def _compare_state_distributions(self) -> float:
        """Compare state value distributions between sim and real data."""
        # Simplified implementation - would use statistical tests in practice
        return 0.1  # Placeholder
    
    def _compare_action_effectiveness(self) -> float:
        """Compare how effective actions are in sim vs real."""
        # Simplified implementation
        return 0.15  # Placeholder
    
    def _compare_dynamics(self) -> float:
        """Compare system dynamics between sim and real."""
        # Simplified implementation
        return 0.08  # Placeholder
    
    def _compare_sensor_characteristics(self) -> float:
        """Compare sensor noise and bias between sim and real."""
        # Simplified implementation
        return 0.12  # Placeholder
    
    def suggest_randomization_improvements(self) -> Dict[str, Any]:
        """Suggest improvements to domain randomization based on gap analysis."""
        
        suggestions = {
            "increase_sensor_noise": self.gap_metrics.get("sensor_gap", 0) > 0.1,
            "expand_parameter_ranges": self.gap_metrics.get("dynamics_gap", 0) > 0.1,
            "add_systematic_biases": self.gap_metrics.get("state_distribution_gap", 0) > 0.15,
            "improve_actuator_modeling": self.gap_metrics.get("action_effectiveness_gap", 0) > 0.1
        }
        
        return suggestions


def create_robust_training_env(
    base_env: Any,
    randomization_level: str = "medium",
    seed: Optional[int] = None
) -> DomainRandomizationWrapper:
    """
    Create a domain-randomized environment for robust training.
    
    Args:
        base_env: Base environment to wrap
        randomization_level: Level of randomization (low, medium, high)
        seed: Random seed for reproducibility
    
    Returns:
        Domain randomized environment wrapper
    """
    
    # Configure randomization based on level
    if randomization_level == "low":
        config = DomainRandomizationConfig(
            thermal_conductivity_range=RandomizationRange(0.95, 1.05),
            actuator_delay_range=RandomizationRange(0.1, 0.2),
            temperature_sensor_noise=0.1,
            randomize_degradation=False
        )
    elif randomization_level == "medium":
        config = DomainRandomizationConfig()  # Use defaults
    elif randomization_level == "high":
        config = DomainRandomizationConfig(
            thermal_conductivity_range=RandomizationRange(0.7, 1.3),
            solar_flux_variation=RandomizationRange(0.6, 1.4),
            actuator_delay_range=RandomizationRange(0.1, 1.0),
            temperature_sensor_noise=1.0,
            pressure_sensor_noise=0.2,
            randomize_degradation=True
        )
    else:
        raise ValueError(f"Unknown randomization level: {randomization_level}")
    
    return DomainRandomizationWrapper(base_env, config, seed)


# Export main classes
__all__ = [
    "RandomizationRange",
    "DomainRandomizationConfig", 
    "ParameterRandomizer",
    "SensorNoiseModel",
    "ActuatorModel",
    "DomainRandomizationWrapper",
    "RealityGapAnalyzer",
    "create_robust_training_env"
]