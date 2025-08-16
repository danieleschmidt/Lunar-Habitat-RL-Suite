"""Lightweight habitat environment without heavy dependencies - Generation 1"""

import random
import math
from typing import Dict, List, Tuple, Optional, Any

from ..core.lightweight_config import HabitatConfig
from ..core.lightweight_state import HabitatState, ActionSpace


class MockObservationSpace:
    """Mock observation space for lightweight implementation."""
    
    def __init__(self, dims: int):
        self.shape = (dims,)
        
    def sample(self) -> List[float]:
        return [random.random() for _ in range(self.shape[0])]


class MockActionSpace:
    """Mock action space for lightweight implementation."""
    
    def __init__(self, dims: int):
        self.shape = (dims,)
        
    def sample(self) -> List[float]:
        return [random.random() for _ in range(self.shape[0])]


class LunarHabitatEnv:
    """Lightweight Lunar Habitat RL Environment - Generation 1 Implementation.
    
    This is a simplified version that works without gymnasium, numpy, or other heavy dependencies.
    It provides basic functionality for testing and development.
    """
    
    def __init__(self, config: Optional[HabitatConfig] = None, crew_size: int = 4):
        """Initialize the habitat environment.
        
        Args:
            config: Habitat configuration. If None, uses default configuration.
            crew_size: Number of crew members (1-6).
        """
        self.config = config or HabitatConfig()
        self.crew_size = min(max(crew_size, 1), 6)
        
        # Initialize state and action spaces
        self.state = HabitatState(max_crew=self.crew_size)
        self.action_parser = ActionSpace()
        
        # Set up spaces
        state_info = self.state.get_observation_space_info()
        action_info = self.action_parser.get_action_space_info()
        
        self.observation_space = MockObservationSpace(state_info['total_dims'])
        self.action_space = MockActionSpace(action_info['total_dims'])
        
        # Environment state
        self.current_step = 0
        self.max_steps = 1000
        self.episode_reward = 0.0
        
        # Physics simulation step size
        self.dt = 60.0  # 1 minute timesteps
        
    def reset(self, seed: Optional[int] = None) -> Tuple[List[float], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            
        # Reset state to nominal conditions
        self.state = HabitatState(max_crew=self.crew_size)
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Add some initial variation
        self.state.atmosphere.temperature += random.uniform(-1.0, 1.0)
        self.state.power.battery_charge += random.uniform(-5.0, 5.0)
        
        observation = self.state.to_array()
        info = {
            'step': self.current_step,
            'total_reward': self.episode_reward,
            'crew_size': self.crew_size,
            'status': 'nominal'
        }
        
        return observation, info
    
    def step(self, action: List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Action to execute (list of floats).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Parse action
        try:
            parsed_action = self.action_parser.parse_action(action)
        except ValueError as e:
            # Invalid action - return penalty
            reward = -10.0
            terminated = True
            truncated = False
            info = {'error': str(e), 'step': self.current_step}
            return self.state.to_array(), reward, terminated, truncated, info
        
        # Apply actions and simulate physics
        reward = self._simulate_step(parsed_action)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Update episode reward
        self.episode_reward += reward
        
        # Prepare observation and info
        observation = self.state.to_array()
        info = {
            'step': self.current_step,
            'total_reward': self.episode_reward,
            'actions_taken': parsed_action,
            'status': self._get_status()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _simulate_step(self, parsed_action: Dict[str, Any]) -> float:
        """Simulate one timestep of habitat physics and return reward.
        
        Args:
            parsed_action: Parsed action dictionary.
            
        Returns:
            Reward for this step.
        """
        reward = 0.0
        
        # Simulate atmosphere dynamics
        reward += self._simulate_atmosphere(parsed_action['life_support'])
        
        # Simulate power systems
        reward += self._simulate_power(parsed_action['power'])
        
        # Simulate thermal systems
        reward += self._simulate_thermal(parsed_action['thermal'])
        
        # Simulate water systems
        reward += self._simulate_water(parsed_action['water'])
        
        # Simulate crew health and productivity
        reward += self._simulate_crew()
        
        # Add small time penalty to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def _simulate_atmosphere(self, life_support_actions: Dict[str, Any]) -> float:
        """Simulate atmospheric systems and return reward component."""
        reward = 0.0
        
        # O2 generation
        o2_rate = life_support_actions['o2_generation_rate']
        o2_generated = o2_rate * 0.1 * self.dt / 3600  # kPa per hour
        self.state.atmosphere.o2_partial_pressure += o2_generated
        
        # CO2 scrubbing
        co2_scrub_power = life_support_actions['co2_scrubber_power']
        co2_removed = co2_scrub_power * 0.05 * self.dt / 3600
        self.state.atmosphere.co2_partial_pressure -= co2_removed
        
        # Crew respiration
        crew_respiration_o2 = self.crew_size * 0.02 * self.dt / 3600  # O2 consumed
        crew_respiration_co2 = self.crew_size * 0.025 * self.dt / 3600  # CO2 produced
        
        self.state.atmosphere.o2_partial_pressure -= crew_respiration_o2
        self.state.atmosphere.co2_partial_pressure += crew_respiration_co2
        
        # Keep values in bounds
        self.state.atmosphere.o2_partial_pressure = max(0, min(30, self.state.atmosphere.o2_partial_pressure))
        self.state.atmosphere.co2_partial_pressure = max(0, min(5, self.state.atmosphere.co2_partial_pressure))
        
        # Update total pressure
        self.state.atmosphere.total_pressure = (
            self.state.atmosphere.o2_partial_pressure + 
            self.state.atmosphere.co2_partial_pressure +
            self.state.atmosphere.n2_partial_pressure
        )
        
        # Reward for maintaining proper atmosphere
        if 18 <= self.state.atmosphere.o2_partial_pressure <= 25:
            reward += 1.0
        else:
            reward -= abs(21.3 - self.state.atmosphere.o2_partial_pressure) * 0.5
            
        if self.state.atmosphere.co2_partial_pressure <= 0.5:
            reward += 0.5
        else:
            reward -= (self.state.atmosphere.co2_partial_pressure - 0.5) * 2.0
            
        return reward
    
    def _simulate_power(self, power_actions: Dict[str, Any]) -> float:
        """Simulate power systems and return reward component."""
        reward = 0.0
        
        # Solar generation (simplified day/night cycle)
        day_phase = self.state.environment.lunar_day_phase
        solar_efficiency = max(0, math.cos((day_phase - 0.5) * 2 * math.pi))
        
        solar_angle = power_actions['solar_panel_angle']
        angle_efficiency = math.cos(math.radians(solar_angle)) * 0.1 + 0.9
        
        self.state.power.solar_generation = self.config.solar_capacity * solar_efficiency * angle_efficiency
        
        # Power consumption
        base_consumption = 5.0  # kW baseline
        life_support_consumption = 2.0  # Life support systems
        thermal_consumption = 1.5  # Thermal systems
        
        total_consumption = base_consumption + life_support_consumption + thermal_consumption
        self.state.power.total_load = total_consumption
        
        # Battery dynamics
        net_power = self.state.power.solar_generation - total_consumption
        fuel_cell_power = power_actions['fuel_cell_activation'] * self.config.fuel_cell_capacity
        
        if net_power < 0 and fuel_cell_power > 0:
            net_power += fuel_cell_power
            
        # Update battery charge
        battery_change = net_power * self.dt / 3600 / self.config.battery_capacity * 100
        self.state.power.battery_charge += battery_change
        self.state.power.battery_charge = max(0, min(100, self.state.power.battery_charge))
        
        # Reward for maintaining power balance
        if self.state.power.battery_charge > 20:
            reward += 0.5
        else:
            reward -= (20 - self.state.power.battery_charge) * 0.1
            
        if net_power >= 0:
            reward += 0.2  # Bonus for positive energy balance
            
        return reward
    
    def _simulate_thermal(self, thermal_actions: Dict[str, Any]) -> float:
        """Simulate thermal systems and return reward component."""
        reward = 0.0
        
        # External temperature variation
        day_phase = self.state.environment.lunar_day_phase
        external_temp = -100 + 150 * max(0, math.sin(day_phase * 2 * math.pi))
        self.state.thermal.external_temp = external_temp
        
        # Internal temperature dynamics (simplified)
        heating_power = sum(thermal_actions['heating_zones']) / len(thermal_actions['heating_zones'])
        heat_loss = (self.state.thermal.internal_temp_zones[0] - external_temp) * 0.001
        
        temp_change = (heating_power * 5 - heat_loss) * self.dt / 3600
        
        for i in range(len(self.state.thermal.internal_temp_zones)):
            self.state.thermal.internal_temp_zones[i] += temp_change * random.uniform(0.8, 1.2)
            self.state.thermal.internal_temp_zones[i] = max(0, min(50, self.state.thermal.internal_temp_zones[i]))
        
        # Reward for maintaining comfortable temperature
        avg_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
        if 20 <= avg_temp <= 25:
            reward += 1.0
        else:
            reward -= abs(22.5 - avg_temp) * 0.2
            
        return reward
    
    def _simulate_water(self, water_actions: Dict[str, Any]) -> float:
        """Simulate water systems and return reward component."""
        reward = 0.0
        
        # Water consumption by crew
        daily_consumption = self.crew_size * 3.0  # 3 liters per person per day
        consumption_rate = daily_consumption / 24 / 3600 * self.dt  # liters per timestep
        
        self.state.water.potable_water -= consumption_rate
        self.state.water.grey_water += consumption_rate * 0.8  # 80% becomes grey water
        
        # Water recycling
        recycling_intensity = water_actions['purification_intensity']
        recycled_water = self.state.water.grey_water * recycling_intensity * 0.1
        recycling_efficiency = self.state.water.recycling_efficiency
        
        clean_water_recovered = recycled_water * recycling_efficiency
        self.state.water.potable_water += clean_water_recovered
        self.state.water.grey_water -= recycled_water
        
        # Keep values in bounds
        self.state.water.potable_water = max(0, self.state.water.potable_water)
        self.state.water.grey_water = max(0, self.state.water.grey_water)
        
        # Reward for maintaining water supply
        if self.state.water.potable_water > 100:  # At least 100L reserve
            reward += 0.3
        else:
            reward -= (100 - self.state.water.potable_water) * 0.01
            
        return reward
    
    def _simulate_crew(self) -> float:
        """Simulate crew health and productivity."""
        reward = 0.0
        
        # Crew health affected by environmental conditions
        for i in range(len(self.state.crew.health)):
            # Health effects from atmosphere
            if self.state.atmosphere.o2_partial_pressure < 16:
                self.state.crew.health[i] -= 0.001
            if self.state.atmosphere.co2_partial_pressure > 1.0:
                self.state.crew.health[i] -= 0.0005
                
            # Health effects from temperature
            avg_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
            if not (18 <= avg_temp <= 28):
                self.state.crew.health[i] -= 0.0002
                
            # Keep health in bounds
            self.state.crew.health[i] = max(0, min(1, self.state.crew.health[i]))
            
            # Update stress based on conditions
            stress_factors = 0
            if self.state.atmosphere.o2_partial_pressure < 18:
                stress_factors += 0.001
            if self.state.atmosphere.co2_partial_pressure > 0.6:
                stress_factors += 0.001
            if self.state.power.battery_charge < 20:
                stress_factors += 0.002
                
            self.state.crew.stress[i] += stress_factors
            self.state.crew.stress[i] = max(0, min(1, self.state.crew.stress[i]))
            
            # Productivity based on health and stress
            self.state.crew.productivity[i] = self.state.crew.health[i] * (1 - self.state.crew.stress[i])
            
        # Reward based on crew status
        avg_health = sum(self.state.crew.health) / len(self.state.crew.health)
        avg_productivity = sum(self.state.crew.productivity) / len(self.state.crew.productivity)
        
        reward += avg_health * 0.5
        reward += avg_productivity * 0.3
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to failure conditions."""
        # Critical atmosphere failure
        if (self.state.atmosphere.o2_partial_pressure < 10 or
            self.state.atmosphere.co2_partial_pressure > 2.0):
            return True
            
        # Power system failure
        if self.state.power.battery_charge <= 0:
            return True
            
        # Crew health failure
        avg_health = sum(self.state.crew.health) / len(self.state.crew.health)
        if avg_health < 0.3:
            return True
            
        # Water depletion
        if self.state.water.potable_water <= 0:
            return True
            
        return False
    
    def _get_status(self) -> str:
        """Get current habitat status."""
        if self._check_termination():
            return 'critical'
        
        # Check warning conditions
        warnings = 0
        if self.state.atmosphere.o2_partial_pressure < 16:
            warnings += 1
        if self.state.atmosphere.co2_partial_pressure > 0.8:
            warnings += 1
        if self.state.power.battery_charge < 30:
            warnings += 1
        if self.state.water.potable_water < 200:
            warnings += 1
            
        if warnings >= 2:
            return 'warning'
        elif warnings >= 1:
            return 'caution'
        else:
            return 'nominal'
    
    def close(self):
        """Close the environment."""
        pass


def make_lunar_env(n_envs: int = 1, scenario: str = 'nominal_operations', 
                   reward_config: str = 'survival_focused'):
    """Create lunar habitat environment(s).
    
    Args:
        n_envs: Number of environments to create.
        scenario: Scenario configuration name.
        reward_config: Reward configuration name.
        
    Returns:
        Single environment if n_envs=1, otherwise list of environments.
    """
    envs = []
    for i in range(n_envs):
        config = HabitatConfig.from_preset("nasa_reference")
        config.scenario.name = scenario
        env = LunarHabitatEnv(config=config)
        envs.append(env)
    
    # Return single environment for n_envs=1, list otherwise
    return envs[0] if n_envs == 1 else envs