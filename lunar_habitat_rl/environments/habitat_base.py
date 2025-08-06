"""Base lunar habitat environment implementing Gymnasium interface."""

from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

from ..core import HabitatConfig, HabitatState, ActionSpace, MissionMetrics, PerformanceTracker
from ..physics import ThermalSimulator, CFDSimulator, ChemistrySimulator

logger = logging.getLogger(__name__)


class LunarHabitatEnv(gym.Env):
    """
    Lunar Habitat Environment for autonomous life support system training.
    
    This environment simulates a lunar habitat with realistic physics modeling
    for atmosphere, thermal, power, and water systems. The goal is to maintain
    crew safety and mission success over extended periods.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, 
                 config: Optional[Union[HabitatConfig, str]] = None,
                 crew_size: int = 4,
                 difficulty: str = "nominal",
                 scenario: str = "nominal_operations",
                 reward_config: str = "survival_focused",
                 render_mode: Optional[str] = None,
                 physics_enabled: bool = True):
        """
        Initialize the lunar habitat environment.
        
        Args:
            config: Habitat configuration or preset name
            crew_size: Number of crew members (1-12)
            difficulty: Difficulty level ('easy', 'nominal', 'hard', 'extreme')
            scenario: Mission scenario type
            reward_config: Reward function configuration
            render_mode: Rendering mode for visualization
            physics_enabled: Whether to use high-fidelity physics simulation
        """
        super().__init__()
        
        # Load configuration
        if isinstance(config, str):
            self.config = HabitatConfig.from_preset(config)
        elif isinstance(config, HabitatConfig):
            self.config = config
        else:
            self.config = HabitatConfig()
            
        # Override config with provided parameters
        self.config.crew.size = crew_size
        self.config.scenario.difficulty = difficulty
        
        self.scenario = scenario
        self.reward_config = reward_config
        self.render_mode = render_mode
        self.physics_enabled = physics_enabled
        
        # Initialize state and action spaces
        self.habitat_state = HabitatState(max_crew=max(crew_size, 6))
        self.action_handler = ActionSpace(num_zones=4, num_radiators=2)
        
        self.observation_space = self.habitat_state.get_observation_space()
        self.action_space = self.action_handler.get_action_space()
        
        # Initialize physics simulators
        self._init_physics_simulators()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.current_metrics = MissionMetrics()
        
        # Episode state
        self.steps_taken = 0
        self.max_steps = int(self.config.scenario.duration_days * 24 * 60)  # 1 step per minute
        self.mission_time = 0.0  # sols
        self.episode_reward = 0.0
        
        # System state
        self.system_failures = []
        self.emergency_active = False
        self.last_emergency_time = 0.0
        
        logger.info(f"Initialized LunarHabitatEnv with {crew_size} crew, "
                   f"difficulty={difficulty}, scenario={scenario}")
    
    def _init_physics_simulators(self):
        """Initialize high-fidelity physics simulators."""
        if not self.physics_enabled:
            self.thermal_sim = None
            self.cfd_sim = None  
            self.chemistry_sim = None
            return
            
        try:
            self.thermal_sim = ThermalSimulator(
                habitat_volume=self.config.volume,
                thermal_mass=5000.0,  # kg
                insulation_r_value=10.0,  # m²K/W
                solver=self.config.physics.thermal_solver
            )
            
            self.cfd_sim = CFDSimulator(
                volume=self.config.volume,
                turbulence_model=self.config.physics.turbulence_model
            )
            
            self.chemistry_sim = ChemistrySimulator(
                volume=self.config.volume,
                pressure=self.config.pressure_nominal,
                database=self.config.physics.reaction_database
            )
            
            logger.info("Physics simulators initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize physics simulators: {e}")
            self.thermal_sim = None
            self.cfd_sim = None
            self.chemistry_sim = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if options is None:
            options = {}
        
        # Reset episode state
        self.steps_taken = 0
        self.mission_time = 0.0
        self.episode_reward = 0.0
        self.system_failures.clear()
        self.emergency_active = False
        
        # Initialize habitat state based on configuration
        self._initialize_habitat_state()
        
        # Reset performance tracking
        self.performance_tracker.reset()
        self.current_metrics = MissionMetrics()
        
        # Generate initial observation
        observation = self.habitat_state.to_array()
        info = self._get_info()
        
        logger.debug("Environment reset successfully")
        return observation, info
    
    def _initialize_habitat_state(self):
        """Initialize habitat state with nominal conditions plus some variance."""
        # Atmosphere - start near nominal with small variations
        self.habitat_state.atmosphere.o2_partial_pressure = self.config.o2_nominal + np.random.normal(0, 0.5)
        self.habitat_state.atmosphere.co2_partial_pressure = 0.1 + np.random.uniform(0, 0.1)
        self.habitat_state.atmosphere.n2_partial_pressure = self.config.n2_nominal + np.random.normal(0, 1.0) 
        self.habitat_state.atmosphere.total_pressure = self.config.pressure_nominal + np.random.normal(0, 2.0)
        self.habitat_state.atmosphere.humidity = 40.0 + np.random.uniform(0, 10)
        self.habitat_state.atmosphere.temperature = self.config.temp_nominal + np.random.normal(0, 1.0)
        self.habitat_state.atmosphere.air_quality_index = 0.9 + np.random.uniform(0, 0.1)
        
        # Power - start with good battery charge
        self.habitat_state.power.battery_charge = 80.0 + np.random.uniform(0, 15)
        self.habitat_state.power.fuel_cell_capacity = 95.0 + np.random.uniform(0, 5)
        self.habitat_state.power.emergency_reserve = 100.0
        self.habitat_state.power.grid_stability = 0.98 + np.random.uniform(0, 0.02)
        
        # Thermal - start at comfortable temperatures
        base_temp = self.config.temp_nominal
        self.habitat_state.thermal.internal_temp_zones = [
            base_temp + np.random.normal(0, 0.5) for _ in range(4)
        ]
        self.habitat_state.thermal.external_temp = -50.0 + np.random.uniform(-20, 20)
        self.habitat_state.thermal.radiator_temps = [15.0 + np.random.normal(0, 2) for _ in range(2)]
        self.habitat_state.thermal.heat_pump_efficiency = 3.0 + np.random.uniform(0, 0.5)
        
        # Water - start with good reserves
        self.habitat_state.water.potable_water = self.config.water_storage * 0.8 + np.random.uniform(0, 100)
        self.habitat_state.water.grey_water = np.random.uniform(50, 150) 
        self.habitat_state.water.black_water = np.random.uniform(20, 60)
        self.habitat_state.water.recycling_efficiency = self.config.recycling_efficiency
        self.habitat_state.water.filter_status = 0.9 + np.random.uniform(0, 0.1)
        
        # Crew - start healthy but with individual variation
        crew_size = self.config.crew.size
        self.habitat_state.crew.health = [0.95 + np.random.normal(0, 0.05) for _ in range(crew_size)]
        self.habitat_state.crew.stress = [0.2 + np.random.uniform(0, 0.2) for _ in range(crew_size)]
        self.habitat_state.crew.productivity = [0.9 + np.random.normal(0, 0.05) for _ in range(crew_size)]
        
        locations = ['hab', 'lab', 'workshop', 'airlock']
        self.habitat_state.crew.locations = [
            np.random.choice(locations) for _ in range(crew_size)
        ]
        
        # Environment
        self.habitat_state.environment.mission_elapsed_time = 0.0
        self.habitat_state.environment.lunar_day_phase = np.random.uniform(0, 1) 
        self.habitat_state.environment.dust_accumulation = np.random.uniform(0, 0.05)
        self.habitat_state.environment.system_degradation = 0.0
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.steps_taken += 1
        self.mission_time = self.steps_taken / (24 * 60)  # Convert to sols
        
        # Parse action
        parsed_action = self.action_handler.parse_action(action)
        
        # Apply physics simulation
        self._apply_physics_step(parsed_action)
        
        # Update system state
        self._update_system_state(parsed_action)
        
        # Apply random events and degradation
        self._apply_random_events()
        
        # Check for emergencies
        self._check_emergency_conditions()
        
        # Compute reward
        reward = self._compute_reward(parsed_action)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.steps_taken >= self.max_steps
        
        # Update performance tracking
        state_dict = self._get_state_dict()
        info = self._get_info()
        self.performance_tracker.update_step(reward, state_dict, parsed_action, info)
        
        # Generate observation
        observation = self.habitat_state.to_array()
        
        if terminated or truncated:
            final_metrics = self.performance_tracker.update_episode(
                self.steps_taken, state_dict, info
            )
            info.update(final_metrics.to_dict())
            
            if terminated:
                logger.info(f"Episode terminated at step {self.steps_taken}, "
                           f"survival time: {self.mission_time:.2f} sols")
        
        return observation, reward, terminated, truncated, info
    
    def _apply_physics_step(self, action: Dict[str, Any]):
        """Apply physics simulation for one timestep."""
        if not self.physics_enabled or not all([self.thermal_sim, self.cfd_sim, self.chemistry_sim]):
            self._apply_simplified_physics(action)
            return
            
        # High-fidelity physics simulation
        dt = 60.0  # 1 minute timestep
        
        # Thermal simulation
        if self.thermal_sim:
            thermal_result = self.thermal_sim.step(
                dt=dt,
                external_temp=self.habitat_state.thermal.external_temp,
                internal_heat_generation=self._compute_internal_heat_generation(),
                heating_power=action['thermal']['heating_zones'],
                radiator_flow=action['thermal']['radiator_flow']
            )
            
            self.habitat_state.thermal.internal_temp_zones = thermal_result['zone_temperatures']
            self.habitat_state.thermal.radiator_temps = thermal_result['radiator_temperatures']
            self.habitat_state.thermal.heat_pump_efficiency = thermal_result['heat_pump_cop']
        
        # Atmosphere chemistry
        if self.chemistry_sim:
            chem_result = self.chemistry_sim.step(
                dt=dt,
                o2_generation_rate=action['life_support']['o2_generation_rate'],
                co2_scrubbing_rate=action['life_support']['co2_scrubber_power'],
                crew_metabolism=self._compute_crew_metabolism()
            )
            
            self.habitat_state.atmosphere.o2_partial_pressure = chem_result['o2_pressure']
            self.habitat_state.atmosphere.co2_partial_pressure = chem_result['co2_pressure']
            self.habitat_state.atmosphere.total_pressure = chem_result['total_pressure']
        
        # Fluid dynamics for air circulation
        if self.cfd_sim:
            flow_result = self.cfd_sim.step(
                dt=dt,
                fan_speed=action['life_support']['air_circulation_speed'],
                temperature_zones=self.habitat_state.thermal.internal_temp_zones
            )
            
            self.habitat_state.atmosphere.air_quality_index = flow_result['mixing_efficiency']
    
    def _apply_simplified_physics(self, action: Dict[str, Any]):
        """Simplified physics model for faster simulation."""
        dt = 1.0 / 60.0  # 1 minute in hours
        
        # Simplified atmosphere dynamics
        crew_o2_consumption = self.config.crew.size * 0.84  # kg/day
        crew_co2_production = self.config.crew.size * 1.04  # kg/day
        
        o2_generation = action['life_support']['o2_generation_rate'] * 2.0  # kg/day max capacity
        co2_scrubbing = action['life_support']['co2_scrubber_power'] * 1.5  # kg/day max capacity
        
        # Update partial pressures (simplified)
        self.habitat_state.atmosphere.o2_partial_pressure += (
            (o2_generation - crew_o2_consumption) * dt * 0.01  # Conversion factor
        )
        
        self.habitat_state.atmosphere.co2_partial_pressure += (
            (crew_co2_production - co2_scrubbing) * dt * 0.01
        )
        
        # Clamp to realistic ranges
        self.habitat_state.atmosphere.o2_partial_pressure = np.clip(
            self.habitat_state.atmosphere.o2_partial_pressure, 0.0, 50.0
        )
        self.habitat_state.atmosphere.co2_partial_pressure = np.clip(
            self.habitat_state.atmosphere.co2_partial_pressure, 0.0, 5.0
        )
        
        # Update total pressure
        self.habitat_state.atmosphere.total_pressure = (
            self.habitat_state.atmosphere.o2_partial_pressure +
            self.habitat_state.atmosphere.co2_partial_pressure +
            self.habitat_state.atmosphere.n2_partial_pressure
        )
        
        # Simplified thermal dynamics
        external_heat_loss = (self.habitat_state.thermal.external_temp - 
                             np.mean(self.habitat_state.thermal.internal_temp_zones)) * 0.001
        
        heating_power = np.sum(action['thermal']['heating_zones']) * self.config.heater_capacity / 1000.0  # kW
        internal_heat = self._compute_internal_heat_generation() / 1000.0  # kW
        
        for i in range(len(self.habitat_state.thermal.internal_temp_zones)):
            zone_heating = action['thermal']['heating_zones'][i] * heating_power / 4.0
            self.habitat_state.thermal.internal_temp_zones[i] += (
                (zone_heating + internal_heat / 4.0 + external_heat_loss) * dt * 0.1
            )
    
    def _compute_internal_heat_generation(self) -> float:
        """Compute internal heat generation from crew and equipment."""
        crew_heat = self.config.crew.size * self.config.crew.metabolic_rate  # W
        equipment_heat = 2000.0  # W (estimated equipment heat)
        lighting_heat = 500.0  # W (lighting heat)
        
        return crew_heat + equipment_heat + lighting_heat
    
    def _compute_crew_metabolism(self) -> Dict[str, float]:
        """Compute crew metabolic rates for atmospheric chemistry."""
        return {
            'o2_consumption_rate': self.config.crew.size * 0.84 / 86400.0,  # kg/s
            'co2_production_rate': self.config.crew.size * 1.04 / 86400.0,  # kg/s
            'water_vapor_rate': self.config.crew.size * 2.5 / 86400.0,  # kg/s
        }
    
    def _update_system_state(self, action: Dict[str, Any]):
        """Update system components based on actions and physics."""
        # Power system updates
        solar_input = self._compute_solar_generation(action['power']['solar_panel_angle'])
        power_consumption = self._compute_power_consumption(action)
        
        net_power = solar_input - power_consumption
        
        # Battery charging/discharging
        if net_power > 0:
            charge_rate = min(action['power']['battery_charge_rate'], 1.0)
            self.habitat_state.power.battery_charge += net_power * charge_rate * 0.1
        else:
            self.habitat_state.power.battery_charge += net_power * 0.1  # Discharge
            
        # Clamp battery charge
        self.habitat_state.power.battery_charge = np.clip(
            self.habitat_state.power.battery_charge, 0.0, 100.0
        )
        
        self.habitat_state.power.solar_generation = solar_input
        self.habitat_state.power.total_load = power_consumption
        
        # Water system updates
        water_consumption = self.config.crew.size * 3.5  # liters/day per person
        water_recovery = (self.habitat_state.water.grey_water + 
                         self.habitat_state.water.black_water) * action['water']['purification_intensity']
        
        self.habitat_state.water.potable_water -= water_consumption / (24 * 60)  # per minute
        self.habitat_state.water.potable_water += water_recovery * self.habitat_state.water.recycling_efficiency / (24 * 60)
        
        # Update grey and black water
        self.habitat_state.water.grey_water += water_consumption * 0.7 / (24 * 60)
        self.habitat_state.water.black_water += water_consumption * 0.3 / (24 * 60) 
        
        # Filter degradation
        filter_usage = action['water']['purification_intensity'] * 0.001
        self.habitat_state.water.filter_status = max(0.0, self.habitat_state.water.filter_status - filter_usage)
        
        # Crew state updates
        self._update_crew_state(action)
    
    def _compute_solar_generation(self, panel_angle: float) -> float:
        """Compute solar power generation based on angle and lunar conditions.""" 
        lunar_day_phase = self.habitat_state.environment.lunar_day_phase
        
        # Solar availability (0 during lunar night)
        if lunar_day_phase < 0.1 or lunar_day_phase > 0.9:
            solar_availability = 0.0
        else:
            # Peak at lunar noon (phase = 0.5)
            solar_availability = np.sin(np.pi * lunar_day_phase)
        
        # Panel angle efficiency (optimal around 0 degrees)
        angle_efficiency = np.cos(np.radians(panel_angle))
        
        # Dust accumulation reduces efficiency
        dust_factor = 1.0 - self.habitat_state.environment.dust_accumulation * 0.5
        
        # System degradation
        degradation_factor = 1.0 - self.habitat_state.environment.system_degradation * 0.3
        
        generation = (self.config.solar_capacity * solar_availability * 
                     angle_efficiency * dust_factor * degradation_factor)
        
        return max(0.0, generation)
    
    def _compute_power_consumption(self, action: Dict[str, Any]) -> float:
        """Compute total power consumption based on active systems."""
        base_load = 2.0  # kW baseline systems
        
        # Life support power
        life_support_power = (
            action['life_support']['o2_generation_rate'] * 1.5 +  # kW
            action['life_support']['co2_scrubber_power'] * 1.0 +   # kW
            action['life_support']['air_circulation_speed'] * self.config.fan_power / 1000.0  # kW
        )
        
        # Thermal system power
        heating_power = np.sum(action['thermal']['heating_zones']) * self.config.heater_capacity / 1000.0  # kW
        heat_pump_power = 0.5 if action['thermal']['heat_pump_mode'] > 0 else 0.0  # kW
        
        # Water system power
        water_power = action['water']['purification_intensity'] * 0.3  # kW
        
        # Load shedding reduces consumption
        load_shed_factor = 1.0 - np.mean(action['power']['load_shedding']) * 0.2
        
        total_consumption = (base_load + life_support_power + heating_power + 
                           heat_pump_power + water_power) * load_shed_factor
        
        return total_consumption
    
    def _update_crew_state(self, action: Dict[str, Any]):
        """Update crew health, stress, and productivity."""
        crew_size = len(self.habitat_state.crew.health)
        
        for i in range(crew_size):
            # Health factors
            health_change = 0.0
            
            # Atmosphere quality impact
            if self.habitat_state.atmosphere.o2_partial_pressure < 18.0:
                health_change -= 0.01  # Hypoxia
            if self.habitat_state.atmosphere.co2_partial_pressure > 0.8:
                health_change -= 0.005  # CO2 toxicity
                
            # Temperature comfort impact  
            zone_temps = self.habitat_state.thermal.internal_temp_zones
            avg_temp = np.mean(zone_temps)
            if avg_temp < 18.0 or avg_temp > 26.0:
                health_change -= 0.002  # Thermal stress
                
            # Apply health change
            self.habitat_state.crew.health[i] = np.clip(
                self.habitat_state.crew.health[i] + health_change, 0.0, 1.0
            )
            
            # Stress factors
            stress_change = 0.0
            
            # Emergency conditions increase stress
            if self.emergency_active:
                stress_change += 0.02
                
            # Poor air quality increases stress
            if self.habitat_state.atmosphere.air_quality_index < 0.8:
                stress_change += 0.01
                
            # Comfort reduces stress
            if 20.0 <= avg_temp <= 24.0 and self.habitat_state.atmosphere.humidity < 60.0:
                stress_change -= 0.001  # Comfort reduces stress
                
            self.habitat_state.crew.stress[i] = np.clip(
                self.habitat_state.crew.stress[i] + stress_change, 0.0, 1.0
            )
            
            # Productivity based on health and stress
            base_productivity = self.config.crew.productivity_base
            health_factor = self.habitat_state.crew.health[i]
            stress_factor = 1.0 - self.habitat_state.crew.stress[i]
            
            self.habitat_state.crew.productivity[i] = base_productivity * health_factor * stress_factor
    
    def _apply_random_events(self):
        """Apply random events and system degradation."""
        # System degradation over time
        degradation_rate = 0.001 / (24 * 60)  # 0.1% per day
        self.habitat_state.environment.system_degradation += degradation_rate
        
        # Dust accumulation
        if self.habitat_state.environment.lunar_day_phase > 0.1:  # During lunar day
            dust_rate = 0.01 / (24 * 60)  # 1% per day during day
            self.habitat_state.environment.dust_accumulation += dust_rate
            
        # Random equipment failures (rare)
        failure_probability = 1e-5  # Very rare failures
        
        if np.random.random() < failure_probability:
            failure_types = ['pump_failure', 'sensor_failure', 'valve_stuck']
            failure = np.random.choice(failure_types)
            self.system_failures.append({
                'type': failure,
                'time': self.mission_time,
                'severity': np.random.uniform(0.1, 0.9)
            })
            logger.warning(f"System failure occurred: {failure} at time {self.mission_time:.2f}")
        
        # Micrometeorite impacts
        impact_probability = self.config.scenario.micrometeorite_rate / (24 * 60)  # Per minute
        
        if np.random.random() < impact_probability:
            impact_severity = np.random.exponential(0.1)  # Most impacts are minor
            if impact_severity > 0.5:
                logger.warning(f"Micrometeorite impact detected, severity: {impact_severity:.2f}")
                # Could cause pressure loss, equipment damage, etc.
    
    def _check_emergency_conditions(self):
        """Check for emergency conditions requiring immediate response."""
        emergency_triggered = False
        emergency_type = None
        
        # Atmosphere emergencies
        if self.habitat_state.atmosphere.o2_partial_pressure < 16.0:
            emergency_triggered = True
            emergency_type = "oxygen_depletion"
            
        if self.habitat_state.atmosphere.co2_partial_pressure > 1.0:
            emergency_triggered = True
            emergency_type = "co2_toxicity"
            
        if self.habitat_state.atmosphere.total_pressure < 50.0:
            emergency_triggered = True
            emergency_type = "pressure_loss"
            
        # Power emergencies
        if (self.habitat_state.power.battery_charge < 5.0 and 
            self.habitat_state.power.solar_generation < 0.5):
            emergency_triggered = True
            emergency_type = "power_critical"
            
        # Thermal emergencies
        avg_temp = np.mean(self.habitat_state.thermal.internal_temp_zones)
        if avg_temp < 10.0 or avg_temp > 35.0:
            emergency_triggered = True
            emergency_type = "thermal_emergency"
            
        # Water emergencies
        if self.habitat_state.water.potable_water < 50.0:  # Less than 2 days supply
            emergency_triggered = True
            emergency_type = "water_shortage"
        
        # Crew health emergencies
        if any(h < 0.3 for h in self.habitat_state.crew.health):
            emergency_triggered = True
            emergency_type = "crew_medical_emergency"
        
        if emergency_triggered and not self.emergency_active:
            self.emergency_active = True
            self.last_emergency_time = self.mission_time
            logger.error(f"EMERGENCY ACTIVATED: {emergency_type} at time {self.mission_time:.2f}")
            
        elif not emergency_triggered and self.emergency_active:
            self.emergency_active = False
            response_time = (self.mission_time - self.last_emergency_time) * 24 * 60  # minutes
            self.current_metrics.emergency_response_time = response_time
            logger.info(f"Emergency resolved after {response_time:.1f} minutes")
    
    def _compute_reward(self, action: Dict[str, Any]) -> float:
        """Compute reward based on current state and actions."""
        reward = 0.0
        
        if self.reward_config == "survival_focused":
            reward = self._survival_focused_reward(action)
        elif self.reward_config == "efficiency_focused":
            reward = self._efficiency_focused_reward(action)
        elif self.reward_config == "exploration_focused":
            reward = self._exploration_focused_reward(action)
        else:
            reward = self._balanced_reward(action)
            
        return reward
    
    def _survival_focused_reward(self, action: Dict[str, Any]) -> float:
        """Reward function focused on crew survival and safety."""
        reward = 0.0
        
        # Base survival reward
        reward += 1.0  # Base reward for staying alive
        
        # Atmosphere quality reward
        o2_optimal = (16.0 <= self.habitat_state.atmosphere.o2_partial_pressure <= 25.0)
        co2_safe = self.habitat_state.atmosphere.co2_partial_pressure < 0.8
        pressure_good = self.habitat_state.atmosphere.total_pressure > 60.0
        
        if o2_optimal and co2_safe and pressure_good:
            reward += 5.0
        else:
            reward -= 2.0
            
        # Crew health reward
        avg_health = np.mean(self.habitat_state.crew.health)
        reward += avg_health * 3.0
        
        # Emergency penalties
        if self.emergency_active:
            reward -= 10.0
            
        # Safety violation penalties
        if self.habitat_state.atmosphere.o2_partial_pressure < 16.0:
            reward -= 20.0  # Severe penalty for hypoxia
            
        if self.habitat_state.atmosphere.co2_partial_pressure > 1.0:
            reward -= 15.0  # Severe penalty for CO2 toxicity
            
        # Power stability reward
        if self.habitat_state.power.battery_charge > 20.0:
            reward += 1.0
        elif self.habitat_state.power.battery_charge < 10.0:
            reward -= 5.0
            
        return reward
    
    def _efficiency_focused_reward(self, action: Dict[str, Any]) -> float:
        """Reward function focused on resource efficiency and optimization."""
        reward = 0.0
        
        # Base survival
        if not self._check_termination():
            reward += 1.0
            
        # Power efficiency
        power_generation = self.habitat_state.power.solar_generation
        power_consumption = self.habitat_state.power.total_load
        
        if power_consumption > 0:
            power_efficiency = min(1.0, power_generation / power_consumption)
            reward += power_efficiency * 2.0
            
        # Water conservation
        recycling_eff = self.habitat_state.water.recycling_efficiency
        reward += recycling_eff * 1.5
        
        # System optimization bonuses
        temp_variance = np.var(self.habitat_state.thermal.internal_temp_zones)
        if temp_variance < 1.0:  # Well-regulated temperatures
            reward += 1.0
            
        # Penalize excessive actions (encourage efficiency)
        action_magnitude = np.mean([
            np.mean(action['life_support']['o2_generation_rate']),
            np.mean(action['power']['battery_charge_rate']),
            np.mean(action['thermal']['heating_zones'])
        ])
        
        if action_magnitude < 0.7:  # Reward moderate actions
            reward += 0.5
            
        return reward
    
    def _exploration_focused_reward(self, action: Dict[str, Any]) -> float:
        """Reward function to encourage exploration of state-action space."""
        reward = self._survival_focused_reward(action) * 0.5  # Base survival
        
        # Encourage diverse actions
        action_entropy = -np.sum(action * np.log(action + 1e-8)) if np.all(action >= 0) else 0
        reward += action_entropy * 0.1
        
        # Reward for experiencing different system states
        state_novelty = 0.0  # Would need state history to compute
        reward += state_novelty
        
        return reward
    
    def _balanced_reward(self, action: Dict[str, Any]) -> float:
        """Balanced reward function combining survival, efficiency, and performance."""
        survival_reward = self._survival_focused_reward(action) * 0.6
        efficiency_reward = self._efficiency_focused_reward(action) * 0.4
        
        return survival_reward + efficiency_reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to mission failure."""
        # Crew death
        if any(h <= 0.0 for h in self.habitat_state.crew.health):
            return True
            
        # Catastrophic atmosphere failure
        if self.habitat_state.atmosphere.o2_partial_pressure < 10.0:
            return True
            
        if self.habitat_state.atmosphere.co2_partial_pressure > 2.0:
            return True
            
        if self.habitat_state.atmosphere.total_pressure < 30.0:
            return True
            
        # Total power failure
        if (self.habitat_state.power.battery_charge <= 0.0 and 
            self.habitat_state.power.solar_generation <= 0.0):
            return True
            
        # Water depletion
        if self.habitat_state.water.potable_water <= 0.0:
            return True
            
        return False
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """Get current state as dictionary for logging."""
        return {
            'o2_partial_pressure': self.habitat_state.atmosphere.o2_partial_pressure,
            'co2_partial_pressure': self.habitat_state.atmosphere.co2_partial_pressure,
            'total_pressure': self.habitat_state.atmosphere.total_pressure,
            'temperature': self.habitat_state.atmosphere.temperature,
            'battery_charge': self.habitat_state.power.battery_charge,
            'solar_generation': self.habitat_state.power.solar_generation,
            'grid_stability': self.habitat_state.power.grid_stability,
            'internal_temp_zones': self.habitat_state.thermal.internal_temp_zones,
            'potable_water': self.habitat_state.water.potable_water,
            'recycling_efficiency': self.habitat_state.water.recycling_efficiency,
            'crew_health': self.habitat_state.crew.health,
            'crew_stress': self.habitat_state.crew.stress,
            'air_quality_index': self.habitat_state.atmosphere.air_quality_index,
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get episode information dictionary."""
        return {
            'mission_time': self.mission_time,
            'steps_taken': self.steps_taken,
            'emergency_active': self.emergency_active,
            'system_failures': len(self.system_failures),
            'survival_time': self.mission_time,
            'episode_reward': self.episode_reward,
            'crew_size': self.config.crew.size,
            'scenario': self.scenario,
            'difficulty': self.config.scenario.difficulty,
        }
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            print(f"\nMission Time: {self.mission_time:.2f} sols")
            print(f"Crew Health: {np.mean(self.habitat_state.crew.health):.2f}")
            print(f"O2: {self.habitat_state.atmosphere.o2_partial_pressure:.1f} kPa")
            print(f"CO2: {self.habitat_state.atmosphere.co2_partial_pressure:.1f} kPa") 
            print(f"Battery: {self.habitat_state.power.battery_charge:.1f}%")
            print(f"Water: {self.habitat_state.water.potable_water:.0f} L")
            if self.emergency_active:
                print("*** EMERGENCY ACTIVE ***")
        
        return None  # Would return RGB array for rgb_array mode
    
    def close(self):
        """Clean up environment resources."""
        pass