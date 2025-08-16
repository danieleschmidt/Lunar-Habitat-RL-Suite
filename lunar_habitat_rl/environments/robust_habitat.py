"""Robust habitat environment with comprehensive error handling and monitoring - Generation 2"""

import time
import random
import math
from typing import Dict, List, Tuple, Optional, Any

from ..core.lightweight_config import HabitatConfig
from ..core.lightweight_state import HabitatState, ActionSpace
from ..utils.robust_logging import get_logger, PerformanceMonitor, log_exception, log_performance
from ..utils.robust_validation import get_validator, validate_and_sanitize_observation, validate_and_sanitize_action
from ..utils.robust_monitoring import get_simulation_monitor, monitor_simulation_performance


class RobustLunarHabitatEnv:
    """Production-ready Lunar Habitat RL Environment - Generation 2 Implementation.
    
    Features comprehensive error handling, input validation, monitoring, logging,
    and safety checks for mission-critical lunar habitat simulations.
    """
    
    def __init__(self, config: Optional[HabitatConfig] = None, crew_size: int = 4, 
                 safety_mode: str = "strict", enable_monitoring: bool = True):
        """Initialize the robust habitat environment.
        
        Args:
            config: Habitat configuration. If None, uses default configuration.
            crew_size: Number of crew members (1-6).
            safety_mode: Safety validation mode ('strict', 'moderate', 'permissive').
            enable_monitoring: Whether to enable performance monitoring.
        """
        # Initialize logging and monitoring
        self.logger = get_logger()
        self.validator = get_validator()
        self.simulation_monitor = get_simulation_monitor()
        
        # Log environment initialization
        self.logger.info(f"Initializing RobustLunarHabitatEnv - crew_size={crew_size}, safety_mode={safety_mode}")
        
        try:
            # Validate inputs
            self._validate_init_parameters(config, crew_size, safety_mode)
            
            # Configuration
            self.config = config or HabitatConfig()
            self.crew_size = min(max(crew_size, 1), 6)
            self.safety_mode = safety_mode
            self.enable_monitoring = enable_monitoring
            
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
            self.episode_count = 0
            
            # Safety and error tracking
            self.safety_violations = []
            self.error_history = []
            self.validation_failures = 0
            
            # Physics simulation
            self.dt = 60.0  # 1 minute timesteps
            self.physics_stability = 1.0  # Physics simulation stability factor
            
            # Emergency protocols
            self.emergency_mode = False
            self.critical_alerts = []
            
            self.logger.info(f"RobustLunarHabitatEnv initialized successfully - obs_dims={state_info['total_dims']}, action_dims={action_info['total_dims']}")
            
        except Exception as e:
            self.logger.critical(f"Failed to initialize RobustLunarHabitatEnv: {e}", error=e)
            raise
    
    def _validate_init_parameters(self, config, crew_size, safety_mode):
        """Validate initialization parameters."""
        if crew_size < 1 or crew_size > 6:
            raise ValueError(f"crew_size must be between 1 and 6, got {crew_size}")
        
        if safety_mode not in ['strict', 'moderate', 'permissive']:
            raise ValueError(f"safety_mode must be 'strict', 'moderate', or 'permissive', got {safety_mode}")
        
        if config is not None and not isinstance(config, HabitatConfig):
            raise TypeError(f"config must be HabitatConfig instance, got {type(config)}")
    
    @log_exception
    @log_performance("environment_reset")
    def reset(self, seed: Optional[int] = None) -> Tuple[List[float], Dict[str, Any]]:
        """Reset the environment to initial state with comprehensive error handling.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (observation, info)
        """
        with PerformanceMonitor(self.logger, "reset", episode=self.episode_count):
            try:
                # Log episode start
                if self.enable_monitoring:
                    self.simulation_monitor.log_episode_start()
                
                # Set random seed if provided
                if seed is not None:
                    random.seed(seed)
                    self.logger.info(f"Random seed set to {seed}")
                
                # Reset state to nominal conditions
                self.state = HabitatState(max_crew=self.crew_size)
                self.current_step = 0
                self.episode_reward = 0.0
                self.episode_count += 1
                
                # Clear episode-specific tracking
                self.safety_violations.clear()
                self.emergency_mode = False
                self.critical_alerts.clear()
                
                # Add realistic initial variation
                self._apply_initial_variations()
                
                # Validate initial state
                observation = self.state.to_array()
                is_valid, sanitized_obs, validation_result = validate_and_sanitize_observation(
                    observation, self.crew_size
                )
                
                if not is_valid:
                    self.logger.warning(f"Initial state validation failed: {validation_result}")
                    observation = sanitized_obs
                    self.validation_failures += 1
                
                # Prepare info
                info = {
                    'step': self.current_step,
                    'total_reward': self.episode_reward,
                    'crew_size': self.crew_size,
                    'status': self._get_status(),
                    'safety_mode': self.safety_mode,
                    'episode_count': self.episode_count,
                    'validation_result': validation_result if not is_valid else None
                }
                
                self.logger.info(f"Environment reset complete - episode {self.episode_count}, status: {info['status']}")
                
                return observation, info
                
            except Exception as e:
                self.logger.critical(f"Environment reset failed: {e}", error=e)
                self._handle_critical_error("reset", e)
                # Return emergency safe state
                return self._get_emergency_state()
    
    @log_exception
    @monitor_simulation_performance
    def step(self, action: List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step with robust error handling and validation.
        
        Args:
            action: Action to execute (list of floats).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        with PerformanceMonitor(self.logger, "step", 
                              step=self.current_step, episode=self.episode_count):
            try:
                self.current_step += 1
                step_start_time = time.time()
                
                # Validate and sanitize action
                is_valid_action, sanitized_action, action_validation = validate_and_sanitize_action(
                    action, expected_dims=self.action_space.shape[0]
                )
                
                if not is_valid_action:
                    self.logger.warning(f"Action validation failed: {action_validation}")
                    self.validation_failures += 1
                    
                    if self.safety_mode == "strict":
                        # In strict mode, invalid actions cause episode termination
                        reward = -100.0
                        terminated = True
                        truncated = False
                        info = {
                            'error': 'Invalid action in strict safety mode',
                            'step': self.current_step,
                            'action_validation': action_validation
                        }
                        return self.state.to_array(), reward, terminated, truncated, info
                
                # Use sanitized action
                action_to_use = sanitized_action
                
                # Parse action with error handling
                try:
                    parsed_action = self.action_parser.parse_action(action_to_use)
                except Exception as e:
                    self.logger.error(f"Action parsing failed: {e}", error=e)
                    # Use safe default action
                    parsed_action = self._get_safe_default_action()
                
                # Apply actions and simulate physics
                reward = self._simulate_step_robust(parsed_action)
                
                # Check termination conditions
                terminated, termination_reason = self._check_termination_robust()
                truncated = self.current_step >= self.max_steps
                
                # Update episode reward
                self.episode_reward += reward
                
                # Validate state after simulation
                observation = self.state.to_array()
                is_valid_obs, sanitized_obs, obs_validation = validate_and_sanitize_observation(
                    observation, self.crew_size
                )
                
                if not is_valid_obs:
                    self.logger.error(f"State validation failed after simulation: {obs_validation}")
                    observation = sanitized_obs
                    self.validation_failures += 1
                    
                    # In strict mode, state validation failure is critical
                    if self.safety_mode == "strict":
                        terminated = True
                        termination_reason = "State validation failure"
                
                # Log simulation state for debugging
                self.logger.log_simulation_state(
                    self.current_step, 
                    self._parse_state_for_logging(observation),
                    reward,
                    self._get_status()
                )
                
                # Prepare comprehensive info
                info = self._prepare_step_info(
                    action_validation, obs_validation, termination_reason,
                    step_start_time, parsed_action
                )
                
                # Handle episode end
                if terminated or truncated:
                    self._handle_episode_end(terminated, truncated, termination_reason)
                
                return observation, reward, terminated, truncated, info
                
            except Exception as e:
                self.logger.critical(f"Environment step failed at step {self.current_step}: {e}", error=e)
                self._handle_critical_error("step", e)
                return self._get_emergency_step_result()
    
    def _apply_initial_variations(self):
        """Apply realistic initial variations to state."""
        try:
            # Small variations in atmospheric conditions
            self.state.atmosphere.temperature += random.uniform(-1.0, 1.0)
            self.state.atmosphere.o2_partial_pressure += random.uniform(-0.5, 0.5)
            self.state.atmosphere.co2_partial_pressure += random.uniform(-0.05, 0.05)
            
            # Power system variations
            self.state.power.battery_charge += random.uniform(-10.0, 10.0)
            self.state.power.solar_generation += random.uniform(-1.0, 1.0)
            
            # Water system variations
            self.state.water.potable_water += random.uniform(-50.0, 50.0)
            
            # Crew variations
            for i in range(len(self.state.crew.health)):
                self.state.crew.health[i] += random.uniform(-0.05, 0.05)
                self.state.crew.stress[i] += random.uniform(-0.1, 0.1)
            
            # Ensure all values remain in valid ranges
            self._clamp_state_values()
            
        except Exception as e:
            self.logger.warning(f"Failed to apply initial variations: {e}")
    
    def _simulate_step_robust(self, parsed_action: Dict[str, Any]) -> float:
        """Simulate one timestep with comprehensive error handling."""
        try:
            reward = 0.0
            simulation_errors = []
            
            # Simulate each subsystem with individual error handling
            try:
                reward += self._simulate_atmosphere(parsed_action['life_support'])
            except Exception as e:
                simulation_errors.append(f"Atmosphere simulation: {e}")
                self.logger.error(f"Atmosphere simulation failed: {e}")
                reward -= 5.0  # Penalty for simulation failure
            
            try:
                reward += self._simulate_power(parsed_action['power'])
            except Exception as e:
                simulation_errors.append(f"Power simulation: {e}")
                self.logger.error(f"Power simulation failed: {e}")
                reward -= 5.0
            
            try:
                reward += self._simulate_thermal(parsed_action['thermal'])
            except Exception as e:
                simulation_errors.append(f"Thermal simulation: {e}")
                self.logger.error(f"Thermal simulation failed: {e}")
                reward -= 3.0
            
            try:
                reward += self._simulate_water(parsed_action['water'])
            except Exception as e:
                simulation_errors.append(f"Water simulation: {e}")
                self.logger.error(f"Water simulation failed: {e}")
                reward -= 3.0
            
            try:
                reward += self._simulate_crew()
            except Exception as e:
                simulation_errors.append(f"Crew simulation: {e}")
                self.logger.error(f"Crew simulation failed: {e}")
                reward -= 2.0
            
            # Apply physics degradation if there were errors
            if simulation_errors:
                self.physics_stability *= 0.99  # Slight degradation
                self.error_history.extend(simulation_errors)
                
                # Keep error history manageable
                if len(self.error_history) > 100:
                    self.error_history = self.error_history[-100:]
            else:
                # Slowly recover physics stability
                self.physics_stability = min(1.0, self.physics_stability + 0.001)
            
            # Apply stability factor to reward
            reward *= self.physics_stability
            
            # Add small time penalty
            reward -= 0.01
            
            # Clamp state values after simulation
            self._clamp_state_values()
            
            return reward
            
        except Exception as e:
            self.logger.critical(f"Critical simulation failure: {e}", error=e)
            self.physics_stability *= 0.95
            return -10.0  # Large penalty for critical failure
    
    def _simulate_atmosphere(self, life_support_actions: Dict[str, Any]) -> float:
        """Simulate atmospheric systems with enhanced error handling."""
        reward = 0.0
        
        # Validate life support actions
        for key in ['o2_generation_rate', 'co2_scrubber_power']:
            if key not in life_support_actions:
                raise ValueError(f"Missing life support action: {key}")
        
        # O2 generation with efficiency factors
        o2_rate = life_support_actions['o2_generation_rate']
        system_efficiency = 0.9 + (self.physics_stability - 1.0) * 0.1  # Degraded efficiency with errors
        o2_generated = o2_rate * 0.1 * self.dt / 3600 * system_efficiency
        self.state.atmosphere.o2_partial_pressure += o2_generated
        
        # CO2 scrubbing
        co2_scrub_power = life_support_actions['co2_scrubber_power']
        co2_removed = co2_scrub_power * 0.05 * self.dt / 3600 * system_efficiency
        self.state.atmosphere.co2_partial_pressure -= co2_removed
        
        # Crew respiration (based on activity and stress levels)
        avg_crew_stress = sum(self.state.crew.stress) / len(self.state.crew.stress)
        respiration_factor = 1.0 + avg_crew_stress * 0.2  # Higher stress = higher respiration
        
        crew_respiration_o2 = self.crew_size * 0.02 * self.dt / 3600 * respiration_factor
        crew_respiration_co2 = self.crew_size * 0.025 * self.dt / 3600 * respiration_factor
        
        self.state.atmosphere.o2_partial_pressure -= crew_respiration_o2
        self.state.atmosphere.co2_partial_pressure += crew_respiration_co2
        
        # Update total pressure
        self.state.atmosphere.total_pressure = (
            self.state.atmosphere.o2_partial_pressure + 
            self.state.atmosphere.co2_partial_pressure +
            self.state.atmosphere.n2_partial_pressure
        )
        
        # Enhanced reward calculation with safety checks
        o2_pressure = self.state.atmosphere.o2_partial_pressure
        co2_pressure = self.state.atmosphere.co2_partial_pressure
        
        # O2 reward with safety zones
        if 18 <= o2_pressure <= 25:
            reward += 2.0  # Optimal range
        elif 16 <= o2_pressure < 18 or 25 < o2_pressure <= 27:
            reward += 0.5  # Acceptable range
        elif o2_pressure < 16:
            reward -= (16 - o2_pressure) * 10  # Dangerous low
            self._add_safety_violation(f"Low O2: {o2_pressure:.1f} kPa")
        elif o2_pressure > 30:
            reward -= (o2_pressure - 30) * 5  # Fire risk
            self._add_safety_violation(f"High O2 fire risk: {o2_pressure:.1f} kPa")
        else:
            reward -= abs(21.3 - o2_pressure) * 0.5
        
        # CO2 reward with health impacts
        if co2_pressure <= 0.4:
            reward += 1.0
        elif co2_pressure <= 0.5:
            reward += 0.2
        else:
            reward -= (co2_pressure - 0.5) * 20  # Severe health impact
            self._add_safety_violation(f"High CO2: {co2_pressure:.2f} kPa")
        
        return reward
    
    def _simulate_power(self, power_actions: Dict[str, Any]) -> float:
        """Simulate power systems with load balancing and efficiency."""
        reward = 0.0
        
        # Solar generation with day/night cycle and panel degradation
        day_phase = self.state.environment.lunar_day_phase
        solar_efficiency = max(0, math.cos((day_phase - 0.5) * 2 * math.pi))
        
        # Panel degradation due to dust and age
        dust_factor = 1.0 - self.state.environment.dust_accumulation * 0.3
        degradation_factor = 1.0 - self.state.environment.system_degradation * 0.2
        
        solar_angle = power_actions.get('solar_panel_angle', 0)
        angle_efficiency = math.cos(math.radians(solar_angle)) * 0.1 + 0.9
        
        self.state.power.solar_generation = (
            self.config.solar_capacity * solar_efficiency * 
            dust_factor * degradation_factor * angle_efficiency
        )
        
        # Dynamic power consumption
        base_consumption = 5.0
        life_support_load = 2.0 + (1 - self.physics_stability) * 1.0  # Higher load if systems degraded
        thermal_load = self._calculate_thermal_load()
        emergency_load = 1.0 if self.emergency_mode else 0.0
        
        # Load shedding impact
        load_shedding = power_actions.get('load_shedding', [False] * 4)
        shed_reduction = sum(load_shedding) * 0.5  # 0.5 kW per shed zone
        
        total_consumption = max(0, base_consumption + life_support_load + thermal_load + emergency_load - shed_reduction)
        self.state.power.total_load = total_consumption
        
        # Battery and fuel cell dynamics
        net_power = self.state.power.solar_generation - total_consumption
        fuel_cell_power = power_actions.get('fuel_cell_activation', 0) * self.config.fuel_cell_capacity
        
        if net_power < 0:
            net_power += fuel_cell_power
            # Fuel cell efficiency and consumption
            fuel_consumed = fuel_cell_power * 0.1  # Fuel consumption rate
            self.state.power.fuel_cell_capacity = max(0, self.state.power.fuel_cell_capacity - fuel_consumed)
        
        # Battery dynamics with charge/discharge efficiency
        charge_efficiency = 0.95 if net_power > 0 else 1.0  # Slight loss when charging
        battery_change = net_power * self.dt / 3600 / self.config.battery_capacity * 100 * charge_efficiency
        self.state.power.battery_charge += battery_change
        
        # Grid stability based on load balance
        load_balance = abs(self.state.power.solar_generation - total_consumption) / max(total_consumption, 1.0)
        self.state.power.grid_stability = max(0.5, 1.0 - load_balance * 0.2)
        
        # Power reward calculation
        battery_level = self.state.power.battery_charge
        if battery_level > 60:
            reward += 1.0
        elif battery_level > 30:
            reward += 0.5
        elif battery_level > 10:
            reward -= (30 - battery_level) * 0.1
        else:
            reward -= (10 - battery_level) * 2.0  # Critical low battery
            self._add_safety_violation(f"Critical battery level: {battery_level:.1f}%")
        
        # Energy efficiency bonus
        if net_power >= 0:
            reward += 0.3
        
        # Grid stability reward
        reward += (self.state.power.grid_stability - 0.9) * 2.0
        
        return reward
    
    def _calculate_thermal_load(self) -> float:
        """Calculate thermal system power consumption."""
        # Base thermal load
        base_load = 1.5
        
        # Additional load based on external temperature
        external_temp = self.state.thermal.external_temp
        if external_temp < -50:
            temp_load = (abs(external_temp) - 50) * 0.02
        else:
            temp_load = 0
        
        return base_load + temp_load
    
    def _simulate_thermal(self, thermal_actions: Dict[str, Any]) -> float:
        """Enhanced thermal simulation with heat transfer physics."""
        reward = 0.0
        
        # External temperature with lunar day/night cycle
        day_phase = self.state.environment.lunar_day_phase
        # Lunar surface temperature range: -173°C to 127°C
        external_temp = -123 + 150 * max(0, math.sin(day_phase * 2 * math.pi))
        self.state.thermal.external_temp = external_temp
        
        # Internal temperature dynamics
        heating_zones = thermal_actions.get('heating_zones', [0.5] * 4)
        heating_power = sum(heating_zones) / len(heating_zones)
        
        # Heat loss calculation (improved physics)
        avg_internal_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
        temp_diff = avg_internal_temp - external_temp
        
        # Heat loss depends on insulation and habitat surface area
        insulation_factor = thermal_actions.get('insulation_deployment', 0.8)
        heat_loss_coefficient = 0.001 * (1 - insulation_factor * 0.5)
        heat_loss = temp_diff * heat_loss_coefficient
        
        # Heat generation from equipment and crew
        equipment_heat = 2.0  # kW from equipment
        crew_heat = self.crew_size * 0.1  # 100W per crew member
        total_heat_input = heating_power * 5 + equipment_heat + crew_heat
        
        # Temperature change calculation
        habitat_thermal_mass = 50000  # J/K (approximate for 200m³ habitat)
        temp_change = (total_heat_input - heat_loss) * self.dt / habitat_thermal_mass
        
        # Update zone temperatures with some variation
        for i in range(len(self.state.thermal.internal_temp_zones)):
            zone_variation = random.uniform(0.8, 1.2)  # Thermal variations between zones
            self.state.thermal.internal_temp_zones[i] += temp_change * zone_variation
        
        # Heat pump efficiency calculation
        heat_pump_mode = thermal_actions.get('heat_pump_mode', 1)
        if heat_pump_mode > 0:
            temp_difference = abs(avg_internal_temp - external_temp)
            # COP decreases with larger temperature differences
            self.state.thermal.heat_pump_efficiency = max(1.5, 4.0 - temp_difference * 0.02)
        
        # Radiator temperature management
        radiator_flow = thermal_actions.get('radiator_flow', [0.5, 0.5])
        for i, flow in enumerate(radiator_flow[:2]):
            if i < len(self.state.thermal.radiator_temps):
                # Radiator effectiveness depends on flow rate
                target_temp = 15.0 + flow * 10  # Higher flow = higher temp
                self.state.thermal.radiator_temps[i] += (target_temp - self.state.thermal.radiator_temps[i]) * 0.1
        
        # Temperature reward with comfort zones
        avg_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
        
        if 20 <= avg_temp <= 25:
            reward += 2.0  # Optimal comfort zone
        elif 18 <= avg_temp < 20 or 25 < avg_temp <= 28:
            reward += 1.0  # Acceptable range
        elif 15 <= avg_temp < 18 or 28 < avg_temp <= 32:
            reward -= (abs(22.5 - avg_temp) - 2.5) * 0.5  # Suboptimal
        else:
            # Dangerous temperatures
            if avg_temp < 15:
                reward -= (15 - avg_temp) * 5
                self._add_safety_violation(f"Dangerous low temperature: {avg_temp:.1f}°C")
            elif avg_temp > 32:
                reward -= (avg_temp - 32) * 5
                self._add_safety_violation(f"Dangerous high temperature: {avg_temp:.1f}°C")
        
        # Energy efficiency reward
        if heating_power < 0.8 and 20 <= avg_temp <= 25:
            reward += 0.5  # Bonus for efficient heating
        
        return reward
    
    def _simulate_water(self, water_actions: Dict[str, Any]) -> float:
        """Enhanced water system simulation with recycling efficiency."""
        reward = 0.0
        
        # Crew water consumption (varies with activity and temperature)
        avg_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
        temp_factor = 1.0 + max(0, (avg_temp - 22.5) / 10)  # Higher consumption in heat
        
        daily_consumption = self.crew_size * 3.0 * temp_factor  # liters per day
        consumption_rate = daily_consumption / 24 / 3600 * self.dt
        
        self.state.water.potable_water -= consumption_rate
        self.state.water.grey_water += consumption_rate * 0.8
        
        # Water recycling with system efficiency
        recycling_priority = water_actions.get('recycling_priority', 2)
        purification_intensity = water_actions.get('purification_intensity', 0.8)
        
        # System efficiency affected by filter status and degradation
        base_efficiency = self.state.water.recycling_efficiency
        filter_efficiency = self.state.water.filter_status
        system_efficiency = base_efficiency * filter_efficiency * self.physics_stability
        
        # Amount recycled depends on priority and intensity
        recycling_rate = min(self.state.water.grey_water, 
                           self.state.water.grey_water * purification_intensity * 0.2)
        
        clean_water_recovered = recycling_rate * system_efficiency
        self.state.water.potable_water += clean_water_recovered
        self.state.water.grey_water -= recycling_rate
        
        # Filter degradation over time
        filter_degradation = 0.001 * self.dt / 3600  # Slow degradation
        if purification_intensity > 0.7:
            filter_degradation *= 1.5  # Faster degradation at high intensity
        
        self.state.water.filter_status = max(0, self.state.water.filter_status - filter_degradation)
        
        # Water rationing impact
        rationing_level = water_actions.get('rationing_level', 0)
        if rationing_level > 0:
            # Rationing reduces consumption but increases crew stress
            consumption_reduction = rationing_level * 0.2
            self.state.water.potable_water += consumption_rate * consumption_reduction
            
            # Increase crew stress due to rationing
            for i in range(len(self.state.crew.stress)):
                self.state.crew.stress[i] += rationing_level * 0.001
        
        # Water supply reward
        water_level = self.state.water.potable_water
        if water_level > 500:
            reward += 1.0
        elif water_level > 200:
            reward += 0.5
        elif water_level > 100:
            reward -= (200 - water_level) * 0.002
        elif water_level > 50:
            reward -= (100 - water_level) * 0.01
            self._add_safety_violation(f"Low water supply: {water_level:.1f}L")
        else:
            reward -= (50 - water_level) * 0.1
            self._add_safety_violation(f"Critical water shortage: {water_level:.1f}L")
        
        # Recycling efficiency reward
        if system_efficiency > 0.9:
            reward += 0.3
        elif system_efficiency < 0.7:
            reward -= 0.2
        
        return reward
    
    def _simulate_crew(self) -> float:
        """Enhanced crew simulation with detailed health modeling."""
        reward = 0.0
        
        # Environmental impact on crew health
        o2_pressure = self.state.atmosphere.o2_partial_pressure
        co2_pressure = self.state.atmosphere.co2_partial_pressure
        avg_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
        
        for i in range(len(self.state.crew.health)):
            health_change = 0.0
            stress_change = 0.0
            
            # Atmospheric effects on health
            if o2_pressure < 16:
                health_change -= 0.002  # Severe hypoxia
                stress_change += 0.003
            elif o2_pressure < 18:
                health_change -= 0.001  # Mild hypoxia
                stress_change += 0.002
            elif o2_pressure > 30:
                stress_change += 0.001  # Fire risk stress
            
            if co2_pressure > 1.0:
                health_change -= 0.001  # CO2 poisoning
                stress_change += 0.002
            elif co2_pressure > 0.5:
                health_change -= 0.0005
                stress_change += 0.001
            
            # Temperature effects
            if avg_temp < 18 or avg_temp > 28:
                health_change -= 0.0003
                stress_change += 0.001
            elif 20 <= avg_temp <= 25:
                stress_change -= 0.0002  # Comfort reduces stress
            
            # Power outage stress
            if self.state.power.battery_charge < 20:
                stress_change += 0.002
            elif self.state.power.battery_charge < 50:
                stress_change += 0.001
            
            # Water shortage stress
            if self.state.water.potable_water < 100:
                stress_change += 0.003
            elif self.state.water.potable_water < 200:
                stress_change += 0.001
            
            # Apply changes
            self.state.crew.health[i] += health_change
            self.state.crew.stress[i] += stress_change
            
            # Natural stress reduction over time
            self.state.crew.stress[i] -= 0.0001  # Gradual stress reduction
            
            # Bounds checking
            self.state.crew.health[i] = max(0, min(1, self.state.crew.health[i]))
            self.state.crew.stress[i] = max(0, min(1, self.state.crew.stress[i]))
            
            # Update productivity based on health and stress
            self.state.crew.productivity[i] = (
                self.state.crew.health[i] * (1 - self.state.crew.stress[i]) * 
                random.uniform(0.95, 1.05)  # Daily variation
            )
            self.state.crew.productivity[i] = max(0, min(1, self.state.crew.productivity[i]))
        
        # Crew performance rewards
        avg_health = sum(self.state.crew.health) / len(self.state.crew.health)
        avg_stress = sum(self.state.crew.stress) / len(self.state.crew.stress)
        avg_productivity = sum(self.state.crew.productivity) / len(self.state.crew.productivity)
        
        # Health reward
        if avg_health > 0.9:
            reward += 1.0
        elif avg_health > 0.7:
            reward += 0.5
        elif avg_health < 0.5:
            reward -= (0.5 - avg_health) * 10
            self._add_safety_violation(f"Poor crew health: {avg_health:.2f}")
        elif avg_health < 0.3:
            reward -= (0.3 - avg_health) * 20
            self._add_safety_violation(f"Critical crew health: {avg_health:.2f}")
        
        # Stress penalty
        if avg_stress > 0.7:
            reward -= (avg_stress - 0.7) * 3
        
        # Productivity reward
        reward += avg_productivity * 0.5
        
        return reward
    
    def _clamp_state_values(self):
        """Ensure all state values remain within physically reasonable bounds."""
        # Atmosphere bounds
        self.state.atmosphere.o2_partial_pressure = max(0, min(50, self.state.atmosphere.o2_partial_pressure))
        self.state.atmosphere.co2_partial_pressure = max(0, min(10, self.state.atmosphere.co2_partial_pressure))
        self.state.atmosphere.n2_partial_pressure = max(0, min(100, self.state.atmosphere.n2_partial_pressure))
        self.state.atmosphere.humidity = max(0, min(100, self.state.atmosphere.humidity))
        self.state.atmosphere.temperature = max(-50, min(60, self.state.atmosphere.temperature))
        self.state.atmosphere.air_quality_index = max(0, min(1, self.state.atmosphere.air_quality_index))
        
        # Power bounds
        self.state.power.battery_charge = max(0, min(100, self.state.power.battery_charge))
        self.state.power.solar_generation = max(0, min(20, self.state.power.solar_generation))
        self.state.power.fuel_cell_capacity = max(0, min(100, self.state.power.fuel_cell_capacity))
        self.state.power.grid_stability = max(0, min(1, self.state.power.grid_stability))
        
        # Thermal bounds
        for i in range(len(self.state.thermal.internal_temp_zones)):
            self.state.thermal.internal_temp_zones[i] = max(-20, min(50, self.state.thermal.internal_temp_zones[i]))
        
        # Water bounds
        self.state.water.potable_water = max(0, min(2000, self.state.water.potable_water))
        self.state.water.grey_water = max(0, min(1000, self.state.water.grey_water))
        self.state.water.black_water = max(0, min(500, self.state.water.black_water))
        self.state.water.recycling_efficiency = max(0, min(1, self.state.water.recycling_efficiency))
        self.state.water.filter_status = max(0, min(1, self.state.water.filter_status))
    
    def _check_termination_robust(self) -> Tuple[bool, Optional[str]]:
        """Enhanced termination checking with detailed reasons."""
        # Critical atmosphere failure
        if self.state.atmosphere.o2_partial_pressure < 10:
            return True, f"Critical hypoxia: O2 {self.state.atmosphere.o2_partial_pressure:.1f} kPa"
        
        if self.state.atmosphere.co2_partial_pressure > 2.0:
            return True, f"CO2 poisoning: CO2 {self.state.atmosphere.co2_partial_pressure:.2f} kPa"
        
        # Power system failure
        if self.state.power.battery_charge <= 0 and self.state.power.solar_generation < 1.0:
            return True, "Total power failure"
        
        # Critical crew health
        critical_crew = sum(1 for h in self.state.crew.health if h < 0.2)
        if critical_crew >= len(self.state.crew.health) // 2:  # More than half crew critical
            return True, f"Crew medical emergency: {critical_crew} critical crew members"
        
        # Water depletion
        if self.state.water.potable_water <= 0:
            return True, "Water supply exhausted"
        
        # Temperature extremes
        avg_temp = sum(self.state.thermal.internal_temp_zones) / len(self.state.thermal.internal_temp_zones)
        if avg_temp < 10 or avg_temp > 40:
            return True, f"Extreme temperature: {avg_temp:.1f}°C"
        
        # Multiple safety violations
        if len(self.safety_violations) > 10:
            return True, f"Multiple safety violations: {len(self.safety_violations)}"
        
        # Physics simulation breakdown
        if self.physics_stability < 0.5:
            return True, f"Simulation instability: {self.physics_stability:.2f}"
        
        return False, None
    
    def _add_safety_violation(self, violation: str):
        """Add safety violation to tracking."""
        self.safety_violations.append({
            'step': self.current_step,
            'violation': violation,
            'timestamp': time.time()
        })
        
        # Emergency mode trigger
        if len(self.safety_violations) > 5:
            self.emergency_mode = True
            self.critical_alerts.append(violation)
    
    def _get_safe_default_action(self) -> Dict[str, Any]:
        """Get safe default action for error recovery."""
        return {
            'life_support': {
                'o2_generation_rate': 0.6,
                'co2_scrubber_power': 0.8,
                'n2_injection': 0.1,
                'air_circulation_speed': 0.7,
                'humidity_target': 0.45,
                'air_filter_mode': 2
            },
            'power': {
                'battery_charge_rate': 0.5,
                'load_shedding': [False] * 4,
                'fuel_cell_activation': 0.3,
                'solar_panel_angle': 0.0,
                'emergency_power_reserve': 0.8
            },
            'thermal': {
                'heating_zones': [0.5] * 4,
                'radiator_flow': [0.5] * 2,
                'heat_pump_mode': 1,
                'insulation_deployment': 0.8
            },
            'water': {
                'recycling_priority': 2,
                'purification_intensity': 0.8,
                'rationing_level': 0
            }
        }
    
    def _get_status(self) -> str:
        """Get enhanced habitat status."""
        if self.emergency_mode:
            return 'emergency'
        
        if self._check_termination_robust()[0]:
            return 'critical'
        
        # Count warning conditions
        warnings = 0
        if self.state.atmosphere.o2_partial_pressure < 18:
            warnings += 1
        if self.state.atmosphere.co2_partial_pressure > 0.6:
            warnings += 1
        if self.state.power.battery_charge < 30:
            warnings += 1
        if self.state.water.potable_water < 200:
            warnings += 1
        if self.physics_stability < 0.9:
            warnings += 1
        
        if warnings >= 3:
            return 'warning'
        elif warnings >= 1:
            return 'caution'
        else:
            return 'nominal'
    
    def _prepare_step_info(self, action_validation, obs_validation, termination_reason, 
                          step_start_time, parsed_action) -> Dict[str, Any]:
        """Prepare comprehensive step information."""
        step_duration = time.time() - step_start_time
        
        info = {
            'step': self.current_step,
            'total_reward': self.episode_reward,
            'status': self._get_status(),
            'step_duration_ms': step_duration * 1000,
            'physics_stability': self.physics_stability,
            'safety_violations': len(self.safety_violations),
            'validation_failures': self.validation_failures,
            'emergency_mode': self.emergency_mode
        }
        
        # Add validation results if there were issues
        if not action_validation.get('valid', True):
            info['action_validation'] = action_validation
        if obs_validation and not obs_validation.get('valid', True):
            info['observation_validation'] = obs_validation
        
        # Add termination reason if terminated
        if termination_reason:
            info['termination_reason'] = termination_reason
        
        # Add recent safety violations
        if self.safety_violations:
            info['recent_violations'] = self.safety_violations[-3:]
        
        # Add system health summary
        info['system_health'] = {
            'atmosphere': {
                'o2_pressure': self.state.atmosphere.o2_partial_pressure,
                'co2_pressure': self.state.atmosphere.co2_partial_pressure
            },
            'power': {
                'battery_charge': self.state.power.battery_charge,
                'solar_generation': self.state.power.solar_generation
            },
            'crew': {
                'avg_health': sum(self.state.crew.health) / len(self.state.crew.health),
                'avg_stress': sum(self.state.crew.stress) / len(self.state.crew.stress)
            }
        }
        
        return info
    
    def _parse_state_for_logging(self, observation: List[float]) -> Dict[str, Any]:
        """Parse observation for structured logging."""
        if len(observation) < 26:
            return {'error': 'Incomplete observation'}
        
        return {
            'atmosphere': {
                'o2_partial_pressure': observation[0],
                'co2_partial_pressure': observation[1],
                'temperature': observation[5]
            },
            'power': {
                'battery_charge': observation[8],
                'solar_generation': observation[7]
            },
            'thermal': {
                'avg_temperature': sum(observation[13:17]) / 4
            },
            'water': {
                'potable_water': observation[21]
            }
        }
    
    def _handle_episode_end(self, terminated: bool, truncated: bool, reason: Optional[str]):
        """Handle episode end with monitoring and logging."""
        if self.enable_monitoring:
            self.simulation_monitor.log_episode_end(self.episode_reward)
        
        end_type = "terminated" if terminated else "truncated"
        self.logger.info(
            f"Episode {self.episode_count} ended ({end_type}) - "
            f"steps: {self.current_step}, reward: {self.episode_reward:.2f}, "
            f"reason: {reason or 'max steps'}"
        )
        
        # Log episode statistics
        self.logger.log_performance(
            f"episode_{self.episode_count}",
            self.current_step,  # Use steps as duration proxy
            episode_reward=self.episode_reward,
            safety_violations=len(self.safety_violations),
            termination_reason=reason or "normal"
        )
    
    def _handle_critical_error(self, operation: str, error: Exception):
        """Handle critical errors that threaten simulation integrity."""
        self.logger.critical(f"Critical error in {operation}: {error}", error=error)
        self.emergency_mode = True
        self.physics_stability *= 0.8
        
        # Add to error history
        self.error_history.append({
            'operation': operation,
            'error': str(error),
            'step': self.current_step,
            'timestamp': time.time()
        })
    
    def _get_emergency_state(self) -> Tuple[List[float], Dict[str, Any]]:
        """Return emergency safe state when initialization fails."""
        safe_obs = ([21.3, 0.3, 79.0, 101.3, 45.0, 22.5, 0.95] +  # atmosphere
                   [8.5, 75.0, 90.0, 6.2, 100.0, 0.98] +  # power
                   [22.5, 23.1, 22.8, 21.9, -45.0, 15.0, 16.2, 3.2] +  # thermal
                   [850.0, 120.0, 45.0, 0.93, 0.87] + # water
                   [0.95] * self.crew_size + [0.3] * self.crew_size + [0.9] * self.crew_size + # crew
                   [127.5, 0.3, 0.15, 0.05])  # environment
        
        info = {
            'step': 0,
            'total_reward': 0.0,
            'crew_size': self.crew_size,
            'status': 'emergency',
            'error': 'Emergency state due to initialization failure'
        }
        
        return safe_obs, info
    
    def _get_emergency_step_result(self) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """Return emergency step result when step fails."""
        obs, info = self._get_emergency_state()
        return obs, -100.0, True, False, {**info, 'error': 'Emergency termination due to step failure'}
    
    def close(self):
        """Close the environment with cleanup."""
        try:
            self.logger.info(f"Closing RobustLunarHabitatEnv - completed {self.episode_count} episodes")
            
            # Final statistics
            if self.enable_monitoring:
                final_stats = self.simulation_monitor.get_simulation_metrics()
                self.logger.info(f"Final simulation statistics: {final_stats}")
            
            # Clean up resources
            self.safety_violations.clear()
            self.error_history.clear()
            
        except Exception as e:
            self.logger.error(f"Error during environment cleanup: {e}")


# Convenience classes for backward compatibility
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


def make_robust_lunar_env(n_envs: int = 1, safety_mode: str = "strict", 
                         enable_monitoring: bool = True, **kwargs):
    """Create robust lunar habitat environment(s).
    
    Args:
        n_envs: Number of environments to create.
        safety_mode: Safety validation mode ('strict', 'moderate', 'permissive').
        enable_monitoring: Whether to enable performance monitoring.
        **kwargs: Additional arguments for environment constructor.
        
    Returns:
        Single environment if n_envs=1, otherwise list of environments.
    """
    envs = []
    for i in range(n_envs):
        env = RobustLunarHabitatEnv(
            safety_mode=safety_mode,
            enable_monitoring=enable_monitoring,
            **kwargs
        )
        envs.append(env)
    
    return envs[0] if n_envs == 1 else envs