"""Thermal simulation engine for habitat temperature modeling."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThermalProperties:
    """Material thermal properties."""
    conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K  
    density: float  # kg/m³
    emissivity: float  # 0-1
    absorptivity: float  # 0-1


@dataclass
class ThermalZone:
    """Thermal zone representation."""
    volume: float  # m³
    surface_area: float  # m²
    thermal_mass: float  # J/K
    current_temp: float  # °C
    target_temp: float  # °C
    heat_sources: List[float]  # W
    

class ThermalSimulator:
    """
    High-fidelity thermal simulation for lunar habitat.
    
    Models heat transfer through conduction, convection, and radiation,
    considering the extreme lunar thermal environment.
    """
    
    def __init__(self,
                 habitat_volume: float = 200.0,  # m³
                 thermal_mass: float = 5000.0,  # kg
                 insulation_r_value: float = 10.0,  # m²K/W
                 solver: str = "finite_difference"):
        """
        Initialize thermal simulator.
        
        Args:
            habitat_volume: Total internal volume
            thermal_mass: Total thermal mass of habitat
            insulation_r_value: Thermal resistance of insulation
            solver: Solver type ('finite_difference', 'finite_element')
        """
        
        self.habitat_volume = habitat_volume
        self.thermal_mass = thermal_mass
        self.insulation_r_value = insulation_r_value
        self.solver = solver
        
        # Material properties
        self.materials = {
            'aluminum': ThermalProperties(237.0, 897.0, 2700.0, 0.05, 0.15),
            'regolith_insulation': ThermalProperties(0.1, 800.0, 1000.0, 0.9, 0.9),
            'air': ThermalProperties(0.026, 1005.0, 1.2, 0.0, 0.0),
        }
        
        # Initialize thermal zones (4 zones default)
        self.zones = self._initialize_zones()
        
        # Stefan-Boltzmann constant
        self.sigma = 5.67e-8  # W/m²·K⁴
        
        # Simulation state
        self.current_time = 0.0
        self.last_step_size = 60.0  # Default 1 minute
        
        logger.info(f"ThermalSimulator initialized with {len(self.zones)} zones, "
                   f"solver: {solver}")
    
    def _initialize_zones(self) -> List[ThermalZone]:
        """Initialize thermal zones with default properties."""
        zone_volume = self.habitat_volume / 4
        zone_thermal_mass = self.thermal_mass / 4
        
        zones = []
        for i in range(4):
            zone = ThermalZone(
                volume=zone_volume,
                surface_area=50.0,  # m² (estimated)
                thermal_mass=zone_thermal_mass,  # J/K
                current_temp=22.0,  # °C
                target_temp=22.0,  # °C
                heat_sources=[100.0]  # W (baseline equipment)
            )
            zones.append(zone)
            
        return zones
    
    def step(self,
             dt: float,
             external_temp: float,
             internal_heat_generation: float,
             heating_power: List[float],
             radiator_flow: List[float]) -> Dict[str, Any]:
        """
        Advance thermal simulation by one timestep.
        
        Args:
            dt: Timestep in seconds
            external_temp: External lunar surface temperature (°C)
            internal_heat_generation: Total internal heat generation (W)
            heating_power: Heating power per zone (0-1 normalized)
            radiator_flow: Radiator coolant flow rates (0-1 normalized)
            
        Returns:
            Dictionary with updated temperatures and thermal metrics
        """
        
        self.last_step_size = dt
        
        if self.solver == "finite_difference":
            return self._step_finite_difference(
                dt, external_temp, internal_heat_generation, heating_power, radiator_flow
            )
        else:
            # Fallback to simplified model
            return self._step_simplified(
                dt, external_temp, internal_heat_generation, heating_power, radiator_flow
            )
    
    def _step_finite_difference(self,
                               dt: float,
                               external_temp: float,
                               internal_heat_generation: float,
                               heating_power: List[float],
                               radiator_flow: List[float]) -> Dict[str, Any]:
        """Finite difference thermal solver."""
        
        # Convert to Kelvin for calculations
        T_ext = external_temp + 273.15
        
        # Update each zone
        new_zone_temps = []
        radiator_temps = []
        
        for i, zone in enumerate(self.zones):
            T_zone = zone.current_temp + 273.15
            
            # Heat generation in this zone
            zone_heat_gen = internal_heat_generation / len(self.zones)
            heater_power = heating_power[i] * 1000.0 if i < len(heating_power) else 0.0  # W
            
            # Heat losses/gains
            heat_conduction = self._compute_conduction_loss(T_zone, T_ext)
            heat_radiation = self._compute_radiation_loss(T_zone, T_ext)
            heat_convection = self._compute_convection_loss(T_zone, T_ext)
            
            # Inter-zone heat transfer
            heat_mixing = self._compute_zone_mixing(i, T_zone)
            
            # Net heat change
            net_heat = (zone_heat_gen + heater_power - 
                       heat_conduction - heat_radiation - heat_convection + heat_mixing)
            
            # Temperature change
            dT = (net_heat * dt) / zone.thermal_mass
            new_temp_k = T_zone + dT
            new_temp_c = new_temp_k - 273.15
            
            # Clamp to reasonable bounds
            new_temp_c = np.clip(new_temp_c, -100.0, 80.0)
            
            zone.current_temp = new_temp_c
            new_zone_temps.append(new_temp_c)
        
        # Compute radiator temperatures
        for i, flow_rate in enumerate(radiator_flow[:2]):  # Max 2 radiators
            if flow_rate > 0.1:
                # Active cooling
                avg_zone_temp = np.mean(new_zone_temps)
                coolant_temp = avg_zone_temp - 5.0 - (flow_rate * 10.0)  # Cooling effect
                radiator_temp = max(-50.0, coolant_temp)
            else:
                # Passive radiation to space
                radiator_temp = min(new_zone_temps) - 20.0
                
            radiator_temps.append(radiator_temp)
        
        # Compute heat pump COP based on temperature differential
        avg_temp = np.mean(new_zone_temps)
        temp_diff = abs(avg_temp - external_temp)
        heat_pump_cop = self._compute_heat_pump_cop(temp_diff)
        
        self.current_time += dt
        
        return {
            'zone_temperatures': new_zone_temps,
            'radiator_temperatures': radiator_temps,
            'heat_pump_cop': heat_pump_cop,
            'total_heat_loss': sum([
                self._compute_conduction_loss(t + 273.15, T_ext) 
                for t in new_zone_temps
            ]),
            'average_temperature': avg_temp,
            'temperature_variance': np.var(new_zone_temps)
        }
    
    def _step_simplified(self,
                        dt: float,
                        external_temp: float,
                        internal_heat_generation: float,
                        heating_power: List[float],
                        radiator_flow: List[float]) -> Dict[str, Any]:
        """Simplified thermal model for faster simulation."""
        
        # Simple thermal time constant model
        thermal_time_constant = 3600.0  # 1 hour in seconds
        
        new_zone_temps = []
        
        for i, zone in enumerate(self.zones):
            # Target temperature based on heating
            heater_power = heating_power[i] * 3000.0 if i < len(heating_power) else 0.0  # W
            
            # Heat balance
            heat_input = internal_heat_generation / len(self.zones) + heater_power
            heat_loss_coeff = 500.0  # W/K (simplified)
            
            equilibrium_temp = external_temp + (heat_input / heat_loss_coeff)
            
            # Exponential approach to equilibrium
            temp_diff = equilibrium_temp - zone.current_temp
            alpha = 1.0 - np.exp(-dt / thermal_time_constant)
            
            new_temp = zone.current_temp + alpha * temp_diff
            
            # Apply limits
            new_temp = np.clip(new_temp, external_temp - 5.0, 50.0)
            
            zone.current_temp = new_temp
            new_zone_temps.append(new_temp)
        
        # Simple radiator model
        radiator_temps = []
        for i, flow_rate in enumerate(radiator_flow[:2]):
            avg_temp = np.mean(new_zone_temps)
            radiator_temp = avg_temp - (flow_rate * 15.0)  # Cooling effect
            radiator_temps.append(radiator_temp)
            
        heat_pump_cop = self._compute_heat_pump_cop(abs(np.mean(new_zone_temps) - external_temp))
        
        return {
            'zone_temperatures': new_zone_temps,
            'radiator_temperatures': radiator_temps,
            'heat_pump_cop': heat_pump_cop,
            'total_heat_loss': 1000.0,  # Placeholder
            'average_temperature': np.mean(new_zone_temps),
            'temperature_variance': np.var(new_zone_temps)
        }
    
    def _compute_conduction_loss(self, T_inside: float, T_outside: float) -> float:
        """Compute conductive heat loss through insulation."""
        # Q = (T_hot - T_cold) / R_thermal
        # R_thermal = thickness / (k * A) = R_value / A
        
        surface_area = 300.0  # m² (estimated total surface)
        R_total = self.insulation_r_value / surface_area  # K/W
        
        heat_loss = (T_inside - T_outside) / R_total  # W
        
        return max(0.0, heat_loss)
    
    def _compute_radiation_loss(self, T_inside: float, T_outside: float) -> float:
        """Compute radiative heat loss to space."""
        # Stefan-Boltzmann law: Q = σ * ε * A * (T₁⁴ - T₂⁴)
        
        surface_area = 50.0  # m² (external radiating surface)
        emissivity = self.materials['aluminum'].emissivity
        
        heat_loss = (self.sigma * emissivity * surface_area * 
                    (T_inside**4 - T_outside**4))  # W
        
        return max(0.0, heat_loss)
    
    def _compute_convection_loss(self, T_inside: float, T_outside: float) -> float:
        """Compute convective heat loss (minimal in lunar vacuum)."""
        # Minimal convection in vacuum - only internal air circulation
        return 0.0
    
    def _compute_zone_mixing(self, zone_idx: int, zone_temp: float) -> float:
        """Compute heat transfer between zones due to air mixing."""
        heat_transfer = 0.0
        mixing_coeff = 50.0  # W/K (air circulation effectiveness)
        
        for i, other_zone in enumerate(self.zones):
            if i != zone_idx:
                other_temp = other_zone.current_temp + 273.15
                temp_diff = other_temp - zone_temp
                heat_transfer += mixing_coeff * temp_diff / len(self.zones)
        
        return heat_transfer
    
    def _compute_heat_pump_cop(self, temp_diff: float) -> float:
        """Compute heat pump coefficient of performance."""
        # Idealized Carnot COP with practical efficiency factor
        if temp_diff < 1.0:
            return 4.0  # High COP for small temperature differences
            
        # COP = T_hot / (T_hot - T_cold) * efficiency
        efficiency = 0.6  # Practical heat pump efficiency
        carnot_cop = 20.0 / temp_diff  # Simplified
        
        practical_cop = carnot_cop * efficiency
        
        # Clamp to realistic range
        return np.clip(practical_cop, 1.0, 6.0)
    
    def create_scenario(self,
                       external_temp_profile: str,
                       internal_heat_sources: Dict[str, float],
                       duration: float = 86400) -> Dict[str, Any]:
        """
        Create thermal scenario configuration.
        
        Args:
            external_temp_profile: Temperature profile name or data
            internal_heat_sources: Heat source definitions
            duration: Scenario duration in seconds
            
        Returns:
            Scenario configuration dictionary
        """
        
        # Load or generate temperature profile
        if external_temp_profile == "14_day_lunar_cycle":
            times = np.linspace(0, 14 * 24 * 3600, 1000)  # 14 days
            temps = -50.0 + 77.0 * np.sin(2 * np.pi * times / (14 * 24 * 3600))  # -50 to +127°C
        elif external_temp_profile == "constant_cold":
            times = np.linspace(0, duration, 100)
            temps = np.full_like(times, -180.0)  # Constant cold
        else:
            # Default profile
            times = np.linspace(0, duration, 100)
            temps = -50.0 + 20.0 * np.sin(2 * np.pi * times / 86400)  # Daily variation
        
        scenario = {
            'duration': duration,
            'time_profile': times,
            'temperature_profile': temps,
            'internal_heat_sources': internal_heat_sources,
            'initial_conditions': {
                'zone_temperatures': [22.0] * len(self.zones),
                'external_temperature': temps[0]
            }
        }
        
        return scenario
    
    def run(self, scenario: Dict[str, Any], timestep: float = 60.0) -> Dict[str, Any]:
        """
        Run thermal simulation for a complete scenario.
        
        Args:
            scenario: Scenario configuration from create_scenario()
            timestep: Simulation timestep in seconds
            
        Returns:
            Complete simulation results
        """
        
        times = scenario['time_profile']
        temps = scenario['temperature_profile']
        duration = scenario['duration']
        
        # Initialize results storage
        results = {
            'times': [],
            'zone_temperatures': [[] for _ in self.zones],
            'external_temperatures': [],
            'average_temperatures': [],
            'heat_losses': [],
            'heat_pump_cops': []
        }
        
        # Reset simulator state
        initial_temps = scenario['initial_conditions']['zone_temperatures']
        for i, zone in enumerate(self.zones):
            zone.current_temp = initial_temps[i]
        
        current_time = 0.0
        
        while current_time < duration:
            # Interpolate external temperature
            ext_temp = np.interp(current_time, times, temps)
            
            # Run simulation step
            internal_heat = sum(scenario['internal_heat_sources'].values())
            heating_power = [0.5] * len(self.zones)  # Moderate heating
            radiator_flow = [0.8, 0.8]  # Active cooling
            
            step_result = self.step(
                dt=timestep,
                external_temp=ext_temp,
                internal_heat_generation=internal_heat,
                heating_power=heating_power,
                radiator_flow=radiator_flow
            )
            
            # Store results
            results['times'].append(current_time)
            results['external_temperatures'].append(ext_temp)
            results['average_temperatures'].append(step_result['average_temperature'])
            results['heat_losses'].append(step_result['total_heat_loss'])
            results['heat_pump_cops'].append(step_result['heat_pump_cop'])
            
            for i, temp in enumerate(step_result['zone_temperatures']):
                results['zone_temperatures'][i].append(temp)
            
            current_time += timestep
            
        logger.info(f"Thermal simulation completed: {current_time:.1f}s simulated in {len(results['times'])} steps")
        
        return results
    
    def visualize_heatmap(self, results: Dict[str, Any], output_path: Optional[str] = None):
        """
        Visualize thermal simulation results as heatmap.
        
        Args:
            results: Simulation results from run()
            output_path: Optional file path to save visualization
        """
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Lunar Habitat Thermal Analysis')
            
            times_hours = np.array(results['times']) / 3600.0  # Convert to hours
            
            # Zone temperature evolution
            ax1 = axes[0, 0]
            for i, zone_temps in enumerate(results['zone_temperatures']):
                ax1.plot(times_hours, zone_temps, label=f'Zone {i+1}')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Temperature (°C)')
            ax1.set_title('Zone Temperatures')
            ax1.legend()
            ax1.grid(True)
            
            # External vs internal temperature
            ax2 = axes[0, 1]
            ax2.plot(times_hours, results['external_temperatures'], 'r-', label='External')
            ax2.plot(times_hours, results['average_temperatures'], 'b-', label='Internal Avg')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Temperature (°C)')
            ax2.set_title('Internal vs External Temperature')
            ax2.legend()
            ax2.grid(True)
            
            # Heat loss over time
            ax3 = axes[1, 0]
            ax3.plot(times_hours, results['heat_losses'])
            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Heat Loss (W)')
            ax3.set_title('Total Heat Loss')
            ax3.grid(True)
            
            # Heat pump performance
            ax4 = axes[1, 1]
            ax4.plot(times_hours, results['heat_pump_cops'])
            ax4.set_xlabel('Time (hours)')
            ax4.set_ylabel('COP')
            ax4.set_title('Heat Pump Coefficient of Performance')
            ax4.grid(True)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Thermal visualization saved to {output_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for visualization")