"""State representation and action space definitions for the habitat environment."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from gymnasium import spaces


@dataclass
class AtmosphereState:
    """Atmospheric conditions state."""
    o2_partial_pressure: float  # kPa
    co2_partial_pressure: float  # kPa
    n2_partial_pressure: float  # kPa
    total_pressure: float  # kPa
    humidity: float  # %
    temperature: float  # °C
    air_quality_index: float  # 0-1
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.o2_partial_pressure,
            self.co2_partial_pressure, 
            self.n2_partial_pressure,
            self.total_pressure,
            self.humidity,
            self.temperature,
            self.air_quality_index
        ], dtype=np.float32)


@dataclass
class PowerState:
    """Power system state."""
    solar_generation: float  # kW
    battery_charge: float  # %
    fuel_cell_capacity: float  # %
    total_load: float  # kW
    emergency_reserve: float  # %
    grid_stability: float  # 0-1
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.solar_generation,
            self.battery_charge,
            self.fuel_cell_capacity,
            self.total_load,
            self.emergency_reserve,
            self.grid_stability
        ], dtype=np.float32)


@dataclass  
class ThermalState:
    """Thermal system state."""
    internal_temp_zones: List[float]  # °C per zone
    external_temp: float  # °C
    radiator_temps: List[float]  # °C per radiator
    heat_pump_efficiency: float  # COP
    
    def to_array(self) -> np.ndarray:
        zones = np.array(self.internal_temp_zones[:4], dtype=np.float32)  # Max 4 zones
        if len(zones) < 4:
            zones = np.pad(zones, (0, 4 - len(zones)), constant_values=22.0)
            
        radiators = np.array(self.radiator_temps[:2], dtype=np.float32)  # Max 2 radiators
        if len(radiators) < 2:
            radiators = np.pad(radiators, (0, 2 - len(radiators)), constant_values=15.0)
            
        return np.concatenate([
            zones,
            [self.external_temp],
            radiators,
            [self.heat_pump_efficiency]
        ])


@dataclass
class WaterState:
    """Water management state."""
    potable_water: float  # liters
    grey_water: float  # liters  
    black_water: float  # liters
    recycling_efficiency: float  # 0-1
    filter_status: float  # 0-1
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.potable_water,
            self.grey_water,
            self.black_water, 
            self.recycling_efficiency,
            self.filter_status
        ], dtype=np.float32)


@dataclass
class CrewState:
    """Crew status and location state."""
    health: List[float]  # 0-1 per crew member
    stress: List[float]  # 0-1 per crew member
    productivity: List[float]  # 0-1 per crew member
    locations: List[str]  # Location per crew member
    
    def to_array(self, max_crew: int = 6) -> np.ndarray:
        """Convert to array format, padding for consistent size."""
        health = np.array(self.health[:max_crew], dtype=np.float32)
        stress = np.array(self.stress[:max_crew], dtype=np.float32)
        productivity = np.array(self.productivity[:max_crew], dtype=np.float32)
        
        # Pad arrays if needed
        if len(health) < max_crew:
            health = np.pad(health, (0, max_crew - len(health)), constant_values=0.95)
        if len(stress) < max_crew:
            stress = np.pad(stress, (0, max_crew - len(stress)), constant_values=0.3)
        if len(productivity) < max_crew:
            productivity = np.pad(productivity, (0, max_crew - len(productivity)), constant_values=0.9)
            
        return np.concatenate([health, stress, productivity])


@dataclass
class EnvironmentState:
    """External environment and time state.""" 
    mission_elapsed_time: float  # sols
    lunar_day_phase: float  # 0-1 (0=sunrise)
    dust_accumulation: float  # 0-1
    system_degradation: float  # 0-1
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.mission_elapsed_time,
            self.lunar_day_phase,
            self.dust_accumulation,
            self.system_degradation
        ], dtype=np.float32)


class HabitatState:
    """Complete habitat state representation."""
    
    def __init__(self, max_crew: int = 6):
        self.max_crew = max_crew
        self.atmosphere = AtmosphereState(
            o2_partial_pressure=21.3,
            co2_partial_pressure=0.4,
            n2_partial_pressure=79.0,
            total_pressure=101.3,
            humidity=45.0,
            temperature=22.5,
            air_quality_index=0.95
        )
        
        self.power = PowerState(
            solar_generation=8.5,
            battery_charge=75.0,
            fuel_cell_capacity=90.0,
            total_load=6.2,
            emergency_reserve=100.0,
            grid_stability=0.98
        )
        
        self.thermal = ThermalState(
            internal_temp_zones=[22.5, 23.1, 22.8, 21.9],
            external_temp=-45.0,
            radiator_temps=[15.0, 16.2],
            heat_pump_efficiency=3.2
        )
        
        self.water = WaterState(
            potable_water=850.0,
            grey_water=120.0,
            black_water=45.0,
            recycling_efficiency=0.93,
            filter_status=0.87
        )
        
        self.crew = CrewState(
            health=[0.95, 0.98, 0.92, 0.96],
            stress=[0.3, 0.2, 0.4, 0.25],
            productivity=[0.9, 0.95, 0.85, 0.92],
            locations=['lab', 'hab', 'airlock', 'hab']
        )
        
        self.environment = EnvironmentState(
            mission_elapsed_time=127.5,
            lunar_day_phase=0.3,
            dust_accumulation=0.15,
            system_degradation=0.05
        )
    
    def to_array(self) -> np.ndarray:
        """Convert complete state to flat numpy array."""
        return np.concatenate([
            self.atmosphere.to_array(),  # 7 dims
            self.power.to_array(),  # 6 dims
            self.thermal.to_array(),  # 8 dims  
            self.water.to_array(),  # 5 dims
            self.crew.to_array(self.max_crew),  # 18 dims (6 crew * 3 metrics)
            self.environment.to_array()  # 4 dims
        ])
    
    def get_observation_space(self) -> spaces.Box:
        """Get the gymnasium observation space for this state."""
        total_dims = 7 + 6 + 8 + 5 + (self.max_crew * 3) + 4  # 48 dims for 6 crew
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dims,),
            dtype=np.float32
        )


class ActionSpace:
    """Multi-dimensional continuous action space for habitat control."""
    
    def __init__(self, num_zones: int = 4, num_radiators: int = 2):
        self.num_zones = num_zones
        self.num_radiators = num_radiators
        
        # Action dimensions:
        # Life Support (6): o2_gen, co2_scrub, n2_inj, air_circ, humidity, filter_mode
        # Power (5): battery_rate, load_shed (4 zones), fuel_cell, solar_angle, emerg_reserve
        # Thermal (8): heating (4 zones), radiator_flow (2), heat_pump_mode, insulation 
        # Water (3): recycling_priority, purification_intensity, rationing_level
        
        self.life_support_dims = 6
        self.power_dims = 5 + num_zones  # +4 for load shedding per zone
        self.thermal_dims = num_zones + num_radiators + 2  # heating, radiator, pump, insulation
        self.water_dims = 3
        
        self.total_dims = self.life_support_dims + self.power_dims + self.thermal_dims + self.water_dims
    
    def get_action_space(self) -> spaces.Box:
        """Get the gymnasium action space."""
        # Most actions are continuous [0, 1], except solar angle [-90, 90]
        low = np.zeros(self.total_dims, dtype=np.float32)
        high = np.ones(self.total_dims, dtype=np.float32)
        
        # Solar panel angle is in degrees
        solar_angle_idx = self.life_support_dims + 4 + self.num_zones + 1  # After load shedding and fuel cell
        low[solar_angle_idx] = -90.0
        high[solar_angle_idx] = 90.0
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def parse_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Parse flat action array into structured action dictionary."""
        idx = 0
        
        # Life support actions
        life_support = {
            'o2_generation_rate': action[idx],
            'co2_scrubber_power': action[idx + 1], 
            'n2_injection': action[idx + 2],
            'air_circulation_speed': action[idx + 3],
            'humidity_target': action[idx + 4],
            'air_filter_mode': int(action[idx + 5] * 3)  # 0-3 discrete
        }
        idx += self.life_support_dims
        
        # Power actions
        power = {
            'battery_charge_rate': action[idx],
            'load_shedding': action[idx + 1:idx + 1 + self.num_zones].astype(bool),
            'fuel_cell_activation': action[idx + 1 + self.num_zones],
            'solar_panel_angle': action[idx + 2 + self.num_zones],
            'emergency_power_reserve': action[idx + 3 + self.num_zones]
        }
        idx += self.power_dims
        
        # Thermal actions
        thermal = {
            'heating_zones': action[idx:idx + self.num_zones],
            'radiator_flow': action[idx + self.num_zones:idx + self.num_zones + self.num_radiators],
            'heat_pump_mode': int(action[idx + self.num_zones + self.num_radiators] * 2),  # 0-2
            'insulation_deployment': action[idx + self.num_zones + self.num_radiators + 1]
        }
        idx += self.thermal_dims
        
        # Water actions
        water = {
            'recycling_priority': int(action[idx] * 3),  # 0-3 discrete
            'purification_intensity': action[idx + 1],
            'rationing_level': int(action[idx + 2] * 3)  # 0-3 discrete
        }
        
        return {
            'life_support': life_support,
            'power': power,
            'thermal': thermal,
            'water': water
        }