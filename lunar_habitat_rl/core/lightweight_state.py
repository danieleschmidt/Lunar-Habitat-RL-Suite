"""Lightweight state representation without numpy/gymnasium dependencies - Generation 1"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class AtmosphereState:
    """Atmospheric conditions state."""
    o2_partial_pressure: float = 21.3  # kPa
    co2_partial_pressure: float = 0.4  # kPa
    n2_partial_pressure: float = 79.0  # kPa
    total_pressure: float = 101.3  # kPa
    humidity: float = 45.0  # %
    temperature: float = 22.5  # °C
    air_quality_index: float = 0.95  # 0-1
    
    def to_array(self) -> List[float]:
        """Convert to list for lightweight implementation."""
        return [
            self.o2_partial_pressure,
            self.co2_partial_pressure, 
            self.n2_partial_pressure,
            self.total_pressure,
            self.humidity,
            self.temperature,
            self.air_quality_index
        ]


@dataclass
class PowerState:
    """Power system state."""
    solar_generation: float = 8.5  # kW
    battery_charge: float = 75.0  # %
    fuel_cell_capacity: float = 90.0  # %
    total_load: float = 6.2  # kW
    emergency_reserve: float = 100.0  # %
    grid_stability: float = 0.98  # 0-1
    
    def to_array(self) -> List[float]:
        return [
            self.solar_generation,
            self.battery_charge,
            self.fuel_cell_capacity,
            self.total_load,
            self.emergency_reserve,
            self.grid_stability
        ]


@dataclass  
class ThermalState:
    """Thermal system state."""
    internal_temp_zones: List[float] = None  # °C per zone
    external_temp: float = -45.0  # °C
    radiator_temps: List[float] = None  # °C per radiator
    heat_pump_efficiency: float = 3.2  # COP
    
    def __post_init__(self):
        if self.internal_temp_zones is None:
            self.internal_temp_zones = [22.5, 23.1, 22.8, 21.9]
        if self.radiator_temps is None:
            self.radiator_temps = [15.0, 16.2]
    
    def to_array(self) -> List[float]:
        zones = self.internal_temp_zones[:4]  # Max 4 zones
        while len(zones) < 4:
            zones.append(22.0)  # Pad with default
            
        radiators = self.radiator_temps[:2]  # Max 2 radiators
        while len(radiators) < 2:
            radiators.append(15.0)  # Pad with default
            
        return zones + [self.external_temp] + radiators + [self.heat_pump_efficiency]


@dataclass
class WaterState:
    """Water management state."""
    potable_water: float = 850.0  # liters
    grey_water: float = 120.0  # liters  
    black_water: float = 45.0  # liters
    recycling_efficiency: float = 0.93  # 0-1
    filter_status: float = 0.87  # 0-1
    
    def to_array(self) -> List[float]:
        return [
            self.potable_water,
            self.grey_water,
            self.black_water, 
            self.recycling_efficiency,
            self.filter_status
        ]


@dataclass
class CrewState:
    """Crew status and location state."""
    health: List[float] = None  # 0-1 per crew member
    stress: List[float] = None  # 0-1 per crew member
    productivity: List[float] = None  # 0-1 per crew member
    locations: List[str] = None  # Location per crew member
    
    def __post_init__(self):
        if self.health is None:
            self.health = [0.95, 0.98, 0.92, 0.96]
        if self.stress is None:
            self.stress = [0.3, 0.2, 0.4, 0.25]
        if self.productivity is None:
            self.productivity = [0.9, 0.95, 0.85, 0.92]
        if self.locations is None:
            self.locations = ['lab', 'hab', 'airlock', 'hab']
    
    def to_array(self, max_crew: int = 6) -> List[float]:
        """Convert to array format, padding for consistent size."""
        health = self.health[:max_crew]
        stress = self.stress[:max_crew]
        productivity = self.productivity[:max_crew]
        
        # Pad arrays if needed
        while len(health) < max_crew:
            health.append(0.95)
        while len(stress) < max_crew:
            stress.append(0.3)
        while len(productivity) < max_crew:
            productivity.append(0.9)
            
        return health + stress + productivity


@dataclass
class EnvironmentState:
    """External environment and time state.""" 
    mission_elapsed_time: float = 127.5  # sols
    lunar_day_phase: float = 0.3  # 0-1 (0=sunrise)
    dust_accumulation: float = 0.15  # 0-1
    system_degradation: float = 0.05  # 0-1
    
    def to_array(self) -> List[float]:
        return [
            self.mission_elapsed_time,
            self.lunar_day_phase,
            self.dust_accumulation,
            self.system_degradation
        ]


class HabitatState:
    """Complete habitat state representation - lightweight implementation."""
    
    def __init__(self, max_crew: int = 6):
        self.max_crew = max_crew
        self.atmosphere = AtmosphereState()
        self.power = PowerState()
        self.thermal = ThermalState()
        self.water = WaterState()
        self.crew = CrewState()
        self.environment = EnvironmentState()
    
    def to_array(self) -> List[float]:
        """Convert complete state to flat list."""
        result = []
        result.extend(self.atmosphere.to_array())  # 7 dims
        result.extend(self.power.to_array())  # 6 dims
        result.extend(self.thermal.to_array())  # 8 dims  
        result.extend(self.water.to_array())  # 5 dims
        result.extend(self.crew.to_array(self.max_crew))  # 18 dims (6 crew * 3 metrics)
        result.extend(self.environment.to_array())  # 4 dims
        return result
    
    def get_observation_space_info(self) -> Dict[str, int]:
        """Get information about the observation space."""
        return {
            'total_dims': 7 + 6 + 8 + 5 + (self.max_crew * 3) + 4,
            'atmosphere_dims': 7,
            'power_dims': 6,
            'thermal_dims': 8,
            'water_dims': 5,
            'crew_dims': self.max_crew * 3,
            'environment_dims': 4
        }


class ActionSpace:
    """Multi-dimensional continuous action space - lightweight implementation."""
    
    def __init__(self, num_zones: int = 4, num_radiators: int = 2):
        self.num_zones = num_zones
        self.num_radiators = num_radiators
        
        self.life_support_dims = 6
        self.power_dims = 5 + num_zones  # +4 for load shedding per zone
        self.thermal_dims = num_zones + num_radiators + 2  # heating, radiator, pump, insulation
        self.water_dims = 3
        
        self.total_dims = self.life_support_dims + self.power_dims + self.thermal_dims + self.water_dims
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space."""
        return {
            'total_dims': self.total_dims,
            'life_support_dims': self.life_support_dims,
            'power_dims': self.power_dims,
            'thermal_dims': self.thermal_dims,
            'water_dims': self.water_dims,
            'bounds': {
                'low': [0.0] * self.total_dims,
                'high': [1.0] * self.total_dims  # Most actions 0-1, except solar angle
            }
        }
    
    def parse_action(self, action: List[float]) -> Dict[str, Any]:
        """Parse flat action list into structured action dictionary."""
        if len(action) != self.total_dims:
            raise ValueError(f"Action must have {self.total_dims} dimensions, got {len(action)}")
            
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
            'load_shedding': [bool(x > 0.5) for x in action[idx + 1:idx + 1 + self.num_zones]],
            'fuel_cell_activation': action[idx + 1 + self.num_zones],
            'solar_panel_angle': action[idx + 2 + self.num_zones] * 180 - 90,  # Map [0,1] to [-90,90]
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