"""Lightweight configuration classes without Pydantic - Generation 1"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class CrewConfig:
    """Lightweight configuration for crew parameters."""
    
    size: int = 4
    health_variance: float = 0.05
    stress_sensitivity: float = 0.3
    productivity_base: float = 0.9
    metabolic_rate: float = 100.0


@dataclass
class PhysicsConfig:
    """Lightweight configuration for physics simulation."""
    
    thermal_enabled: bool = True
    fluid_enabled: bool = True
    chemistry_enabled: bool = True
    thermal_timestep: float = 60.0
    thermal_solver: str = "finite_difference"
    fluid_timestep: float = 1.0
    turbulence_model: str = "k_epsilon"
    chemistry_timestep: float = 1.0
    reaction_database: str = "nist"


@dataclass
class ScenarioConfig:
    """Lightweight configuration for mission scenarios."""
    
    name: str = "nominal_operations"
    duration_days: int = 30
    difficulty: str = "nominal"
    location: str = "lunar_south_pole"
    external_temp_profile: str = "14_day_lunar_cycle"
    dust_level: float = 0.1
    micrometeorite_rate: float = 1e-6
    events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []


@dataclass
class HabitatConfig:
    """Lightweight main configuration for the lunar habitat."""
    
    # Basic parameters
    name: str = "nasa_reference_habitat"
    volume: float = 200.0
    pressure_nominal: float = 101.3
    
    # Atmosphere composition (partial pressures in kPa)
    o2_nominal: float = 21.3
    co2_limit: float = 0.4
    n2_nominal: float = 79.0
    
    # Thermal parameters
    temp_nominal: float = 22.5
    temp_tolerance: float = 3.0
    
    # Power system
    solar_capacity: float = 10.0
    battery_capacity: float = 50.0
    fuel_cell_capacity: float = 5.0
    
    # Water system
    water_storage: float = 1000.0
    recycling_efficiency: float = 0.93
    
    # Equipment parameters
    pump_efficiency: float = 0.85
    fan_power: float = 100.0
    heater_capacity: float = 3000.0
    
    # Safety limits
    emergency_o2_reserve: float = 72.0
    emergency_power_reserve: float = 48.0
    
    # Sub-configurations
    crew: CrewConfig = None
    physics: PhysicsConfig = None
    scenario: ScenarioConfig = None
    
    def __post_init__(self):
        if self.crew is None:
            self.crew = CrewConfig()
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.scenario is None:
            self.scenario = ScenarioConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'name': self.name,
            'volume': self.volume,
            'pressure_nominal': self.pressure_nominal,
            'o2_nominal': self.o2_nominal,
            'co2_limit': self.co2_limit,
            'n2_nominal': self.n2_nominal,
            'temp_nominal': self.temp_nominal,
            'temp_tolerance': self.temp_tolerance,
            'solar_capacity': self.solar_capacity,
            'battery_capacity': self.battery_capacity,
            'fuel_cell_capacity': self.fuel_cell_capacity,
            'water_storage': self.water_storage,
            'recycling_efficiency': self.recycling_efficiency,
            'pump_efficiency': self.pump_efficiency,
            'fan_power': self.fan_power,
            'heater_capacity': self.heater_capacity,
            'emergency_o2_reserve': self.emergency_o2_reserve,
            'emergency_power_reserve': self.emergency_power_reserve,
            'crew': self.crew.__dict__,
            'physics': self.physics.__dict__,
            'scenario': self.scenario.__dict__
        }
    
    @classmethod
    def from_preset(cls, preset: str) -> 'HabitatConfig':
        """Load configuration from predefined preset."""
        if preset == "nasa_reference":
            return cls()
        elif preset == "apollo_derived":
            return cls(
                volume=150.0,
                pressure_nominal=68.9,  # 10 psi
                o2_nominal=68.9,
                n2_nominal=0.0
            )
        elif preset == "mars_analog":
            return cls(
                volume=300.0,
                co2_limit=0.6,
                temp_nominal=20.0
            )
        else:
            raise ValueError(f"Unknown preset: {preset}. Available: nasa_reference, apollo_derived, mars_analog")