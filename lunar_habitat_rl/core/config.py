"""Configuration classes for habitat, crew, and scenario parameters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, List, Optional, Union, Any
try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    from lightweight_pydantic import BaseModel, Field, validator

try:
    import numpy as np
except ImportError:
    class MockNumpy:
        @staticmethod
        def array(x): return x
    np = MockNumpy()


class CrewConfig(BaseModel):
    """Configuration for crew parameters and behavior."""
    
    size: int = Field(default=4, ge=1, le=12, description="Number of crew members")
    health_variance: float = Field(default=0.05, ge=0.0, le=1.0, description="Health variance factor")
    stress_sensitivity: float = Field(default=0.3, ge=0.0, le=1.0, description="Stress response factor") 
    productivity_base: float = Field(default=0.9, ge=0.0, le=1.0, description="Base productivity level")
    metabolic_rate: float = Field(default=100.0, ge=50.0, le=200.0, description="Watts per person")
    
    @validator('size')
    def validate_crew_size(cls, v):
        if v not in range(1, 13):
            raise ValueError('Crew size must be between 1 and 12')
        return v


class PhysicsConfig(BaseModel):
    """Configuration for physics simulation parameters."""
    
    thermal_enabled: bool = Field(default=True, description="Enable thermal simulation")
    fluid_enabled: bool = Field(default=True, description="Enable fluid dynamics")
    chemistry_enabled: bool = Field(default=True, description="Enable chemical reactions")
    
    # Thermal parameters
    thermal_timestep: float = Field(default=60.0, ge=1.0, description="Thermal sim timestep (seconds)")
    thermal_solver: str = Field(default="finite_difference", description="Thermal solver type")
    
    # Fluid parameters  
    fluid_timestep: float = Field(default=1.0, ge=0.1, description="Fluid sim timestep (seconds)")
    turbulence_model: str = Field(default="k_epsilon", description="Turbulence model")
    
    # Chemistry parameters
    chemistry_timestep: float = Field(default=1.0, ge=0.1, description="Chemistry timestep (seconds)")
    reaction_database: str = Field(default="nist", description="Reaction database to use")
    
    @validator('thermal_solver')
    def validate_thermal_solver(cls, v):
        valid_solvers = ["finite_difference", "finite_element", "spectral"]
        if v not in valid_solvers:
            raise ValueError(f'Thermal solver must be one of {valid_solvers}')
        return v


class ScenarioConfig(BaseModel):
    """Configuration for mission scenarios and events."""
    
    name: str = Field(default="nominal_operations", description="Scenario name")
    duration_days: int = Field(default=30, ge=1, le=1000, description="Mission duration in days")
    difficulty: str = Field(default="nominal", description="Difficulty level")
    location: str = Field(default="lunar_south_pole", description="Mission location")
    
    # Environmental conditions
    external_temp_profile: str = Field(default="14_day_lunar_cycle", description="Temperature profile")
    dust_level: float = Field(default=0.1, ge=0.0, le=1.0, description="Dust accumulation rate")
    micrometeorite_rate: float = Field(default=1e-6, ge=0.0, description="Impacts per day")
    
    # Scheduled events
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Scheduled events")
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        valid_difficulties = ["easy", "nominal", "hard", "extreme"]
        if v not in valid_difficulties:
            raise ValueError(f'Difficulty must be one of {valid_difficulties}')
        return v


class HabitatConfig(BaseModel):
    """Main configuration class for the lunar habitat."""
    
    # Basic parameters
    name: str = Field(default="nasa_reference_habitat", description="Habitat configuration name")
    volume: float = Field(default=200.0, ge=10.0, description="Internal volume (m³)")
    pressure_nominal: float = Field(default=101.3, ge=50.0, le=120.0, description="Nominal pressure (kPa)")
    
    # Atmosphere composition (partial pressures in kPa)
    o2_nominal: float = Field(default=21.3, ge=16.0, le=30.0, description="O₂ partial pressure")
    co2_limit: float = Field(default=0.4, ge=0.1, le=1.0, description="CO₂ limit")
    n2_nominal: float = Field(default=79.0, ge=60.0, le=85.0, description="N₂ partial pressure")
    
    # Thermal parameters
    temp_nominal: float = Field(default=22.5, ge=18.0, le=28.0, description="Nominal temperature (°C)")
    temp_tolerance: float = Field(default=3.0, ge=1.0, le=5.0, description="Temperature tolerance")
    
    # Power system
    solar_capacity: float = Field(default=10.0, ge=1.0, description="Solar array capacity (kW)")
    battery_capacity: float = Field(default=50.0, ge=10.0, description="Battery capacity (kWh)")
    fuel_cell_capacity: float = Field(default=5.0, ge=0.0, description="Fuel cell capacity (kW)")
    
    # Water system
    water_storage: float = Field(default=1000.0, ge=100.0, description="Water storage (liters)")
    recycling_efficiency: float = Field(default=0.93, ge=0.8, le=0.99, description="Water recycling efficiency")
    
    # Equipment parameters
    pump_efficiency: float = Field(default=0.85, ge=0.7, le=0.95, description="Pump efficiency")
    fan_power: float = Field(default=100.0, ge=50.0, le=500.0, description="Fan power (W)")
    heater_capacity: float = Field(default=3000.0, ge=1000.0, description="Heater capacity (W)")
    
    # Safety limits
    emergency_o2_reserve: float = Field(default=72.0, ge=24.0, description="Emergency O₂ hours")
    emergency_power_reserve: float = Field(default=48.0, ge=12.0, description="Emergency power hours")
    
    # Sub-configurations
    crew: CrewConfig = Field(default_factory=CrewConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig) 
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    
    @validator('pressure_nominal')
    def validate_pressure(cls, v, values):
        if 'o2_nominal' in values and 'n2_nominal' in values:
            total_partial = values['o2_nominal'] + values['n2_nominal']
            if abs(v - total_partial) > 5.0:
                raise ValueError('Total pressure should approximately equal sum of partial pressures')
        return v
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return self.dict()
        
    @classmethod
    def from_preset(cls, preset: str) -> 'HabitatConfig':
        """Load configuration from predefined preset."""
        presets = {
            "nasa_reference": cls(),
            "apollo_derived": cls(
                volume=150.0,
                pressure_nominal=68.9,  # 10 psi
                o2_nominal=68.9,
                n2_nominal=0.0
            ),
            "mars_analog": cls(
                volume=300.0,
                co2_limit=0.6,
                temp_nominal=20.0
            )
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
            
        return presets[preset]