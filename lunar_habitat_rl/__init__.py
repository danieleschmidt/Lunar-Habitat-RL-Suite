"""
Lunar Habitat RL Suite - Reinforcement Learning Environment for Autonomous Life Support Systems

A comprehensive benchmarking suite for training and evaluating RL agents on lunar habitat
environmental control tasks, featuring high-fidelity physics simulation and NASA-validated scenarios.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Try to import full implementations first, fallback to lightweight
try:
    from .environments import LunarHabitatEnv, make_lunar_env
    from .core import HabitatConfig, CrewConfig, ScenarioConfig
    from .physics import ThermalSimulator, CFDSimulator, ChemistrySimulator
except ImportError:
    # Fallback to lightweight implementations for basic functionality
    from .environments.lightweight_habitat import LunarHabitatEnv, make_lunar_env
    from .core.lightweight_config import HabitatConfig, CrewConfig, ScenarioConfig
    # Mock physics simulators for lightweight mode
    class ThermalSimulator:
        def __init__(self, *args, **kwargs): pass
    class CFDSimulator:
        def __init__(self, *args, **kwargs): pass  
    class ChemistrySimulator:
        def __init__(self, *args, **kwargs): pass

__all__ = [
    "__version__",
    "__author__",
    "LunarHabitatEnv", 
    "make_lunar_env",
    "HabitatConfig",
    "CrewConfig", 
    "ScenarioConfig",
    "ThermalSimulator",
    "CFDSimulator",
    "ChemistrySimulator",
]