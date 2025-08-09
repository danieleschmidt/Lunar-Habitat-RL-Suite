"""Core configuration classes and data structures for the Lunar Habitat RL Suite."""

# Try full implementations first, fallback to lightweight
try:
    from .config import HabitatConfig, CrewConfig, ScenarioConfig, PhysicsConfig
    from .state import HabitatState, ActionSpace
    from .metrics import MissionMetrics, PerformanceTracker
except ImportError:
    # Fallback to lightweight implementations
    from .lightweight_config import HabitatConfig, CrewConfig, ScenarioConfig
    from .lightweight_state import HabitatState, ActionSpace
    # Mock additional classes for lightweight mode
    class PhysicsConfig:
        def __init__(self, *args, **kwargs): pass
    class MissionMetrics:
        def __init__(self, *args, **kwargs): pass
    class PerformanceTracker:
        def __init__(self, *args, **kwargs): pass

__all__ = [
    "HabitatConfig",
    "CrewConfig", 
    "ScenarioConfig",
    "PhysicsConfig",
    "HabitatState",
    "ActionSpace", 
    "MissionMetrics",
    "PerformanceTracker",
]