"""Gymnasium environments for lunar habitat simulation."""

# Try to import full implementation first, fallback to lightweight
try:
    from .habitat_base import LunarHabitatEnv
    from .multi_env import make_lunar_env, VectorizedHabitatEnv
except ImportError:
    from .lightweight_habitat import LunarHabitatEnv, make_lunar_env
    # Mock VectorizedHabitatEnv for lightweight mode
    class VectorizedHabitatEnv:
        def __init__(self, *args, **kwargs): pass

__all__ = [
    "LunarHabitatEnv",
    "make_lunar_env", 
    "VectorizedHabitatEnv",
]