
"""Lightweight Gymnasium replacement for basic environment interface."""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class Space:
    """Base space class."""
    
    def sample(self):
        """Sample from the space."""
        return None


class Box(Space):
    """Continuous space."""
    
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
    
    def sample(self):
        """Sample from box space."""
        if self.shape:
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
        return np.random.uniform(self.low, self.high)


class Discrete(Space):
    """Discrete space."""
    
    def __init__(self, n):
        self.n = n
    
    def sample(self):
        """Sample from discrete space."""
        return np.random.randint(0, self.n)


class Env:
    """Base environment class."""
    
    def __init__(self):
        self.action_space = None
        self.observation_space = None
    
    def reset(self):
        """Reset environment."""
        return {}, {}
    
    def step(self, action):
        """Take environment step."""
        return {}, 0.0, False, False, {}
    
    def close(self):
        """Close environment."""
        pass


class LunarHabitatEnv(Env):
    """Lightweight lunar habitat environment."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.action_space = Box(low=-1, high=1, shape=(10,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(50,))
    
    def reset(self):
        """Reset to initial state."""
        obs = self.observation_space.sample()
        info = {"mission_day": 0, "crew_health": 1.0}
        return obs, info
    
    def step(self, action):
        """Execute action."""
        obs = self.observation_space.sample()
        reward = np.random.normal(0, 1)
        terminated = False
        truncated = False
        info = {"step_info": "lightweight_simulation"}
        return obs, reward, terminated, truncated, info


# Registry for environments
_env_registry = {
    "LunarHabitat-v1": LunarHabitatEnv
}


def make(env_id: str, **kwargs) -> Env:
    """Create environment instance."""
    if env_id in _env_registry:
        return _env_registry[env_id](**kwargs)
    else:
        raise ValueError(f"Environment {env_id} not found in registry")


def register(id: str, entry_point):
    """Register new environment."""
    _env_registry[id] = entry_point
