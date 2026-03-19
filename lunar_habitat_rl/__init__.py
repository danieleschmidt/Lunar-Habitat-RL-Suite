"""
Lunar Habitat RL Suite
======================
Reinforcement learning for autonomous lunar base management.

Components:
  - LunarHabitatEnv: Gym-compatible environment simulating a lunar outpost
  - HabitatPolicyNetwork: MLP actor-critic policy
  - PPOHabitatAgent: PPO trainer
  - SafetyMonitor: Real-time hazard monitoring with emergency protocols
"""

from .environment import LunarHabitatEnv
from .policy import HabitatPolicyNetwork
from .agent import PPOHabitatAgent
from .safety import SafetyMonitor

__all__ = [
    "LunarHabitatEnv",
    "HabitatPolicyNetwork",
    "PPOHabitatAgent",
    "SafetyMonitor",
]

__version__ = "1.0.0"
