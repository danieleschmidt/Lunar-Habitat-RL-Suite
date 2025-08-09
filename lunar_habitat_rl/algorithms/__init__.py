"""Reinforcement Learning algorithms for lunar habitat control."""

# Import baseline agents (no torch dependency)
from .baselines import RandomAgent, HeuristicAgent

# Conditionally import torch-based agents
__all__ = ["RandomAgent", "HeuristicAgent"]

try:
    import torch
    from .baselines import PPOAgent, SACAgent
    __all__.extend(["PPOAgent", "SACAgent"])
except ImportError:
    pass

# Conditionally import advanced algorithms
try:
    from .offline_rl import CQL, IQL, AWAC
    __all__.extend(["CQL", "IQL", "AWAC"])
except ImportError:
    pass

try:
    from .model_based import MuZero, DreamerV3, PlaNet
    __all__.extend(["MuZero", "DreamerV3", "PlaNet"])
except ImportError:
    pass

try:
    from .training import TrainingManager, ExperimentRunner
    __all__.extend(["TrainingManager", "ExperimentRunner"])
except ImportError:
    pass