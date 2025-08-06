"""Vectorized environment wrapper and helper functions."""

from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv, AsyncVectorEnv, SyncVectorEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .habitat_base import LunarHabitatEnv
from ..core import HabitatConfig


def make_lunar_env(n_envs: int = 1,
                   config: Optional[Union[HabitatConfig, str]] = None,
                   crew_size: int = 4,
                   scenario: str = "nominal_operations", 
                   reward_config: str = "survival_focused",
                   difficulty: str = "nominal",
                   parallel: bool = True,
                   seed: Optional[int] = None) -> VectorEnv:
    """
    Create vectorized lunar habitat environments for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        config: Habitat configuration or preset name
        crew_size: Number of crew members
        scenario: Mission scenario type
        reward_config: Reward function configuration
        difficulty: Difficulty level
        parallel: Use parallel processing (SubprocVecEnv) vs sequential (DummyVecEnv)
        seed: Random seed for reproducibility
        
    Returns:
        Vectorized environment wrapper
    """
    
    def _make_env(rank: int = 0) -> Callable[[], LunarHabitatEnv]:
        """Create a single environment with unique seed."""
        def _init() -> LunarHabitatEnv:
            env = LunarHabitatEnv(
                config=config,
                crew_size=crew_size,
                scenario=scenario,
                reward_config=reward_config,
                difficulty=difficulty,
                physics_enabled=True
            )
            
            if seed is not None:
                env.reset(seed=seed + rank)
                
            return env
        return _init
    
    # Create environment factories
    env_fns = [_make_env(i) for i in range(n_envs)]
    
    # Use parallel processing for multiple environments
    if n_envs > 1 and parallel:
        try:
            vec_env = SubprocVecEnv(env_fns)
        except Exception:
            # Fallback to sequential if parallel fails
            vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    
    return vec_env


class VectorizedHabitatEnv(VectorEnv):
    """
    Custom vectorized environment with enhanced features for lunar habitat training.
    
    Provides additional functionality like curriculum learning, adaptive difficulty,
    and performance-based environment configuration.
    """
    
    def __init__(self,
                 n_envs: int,
                 config: Optional[Union[HabitatConfig, str]] = None,
                 adaptive_difficulty: bool = False,
                 curriculum_learning: bool = False,
                 performance_threshold: float = 0.8):
        """
        Initialize vectorized environment with advanced features.
        
        Args:
            n_envs: Number of parallel environments
            config: Base configuration for environments  
            adaptive_difficulty: Enable adaptive difficulty scaling
            curriculum_learning: Enable curriculum learning progression
            performance_threshold: Performance threshold for difficulty increases
        """
        
        # Create base environments
        self.base_config = config if isinstance(config, HabitatConfig) else HabitatConfig.from_preset(config or "nasa_reference")
        self.adaptive_difficulty = adaptive_difficulty
        self.curriculum_learning = curriculum_learning
        self.performance_threshold = performance_threshold
        
        # Performance tracking for adaptive features
        self.episode_returns = [[] for _ in range(n_envs)]
        self.episode_lengths = [[] for _ in range(n_envs)]
        self.difficulty_levels = ["easy"] * n_envs
        self.curriculum_stages = [0] * n_envs
        
        # Create environments with different configurations
        self.envs = []
        for i in range(n_envs):
            env_config = self._get_env_config(i)
            env = LunarHabitatEnv(
                config=env_config,
                crew_size=self.base_config.crew.size,
                scenario="nominal_operations", 
                difficulty=self.difficulty_levels[i],
                physics_enabled=True
            )
            self.envs.append(env)
        
        # Initialize parent class
        super().__init__(
            num_envs=n_envs,
            observation_space=self.envs[0].observation_space,
            action_space=self.envs[0].action_space
        )
        
    def _get_env_config(self, env_idx: int) -> HabitatConfig:
        """Get configuration for specific environment index."""
        config = HabitatConfig.from_preset("nasa_reference")
        
        if self.curriculum_learning:
            stage = self.curriculum_stages[env_idx]
            
            if stage == 0:  # Basic operations
                config.scenario.duration_days = 7
                config.scenario.difficulty = "easy"
                config.crew.size = 2
                
            elif stage == 1:  # Nominal operations
                config.scenario.duration_days = 30
                config.scenario.difficulty = "nominal"
                config.crew.size = 4
                
            elif stage == 2:  # Extended missions
                config.scenario.duration_days = 60
                config.scenario.difficulty = "hard"
                config.crew.size = 6
                
            else:  # Extreme scenarios
                config.scenario.duration_days = 180
                config.scenario.difficulty = "extreme"
                config.crew.size = 8
                config.scenario.events = [
                    {'day': 30, 'type': 'solar_array_fault', 'severity': 0.3},
                    {'day': 60, 'type': 'micrometeorite_strike', 'severity': 0.5},
                    {'day': 90, 'type': 'crew_medical_emergency', 'crew_id': 2},
                ]
        
        return config
    
    def reset(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[Dict] = None):
        """Reset all environments."""
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = seed[:self.num_envs]
            
        observations = []
        infos = []
        
        for i, (env, env_seed) in enumerate(zip(self.envs, seeds)):
            obs, info = env.reset(seed=env_seed, options=options)
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations), infos
    
    def step(self, actions: np.ndarray):
        """Step all environments."""
        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(action)
            
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
            
            # Track performance for adaptive features
            if done or truncated:
                self._update_performance_tracking(i, reward, info)
                self._update_adaptive_features(i, info)
        
        return (np.array(observations), 
                np.array(rewards), 
                np.array(dones),
                np.array(truncateds), 
                infos)
    
    def _update_performance_tracking(self, env_idx: int, final_reward: float, info: Dict[str, Any]):
        """Update performance tracking for environment."""
        episode_return = info.get('episode_reward', final_reward)
        episode_length = info.get('steps_taken', 0)
        
        self.episode_returns[env_idx].append(episode_return)
        self.episode_lengths[env_idx].append(episode_length)
        
        # Keep only recent episodes (sliding window)
        max_history = 100
        if len(self.episode_returns[env_idx]) > max_history:
            self.episode_returns[env_idx] = self.episode_returns[env_idx][-max_history:]
            self.episode_lengths[env_idx] = self.episode_lengths[env_idx][-max_history:]
    
    def _update_adaptive_features(self, env_idx: int, info: Dict[str, Any]):
        """Update adaptive difficulty and curriculum based on performance."""
        
        if self.adaptive_difficulty:
            self._update_adaptive_difficulty(env_idx, info)
            
        if self.curriculum_learning:
            self._update_curriculum(env_idx, info)
    
    def _update_adaptive_difficulty(self, env_idx: int, info: Dict[str, Any]):
        """Adjust difficulty based on recent performance."""
        if len(self.episode_returns[env_idx]) < 10:
            return  # Need sufficient data
        
        recent_performance = np.mean(self.episode_returns[env_idx][-10:])
        survival_rate = np.mean([1.0 if info.get('survival_time', 0) > 7 else 0.0 
                                for info in [info]])  # Would need episode history
        
        current_difficulty = self.difficulty_levels[env_idx]
        
        # Increase difficulty if performing well
        if (recent_performance > self.performance_threshold and 
            survival_rate > 0.8 and 
            current_difficulty == "easy"):
            self.difficulty_levels[env_idx] = "nominal"
            self._reconfigure_env(env_idx)
            
        elif (recent_performance > self.performance_threshold and
              survival_rate > 0.9 and  
              current_difficulty == "nominal"):
            self.difficulty_levels[env_idx] = "hard"
            self._reconfigure_env(env_idx)
            
        # Decrease difficulty if struggling
        elif (recent_performance < 0.3 and
              current_difficulty == "hard"):
            self.difficulty_levels[env_idx] = "nominal" 
            self._reconfigure_env(env_idx)
            
        elif (recent_performance < 0.1 and
              current_difficulty == "nominal"):
            self.difficulty_levels[env_idx] = "easy"
            self._reconfigure_env(env_idx)
    
    def _update_curriculum(self, env_idx: int, info: Dict[str, Any]):
        """Progress curriculum stages based on mastery."""
        if len(self.episode_returns[env_idx]) < 20:
            return
            
        recent_success_rate = np.mean([
            1.0 if ret > 0 else 0.0 
            for ret in self.episode_returns[env_idx][-20:]
        ])
        
        current_stage = self.curriculum_stages[env_idx]
        
        # Progress to next stage if mastered current stage
        if recent_success_rate > 0.85 and current_stage < 3:
            self.curriculum_stages[env_idx] += 1
            self._reconfigure_env(env_idx)
            
        # Regress if struggling too much
        elif recent_success_rate < 0.3 and current_stage > 0:
            self.curriculum_stages[env_idx] = max(0, current_stage - 1)
            self._reconfigure_env(env_idx)
    
    def _reconfigure_env(self, env_idx: int):
        """Reconfigure environment with new settings."""
        new_config = self._get_env_config(env_idx)
        
        # Create new environment with updated configuration
        self.envs[env_idx] = LunarHabitatEnv(
            config=new_config,
            crew_size=new_config.crew.size,
            scenario="nominal_operations",
            difficulty=self.difficulty_levels[env_idx],
            physics_enabled=True
        )
        
        # Reset the reconfigured environment
        self.envs[env_idx].reset()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics across all environments."""
        all_returns = [ret for env_returns in self.episode_returns for ret in env_returns]
        all_lengths = [length for env_lengths in self.episode_lengths for length in env_lengths]
        
        stats = {
            'total_episodes': len(all_returns),
            'mean_return': np.mean(all_returns) if all_returns else 0.0,
            'std_return': np.std(all_returns) if all_returns else 0.0,
            'mean_length': np.mean(all_lengths) if all_lengths else 0.0,
            'difficulty_distribution': {
                diff: self.difficulty_levels.count(diff) 
                for diff in set(self.difficulty_levels)
            },
            'curriculum_distribution': {
                stage: self.curriculum_stages.count(stage)
                for stage in set(self.curriculum_stages)
            }
        }
        
        return stats
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()