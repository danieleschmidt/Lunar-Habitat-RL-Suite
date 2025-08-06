"""
Training Management and Experiment Runner for Lunar Habitat RL

This module provides comprehensive training management capabilities for reinforcement
learning experiments in lunar habitat environments. Supports both research-grade
experimentation and production deployment scenarios.

Features:
- Multi-algorithm training orchestration
- Hyperparameter optimization with safety constraints
- Distributed training across multiple environments
- Comprehensive experiment tracking and reproducibility
- Research-quality benchmarking and evaluation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
import pickle
import multiprocessing as mp
from datetime import datetime
import shutil
import wandb
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import optuna
from collections import defaultdict, deque

from ..core.state import HabitatState
from ..core.metrics import PerformanceTracker, SafetyMonitor
from ..core.config import HabitatConfig
from ..utils.logging import get_logger
from ..utils.exceptions import TrainingError, ExperimentError
from ..utils.validation import validate_training_config
from ..utils.security import SecurityManager
from .baselines import RandomAgent, HeuristicAgent, PPOAgent, SACAgent
from .offline_rl import CQL, IQL, AWAC
from .model_based import DreamerV3, MuZero, PlaNet, create_model_based_agent

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training experiments."""
    # Basic training setup
    algorithm: str = "dreamer_v3"  # Options: ppo, sac, cql, iql, awac, dreamer_v3, muzero, planet
    total_timesteps: int = 10_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000
    log_freq: int = 1000
    
    # Environment configuration
    env_id: str = "LunarHabitat-v1"
    n_envs: int = 8
    env_config: Optional[Dict[str, Any]] = None
    
    # Algorithm hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    gamma: float = 0.99
    tau: float = 0.005
    
    # Training infrastructure
    device: str = "auto"
    n_workers: int = 4
    seed: int = 42
    deterministic: bool = True
    
    # Logging and checkpoints
    experiment_name: str = "lunar_habitat_experiment"
    project_name: str = "lunar-habitat-rl"
    log_dir: Path = Path("./experiments")
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    # Safety and monitoring
    safety_monitoring: bool = True
    early_stopping_patience: int = 100
    min_performance_threshold: float = 0.1
    max_training_time: int = 24 * 3600  # 24 hours in seconds
    
    # Research features
    enable_hyperparameter_tuning: bool = False
    n_trials: int = 100
    study_name: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create experiment directory
        self.log_dir = Path(self.log_dir)
        self.experiment_dir = self.log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.experiment_dir / "training_config.json", "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)


@dataclass
class ExperimentResult:
    """Results from a single training experiment."""
    experiment_id: str
    algorithm: str
    total_timesteps: int
    final_performance: float
    best_performance: float
    training_time: float
    convergence_step: Optional[int]
    hyperparameters: Dict[str, Any]
    safety_violations: int
    final_model_path: Path
    logs: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "final_performance": self.final_performance,
            "best_performance": self.best_performance,
            "training_time": self.training_time,
            "convergence_step": self.convergence_step,
            "hyperparameters": self.hyperparameters,
            "safety_violations": self.safety_violations,
            "final_model_path": str(self.final_model_path),
            "logs": self.logs
        }


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0
        
        # Pre-allocate memory
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: float, 
            next_obs: torch.Tensor, done: bool) -> None:
        """Add experience to buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices]
        }
    
    def __len__(self) -> int:
        return self.size


class TrainingManager:
    """
    Comprehensive training manager for lunar habitat RL experiments.
    
    Handles single algorithm training with full experiment tracking,
    safety monitoring, and research-quality evaluation.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        validate_training_config(config)
        
        # Setup logging
        self.logger = get_logger(f"training_manager_{config.experiment_name}")
        
        # Initialize tracking systems
        self.performance_tracker = PerformanceTracker()
        self.safety_monitor = SafetyMonitor() if config.safety_monitoring else None
        self.security_manager = SecurityManager()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_performance = -float('inf')
        self.training_start_time = None
        self.safety_violations = 0
        
        # Initialize logging systems
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=config.__dict__,
                dir=str(config.experiment_dir)
            )
        
        self.logger.info(f"Initialized TrainingManager for {config.algorithm}")
        self.logger.info(f"Experiment directory: {config.experiment_dir}")
    
    def create_agent(self, obs_dim: int, action_dim: int) -> nn.Module:
        """Create the specified RL agent."""
        if self.config.algorithm == "ppo":
            return PPOAgent(obs_dim, action_dim, lr=self.config.learning_rate)
        elif self.config.algorithm == "sac":
            return SACAgent(obs_dim, action_dim, lr=self.config.learning_rate)
        elif self.config.algorithm == "cql":
            return CQL(obs_dim, action_dim, lr=self.config.learning_rate)
        elif self.config.algorithm == "iql":
            return IQL(obs_dim, action_dim, lr=self.config.learning_rate)
        elif self.config.algorithm == "awac":
            return AWAC(obs_dim, action_dim, lr=self.config.learning_rate)
        elif self.config.algorithm == "dreamer_v3":
            from .model_based import ModelBasedConfig
            mb_config = ModelBasedConfig()
            return DreamerV3(obs_dim, action_dim, mb_config)
        elif self.config.algorithm == "muzero":
            from .model_based import ModelBasedConfig
            mb_config = ModelBasedConfig()
            return MuZero(obs_dim, action_dim, mb_config)
        elif self.config.algorithm == "planet":
            from .model_based import ModelBasedConfig
            mb_config = ModelBasedConfig()
            return PlaNet(obs_dim, action_dim, mb_config)
        else:
            raise TrainingError(f"Unknown algorithm: {self.config.algorithm}")
    
    def create_environment(self) -> Any:
        """Create training environment."""
        # Import here to avoid circular imports
        from ..environments import make_lunar_env
        
        env_config = self.config.env_config or {}
        env = make_lunar_env(
            n_envs=self.config.n_envs,
            **env_config
        )
        return env
    
    def evaluate_agent(self, agent: nn.Module, env: Any, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance."""
        agent.eval()
        episode_rewards = []
        episode_lengths = []
        safety_violations = 0
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                    action = agent.act(obs_tensor)
                    action = action.cpu().numpy().flatten()
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Check for safety violations
                if self.safety_monitor:
                    if info.get('safety_violation', False):
                        safety_violations += 1
                
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        agent.train()
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "safety_violations": safety_violations / n_episodes
        }
    
    def save_checkpoint(self, agent: nn.Module, step: int, is_best: bool = False) -> Path:
        """Save training checkpoint."""
        checkpoint = {
            "agent_state_dict": agent.state_dict(),
            "step": step,
            "episode_count": self.episode_count,
            "best_performance": self.best_performance,
            "safety_violations": self.safety_violations,
            "config": self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.experiment_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.config.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at step {step}")
        
        return checkpoint_path
    
    def train(self) -> ExperimentResult:
        """Main training loop."""
        self.training_start_time = time.time()
        
        # Create environment and agent
        env = self.create_environment()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = self.create_agent(obs_dim, action_dim)
        agent.to(self.config.device)
        
        # Create replay buffer for off-policy algorithms
        if self.config.algorithm in ["sac", "cql", "iql", "awac"]:
            replay_buffer = ReplayBuffer(
                self.config.buffer_size, obs_dim, action_dim, self.config.device
            )
        
        # Training metrics
        episode_rewards = deque(maxlen=100)
        training_logs = []
        patience_counter = 0
        
        self.logger.info("Starting training...")
        
        try:
            obs, _ = env.reset()
            
            for step in range(self.config.total_timesteps):
                self.global_step = step
                
                # Select action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.config.device)
                    action = agent.act(obs_tensor)
                    action_np = action.cpu().numpy().flatten()
                
                # Take environment step
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                
                # Store experience (for off-policy algorithms)
                if self.config.algorithm in ["sac", "cql", "iql", "awac"]:
                    replay_buffer.add(
                        torch.FloatTensor(obs).to(self.config.device),
                        action.squeeze(0),
                        reward,
                        torch.FloatTensor(next_obs).to(self.config.device),
                        done
                    )
                
                # Check for safety violations
                if self.safety_monitor and info.get('safety_violation', False):
                    self.safety_violations += 1
                    self.logger.warning(f"Safety violation detected at step {step}")
                
                if done:
                    episode_rewards.append(info.get('episode_reward', reward))
                    self.episode_count += 1
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                
                # Training updates
                if hasattr(agent, 'update') and len(replay_buffer) > self.config.batch_size:
                    batch = replay_buffer.sample(self.config.batch_size)
                    loss_info = agent.update(batch)
                else:
                    loss_info = {}
                
                # Logging
                if step % self.config.log_freq == 0 and episode_rewards:
                    mean_reward = np.mean(episode_rewards)
                    log_data = {
                        "step": step,
                        "episode": self.episode_count,
                        "mean_reward": mean_reward,
                        "safety_violations": self.safety_violations,
                        "training_time": time.time() - self.training_start_time,
                        **loss_info
                    }
                    
                    training_logs.append(log_data)
                    
                    if self.config.use_wandb:
                        wandb.log(log_data)
                    
                    if step % (self.config.log_freq * 10) == 0:
                        self.logger.info(
                            f"Step {step}: Mean Reward = {mean_reward:.2f}, "
                            f"Episodes = {self.episode_count}, "
                            f"Safety Violations = {self.safety_violations}"
                        )
                
                # Evaluation
                if step % self.config.eval_freq == 0 and step > 0:
                    eval_results = self.evaluate_agent(agent, env)
                    
                    # Check for improvement
                    current_performance = eval_results["mean_reward"]
                    is_best = current_performance > self.best_performance
                    
                    if is_best:
                        self.best_performance = current_performance
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Save checkpoint
                    if step % self.config.save_freq == 0:
                        self.save_checkpoint(agent, step, is_best)
                    
                    # Early stopping
                    if patience_counter >= self.config.early_stopping_patience:
                        self.logger.info(f"Early stopping at step {step} due to no improvement")
                        break
                    
                    # Safety check
                    if (current_performance < self.config.min_performance_threshold and 
                        step > self.config.total_timesteps * 0.1):
                        self.logger.warning(f"Performance below threshold, stopping training")
                        break
                    
                    # Time limit check
                    if time.time() - self.training_start_time > self.config.max_training_time:
                        self.logger.info(f"Training time limit reached, stopping")
                        break
        
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise TrainingError(f"Training failed: {str(e)}")
        
        finally:
            env.close()
            if self.config.use_wandb:
                wandb.finish()
        
        # Final evaluation and save
        final_performance = self.evaluate_agent(agent, self.create_environment())["mean_reward"]
        final_model_path = self.save_checkpoint(agent, self.global_step, True)
        
        training_time = time.time() - self.training_start_time
        
        # Create experiment result
        result = ExperimentResult(
            experiment_id=self.config.experiment_name,
            algorithm=self.config.algorithm,
            total_timesteps=self.global_step,
            final_performance=final_performance,
            best_performance=self.best_performance,
            training_time=training_time,
            convergence_step=None,  # Could be computed from logs
            hyperparameters=self.config.__dict__,
            safety_violations=self.safety_violations,
            final_model_path=final_model_path,
            logs=training_logs
        )
        
        # Save experiment result
        with open(self.config.experiment_dir / "experiment_result.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Training completed successfully!")
        self.logger.info(f"Final performance: {final_performance:.2f}")
        self.logger.info(f"Best performance: {self.best_performance:.2f}")
        self.logger.info(f"Total training time: {training_time:.2f} seconds")
        
        return result


class ExperimentRunner:
    """
    Advanced experiment runner for research-quality benchmarking and comparison.
    
    Supports:
    - Multi-algorithm comparison studies
    - Hyperparameter optimization
    - Statistical significance testing  
    - Reproducible experiment management
    """
    
    def __init__(self, base_config: TrainingConfig, results_dir: Path = Path("./experiment_results")):
        self.base_config = base_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("experiment_runner")
        self.security_manager = SecurityManager()
        
        # Experiment tracking
        self.experiment_results: List[ExperimentResult] = []
        self.comparison_results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        
        self.logger.info(f"Initialized ExperimentRunner with results directory: {results_dir}")
    
    def run_single_experiment(
        self, 
        algorithm: str, 
        config_overrides: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None
    ) -> ExperimentResult:
        """Run a single training experiment."""
        # Create experiment-specific config
        config = TrainingConfig(**self.base_config.__dict__)
        config.algorithm = algorithm
        
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        
        # Add run ID to experiment name
        if run_id:
            config.experiment_name = f"{algorithm}_{run_id}"
        else:
            config.experiment_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run training
        trainer = TrainingManager(config)
        result = trainer.train()
        
        # Store result
        self.experiment_results.append(result)
        self.comparison_results[algorithm].append(result)
        
        return result
    
    def run_algorithm_comparison(
        self, 
        algorithms: List[str], 
        n_runs_per_algorithm: int = 3,
        config_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run comprehensive comparison of multiple algorithms.
        
        Each algorithm is run multiple times for statistical significance.
        """
        self.logger.info(f"Starting algorithm comparison: {algorithms}")
        self.logger.info(f"Running {n_runs_per_algorithm} experiments per algorithm")
        
        results = defaultdict(list)
        
        for algorithm in algorithms:
            self.logger.info(f"Running experiments for {algorithm}")
            
            for run in range(n_runs_per_algorithm):
                run_id = f"run_{run + 1}"
                overrides = config_overrides.get(algorithm, {}) if config_overrides else {}
                
                try:
                    result = self.run_single_experiment(algorithm, overrides, run_id)
                    results[algorithm].append(result)
                    
                    self.logger.info(
                        f"Completed {algorithm} {run_id}: "
                        f"Performance = {result.final_performance:.2f}"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to run {algorithm} {run_id}: {str(e)}")
                    continue
        
        # Save comparison results
        self._save_comparison_results(results)
        
        return dict(results)
    
    def optimize_hyperparameters(
        self, 
        algorithm: str,
        param_space: Dict[str, Any],
        n_trials: int = 50,
        study_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            algorithm: Algorithm to optimize
            param_space: Dictionary defining parameter search space
            n_trials: Number of optimization trials
            study_name: Optional study name for persistence
        """
        if study_name is None:
            study_name = f"{algorithm}_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            config_overrides = {}
            
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    config_overrides[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                elif param_config["type"] == "float":
                    config_overrides[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    config_overrides[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"],
                        log=param_config.get("log", False)
                    )
            
            # Reduce training time for hyperparameter search
            config_overrides["total_timesteps"] = min(
                self.base_config.total_timesteps // 4,
                1_000_000
            )
            config_overrides["eval_freq"] = config_overrides["total_timesteps"] // 20
            
            # Run experiment
            run_id = f"trial_{trial.number}"
            try:
                result = self.run_single_experiment(algorithm, config_overrides, run_id)
                return result.best_performance
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {str(e)}")
                return -float('inf')
        
        # Create and run study
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=f"sqlite:///{self.results_dir / 'hyperopt.db'}",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Log results
        self.logger.info(f"Hyperparameter optimization completed for {algorithm}")
        self.logger.info(f"Best parameters: {study.best_params}")
        self.logger.info(f"Best performance: {study.best_value:.2f}")
        
        # Save optimization results
        with open(self.results_dir / f"{algorithm}_hyperopt_results.json", "w") as f:
            json.dump({
                "algorithm": algorithm,
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": n_trials,
                "study_name": study_name
            }, f, indent=2)
        
        return study.best_params
    
    def _save_comparison_results(self, results: Dict[str, List[ExperimentResult]]) -> None:
        """Save algorithm comparison results."""
        comparison_data = {}
        
        for algorithm, runs in results.items():
            performances = [r.final_performance for r in runs]
            comparison_data[algorithm] = {
                "mean_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "min_performance": np.min(performances),
                "max_performance": np.max(performances),
                "n_runs": len(runs),
                "runs": [r.to_dict() for r in runs]
            }
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"algorithm_comparison_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info(f"Comparison results saved to {results_file}")


# Export main classes
__all__ = [
    "TrainingConfig",
    "ExperimentResult", 
    "TrainingManager",
    "ExperimentRunner",
    "ReplayBuffer"
]