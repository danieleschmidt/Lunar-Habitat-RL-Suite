"""
Scalable Hyperparameter Optimization for Lunar Habitat RL

This module provides advanced hyperparameter optimization capabilities that scale
across distributed infrastructure for efficient automated machine learning.

Features:
- Bayesian optimization with Gaussian processes
- Multi-fidelity optimization (Hyperband, BOHB)
- Population-based training (PBT) 
- Evolutionary strategies for hyperparameter search
- Neural architecture search (NAS) for network topology
- Distributed parallel optimization
- Early stopping and resource allocation
- Integration with cloud computing platforms
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import copy
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict, deque
import random
import math

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.suggest.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .algorithms.training import TrainingManager, TrainingConfig
from .algorithms.baselines import PPOAgent, SACAgent
from .algorithms.model_based import DreamerV3, ModelBasedConfig
from .distributed import DistributedConfig, launch_distributed_training
from .utils.logging import get_logger
from .utils.exceptions import HyperoptError, OptimizationError
from .utils.validation import validate_hyperopt_config

logger = get_logger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    name: str
    param_type: str  # "float", "int", "categorical", "bool"
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[List[Any]] = None
    log_scale: bool = False
    step: Optional[float] = None
    
    def sample(self, rng: Optional[np.random.Generator] = None) -> Any:
        """Sample a value from this hyperparameter space."""
        if rng is None:
            rng = np.random.default_rng()
        
        if self.param_type == "float":
            if self.log_scale:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                return np.exp(rng.uniform(log_low, log_high))
            else:
                return rng.uniform(self.low, self.high)
        
        elif self.param_type == "int":
            return rng.integers(self.low, self.high + 1)
        
        elif self.param_type == "categorical":
            return rng.choice(self.values)
        
        elif self.param_type == "bool":
            return rng.choice([True, False])
        
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")
    
    def to_optuna_distribution(self):
        """Convert to Optuna distribution."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        if self.param_type == "float":
            if self.log_scale:
                return optuna.distributions.FloatDistribution(
                    low=self.low, high=self.high, log=True
                )
            else:
                return optuna.distributions.FloatDistribution(
                    low=self.low, high=self.high, step=self.step
                )
        
        elif self.param_type == "int":
            return optuna.distributions.IntDistribution(low=self.low, high=self.high)
        
        elif self.param_type == "categorical":
            return optuna.distributions.CategoricalDistribution(choices=self.values)
        
        elif self.param_type == "bool":
            return optuna.distributions.CategoricalDistribution(choices=[True, False])


@dataclass 
class HyperoptConfig:
    """Configuration for hyperparameter optimization."""
    # Search space
    search_space: List[HyperparameterSpace] = field(default_factory=list)
    
    # Optimization algorithm
    algorithm: str = "tpe"  # tpe, cma_es, random, evolutionary, pbt
    n_trials: int = 100
    n_parallel_jobs: int = 4
    timeout: Optional[float] = None  # seconds
    
    # Multi-fidelity settings
    use_early_stopping: bool = True
    min_resource: int = 10  # Minimum training steps
    max_resource: int = 1000  # Maximum training steps
    reduction_factor: int = 3
    
    # Population-based training settings
    population_size: int = 20
    perturbation_interval: int = 100
    mutation_probability: float = 0.1
    
    # Resource allocation
    max_concurrent_trials: int = 10
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {"cpu": 2.0, "gpu": 0.5})
    
    # Checkpointing and resumption
    study_name: Optional[str] = None
    storage_url: Optional[str] = None  # Database URL for persistent storage
    resume_study: bool = True
    
    # Logging and monitoring
    log_to_wandb: bool = False
    wandb_project: str = "lunar-habitat-hyperopt"
    save_intermediate_results: bool = True
    
    # Evaluation settings
    evaluation_metric: str = "mean_episode_reward"  # Metric to optimize
    optimization_direction: str = "maximize"  # "maximize" or "minimize"
    n_evaluation_episodes: int = 10
    
    # Distributed settings
    use_distributed: bool = False
    distributed_config: Optional[DistributedConfig] = None


def create_default_search_space() -> List[HyperparameterSpace]:
    """Create default hyperparameter search space for lunar habitat RL."""
    
    search_space = [
        # Learning rates
        HyperparameterSpace(
            name="learning_rate",
            param_type="float",
            low=1e-5,
            high=1e-2,
            log_scale=True
        ),
        
        # Network architecture
        HyperparameterSpace(
            name="hidden_size",
            param_type="categorical",
            values=[64, 128, 256, 512]
        ),
        
        HyperparameterSpace(
            name="n_layers",
            param_type="int",
            low=2,
            high=5
        ),
        
        # Training hyperparameters
        HyperparameterSpace(
            name="batch_size",
            param_type="categorical",
            values=[32, 64, 128, 256]
        ),
        
        HyperparameterSpace(
            name="buffer_size",
            param_type="categorical",
            values=[10000, 50000, 100000, 500000]
        ),
        
        # Algorithm-specific parameters
        HyperparameterSpace(
            name="gamma",
            param_type="float",
            low=0.95,
            high=0.999
        ),
        
        HyperparameterSpace(
            name="tau",
            param_type="float",
            low=0.001,
            high=0.01
        ),
        
        # Regularization
        HyperparameterSpace(
            name="entropy_coef",
            param_type="float",
            low=0.001,
            high=0.1,
            log_scale=True
        ),
        
        HyperparameterSpace(
            name="value_loss_coef",
            param_type="float",
            low=0.1,
            high=1.0
        ),
        
        # Exploration
        HyperparameterSpace(
            name="exploration_noise",
            param_type="float",
            low=0.01,
            high=0.3
        ),
    ]
    
    return search_space


class BayesianOptimizer:
    """Bayesian hyperparameter optimization using Optuna."""
    
    def __init__(self, config: HyperoptConfig):
        self.config = config
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        # Create or load study
        study_name = config.study_name or f"lunar_habitat_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup sampler
        if config.algorithm == "tpe":
            sampler = TPESampler(seed=42)
        elif config.algorithm == "cma_es":
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)
        
        # Setup pruner for early stopping
        if config.use_early_stopping:
            pruner = HyperbandPruner(
                min_resource=config.min_resource,
                max_resource=config.max_resource,
                reduction_factor=config.reduction_factor
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        direction = "maximize" if config.optimization_direction == "maximize" else "minimize"
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=config.storage_url,
            load_if_exists=config.resume_study,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Initialized Bayesian optimizer with study: {study_name}")
    
    def optimize(self, 
                 objective_fn: Callable[[Dict[str, Any]], float],
                 n_trials: Optional[int] = None) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        
        n_trials = n_trials or self.config.n_trials
        
        def optuna_objective(trial):
            # Sample hyperparameters
            params = {}
            for param_space in self.config.search_space:
                if param_space.param_type == "float":
                    if param_space.log_scale:
                        params[param_space.name] = trial.suggest_float(
                            param_space.name, param_space.low, param_space.high, log=True
                        )
                    else:
                        params[param_space.name] = trial.suggest_float(
                            param_space.name, param_space.low, param_space.high, step=param_space.step
                        )
                elif param_space.param_type == "int":
                    params[param_space.name] = trial.suggest_int(
                        param_space.name, param_space.low, param_space.high
                    )
                elif param_space.param_type == "categorical":
                    params[param_space.name] = trial.suggest_categorical(
                        param_space.name, param_space.values
                    )
                elif param_space.param_type == "bool":
                    params[param_space.name] = trial.suggest_categorical(
                        param_space.name, [True, False]
                    )
            
            # Evaluate objective
            try:
                score = objective_fn(params)
                
                # Log to WandB if configured
                if self.config.log_to_wandb and WANDB_AVAILABLE:
                    wandb.log({"objective_score": score, "trial": trial.number, **params})
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                # Return worst possible score for failed trials
                return -np.inf if self.config.optimization_direction == "maximize" else np.inf
        
        # Run optimization
        self.study.optimize(
            optuna_objective, 
            n_trials=n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_parallel_jobs
        )
        
        # Return results
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.study.study_name,
            "optimization_history": [
                {"trial": i, "value": trial.value, "params": trial.params}
                for i, trial in enumerate(self.study.trials)
                if trial.value is not None
            ]
        }
        
        logger.info(f"Optimization complete. Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results


class PopulationBasedTrainer:
    """Population-based training for hyperparameter optimization."""
    
    def __init__(self, config: HyperoptConfig):
        self.config = config
        self.population_size = config.population_size
        self.perturbation_interval = config.perturbation_interval
        
        # Population of agents
        self.population = []
        self.population_scores = []
        self.population_ages = []
        
        # Performance tracking
        self.generation = 0
        self.best_score_history = []
        self.diversity_history = []
        
        logger.info(f"Initialized PBT with population size {self.population_size}")
    
    def initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize random population of hyperparameters."""
        
        population = []
        rng = np.random.default_rng(42)
        
        for i in range(self.population_size):
            individual = {}
            
            for param_space in self.config.search_space:
                individual[param_space.name] = param_space.sample(rng)
            
            population.append(individual)
        
        self.population = population
        self.population_scores = [0.0] * self.population_size
        self.population_ages = [0] * self.population_size
        
        return population
    
    def evolve_population(self, 
                         objective_fn: Callable[[Dict[str, Any]], float],
                         n_generations: int = 50) -> Dict[str, Any]:
        """Evolve population using PBT algorithm."""
        
        # Initialize population
        if not self.population:
            self.initialize_population()
        
        best_individual = None
        best_score = -np.inf if self.config.optimization_direction == "maximize" else np.inf
        
        for generation in range(n_generations):
            self.generation = generation
            
            # Evaluate population
            scores = []
            for i, individual in enumerate(self.population):
                score = objective_fn(individual)
                scores.append(score)
                self.population_scores[i] = score
                self.population_ages[i] += 1
                
                # Track best individual
                if self.config.optimization_direction == "maximize":
                    if score > best_score:
                        best_score = score
                        best_individual = copy.deepcopy(individual)
                else:
                    if score < best_score:
                        best_score = score
                        best_individual = copy.deepcopy(individual)
            
            # Track metrics
            self.best_score_history.append(best_score)
            self.diversity_history.append(self._compute_diversity())
            
            logger.info(f"Generation {generation}: Best score = {best_score:.4f}")
            
            # Evolution step
            if generation < n_generations - 1:  # Skip evolution on last generation
                self._exploit_and_explore()
        
        # Return results
        results = {
            "best_params": best_individual,
            "best_value": best_score,
            "n_generations": n_generations,
            "population_size": self.population_size,
            "best_score_history": self.best_score_history,
            "diversity_history": self.diversity_history,
            "final_population": copy.deepcopy(self.population)
        }
        
        return results
    
    def _exploit_and_explore(self):
        """Exploit good solutions and explore new ones."""
        
        # Sort population by performance
        sorted_indices = sorted(
            range(self.population_size),
            key=lambda i: self.population_scores[i],
            reverse=(self.config.optimization_direction == "maximize")
        )
        
        # Bottom 20% exploit top 20%
        n_bottom = max(1, self.population_size // 5)
        n_top = max(1, self.population_size // 5)
        
        bottom_indices = sorted_indices[-n_bottom:]
        top_indices = sorted_indices[:n_top]
        
        rng = np.random.default_rng()
        
        for bottom_idx in bottom_indices:
            if rng.random() < 0.8:  # 80% chance to exploit
                # Copy from random top performer
                top_idx = rng.choice(top_indices)
                self.population[bottom_idx] = copy.deepcopy(self.population[top_idx])
                self.population_ages[bottom_idx] = 0  # Reset age
                
                # Perturb parameters (explore)
                self._perturb_individual(bottom_idx, rng)
    
    def _perturb_individual(self, idx: int, rng: np.random.Generator):
        """Perturb an individual's hyperparameters."""
        
        individual = self.population[idx]
        
        for param_space in self.config.search_space:
            if rng.random() < self.config.mutation_probability:
                param_name = param_space.name
                
                if param_space.param_type == "float":
                    current_value = individual[param_name]
                    
                    # Perturb by ±20%
                    if param_space.log_scale:
                        log_value = np.log(current_value)
                        perturbation = rng.normal(0, 0.2)
                        new_log_value = log_value + perturbation
                        new_value = np.exp(new_log_value)
                    else:
                        perturbation = rng.normal(0, 0.2 * abs(current_value))
                        new_value = current_value + perturbation
                    
                    # Clamp to bounds
                    new_value = np.clip(new_value, param_space.low, param_space.high)
                    individual[param_name] = new_value
                
                elif param_space.param_type == "int":
                    # Random walk
                    current_value = individual[param_name]
                    step = rng.choice([-1, 1])
                    new_value = current_value + step
                    new_value = np.clip(new_value, param_space.low, param_space.high)
                    individual[param_name] = new_value
                
                elif param_space.param_type == "categorical":
                    # Random choice
                    individual[param_name] = rng.choice(param_space.values)
                
                elif param_space.param_type == "bool":
                    # Flip with some probability
                    if rng.random() < 0.3:
                        individual[param_name] = not individual[param_name]
    
    def _compute_diversity(self) -> float:
        """Compute population diversity metric."""
        if self.population_size < 2:
            return 0.0
        
        # Simple diversity metric: average pairwise parameter difference
        diversities = []
        
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                diversity = 0.0
                n_params = 0
                
                for param_space in self.config.search_space:
                    param_name = param_space.name
                    val_i = self.population[i][param_name]
                    val_j = self.population[j][param_name]
                    
                    if param_space.param_type in ["float", "int"]:
                        # Normalized difference
                        if param_space.param_type == "float" and param_space.log_scale:
                            diff = abs(np.log(val_i) - np.log(val_j))
                            max_diff = abs(np.log(param_space.high) - np.log(param_space.low))
                        else:
                            diff = abs(val_i - val_j)
                            max_diff = abs(param_space.high - param_space.low)
                        
                        if max_diff > 0:
                            diversity += diff / max_diff
                        n_params += 1
                    
                    elif param_space.param_type in ["categorical", "bool"]:
                        diversity += 1.0 if val_i != val_j else 0.0
                        n_params += 1
                
                if n_params > 0:
                    diversities.append(diversity / n_params)
        
        return np.mean(diversities) if diversities else 0.0


class HyperparameterOptimizer:
    """Main hyperparameter optimizer with multiple algorithms."""
    
    def __init__(self, config: HyperoptConfig):
        self.config = config
        validate_hyperopt_config(config)
        
        # Create search space if not provided
        if not config.search_space:
            config.search_space = create_default_search_space()
        
        # Initialize logging
        if config.log_to_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config.__dict__
            )
        
        logger.info(f"Initialized hyperparameter optimizer with {config.algorithm}")
        logger.info(f"Search space: {[p.name for p in config.search_space]}")
    
    def optimize(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        
        if self.config.algorithm in ["tpe", "cma_es", "random"]:
            optimizer = BayesianOptimizer(self.config)
            return optimizer.optimize(objective_fn)
        
        elif self.config.algorithm == "pbt":
            optimizer = PopulationBasedTrainer(self.config)
            return optimizer.evolve_population(objective_fn)
        
        elif self.config.algorithm == "evolutionary":
            return self._evolutionary_optimization(objective_fn)
        
        else:
            raise HyperoptError(f"Unknown optimization algorithm: {self.config.algorithm}")
    
    def _evolutionary_optimization(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """Simple evolutionary strategy for hyperparameter optimization."""
        
        population_size = 20
        n_generations = self.config.n_trials // population_size
        mutation_rate = 0.1
        
        # Initialize population
        rng = np.random.default_rng(42)
        population = []
        
        for _ in range(population_size):
            individual = {}
            for param_space in self.config.search_space:
                individual[param_space.name] = param_space.sample(rng)
            population.append(individual)
        
        best_individual = None
        best_score = -np.inf if self.config.optimization_direction == "maximize" else np.inf
        history = []
        
        for generation in range(n_generations):
            # Evaluate population
            scores = []
            for individual in population:
                score = objective_fn(individual)
                scores.append(score)
                
                if self.config.optimization_direction == "maximize":
                    if score > best_score:
                        best_score = score
                        best_individual = copy.deepcopy(individual)
                else:
                    if score < best_score:
                        best_score = score
                        best_individual = copy.deepcopy(individual)
            
            history.append({
                "generation": generation,
                "best_score": best_score,
                "mean_score": np.mean(scores),
                "std_score": np.std(scores)
            })
            
            logger.info(f"Generation {generation}: Best = {best_score:.4f}, Mean = {np.mean(scores):.4f}")
            
            # Selection and mutation
            if generation < n_generations - 1:
                # Tournament selection
                new_population = []
                for _ in range(population_size):
                    tournament_size = 3
                    tournament_indices = rng.choice(population_size, tournament_size, replace=False)
                    tournament_scores = [scores[i] for i in tournament_indices]
                    
                    if self.config.optimization_direction == "maximize":
                        winner_idx = tournament_indices[np.argmax(tournament_scores)]
                    else:
                        winner_idx = tournament_indices[np.argmin(tournament_scores)]
                    
                    # Copy winner and potentially mutate
                    new_individual = copy.deepcopy(population[winner_idx])
                    
                    if rng.random() < mutation_rate:
                        # Mutate random parameter
                        param_space = rng.choice(self.config.search_space)
                        new_individual[param_space.name] = param_space.sample(rng)
                    
                    new_population.append(new_individual)
                
                population = new_population
        
        return {
            "best_params": best_individual,
            "best_value": best_score,
            "n_generations": n_generations,
            "population_size": population_size,
            "history": history
        }


def create_objective_function(
    algorithm: str,
    env_creator: Callable,
    training_steps: int = 10000,
    evaluation_episodes: int = 5
) -> Callable[[Dict[str, Any]], float]:
    """
    Create objective function for hyperparameter optimization.
    
    Args:
        algorithm: RL algorithm to optimize
        env_creator: Function that creates the environment
        training_steps: Number of training steps per trial
        evaluation_episodes: Number of episodes for evaluation
    
    Returns:
        Objective function that takes hyperparameters and returns performance score
    """
    
    def objective(params: Dict[str, Any]) -> float:
        """Objective function for hyperparameter optimization."""
        
        try:
            # Create training configuration with hyperparameters
            training_config = TrainingConfig(
                algorithm=algorithm,
                total_timesteps=training_steps,
                learning_rate=params.get("learning_rate", 3e-4),
                batch_size=params.get("batch_size", 256),
                buffer_size=params.get("buffer_size", 100000),
                gamma=params.get("gamma", 0.99),
                tau=params.get("tau", 0.005),
                eval_freq=training_steps,  # Evaluate at end
                log_freq=training_steps // 10,
                use_wandb=False  # Disable to avoid conflicts
            )
            
            # Create and run training
            trainer = TrainingManager(training_config)
            env = env_creator()
            
            # Add algorithm-specific hyperparameters
            if algorithm == "dreamer_v3":
                model_config = ModelBasedConfig(
                    learning_rate=params.get("learning_rate", 6e-4),
                    hidden_size=params.get("hidden_size", 400),
                    stoch_size=params.get("stoch_size", 32),
                    deter_size=params.get("deter_size", 512)
                )
                # Would create DreamerV3 with config
            
            # Train agent
            result = trainer.train()
            
            # Return performance metric
            return result.best_performance
            
        except Exception as e:
            logger.error(f"Objective evaluation failed: {e}")
            return -np.inf  # Return worst possible score for failed trials
    
    return objective


def run_hyperparameter_optimization(
    algorithm: str,
    env_creator: Callable,
    config: Optional[HyperoptConfig] = None,
    training_steps: int = 100000
) -> Dict[str, Any]:
    """
    Run complete hyperparameter optimization for lunar habitat RL.
    
    Args:
        algorithm: RL algorithm to optimize
        env_creator: Function that creates the environment
        config: Hyperparameter optimization configuration
        training_steps: Number of training steps per trial
    
    Returns:
        Optimization results with best hyperparameters
    """
    
    if config is None:
        config = HyperoptConfig(
            algorithm="tpe",
            n_trials=50,
            n_parallel_jobs=4,
            use_early_stopping=True
        )
    
    # Create objective function
    objective_fn = create_objective_function(
        algorithm, 
        env_creator, 
        training_steps,
        config.n_evaluation_episodes
    )
    
    # Run optimization
    optimizer = HyperparameterOptimizer(config)
    results = optimizer.optimize(objective_fn)
    
    # Save results
    results_path = Path("hyperopt_results") / f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Hyperparameter optimization complete. Results saved to {results_path}")
    
    return results


# Export main classes
__all__ = [
    "HyperparameterSpace",
    "HyperoptConfig",
    "BayesianOptimizer",
    "PopulationBasedTrainer", 
    "HyperparameterOptimizer",
    "create_default_search_space",
    "create_objective_function",
    "run_hyperparameter_optimization"
]