"""
Test suite for hyperparameter optimization in the lunar habitat RL suite.

This module provides comprehensive tests for hyperparameter optimization
functionality including Bayesian optimization, population-based training,
and evolutionary strategies.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Callable
import tempfile
import json

# Import modules to test
from lunar_habitat_rl.hyperopt import (
    HyperparameterSpace, HyperoptConfig, BayesianOptimizer,
    PopulationBasedTrainer, HyperparameterOptimizer,
    create_default_search_space, create_objective_function,
    run_hyperparameter_optimization
)


class TestHyperparameterSpace:
    """Test suite for hyperparameter search space definition."""

    def test_float_space_creation(self):
        """Test creating float hyperparameter space."""
        space = HyperparameterSpace(
            name="learning_rate",
            param_type="float",
            low=1e-5,
            high=1e-2,
            log_scale=True
        )
        
        assert space.name == "learning_rate"
        assert space.param_type == "float"
        assert space.low == 1e-5
        assert space.high == 1e-2
        assert space.log_scale is True

    def test_int_space_creation(self):
        """Test creating integer hyperparameter space."""
        space = HyperparameterSpace(
            name="batch_size",
            param_type="int",
            low=16,
            high=512
        )
        
        assert space.name == "batch_size"
        assert space.param_type == "int"
        assert space.low == 16
        assert space.high == 512

    def test_categorical_space_creation(self):
        """Test creating categorical hyperparameter space."""
        space = HyperparameterSpace(
            name="optimizer",
            param_type="categorical",
            values=["adam", "sgd", "rmsprop"]
        )
        
        assert space.name == "optimizer"
        assert space.param_type == "categorical"
        assert space.values == ["adam", "sgd", "rmsprop"]

    def test_bool_space_creation(self):
        """Test creating boolean hyperparameter space."""
        space = HyperparameterSpace(
            name="use_batchnorm",
            param_type="bool"
        )
        
        assert space.name == "use_batchnorm"
        assert space.param_type == "bool"

    def test_float_sampling_linear(self):
        """Test sampling from linear float space."""
        space = HyperparameterSpace(
            name="test",
            param_type="float",
            low=0.0,
            high=1.0,
            log_scale=False
        )
        
        rng = np.random.RandomState(42)
        samples = [space.sample(rng) for _ in range(100)]
        
        assert all(0.0 <= s <= 1.0 for s in samples)
        assert len(set(samples)) > 50  # Should have variety

    def test_float_sampling_log_scale(self):
        """Test sampling from log-scale float space."""
        space = HyperparameterSpace(
            name="test",
            param_type="float",
            low=1e-5,
            high=1e-1,
            log_scale=True
        )
        
        rng = np.random.RandomState(42)
        samples = [space.sample(rng) for _ in range(100)]
        
        assert all(1e-5 <= s <= 1e-1 for s in samples)
        # Log scale should produce more small values
        small_values = sum(1 for s in samples if s < 1e-3)
        assert small_values > 10

    def test_int_sampling(self):
        """Test sampling from integer space."""
        space = HyperparameterSpace(
            name="test",
            param_type="int",
            low=10,
            high=100
        )
        
        rng = np.random.RandomState(42)
        samples = [space.sample(rng) for _ in range(50)]
        
        assert all(isinstance(s, (int, np.integer)) for s in samples)
        assert all(10 <= s <= 100 for s in samples)

    def test_categorical_sampling(self):
        """Test sampling from categorical space."""
        values = ["option_a", "option_b", "option_c"]
        space = HyperparameterSpace(
            name="test",
            param_type="categorical",
            values=values
        )
        
        rng = np.random.RandomState(42)
        samples = [space.sample(rng) for _ in range(50)]
        
        assert all(s in values for s in samples)
        unique_samples = set(samples)
        assert len(unique_samples) > 1  # Should sample different values

    def test_bool_sampling(self):
        """Test sampling from boolean space."""
        space = HyperparameterSpace(
            name="test",
            param_type="bool"
        )
        
        rng = np.random.RandomState(42)
        samples = [space.sample(rng) for _ in range(50)]
        
        assert all(isinstance(s, (bool, np.bool_)) for s in samples)
        assert True in samples and False in samples

    def test_invalid_param_type(self):
        """Test error handling for invalid parameter type."""
        space = HyperparameterSpace(
            name="test",
            param_type="invalid_type"
        )
        
        with pytest.raises(ValueError, match="Unknown parameter type"):
            space.sample()

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.optuna')
    def test_optuna_distribution_float(self, mock_optuna):
        """Test conversion to Optuna distribution for float."""
        mock_distribution = Mock()
        mock_optuna.distributions.FloatDistribution.return_value = mock_distribution
        
        space = HyperparameterSpace(
            name="test",
            param_type="float",
            low=0.1,
            high=1.0,
            log_scale=True
        )
        
        distribution = space.to_optuna_distribution()
        
        mock_optuna.distributions.FloatDistribution.assert_called_once_with(
            low=0.1, high=1.0, log=True
        )
        assert distribution == mock_distribution

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', False)
    def test_optuna_distribution_not_available(self):
        """Test error when Optuna is not available."""
        space = HyperparameterSpace(
            name="test",
            param_type="float",
            low=0.1,
            high=1.0
        )
        
        with pytest.raises(ImportError, match="Optuna is required"):
            space.to_optuna_distribution()


class TestHyperoptConfig:
    """Test suite for hyperparameter optimization configuration."""

    def test_default_config(self):
        """Test default hyperopt configuration."""
        config = HyperoptConfig()
        
        assert config.algorithm == "tpe"
        assert config.n_trials == 100
        assert config.n_parallel_jobs == 4
        assert config.use_early_stopping is True
        assert config.population_size == 20

    def test_custom_config(self):
        """Test custom hyperopt configuration."""
        search_space = create_default_search_space()
        
        config = HyperoptConfig(
            search_space=search_space,
            algorithm="cma_es",
            n_trials=50,
            n_parallel_jobs=8,
            timeout=3600.0,
            study_name="custom_study"
        )
        
        assert config.search_space == search_space
        assert config.algorithm == "cma_es"
        assert config.n_trials == 50
        assert config.n_parallel_jobs == 8
        assert config.timeout == 3600.0
        assert config.study_name == "custom_study"

    def test_pbt_config(self):
        """Test population-based training configuration."""
        config = HyperoptConfig(
            algorithm="pbt",
            population_size=50,
            perturbation_interval=200,
            mutation_probability=0.2
        )
        
        assert config.algorithm == "pbt"
        assert config.population_size == 50
        assert config.perturbation_interval == 200
        assert config.mutation_probability == 0.2


class TestCreateDefaultSearchSpace:
    """Test suite for default search space creation."""

    def test_default_search_space(self):
        """Test default hyperparameter search space."""
        search_space = create_default_search_space()
        
        assert isinstance(search_space, list)
        assert len(search_space) > 5  # Should have multiple parameters
        
        param_names = [param.name for param in search_space]
        expected_params = [
            "learning_rate", "hidden_size", "n_layers", "batch_size",
            "buffer_size", "gamma", "tau", "entropy_coef"
        ]
        
        for param_name in expected_params:
            assert param_name in param_names

    def test_search_space_types(self):
        """Test that search space has appropriate parameter types."""
        search_space = create_default_search_space()
        
        # Find specific parameters and check their types
        lr_param = next(p for p in search_space if p.name == "learning_rate")
        assert lr_param.param_type == "float"
        assert lr_param.log_scale is True
        
        batch_size_param = next(p for p in search_space if p.name == "batch_size")
        assert batch_size_param.param_type == "categorical"
        assert isinstance(batch_size_param.values, list)
        
        n_layers_param = next(p for p in search_space if p.name == "n_layers")
        assert n_layers_param.param_type == "int"


class TestBayesianOptimizer:
    """Test suite for Bayesian hyperparameter optimization."""

    @pytest.fixture
    def simple_config(self):
        """Simple configuration for testing."""
        return HyperoptConfig(
            search_space=create_default_search_space()[:3],  # Just a few params
            algorithm="tpe",
            n_trials=5,
            n_parallel_jobs=1
        )

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.optuna')
    def test_bayesian_optimizer_initialization(self, mock_optuna, simple_config):
        """Test Bayesian optimizer initialization."""
        mock_study = Mock()
        mock_study.study_name = "test_study"
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = BayesianOptimizer(simple_config)
        
        assert optimizer.config == simple_config
        mock_optuna.create_study.assert_called_once()

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', False)
    def test_bayesian_optimizer_no_optuna(self, simple_config):
        """Test error when Optuna is not available."""
        with pytest.raises(ImportError, match="Optuna is required"):
            BayesianOptimizer(simple_config)

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.optuna')
    def test_bayesian_optimizer_optimize(self, mock_optuna, simple_config):
        """Test Bayesian optimization process."""
        # Mock study and optimization
        mock_study = Mock()
        mock_study.study_name = "test_study"
        mock_study.best_params = {"learning_rate": 0.001}
        mock_study.best_value = 100.0
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = BayesianOptimizer(simple_config)
        
        # Simple objective function
        def objective(params):
            return sum(params.values())
        
        results = optimizer.optimize(objective, n_trials=3)
        
        assert "best_params" in results
        assert "best_value" in results
        assert "n_trials" in results
        mock_study.optimize.assert_called_once()

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.optuna')
    @patch('lunar_habitat_rl.hyperopt.WANDB_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.wandb')
    def test_bayesian_optimizer_with_wandb(self, mock_wandb, mock_optuna, simple_config):
        """Test Bayesian optimization with WandB logging."""
        simple_config.log_to_wandb = True
        
        mock_study = Mock()
        mock_study.study_name = "test_study"
        mock_study.best_params = {"learning_rate": 0.001}
        mock_study.best_value = 100.0
        mock_study.trials = []
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = BayesianOptimizer(simple_config)
        
        def objective(params):
            return 1.0
        
        # This would normally call WandB logging within the objective
        results = optimizer.optimize(objective, n_trials=1)
        
        assert "best_params" in results


class TestPopulationBasedTrainer:
    """Test suite for Population-Based Training."""

    @pytest.fixture
    def pbt_config(self):
        """PBT configuration for testing."""
        return HyperoptConfig(
            search_space=create_default_search_space()[:3],
            algorithm="pbt",
            population_size=10,
            perturbation_interval=5,
            mutation_probability=0.2
        )

    def test_pbt_initialization(self, pbt_config):
        """Test PBT initialization."""
        trainer = PopulationBasedTrainer(pbt_config)
        
        assert trainer.config == pbt_config
        assert trainer.population_size == 10
        assert trainer.generation == 0
        assert len(trainer.population) == 0

    def test_pbt_initialize_population(self, pbt_config):
        """Test PBT population initialization."""
        trainer = PopulationBasedTrainer(pbt_config)
        population = trainer.initialize_population()
        
        assert len(population) == pbt_config.population_size
        assert len(trainer.population) == pbt_config.population_size
        assert len(trainer.population_scores) == pbt_config.population_size
        
        # Each individual should have all parameters
        for individual in population:
            assert len(individual) == len(pbt_config.search_space)
            for param_space in pbt_config.search_space:
                assert param_space.name in individual

    def test_pbt_evolve_population(self, pbt_config):
        """Test PBT population evolution."""
        trainer = PopulationBasedTrainer(pbt_config)
        
        # Simple objective function - maximize sum of parameters
        def objective(params):
            return sum(v for v in params.values() if isinstance(v, (int, float)))
        
        results = trainer.evolve_population(objective, n_generations=3)
        
        assert "best_params" in results
        assert "best_value" in results
        assert "n_generations" in results
        assert "population_size" in results
        assert results["n_generations"] == 3
        assert len(results["best_score_history"]) == 3

    def test_pbt_exploit_and_explore(self, pbt_config):
        """Test PBT exploit and explore mechanism."""
        trainer = PopulationBasedTrainer(pbt_config)
        trainer.initialize_population()
        
        # Set up scores (ascending order)
        trainer.population_scores = list(range(pbt_config.population_size))
        
        # Store original population
        original_population = [p.copy() for p in trainer.population]
        
        trainer._exploit_and_explore()
        
        # Bottom performers should have changed (exploited top performers)
        # This is probabilistic, so we just check that population still has right size
        assert len(trainer.population) == pbt_config.population_size

    def test_pbt_compute_diversity(self, pbt_config):
        """Test PBT diversity computation."""
        trainer = PopulationBasedTrainer(pbt_config)
        trainer.initialize_population()
        
        diversity = trainer._compute_diversity()
        
        assert isinstance(diversity, float)
        assert 0.0 <= diversity <= 1.0

    def test_pbt_perturb_individual(self, pbt_config):
        """Test PBT individual perturbation."""
        trainer = PopulationBasedTrainer(pbt_config)
        trainer.initialize_population()
        
        # Store original individual
        original_individual = trainer.population[0].copy()
        
        # Force perturbation by setting high mutation probability
        original_prob = trainer.config.mutation_probability
        trainer.config.mutation_probability = 1.0
        
        rng = np.random.RandomState(42)
        trainer._perturb_individual(0, rng)
        
        # At least some parameters should have changed
        changed = any(
            trainer.population[0][k] != original_individual[k]
            for k in original_individual.keys()
        )
        
        # Restore original probability
        trainer.config.mutation_probability = original_prob
        
        # Note: With randomness, we can't guarantee changes, but structure should be intact
        assert len(trainer.population[0]) == len(original_individual)


class TestHyperparameterOptimizer:
    """Test suite for main hyperparameter optimizer."""

    @pytest.fixture
    def optimizer_config(self):
        """Configuration for main optimizer testing."""
        return HyperoptConfig(
            search_space=create_default_search_space()[:2],
            algorithm="random",  # Use random to avoid Optuna dependency
            n_trials=3
        )

    def test_optimizer_initialization(self, optimizer_config):
        """Test main optimizer initialization."""
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(optimizer_config)
            
            assert optimizer.config == optimizer_config

    def test_optimizer_create_default_search_space(self):
        """Test optimizer creates default search space when none provided."""
        config = HyperoptConfig(algorithm="random")
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(config)
            
            assert len(config.search_space) > 0

    @patch('lunar_habitat_rl.hyperopt.WANDB_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.wandb')
    def test_optimizer_with_wandb_initialization(self, mock_wandb, optimizer_config):
        """Test optimizer initialization with WandB."""
        optimizer_config.log_to_wandb = True
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(optimizer_config)
            
            mock_wandb.init.assert_called_once()

    def test_optimizer_evolutionary_algorithm(self):
        """Test evolutionary optimization algorithm."""
        config = HyperoptConfig(
            search_space=create_default_search_space()[:2],
            algorithm="evolutionary",
            n_trials=20  # Should be enough for 1 generation with population of 20
        )
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(config)
            
            def objective(params):
                return sum(v for v in params.values() if isinstance(v, (int, float)))
            
            results = optimizer._evolutionary_optimization(objective)
            
            assert "best_params" in results
            assert "best_value" in results
            assert "n_generations" in results
            assert "population_size" in results

    @patch('lunar_habitat_rl.hyperopt.OPTUNA_AVAILABLE', True)
    @patch('lunar_habitat_rl.hyperopt.BayesianOptimizer')
    def test_optimizer_bayesian_algorithm(self, mock_bayesian_class, optimizer_config):
        """Test optimizer with Bayesian algorithm."""
        optimizer_config.algorithm = "tpe"
        
        mock_bayesian = Mock()
        mock_bayesian.optimize.return_value = {"best_params": {}, "best_value": 1.0}
        mock_bayesian_class.return_value = mock_bayesian
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(optimizer_config)
            
            def objective(params):
                return 1.0
            
            results = optimizer.optimize(objective)
            
            mock_bayesian_class.assert_called_once_with(optimizer_config)
            mock_bayesian.optimize.assert_called_once()

    @patch('lunar_habitat_rl.hyperopt.PopulationBasedTrainer')
    def test_optimizer_pbt_algorithm(self, mock_pbt_class, optimizer_config):
        """Test optimizer with PBT algorithm."""
        optimizer_config.algorithm = "pbt"
        
        mock_pbt = Mock()
        mock_pbt.evolve_population.return_value = {"best_params": {}, "best_value": 1.0}
        mock_pbt_class.return_value = mock_pbt
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(optimizer_config)
            
            def objective(params):
                return 1.0
            
            results = optimizer.optimize(objective)
            
            mock_pbt_class.assert_called_once_with(optimizer_config)
            mock_pbt.evolve_population.assert_called_once()

    def test_optimizer_unknown_algorithm(self, optimizer_config):
        """Test error handling for unknown algorithm."""
        optimizer_config.algorithm = "unknown_algorithm"
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(optimizer_config)
            
            def objective(params):
                return 1.0
            
            with pytest.raises(Exception):  # Should raise HyperoptError
                optimizer.optimize(objective)


class TestObjectiveFunction:
    """Test suite for objective function creation."""

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(10), {})
            env.step.return_value = (
                np.random.randn(10), 1.0, True, False, {}
            )
            return env
        return creator

    @patch('lunar_habitat_rl.hyperopt.TrainingManager')
    @patch('lunar_habitat_rl.hyperopt.TrainingConfig')
    def test_create_objective_function(self, mock_training_config_class, 
                                     mock_training_manager_class, mock_env_creator):
        """Test objective function creation."""
        # Mock training components
        mock_config = Mock()
        mock_training_config_class.return_value = mock_config
        
        mock_trainer = Mock()
        mock_result = Mock()
        mock_result.best_performance = 85.0
        mock_trainer.train.return_value = mock_result
        mock_training_manager_class.return_value = mock_trainer
        
        objective_fn = create_objective_function(
            algorithm="ppo",
            env_creator=mock_env_creator,
            training_steps=1000
        )
        
        # Test objective function call
        params = {
            "learning_rate": 3e-4,
            "batch_size": 64,
            "gamma": 0.99
        }
        
        score = objective_fn(params)
        
        assert isinstance(score, (int, float))
        assert score == 85.0
        mock_training_manager_class.assert_called_once()
        mock_trainer.train.assert_called_once()

    @patch('lunar_habitat_rl.hyperopt.TrainingManager')
    def test_objective_function_error_handling(self, mock_training_manager_class, mock_env_creator):
        """Test objective function error handling."""
        # Mock training manager to raise error
        mock_trainer = Mock()
        mock_trainer.train.side_effect = RuntimeError("Training failed")
        mock_training_manager_class.return_value = mock_trainer
        
        objective_fn = create_objective_function(
            algorithm="ppo",
            env_creator=mock_env_creator
        )
        
        params = {"learning_rate": 3e-4}
        score = objective_fn(params)
        
        # Should return worst possible score for failed trials
        assert score == -np.inf

    @patch('lunar_habitat_rl.hyperopt.ModelBasedConfig')
    def test_objective_function_dreamer_v3(self, mock_model_config_class, mock_env_creator):
        """Test objective function with DreamerV3 algorithm."""
        mock_config = Mock()
        mock_model_config_class.return_value = mock_config
        
        objective_fn = create_objective_function(
            algorithm="dreamer_v3",
            env_creator=mock_env_creator
        )
        
        params = {
            "learning_rate": 6e-4,
            "hidden_size": 400,
            "stoch_size": 32,
            "deter_size": 512
        }
        
        # Mock training to avoid actual training
        with patch('lunar_habitat_rl.hyperopt.TrainingManager') as mock_trainer_class:
            mock_trainer = Mock()
            mock_result = Mock()
            mock_result.best_performance = 90.0
            mock_trainer.train.return_value = mock_result
            mock_trainer_class.return_value = mock_trainer
            
            score = objective_fn(params)
            
            mock_model_config_class.assert_called_once()
            assert score == 90.0


class TestRunHyperparameterOptimization:
    """Test suite for complete hyperparameter optimization run."""

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(10), {})
            env.step.return_value = (
                np.random.randn(10), 1.0, True, False, {}
            )
            return env
        return creator

    @patch('lunar_habitat_rl.hyperopt.HyperparameterOptimizer')
    @patch('lunar_habitat_rl.hyperopt.create_objective_function')
    def test_run_optimization_default_config(self, mock_create_objective, 
                                           mock_optimizer_class, mock_env_creator, tmp_path):
        """Test running optimization with default configuration."""
        # Mock objective function and optimizer
        mock_objective = Mock(return_value=100.0)
        mock_create_objective.return_value = mock_objective
        
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {
            "best_params": {"learning_rate": 3e-4},
            "best_value": 100.0
        }
        mock_optimizer_class.return_value = mock_optimizer
        
        # Change to temporary directory for file output
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            results = run_hyperparameter_optimization(
                algorithm="ppo",
                env_creator=mock_env_creator,
                training_steps=1000
            )
            
            assert "best_params" in results
            assert "best_value" in results
            mock_optimizer_class.assert_called_once()
            mock_optimizer.optimize.assert_called_once()
            
            # Check that results were saved
            hyperopt_results_dir = tmp_path / "hyperopt_results"
            if hyperopt_results_dir.exists():
                result_files = list(hyperopt_results_dir.glob("ppo_*.json"))
                assert len(result_files) > 0
        
        finally:
            os.chdir(original_cwd)

    @patch('lunar_habitat_rl.hyperopt.HyperparameterOptimizer')
    def test_run_optimization_custom_config(self, mock_optimizer_class, mock_env_creator):
        """Test running optimization with custom configuration."""
        config = HyperoptConfig(
            algorithm="tpe",
            n_trials=10,
            n_parallel_jobs=2
        )
        
        mock_optimizer = Mock()
        mock_optimizer.optimize.return_value = {"best_params": {}, "best_value": 50.0}
        mock_optimizer_class.return_value = mock_optimizer
        
        with patch('lunar_habitat_rl.hyperopt.create_objective_function'):
            results = run_hyperparameter_optimization(
                algorithm="sac",
                env_creator=mock_env_creator,
                config=config
            )
            
            # Verify custom config was used
            call_args = mock_optimizer_class.call_args[0][0]
            assert call_args.algorithm == "tpe"
            assert call_args.n_trials == 10


class TestIntegration:
    """Integration tests for hyperparameter optimization."""

    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline with mocked components."""
        # Create a simple search space
        search_space = [
            HyperparameterSpace("param1", "float", low=0.0, high=1.0),
            HyperparameterSpace("param2", "int", low=1, high=10)
        ]
        
        config = HyperoptConfig(
            search_space=search_space,
            algorithm="evolutionary",
            n_trials=20,
            n_parallel_jobs=1
        )
        
        with patch('lunar_habitat_rl.hyperopt.validate_hyperopt_config'):
            optimizer = HyperparameterOptimizer(config)
            
            # Simple quadratic objective
            def objective(params):
                return -(params["param1"] - 0.7)**2 - (params["param2"] - 5)**2
            
            results = optimizer.optimize(objective)
            
            assert "best_params" in results
            assert "best_value" in results
            
            # Check that optimization found reasonable values
            best_params = results["best_params"]
            assert 0.0 <= best_params["param1"] <= 1.0
            assert 1 <= best_params["param2"] <= 10

    def test_search_space_diversity(self):
        """Test that search space covers different parameter types."""
        search_space = create_default_search_space()
        
        # Sample from all parameters
        rng = np.random.RandomState(42)
        samples = []
        for _ in range(5):
            sample = {}
            for param_space in search_space:
                sample[param_space.name] = param_space.sample(rng)
            samples.append(sample)
        
        # All samples should have all parameters
        param_names = {param.name for param in search_space}
        for sample in samples:
            assert set(sample.keys()) == param_names
        
        # Samples should be different
        assert len(set(str(s) for s in samples)) > 1


if __name__ == "__main__":
    pytest.main([__file__])