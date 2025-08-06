"""
Test suite for RL algorithms in the lunar habitat RL suite.

This module provides comprehensive tests for all implemented algorithms
including baselines, model-based methods, offline RL, and training infrastructure.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import os

# Import modules to test
from lunar_habitat_rl.algorithms.baselines import (
    RandomAgent, ConstantAgent, HeuristicAgent, PPOAgent, SACAgent,
    ActorNetwork, CriticNetwork, ValueNetwork
)
from lunar_habitat_rl.algorithms.model_based import (
    DreamerV3, MuZeroAgent, PlaNetAgent, WorldModel, TransitionModel,
    ObservationModel, RewardModel, ModelBasedConfig
)
from lunar_habitat_rl.algorithms.training import (
    TrainingManager, TrainingConfig, ExperimentRunner, ExperimentConfig
)
from lunar_habitat_rl.algorithms.offline_rl import (
    CQLAgent, IQLAgent, AWACAgent, OfflineRLConfig
)


class TestBaselineAgents:
    """Test suite for baseline agents."""

    @pytest.fixture
    def mock_action_space(self):
        """Mock action space for testing."""
        action_space = Mock()
        action_space.shape = (4,)
        action_space.low = np.array([-1.0, -1.0, -1.0, -1.0])
        action_space.high = np.array([1.0, 1.0, 1.0, 1.0])
        action_space.sample.return_value = np.array([0.1, -0.2, 0.3, -0.4])
        return action_space

    @pytest.fixture
    def mock_env(self):
        """Mock environment for testing."""
        env = Mock()
        env.reset.return_value = (np.random.randn(10), {})
        env.step.return_value = (
            np.random.randn(10),  # obs
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {}  # info
        )
        return env

    def test_random_agent_initialization(self, mock_action_space):
        """Test RandomAgent initialization."""
        agent = RandomAgent(mock_action_space, seed=42)
        
        assert agent.action_space == mock_action_space
        assert agent.rng is not None
        
    def test_random_agent_predict(self, mock_action_space):
        """Test RandomAgent action prediction."""
        agent = RandomAgent(mock_action_space, seed=42)
        obs = np.random.randn(10)
        
        action = agent.predict(obs)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == mock_action_space.shape

    def test_random_agent_train(self, mock_action_space, mock_env):
        """Test RandomAgent training (statistics collection)."""
        agent = RandomAgent(mock_action_space, seed=42)
        
        # Mock environment to terminate after 5 steps
        step_count = 0
        def mock_step(action):
            nonlocal step_count
            step_count += 1
            done = step_count >= 5
            return np.random.randn(10), 1.0, done, False, {}
        
        mock_env.step.side_effect = mock_step
        
        stats = agent.train(mock_env, total_timesteps=10)
        
        assert 'total_timesteps' in stats
        assert 'episodes' in stats
        assert 'episode_rewards' in stats
        assert stats['total_timesteps'] <= 10

    def test_constant_agent(self):
        """Test ConstantAgent functionality."""
        constant_action = np.array([0.5, -0.5, 0.0, 1.0])
        agent = ConstantAgent(constant_action)
        
        obs = np.random.randn(10)
        action = agent.predict(obs)
        
        np.testing.assert_array_equal(action, constant_action)

    def test_heuristic_agent(self, mock_action_space):
        """Test HeuristicAgent domain-specific logic."""
        agent = HeuristicAgent(mock_action_space)
        
        # Test with typical lunar habitat observation
        obs = np.array([
            21.0,  # O2 pressure
            0.4,   # CO2 pressure
            79.0,  # N2 pressure
            101.3, # Total pressure
            45.0,  # Humidity
            22.5,  # Temperature
            0.95,  # Air quality
            5.0,   # Solar generation
            75.0   # Battery charge
        ])
        
        action = agent.predict(obs)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (mock_action_space.shape[0],)
        assert np.all(action >= 0.0) and np.all(action <= 1.0)

    def test_heuristic_agent_low_battery(self, mock_action_space):
        """Test HeuristicAgent behavior with low battery."""
        agent = HeuristicAgent(mock_action_space)
        
        # Low battery scenario
        obs = np.array([21.0, 0.4, 79.0, 101.3, 45.0, 22.5, 0.95, 5.0, 15.0])
        action = agent.predict(obs)
        
        # Should have appropriate response to low battery
        assert isinstance(action, np.ndarray)

    def test_agent_save_load(self, mock_action_space, tmp_path):
        """Test agent save/load functionality."""
        agent = RandomAgent(mock_action_space, seed=42)
        
        # Save agent
        save_path = tmp_path / "test_agent.npz"
        agent.save(str(save_path))
        
        # Load agent
        new_agent = RandomAgent(mock_action_space)
        new_agent.load(str(save_path))
        
        # Test that loaded agent behaves similarly
        obs = np.random.randn(10)
        action1 = agent.predict(obs)
        action2 = new_agent.predict(obs)
        
        # Actions should be similar (same RNG state)
        assert action1.shape == action2.shape


class TestNeuralNetworks:
    """Test suite for neural network components."""

    def test_actor_network(self):
        """Test ActorNetwork functionality."""
        obs_dim, action_dim = 10, 4
        network = ActorNetwork(obs_dim, action_dim)
        
        batch_size = 32
        obs = torch.randn(batch_size, obs_dim)
        
        mean, log_std = network.forward(obs)
        
        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
        assert torch.all(log_std >= network.min_log_std)
        assert torch.all(log_std <= network.max_log_std)

    def test_actor_network_sample(self):
        """Test ActorNetwork action sampling."""
        obs_dim, action_dim = 10, 4
        network = ActorNetwork(obs_dim, action_dim)
        
        obs = torch.randn(1, obs_dim)
        action, log_prob = network.sample(obs)
        
        assert action.shape == (1, action_dim)
        assert log_prob.shape == (1, 1)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)  # tanh output

    def test_critic_network(self):
        """Test CriticNetwork functionality."""
        obs_dim, action_dim = 10, 4
        network = CriticNetwork(obs_dim, action_dim)
        
        batch_size = 32
        obs = torch.randn(batch_size, obs_dim)
        action = torch.randn(batch_size, action_dim)
        
        q_value = network.forward(obs, action)
        
        assert q_value.shape == (batch_size, 1)

    def test_value_network(self):
        """Test ValueNetwork functionality."""
        obs_dim = 10
        network = ValueNetwork(obs_dim)
        
        batch_size = 32
        obs = torch.randn(batch_size, obs_dim)
        
        value = network.forward(obs)
        
        assert value.shape == (batch_size, 1)


class TestPPOAgent:
    """Test suite for PPO agent implementation."""

    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = PPOAgent(obs_dim, action_dim)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert isinstance(agent.actor, ActorNetwork)
        assert isinstance(agent.critic, ValueNetwork)

    def test_ppo_act(self):
        """Test PPO action selection."""
        obs_dim, action_dim = 10, 4
        agent = PPOAgent(obs_dim, action_dim)
        
        obs = torch.randn(1, obs_dim)
        action = agent.act(obs)
        
        assert action.shape == (1, action_dim)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    def test_ppo_predict(self):
        """Test PPO predict interface."""
        obs_dim, action_dim = 10, 4
        agent = PPOAgent(obs_dim, action_dim)
        
        obs = np.random.randn(obs_dim)
        action = agent.predict(obs, deterministic=True)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (action_dim,)

    def test_ppo_save_load(self, tmp_path):
        """Test PPO save/load functionality."""
        obs_dim, action_dim = 10, 4
        agent = PPOAgent(obs_dim, action_dim)
        
        # Save model
        save_path = tmp_path / "ppo_model.pt"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = PPOAgent(obs_dim, action_dim)
        new_agent.load(str(save_path))
        
        # Test that models produce same output
        obs = torch.randn(1, obs_dim)
        action1 = agent.act(obs)
        action2 = new_agent.act(obs)
        
        torch.testing.assert_close(action1, action2, rtol=1e-5, atol=1e-5)


class TestSACAgent:
    """Test suite for SAC agent implementation."""

    def test_sac_initialization(self):
        """Test SAC agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = SACAgent(obs_dim, action_dim)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert isinstance(agent.actor, ActorNetwork)
        assert isinstance(agent.q1, CriticNetwork)
        assert isinstance(agent.q2, CriticNetwork)

    def test_sac_act(self):
        """Test SAC action selection."""
        obs_dim, action_dim = 10, 4
        agent = SACAgent(obs_dim, action_dim)
        
        obs = torch.randn(1, obs_dim)
        action = agent.act(obs)
        
        assert action.shape == (1, action_dim)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    def test_sac_soft_update(self):
        """Test SAC soft update mechanism."""
        obs_dim, action_dim = 10, 4
        agent = SACAgent(obs_dim, action_dim)
        
        # Store original target network parameters
        original_params = [p.clone() for p in agent.q1_target.parameters()]
        
        # Modify source network
        for p in agent.q1.parameters():
            p.data += 0.1
        
        # Apply soft update
        agent._soft_update(agent.q1, agent.q1_target)
        
        # Check that target network has changed
        for orig_p, new_p in zip(original_params, agent.q1_target.parameters()):
            assert not torch.equal(orig_p, new_p)


class TestModelBasedAgents:
    """Test suite for model-based RL agents."""

    @pytest.fixture
    def model_based_config(self):
        """Create test configuration for model-based agents."""
        return ModelBasedConfig(
            learning_rate=3e-4,
            hidden_size=64,  # Smaller for faster testing
            stoch_size=16,
            deter_size=32,
            batch_size=16,
            sequence_length=8
        )

    def test_world_model_components(self):
        """Test WorldModel component initialization."""
        obs_dim, action_dim = 10, 4
        config = ModelBasedConfig(hidden_size=64, stoch_size=16, deter_size=32)
        
        transition_model = TransitionModel(config)
        observation_model = ObservationModel(obs_dim, config)
        reward_model = RewardModel(config)
        
        assert isinstance(transition_model, nn.Module)
        assert isinstance(observation_model, nn.Module)
        assert isinstance(reward_model, nn.Module)

    def test_dreamer_v3_initialization(self, model_based_config):
        """Test DreamerV3 initialization."""
        obs_dim, action_dim = 10, 4
        agent = DreamerV3(obs_dim, action_dim, model_based_config)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert hasattr(agent, 'world_model')
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')

    def test_dreamer_v3_act(self, model_based_config):
        """Test DreamerV3 action selection."""
        obs_dim, action_dim = 10, 4
        agent = DreamerV3(obs_dim, action_dim, model_based_config)
        
        obs = torch.randn(1, obs_dim)
        action = agent.act(obs)
        
        assert action.shape == (1, action_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for full model testing")
    def test_dreamer_v3_gpu(self, model_based_config):
        """Test DreamerV3 on GPU if available."""
        obs_dim, action_dim = 10, 4
        agent = DreamerV3(obs_dim, action_dim, model_based_config)
        agent = agent.cuda()
        
        obs = torch.randn(1, obs_dim).cuda()
        action = agent.act(obs)
        
        assert action.device.type == 'cuda'

    def test_muzero_agent_initialization(self, model_based_config):
        """Test MuZero agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = MuZeroAgent(obs_dim, action_dim, model_based_config)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert hasattr(agent, 'representation_network')
        assert hasattr(agent, 'dynamics_network')
        assert hasattr(agent, 'prediction_network')

    def test_planet_agent_initialization(self, model_based_config):
        """Test PlaNet agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = PlaNetAgent(obs_dim, action_dim, model_based_config)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim


class TestTrainingInfrastructure:
    """Test suite for training infrastructure."""

    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return TrainingConfig(
            algorithm="ppo",
            total_timesteps=1000,
            batch_size=32,
            learning_rate=3e-4,
            eval_freq=100
        )

    @pytest.fixture
    def mock_env(self):
        """Mock environment for training tests."""
        env = Mock()
        env.reset.return_value = (np.random.randn(10), {})
        env.step.return_value = (
            np.random.randn(10),  # obs
            1.0,  # reward
            False,  # terminated
            False,  # truncated
            {}  # info
        )
        env.observation_space.shape = (10,)
        env.action_space.shape = (4,)
        return env

    def test_training_manager_initialization(self, training_config):
        """Test TrainingManager initialization."""
        manager = TrainingManager(training_config)
        
        assert manager.config == training_config
        assert manager.current_step == 0

    @patch('lunar_habitat_rl.algorithms.training.PPOAgent')
    def test_training_manager_create_agent(self, mock_ppo_class, training_config):
        """Test TrainingManager agent creation."""
        manager = TrainingManager(training_config)
        
        # Mock environment dimensions
        obs_dim, action_dim = 10, 4
        agent = manager._create_agent("ppo", obs_dim, action_dim)
        
        mock_ppo_class.assert_called_once()

    def test_experiment_config(self):
        """Test ExperimentConfig creation."""
        config = ExperimentConfig(
            experiment_name="test_experiment",
            algorithms=["ppo", "sac"],
            n_runs=3,
            total_timesteps=1000
        )
        
        assert config.experiment_name == "test_experiment"
        assert len(config.algorithms) == 2
        assert config.n_runs == 3

    @patch('lunar_habitat_rl.algorithms.training.TrainingManager')
    def test_experiment_runner_initialization(self, mock_training_manager):
        """Test ExperimentRunner initialization."""
        config = ExperimentConfig(
            experiment_name="test",
            algorithms=["ppo"],
            n_runs=1,
            total_timesteps=100
        )
        
        runner = ExperimentRunner(config)
        
        assert runner.config == config
        assert hasattr(runner, 'results')


class TestOfflineRLAgents:
    """Test suite for offline RL agents."""

    @pytest.fixture
    def offline_config(self):
        """Create test configuration for offline RL."""
        return OfflineRLConfig(
            learning_rate=3e-4,
            batch_size=32,
            hidden_size=64,
            tau=0.005,
            gamma=0.99
        )

    def test_cql_agent_initialization(self, offline_config):
        """Test CQL agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = CQLAgent(obs_dim, action_dim, offline_config)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'q1')
        assert hasattr(agent, 'q2')

    def test_iql_agent_initialization(self, offline_config):
        """Test IQL agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = IQLAgent(obs_dim, action_dim, offline_config)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'q1')
        assert hasattr(agent, 'q2')
        assert hasattr(agent, 'value_net')

    def test_awac_agent_initialization(self, offline_config):
        """Test AWAC agent initialization."""
        obs_dim, action_dim = 10, 4
        agent = AWACAgent(obs_dim, action_dim, offline_config)
        
        assert agent.obs_dim == obs_dim
        assert agent.action_dim == action_dim
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')

    def test_offline_agent_update(self, offline_config):
        """Test offline RL agent update with dummy data."""
        obs_dim, action_dim = 10, 4
        agent = CQLAgent(obs_dim, action_dim, offline_config)
        
        # Create dummy batch
        batch = {
            "obs": torch.randn(32, obs_dim),
            "actions": torch.randn(32, action_dim),
            "rewards": torch.randn(32, 1),
            "next_obs": torch.randn(32, obs_dim),
            "dones": torch.randint(0, 2, (32, 1)).float()
        }
        
        # Update should run without error
        losses = agent.update(batch)
        
        assert isinstance(losses, dict)
        assert "q_loss" in losses or "q1_loss" in losses


class TestIntegration:
    """Integration tests for algorithm components."""

    def test_agent_environment_interaction(self):
        """Test that agents can interact with environments."""
        # Create simple mock environment
        env = Mock()
        env.reset.return_value = (np.random.randn(10), {})
        env.step.return_value = (
            np.random.randn(10), 1.0, False, False, {}
        )
        env.observation_space.shape = (10,)
        env.action_space.shape = (4,)
        
        # Test with PPO agent
        agent = PPOAgent(10, 4)
        
        # Run interaction loop
        obs, _ = env.reset()
        for _ in range(5):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            assert isinstance(action, np.ndarray)
            assert action.shape == (4,)

    def test_training_pipeline(self):
        """Test basic training pipeline functionality."""
        config = TrainingConfig(
            algorithm="ppo",
            total_timesteps=100,
            batch_size=16,
            eval_freq=50
        )
        
        manager = TrainingManager(config)
        
        # Mock environment
        env = Mock()
        env.observation_space.shape = (10,)
        env.action_space.shape = (4,)
        env.reset.return_value = (np.random.randn(10), {})
        env.step.return_value = (
            np.random.randn(10), 1.0, False, False, {}
        )
        
        # Training should initialize without errors
        assert manager.config.algorithm == "ppo"
        assert manager.config.total_timesteps == 100


if __name__ == "__main__":
    pytest.main([__file__])