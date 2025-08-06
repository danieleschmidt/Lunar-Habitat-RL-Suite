"""Tests for environment implementations."""

import pytest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

from lunar_habitat_rl.environments import LunarHabitatEnv, make_lunar_env
from lunar_habitat_rl.core import HabitatConfig
from lunar_habitat_rl.utils.exceptions import ValidationError, SafetyError
from tests import TEST_CONFIG, assert_array_close, MockPhysicsSimulator


class TestLunarHabitatEnv:
    """Test the main lunar habitat environment."""
    
    def test_environment_creation(self):
        """Test basic environment creation."""
        env = LunarHabitatEnv()
        
        assert env is not None
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.config is not None
        assert isinstance(env.config, HabitatConfig)
    
    def test_environment_with_custom_config(self):
        """Test environment creation with custom configuration."""
        config = HabitatConfig(
            volume=150.0,
            crew={"size": 6}
        )
        
        env = LunarHabitatEnv(config=config)
        
        assert env.config.volume == 150.0
        assert env.config.crew.size == 6
    
    def test_environment_with_preset_config(self):
        """Test environment creation with preset configuration."""
        env = LunarHabitatEnv(config="nasa_reference")
        
        assert env.config.name == "nasa_reference_habitat"
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = LunarHabitatEnv(crew_size=4)
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))
        
        assert isinstance(info, dict)
        assert 'mission_time' in info
        assert info['mission_time'] == 0.0
        
        # Test reset determinism with same seed
        obs2, info2 = env.reset(seed=TEST_CONFIG['random_seed'])
        assert_array_close(obs, obs2)
    
    def test_environment_step(self):
        """Test environment step functionality."""
        env = LunarHabitatEnv(crew_size=4)
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Test valid action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(next_obs))
        
        assert isinstance(reward, (float, int))
        assert np.isfinite(reward)
        
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
        
        assert 'mission_time' in step_info
        assert step_info['mission_time'] > 0
    
    def test_environment_action_validation(self):
        """Test action validation in environment."""
        env = LunarHabitatEnv()
        env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Test invalid action shapes
        with pytest.raises((ValueError, ValidationError)):
            env.step(np.array([1.0]))  # Too few dimensions
        
        with pytest.raises((ValueError, ValidationError)):
            env.step(np.array([np.inf, 0.5, 0.5]))  # Contains infinity
        
        with pytest.raises((ValueError, ValidationError)):
            env.step(np.array([np.nan, 0.5, 0.5]))  # Contains NaN
    
    def test_environment_reward_functions(self):
        """Test different reward function configurations.""" 
        reward_configs = ["survival_focused", "efficiency_focused", "balanced"]
        
        for reward_config in reward_configs:
            env = LunarHabitatEnv(reward_config=reward_config)
            obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
            
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            assert isinstance(reward, (float, int))
            assert np.isfinite(reward)
    
    def test_environment_termination_conditions(self):
        """Test environment termination conditions."""
        env = LunarHabitatEnv()
        
        # Test normal operation doesn't terminate immediately
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        action = env.action_space.sample()
        action = np.clip(action, 0.3, 0.7)  # Safe action range
        
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert not terminated  # Should not terminate on first step with safe actions
    
    def test_environment_with_physics_simulation(self):
        """Test environment with physics simulation enabled."""
        env = LunarHabitatEnv(physics_enabled=True)
        
        # Mock physics simulators to avoid dependencies
        env.thermal_sim = MockPhysicsSimulator()
        env.cfd_sim = MockPhysicsSimulator() 
        env.chemistry_sim = MockPhysicsSimulator()
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Should work with mocked physics
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (float, int))
    
    def test_environment_without_physics_simulation(self):
        """Test environment with physics simulation disabled."""
        env = LunarHabitatEnv(physics_enabled=False)
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Should work with simplified physics
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, (float, int))
    
    def test_environment_episode_length(self):
        """Test environment episode length limits.""" 
        env = LunarHabitatEnv()
        env.max_steps = 10  # Set short episode for testing
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        for step in range(15):  # More than max_steps
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should truncate at max_steps
        assert step >= 9  # Should reach near max_steps
        assert truncated or terminated
    
    def test_environment_info_dict(self):
        """Test environment info dictionary contents."""
        env = LunarHabitatEnv(crew_size=4)
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Check initial info
        required_keys = ['mission_time', 'steps_taken', 'crew_size', 'scenario', 'difficulty']
        for key in required_keys:
            assert key in info
        
        # Check step info
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        for key in required_keys:
            assert key in step_info
        
        assert step_info['steps_taken'] > info['steps_taken']
    
    def test_environment_different_crew_sizes(self):
        """Test environment with different crew sizes."""
        crew_sizes = [2, 4, 6, 8]
        
        for crew_size in crew_sizes:
            env = LunarHabitatEnv(crew_size=crew_size)
            obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
            
            assert env.config.crew.size == crew_size
            assert info['crew_size'] == crew_size
            
            # Check observation space adapts to crew size
            assert obs.shape == env.observation_space.shape
    
    def test_environment_different_scenarios(self):
        """Test environment with different scenarios."""
        scenarios = ["nominal_operations", "emergency_scenarios", "long_term_mission"]
        
        for scenario in scenarios:
            env = LunarHabitatEnv(scenario=scenario)
            obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
            
            assert info['scenario'] == scenario
    
    def test_environment_render(self):
        """Test environment rendering functionality."""
        env = LunarHabitatEnv(render_mode="human")
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Render should not raise exceptions
        rendered = env.render()
        # For text-based rendering, this returns None
        
        env.close()
    
    def test_environment_close(self):
        """Test environment cleanup."""
        env = LunarHabitatEnv()
        env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Should not raise exceptions
        env.close()


class TestMakeLunarEnv:
    """Test the vectorized environment creation function."""
    
    def test_make_single_env(self):
        """Test creating single environment."""
        vec_env = make_lunar_env(n_envs=1, crew_size=4)
        
        assert vec_env is not None
        assert vec_env.num_envs == 1
        
        observations = vec_env.reset()
        assert len(observations) == 2  # obs, infos
        assert len(observations[0]) == 1  # Single environment
        
        vec_env.close()
    
    def test_make_multiple_envs(self):
        """Test creating multiple environments."""
        n_envs = 4
        vec_env = make_lunar_env(n_envs=n_envs, crew_size=4, parallel=False)  # Use sequential for testing
        
        assert vec_env.num_envs == n_envs
        
        observations = vec_env.reset()
        assert len(observations[0]) == n_envs
        
        # Test stepping
        actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
        next_obs, rewards, dones, truncateds, infos = vec_env.step(actions)
        
        assert len(next_obs) == n_envs
        assert len(rewards) == n_envs
        assert len(dones) == n_envs
        assert len(infos) == n_envs
        
        vec_env.close()
    
    def test_make_env_with_config(self):
        """Test creating vectorized environment with configuration."""
        config = HabitatConfig(volume=150.0)
        vec_env = make_lunar_env(n_envs=2, config=config, parallel=False)
        
        assert vec_env.num_envs == 2
        
        vec_env.close()
    
    def test_make_env_different_scenarios(self):
        """Test vectorized environment with different scenarios."""
        scenarios = ["nominal_operations", "emergency_scenarios"]
        
        for scenario in scenarios:
            vec_env = make_lunar_env(
                n_envs=2, 
                scenario=scenario, 
                parallel=False
            )
            
            observations = vec_env.reset()
            assert len(observations[0]) == 2
            
            vec_env.close()
    
    def test_make_env_with_seed(self):
        """Test vectorized environment with reproducible seeding."""
        seed = TEST_CONFIG['random_seed']
        
        # Create two identical environments
        vec_env1 = make_lunar_env(n_envs=2, seed=seed, parallel=False)
        vec_env2 = make_lunar_env(n_envs=2, seed=seed, parallel=False)
        
        obs1 = vec_env1.reset()
        obs2 = vec_env2.reset()
        
        # Should produce same initial observations (approximately)
        assert_array_close(obs1[0], obs2[0], rtol=1e-3)
        
        vec_env1.close()
        vec_env2.close()


class TestEnvironmentIntegration:
    """Integration tests for environment components."""
    
    def test_environment_physics_integration(self):
        """Test integration with physics simulators."""
        env = LunarHabitatEnv(physics_enabled=True)
        
        # Mock the physics simulators
        mock_thermal = Mock()
        mock_thermal.step.return_value = {
            'zone_temperatures': [22.0, 23.0, 22.5, 21.8],
            'radiator_temperatures': [15.0, 16.0],
            'heat_pump_cop': 3.2,
            'total_heat_loss': 1000.0,
            'average_temperature': 22.3,
            'temperature_variance': 0.5
        }
        
        mock_cfd = Mock()
        mock_cfd.step.return_value = {
            'mixing_efficiency': 0.9,
            'air_exchange_rate': 2.5
        }
        
        mock_chemistry = Mock()
        mock_chemistry.step.return_value = {
            'o2_pressure': 21.0,
            'co2_pressure': 0.4,
            'n2_pressure': 79.0,
            'total_pressure': 100.4
        }
        
        env.thermal_sim = mock_thermal
        env.cfd_sim = mock_cfd
        env.chemistry_sim = mock_chemistry
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Verify physics simulators were called
        assert mock_thermal.step.called
        assert mock_cfd.step.called
        assert mock_chemistry.step.called
        
        # Verify results are incorporated into state
        assert isinstance(next_obs, np.ndarray)
        assert np.all(np.isfinite(next_obs))
    
    def test_environment_performance_tracking(self):
        """Test integration with performance tracking."""
        env = LunarHabitatEnv()
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Run several steps
        for _ in range(10):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Check performance tracker has recorded metrics
        stats = env.performance_tracker.get_statistics()
        
        assert 'total_steps' in stats
        assert stats['total_steps'] > 0
        
        if stats['avg_reward'] != 0.0:  # May be 0 for some reward functions
            assert isinstance(stats['avg_reward'], (int, float))
    
    def test_environment_safety_violations(self):
        """Test detection of safety violations."""
        env = LunarHabitatEnv()
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        # Manually trigger unsafe conditions for testing
        env.habitat_state.atmosphere.o2_partial_pressure = 10.0  # Dangerously low
        env.habitat_state.atmosphere.co2_partial_pressure = 1.5  # Dangerously high
        
        # Take a step
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        # Should detect safety issues
        stats = env.performance_tracker.get_statistics()
        assert stats['total_safety_violations'] > 0
    
    @pytest.mark.slow
    def test_environment_long_episode(self):
        """Test environment stability over long episodes."""
        env = LunarHabitatEnv()
        env.max_steps = 1000  # Longer episode
        
        obs, info = env.reset(seed=TEST_CONFIG['random_seed'])
        
        steps = 0
        total_reward = 0.0
        
        while steps < 100:  # Test subset for speed
            action = env.action_space.sample()
            # Bias actions towards safe middle range
            action = np.clip(action, 0.3, 0.7)
            
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            assert np.all(np.isfinite(next_obs))
            assert np.isfinite(reward)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        assert np.isfinite(total_reward)