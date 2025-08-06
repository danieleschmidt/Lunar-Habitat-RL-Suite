"""
Test suite for evaluation capabilities in the lunar habitat RL suite.

This module provides comprehensive tests for the evaluation system
including benchmark scenarios, statistical analysis, and performance metrics.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import json

# Import modules to test
from lunar_habitat_rl.evaluation import (
    BenchmarkSuite, EvaluationMetrics, StatisticalAnalyzer,
    PerformanceProfiler, RobustnessEvaluator, TransferLearningEvaluator,
    run_benchmark_suite, create_evaluation_report
)


class TestBenchmarkSuite:
    """Test suite for benchmark scenario management."""

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator for testing."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (
                np.random.randn(20), 1.0, False, False, 
                {'safety_score': 0.9, 'efficiency_score': 0.8}
            )
            return env
        return creator

    def test_benchmark_suite_initialization(self, mock_env_creator):
        """Test BenchmarkSuite initialization."""
        suite = BenchmarkSuite(mock_env_creator)
        
        assert suite.env_creator == mock_env_creator
        assert len(suite.scenarios) > 0
        assert hasattr(suite, 'results_history')

    def test_default_scenarios_loaded(self, mock_env_creator):
        """Test that default scenarios are loaded correctly."""
        suite = BenchmarkSuite(mock_env_creator)
        
        # Check that expected scenarios are present
        scenario_names = [s['name'] for s in suite.scenarios]
        expected_scenarios = [
            'nominal_30_day', 'eclss_failure', 'thermal_stress',
            'power_shortage', 'dust_storm', 'micrometeorite_damage'
        ]
        
        for expected in expected_scenarios:
            assert expected in scenario_names

    def test_scenario_structure(self, mock_env_creator):
        """Test that scenarios have required structure."""
        suite = BenchmarkSuite(mock_env_creator)
        
        for scenario in suite.scenarios:
            assert 'name' in scenario
            assert 'description' in scenario
            assert 'duration' in scenario
            assert 'success_threshold' in scenario
            assert 'config' in scenario
            assert isinstance(scenario['config'], dict)

    def test_add_custom_scenario(self, mock_env_creator):
        """Test adding custom benchmark scenarios."""
        suite = BenchmarkSuite(mock_env_creator)
        
        initial_count = len(suite.scenarios)
        
        custom_scenario = {
            'name': 'custom_test',
            'description': 'Custom test scenario',
            'duration': 100,
            'success_threshold': 0.8,
            'config': {'test_param': 42}
        }
        
        suite.add_scenario(custom_scenario)
        
        assert len(suite.scenarios) == initial_count + 1
        assert custom_scenario in suite.scenarios

    def test_get_scenario_by_name(self, mock_env_creator):
        """Test retrieving scenarios by name."""
        suite = BenchmarkSuite(mock_env_creator)
        
        scenario = suite.get_scenario('nominal_30_day')
        
        assert scenario is not None
        assert scenario['name'] == 'nominal_30_day'

    def test_get_nonexistent_scenario(self, mock_env_creator):
        """Test handling of nonexistent scenario."""
        suite = BenchmarkSuite(mock_env_creator)
        
        scenario = suite.get_scenario('nonexistent_scenario')
        
        assert scenario is None


class TestEvaluationMetrics:
    """Test suite for evaluation metrics calculation."""

    def test_metrics_initialization(self):
        """Test EvaluationMetrics initialization."""
        metrics = EvaluationMetrics()
        
        assert hasattr(metrics, 'episode_rewards')
        assert hasattr(metrics, 'episode_lengths')
        assert hasattr(metrics, 'safety_violations')

    def test_add_episode_data(self):
        """Test adding episode data to metrics."""
        metrics = EvaluationMetrics()
        
        metrics.add_episode(
            reward=100.0,
            length=250,
            success=True,
            safety_score=0.95,
            custom_metric=42.0
        )
        
        assert len(metrics.episode_rewards) == 1
        assert metrics.episode_rewards[0] == 100.0
        assert metrics.episode_lengths[0] == 250
        assert metrics.success_rate == 1.0

    def test_multiple_episodes(self):
        """Test metrics with multiple episodes."""
        metrics = EvaluationMetrics()
        
        # Add multiple episodes
        episodes_data = [
            {'reward': 100.0, 'length': 200, 'success': True, 'safety_score': 0.9},
            {'reward': 80.0, 'length': 180, 'success': False, 'safety_score': 0.8},
            {'reward': 120.0, 'length': 220, 'success': True, 'safety_score': 0.95},
        ]
        
        for episode in episodes_data:
            metrics.add_episode(**episode)
        
        assert len(metrics.episode_rewards) == 3
        assert metrics.mean_reward == 100.0  # (100 + 80 + 120) / 3
        assert metrics.success_rate == 2/3  # 2 successes out of 3

    def test_statistical_properties(self):
        """Test statistical properties calculation."""
        metrics = EvaluationMetrics()
        
        # Add episodes with known statistics
        rewards = [10.0, 20.0, 30.0, 40.0, 50.0]
        for reward in rewards:
            metrics.add_episode(reward=reward, length=100, success=True)
        
        assert metrics.mean_reward == 30.0
        assert metrics.std_reward == np.std(rewards, ddof=1)
        assert metrics.min_reward == 10.0
        assert metrics.max_reward == 50.0

    def test_safety_metrics(self):
        """Test safety-specific metrics."""
        metrics = EvaluationMetrics()
        
        # Add episodes with safety violations
        safety_data = [
            {'safety_score': 0.9, 'has_violation': False},
            {'safety_score': 0.7, 'has_violation': True},
            {'safety_score': 0.95, 'has_violation': False},
            {'safety_score': 0.6, 'has_violation': True},
        ]
        
        for data in safety_data:
            metrics.add_episode(
                reward=100.0, length=200, success=True,
                safety_score=data['safety_score']
            )
            if data['has_violation']:
                metrics.safety_violations.append('test_violation')
        
        assert len(metrics.safety_violations) == 2
        assert metrics.mean_safety_score == 0.8  # (0.9 + 0.7 + 0.95 + 0.6) / 4

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = EvaluationMetrics()
        
        # Add some data
        for i in range(3):
            metrics.add_episode(reward=i*10, length=100+i*10, success=i%2==0)
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'mean_reward' in metrics_dict
        assert 'std_reward' in metrics_dict
        assert 'success_rate' in metrics_dict
        assert 'n_episodes' in metrics_dict


class TestStatisticalAnalyzer:
    """Test suite for statistical analysis capabilities."""

    @pytest.fixture
    def sample_results(self):
        """Sample benchmark results for testing."""
        return [
            {
                'algorithm': 'PPO',
                'scenario': 'nominal_30_day',
                'run_id': 0,
                'mean_reward': 85.0,
                'success_rate': 0.9,
                'episode_rewards': [80, 85, 90, 85, 80]
            },
            {
                'algorithm': 'PPO',
                'scenario': 'nominal_30_day',
                'run_id': 1,
                'mean_reward': 82.0,
                'success_rate': 0.8,
                'episode_rewards': [78, 82, 86, 82, 77]
            },
            {
                'algorithm': 'SAC',
                'scenario': 'nominal_30_day',
                'run_id': 0,
                'mean_reward': 90.0,
                'success_rate': 0.95,
                'episode_rewards': [88, 90, 92, 90, 90]
            }
        ]

    def test_analyzer_initialization(self):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer()
        
        assert hasattr(analyzer, 'results')
        assert hasattr(analyzer, 'analyses')

    def test_add_results(self, sample_results):
        """Test adding results to analyzer."""
        analyzer = StatisticalAnalyzer()
        
        for result in sample_results:
            analyzer.add_result(result)
        
        assert len(analyzer.results) == 3

    def test_compute_aggregate_statistics(self, sample_results):
        """Test computing aggregate statistics."""
        analyzer = StatisticalAnalyzer()
        
        for result in sample_results:
            analyzer.add_result(result)
        
        stats = analyzer.compute_aggregate_statistics()
        
        assert 'PPO' in stats
        assert 'SAC' in stats
        assert 'mean_reward_mean' in stats['PPO']['nominal_30_day']
        assert 'mean_reward_std' in stats['PPO']['nominal_30_day']

    def test_statistical_significance_test(self, sample_results):
        """Test statistical significance testing."""
        analyzer = StatisticalAnalyzer()
        
        for result in sample_results:
            analyzer.add_result(result)
        
        significance = analyzer.test_statistical_significance(
            algorithm_a='PPO',
            algorithm_b='SAC',
            scenario='nominal_30_day',
            metric='mean_reward'
        )
        
        assert 'p_value' in significance
        assert 'significant' in significance
        assert 'effect_size' in significance

    def test_confidence_intervals(self, sample_results):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer()
        
        for result in sample_results:
            analyzer.add_result(result)
        
        ci = analyzer.compute_confidence_intervals(
            algorithm='PPO',
            scenario='nominal_30_day',
            metric='mean_reward',
            confidence_level=0.95
        )
        
        assert 'lower' in ci
        assert 'upper' in ci
        assert 'mean' in ci
        assert ci['lower'] <= ci['mean'] <= ci['upper']

    def test_rank_algorithms(self, sample_results):
        """Test algorithm ranking functionality."""
        analyzer = StatisticalAnalyzer()
        
        for result in sample_results:
            analyzer.add_result(result)
        
        rankings = analyzer.rank_algorithms(
            scenario='nominal_30_day',
            metric='mean_reward'
        )
        
        assert isinstance(rankings, list)
        assert len(rankings) == 2  # PPO and SAC
        assert rankings[0]['algorithm'] == 'SAC'  # Should be top (90.0 > 83.5)


class TestPerformanceProfiler:
    """Test suite for performance profiling capabilities."""

    @pytest.fixture
    def mock_agent(self):
        """Mock agent for profiling."""
        agent = Mock()
        agent.predict.return_value = np.array([0.1, -0.2, 0.3, -0.4])
        return agent

    @pytest.fixture
    def mock_env(self):
        """Mock environment for profiling."""
        env = Mock()
        env.reset.return_value = (np.random.randn(20), {})
        env.step.return_value = (np.random.randn(20), 1.0, False, False, {})
        return env

    def test_profiler_initialization(self):
        """Test PerformanceProfiler initialization."""
        profiler = PerformanceProfiler()
        
        assert hasattr(profiler, 'timing_results')
        assert hasattr(profiler, 'memory_results')

    def test_profile_inference_time(self, mock_agent, mock_env):
        """Test profiling inference time."""
        profiler = PerformanceProfiler()
        
        timing_stats = profiler.profile_inference_time(
            agent=mock_agent,
            env=mock_env,
            n_steps=10
        )
        
        assert 'mean_inference_time' in timing_stats
        assert 'std_inference_time' in timing_stats
        assert 'min_inference_time' in timing_stats
        assert 'max_inference_time' in timing_stats
        assert timing_stats['n_steps'] == 10

    def test_profile_memory_usage(self, mock_agent):
        """Test profiling memory usage."""
        profiler = PerformanceProfiler()
        
        memory_stats = profiler.profile_memory_usage(
            agent=mock_agent,
            batch_sizes=[1, 16, 32]
        )
        
        assert 'peak_memory' in memory_stats
        assert 'memory_by_batch_size' in memory_stats
        assert len(memory_stats['memory_by_batch_size']) == 3

    def test_profile_throughput(self, mock_agent, mock_env):
        """Test profiling throughput."""
        profiler = PerformanceProfiler()
        
        throughput_stats = profiler.profile_throughput(
            agent=mock_agent,
            env=mock_env,
            duration_seconds=1.0
        )
        
        assert 'steps_per_second' in throughput_stats
        assert 'total_steps' in throughput_stats
        assert 'actual_duration' in throughput_stats

    def test_hardware_utilization(self):
        """Test hardware utilization monitoring."""
        profiler = PerformanceProfiler()
        
        # Mock the utilization monitoring
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 60.0
            
            utilization = profiler.monitor_hardware_utilization(duration=0.1)
            
            assert 'cpu_percent' in utilization
            assert 'memory_percent' in utilization


class TestRobustnessEvaluator:
    """Test suite for robustness evaluation capabilities."""

    @pytest.fixture
    def mock_agent(self):
        """Mock agent for robustness testing."""
        agent = Mock()
        agent.predict.return_value = np.array([0.1, -0.2, 0.3, -0.4])
        return agent

    def test_evaluator_initialization(self):
        """Test RobustnessEvaluator initialization."""
        evaluator = RobustnessEvaluator()
        
        assert hasattr(evaluator, 'robustness_results')

    def test_noise_sensitivity(self, mock_agent):
        """Test noise sensitivity evaluation."""
        evaluator = RobustnessEvaluator()
        
        # Mock environment with noise
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (np.random.randn(20), 1.0, True, False, {})
            return env
        
        sensitivity_results = evaluator.evaluate_noise_sensitivity(
            agent=mock_agent,
            env_creator=mock_env_creator,
            noise_levels=[0.0, 0.1, 0.2],
            n_episodes=2
        )
        
        assert 'noise_levels' in sensitivity_results
        assert 'performance_degradation' in sensitivity_results
        assert len(sensitivity_results['noise_levels']) == 3

    def test_parameter_sensitivity(self, mock_agent):
        """Test parameter sensitivity analysis."""
        evaluator = RobustnessEvaluator()
        
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (np.random.randn(20), 1.0, True, False, {})
            return env
        
        parameter_variations = {
            'param1': [1.0, 1.1, 1.2],
            'param2': [0.5, 0.6, 0.7]
        }
        
        sensitivity_results = evaluator.evaluate_parameter_sensitivity(
            agent=mock_agent,
            env_creator=mock_env_creator,
            parameter_variations=parameter_variations,
            n_episodes=2
        )
        
        assert 'parameter_sensitivity' in sensitivity_results
        assert 'param1' in sensitivity_results['parameter_sensitivity']
        assert 'param2' in sensitivity_results['parameter_sensitivity']

    def test_distributional_shift(self, mock_agent):
        """Test distributional shift robustness."""
        evaluator = RobustnessEvaluator()
        
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (np.random.randn(20), 1.0, True, False, {})
            return env
        
        shift_configs = [
            {'name': 'nominal', 'shift_factor': 0.0},
            {'name': 'mild_shift', 'shift_factor': 0.1},
            {'name': 'severe_shift', 'shift_factor': 0.3}
        ]
        
        shift_results = evaluator.evaluate_distributional_shift(
            agent=mock_agent,
            env_creator=mock_env_creator,
            shift_configs=shift_configs,
            n_episodes=2
        )
        
        assert 'shift_robustness' in shift_results
        assert len(shift_results['shift_robustness']) == 3


class TestTransferLearningEvaluator:
    """Test suite for transfer learning evaluation."""

    @pytest.fixture
    def mock_source_agent(self):
        """Mock source agent for transfer learning."""
        agent = Mock()
        agent.predict.return_value = np.array([0.1, -0.2, 0.3, -0.4])
        agent.save = Mock()
        agent.load = Mock()
        return agent

    def test_evaluator_initialization(self):
        """Test TransferLearningEvaluator initialization."""
        evaluator = TransferLearningEvaluator()
        
        assert hasattr(evaluator, 'transfer_results')

    def test_cross_scenario_transfer(self, mock_source_agent):
        """Test cross-scenario transfer evaluation."""
        evaluator = TransferLearningEvaluator()
        
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (np.random.randn(20), 1.0, True, False, {})
            return env
        
        source_scenarios = ['nominal_30_day']
        target_scenarios = ['eclss_failure', 'thermal_stress']
        
        with patch('lunar_habitat_rl.evaluation.BenchmarkSuite') as mock_suite_class:
            mock_suite = Mock()
            mock_suite.scenarios = [
                {'name': 'nominal_30_day', 'config': {}},
                {'name': 'eclss_failure', 'config': {}},
                {'name': 'thermal_stress', 'config': {}}
            ]
            mock_suite_class.return_value = mock_suite
            
            transfer_results = evaluator.evaluate_cross_scenario_transfer(
                source_agent=mock_source_agent,
                env_creator=mock_env_creator,
                source_scenarios=source_scenarios,
                target_scenarios=target_scenarios,
                n_evaluation_episodes=2
            )
            
            assert 'transfer_performance' in transfer_results
            assert len(transfer_results['transfer_performance']) == 2

    def test_fine_tuning_analysis(self, mock_source_agent):
        """Test fine-tuning effectiveness analysis."""
        evaluator = TransferLearningEvaluator()
        
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (np.random.randn(20), 1.0, True, False, {})
            return env
        
        # Mock training function
        def mock_fine_tune_fn(agent, env, steps):
            return {'final_performance': 85.0, 'training_time': 120.0}
        
        fine_tuning_results = evaluator.evaluate_fine_tuning_effectiveness(
            source_agent=mock_source_agent,
            target_env_creator=mock_env_creator,
            fine_tune_steps=[100, 500, 1000],
            fine_tune_fn=mock_fine_tune_fn,
            n_evaluation_episodes=2
        )
        
        assert 'fine_tuning_curves' in fine_tuning_results
        assert len(fine_tuning_results['fine_tuning_curves']) == 3


class TestBenchmarkExecution:
    """Test suite for benchmark execution functions."""

    @pytest.fixture
    def mock_agents(self):
        """Mock agents for benchmarking."""
        agents = {}
        for name in ['PPO', 'SAC', 'Random']:
            agent = Mock()
            agent.predict.return_value = np.array([0.1, -0.2, 0.3, -0.4])
            agents[name] = agent
        return agents

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (
                np.random.randn(20), 1.0, True, False,
                {'safety_score': 0.9, 'efficiency_score': 0.8}
            )
            return env
        return creator

    @patch('lunar_habitat_rl.evaluation.BenchmarkSuite')
    def test_run_benchmark_suite(self, mock_suite_class, mock_agents, mock_env_creator):
        """Test running complete benchmark suite."""
        # Mock benchmark suite
        mock_suite = Mock()
        mock_suite.scenarios = [
            {
                'name': 'test_scenario',
                'description': 'Test scenario',
                'duration': 100,
                'success_threshold': 0.8,
                'config': {}
            }
        ]
        mock_suite_class.return_value = mock_suite
        
        results = run_benchmark_suite(
            agents=mock_agents,
            env_creator=mock_env_creator,
            scenarios=['test_scenario'],
            n_runs=2,
            n_episodes=3
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert 'algorithm' in result
            assert 'scenario' in result
            assert 'run_id' in result
            assert 'metrics' in result

    def test_create_evaluation_report(self):
        """Test creating evaluation report."""
        # Sample benchmark results
        sample_results = [
            {
                'algorithm': 'PPO',
                'scenario': 'nominal_30_day',
                'run_id': 0,
                'metrics': {
                    'mean_reward': 85.0,
                    'success_rate': 0.9,
                    'mean_safety_score': 0.95
                }
            },
            {
                'algorithm': 'SAC',
                'scenario': 'nominal_30_day',
                'run_id': 0,
                'metrics': {
                    'mean_reward': 90.0,
                    'success_rate': 0.95,
                    'mean_safety_score': 0.93
                }
            }
        ]
        
        report = create_evaluation_report(
            results=sample_results,
            output_format='dict'
        )
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'statistical_analysis' in report

    def test_create_evaluation_report_json(self, tmp_path):
        """Test creating JSON evaluation report."""
        sample_results = [
            {
                'algorithm': 'PPO',
                'scenario': 'test',
                'run_id': 0,
                'metrics': {'mean_reward': 85.0}
            }
        ]
        
        output_file = tmp_path / "evaluation_report.json"
        
        create_evaluation_report(
            results=sample_results,
            output_format='json',
            output_file=str(output_file)
        )
        
        assert output_file.exists()
        
        # Verify JSON content
        with open(output_file, 'r') as f:
            report = json.load(f)
        
        assert isinstance(report, dict)
        assert 'summary' in report


class TestIntegration:
    """Integration tests for evaluation system."""

    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Mock components
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (
                np.random.randn(20), 1.0, True, False,
                {'safety_score': 0.9}
            )
            return env
        
        mock_agent = Mock()
        mock_agent.predict.return_value = np.array([0.1, -0.2, 0.3, -0.4])
        
        # Test evaluation metrics collection
        metrics = EvaluationMetrics()
        
        # Simulate episodes
        for i in range(5):
            metrics.add_episode(
                reward=80 + i*5,
                length=200 + i*10,
                success=i % 2 == 0,
                safety_score=0.9 + i*0.01
            )
        
        # Test statistical analysis
        analyzer = StatisticalAnalyzer()
        analyzer.add_result({
            'algorithm': 'TestAgent',
            'scenario': 'test_scenario',
            'run_id': 0,
            'mean_reward': metrics.mean_reward,
            'success_rate': metrics.success_rate
        })
        
        stats = analyzer.compute_aggregate_statistics()
        
        assert 'TestAgent' in stats
        assert 'test_scenario' in stats['TestAgent']

    def test_benchmark_suite_with_multiple_agents(self):
        """Test benchmark suite with multiple agents."""
        def mock_env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(20), {})
            env.step.return_value = (np.random.randn(20), 1.0, True, False, {})
            return env
        
        # Create benchmark suite
        suite = BenchmarkSuite(mock_env_creator)
        
        # Mock agents
        agents = {}
        for name in ['Agent1', 'Agent2']:
            agent = Mock()
            agent.predict.return_value = np.array([0.1, -0.2, 0.3, -0.4])
            agents[name] = agent
        
        # Test that suite can work with multiple agents
        assert len(agents) == 2
        assert len(suite.scenarios) > 0
        
        # Each agent should be able to interact with scenarios
        for agent_name, agent in agents.items():
            for scenario in suite.scenarios[:1]:  # Test with first scenario
                # This would normally run evaluation
                env = mock_env_creator()
                obs, _ = env.reset()
                action = agent.predict(obs)
                assert action is not None


if __name__ == "__main__":
    pytest.main([__file__])