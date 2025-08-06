"""
Test suite for distributed training capabilities in the lunar habitat RL suite.

This module provides comprehensive tests for distributed training infrastructure,
including DDP, parameter server architecture, and auto-scaling functionality.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import time
import threading

# Import modules to test
from lunar_habitat_rl.distributed import (
    DistributedConfig, DistributedBuffer, ParameterServer,
    DistributedWorker, DistributedDataParallel, AutoScaler,
    DistributedTrainingCoordinator, launch_distributed_training
)


class TestDistributedConfig:
    """Test suite for distributed configuration."""

    def test_default_config(self):
        """Test default distributed configuration."""
        config = DistributedConfig()
        
        assert config.world_size == 1
        assert config.rank == 0
        assert config.architecture == "ddp"
        assert config.backend == "nccl"
        assert config.master_addr == "localhost"

    def test_custom_config(self):
        """Test custom distributed configuration."""
        config = DistributedConfig(
            world_size=4,
            rank=2,
            architecture="parameter_server",
            n_workers=8,
            backend="gloo"
        )
        
        assert config.world_size == 4
        assert config.rank == 2
        assert config.architecture == "parameter_server"
        assert config.n_workers == 8
        assert config.backend == "gloo"

    def test_cloud_config(self):
        """Test cloud-specific configuration."""
        config = DistributedConfig(
            cloud_provider="aws",
            auto_scaling=True,
            min_instances=2,
            max_instances=10,
            target_utilization=0.8
        )
        
        assert config.cloud_provider == "aws"
        assert config.auto_scaling is True
        assert config.min_instances == 2
        assert config.max_instances == 10
        assert config.target_utilization == 0.8


class TestDistributedBuffer:
    """Test suite for distributed experience buffer."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        redis_client = Mock()
        redis_client.exists.return_value = False
        redis_client.set.return_value = True
        redis_client.get.return_value = b'0'
        redis_client.hset.return_value = True
        redis_client.hlen.return_value = 5
        redis_client.hget.return_value = b'mock_data'
        return redis_client

    @patch('lunar_habitat_rl.distributed.redis.Redis')
    def test_buffer_initialization(self, mock_redis_class, mock_redis):
        """Test DistributedBuffer initialization."""
        mock_redis_class.return_value = mock_redis
        
        buffer = DistributedBuffer(buffer_size=1000)
        
        assert buffer.buffer_size == 1000
        assert buffer.redis_client == mock_redis
        mock_redis.set.assert_called_once()

    @patch('lunar_habitat_rl.distributed.redis.Redis')
    def test_buffer_add_experience(self, mock_redis_class, mock_redis):
        """Test adding experience to distributed buffer."""
        mock_redis_class.return_value = mock_redis
        
        buffer = DistributedBuffer(buffer_size=1000)
        experience = {
            'obs': np.array([1, 2, 3]),
            'action': np.array([0.5]),
            'reward': 1.0,
            'done': False
        }
        
        buffer.add_experience(experience)
        
        # Should call Redis methods
        assert mock_redis.hset.called
        assert mock_redis.set.called

    @patch('lunar_habitat_rl.distributed.redis.Redis')
    @patch('lunar_habitat_rl.distributed.pickle.loads')
    def test_buffer_sample_batch(self, mock_pickle_loads, mock_redis_class, mock_redis):
        """Test sampling batch from distributed buffer."""
        mock_redis_class.return_value = mock_redis
        mock_pickle_loads.return_value = {'obs': np.array([1, 2, 3])}
        
        buffer = DistributedBuffer(buffer_size=1000)
        
        batch = buffer.sample_batch(batch_size=2)
        
        assert isinstance(batch, list)
        assert mock_redis.hget.called

    @patch('lunar_habitat_rl.distributed.redis.Redis')
    def test_buffer_get_size(self, mock_redis_class, mock_redis):
        """Test getting buffer size."""
        mock_redis_class.return_value = mock_redis
        mock_redis.hlen.return_value = 42
        
        buffer = DistributedBuffer(buffer_size=1000)
        size = buffer.get_buffer_size()
        
        assert size == 42
        mock_redis.hlen.assert_called_once()


class TestParameterServer:
    """Test suite for parameter server architecture."""

    @pytest.fixture
    def simple_model(self):
        """Simple neural network for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    @pytest.fixture
    def distributed_config(self):
        """Test distributed configuration."""
        return DistributedConfig(
            architecture="parameter_server",
            n_workers=2
        )

    @patch('lunar_habitat_rl.distributed.zmq.Context')
    def test_parameter_server_initialization(self, mock_zmq_context, simple_model, distributed_config):
        """Test ParameterServer initialization."""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context
        
        ps = ParameterServer(distributed_config, simple_model)
        
        assert ps.config == distributed_config
        assert ps.model == simple_model
        assert ps.global_step == 0
        mock_socket.bind.assert_called_once()

    @patch('lunar_habitat_rl.distributed.zmq.Context')
    def test_parameter_server_get_parameters(self, mock_zmq_context, simple_model, distributed_config):
        """Test parameter server get parameters request."""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context
        
        ps = ParameterServer(distributed_config, simple_model)
        
        # Test get_parameters request
        message = {"type": "get_parameters", "worker_id": "worker_0"}
        response = ps._handle_request(message)
        
        assert "parameters" in response
        assert "global_step" in response
        assert response["global_step"] == 0

    @patch('lunar_habitat_rl.distributed.zmq.Context')
    def test_parameter_server_update_parameters(self, mock_zmq_context, simple_model, distributed_config):
        """Test parameter server update parameters request."""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context
        
        ps = ParameterServer(distributed_config, simple_model)
        
        # Create fake gradients
        gradients = {}
        for name, param in simple_model.named_parameters():
            gradients[name] = torch.randn_like(param)
        
        message = {
            "type": "update_parameters",
            "worker_id": "worker_0",
            "gradients": gradients,
            "learning_rate": 3e-4
        }
        
        response = ps._handle_request(message)
        
        assert response["success"] is True
        assert ps.global_step == 1

    @patch('lunar_habitat_rl.distributed.zmq.Context')
    def test_parameter_server_get_stats(self, mock_zmq_context, simple_model, distributed_config):
        """Test parameter server statistics request."""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context
        
        ps = ParameterServer(distributed_config, simple_model)
        
        message = {"type": "get_stats"}
        response = ps._handle_request(message)
        
        assert "global_step" in response
        assert "worker_stats" in response
        assert "num_workers" in response


class TestDistributedWorker:
    """Test suite for distributed worker."""

    @pytest.fixture
    def simple_model(self):
        """Simple neural network for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(10), {})
            env.step.return_value = (
                np.random.randn(10), 1.0, False, False, {}
            )
            return env
        return creator

    @pytest.fixture
    def distributed_config(self):
        """Test distributed configuration."""
        return DistributedConfig(
            architecture="parameter_server",
            sync_frequency=10
        )

    @patch('lunar_habitat_rl.distributed.zmq.Context')
    @patch('lunar_habitat_rl.distributed.DistributedBuffer')
    def test_worker_initialization(self, mock_buffer, mock_zmq_context, 
                                 simple_model, mock_env_creator, distributed_config):
        """Test DistributedWorker initialization."""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context
        
        worker = DistributedWorker(
            worker_id="worker_0",
            config=distributed_config,
            model=simple_model,
            env_creator=mock_env_creator
        )
        
        assert worker.worker_id == "worker_0"
        assert worker.config == distributed_config
        assert worker.local_step == 0
        mock_socket.connect.assert_called_once()

    @patch('lunar_habitat_rl.distributed.zmq.Context')
    @patch('lunar_habitat_rl.distributed.DistributedBuffer')
    def test_worker_sync_parameters(self, mock_buffer, mock_zmq_context,
                                   simple_model, mock_env_creator, distributed_config):
        """Test worker parameter synchronization."""
        mock_context = Mock()
        mock_socket = Mock()
        mock_context.socket.return_value = mock_socket
        mock_zmq_context.return_value = mock_context
        
        # Mock parameter server response
        mock_params = {}
        for name, param in simple_model.named_parameters():
            mock_params[name] = torch.randn_like(param)
        
        mock_socket.recv_pyobj.return_value = {
            "parameters": mock_params,
            "global_step": 5
        }
        
        worker = DistributedWorker(
            worker_id="worker_0",
            config=distributed_config,
            model=simple_model,
            env_creator=mock_env_creator
        )
        
        worker._sync_parameters()
        
        mock_socket.send_pyobj.assert_called()
        mock_socket.recv_pyobj.assert_called()


class TestAutoScaler:
    """Test suite for auto-scaling functionality."""

    def test_autoscaler_initialization(self):
        """Test AutoScaler initialization."""
        config = DistributedConfig(
            auto_scaling=True,
            min_instances=2,
            max_instances=8,
            target_utilization=0.7
        )
        
        scaler = AutoScaler(config)
        
        assert scaler.current_instances == config.min_instances
        assert scaler.target_utilization == 0.7
        assert len(scaler.utilization_history) == 0

    def test_autoscaler_update_metrics(self):
        """Test updating scaling metrics."""
        config = DistributedConfig(auto_scaling=True)
        scaler = AutoScaler(config)
        
        scaler.update_metrics(utilization=0.8, performance=100.0)
        
        assert len(scaler.utilization_history) == 1
        assert len(scaler.performance_history) == 1
        assert scaler.utilization_history[0] == 0.8

    def test_autoscaler_scale_up_decision(self):
        """Test scale-up decision logic."""
        config = DistributedConfig(
            auto_scaling=True,
            max_instances=10,
            target_utilization=0.7
        )
        scaler = AutoScaler(config)
        
        # Add high utilization metrics
        for _ in range(6):
            scaler.update_metrics(utilization=0.9, performance=100.0)
        
        should_scale = scaler.should_scale_up()
        assert should_scale is True

    def test_autoscaler_scale_down_decision(self):
        """Test scale-down decision logic."""
        config = DistributedConfig(
            auto_scaling=True,
            min_instances=1,
            target_utilization=0.7
        )
        scaler = AutoScaler(config)
        scaler.current_instances = 5  # Start with more instances
        
        # Add low utilization metrics
        for _ in range(6):
            scaler.update_metrics(utilization=0.3, performance=100.0)
        
        should_scale = scaler.should_scale_down()
        assert should_scale is True

    def test_autoscaler_no_scale_at_limits(self):
        """Test that scaling respects min/max limits."""
        config = DistributedConfig(
            auto_scaling=True,
            min_instances=1,
            max_instances=2
        )
        scaler = AutoScaler(config)
        
        # At max instances - shouldn't scale up
        scaler.current_instances = config.max_instances
        for _ in range(6):
            scaler.update_metrics(utilization=0.9, performance=100.0)
        
        assert scaler.should_scale_up() is False
        
        # At min instances - shouldn't scale down  
        scaler.current_instances = config.min_instances
        for _ in range(6):
            scaler.update_metrics(utilization=0.2, performance=100.0)
        
        assert scaler.should_scale_down() is False


class TestDistributedDataParallel:
    """Test suite for Distributed Data Parallel training."""

    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="DDP tests require CUDA")
    def test_ddp_initialization_gpu(self):
        """Test DDP initialization with GPU."""
        config = DistributedConfig(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl"
        )
        
        # Mock distributed initialization
        with patch('torch.distributed.init_process_group'), \
             patch('torch.cuda.set_device'):
            
            ddp = DistributedDataParallel(config)
            
            assert ddp.rank == 0
            assert ddp.world_size == 1
            assert ddp.device.type == 'cuda'

    def test_ddp_initialization_cpu(self):
        """Test DDP initialization with CPU."""
        config = DistributedConfig(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo"
        )
        
        with patch('torch.distributed.init_process_group'), \
             patch('torch.cuda.is_available', return_value=False):
            
            ddp = DistributedDataParallel(config)
            
            assert ddp.device.type == 'cpu'

    def test_ddp_setup_model(self):
        """Test DDP model setup."""
        config = DistributedConfig(world_size=1, rank=0, local_rank=0)
        
        with patch('torch.distributed.init_process_group'), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
            
            ddp = DistributedDataParallel(config)
            model = nn.Linear(10, 4)
            
            ddp_model = ddp.setup_model(model)
            
            mock_ddp.assert_called_once()

    def test_ddp_all_reduce_metrics(self):
        """Test all-reduce metrics functionality."""
        config = DistributedConfig(world_size=1, rank=0)
        
        with patch('torch.distributed.init_process_group'), \
             patch('torch.distributed.all_reduce') as mock_all_reduce, \
             patch('torch.cuda.is_available', return_value=False):
            
            ddp = DistributedDataParallel(config)
            
            metrics = {"loss": 1.5, "accuracy": 0.8}
            reduced_metrics = ddp.all_reduce_metrics(metrics)
            
            assert "loss" in reduced_metrics
            assert "accuracy" in reduced_metrics
            assert mock_all_reduce.call_count == 2


class TestDistributedTrainingCoordinator:
    """Test suite for distributed training coordinator."""

    @pytest.fixture
    def simple_model(self):
        """Simple neural network for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(), 
            nn.Linear(32, 4)
        )

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(10), {})
            env.step.return_value = (
                np.random.randn(10), 1.0, False, False, {}
            )
            return env
        return creator

    def test_coordinator_ddp_initialization(self, simple_model, mock_env_creator):
        """Test coordinator initialization with DDP architecture."""
        config = DistributedConfig(
            architecture="ddp",
            world_size=1,
            rank=0
        )
        
        with patch('lunar_habitat_rl.distributed.DistributedDataParallel') as mock_ddp_class:
            mock_ddp = Mock()
            mock_ddp_class.return_value = mock_ddp
            
            coordinator = DistributedTrainingCoordinator(
                config, simple_model, mock_env_creator
            )
            
            assert coordinator.config == config
            assert coordinator.model == simple_model
            mock_ddp_class.assert_called_once_with(config)

    def test_coordinator_parameter_server_initialization(self, simple_model, mock_env_creator):
        """Test coordinator initialization with parameter server architecture."""
        config = DistributedConfig(
            architecture="parameter_server",
            n_workers=2
        )
        
        with patch('lunar_habitat_rl.distributed.ParameterServer') as mock_ps_class:
            mock_ps = Mock()
            mock_ps_class.return_value = mock_ps
            
            coordinator = DistributedTrainingCoordinator(
                config, simple_model, mock_env_creator
            )
            
            assert coordinator.config == config
            mock_ps_class.assert_called_once_with(config, simple_model)

    def test_coordinator_unknown_architecture(self, simple_model, mock_env_creator):
        """Test coordinator with unknown architecture."""
        config = DistributedConfig(architecture="unknown")
        
        with pytest.raises(Exception):  # Should raise DistributedTrainingError
            DistributedTrainingCoordinator(config, simple_model, mock_env_creator)

    def test_coordinator_with_autoscaling(self, simple_model, mock_env_creator):
        """Test coordinator with auto-scaling enabled."""
        config = DistributedConfig(
            architecture="ddp",
            auto_scaling=True,
            min_instances=1,
            max_instances=5
        )
        
        with patch('lunar_habitat_rl.distributed.DistributedDataParallel'), \
             patch('lunar_habitat_rl.distributed.AutoScaler') as mock_scaler_class:
            
            mock_scaler = Mock()
            mock_scaler_class.return_value = mock_scaler
            
            coordinator = DistributedTrainingCoordinator(
                config, simple_model, mock_env_creator
            )
            
            assert coordinator.auto_scaler == mock_scaler
            mock_scaler_class.assert_called_once_with(config)


class TestLaunchDistributedTraining:
    """Test suite for distributed training launch function."""

    @pytest.fixture
    def simple_model_creator(self):
        """Simple model creator for testing."""
        def creator():
            return nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 4)
            )
        return creator

    @pytest.fixture
    def mock_env_creator(self):
        """Mock environment creator."""
        def creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(10), {})
            env.step.return_value = (
                np.random.randn(10), 1.0, False, False, {}
            )
            return env
        return creator

    def test_launch_single_process(self, simple_model_creator, mock_env_creator):
        """Test launching single process training."""
        config = DistributedConfig(
            architecture="ddp",
            world_size=1,
            rank=0
        )
        
        with patch('lunar_habitat_rl.distributed.DistributedTrainingCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.train.return_value = {"status": "completed"}
            mock_coordinator_class.return_value = mock_coordinator
            
            results = launch_distributed_training(
                config, simple_model_creator, mock_env_creator, total_steps=100
            )
            
            assert "status" in results
            mock_coordinator.train.assert_called_once_with(100)
            mock_coordinator.shutdown.assert_called_once()

    @patch('torch.multiprocessing.spawn')
    def test_launch_multiprocess_ddp(self, mock_spawn, simple_model_creator, mock_env_creator):
        """Test launching multi-process DDP training."""
        config = DistributedConfig(
            architecture="ddp",
            world_size=4,
            rank=0
        )
        
        results = launch_distributed_training(
            config, simple_model_creator, mock_env_creator, total_steps=1000
        )
        
        mock_spawn.assert_called_once()
        assert "world_size" in results
        assert results["world_size"] == 4

    def test_launch_with_validation_error(self, simple_model_creator, mock_env_creator):
        """Test launch with configuration validation error."""
        # Invalid configuration
        config = DistributedConfig(world_size=-1)  # Invalid world size
        
        with patch('lunar_habitat_rl.distributed.validate_distributed_config') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid configuration")
            
            with pytest.raises(ValueError):
                launch_distributed_training(
                    config, simple_model_creator, mock_env_creator
                )


class TestIntegration:
    """Integration tests for distributed training components."""

    def test_coordinator_training_flow(self):
        """Test basic training flow through coordinator."""
        config = DistributedConfig(
            architecture="ddp",
            world_size=1,
            rank=0
        )
        
        model = nn.Sequential(nn.Linear(10, 4))
        
        def env_creator():
            env = Mock()
            env.reset.return_value = (np.random.randn(10), {})
            env.step.return_value = (
                np.random.randn(10), 1.0, True, False, {}  # Done after 1 step
            )
            return env
        
        with patch('lunar_habitat_rl.distributed.DistributedDataParallel') as mock_ddp_class:
            mock_ddp = Mock()
            mock_ddp.device = torch.device('cpu')
            mock_ddp.setup_model.return_value = model
            mock_ddp.all_reduce_metrics.return_value = {"mean_episode_reward": 1.0}
            mock_ddp_class.return_value = mock_ddp
            
            coordinator = DistributedTrainingCoordinator(config, model, env_creator)
            
            # Run short training
            results = coordinator._train_ddp(total_steps=5, checkpoint_freq=10)
            
            assert isinstance(results, dict)
            assert "mean_episode_reward" in results

    def test_component_interaction(self):
        """Test interaction between distributed components."""
        # Test that components can be created together
        config = DistributedConfig(
            architecture="parameter_server",
            auto_scaling=True,
            n_workers=2
        )
        
        model = nn.Linear(10, 4)
        
        with patch('lunar_habitat_rl.distributed.ParameterServer'), \
             patch('lunar_habitat_rl.distributed.AutoScaler'):
            
            # Should create both parameter server and auto scaler
            def env_creator():
                return Mock()
            
            coordinator = DistributedTrainingCoordinator(config, model, env_creator)
            
            assert hasattr(coordinator, 'parameter_server')
            assert hasattr(coordinator, 'auto_scaler')


if __name__ == "__main__":
    pytest.main([__file__])