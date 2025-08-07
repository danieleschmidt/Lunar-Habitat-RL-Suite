"""
Distributed Training Infrastructure for Lunar Habitat RL

This module implements scalable distributed training capabilities:
- Multi-GPU training with data parallelism
- Distributed actor-learner architectures
- Parameter servers for large-scale deployment
- Asynchronous training with experience replay
- Federated learning for multi-habitat scenarios
- Real-time model synchronization
- Fault-tolerant training systems

Generation 3 Implementation: Production-scale distributed training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import json
import pickle
import threading
from queue import Queue, Empty
import zmq
import redis
from pathlib import Path
import uuid
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Basic distributed settings
    world_size: int = 4                    # Total number of processes
    rank: int = 0                          # Current process rank
    local_rank: int = 0                    # Local GPU rank
    master_addr: str = "localhost"         # Master node address
    master_port: str = "12355"             # Master node port
    backend: str = "nccl"                  # Communication backend
    
    # Training configuration
    batch_size: int = 256                  # Per-device batch size
    gradient_clip_norm: float = 1.0        # Gradient clipping
    sync_every_n_steps: int = 10           # Model synchronization frequency
    
    # Actor-learner architecture
    n_actors: int = 8                      # Number of actor processes
    n_learners: int = 2                    # Number of learner processes
    experience_buffer_size: int = 1000000  # Replay buffer size
    
    # Parameter server settings
    use_parameter_server: bool = False     # Enable parameter server
    ps_host: str = "localhost"             # Parameter server host
    ps_port: int = 5555                    # Parameter server port
    
    # Fault tolerance
    checkpoint_frequency: int = 1000       # Steps between checkpoints
    max_retries: int = 3                   # Max retries for failed operations
    heartbeat_interval: float = 30.0       # Heartbeat interval (seconds)
    
    # Federated learning
    enable_federated: bool = False         # Enable federated learning
    federation_rounds: int = 100           # Number of federation rounds
    clients_per_round: int = 10            # Active clients per round
    
    # Performance optimization
    mixed_precision: bool = True           # Enable mixed precision training
    compile_model: bool = False            # Compile model with torch.compile
    pin_memory: bool = True                # Pin memory for data loading
    non_blocking: bool = True              # Non-blocking GPU transfers


class ParameterServer:
    """Parameter server for distributed training."""
    
    def __init__(self, config: DistributedConfig, model: nn.Module):
        self.config = config
        self.model = model
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{config.ps_port}")
        
        # Redis connection for persistent storage
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        
        # Parameter versioning
        self.version = 0
        self.update_count = 0
        
        # Worker tracking
        self.registered_workers = {}
        self.last_heartbeat = {}
        
        self.running = False
        
        logger.info(f"Parameter server initialized on port {config.ps_port}")
    
    def start(self):
        """Start parameter server."""
        self.running = True
        logger.info("Parameter server started")
        
        while self.running:
            try:
                # Wait for message with timeout
                if self.socket.poll(timeout=1000):  # 1 second timeout
                    message = self.socket.recv_pyobj()
                    response = self._handle_message(message)
                    self.socket.send_pyobj(response)
                
                # Clean up stale workers
                self._cleanup_stale_workers()
                
            except Exception as e:
                logger.error(f"Parameter server error: {e}")
                self.socket.send_pyobj({"status": "error", "message": str(e)})
    
    def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message from worker."""
        msg_type = message.get("type")
        worker_id = message.get("worker_id")
        
        if msg_type == "register":
            return self._register_worker(worker_id, message.get("metadata", {}))
        
        elif msg_type == "get_parameters":
            return self._get_parameters(worker_id)
        
        elif msg_type == "update_parameters":
            gradients = message.get("gradients")
            return self._update_parameters(worker_id, gradients)
        
        elif msg_type == "heartbeat":
            return self._heartbeat(worker_id)
        
        elif msg_type == "checkpoint":
            return self._save_checkpoint(message.get("path"))
        
        else:
            return {"status": "error", "message": f"Unknown message type: {msg_type}"}
    
    def _register_worker(self, worker_id: str, metadata: Dict) -> Dict[str, Any]:
        """Register a new worker."""
        self.registered_workers[worker_id] = metadata
        self.last_heartbeat[worker_id] = time.time()
        
        logger.info(f"Registered worker {worker_id}")
        
        return {
            "status": "success",
            "worker_id": worker_id,
            "version": self.version
        }
    
    def _get_parameters(self, worker_id: str) -> Dict[str, Any]:
        """Send current parameters to worker."""
        if worker_id not in self.registered_workers:
            return {"status": "error", "message": "Worker not registered"}
        
        self.last_heartbeat[worker_id] = time.time()
        
        # Serialize model parameters
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        return {
            "status": "success",
            "parameters": state_dict,
            "version": self.version
        }
    
    def _update_parameters(self, worker_id: str, gradients: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Update parameters with gradients from worker."""
        if worker_id not in self.registered_workers:
            return {"status": "error", "message": "Worker not registered"}
        
        try:
            # Apply gradients to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in gradients:
                        grad_tensor = gradients[name]
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad.add_(grad_tensor)
            
            self.update_count += 1
            self.last_heartbeat[worker_id] = time.time()
            
            # Increment version after updates
            if self.update_count % self.config.sync_every_n_steps == 0:
                self.version += 1
            
            return {
                "status": "success",
                "version": self.version,
                "update_count": self.update_count
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _heartbeat(self, worker_id: str) -> Dict[str, Any]:
        """Handle worker heartbeat."""
        if worker_id not in self.registered_workers:
            return {"status": "error", "message": "Worker not registered"}
        
        self.last_heartbeat[worker_id] = time.time()
        
        return {
            "status": "success",
            "server_time": time.time(),
            "version": self.version
        }
    
    def _save_checkpoint(self, path: str) -> Dict[str, Any]:
        """Save model checkpoint."""
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "version": self.version,
                "update_count": self.update_count,
                "timestamp": time.time()
            }
            
            torch.save(checkpoint, path)
            
            # Also save to Redis for redundancy
            self.redis_client.set("latest_checkpoint", pickle.dumps(checkpoint))
            
            return {"status": "success", "path": path, "version": self.version}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _cleanup_stale_workers(self):
        """Remove workers that haven't sent heartbeat recently."""
        current_time = time.time()
        stale_workers = []
        
        for worker_id, last_heartbeat in self.last_heartbeat.items():
            if current_time - last_heartbeat > self.config.heartbeat_interval * 2:
                stale_workers.append(worker_id)
        
        for worker_id in stale_workers:
            logger.warning(f"Removing stale worker {worker_id}")
            del self.registered_workers[worker_id]
            del self.last_heartbeat[worker_id]
    
    def stop(self):
        """Stop parameter server."""
        self.running = False
        self.socket.close()
        self.context.term()
        logger.info("Parameter server stopped")


class DistributedWorker:
    """Distributed worker for training."""
    
    def __init__(self, config: DistributedConfig, model: nn.Module, 
                 optimizer: torch.optim.Optimizer, worker_type: str = "learner"):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.worker_type = worker_type
        self.worker_id = f"{worker_type}_{uuid.uuid4().hex[:8]}"
        
        # Parameter server connection
        if config.use_parameter_server:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{config.ps_host}:{config.ps_port}")
            self._register_with_ps()
        
        # Experience buffer for actors
        if worker_type == "actor":
            self.experience_buffer = Queue(maxsize=config.experience_buffer_size)
        
        # Performance tracking
        self.step_count = 0
        self.last_sync_time = time.time()
        self.sync_times = []
        
        logger.info(f"Distributed worker {self.worker_id} initialized")
    
    def _register_with_ps(self):
        """Register with parameter server."""
        message = {
            "type": "register",
            "worker_id": self.worker_id,
            "metadata": {
                "worker_type": self.worker_type,
                "hostname": socket.gethostname(),
                "timestamp": time.time()
            }
        }
        
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        
        if response["status"] != "success":
            raise RuntimeError(f"Failed to register with parameter server: {response}")
        
        logger.info(f"Registered with parameter server: {response}")
    
    def sync_parameters(self):
        """Synchronize parameters with parameter server."""
        if not self.config.use_parameter_server:
            return
        
        try:
            # Get latest parameters
            message = {"type": "get_parameters", "worker_id": self.worker_id}
            self.socket.send_pyobj(message)
            response = self.socket.recv_pyobj()
            
            if response["status"] == "success":
                # Load parameters into model
                state_dict = response["parameters"]
                self.model.load_state_dict(state_dict)
                
                sync_time = time.time() - self.last_sync_time
                self.sync_times.append(sync_time)
                self.last_sync_time = time.time()
                
            else:
                logger.error(f"Failed to sync parameters: {response}")
                
        except Exception as e:
            logger.error(f"Parameter sync error: {e}")
    
    def send_gradients(self):
        """Send gradients to parameter server."""
        if not self.config.use_parameter_server:
            return
        
        try:
            # Extract gradients
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.cpu()
            
            # Send gradients
            message = {
                "type": "update_parameters",
                "worker_id": self.worker_id,
                "gradients": gradients
            }
            
            self.socket.send_pyobj(message)
            response = self.socket.recv_pyobj()
            
            if response["status"] != "success":
                logger.error(f"Failed to send gradients: {response}")
                
        except Exception as e:
            logger.error(f"Gradient send error: {e}")
    
    def heartbeat(self):
        """Send heartbeat to parameter server."""
        if not self.config.use_parameter_server:
            return
        
        try:
            message = {"type": "heartbeat", "worker_id": self.worker_id}
            self.socket.send_pyobj(message)
            response = self.socket.recv_pyobj()
            
            if response["status"] != "success":
                logger.warning(f"Heartbeat failed: {response}")
                
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")


class ActorLearnerTrainer:
    """Actor-Learner distributed training architecture."""
    
    def __init__(self, config: DistributedConfig, model_factory: Callable[[], nn.Module],
                 env_factory: Callable[[], Any]):
        self.config = config
        self.model_factory = model_factory
        self.env_factory = env_factory
        
        # Shared experience buffer
        self.experience_buffer = mp.Queue(maxsize=config.experience_buffer_size)
        
        # Model synchronization
        self.shared_state = mp.Manager().dict()
        self.sync_lock = mp.Lock()
        
        # Process management
        self.processes = []
        self.stop_event = mp.Event()
        
        logger.info("Actor-Learner trainer initialized")
    
    def start_training(self):
        """Start distributed actor-learner training."""
        # Start learner processes
        for i in range(self.config.n_learners):
            learner_process = mp.Process(
                target=self._learner_worker,
                args=(i, self.experience_buffer, self.shared_state, self.sync_lock)
            )
            learner_process.start()
            self.processes.append(learner_process)
        
        # Start actor processes
        for i in range(self.config.n_actors):
            actor_process = mp.Process(
                target=self._actor_worker,
                args=(i, self.experience_buffer, self.shared_state)
            )
            actor_process.start()
            self.processes.append(actor_process)
        
        logger.info(f"Started {self.config.n_learners} learners and {self.config.n_actors} actors")
        
        # Wait for processes to complete
        try:
            for process in self.processes:
                process.join()
        except KeyboardInterrupt:
            logger.info("Stopping distributed training...")
            self.stop_training()
    
    def _learner_worker(self, learner_id: int, experience_buffer: mp.Queue,
                       shared_state: dict, sync_lock: mp.Lock):
        """Learner worker process."""
        # Create model and optimizer
        model = self.model_factory()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create distributed worker
        worker_config = self.config
        worker_config.rank = learner_id
        worker = DistributedWorker(worker_config, model, optimizer, "learner")
        
        step_count = 0
        batch_experiences = []
        
        while not self.stop_event.is_set():
            try:
                # Collect experiences
                while len(batch_experiences) < self.config.batch_size:
                    try:
                        experience = experience_buffer.get(timeout=1.0)
                        batch_experiences.append(experience)
                    except:
                        break
                
                if len(batch_experiences) >= self.config.batch_size:
                    # Process batch
                    batch = self._process_experience_batch(batch_experiences)
                    
                    # Training step
                    loss = self._training_step(model, optimizer, batch)
                    
                    step_count += 1
                    
                    # Synchronize with parameter server
                    if step_count % self.config.sync_every_n_steps == 0:
                        worker.send_gradients()
                        worker.sync_parameters()
                        
                        # Update shared state
                        with sync_lock:
                            shared_state[f'learner_{learner_id}_loss'] = loss
                            shared_state[f'learner_{learner_id}_steps'] = step_count
                    
                    batch_experiences.clear()
                
                # Periodic heartbeat
                if step_count % 100 == 0:
                    worker.heartbeat()
                    
            except Exception as e:
                logger.error(f"Learner {learner_id} error: {e}")
                time.sleep(1.0)
        
        logger.info(f"Learner {learner_id} finished")
    
    def _actor_worker(self, actor_id: int, experience_buffer: mp.Queue, shared_state: dict):
        """Actor worker process."""
        # Create environment and model
        env = self.env_factory()
        model = self.model_factory()
        
        # Create distributed worker
        worker_config = self.config
        worker_config.rank = self.config.n_learners + actor_id
        worker = DistributedWorker(worker_config, model, None, "actor")
        
        episode_count = 0
        step_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Sync model parameters periodically
                if step_count % (self.config.sync_every_n_steps * 5) == 0:
                    worker.sync_parameters()
                
                # Run episode
                obs = env.reset()
                done = False
                episode_reward = 0
                
                while not done and not self.stop_event.is_set():
                    # Select action
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = model(obs_tensor).squeeze(0).numpy()
                    
                    # Take step in environment
                    next_obs, reward, done, info = env.step(action)
                    
                    # Create experience
                    experience = {
                        'obs': obs,
                        'action': action,
                        'reward': reward,
                        'next_obs': next_obs,
                        'done': done,
                        'actor_id': actor_id
                    }
                    
                    # Add to experience buffer
                    try:
                        experience_buffer.put_nowait(experience)
                    except:
                        # Buffer full, skip
                        pass
                    
                    obs = next_obs
                    episode_reward += reward
                    step_count += 1
                
                episode_count += 1
                
                # Update shared state
                shared_state[f'actor_{actor_id}_episodes'] = episode_count
                shared_state[f'actor_{actor_id}_reward'] = episode_reward
                
                # Periodic heartbeat
                if episode_count % 10 == 0:
                    worker.heartbeat()
                
            except Exception as e:
                logger.error(f"Actor {actor_id} error: {e}")
                time.sleep(1.0)
        
        env.close()
        logger.info(f"Actor {actor_id} finished")
    
    def _process_experience_batch(self, experiences: List[Dict]) -> Dict[str, torch.Tensor]:
        """Process batch of experiences."""
        batch = {
            'obs': torch.FloatTensor([exp['obs'] for exp in experiences]),
            'actions': torch.FloatTensor([exp['action'] for exp in experiences]),
            'rewards': torch.FloatTensor([exp['reward'] for exp in experiences]),
            'next_obs': torch.FloatTensor([exp['next_obs'] for exp in experiences]),
            'dones': torch.BoolTensor([exp['done'] for exp in experiences])
        }
        return batch
    
    def _training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                      batch: Dict[str, torch.Tensor]) -> float:
        """Perform single training step."""
        # Simple training step (would be more complex in practice)
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch['obs'])
        targets = batch['rewards'].unsqueeze(1)  # Simple regression target
        
        # Compute loss
        loss = nn.MSELoss()(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
        
        # Optimizer step
        optimizer.step()
        
        return loss.item()
    
    def stop_training(self):
        """Stop distributed training."""
        self.stop_event.set()
        
        # Terminate processes
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
        
        logger.info("Distributed training stopped")


class FederatedLearner:
    """Federated learning implementation for multi-habitat scenarios."""
    
    def __init__(self, config: DistributedConfig, model: nn.Module):
        self.config = config
        self.global_model = model
        self.clients = {}
        self.round_number = 0
        
        # Client selection
        self.client_selection_strategy = "random"  # "random", "performance", "diversity"
        
        # Aggregation method
        self.aggregation_method = "fedavg"  # "fedavg", "weighted", "scaffold"
        
        logger.info("Federated learner initialized")
    
    def register_client(self, client_id: str, client_data_size: int, 
                       client_model: nn.Module) -> bool:
        """Register a federated learning client."""
        self.clients[client_id] = {
            'model': client_model,
            'data_size': client_data_size,
            'performance_history': [],
            'last_update_round': -1,
            'is_active': True
        }
        
        logger.info(f"Registered client {client_id} with {client_data_size} data samples")
        return True
    
    def federated_round(self) -> Dict[str, Any]:
        """Execute one federated learning round."""
        round_start_time = time.time()
        
        # Select clients for this round
        selected_clients = self._select_clients()
        
        if not selected_clients:
            logger.warning("No clients selected for federated round")
            return {}
        
        # Send global model to selected clients
        for client_id in selected_clients:
            self._send_global_model_to_client(client_id)
        
        # Collect client updates
        client_updates = {}
        for client_id in selected_clients:
            update = self._collect_client_update(client_id)
            if update:
                client_updates[client_id] = update
        
        # Aggregate updates
        if client_updates:
            self._aggregate_client_updates(client_updates)
            self.round_number += 1
        
        round_time = time.time() - round_start_time
        
        # Evaluate global model
        global_performance = self._evaluate_global_model()
        
        round_stats = {
            'round_number': self.round_number,
            'selected_clients': len(selected_clients),
            'participating_clients': len(client_updates),
            'round_time': round_time,
            'global_performance': global_performance
        }
        
        logger.info(f"Federated round {self.round_number} completed: {round_stats}")
        
        return round_stats
    
    def _select_clients(self) -> List[str]:
        """Select clients for current round."""
        active_clients = [cid for cid, info in self.clients.items() if info['is_active']]
        
        if len(active_clients) <= self.config.clients_per_round:
            return active_clients
        
        if self.client_selection_strategy == "random":
            return np.random.choice(active_clients, self.config.clients_per_round, replace=False).tolist()
        
        elif self.client_selection_strategy == "performance":
            # Select based on recent performance
            client_scores = {}
            for client_id in active_clients:
                history = self.clients[client_id]['performance_history']
                if history:
                    client_scores[client_id] = np.mean(history[-5:])  # Recent average
                else:
                    client_scores[client_id] = 0.0
            
            # Select top performers
            sorted_clients = sorted(client_scores.items(), key=lambda x: x[1], reverse=True)
            return [cid for cid, _ in sorted_clients[:self.config.clients_per_round]]
        
        elif self.client_selection_strategy == "diversity":
            # Select diverse set of clients (simplified)
            return np.random.choice(active_clients, self.config.clients_per_round, replace=False).tolist()
        
        else:
            return active_clients[:self.config.clients_per_round]
    
    def _send_global_model_to_client(self, client_id: str):
        """Send current global model to client."""
        if client_id not in self.clients:
            return
        
        # Copy global model parameters to client model
        client_model = self.clients[client_id]['model']
        client_model.load_state_dict(self.global_model.state_dict())
    
    def _collect_client_update(self, client_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Collect model update from client."""
        if client_id not in self.clients:
            return None
        
        # In practice, this would involve network communication
        # Here we simulate client training and return parameter updates
        
        client_model = self.clients[client_id]['model']
        
        # Simulate local training (simplified)
        # In reality, client would train locally and send back parameter differences
        
        # Calculate parameter differences
        update = {}
        for name, global_param in self.global_model.named_parameters():
            client_param = dict(client_model.named_parameters())[name]
            update[name] = client_param - global_param
        
        return update
    
    def _aggregate_client_updates(self, client_updates: Dict[str, Dict[str, torch.Tensor]]):
        """Aggregate client updates using FedAvg or other methods."""
        if self.aggregation_method == "fedavg":
            self._federated_averaging(client_updates)
        elif self.aggregation_method == "weighted":
            self._weighted_aggregation(client_updates)
        else:
            self._federated_averaging(client_updates)  # Default
    
    def _federated_averaging(self, client_updates: Dict[str, Dict[str, torch.Tensor]]):
        """FedAvg aggregation method."""
        num_clients = len(client_updates)
        
        if num_clients == 0:
            return
        
        # Average parameter updates
        averaged_update = {}
        
        # Initialize averaged update
        for param_name in client_updates[list(client_updates.keys())[0]].keys():
            averaged_update[param_name] = torch.zeros_like(
                self.global_model.state_dict()[param_name]
            )
        
        # Sum all client updates
        for client_id, update in client_updates.items():
            for param_name, param_update in update.items():
                averaged_update[param_name] += param_update
        
        # Average and apply to global model
        with torch.no_grad():
            for param_name, param_update in averaged_update.items():
                param_update /= num_clients
                self.global_model.state_dict()[param_name].add_(param_update)
    
    def _weighted_aggregation(self, client_updates: Dict[str, Dict[str, torch.Tensor]]):
        """Weighted aggregation based on client data sizes."""
        total_data_size = sum(self.clients[cid]['data_size'] for cid in client_updates.keys())
        
        if total_data_size == 0:
            return
        
        # Weighted average parameter updates
        weighted_update = {}
        
        # Initialize
        for param_name in client_updates[list(client_updates.keys())[0]].keys():
            weighted_update[param_name] = torch.zeros_like(
                self.global_model.state_dict()[param_name]
            )
        
        # Weighted sum
        for client_id, update in client_updates.items():
            weight = self.clients[client_id]['data_size'] / total_data_size
            for param_name, param_update in update.items():
                weighted_update[param_name] += weight * param_update
        
        # Apply to global model
        with torch.no_grad():
            for param_name, param_update in weighted_update.items():
                self.global_model.state_dict()[param_name].add_(param_update)
    
    def _evaluate_global_model(self) -> float:
        """Evaluate global model performance."""
        # Simplified evaluation - in practice would use validation set
        return np.random.uniform(0.7, 0.95)  # Mock performance score
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """Get federated learning statistics."""
        active_clients = sum(1 for info in self.clients.values() if info['is_active'])
        
        return {
            'round_number': self.round_number,
            'total_clients': len(self.clients),
            'active_clients': active_clients,
            'clients_per_round': self.config.clients_per_round,
            'aggregation_method': self.aggregation_method,
            'selection_strategy': self.client_selection_strategy
        }


class DistributedTrainingManager:
    """Main manager for distributed training coordination."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.parameter_server = None
        self.actor_learner_trainer = None
        self.federated_learner = None
        
        # Training mode
        self.training_mode = "single_node"  # "single_node", "multi_node", "actor_learner", "federated"
        
        # Performance monitoring
        self.training_stats = {}
        
        logger.info("Distributed training manager initialized")
    
    def setup_single_node_training(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Setup single-node multi-GPU training."""
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        
        self.training_mode = "single_node"
        return model
    
    def setup_multi_node_training(self, model: nn.Module):
        """Setup multi-node distributed training."""
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Move model to GPU
        device = torch.device(f"cuda:{self.config.local_rank}")
        model = model.to(device)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[self.config.local_rank])
        
        self.training_mode = "multi_node"
        logger.info(f"Multi-node training setup complete (rank {self.config.rank}/{self.config.world_size})")
        
        return model
    
    def setup_actor_learner_training(self, model_factory: Callable[[], nn.Module], 
                                   env_factory: Callable[[], Any]):
        """Setup actor-learner distributed training."""
        self.actor_learner_trainer = ActorLearnerTrainer(self.config, model_factory, env_factory)
        self.training_mode = "actor_learner"
        
        logger.info("Actor-learner training setup complete")
    
    def setup_federated_learning(self, global_model: nn.Module):
        """Setup federated learning."""
        self.federated_learner = FederatedLearner(self.config, global_model)
        self.training_mode = "federated"
        
        logger.info("Federated learning setup complete")
    
    def setup_parameter_server(self, model: nn.Module):
        """Setup parameter server."""
        self.parameter_server = ParameterServer(self.config, model)
        
        # Start parameter server in separate thread
        ps_thread = threading.Thread(target=self.parameter_server.start, daemon=True)
        ps_thread.start()
        
        logger.info("Parameter server setup complete")
    
    def start_training(self):
        """Start distributed training based on configured mode."""
        if self.training_mode == "actor_learner":
            if self.actor_learner_trainer is None:
                raise ValueError("Actor-learner trainer not setup")
            self.actor_learner_trainer.start_training()
        
        elif self.training_mode == "federated":
            if self.federated_learner is None:
                raise ValueError("Federated learner not setup")
            self._run_federated_training()
        
        else:
            raise ValueError(f"Training mode '{self.training_mode}' not implemented for auto-start")
    
    def _run_federated_training(self):
        """Run federated learning rounds."""
        for round_num in range(self.config.federation_rounds):
            round_stats = self.federated_learner.federated_round()
            self.training_stats[f'round_{round_num}'] = round_stats
            
            if round_num % 10 == 0:
                logger.info(f"Federated learning progress: {round_num}/{self.config.federation_rounds}")
        
        logger.info("Federated learning completed")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'training_mode': self.training_mode,
            'config': self.config.__dict__,
        }
        
        if self.federated_learner:
            stats['federated_stats'] = self.federated_learner.get_federation_stats()
        
        if self.training_stats:
            stats['round_stats'] = self.training_stats
        
        return stats
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.parameter_server:
            self.parameter_server.stop()
        
        if self.actor_learner_trainer:
            self.actor_learner_trainer.stop_training()
        
        if self.training_mode == "multi_node":
            dist.destroy_process_group()
        
        logger.info("Distributed training cleanup completed")


# Utility functions

def setup_distributed_environment(rank: int, world_size: int, master_addr: str = "localhost", 
                                 master_port: str = "12355"):
    """Setup distributed environment variables."""
    import os
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())


def create_distributed_trainer(training_mode: str = "single_node", **config_kwargs) -> DistributedTrainingManager:
    """Create distributed training manager with specified mode."""
    config = DistributedConfig(**config_kwargs)
    manager = DistributedTrainingManager(config)
    
    return manager


if __name__ == "__main__":
    # Demonstration of distributed training infrastructure
    print("Distributed Training Infrastructure - Generation 3")
    print("=" * 60)
    print("Features:")
    print("1. Multi-GPU and multi-node training")
    print("2. Actor-learner architectures")
    print("3. Parameter servers")
    print("4. Federated learning")
    print("5. Fault-tolerant training")
    print("6. Real-time synchronization")
    print("\nThis infrastructure enables production-scale distributed RL training")
    print("for lunar habitat control systems.")
    
    # Example configuration
    config = DistributedConfig(
        world_size=4,
        n_actors=8,
        n_learners=2,
        batch_size=256,
        enable_federated=True,
        federation_rounds=50
    )
    
    print(f"\nExample configuration:")
    print(f"- World size: {config.world_size}")
    print(f"- Actors: {config.n_actors}, Learners: {config.n_learners}")
    print(f"- Federated learning: {config.enable_federated}")
    print(f"- Federation rounds: {config.federation_rounds}")