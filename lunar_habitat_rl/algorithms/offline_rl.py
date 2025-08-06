"""Offline reinforcement learning algorithms for lunar habitat control."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import copy
import logging

from ..utils.logging import get_logger

logger = get_logger("offline_rl")


@dataclass
class OfflineRLConfig:
    """Configuration for offline RL algorithms."""
    batch_size: int = 256
    learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    hidden_size: int = 256
    n_hidden: int = 2
    device: str = "auto"  # auto, cpu, cuda
    
    # Algorithm-specific parameters
    alpha: float = 0.2  # CQL regularization weight
    beta: float = 3.0   # IQL expectile parameter
    awac_lambda: float = 1.0  # AWAC weight coefficient


class QNetwork(nn.Module):
    """Q-network for value-based methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256, n_hidden: int = 2):
        super().__init__()
        
        layers = [nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Policy network for actor-critic methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256, n_hidden: int = 2):
        super().__init__()
        
        layers = [nn.Linear(state_dim, hidden_size), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        
        self.backbone = nn.Sequential(*layers)
        
        # Output mean and log standard deviation
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)
        
        # Action bounds for tanh squashing
        self.action_scale = 1.0
        self.action_bias = 0.0
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network."""
        x = self.backbone(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class CQL:
    """Conservative Q-Learning for offline RL."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[OfflineRLConfig] = None):
        self.config = config or OfflineRLConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Initialize networks
        self.q1 = QNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        self.policy = PolicyNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        
        # Training statistics
        self.training_stats = {
            'q1_loss': [],
            'q2_loss': [],
            'policy_loss': [],
            'cql_loss': []
        }
        
        logger.info(f"Initialized CQL agent on {self.device}")
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update CQL agent with a batch of data."""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.discount * target_q
        
        # Current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        # Bellman loss
        q1_bellman_loss = nn.MSELoss()(current_q1, target_q)
        q2_bellman_loss = nn.MSELoss()(current_q2, target_q)
        
        # CQL regularization loss
        cql_q1_loss = self._compute_cql_loss(self.q1, states, actions)
        cql_q2_loss = self._compute_cql_loss(self.q2, states, actions)
        
        # Total Q-losses
        q1_loss = q1_bellman_loss + self.config.alpha * cql_q1_loss
        q2_loss = q2_bellman_loss + self.config.alpha * cql_q2_loss
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Policy loss
        policy_actions, policy_log_probs = self.policy.sample(states)
        policy_q1 = self.q1(states, policy_actions)
        policy_q2 = self.q2(states, policy_actions)
        policy_q = torch.min(policy_q1, policy_q2)
        
        policy_loss = -policy_q.mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        # Update statistics
        stats = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'cql_loss': (cql_q1_loss + cql_q2_loss).item() / 2
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def _compute_cql_loss(self, q_network: QNetwork, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute CQL regularization loss."""
        batch_size = states.shape[0]
        
        # Sample random actions
        random_actions = torch.rand(batch_size, self.action_dim, device=self.device) * 2 - 1
        
        # Sample actions from current policy
        policy_actions, _ = self.policy.sample(states)
        
        # Q-values for random and policy actions
        q_random = q_network(states, random_actions)
        q_policy = q_network(states, policy_actions)
        q_data = q_network(states, actions)
        
        # CQL loss: maximize Q for random/policy actions, minimize for data actions
        cql_loss = torch.logsumexp(torch.cat([q_random, q_policy], dim=1), dim=1).mean() - q_data.mean()
        
        return cql_loss
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau
            )
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_tensor)
        
        return action.cpu().numpy().squeeze(0)
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        
        logger.info(f"Saved CQL model to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        logger.info(f"Loaded CQL model from {filepath}")


class IQL:
    """Implicit Q-Learning for offline RL."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[OfflineRLConfig] = None):
        self.config = config or OfflineRLConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Initialize networks
        self.q1 = QNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        
        # Value network for IQL
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, self.config.hidden_size),
            nn.ReLU(),
            *[layer for _ in range(self.config.n_hidden - 1) 
              for layer in [nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.ReLU()]],
            nn.Linear(self.config.hidden_size, 1)
        ).to(self.device)
        
        self.policy = PolicyNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        
        # Training statistics
        self.training_stats = {
            'q_loss': [],
            'value_loss': [],
            'policy_loss': []
        }
        
        logger.info(f"Initialized IQL agent on {self.device}")
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update IQL agent with a batch of data."""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update Q-networks
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        with torch.no_grad():
            next_values = self.value_network(next_states)
            target_q = rewards + (1 - dones) * self.config.discount * next_values
        
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        # Update value network with expectile regression
        with torch.no_grad():
            target_q1 = self.q1(states, actions)
            target_q2 = self.q2(states, actions)
            target_q = torch.min(target_q1, target_q2)
        
        current_values = self.value_network(states)
        value_loss = self._expectile_loss(current_values - target_q, self.config.beta)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy with advantage weighting
        with torch.no_grad():
            advantages = target_q - current_values
            weights = torch.exp(advantages / self.config.beta)
            weights = torch.clamp(weights, max=100.0)  # Prevent explosion
        
        policy_actions, policy_log_probs = self.policy.sample(states)
        policy_loss = -(weights * policy_log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update statistics
        stats = {
            'q_loss': q_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def _expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """Compute expectile loss for value function."""
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_tensor)
        
        return action.cpu().numpy().squeeze(0)
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        
        logger.info(f"Saved IQL model to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        logger.info(f"Loaded IQL model from {filepath}")


class AWAC:
    """Advantage-Weighted Actor-Critic for offline RL."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[OfflineRLConfig] = None):
        self.config = config or OfflineRLConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Initialize networks
        self.q1 = QNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        self.policy = PolicyNetwork(state_dim, action_dim, self.config.hidden_size, self.config.n_hidden).to(self.device)
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        
        # Training statistics
        self.training_stats = {
            'q1_loss': [],
            'q2_loss': [],
            'policy_loss': []
        }
        
        logger.info(f"Initialized AWAC agent on {self.device}")
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update AWAC agent with a batch of data."""
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update Q-networks (standard TD learning)
        with torch.no_grad():
            next_actions, _ = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.discount * target_q
        
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        
        # Update policy with advantage weighting
        with torch.no_grad():
            # Compute advantages
            current_q = torch.min(self.q1(states, actions), self.q2(states, actions))
            baseline_actions, _ = self.policy.sample(states)
            baseline_q = torch.min(self.q1(states, baseline_actions), self.q2(states, baseline_actions))
            
            advantages = current_q - baseline_q
            weights = torch.exp(advantages / self.config.awac_lambda)
            weights = torch.clamp(weights, max=20.0)  # Prevent explosion
        
        # Policy loss with advantage weighting
        _, log_probs = self.policy.sample(states)
        policy_loss = -(weights * log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update statistics
        stats = {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau
            )
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_tensor)
        
        return action.cpu().numpy().squeeze(0)
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        
        logger.info(f"Saved AWAC model to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        logger.info(f"Loaded AWAC model from {filepath}")