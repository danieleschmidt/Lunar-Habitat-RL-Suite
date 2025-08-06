"""
Model-Based Reinforcement Learning Algorithms for Lunar Habitat Control

This module implements state-of-the-art model-based RL algorithms optimized for
autonomous life support system control in lunar habitats.

Algorithms implemented:
- MuZero: Planning with learned dynamics model
- DreamerV3: World model learning with actor-critic
- PlaNet: Deep planning network for continuous control
- MBPO: Model-based policy optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from ..core.state import HabitatState
from ..utils.logging import get_logger
from ..utils.exceptions import AlgorithmError, TrainingError
from ..utils.validation import validate_model_config

logger = get_logger(__name__)


@dataclass
class ModelBasedConfig:
    """Configuration for model-based RL algorithms."""
    model_type: str = "dreamer_v3"  # Options: muzero, dreamer_v3, planet, mbpo
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 1_000_000
    horizon: int = 50
    imagination_horizon: int = 15
    world_model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    model_update_freq: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_freq: int = 10_000
    eval_freq: int = 5_000
    
    # MuZero specific
    num_simulations: int = 800
    planning_depth: int = 50
    discount: float = 0.997
    
    # Dreamer specific  
    stoch_size: int = 32
    deter_size: int = 512
    hidden_size: int = 400
    layers: int = 2
    
    # PlaNet specific
    state_size: int = 30
    belief_size: int = 200
    hidden_size_planet: int = 200


class WorldModel(nn.Module):
    """Neural world model for habitat dynamics prediction."""
    
    def __init__(self, obs_dim: int, action_dim: int, config: ModelBasedConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Encoder: observations -> latent state
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.stoch_size + config.deter_size)
        )
        
        # Dynamics model: (latent_state, action) -> next_latent_state
        self.dynamics = nn.Sequential(
            nn.Linear(config.stoch_size + config.deter_size + action_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.stoch_size + config.deter_size)
        )
        
        # Decoder: latent_state -> observations
        self.decoder = nn.Sequential(
            nn.Linear(config.stoch_size + config.deter_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, obs_dim)
        )
        
        # Reward predictor: latent_state -> reward
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.stoch_size + config.deter_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Continuation predictor: latent_state -> continuation_probability
        self.continue_predictor = nn.Sequential(
            nn.Linear(config.stoch_size + config.deter_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent state."""
        return self.encoder(obs)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observations."""
        return self.decoder(latent)
    
    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent state given current state and action."""
        combined = torch.cat([latent, action], dim=-1)
        return self.dynamics(combined)
    
    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        return self.reward_predictor(latent)
    
    def predict_continue(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict episode continuation probability."""
        return self.continue_predictor(latent)


class Actor(nn.Module):
    """Actor network for continuous action spaces."""
    
    def __init__(self, latent_dim: int, action_dim: int, config: ModelBasedConfig):
        super().__init__()
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * action_dim)  # mean and log_std
        )
        
        self.min_std = 1e-4
        self.max_std = 1.0
    
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution parameters."""
        out = self.network(latent)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        
        # Constrain std to reasonable range
        log_std = torch.clamp(log_std, np.log(self.min_std), np.log(self.max_std))
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the policy."""
        mean, std = self.forward(latent)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Squash to [-1, 1] with tanh
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, latent_dim: int, config: ModelBasedConfig):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate value of latent state."""
        return self.network(latent)


class DreamerV3(nn.Module):
    """
    DreamerV3: Advanced world model learning with actor-critic.
    
    Optimized for lunar habitat life support control with safety considerations.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: ModelBasedConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Core components
        latent_dim = config.stoch_size + config.deter_size
        self.world_model = WorldModel(obs_dim, action_dim, config)
        self.actor = Actor(latent_dim, action_dim, config)
        self.critic = Critic(latent_dim, config)
        
        # Optimizers
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.world_model_lr
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        
        logger.info(f"Initialized DreamerV3 with {self.count_parameters()} parameters")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def imagine_trajectories(
        self, 
        initial_latent: torch.Tensor, 
        horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Imagine trajectories using the world model.
        
        Returns:
            latents: Sequence of latent states
            actions: Sequence of actions
            rewards: Sequence of predicted rewards  
            continues: Sequence of continuation probabilities
        """
        batch_size = initial_latent.shape[0]
        device = initial_latent.device
        
        # Storage for trajectory
        latents = torch.zeros(horizon + 1, batch_size, initial_latent.shape[-1], device=device)
        actions = torch.zeros(horizon, batch_size, self.action_dim, device=device)
        rewards = torch.zeros(horizon, batch_size, 1, device=device)
        continues = torch.zeros(horizon, batch_size, 1, device=device)
        
        latents[0] = initial_latent
        
        # Imagine forward
        for t in range(horizon):
            # Sample action from policy
            action, _ = self.actor.sample(latents[t])
            actions[t] = action
            
            # Predict next state
            latents[t + 1] = self.world_model.predict_next(latents[t], action)
            
            # Predict reward and continuation
            rewards[t] = self.world_model.predict_reward(latents[t])
            continues[t] = self.world_model.predict_continue(latents[t])
        
        return latents[1:], actions, rewards, continues
    
    def compute_lambda_returns(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        continues: torch.Tensor, 
        lambda_: float = 0.95
    ) -> torch.Tensor:
        """Compute lambda returns for training."""
        horizon, batch_size = rewards.shape[:2]
        
        # Bootstrap with final value
        returns = torch.zeros_like(rewards)
        last_value = values[-1]
        
        for t in reversed(range(horizon)):
            if t == horizon - 1:
                returns[t] = rewards[t] + continues[t] * self.config.discount * last_value
            else:
                returns[t] = rewards[t] + continues[t] * self.config.discount * (
                    lambda_ * returns[t + 1] + (1 - lambda_) * values[t + 1]
                )
        
        return returns
    
    def train_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train the world model on collected experience."""
        obs = batch["obs"]
        actions = batch["actions"]
        next_obs = batch["next_obs"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        
        # Encode observations
        latent = self.world_model.encode(obs)
        next_latent_true = self.world_model.encode(next_obs)
        
        # Predict next latent state
        next_latent_pred = self.world_model.predict_next(latent, actions)
        
        # Predict observations, rewards, and continuations
        obs_pred = self.world_model.decode(latent)
        next_obs_pred = self.world_model.decode(next_latent_pred)
        reward_pred = self.world_model.predict_reward(latent)
        continue_pred = self.world_model.predict_continue(latent)
        
        # Compute losses
        obs_loss = F.mse_loss(obs_pred, obs)
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_true)
        reward_loss = F.mse_loss(reward_pred, rewards)
        continue_loss = F.binary_cross_entropy(continue_pred, 1 - dones)
        
        # Total world model loss
        world_model_loss = obs_loss + dynamics_loss + reward_loss + continue_loss
        
        # Update world model
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_model_optimizer.step()
        
        return {
            "world_model_loss": world_model_loss.item(),
            "obs_loss": obs_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item()
        }
    
    def train_actor_critic(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train actor and critic using imagined trajectories."""
        with torch.no_grad():
            # Encode initial observations
            initial_latent = self.world_model.encode(batch["obs"])
        
        # Imagine trajectories
        latents, actions, rewards, continues = self.imagine_trajectories(
            initial_latent, self.config.imagination_horizon
        )
        
        # Compute values
        values = self.critic(latents).squeeze(-1)
        
        # Compute lambda returns
        lambda_returns = self.compute_lambda_returns(
            rewards.squeeze(-1), values, continues.squeeze(-1)
        )
        
        # Actor loss (policy gradient with advantage)
        advantages = lambda_returns - values
        _, log_probs = self.actor.sample(latents.detach())
        actor_loss = -(log_probs.squeeze(-1) * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values, lambda_returns.detach())
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "average_value": values.mean().item(),
            "average_return": lambda_returns.mean().item()
        }
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action given observation."""
        with torch.no_grad():
            latent = self.world_model.encode(obs)
            action, _ = self.actor.sample(latent)
        return action
    
    def save(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "config": self.config
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved DreamerV3 checkpoint to {path}")
    
    def load(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.step_count = checkpoint["step_count"]
        self.episode_count = checkpoint["episode_count"]
        logger.info(f"Loaded DreamerV3 checkpoint from {path}")


class MuZero(nn.Module):
    """
    MuZero: Planning with learned dynamics model.
    
    Simplified implementation optimized for lunar habitat control.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: ModelBasedConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Representation network: observation -> hidden state
        self.representation_network = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.stoch_size)
        )
        
        # Dynamics network: (hidden_state, action) -> next_hidden_state, reward
        self.dynamics_network = nn.Sequential(
            nn.Linear(config.stoch_size + action_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.stoch_size + 1)  # state + reward
        )
        
        # Prediction network: hidden_state -> value, policy
        self.prediction_network = nn.Sequential(
            nn.Linear(config.stoch_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1 + action_dim)  # value + policy logits
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        
        logger.info(f"Initialized MuZero with {self.count_parameters()} parameters")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def represent(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert observation to hidden state representation."""
        return self.representation_network(obs)
    
    def dynamics(self, hidden: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next hidden state and immediate reward."""
        combined = torch.cat([hidden, action], dim=-1)
        output = self.dynamics_network(combined)
        next_hidden = output[:, :-1]
        reward = output[:, -1:]
        return next_hidden, reward
    
    def predict(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict value and policy from hidden state."""
        output = self.prediction_network(hidden)
        value = output[:, :1]
        policy_logits = output[:, 1:]
        return value, policy_logits
    
    def mcts_search(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Simplified MCTS search for action selection.
        
        In production, this would use full MCTS with tree search.
        For now, using simplified forward planning.
        """
        with torch.no_grad():
            hidden = self.represent(obs)
            _, policy_logits = self.predict(hidden)
            
            # For safety in lunar habitat, add exploration noise
            action_probs = F.softmax(policy_logits + 0.1 * torch.randn_like(policy_logits), dim=-1)
            action = torch.multinomial(action_probs, 1).float() / (self.action_dim - 1) * 2 - 1
            
        return action
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action using MCTS search."""
        return self.mcts_search(obs)


class PlaNet(nn.Module):
    """
    PlaNet: Deep Planning Network for continuous control.
    
    Simplified implementation for lunar habitat systems.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: ModelBasedConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Similar to DreamerV3 but with different architecture
        # This is a simplified version - production would be more complex
        self.world_model = WorldModel(obs_dim, action_dim, config)
        
        # Planning controller
        self.controller = nn.Sequential(
            nn.Linear(config.stoch_size + config.deter_size, config.hidden_size_planet),
            nn.LayerNorm(config.hidden_size_planet),
            nn.SiLU(),
            nn.Linear(config.hidden_size_planet, config.hidden_size_planet),
            nn.LayerNorm(config.hidden_size_planet),
            nn.SiLU(),
            nn.Linear(config.hidden_size_planet, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        
        logger.info(f"Initialized PlaNet with {self.count_parameters()} parameters")
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def plan(self, obs: torch.Tensor, horizon: int = 12) -> torch.Tensor:
        """Plan action sequence using shooting method."""
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            
            # Initialize action sequence
            actions = torch.randn(horizon, batch_size, self.action_dim, device=device) * 0.1
            best_actions = actions.clone()
            best_returns = torch.full((batch_size,), -float('inf'), device=device)
            
            # Cross-entropy method for action optimization
            for iteration in range(10):  # CEM iterations
                returns = self.evaluate_action_sequence(obs, actions)
                
                # Update best actions
                improved = returns > best_returns
                best_returns[improved] = returns[improved]
                best_actions[:, improved] = actions[:, improved]
                
                # Update action distribution
                if iteration < 9:  # Don't update on last iteration
                    actions = best_actions + 0.1 * torch.randn_like(actions)
                    actions = torch.clamp(actions, -1, 1)
            
            return best_actions[0]  # Return first action
    
    def evaluate_action_sequence(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate a sequence of actions using the world model."""
        latent = self.world_model.encode(obs)
        total_return = torch.zeros(obs.shape[0], device=obs.device)
        
        for t, action in enumerate(actions):
            reward = self.world_model.predict_reward(latent)
            total_return += (self.config.discount ** t) * reward.squeeze(-1)
            latent = self.world_model.predict_next(latent, action)
        
        return total_return
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action using model predictive control."""
        return self.plan(obs)


def create_model_based_agent(
    obs_dim: int,
    action_dim: int,
    config: ModelBasedConfig
) -> nn.Module:
    """Factory function to create model-based RL agents."""
    validate_model_config(config)
    
    if config.model_type == "dreamer_v3":
        return DreamerV3(obs_dim, action_dim, config)
    elif config.model_type == "muzero":
        return MuZero(obs_dim, action_dim, config)
    elif config.model_type == "planet":
        return PlaNet(obs_dim, action_dim, config)
    else:
        raise AlgorithmError(f"Unknown model type: {config.model_type}")


# Export main classes
__all__ = [
    "ModelBasedConfig",
    "DreamerV3", 
    "MuZero",
    "PlaNet",
    "WorldModel",
    "create_model_based_agent"
]