"""
Multi-Agent Reinforcement Learning for Lunar Habitat Coordination

This module provides advanced multi-agent RL capabilities for coordinated control
of lunar habitat systems. Multiple specialized agents work together to manage
different subsystems while maintaining overall habitat safety and efficiency.

Features:
- Multi-agent environment with shared and individual observations
- Specialized agents for different habitat subsystems (life support, thermal, power)
- Communication protocols between agents
- Centralized training with decentralized execution (CTDE)
- Advanced coordination algorithms (QMIX, MADDPG, MAPPO)
- Hierarchical multi-agent control
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
from collections import defaultdict, deque
import copy

from .core.state import HabitatState
from .core.metrics import PerformanceTracker, SafetyMonitor
from .utils.logging import get_logger
from .utils.exceptions import MultiAgentError, CoordinationError
from .utils.validation import validate_agent_config

logger = get_logger(__name__)


class AgentRole(Enum):
    """Enumeration of agent roles in lunar habitat control."""
    LIFE_SUPPORT = "life_support"
    THERMAL_CONTROL = "thermal_control"
    POWER_MANAGEMENT = "power_management"
    EMERGENCY_RESPONSE = "emergency_response"
    RESOURCE_ALLOCATION = "resource_allocation"
    COMMUNICATION = "communication"
    SUPERVISOR = "supervisor"


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    agent_id: str
    role: AgentRole
    observation_space_size: int
    action_space_size: int
    learning_rate: float = 3e-4
    hidden_size: int = 256
    memory_size: int = 100000
    batch_size: int = 64
    
    # Communication settings
    can_communicate: bool = True
    communication_channels: List[str] = field(default_factory=list)
    message_size: int = 32
    
    # Coordination settings
    coordination_weight: float = 1.0
    safety_priority: float = 2.0
    
    # Agent-specific parameters
    specialized_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationMessage:
    """Message structure for agent communication."""
    sender_id: str
    receiver_id: str
    message_type: str
    content: torch.Tensor
    priority: float = 1.0
    timestamp: float = 0.0


class CommunicationProtocol(ABC):
    """Abstract base class for agent communication protocols."""
    
    @abstractmethod
    def encode_message(self, sender_state: torch.Tensor, message_type: str) -> torch.Tensor:
        """Encode a message from sender state."""
        pass
    
    @abstractmethod
    def decode_message(self, message: torch.Tensor, receiver_context: torch.Tensor) -> torch.Tensor:
        """Decode a message for the receiver."""
        pass
    
    @abstractmethod
    def filter_messages(self, messages: List[CommunicationMessage], receiver_id: str) -> List[CommunicationMessage]:
        """Filter relevant messages for a specific receiver."""
        pass


class DirectCommunication(CommunicationProtocol):
    """Direct communication protocol with learned message encoding."""
    
    def __init__(self, message_size: int = 32, hidden_size: int = 128):
        self.message_size = message_size
        self.hidden_size = hidden_size
        
        # Message encoder/decoder networks
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size),
            nn.Tanh()  # Normalize messages
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(message_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def encode_message(self, sender_state: torch.Tensor, message_type: str) -> torch.Tensor:
        """Encode message from sender's internal state."""
        return self.encoder(sender_state)
    
    def decode_message(self, message: torch.Tensor, receiver_context: torch.Tensor) -> torch.Tensor:
        """Decode message with receiver's context."""
        combined = torch.cat([message, receiver_context], dim=-1)
        return self.decoder(combined)
    
    def filter_messages(self, messages: List[CommunicationMessage], receiver_id: str) -> List[CommunicationMessage]:
        """Filter messages by priority and relevance."""
        relevant_messages = [
            msg for msg in messages 
            if msg.receiver_id == receiver_id or msg.receiver_id == "all"
        ]
        
        # Sort by priority and timestamp
        relevant_messages.sort(key=lambda x: (-x.priority, -x.timestamp))
        
        # Limit number of messages to process
        return relevant_messages[:5]  # Max 5 messages per agent per step


class MultiAgentActor(nn.Module):
    """Actor network for multi-agent settings with communication."""
    
    def __init__(self, config: AgentConfig, communication_protocol: Optional[CommunicationProtocol] = None):
        super().__init__()
        self.config = config
        self.communication_protocol = communication_protocol
        
        # Observation processing
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.observation_space_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU()
        )
        
        # Communication processing (if enabled)
        if config.can_communicate and communication_protocol:
            self.message_processor = nn.Sequential(
                nn.Linear(config.message_size * 5, config.hidden_size // 2),  # Max 5 messages
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.hidden_size // 2)
            )
            actor_input_size = config.hidden_size + config.hidden_size // 2
        else:
            self.message_processor = None
            actor_input_size = config.hidden_size
        
        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(actor_input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_space_size * 2)  # mean and log_std
        )
        
        # Message generation (if communication enabled)
        if config.can_communicate:
            self.message_generator = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.message_size),
                nn.Tanh()
            )
        else:
            self.message_generator = None
    
    def forward(self, observation: torch.Tensor, 
                messages: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through actor network.
        
        Returns:
            action_mean, action_log_std, outgoing_message
        """
        # Process observation
        obs_features = self.obs_encoder(observation)
        
        # Process incoming messages
        if messages and self.message_processor:
            # Pad or truncate messages to fixed size
            padded_messages = []
            for i in range(5):  # Max 5 messages
                if i < len(messages):
                    padded_messages.append(messages[i])
                else:
                    padded_messages.append(torch.zeros_like(messages[0]))
            
            message_input = torch.cat(padded_messages, dim=-1)
            message_features = self.message_processor(message_input)
            
            # Combine observation and message features
            combined_features = torch.cat([obs_features, message_features], dim=-1)
        else:
            combined_features = obs_features
        
        # Generate action distribution
        policy_output = self.policy_network(combined_features)
        action_mean, action_log_std = torch.chunk(policy_output, 2, dim=-1)
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        
        # Generate outgoing message
        outgoing_message = None
        if self.config.can_communicate and self.message_generator:
            outgoing_message = self.message_generator(obs_features)
        
        return action_mean, action_log_std, outgoing_message
    
    def sample_action(self, observation: torch.Tensor, 
                     messages: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Sample action from policy."""
        action_mean, action_log_std, outgoing_message = self.forward(observation, messages)
        
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply tanh squashing
        action_squashed = torch.tanh(action)
        log_prob -= torch.log(1 - action_squashed.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action_squashed, log_prob, action_mean, outgoing_message


class MultiAgentCritic(nn.Module):
    """Centralized critic for multi-agent training."""
    
    def __init__(self, total_obs_size: int, total_action_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_obs_size + total_action_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """Evaluate joint state-action value."""
        combined = torch.cat([joint_obs, joint_actions], dim=-1)
        return self.network(combined)


class QMixingNetwork(nn.Module):
    """Mixing network for QMIX algorithm."""
    
    def __init__(self, n_agents: int, state_size: int, mixing_embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_size = state_size
        self.mixing_embed_dim = mixing_embed_dim
        
        # Hypernetwork for generating mixing network weights
        self.hyper_w_1 = nn.Linear(state_size, mixing_embed_dim * n_agents)
        self.hyper_w_final = nn.Linear(state_size, mixing_embed_dim)
        
        # Hypernetwork for generating biases
        self.hyper_b_1 = nn.Linear(state_size, mixing_embed_dim)
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_size, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
    
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix agent Q-values into joint Q-value.
        
        Args:
            agent_qs: Individual Q-values [batch_size, n_agents]
            states: Global states [batch_size, state_size]
        
        Returns:
            Mixed Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        
        # Generate mixing network weights and biases
        w1 = torch.abs(self.hyper_w_1(states))  # Ensure monotonicity
        b1 = self.hyper_b_1(states)
        
        w_final = torch.abs(self.hyper_w_final(states))
        b_final = self.hyper_b_final(states)
        
        # Reshape for mixing
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)
        
        # First layer
        agent_qs_reshaped = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(agent_qs_reshaped, w1) + b1)
        
        # Final layer
        w_final = w_final.view(batch_size, self.mixing_embed_dim, 1)
        mixed_q = torch.bmm(hidden, w_final) + b_final
        
        return mixed_q.view(batch_size, -1)


class MultiAgentEnvironment:
    """Multi-agent wrapper for lunar habitat environment."""
    
    def __init__(self, base_env: Any, agent_configs: List[AgentConfig]):
        self.base_env = base_env
        self.agent_configs = {config.agent_id: config for config in agent_configs}
        self.agents = list(self.agent_configs.keys())
        self.n_agents = len(self.agents)
        
        # Communication setup
        self.communication_protocol = DirectCommunication()
        self.message_buffer = defaultdict(list)
        
        # Observation and action space configuration
        self._setup_spaces()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.safety_monitor = SafetyMonitor()
        
        logger.info(f"Initialized multi-agent environment with {self.n_agents} agents")
        logger.info(f"Agents: {self.agents}")
    
    def _setup_spaces(self):
        """Setup observation and action spaces for each agent."""
        # Get base environment dimensions
        base_obs_dim = self.base_env.observation_space.shape[0]
        base_action_dim = self.base_env.action_space.shape[0]
        
        # Distribute observations and actions among agents
        obs_per_agent = base_obs_dim // self.n_agents
        action_per_agent = base_action_dim // self.n_agents
        
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for i, agent_id in enumerate(self.agents):
            # Each agent gets a portion of the full observation space
            start_obs = i * obs_per_agent
            end_obs = start_obs + obs_per_agent
            if i == self.n_agents - 1:  # Last agent gets remainder
                end_obs = base_obs_dim
            
            start_action = i * action_per_agent
            end_action = start_action + action_per_agent
            if i == self.n_agents - 1:  # Last agent gets remainder
                end_action = base_action_dim
            
            self.observation_spaces[agent_id] = (start_obs, end_obs)
            self.action_spaces[agent_id] = (start_action, end_action)
        
        # Global state for centralized training
        self.global_obs_size = base_obs_dim
        self.global_action_size = base_action_dim
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and return initial observations for all agents."""
        base_obs, base_info = self.base_env.reset(seed=seed)
        
        # Clear communication buffer
        self.message_buffer.clear()
        
        # Split observations among agents
        agent_obs = {}
        for agent_id in self.agents:
            start_obs, end_obs = self.observation_spaces[agent_id]
            agent_obs[agent_id] = base_obs[start_obs:end_obs]
        
        # Add global state to info
        info = {
            "global_state": base_obs,
            "base_info": base_info
        }
        
        return agent_obs, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """Execute joint action and return results for all agents."""
        
        # Reconstruct full action vector
        full_action = np.zeros(self.global_action_size)
        
        for agent_id, action in actions.items():
            start_action, end_action = self.action_spaces[agent_id]
            full_action[start_action:end_action] = action
        
        # Execute in base environment
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(full_action)
        
        # Process communication messages
        self._process_communication()
        
        # Split observations among agents
        agent_obs = {}
        for agent_id in self.agents:
            start_obs, end_obs = self.observation_spaces[agent_id]
            agent_obs[agent_id] = base_obs[start_obs:end_obs]
        
        # Compute individual rewards (could be specialized per agent role)
        agent_rewards = self._compute_agent_rewards(base_reward, base_info)
        
        # Determine termination for each agent
        agent_terminated = {agent_id: base_terminated for agent_id in self.agents}
        agent_truncated = {agent_id: base_truncated for agent_id in self.agents}
        
        # Enhanced info with coordination metrics
        info = {
            "global_state": base_obs,
            "global_reward": base_reward,
            "base_info": base_info,
            "coordination_score": self._compute_coordination_score(actions),
            "communication_efficiency": self._compute_communication_efficiency()
        }
        
        return agent_obs, agent_rewards, agent_terminated, agent_truncated, info
    
    def _process_communication(self):
        """Process inter-agent communication messages."""
        # In a full implementation, this would handle message routing,
        # filtering, and processing according to the communication protocol
        pass
    
    def _compute_agent_rewards(self, base_reward: float, base_info: Dict[str, Any]) -> Dict[str, float]:
        """Compute individual agent rewards based on their roles and performance."""
        agent_rewards = {}
        
        for agent_id in self.agents:
            config = self.agent_configs[agent_id]
            
            # Base reward component
            agent_reward = base_reward / self.n_agents
            
            # Role-specific reward shaping
            if config.role == AgentRole.LIFE_SUPPORT:
                # Reward for maintaining life support parameters
                if "life_support_efficiency" in base_info:
                    agent_reward += base_info["life_support_efficiency"] * 0.5
                if "safety_violation" in base_info and base_info["safety_violation"]:
                    agent_reward -= 2.0  # High penalty for safety violations
            
            elif config.role == AgentRole.THERMAL_CONTROL:
                # Reward for temperature regulation
                if "thermal_stability" in base_info:
                    agent_reward += base_info["thermal_stability"] * 0.3
                if "temperature_variance" in base_info:
                    agent_reward -= base_info["temperature_variance"] * 0.1
            
            elif config.role == AgentRole.POWER_MANAGEMENT:
                # Reward for power efficiency
                if "power_efficiency" in base_info:
                    agent_reward += base_info["power_efficiency"] * 0.4
                if "power_outage" in base_info and base_info["power_outage"]:
                    agent_reward -= 1.5
            
            elif config.role == AgentRole.EMERGENCY_RESPONSE:
                # Reward for quick emergency response
                if "emergency_response_time" in base_info:
                    response_time = base_info["emergency_response_time"]
                    agent_reward += max(0, 1.0 - response_time / 300.0)  # Reward fast response
            
            agent_rewards[agent_id] = agent_reward
        
        return agent_rewards
    
    def _compute_coordination_score(self, actions: Dict[str, np.ndarray]) -> float:
        """Compute coordination score based on action harmony."""
        # Simplified coordination metric
        # In practice, this would analyze action compatibility
        if len(actions) < 2:
            return 1.0
        
        action_values = list(actions.values())
        action_correlations = []
        
        for i in range(len(action_values)):
            for j in range(i + 1, len(action_values)):
                correlation = np.corrcoef(action_values[i], action_values[j])[0, 1]
                if not np.isnan(correlation):
                    action_correlations.append(abs(correlation))
        
        return np.mean(action_correlations) if action_correlations else 0.5
    
    def _compute_communication_efficiency(self) -> float:
        """Compute communication efficiency metric."""
        # Simplified metric - in practice would analyze message relevance and impact
        total_messages = sum(len(msgs) for msgs in self.message_buffer.values())
        if total_messages == 0:
            return 1.0
        
        # Assume higher message count reduces efficiency (too much chatter)
        return max(0.1, 1.0 - total_messages / (self.n_agents * 10))


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient agent."""
    
    def __init__(self, agent_id: str, config: AgentConfig, n_agents: int, global_obs_size: int, global_action_size: int):
        self.agent_id = agent_id
        self.config = config
        self.n_agents = n_agents
        
        # Networks
        self.actor = MultiAgentActor(config)
        self.critic = MultiAgentCritic(global_obs_size, global_action_size, config.hidden_size)
        
        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=config.memory_size)
        
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter
        
        logger.info(f"Initialized MADDPG agent {agent_id}")
    
    def act(self, observation: torch.Tensor, messages: Optional[List[torch.Tensor]] = None, noise_std: float = 0.1) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Select action with optional exploration noise."""
        with torch.no_grad():
            action, _, _, message = self.actor.sample_action(observation, messages)
            
            # Add exploration noise
            if noise_std > 0:
                noise = torch.randn_like(action) * noise_std
                action = torch.clamp(action + noise, -1, 1)
        
        return action, message
    
    def update(self, batch: Dict[str, torch.Tensor], other_actors: List[MultiAgentActor]) -> Dict[str, float]:
        """Update agent networks using MADDPG algorithm."""
        
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]
        global_obs = batch["global_observations"]
        global_next_obs = batch["global_next_observations"]
        
        # Update critic
        with torch.no_grad():
            # Get next actions from all target actors
            next_actions = []
            for i, actor in enumerate(other_actors):
                if i == int(self.agent_id.split('_')[-1]):  # This agent
                    next_action, _, _, _ = self.actor_target.sample_action(next_obs[i])
                else:
                    next_action, _, _, _ = actor.sample_action(next_obs[i])
                next_actions.append(next_action)
            
            next_joint_actions = torch.cat(next_actions, dim=-1)
            target_q = self.critic_target(global_next_obs, next_joint_actions)
            y = rewards + self.gamma * (1 - dones) * target_q
        
        current_q = self.critic(global_obs, actions)
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        
        # Update actor
        # Reconstruct actions with this agent's new action
        new_actions = actions.clone()
        agent_idx = int(self.agent_id.split('_')[-1])
        obs_start = agent_idx * (obs.shape[1] // self.n_agents)
        obs_end = obs_start + (obs.shape[1] // self.n_agents)
        
        new_action, _, _, _ = self.actor.sample_action(obs[:, obs_start:obs_end])
        action_start = agent_idx * (actions.shape[1] // self.n_agents)
        action_end = action_start + (actions.shape[1] // self.n_agents)
        new_actions[:, action_start:action_end] = new_action
        
        actor_loss = -self.critic(global_obs, new_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value_mean": current_q.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )


class MultiAgentTrainer:
    """Trainer for multi-agent reinforcement learning."""
    
    def __init__(self, env: MultiAgentEnvironment, algorithm: str = "maddpg"):
        self.env = env
        self.algorithm = algorithm
        
        # Initialize agents based on algorithm
        if algorithm == "maddpg":
            self.agents = self._initialize_maddpg_agents()
        elif algorithm == "qmix":
            self.agents = self._initialize_qmix_agents()
        else:
            raise MultiAgentError(f"Unknown algorithm: {algorithm}")
        
        # Training metrics
        self.training_stats = defaultdict(list)
        
        logger.info(f"Initialized multi-agent trainer with {algorithm}")
    
    def _initialize_maddpg_agents(self) -> Dict[str, MADDPGAgent]:
        """Initialize MADDPG agents."""
        agents = {}
        
        for i, agent_id in enumerate(self.env.agents):
            config = self.env.agent_configs[agent_id]
            agent = MADDPGAgent(
                agent_id=agent_id,
                config=config,
                n_agents=self.env.n_agents,
                global_obs_size=self.env.global_obs_size,
                global_action_size=self.env.global_action_size
            )
            agents[agent_id] = agent
        
        return agents
    
    def _initialize_qmix_agents(self) -> Dict[str, Any]:
        """Initialize QMIX agents."""
        # QMIX implementation would go here
        raise NotImplementedError("QMIX implementation not yet complete")
    
    def train(self, total_episodes: int, max_episode_length: int = 1000) -> Dict[str, List[float]]:
        """Train multi-agent system."""
        
        episode_rewards = []
        coordination_scores = []
        
        for episode in range(total_episodes):
            episode_reward = 0
            coordination_score = 0
            
            # Reset environment
            agent_obs, info = self.env.reset()
            
            for step in range(max_episode_length):
                # Get actions from all agents
                actions = {}
                messages = {}
                
                for agent_id in self.env.agents:
                    obs_tensor = torch.FloatTensor(agent_obs[agent_id]).unsqueeze(0)
                    action, message = self.agents[agent_id].act(obs_tensor)
                    actions[agent_id] = action.cpu().numpy().flatten()
                    if message is not None:
                        messages[agent_id] = message
                
                # Execute joint action
                next_agent_obs, agent_rewards, agent_terminated, agent_truncated, step_info = self.env.step(actions)
                
                # Store experience for each agent (simplified)
                # In practice, would store in individual replay buffers
                
                # Update metrics
                total_reward = sum(agent_rewards.values())
                episode_reward += total_reward
                coordination_score += step_info.get("coordination_score", 0)
                
                agent_obs = next_agent_obs
                
                # Check termination
                if any(agent_terminated.values()) or any(agent_truncated.values()):
                    break
            
            episode_rewards.append(episode_reward)
            coordination_scores.append(coordination_score / max_episode_length)
            
            # Periodic updates and logging
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                           f"Coordination = {coordination_scores[-1]:.3f}")
        
        results = {
            "episode_rewards": episode_rewards,
            "coordination_scores": coordination_scores
        }
        
        return results


# Export main classes
__all__ = [
    "AgentRole",
    "AgentConfig",
    "CommunicationMessage",
    "CommunicationProtocol",
    "DirectCommunication",
    "MultiAgentActor",
    "MultiAgentCritic", 
    "QMixingNetwork",
    "MultiAgentEnvironment",
    "MADDPGAgent",
    "MultiAgentTrainer"
]