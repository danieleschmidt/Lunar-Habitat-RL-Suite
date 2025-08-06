"""Baseline agents for comparison and benchmarking."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from collections import deque

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
    from stable_baselines3.common.vec_env import VecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger("baselines")


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation."""
        pass
    
    @abstractmethod
    def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train the agent."""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save agent to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent from file."""
        pass


class RandomAgent(BaseAgent):
    """Random baseline agent for comparison."""
    
    def __init__(self, action_space: Any, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            action_space: Gymnasium action space
            seed: Random seed for reproducibility
        """
        self.action_space = action_space
        self.rng = np.random.RandomState(seed)
        
        logger.info("Initialized Random agent")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Sample random action from action space."""
        # Use internal RNG for consistency
        if hasattr(self.action_space, 'sample'):
            # Set seed for action space sampling
            if hasattr(self.action_space, 'seed'):
                self.action_space.seed(self.rng.randint(0, 2**32 - 1))
            return self.action_space.sample()
        else:
            # Fallback for simple spaces
            return self.rng.uniform(
                low=self.action_space.low,
                high=self.action_space.high,
                size=self.action_space.shape
            ).astype(np.float32)
    
    def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Random agent doesn't train - just collect statistics."""
        logger.info(f"Running random agent for {total_timesteps} timesteps")
        
        stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': 0.0
        }
        
        timesteps = 0
        episodes = 0
        
        while timesteps < total_timesteps:
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated) and timesteps < total_timesteps:
                action = self.predict(obs)
                obs, reward, done, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                timesteps += 1
            
            episodes += 1
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            
            # Log progress periodically
            if episodes % 10 == 0:
                avg_reward = np.mean(stats['episode_rewards'][-10:])
                logger.info(f"Episode {episodes}, Avg Reward: {avg_reward:.2f}")
        
        stats['total_timesteps'] = timesteps
        stats['episodes'] = episodes
        
        if stats['episode_rewards']:
            stats['mean_reward'] = np.mean(stats['episode_rewards'])
            stats['std_reward'] = np.std(stats['episode_rewards'])
            stats['mean_length'] = np.mean(stats['episode_lengths'])
        
        logger.info(f"Random agent completed {episodes} episodes")
        return stats
    
    def save(self, filepath: str):
        """Save random agent configuration."""
        np.savez(filepath, 
                seed=self.rng.get_state(),
                action_space_info={
                    'shape': getattr(self.action_space, 'shape', None),
                    'low': getattr(self.action_space, 'low', None),
                    'high': getattr(self.action_space, 'high', None)
                })
        logger.info(f"Saved random agent to {filepath}")
    
    def load(self, filepath: str):
        """Load random agent configuration."""
        data = np.load(filepath, allow_pickle=True)
        if 'seed' in data:
            self.rng.set_state(data['seed'].item())
        logger.info(f"Loaded random agent from {filepath}")


class ConstantAgent(BaseAgent):
    """Constant action baseline agent."""
    
    def __init__(self, action: np.ndarray):
        """
        Initialize constant agent.
        
        Args:
            action: Constant action to always take
        """
        self.action = np.array(action, dtype=np.float32)
        logger.info(f"Initialized Constant agent with action: {self.action}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return constant action."""
        return self.action.copy()
    
    def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Constant agent doesn't train - just collect statistics."""
        logger.info(f"Running constant agent for {total_timesteps} timesteps")
        
        stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        timesteps = 0
        episodes = 0
        
        while timesteps < total_timesteps:
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated) and timesteps < total_timesteps:
                action = self.predict(obs)
                obs, reward, done, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                timesteps += 1
            
            episodes += 1
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
        
        stats['total_timesteps'] = timesteps
        stats['episodes'] = episodes
        
        if stats['episode_rewards']:
            stats['mean_reward'] = np.mean(stats['episode_rewards'])
            stats['std_reward'] = np.std(stats['episode_rewards'])
            stats['mean_length'] = np.mean(stats['episode_lengths'])
        
        return stats
    
    def save(self, filepath: str):
        """Save constant agent."""
        np.save(filepath, self.action)
        logger.info(f"Saved constant agent to {filepath}")
    
    def load(self, filepath: str):
        """Load constant agent."""
        self.action = np.load(filepath).astype(np.float32)
        logger.info(f"Loaded constant agent from {filepath}")


class HeuristicAgent(BaseAgent):
    """Heuristic agent with domain knowledge for lunar habitat control."""
    
    def __init__(self, action_space: Any):
        """
        Initialize heuristic agent.
        
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space
        self.action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else len(action_space.high)
        
        # Define heuristic control parameters
        self.control_gains = {
            'o2_gain': 0.1,
            'co2_gain': 0.2,
            'temp_gain': 0.05,
            'power_gain': 0.1
        }
        
        logger.info("Initialized Heuristic agent with domain knowledge")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Generate action using heuristic control logic.
        
        This implements basic control logic for lunar habitat systems:
        - Increase O2 generation if O2 is low
        - Increase CO2 scrubbing if CO2 is high  
        - Adjust heating/cooling based on temperature
        - Manage power based on battery level and solar availability
        """
        # Extract state variables (approximate indices based on typical state structure)
        obs_len = len(observation)
        
        if obs_len >= 7:  # Minimum expected state size
            # Atmosphere variables
            o2_pressure = observation[0]  # kPa
            co2_pressure = observation[1]  # kPa
            n2_pressure = observation[2] if obs_len > 2 else 79.0
            total_pressure = observation[3] if obs_len > 3 else 101.3
            humidity = observation[4] if obs_len > 4 else 45.0
            temperature = observation[5] if obs_len > 5 else 22.5
            air_quality = observation[6] if obs_len > 6 else 0.95
            
            # Power variables (if available)
            solar_gen = observation[7] if obs_len > 7 else 5.0
            battery_charge = observation[8] if obs_len > 8 else 75.0
            
        else:
            # Fallback to default values if observation is too short
            o2_pressure = 21.0
            co2_pressure = 0.4
            temperature = 22.5
            battery_charge = 75.0
            solar_gen = 5.0
        
        # Initialize action array
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # Heuristic control logic
        idx = 0
        
        # Life support control (first 6 actions assumed)
        if idx < self.action_dim:
            # O2 generation: increase if O2 is low
            o2_error = max(0, 20.0 - o2_pressure)  # Target 20 kPa
            action[idx] = np.clip(0.5 + self.control_gains['o2_gain'] * o2_error, 0.0, 1.0)
            idx += 1
        
        if idx < self.action_dim:
            # CO2 scrubbing: increase if CO2 is high
            co2_error = max(0, co2_pressure - 0.5)  # Target max 0.5 kPa
            action[idx] = np.clip(0.5 + self.control_gains['co2_gain'] * co2_error, 0.0, 1.0)
            idx += 1
        
        if idx < self.action_dim:
            # N2 injection: minimal unless pressure is low
            pressure_error = max(0, 95.0 - total_pressure)
            action[idx] = np.clip(pressure_error * 0.01, 0.0, 1.0)
            idx += 1
        
        if idx < self.action_dim:
            # Air circulation: higher if air quality is poor
            circulation_need = max(0.3, 1.0 - air_quality)
            action[idx] = np.clip(circulation_need, 0.0, 1.0)
            idx += 1
        
        if idx < self.action_dim:
            # Humidity control: target 45%
            humidity_error = abs(humidity - 45.0) / 45.0
            action[idx] = np.clip(0.4 + humidity_error * 0.2, 0.0, 1.0)
            idx += 1
        
        if idx < self.action_dim:
            # Air filter mode: higher if air quality is poor
            filter_mode = 1.0 if air_quality < 0.9 else 0.3
            action[idx] = filter_mode
            idx += 1
        
        # Power management
        if idx < self.action_dim:
            # Battery charging: charge more if battery is low and solar is available
            if battery_charge < 50.0 and solar_gen > 2.0:
                action[idx] = 0.8
            elif battery_charge > 90.0:
                action[idx] = 0.2
            else:
                action[idx] = 0.5
            idx += 1
        
        # Load shedding (4 zones) - shed load if battery is critically low
        for _ in range(min(4, self.action_dim - idx)):
            if battery_charge < 20.0:
                action[idx] = 0.0  # Shed non-essential loads
            else:
                action[idx] = 1.0  # Keep all loads on
            idx += 1
        
        # Remaining actions (thermal, water, etc.) - use moderate values
        while idx < self.action_dim:
            if idx == self.action_dim - 5:  # Assume solar panel angle
                action[idx] = 0.0  # 0 degrees (optimal for equatorial)
            else:
                # Thermal control based on temperature
                if temperature < 20.0:
                    action[idx] = 0.7  # More heating
                elif temperature > 25.0:
                    action[idx] = 0.3  # Less heating/more cooling
                else:
                    action[idx] = 0.5  # Moderate control
            idx += 1
        
        return action
    
    def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Heuristic agent doesn't train - just collect statistics."""
        logger.info(f"Running heuristic agent for {total_timesteps} timesteps")
        
        stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_o2_pressure': [],
            'avg_co2_pressure': [],
            'avg_temperature': []
        }
        
        timesteps = 0
        episodes = 0
        
        while timesteps < total_timesteps:
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_o2 = []
            episode_co2 = []
            episode_temp = []
            done = False
            truncated = False
            
            while not (done or truncated) and timesteps < total_timesteps:
                action = self.predict(obs)
                obs, reward, done, truncated, step_info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                timesteps += 1
                
                # Track key variables
                if len(obs) >= 6:
                    episode_o2.append(obs[0])
                    episode_co2.append(obs[1])
                    episode_temp.append(obs[5])
            
            episodes += 1
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
            
            if episode_o2:
                stats['avg_o2_pressure'].append(np.mean(episode_o2))
                stats['avg_co2_pressure'].append(np.mean(episode_co2))
                stats['avg_temperature'].append(np.mean(episode_temp))
            
            # Log progress
            if episodes % 10 == 0:
                avg_reward = np.mean(stats['episode_rewards'][-10:])
                logger.info(f"Episode {episodes}, Avg Reward: {avg_reward:.2f}")
        
        stats['total_timesteps'] = timesteps
        stats['episodes'] = episodes
        
        if stats['episode_rewards']:
            stats['mean_reward'] = np.mean(stats['episode_rewards'])
            stats['std_reward'] = np.std(stats['episode_rewards'])
            stats['mean_length'] = np.mean(stats['episode_lengths'])
        
        return stats
    
    def save(self, filepath: str):
        """Save heuristic agent parameters."""
        np.savez(filepath, 
                control_gains=self.control_gains,
                action_dim=self.action_dim)
        logger.info(f"Saved heuristic agent to {filepath}")
    
    def load(self, filepath: str):
        """Load heuristic agent parameters."""
        data = np.load(filepath, allow_pickle=True)
        if 'control_gains' in data:
            self.control_gains = data['control_gains'].item()
        if 'action_dim' in data:
            self.action_dim = data['action_dim'].item()
        logger.info(f"Loaded heuristic agent from {filepath}")


class ActorNetwork(nn.Module):
    """Actor network for continuous action spaces."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2)  # mean and log_std
        )
        self.action_dim = action_dim
        self.max_log_std = 2.0
        self.min_log_std = -20.0
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log std."""
        out = self.network(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-value."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, obs_dim: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value."""
        return self.network(state)


class PPOAgent(nn.Module):
    """PPO agent implementation compatible with training infrastructure."""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_size)
        self.critic = ValueNetwork(obs_dim, hidden_size)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Training data storage
        self.trajectories = []
        self.episode_rewards = deque(maxlen=100)
        
        logger.info(f"Initialized PPO agent with {sum(p.numel() for p in self.parameters())} parameters")
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action given observation."""
        with torch.no_grad():
            action, _ = self.actor.sample(obs)
        return action
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation (BaseAgent interface)."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(obs_tensor)
        
        return action.cpu().numpy().flatten()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update PPO agent with trajectory data."""
        # Simplified update for basic functionality
        # In practice, this would use proper PPO update with GAE
        return {"actor_loss": 0.0, "critic_loss": 0.0}
    
    def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train PPO agent (BaseAgent interface)."""
        return {"total_timesteps": total_timesteps}
    
    def save(self, filepath: str):
        """Save PPO model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, filepath)
        logger.info(f"Saved PPO model to {filepath}")
    
    def load(self, filepath: str):
        """Load PPO model."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        logger.info(f"Loaded PPO model from {filepath}")


class SACAgent(nn.Module):
    """SAC agent implementation compatible with training infrastructure."""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, hidden_size: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_size)
        self.q1 = CriticNetwork(obs_dim, action_dim, hidden_size)
        self.q2 = CriticNetwork(obs_dim, action_dim, hidden_size)
        self.q1_target = CriticNetwork(obs_dim, action_dim, hidden_size)
        self.q2_target = CriticNetwork(obs_dim, action_dim, hidden_size)
        
        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        
        # SAC hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter
        
        logger.info(f"Initialized SAC agent with {sum(p.numel() for p in self.parameters())} parameters")
    
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action given observation."""
        with torch.no_grad():
            action, _ = self.actor.sample(obs)
        return action
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given observation (BaseAgent interface)."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(obs_tensor)
        
        return action.cpu().numpy().flatten()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update SAC agent with a batch of data."""
        states = batch["obs"]
        actions = batch["actions"] 
        rewards = batch["rewards"]
        next_states = batch["next_obs"]
        dones = batch["dones"]
        
        # Simplified SAC update for basic functionality
        # In practice, this would implement full SAC algorithm
        return {"q1_loss": 0.0, "q2_loss": 0.0, "actor_loss": 0.0}
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """Train SAC agent (BaseAgent interface)."""
        return {"total_timesteps": total_timesteps}
    
    def save(self, filepath: str):
        """Save SAC model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
        }, filepath)
        logger.info(f"Saved SAC model to {filepath}")
    
    def load(self, filepath: str):
        """Load SAC model."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        logger.info(f"Loaded SAC model from {filepath}")


# Legacy Stable Baselines3 wrapper classes
if SB3_AVAILABLE:
    class PPOAgentSB3(BaseAgent):
        """PPO agent using Stable Baselines3."""
        
        def __init__(self, env: Any, policy: str = "MlpPolicy", **kwargs):
            """Initialize PPO agent."""
            self.env = env
            self.model = PPO(policy, env, verbose=1, **kwargs)
            logger.info("Initialized PPO agent")
        
        def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
            """Predict action using PPO policy."""
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action
        
        def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
            """Train PPO agent."""
            logger.info(f"Training PPO agent for {total_timesteps} timesteps")
            self.model.learn(total_timesteps=total_timesteps, **kwargs)
            
            return {
                'total_timesteps': total_timesteps,
                'algorithm': 'PPO'
            }
        
        def save(self, filepath: str):
            """Save PPO model."""
            self.model.save(filepath)
            logger.info(f"Saved PPO agent to {filepath}")
        
        def load(self, filepath: str):
            """Load PPO model."""
            self.model = PPO.load(filepath, env=self.env)
            logger.info(f"Loaded PPO agent from {filepath}")
    
    
    class SACAgentSB3(BaseAgent):
        """SAC agent using Stable Baselines3."""
        
        def __init__(self, env: Any, policy: str = "MlpPolicy", **kwargs):
            """Initialize SAC agent."""
            self.env = env
            self.model = SAC(policy, env, verbose=1, **kwargs)
            logger.info("Initialized SAC agent")
        
        def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
            """Predict action using SAC policy."""
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return action
        
        def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
            """Train SAC agent.""" 
            logger.info(f"Training SAC agent for {total_timesteps} timesteps")
            self.model.learn(total_timesteps=total_timesteps, **kwargs)
            
            return {
                'total_timesteps': total_timesteps,
                'algorithm': 'SAC'
            }
        
        def save(self, filepath: str):
            """Save SAC model."""
            self.model.save(filepath)
            logger.info(f"Saved SAC agent to {filepath}")
        
        def load(self, filepath: str):
            """Load SAC model."""
            self.model = SAC.load(filepath, env=self.env)
            logger.info(f"Loaded SAC agent from {filepath}")

else:
    # Placeholder classes when SB3 is not available
    class PPOAgentSB3Fallback(BaseAgent):
        def __init__(self, *args, **kwargs):
            raise ImportError("Stable Baselines3 not available. Install with: pip install stable-baselines3")
        
        def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
            raise NotImplementedError
        
        def train(self, env: Any, total_timesteps: int, **kwargs) -> Dict[str, Any]:
            raise NotImplementedError
        
        def save(self, filepath: str):
            raise NotImplementedError
        
        def load(self, filepath: str):
            raise NotImplementedError
    
