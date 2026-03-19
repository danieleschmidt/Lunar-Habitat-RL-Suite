"""
HabitatPolicyNetwork — MLP actor-critic policy for lunar habitat control.

Architecture:
  Shared trunk: Linear(6) → 128 → 128 → 64
  Actor head:   64 → 4  (continuous actions via tanh)
  Critic head:  64 → 1  (state value estimate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


class HabitatPolicyNetwork(nn.Module):
    """Actor-critic MLP for LunarHabitatEnv (state_dim=6, action_dim=4)."""

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Actor: mean of Gaussian action distribution
        self.actor_mean = nn.Linear(in_dim, action_dim)
        # Log-std as learnable parameter (not state-dependent, simple case)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        # Critic: scalar value
        self.critic = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Smaller init for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Returns:
            dist:  Normal distribution over actions (batch)
            value: State-value estimate (batch, 1)
        """
        features = self.trunk(state)
        mean = torch.tanh(self.actor_mean(features))   # actions ∈ (-1, 1)
        std = self.log_std.exp().clamp(1e-4, 2.0)
        dist = Normal(mean, std.expand_as(mean))
        value = self.critic(features)
        return dist, value

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or take greedy) action.

        Returns:
            action:   sampled action, clipped to [-1, 1]
            log_prob: log probability of the action
            value:    state value estimate
        """
        dist, value = self(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under current policy (used in PPO update).

        Returns:
            log_probs:  log probabilities of actions
            values:     state values
            entropy:    policy entropy (for exploration bonus)
        """
        dist, values = self(states)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_probs, values, entropy
