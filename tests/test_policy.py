"""Tests for HabitatPolicyNetwork."""

import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, ".")

from lunar_habitat_rl.policy import HabitatPolicyNetwork


class TestHabitatPolicyNetwork:
    def setup_method(self):
        torch.manual_seed(0)
        self.policy = HabitatPolicyNetwork()

    def test_forward_shapes(self):
        batch = 8
        state = torch.randn(batch, 6)
        dist, value = self.policy(state)
        assert dist.mean.shape == (batch, 4), "Action mean must be (B, 4)"
        assert value.shape == (batch, 1), "Value must be (B, 1)"

    def test_action_bounds(self):
        """Sampled actions must be clipped to [-1, 1]."""
        state = torch.randn(32, 6)
        action, log_prob, value = self.policy.get_action(state)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0), \
            "Actions must be in [-1, 1]"

    def test_deterministic_action_is_reproducible(self):
        state = torch.randn(1, 6)
        a1, _, _ = self.policy.get_action(state, deterministic=True)
        a2, _, _ = self.policy.get_action(state, deterministic=True)
        assert torch.allclose(a1, a2), "Deterministic action must be reproducible"

    def test_evaluate_actions_shapes(self):
        batch = 16
        states  = torch.randn(batch, 6)
        actions = torch.zeros(batch, 4)
        log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
        assert log_probs.shape == (batch, 1)
        assert values.shape   == (batch, 1)
        assert entropy.shape  == (batch, 1)

    def test_gradient_flows(self):
        state  = torch.randn(4, 6)
        action = torch.zeros(4, 4)
        log_probs, values, entropy = self.policy.evaluate_actions(state, action)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        for name, param in self.policy.named_parameters():
            assert param.grad is not None, f"Param {name} has no gradient"

    def test_custom_architecture(self):
        policy = HabitatPolicyNetwork(
            state_dim=6, action_dim=4, hidden_dims=(64, 32)
        )
        state = torch.randn(1, 6)
        dist, value = policy(state)
        assert dist.mean.shape == (1, 4)
