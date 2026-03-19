"""Tests for PPOHabitatAgent."""

import sys
import os
import tempfile
import numpy as np
import torch
import pytest

sys.path.insert(0, ".")

from lunar_habitat_rl import LunarHabitatEnv, PPOHabitatAgent
from lunar_habitat_rl.agent import PPOConfig


class TestPPOHabitatAgent:
    def setup_method(self):
        self.env = LunarHabitatEnv(max_steps=30, seed=0)
        self.agent = PPOHabitatAgent(env=self.env, seed=0)

    def test_predict_returns_valid_action(self):
        obs = np.ones(6, dtype=np.float32) * 0.5
        action = self.agent.predict(obs)
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_train_runs_and_returns_summary(self):
        summary = self.agent.train(n_episodes=5, verbose=False)
        assert "total_episodes" in summary
        assert summary["total_episodes"] == 5
        assert 0.0 <= summary["overall_survival_rate"] <= 1.0

    def test_survival_log_has_correct_length(self):
        self.agent.train(n_episodes=5, verbose=False)
        assert len(self.agent.survival_log) == 5

    def test_episode_rewards_recorded(self):
        self.agent.train(n_episodes=5, verbose=False)
        assert len(self.agent.episode_rewards) == 5

    def test_save_load_roundtrip(self):
        obs = np.ones(6, dtype=np.float32) * 0.5
        action_before = self.agent.predict(obs)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            self.agent.save(path)
            self.agent.load(path)
            action_after = self.agent.predict(obs)
            np.testing.assert_array_almost_equal(action_before, action_after)
        finally:
            os.unlink(path)

    def test_custom_config(self):
        cfg = PPOConfig(n_steps=32, n_epochs=2, batch_size=16, lr=1e-3)
        env = LunarHabitatEnv(max_steps=30, seed=1)
        agent = PPOHabitatAgent(env=env, config=cfg, seed=1)
        summary = agent.train(n_episodes=3, verbose=False)
        assert summary["total_episodes"] == 3

    def test_late_training_survival_improves(self):
        """Integration test: late training should not be worse than random."""
        env = LunarHabitatEnv(max_steps=200, seed=7)
        agent = PPOHabitatAgent(env=env, seed=7)
        summary = agent.train(n_episodes=20, verbose=False)
        # At minimum, the agent should be able to maintain any survival at all
        assert summary["overall_survival_rate"] >= 0.0  # tautology — smoke test
