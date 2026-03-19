"""Tests for LunarHabitatEnv."""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from lunar_habitat_rl.environment import LunarHabitatEnv, CRITICAL


class TestLunarHabitatEnv:
    def setup_method(self):
        self.env = LunarHabitatEnv(max_steps=50, seed=0)

    def test_reset_returns_valid_state(self):
        obs, info = self.env.reset()
        assert obs.shape == (6,), "State must be 6-dimensional"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "State must be in [0, 1]"
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        self.env.reset()
        action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_clipping(self):
        """Actions outside [-1, 1] should be clipped without crashing."""
        self.env.reset()
        action = np.array([5.0, -5.0, 100.0, -100.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_episode_terminates_on_crew_death(self):
        """Environment must terminate when crew_health reaches 0."""
        env = LunarHabitatEnv(max_steps=10000, seed=1)
        obs, _ = env.reset()
        # Force crew health to 0
        env._state[3] = 0.001
        env._state[0] = 0.0   # no oxygen
        done = False
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(np.zeros(4))
            if terminated:
                done = True
                break
        assert done, "Environment must terminate when crew dies"

    def test_truncation_at_max_steps(self):
        obs, _ = self.env.reset()
        done = False
        for _ in range(51):  # max_steps=50
            obs, _, terminated, truncated, _ = self.env.step(np.zeros(4))
            if terminated or truncated:
                done = True
                break
        assert done, "Episode must end by max_steps"

    def test_state_stays_in_bounds(self):
        """State should always be clipped to [0, 1] after any action."""
        self.env.reset()
        for _ in range(50):
            action = np.random.uniform(-1, 1, size=4).astype(np.float32)
            obs, _, terminated, truncated, _ = self.env.step(action)
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
                f"State out of bounds: {obs}"
            if terminated or truncated:
                break

    def test_render_does_not_crash(self):
        self.env.reset()
        self.env.step(np.zeros(4))
        self.env.render()  # should print without raising

    def test_survival_status_labels(self):
        env = LunarHabitatEnv(seed=0)
        env.reset()
        env._state[3] = 0.9
        assert env.get_survival_status() == "NOMINAL"
        env._state[3] = 0.5
        assert env.get_survival_status() == "DEGRADED"
        env._state[3] = 0.2
        assert env.get_survival_status() == "CRITICAL"
        env._state[3] = 0.0
        assert env.get_survival_status() == "FAILED"

    def test_multiple_resets_are_independent(self):
        obs1, _ = self.env.reset(seed=1)
        obs2, _ = self.env.reset(seed=2)
        # Different seeds should (usually) produce different starting states
        # This is probabilistic but seeds make it deterministic
        assert not np.array_equal(obs1, obs2), \
            "Different seeds should yield different initial states"
