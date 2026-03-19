"""
PPOHabitatAgent — Proximal Policy Optimization for LunarHabitatEnv.

Implements the PPO-Clip algorithm (Schulman et al., 2017) with:
  - Generalized Advantage Estimation (GAE)
  - Value function clipping
  - Entropy bonus for exploration
  - Gradient norm clipping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .environment import LunarHabitatEnv
from .policy import HabitatPolicyNetwork
from .safety import SafetyMonitor


@dataclass
class PPOConfig:
    # Rollout
    n_steps: int = 512          # steps per rollout per update
    n_epochs: int = 5           # PPO update epochs
    batch_size: int = 64

    # PPO hyperparameters
    clip_range: float = 0.2
    gamma: float = 0.995        # longer horizon — survival is the goal
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    ent_coef: float = 0.02      # higher entropy → more exploration
    max_grad_norm: float = 0.5

    # Optimizer
    lr: float = 3e-4

    # Logging
    log_interval: int = 10


@dataclass
class RolloutBuffer:
    states:     List[np.ndarray] = field(default_factory=list)
    actions:    List[np.ndarray] = field(default_factory=list)
    log_probs:  List[float]      = field(default_factory=list)
    rewards:    List[float]      = field(default_factory=list)
    dones:      List[bool]       = field(default_factory=list)
    values:     List[float]      = field(default_factory=list)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.rewards)


class PPOHabitatAgent:
    """PPO agent for autonomous lunar habitat management."""

    def __init__(
        self,
        env: Optional[LunarHabitatEnv] = None,
        config: Optional[PPOConfig] = None,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.env = env or LunarHabitatEnv(seed=seed)
        self.config = config or PPOConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy = HabitatPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()
        self.safety_monitor = SafetyMonitor()

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int]   = []
        self.survival_log: List[bool]     = []  # did crew survive?
        self.training_losses: List[Dict]  = []

    # ─── Public API ────────────────────────────────────────────────────────────

    def train(self, n_episodes: int = 100, verbose: bool = True) -> Dict:
        """Train for n_episodes, return summary metrics."""
        obs, _ = self.env.reset()
        ep_reward = 0.0
        ep_length = 0
        episode = 0

        total_steps = n_episodes * self.env.max_steps  # upper bound
        step = 0

        while episode < n_episodes:
            # Collect rollout
            for _ in range(self.config.n_steps):
                action, log_prob, value = self._select_action(obs)

                # Safety override if needed
                safe_action, override = self.safety_monitor.check_and_override(
                    obs, action
                )

                next_obs, reward, terminated, truncated, info = self.env.step(
                    safe_action
                )
                done = terminated or truncated

                self.buffer.states.append(obs.copy())
                self.buffer.actions.append(safe_action.copy())
                self.buffer.log_probs.append(log_prob)
                self.buffer.rewards.append(reward)
                self.buffer.dones.append(done)
                self.buffer.values.append(value)

                obs = next_obs
                ep_reward += reward
                ep_length += 1
                step += 1

                if done:
                    survived = not terminated  # truncated = time-up (survived)
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    self.survival_log.append(survived)
                    episode += 1

                    if verbose and episode % self.config.log_interval == 0:
                        self._log_progress(episode, n_episodes)

                    obs, _ = self.env.reset()
                    ep_reward = 0.0
                    ep_length = 0

                    if episode >= n_episodes:
                        break

            # PPO update
            if len(self.buffer) > 0:
                loss_info = self._update()
                self.training_losses.append(loss_info)
                self.buffer.clear()

        return self._summary()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action for a given observation (inference mode)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _select_action(self, obs: np.ndarray):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def _compute_gae(
        self, rewards, dones, values, last_value: float
    ) -> tuple:
        """Compute GAE advantages and returns."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        values_arr = np.array(values + [last_value], dtype=np.float32)

        for t in reversed(range(n)):
            delta = (
                rewards[t]
                + self.config.gamma * values_arr[t + 1] * (1 - dones[t])
                - values_arr[t]
            )
            last_gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def _update(self) -> Dict:
        """Run PPO-Clip update on the current rollout buffer."""
        cfg = self.config

        # Get bootstrap value for last state
        last_obs = self.buffer.states[-1] if self.buffer.states else np.zeros(6)
        last_obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.policy(last_obs_t)
        last_value = last_value.item()

        advantages, returns = self._compute_gae(
            self.buffer.rewards,
            [float(d) for d in self.buffer.dones],
            self.buffer.values,
            last_value,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t  = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_lp_t  = torch.FloatTensor(self.buffer.log_probs).unsqueeze(1).to(self.device)
        adv_t     = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        ret_t     = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        old_val_t = torch.FloatTensor(self.buffer.values).unsqueeze(1).to(self.device)

        n = len(states_t)
        indices = np.arange(n)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, cfg.batch_size):
                batch = indices[start: start + cfg.batch_size]
                new_lp, new_val, entropy = self.policy.evaluate_actions(
                    states_t[batch], actions_t[batch]
                )

                # Policy (actor) loss
                ratio = (new_lp - old_lp_t[batch]).exp()
                adv_b = adv_t[batch]
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping
                val_clipped = old_val_t[batch] + (new_val - old_val_t[batch]).clamp(
                    -cfg.clip_range, cfg.clip_range
                )
                vf_loss1 = (new_val - ret_t[batch]).pow(2)
                vf_loss2 = (val_clipped - ret_t[batch]).pow(2)
                value_loss = torch.max(vf_loss1, vf_loss2).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    + cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += (-entropy_loss).item()
                n_updates += 1

        k = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss":  total_value_loss  / k,
            "entropy":     total_entropy     / k,
        }

    def _log_progress(self, episode: int, total: int):
        recent = min(10, len(self.episode_rewards))
        avg_r = np.mean(self.episode_rewards[-recent:])
        survival_rate = (
            sum(self.survival_log[-recent:]) / recent * 100
            if recent else 0.0
        )
        print(
            f"  Episode {episode:4d}/{total} | "
            f"Avg Reward: {avg_r:+7.2f} | "
            f"Survival Rate (last {recent}): {survival_rate:.0f}%"
        )

    def _summary(self) -> Dict:
        n = len(self.survival_log)
        first_half = self.survival_log[:n // 2] if n > 1 else []
        second_half = self.survival_log[n // 2:] if n > 1 else self.survival_log
        return {
            "total_episodes":         n,
            "overall_survival_rate":  sum(self.survival_log) / max(n, 1),
            "early_survival_rate":    sum(first_half) / max(len(first_half), 1),
            "late_survival_rate":     sum(second_half) / max(len(second_half), 1),
            "avg_episode_reward":     float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "avg_episode_length":     float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "total_hazard_events":    sum(
                info.get("emergency_events", 0)
                for info in (self.training_losses or [])
            ),
        }
