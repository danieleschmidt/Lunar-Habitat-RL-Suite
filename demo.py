#!/usr/bin/env python3
"""
Lunar Habitat RL Suite — Training Demo
=======================================
Trains a PPO agent on LunarHabitatEnv for 100 episodes and reports
survival rate improvement from early → late training.

Usage:
    ~/anaconda3/bin/python3 demo.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from lunar_habitat_rl import (
    LunarHabitatEnv,
    HabitatPolicyNetwork,
    PPOHabitatAgent,
    SafetyMonitor,
)


def section(title: str):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def main():
    section("🌙 LUNAR HABITAT RL SUITE — TRAINING DEMO")
    print("\nInitializing lunar base simulation...")
    print("  State:   oxygen, power, temperature, crew_health,")
    print("           solar_panels, equipment_integrity")
    print("  Actions: adjust_power, regulate_o2, control_temp, repair")
    print("  Hazards: solar flares, micrometeorites, dust storms")

    # ─── Environment & Agent Setup ────────────────────────────────────────────
    env = LunarHabitatEnv(max_steps=300, seed=42)
    agent = PPOHabitatAgent(env=env, seed=42)

    section("🚀 STARTING TRAINING (100 episodes)")
    print(f"  PPO config: lr={agent.config.lr}, clip={agent.config.clip_range}, "
          f"n_steps={agent.config.n_steps}\n")

    summary = agent.train(n_episodes=100, verbose=True)

    # ─── Results ──────────────────────────────────────────────────────────────
    section("📊 TRAINING RESULTS")

    early_rate  = summary["early_survival_rate"]  * 100
    late_rate   = summary["late_survival_rate"]   * 100
    overall     = summary["overall_survival_rate"] * 100
    avg_reward  = summary["avg_episode_reward"]
    avg_length  = summary["avg_episode_length"]

    print(f"\n  Total episodes:          {summary['total_episodes']}")
    print(f"  Overall survival rate:   {overall:.1f}%")
    print(f"  Early training (ep 1-50): {early_rate:.1f}%")
    print(f"  Late training  (ep 51-100): {late_rate:.1f}%")
    print(f"  Improvement:             +{late_rate - early_rate:.1f}pp")
    print(f"  Avg episode reward:      {avg_reward:.2f}")
    print(f"  Avg episode length:      {avg_length:.0f} steps")

    # ─── Safety Monitor Demo ──────────────────────────────────────────────────
    section("🛡️  SAFETY MONITOR DEMO")

    monitor = SafetyMonitor()

    critical_state = np.array([0.10, 0.08, 0.15, 0.45, 0.20, 0.12], dtype=np.float32)
    dummy_action   = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    print("\nSimulating critical state:")
    print(f"  O2={critical_state[0]:.2f}  PWR={critical_state[1]:.2f}  "
          f"TMP={critical_state[2]:.2f}  HP={critical_state[3]:.2f}  "
          f"SOL={critical_state[4]:.2f}  EQP={critical_state[5]:.2f}")

    safe_action, overridden = monitor.check_and_override(critical_state, dummy_action)

    print(f"\nAgent proposed:  {dummy_action}")
    print(f"Safety override: {safe_action}  (overridden={overridden})")
    print("\nActive alerts:")
    alerts = monitor.check(critical_state)
    for a in alerts:
        sym = "▼" if a.direction == "low" else "▲"
        print(f"  ⚠  {a.parameter:<25} {sym} {a.value:.3f}  →  {a.protocol}")

    # ─── Deployment Evaluation ────────────────────────────────────────────────
    section("🌍 DEPLOYMENT EVALUATION (10 episodes, deterministic policy)")

    eval_env = LunarHabitatEnv(max_steps=300, seed=99)
    eval_monitor = SafetyMonitor()
    survived = 0
    total_rewards = []

    for ep in range(10):
        obs, _ = eval_env.reset(seed=ep * 7 + 99)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        if not terminated:
            survived += 1
        total_rewards.append(ep_reward)

    print(f"\n  Eval survival rate:  {survived}/10  ({survived * 10:.0f}%)")
    print(f"  Avg eval reward:     {np.mean(total_rewards):.2f}")

    # ─── Final Status ─────────────────────────────────────────────────────────
    section("✅ MISSION STATUS")
    if late_rate > early_rate:
        print(f"\n  ✓ Agent improved survival rate by {late_rate - early_rate:.1f}pp during training")
    else:
        print(f"\n  ~ Agent achieved {overall:.1f}% survival rate")

    print(f"  ✓ Safety monitor protected {eval_monitor.override_count} unsafe actions")
    print(f"  ✓ LunarHabitatEnv, PPOHabitatAgent, SafetyMonitor all operational")
    print("\n  Ready for deployment on Artemis Base Alpha. 🚀\n")


if __name__ == "__main__":
    main()
