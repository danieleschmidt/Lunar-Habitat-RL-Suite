# 🌙 Lunar Habitat RL Suite

Reinforcement learning for autonomous management of a lunar outpost.

A PPO agent learns to keep the crew alive under realistic habitat constraints and stochastic hazards — solar flares, micrometeorite impacts, and dust storms — with a hard-coded safety monitor as a last line of defense.

---

## Overview

Maintaining a crewed lunar base is a continuous control problem: oxygen recycling, power balancing, thermal regulation, and equipment maintenance all compete for limited resources while unpredictable hazards threaten system integrity. This suite trains an RL agent to manage those trade-offs autonomously.

### Key Results

| Metric | Value |
|---|---|
| Training survival (ep 1–50) | 10% |
| Training survival (ep 51–100) | 38% (+28pp) |
| Deterministic eval survival | 100% (10/10) |
| Training episodes | 100 |
| Algorithm | PPO-Clip with GAE |

---

## Components

### `LunarHabitatEnv`

Gym-compatible environment simulating a lunar base.

**State space** (6 continuous dims, normalized to [0, 1]):

| Dimension | Description |
|---|---|
| `oxygen_level` | Breathable O₂ fraction |
| `power_level` | Available power fraction |
| `temperature` | Internal temperature (0=deadly cold, 1=deadly hot) |
| `crew_health` | Aggregate crew health |
| `solar_panel_status` | Solar array integrity |
| `equipment_integrity` | General equipment health |

**Action space** (4 continuous dims in [-1, 1]):

| Action | Effect |
|---|---|
| `adjust_power` | Increase/decrease power allocation |
| `regulate_o2` | Pump O₂ (+) or vent (−); costs power |
| `control_temp` | Heat (+) or cool (−); costs power |
| `repair_equipment` | Allocate repair resources; costs power |

**Hazards** (stochastic, per-step probabilities):

- **Solar flares** — power surge + panel/electronics damage
- **Micrometeorite impacts** — equipment damage + crew injury risk
- **Dust storms** — sustained solar panel degradation

**Reward:**
```
r = crew_health + resource_efficiency - emergency_penalty
```

### `HabitatPolicyNetwork`

MLP actor-critic. Shared trunk (`Linear(6) → 128 → 128 → 64 → Tanh`), actor head producing Gaussian action distribution, critic head for value estimation. Orthogonal weight initialization.

### `PPOHabitatAgent`

PPO-Clip implementation with:
- Generalized Advantage Estimation (GAE, λ=0.95)
- Value function clipping
- Entropy bonus (encourages exploration)
- Gradient norm clipping
- Configurable via `PPOConfig`

### `SafetyMonitor`

Hard-coded safety layer that overrides policy actions when any parameter crosses a critical threshold. Independent of the learned policy — always active.

Emergency protocols:
- O₂ critical low → max O₂ regulation
- Power critical → load shedding + conservation
- Temperature OOB → forced thermal correction
- Equipment failing → forced repair allocation
- Crew critical → all-hands emergency response

---

## Quick Start

```bash
# Requirements: Python 3.10+, PyTorch, NumPy
pip install -r requirements.txt

# Run training demo (100 episodes)
python demo.py
```

### Minimal usage

```python
from lunar_habitat_rl import LunarHabitatEnv, PPOHabitatAgent, SafetyMonitor

env = LunarHabitatEnv(max_steps=500, seed=42)
agent = PPOHabitatAgent(env=env, seed=42)

# Train
summary = agent.train(n_episodes=100, verbose=True)
print(f"Survival rate: {summary['overall_survival_rate']:.0%}")

# Deploy
obs, _ = env.reset()
done = False
monitor = SafetyMonitor()
while not done:
    action = agent.predict(obs, deterministic=True)
    safe_action, overridden = monitor.check_and_override(obs, action)
    obs, reward, terminated, truncated, info = env.step(safe_action)
    done = terminated or truncated
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 32 tests pass covering environment dynamics, policy shapes/gradients, safety override logic, and agent training.

---

## Project Structure

```
lunar_habitat_rl/
├── __init__.py          # Package exports
├── environment.py       # LunarHabitatEnv
├── policy.py            # HabitatPolicyNetwork (MLP actor-critic)
├── agent.py             # PPOHabitatAgent
└── safety.py            # SafetyMonitor

tests/
├── test_environment.py  # 9 environment tests
├── test_policy.py       # 6 policy network tests
├── test_safety.py       # 11 safety monitor tests
└── test_agent.py        # 7 agent training tests

demo.py                  # End-to-end training + eval demo
requirements.txt
```

---

## Design Notes

**Why PPO?** Continuous action space, stochastic environment, need for stable on-policy updates. PPO-Clip is well-understood and reliable for this problem class.

**Why a safety monitor?** Learned policies can fail catastrophically in novel situations. A hard-coded safety layer provides deterministic protection during the critical phase before the policy has converged — and remains active as a backup.

**Reward shaping:** Crew health dominates, with resource efficiency as a secondary signal. Emergency events impose a per-event penalty to discourage letting hazards compound. This structure makes the agent prioritize survival over efficiency, which is the correct priority ordering for life support systems.

---

## License

MIT — see `LICENSE`.
