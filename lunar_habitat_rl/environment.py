"""
LunarHabitatEnv — RL environment for autonomous lunar base management.

State space (6 continuous dims, all in [0, 1] unless noted):
  oxygen_level        – crew breathable O2 fraction (0=none, 1=full)
  power_level         – available power fraction
  temperature         – internal temperature (0=deadly cold, 1=deadly hot, 0.5=nominal)
  crew_health         – aggregate crew health (0=all dead, 1=all healthy)
  solar_panel_status  – solar array integrity (0=destroyed, 1=pristine)
  equipment_integrity – general equipment health (0=failed, 1=nominal)

Action space (4 continuous dims, each in [-1, 1]):
  adjust_power   – increase (+) / decrease (-) power allocation
  regulate_o2    – pump more O2 (+) or vent (-) (costs power)
  control_temp   – heat (+) or cool (-) (costs power)
  repair_equipment – allocate repair resources (+) (costs power/time)

Hazards (stochastic):
  Solar flares    – sudden power spike that damages panels & electronics
  Micrometeorite  – equipment damage
  Dust storms     – solar panel degradation

Reward:
  r = crew_health + resource_efficiency - emergency_penalty
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


# ─── Thresholds ────────────────────────────────────────────────────────────────

CRITICAL = {
    "oxygen_level":       (0.15, 0.95),   # (low_critical, high_critical)
    "power_level":        (0.10, 0.98),
    "temperature":        (0.20, 0.80),
    "crew_health":        (0.05, 1.01),   # no upper critical for health
    "solar_panel_status": (0.05, 1.01),
    "equipment_integrity":(0.05, 1.01),
}

NOMINAL = {
    "oxygen_level":       (0.30, 0.75),
    "power_level":        (0.25, 0.85),
    "temperature":        (0.35, 0.65),
}

# Hazard probabilities per step
HAZARD_PROBS = {
    "solar_flare":       0.004,   # ~1 per 250 steps
    "micrometeorite":    0.006,
    "dust_storm_start":  0.002,
    "dust_storm_end":    0.06,    # per step while active
}


@dataclass
class HazardEvent:
    name: str
    magnitude: float
    description: str


class LunarHabitatEnv:
    """Gym-compatible RL environment for autonomous lunar habitat management."""

    STATE_DIM = 6
    ACTION_DIM = 4

    STATE_KEYS = [
        "oxygen_level",
        "power_level",
        "temperature",
        "crew_health",
        "solar_panel_status",
        "equipment_integrity",
    ]

    def __init__(
        self,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Spaces (informal — numpy arrays)
        self.observation_space_shape = (self.STATE_DIM,)
        self.action_space_shape = (self.ACTION_DIM,)

        self._state: np.ndarray = np.zeros(self.STATE_DIM, dtype=np.float32)
        self._step = 0
        self._dust_storm_active = False
        self._emergency_events = 0
        self._total_reward = 0.0
        self._hazard_log: list = []

    # ─── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step = 0
        self._dust_storm_active = False
        self._emergency_events = 0
        self._total_reward = 0.0
        self._hazard_log = []

        # Start near-nominal with small noise
        self._state = np.array([
            self.rng.uniform(0.55, 0.75),  # oxygen
            self.rng.uniform(0.55, 0.80),  # power
            self.rng.uniform(0.40, 0.60),  # temperature
            self.rng.uniform(0.80, 1.00),  # crew_health
            self.rng.uniform(0.75, 1.00),  # solar_panels
            self.rng.uniform(0.75, 1.00),  # equipment
        ], dtype=np.float32)

        return self._state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._step += 1

        # Apply actions
        self._apply_action(action)

        # Simulate hazards
        hazards = self._simulate_hazards()

        # Physics / passive decay
        self._passive_dynamics()

        # Clip state to [0, 1]
        self._state = np.clip(self._state, 0.0, 1.0)

        # Compute reward
        reward = self._compute_reward()
        self._total_reward += reward

        # Check termination
        terminated = self._is_terminal()
        truncated = self._step >= self.max_steps

        info = {
            "step": self._step,
            "hazards": [h.name for h in hazards],
            "emergency_events": self._emergency_events,
            "state": self._get_state_dict(),
            "dust_storm": self._dust_storm_active,
        }
        return self._state.copy(), reward, terminated, truncated, info

    # ─── Internal mechanics ────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray):
        """Apply the 4-dim action vector to habitat state."""
        adjust_power, regulate_o2, control_temp, repair = action

        o2, pwr, temp, health, panels, equip = self._state

        # Power management
        pwr += adjust_power * 0.05
        # O2 regulation costs power
        o2 += regulate_o2 * 0.04
        pwr -= abs(regulate_o2) * 0.02
        # Thermal control costs power
        temp += control_temp * 0.03
        pwr -= abs(control_temp) * 0.015
        # Repair costs power but improves equipment
        equip += max(0, repair) * 0.06
        pwr -= max(0, repair) * 0.025

        self._state[:] = [o2, pwr, temp, health, panels, equip]

    def _passive_dynamics(self):
        """Natural decay / consumption each step."""
        o2, pwr, temp, health, panels, equip = self._state

        # O2 consumed by crew
        o2 -= 0.0015
        # Power from solar panels (degraded by dust/damage)
        pwr += panels * 0.018 - 0.008
        # Temperature drifts toward cold (lunar night effect)
        temp -= 0.0008
        # Crew health degrades when resources are critical
        health -= self._health_decay(o2, pwr, temp, equip)
        # Slow equipment wear
        equip -= 0.0003

        self._state[:] = [o2, pwr, temp, health, panels, equip]

    def _health_decay(self, o2, pwr, temp, equip) -> float:
        """Extra health penalty when life-support is failing."""
        decay = 0.0
        if o2 < CRITICAL["oxygen_level"][0]:
            decay += 0.015  # suffocation
        if pwr < CRITICAL["power_level"][0]:
            decay += 0.005  # life support failing
        if temp < CRITICAL["temperature"][0] or temp > CRITICAL["temperature"][1]:
            decay += 0.010  # hypothermia / hyperthermia
        if equip < CRITICAL["equipment_integrity"][0]:
            decay += 0.003
        return decay

    def _simulate_hazards(self) -> list:
        """Randomly trigger hazard events, mutate state, return list of events."""
        events = []

        # Solar flare
        if self.rng.random() < HAZARD_PROBS["solar_flare"]:
            magnitude = self.rng.uniform(0.1, 0.35)
            self._state[1] += magnitude * 0.5    # power spike (can overload)
            self._state[4] -= magnitude * 0.3    # panel damage from radiation
            self._state[5] -= magnitude * 0.2    # electronics damage
            self._emergency_events += 1
            evt = HazardEvent("solar_flare", magnitude,
                              f"Solar flare! Magnitude {magnitude:.2f}")
            events.append(evt)
            self._hazard_log.append(evt)

        # Micrometeorite impact
        if self.rng.random() < HAZARD_PROBS["micrometeorite"]:
            magnitude = self.rng.uniform(0.05, 0.20)
            self._state[5] -= magnitude          # equipment damage
            self._state[3] -= magnitude * 0.1    # crew injury risk
            self._emergency_events += 1
            evt = HazardEvent("micrometeorite", magnitude,
                              f"Micrometeorite impact! Damage {magnitude:.2f}")
            events.append(evt)
            self._hazard_log.append(evt)

        # Dust storm
        if self._dust_storm_active:
            self._state[4] -= 0.003              # panel degradation
            if self.rng.random() < HAZARD_PROBS["dust_storm_end"]:
                self._dust_storm_active = False
        else:
            if self.rng.random() < HAZARD_PROBS["dust_storm_start"]:
                self._dust_storm_active = True
                self._emergency_events += 1
                evt = HazardEvent("dust_storm", 0.0, "Dust storm commenced!")
                events.append(evt)
                self._hazard_log.append(evt)

        return events

    def _compute_reward(self) -> float:
        """r = crew_health + resource_efficiency - emergency_penalty."""
        o2, pwr, temp, health, panels, equip = self._state

        # Crew survival is the primary objective
        crew_reward = health

        # Resource efficiency: reward being in nominal range
        o2_eff   = 1.0 - abs(o2   - 0.55) / 0.45
        pwr_eff  = 1.0 - abs(pwr  - 0.55) / 0.45
        temp_eff = 1.0 - abs(temp - 0.50) / 0.50
        resource_efficiency = (o2_eff + pwr_eff + temp_eff) / 3.0 * 0.5

        # Emergency events per step penalty
        emergency_penalty = self._emergency_events * 0.02

        return float(crew_reward + resource_efficiency - emergency_penalty)

    def _is_terminal(self) -> bool:
        """Episode ends if crew dies."""
        return bool(self._state[3] <= 0.0)  # crew_health

    def _get_state_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.STATE_KEYS, self._state)}

    # ─── Utilities ─────────────────────────────────────────────────────────────

    def get_survival_status(self) -> str:
        health = self._state[3]
        if health > 0.7:
            return "NOMINAL"
        elif health > 0.4:
            return "DEGRADED"
        elif health > 0.1:
            return "CRITICAL"
        return "FAILED"

    def render(self):
        d = self._get_state_dict()
        storm = " 🌪️ DUST STORM" if self._dust_storm_active else ""
        print(
            f"Step {self._step:4d} | "
            f"O2={d['oxygen_level']:.2f} PWR={d['power_level']:.2f} "
            f"TMP={d['temperature']:.2f} HP={d['crew_health']:.2f} "
            f"SOL={d['solar_panel_status']:.2f} EQP={d['equipment_integrity']:.2f} "
            f"| {self.get_survival_status()}{storm}"
        )
