"""
SafetyMonitor — real-time hazard detection and emergency protocol engine.

Monitors all habitat state parameters and overrides agent actions when any
parameter crosses a critical threshold. This is a hard safety layer on top
of the learned policy.

Emergency protocols:
  - O2 CRITICAL LOW  → force maximum O2 regulation
  - POWER CRITICAL   → shed non-essential loads, maximize solar harvest
  - TEMPERATURE OOB  → force thermal correction
  - EQUIPMENT FAIL   → force repair allocation
  - CREW CRITICAL    → all-hands emergency, optimize for health
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class EmergencyAlert:
    parameter: str
    value: float
    threshold: float
    direction: str   # "low" | "high"
    protocol: str


# Critical thresholds (same as environment.py CRITICAL dict)
CRITICAL_LOW = {
    "oxygen_level":       0.15,
    "power_level":        0.10,
    "temperature":        0.20,
    "crew_health":        0.10,
    "solar_panel_status": 0.05,
    "equipment_integrity":0.05,
}

CRITICAL_HIGH = {
    "temperature": 0.80,
    "power_level": 0.95,   # overload risk
}

STATE_KEYS = [
    "oxygen_level",
    "power_level",
    "temperature",
    "crew_health",
    "solar_panel_status",
    "equipment_integrity",
]

# Action indices
ACTION_POWER   = 0
ACTION_O2      = 1
ACTION_TEMP    = 2
ACTION_REPAIR  = 3


class SafetyMonitor:
    """
    Hard-coded safety layer that overrides policy actions during emergencies.

    The monitor observes the current state vector and produces a modified
    action that prioritizes life safety over policy objectives.
    """

    def __init__(self, log_alerts: bool = True):
        self.log_alerts = log_alerts
        self.alert_history: List[EmergencyAlert] = []
        self.override_count = 0

    def check(self, state: np.ndarray) -> List[EmergencyAlert]:
        """
        Inspect state and return a list of active emergency alerts.

        Args:
            state: length-6 state vector from LunarHabitatEnv

        Returns:
            List of EmergencyAlert objects (empty = all clear)
        """
        alerts = []
        for i, key in enumerate(STATE_KEYS):
            val = float(state[i])

            # Check low threshold
            if key in CRITICAL_LOW and val < CRITICAL_LOW[key]:
                alerts.append(EmergencyAlert(
                    parameter=key,
                    value=val,
                    threshold=CRITICAL_LOW[key],
                    direction="low",
                    protocol=self._protocol_low(key),
                ))

            # Check high threshold
            if key in CRITICAL_HIGH and val > CRITICAL_HIGH[key]:
                alerts.append(EmergencyAlert(
                    parameter=key,
                    value=val,
                    threshold=CRITICAL_HIGH[key],
                    direction="high",
                    protocol=self._protocol_high(key),
                ))

        if self.log_alerts:
            self.alert_history.extend(alerts)

        return alerts

    def check_and_override(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Check state for emergencies; if any detected, override action.

        Args:
            state:  current habitat state (length 6)
            action: proposed agent action (length 4)

        Returns:
            (safe_action, was_overridden)
        """
        alerts = self.check(state)
        if not alerts:
            return action.copy(), False

        self.override_count += 1
        safe_action = action.copy()

        for alert in alerts:
            safe_action = self._apply_emergency_protocol(safe_action, alert, state)

        return np.clip(safe_action, -1.0, 1.0), True

    def status_report(self) -> str:
        """Human-readable status summary."""
        lines = [
            "╔══════════════════════════════════════╗",
            "║       SAFETY MONITOR STATUS          ║",
            f"║  Total overrides: {self.override_count:<18}║",
            f"║  Total alerts:    {len(self.alert_history):<18}║",
            "╚══════════════════════════════════════╝",
        ]
        if self.alert_history:
            lines.append("\nRecent alerts (last 5):")
            for a in self.alert_history[-5:]:
                lines.append(
                    f"  ⚠  {a.parameter} = {a.value:.3f} "
                    f"({'<' if a.direction == 'low' else '>'} {a.threshold:.3f}) "
                    f"→ {a.protocol}"
                )
        return "\n".join(lines)

    # ─── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _protocol_low(key: str) -> str:
        return {
            "oxygen_level":        "EMERGENCY O2 BOOST",
            "power_level":         "EMERGENCY POWER CONSERVATION",
            "temperature":         "EMERGENCY HEATING",
            "crew_health":         "CREW EMERGENCY — ALL SYSTEMS OPTIMAL",
            "solar_panel_status":  "SOLAR PANEL REPAIR PRIORITY",
            "equipment_integrity": "EMERGENCY EQUIPMENT REPAIR",
        }.get(key, "GENERIC EMERGENCY")

    @staticmethod
    def _protocol_high(key: str) -> str:
        return {
            "temperature": "EMERGENCY COOLING",
            "power_level": "EMERGENCY LOAD SHEDDING",
        }.get(key, "GENERIC HIGH ALERT")

    def _apply_emergency_protocol(
        self,
        action: np.ndarray,
        alert: EmergencyAlert,
        state: np.ndarray,
    ) -> np.ndarray:
        """Modify action to execute the required emergency protocol."""
        key = alert.parameter
        direction = alert.direction

        if key == "oxygen_level" and direction == "low":
            # Max O2 regulation
            action[ACTION_O2] = 1.0
            # Ensure we have power to do it
            if state[1] < 0.3:
                action[ACTION_POWER] = 0.5

        elif key == "power_level" and direction == "low":
            # Conservative power — reduce non-essential usage
            action[ACTION_POWER] = 0.8
            action[ACTION_O2]    = min(action[ACTION_O2], 0.0)
            action[ACTION_TEMP]  = min(abs(action[ACTION_TEMP]), 0.2) * np.sign(action[ACTION_TEMP] + 1e-9)

        elif key == "power_level" and direction == "high":
            # Shed load to prevent overload
            action[ACTION_POWER] = -0.8

        elif key == "temperature" and direction == "low":
            action[ACTION_TEMP] = 1.0   # max heat

        elif key == "temperature" and direction == "high":
            action[ACTION_TEMP] = -1.0  # max cooling

        elif key == "equipment_integrity" and direction == "low":
            action[ACTION_REPAIR] = 1.0

        elif key == "solar_panel_status" and direction == "low":
            action[ACTION_REPAIR] = max(action[ACTION_REPAIR], 0.8)

        elif key == "crew_health" and direction == "low":
            # All-hands emergency: fix everything simultaneously
            action[ACTION_O2]    = 1.0
            action[ACTION_POWER] = 0.5
            action[ACTION_REPAIR]= 0.8
            # Hold temperature at safe midpoint
            temp = float(state[2])
            if temp < 0.35:
                action[ACTION_TEMP] = 1.0
            elif temp > 0.65:
                action[ACTION_TEMP] = -1.0
            else:
                action[ACTION_TEMP] = 0.0

        return action
