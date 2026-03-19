"""Tests for SafetyMonitor."""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from lunar_habitat_rl.safety import SafetyMonitor


class TestSafetyMonitor:
    def setup_method(self):
        self.monitor = SafetyMonitor(log_alerts=True)

    def _nominal_state(self) -> np.ndarray:
        return np.array([0.60, 0.65, 0.50, 0.85, 0.80, 0.80], dtype=np.float32)

    def _critical_state(self) -> np.ndarray:
        # o2=0.10 (critical low), pwr=0.08 (critical low), tmp=0.15 (critical low)
        return np.array([0.10, 0.08, 0.15, 0.50, 0.20, 0.10], dtype=np.float32)

    def test_no_alert_in_nominal_state(self):
        alerts = self.monitor.check(self._nominal_state())
        assert len(alerts) == 0, "Nominal state should produce no alerts"

    def test_alerts_in_critical_state(self):
        alerts = self.monitor.check(self._critical_state())
        assert len(alerts) > 0, "Critical state must produce alerts"

    def test_no_override_for_nominal(self):
        state  = self._nominal_state()
        action = np.zeros(4, dtype=np.float32)
        safe_action, overridden = self.monitor.check_and_override(state, action)
        assert not overridden
        np.testing.assert_array_equal(safe_action, action)

    def test_override_in_critical_state(self):
        state  = self._critical_state()
        action = np.zeros(4, dtype=np.float32)
        safe_action, overridden = self.monitor.check_and_override(state, action)
        assert overridden, "Should override in critical state"

    def test_o2_emergency_forces_o2_action(self):
        state = self._nominal_state()
        state[0] = 0.08   # o2 critical low
        action = np.zeros(4, dtype=np.float32)
        safe_action, overridden = self.monitor.check_and_override(state, action)
        assert safe_action[1] == 1.0, "O2 emergency must set regulate_o2=1"

    def test_temp_low_emergency_forces_heating(self):
        state = self._nominal_state()
        state[2] = 0.10   # temp critical low
        action = np.zeros(4, dtype=np.float32)
        safe_action, _ = self.monitor.check_and_override(state, action)
        assert safe_action[2] == 1.0, "Low temp must set control_temp=1"

    def test_temp_high_emergency_forces_cooling(self):
        state = self._nominal_state()
        state[2] = 0.90   # temp critical high
        action = np.zeros(4, dtype=np.float32)
        safe_action, _ = self.monitor.check_and_override(state, action)
        assert safe_action[2] == -1.0, "High temp must set control_temp=-1"

    def test_override_produces_valid_actions(self):
        """Override actions must always be in [-1, 1]."""
        state = self._critical_state()
        action = np.random.uniform(-1, 1, size=4).astype(np.float32)
        safe_action, _ = self.monitor.check_and_override(state, action)
        assert np.all(safe_action >= -1.0) and np.all(safe_action <= 1.0)

    def test_alert_history_logged(self):
        self.monitor.check(self._critical_state())
        assert len(self.monitor.alert_history) > 0

    def test_status_report_is_string(self):
        self.monitor.check(self._critical_state())
        report = self.monitor.status_report()
        assert isinstance(report, str)
        assert "SAFETY MONITOR" in report
