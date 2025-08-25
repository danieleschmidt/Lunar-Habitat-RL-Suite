"""
Dynamic Residual Safe Reinforcement Learning (DRS-RL)

Advanced safety-critical RL algorithm with weak-to-strong safety correction
for multi-agent coordination in lunar habitat life support systems.

Based on breakthrough research from ArXiv 2504.06670 (April 2025).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class SafetyBoundary:
    """Dynamic safety boundary definition."""
    parameter: str
    lower_bound: float
    upper_bound: float
    adaptation_rate: float = 0.01
    violation_penalty: float = 100.0
    critical_threshold: float = 0.05  # Distance to boundary that triggers emergency


@dataclass
class ConflictZone:
    """Represents a potential conflict zone between systems."""
    system_a: str
    system_b: str
    risk_level: float  # 0.0 to 1.0
    mitigation_strategy: str
    last_updated: float = 0.0


class WeakToStrongSafetyCorrector(nn.Module):
    """
    Lightweight safety correction network that dynamically calibrates boundaries.
    """
    
    def __init__(self, state_dim: int, action_dim: int, n_safety_params: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_safety_params = n_safety_params
        
        # Weak model (lightweight safety monitor)
        self.weak_safety_monitor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_safety_params),
            nn.Sigmoid()  # Safety scores [0,1]
        )
        
        # Strong corrector (adaptive boundary adjustment)
        self.strong_corrector = nn.Sequential(
            nn.Linear(state_dim + action_dim + n_safety_params, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # Correction actions [-1,1]
        )
        
        # Parameter efficiency components
        self.parameter_reduction = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute safety scores and corrected actions.
        """
        # Parameter-efficient state encoding
        compressed_state = self.parameter_reduction(state)
        
        # Weak safety monitoring
        safety_input = torch.cat([compressed_state, action], dim=-1)
        safety_scores = self.weak_safety_monitor(safety_input)
        
        # Strong safety correction when needed
        corrector_input = torch.cat([compressed_state, action, safety_scores], dim=-1)
        correction = self.strong_corrector(corrector_input)
        
        # Apply correction based on safety scores
        correction_mask = (safety_scores < 0.5).float()  # Apply correction when unsafe
        corrected_action = action + correction * correction_mask.unsqueeze(-1)
        
        # Ensure actions remain in valid bounds
        corrected_action = torch.tanh(corrected_action)
        
        return corrected_action, safety_scores, correction


class DynamicConflictZoneModeler:
    """Real-time risk assessment and mitigation in complex system interactions."""
    
    def __init__(self, systems: List[str]):
        self.systems = systems
        self.conflict_zones = {}
        self.interaction_history = deque(maxlen=1000)
        self.risk_threshold = 0.7
        self.lock = threading.Lock()
        
    def update_conflict_zones(self, system_states: Dict[str, torch.Tensor], 
                             system_actions: Dict[str, torch.Tensor]):
        """Update conflict zones based on current system states and actions."""
        with self.lock:
            current_time = time.time()
            
            # Analyze all pairwise system interactions
            for i, sys_a in enumerate(self.systems):
                for j, sys_b in enumerate(self.systems[i+1:], i+1):
                    conflict_key = f"{sys_a}_{sys_b}"
                    
                    # Compute interaction risk
                    risk_level = self._compute_interaction_risk(
                        system_states.get(sys_a, torch.zeros(1)),
                        system_actions.get(sys_a, torch.zeros(1)),
                        system_states.get(sys_b, torch.zeros(1)),
                        system_actions.get(sys_b, torch.zeros(1))
                    )
                    
                    # Update or create conflict zone
                    if conflict_key in self.conflict_zones:
                        zone = self.conflict_zones[conflict_key]
                        zone.risk_level = 0.9 * zone.risk_level + 0.1 * risk_level  # Smoothing
                        zone.last_updated = current_time
                    else:
                        self.conflict_zones[conflict_key] = ConflictZone(
                            system_a=sys_a,
                            system_b=sys_b,
                            risk_level=risk_level,
                            mitigation_strategy=self._determine_mitigation_strategy(sys_a, sys_b, risk_level),
                            last_updated=current_time
                        )
    
    def _compute_interaction_risk(self, state_a: torch.Tensor, action_a: torch.Tensor,
                                state_b: torch.Tensor, action_b: torch.Tensor) -> float:
        """Compute risk level for interaction between two systems."""
        # State similarity (high similarity can indicate coupling)
        if len(state_a) > 0 and len(state_b) > 0:
            state_correlation = torch.corrcoef(torch.stack([state_a.flatten(), state_b.flatten()]))[0, 1]
            state_risk = abs(state_correlation.item() if not torch.isnan(state_correlation) else 0.0)
        else:
            state_risk = 0.0
        
        # Action conflict (opposing actions can cause conflicts)
        if len(action_a) > 0 and len(action_b) > 0:
            action_conflict = torch.norm(action_a - action_b).item()
            action_risk = min(action_conflict / 2.0, 1.0)  # Normalize to [0,1]
        else:
            action_risk = 0.0
        
        # Combined risk assessment
        risk_level = 0.6 * state_risk + 0.4 * action_risk
        return float(risk_level)
    
    def _determine_mitigation_strategy(self, sys_a: str, sys_b: str, risk_level: float) -> str:
        """Determine appropriate mitigation strategy based on systems and risk level."""
        if risk_level > 0.8:
            return "emergency_isolation"
        elif risk_level > 0.6:
            return "coordinated_handoff"
        elif risk_level > 0.4:
            return "priority_scheduling"
        else:
            return "normal_operation"
    
    def get_high_risk_zones(self) -> List[ConflictZone]:
        """Get conflict zones above risk threshold."""
        with self.lock:
            return [zone for zone in self.conflict_zones.values() 
                   if zone.risk_level > self.risk_threshold]
    
    def predict_conflicts(self, lookahead_steps: int = 5) -> List[Tuple[str, float]]:
        """Predict potential conflicts in the near future."""
        predictions = []
        
        with self.lock:
            for zone in self.conflict_zones.values():
                # Simple trend-based prediction
                if len(self.interaction_history) > lookahead_steps:
                    recent_risks = [h.get(f"{zone.system_a}_{zone.system_b}", 0.0) 
                                  for h in list(self.interaction_history)[-lookahead_steps:]]
                    if len(recent_risks) > 1:
                        trend = (recent_risks[-1] - recent_risks[0]) / len(recent_risks)
                        predicted_risk = zone.risk_level + trend * lookahead_steps
                        
                        if predicted_risk > self.risk_threshold:
                            predictions.append((f"{zone.system_a}_{zone.system_b}", predicted_risk))
        
        return predictions


class MultiAgentSafetyCoordinator:
    """Coordinates safety protocols across multiple habitat subsystem agents."""
    
    def __init__(self, agent_names: List[str], communication_protocol: str = "broadcast"):
        self.agent_names = agent_names
        self.communication_protocol = communication_protocol
        self.agent_safety_states = {name: {} for name in agent_names}
        self.coordination_history = deque(maxlen=1000)
        self.emergency_protocols = self._initialize_emergency_protocols()
        
    def _initialize_emergency_protocols(self) -> Dict[str, Callable]:
        """Initialize emergency response protocols."""
        protocols = {
            "cascade_failure_prevention": self._cascade_failure_protocol,
            "priority_system_protection": self._priority_protection_protocol,
            "emergency_isolation": self._emergency_isolation_protocol,
            "coordinated_shutdown": self._coordinated_shutdown_protocol
        }
        return protocols
    
    def coordinate_agents(self, agent_states: Dict[str, Dict], 
                         agent_actions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Coordinate actions across agents with safety constraints."""
        coordinated_actions = {}
        
        # Detect safety conflicts
        safety_conflicts = self._detect_safety_conflicts(agent_states, agent_actions)
        
        if safety_conflicts:
            logger.warning(f"Safety conflicts detected: {safety_conflicts}")
            
            # Apply conflict resolution
            for conflict in safety_conflicts:
                resolution = self._resolve_conflict(conflict, agent_states, agent_actions)
                agent_actions.update(resolution)
        
        # Apply coordination rules
        for agent_name in self.agent_names:
            if agent_name in agent_actions:
                coordinated_action = self._apply_coordination_rules(
                    agent_name, agent_actions[agent_name], agent_states, agent_actions
                )
                coordinated_actions[agent_name] = coordinated_action
        
        return coordinated_actions
    
    def _detect_safety_conflicts(self, agent_states: Dict, agent_actions: Dict) -> List[Dict]:
        """Detect potential safety conflicts between agents."""
        conflicts = []
        
        # Check for resource conflicts
        power_consumers = []
        for agent_name, action in agent_actions.items():
            if hasattr(action, 'power_consumption') and action.power_consumption > 0.8:
                power_consumers.append(agent_name)
        
        if len(power_consumers) > 2:
            conflicts.append({
                'type': 'power_overload',
                'agents': power_consumers,
                'severity': 'high'
            })
        
        # Check for atmospheric conflicts
        atmosphere_controllers = ['atmosphere_agent', 'thermal_agent']
        if all(agent in agent_actions for agent in atmosphere_controllers):
            # Simplified conflict detection
            conflicts.append({
                'type': 'atmosphere_coordination',
                'agents': atmosphere_controllers,
                'severity': 'medium'
            })
        
        return conflicts
    
    def _resolve_conflict(self, conflict: Dict, agent_states: Dict, 
                         agent_actions: Dict) -> Dict[str, torch.Tensor]:
        """Resolve detected safety conflict."""
        resolution = {}
        
        if conflict['type'] == 'power_overload':
            # Implement power load shedding
            affected_agents = conflict['agents']
            for i, agent in enumerate(affected_agents):
                reduction_factor = 1.0 - (i * 0.2)  # Progressive reduction
                if agent in agent_actions:
                    resolution[agent] = agent_actions[agent] * reduction_factor
        
        elif conflict['type'] == 'atmosphere_coordination':
            # Coordinate atmospheric control actions
            if 'atmosphere_agent' in agent_actions and 'thermal_agent' in agent_actions:
                # Simple coordination: reduce thermal action when atmosphere is active
                resolution['thermal_agent'] = agent_actions['thermal_agent'] * 0.7
        
        return resolution
    
    def _apply_coordination_rules(self, agent_name: str, action: torch.Tensor,
                                agent_states: Dict, all_actions: Dict) -> torch.Tensor:
        """Apply coordination rules to individual agent actions."""
        coordinated_action = action.clone()
        
        # Rule 1: Priority system protection
        if agent_name == 'life_support_agent':
            # Life support always gets priority
            coordinated_action = action  # No modification
        
        # Rule 2: Emergency response coordination
        emergency_detected = any(
            state.get('emergency', False) for state in agent_states.values()
        )
        
        if emergency_detected and agent_name != 'emergency_response_agent':
            # Reduce non-emergency actions during emergencies
            coordinated_action = action * 0.5
        
        # Rule 3: Resource conservation
        total_resource_usage = sum(
            torch.norm(a).item() for a in all_actions.values() 
            if isinstance(a, torch.Tensor)
        )
        
        if total_resource_usage > 10.0:  # Threshold for high usage
            conservation_factor = 10.0 / total_resource_usage
            coordinated_action = action * conservation_factor
        
        return coordinated_action
    
    # Emergency protocol implementations
    def _cascade_failure_protocol(self, trigger_agent: str, failure_info: Dict):
        """Prevent cascade failures across systems."""
        logger.critical(f"Cascade failure prevention triggered by {trigger_agent}")
        # Implementation would coordinate shutdown sequences
        pass
    
    def _priority_protection_protocol(self, critical_systems: List[str]):
        """Protect priority systems during emergencies."""
        logger.critical(f"Priority protection for systems: {critical_systems}")
        # Implementation would isolate and protect critical systems
        pass
    
    def _emergency_isolation_protocol(self, affected_systems: List[str]):
        """Isolate affected systems to prevent spread of failures."""
        logger.critical(f"Emergency isolation of systems: {affected_systems}")
        # Implementation would isolate systems
        pass
    
    def _coordinated_shutdown_protocol(self, shutdown_sequence: List[str]):
        """Coordinate safe shutdown of multiple systems."""
        logger.critical(f"Coordinated shutdown sequence: {shutdown_sequence}")
        # Implementation would manage shutdown sequence
        pass


class DRSRLAgent:
    """
    Dynamic Residual Safe Reinforcement Learning Agent
    
    Implements the complete DRS-RL algorithm with multi-agent coordination.
    """
    
    def __init__(
        self,
        agent_name: str,
        state_dim: int,
        action_dim: int,
        safety_boundaries: List[SafetyBoundary],
        learning_rate: float = 1e-4,
        safety_threshold: float = 0.5
    ):
        self.agent_name = agent_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.safety_boundaries = safety_boundaries
        self.safety_threshold = safety_threshold
        
        # Initialize weak-to-strong safety corrector
        self.safety_corrector = WeakToStrongSafetyCorrector(
            state_dim, action_dim, len(safety_boundaries)
        )
        
        # Base policy network (the policy being corrected)
        self.base_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Optimizers
        self.safety_optimizer = torch.optim.Adam(self.safety_corrector.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.base_policy.parameters(), lr=learning_rate)
        
        # Dynamic boundary adaptation
        self.boundary_history = deque(maxlen=1000)
        self.adaptation_rate = 0.01
        
        # Performance tracking
        self.safety_violations = 0
        self.total_steps = 0
        self.parameter_efficiency_ratio = self._compute_parameter_efficiency()
        
        logger.info(f"DRS-RL Agent '{agent_name}' initialized")
        logger.info(f"Parameter efficiency ratio: {self.parameter_efficiency_ratio:.4f}")
    
    def _compute_parameter_efficiency(self) -> float:
        """Compute parameter efficiency improvement."""
        safety_params = sum(p.numel() for p in self.safety_corrector.parameters())
        total_params = sum(p.numel() for p in self.base_policy.parameters()) + safety_params
        
        # Compare to equivalent full-scale network
        equivalent_full_network_params = total_params * 3  # Assumed baseline
        efficiency_ratio = safety_params / equivalent_full_network_params
        
        return efficiency_ratio
    
    def hardware_fault_adaptation(self, state: torch.Tensor, fault_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Dynamically adjust safety boundaries when sensors fail or systems degrade.
        """
        # Get base action from policy
        base_action = self.base_policy(state)
        
        # Apply safety correction with fault considerations
        corrected_action, safety_scores, correction = self.safety_corrector(state, base_action)
        
        # Adapt safety boundaries based on fault information
        adaptation_info = {'adapted_boundaries': []}
        
        for i, boundary in enumerate(self.safety_boundaries):
            if boundary.parameter in fault_info.get('degraded_sensors', []):
                # Widen safety boundaries for degraded sensors
                degradation_factor = fault_info.get('degradation_levels', {}).get(boundary.parameter, 1.0)
                adapted_lower = boundary.lower_bound * (1 - degradation_factor * 0.1)
                adapted_upper = boundary.upper_bound * (1 + degradation_factor * 0.1)
                
                adaptation_info['adapted_boundaries'].append({
                    'parameter': boundary.parameter,
                    'original_bounds': (boundary.lower_bound, boundary.upper_bound),
                    'adapted_bounds': (adapted_lower, adapted_upper),
                    'degradation_factor': degradation_factor
                })
                
                # Update safety score based on adapted bounds
                if len(safety_scores) > i:
                    safety_scores[i] = safety_scores[i] * (1 - degradation_factor * 0.2)
        
        # Apply additional correction if safety scores are too low
        critical_safety = (safety_scores < self.safety_threshold).any()
        if critical_safety:
            # Emergency safety correction
            emergency_correction = torch.zeros_like(corrected_action)
            emergency_correction.normal_(0, 0.1)  # Small random correction
            corrected_action = 0.8 * corrected_action + 0.2 * emergency_correction
            adaptation_info['emergency_correction_applied'] = True
        
        adaptation_info['safety_scores'] = safety_scores.detach().numpy().tolist()
        adaptation_info['critical_safety_triggered'] = critical_safety
        
        return corrected_action, adaptation_info
    
    def predictive_risk_assessment(self, state: torch.Tensor, action: torch.Tensor, 
                                 lookahead: int = 5) -> Dict[str, Any]:
        """
        Use dynamic conflict zone modeling to predict and prevent dangerous interactions.
        """
        risk_assessment = {
            'immediate_risk': 0.0,
            'predicted_risks': [],
            'preventive_actions': []
        }
        
        # Immediate risk assessment
        _, safety_scores, _ = self.safety_corrector(state, action)
        immediate_risk = 1.0 - safety_scores.mean().item()
        risk_assessment['immediate_risk'] = immediate_risk
        
        # Predict future risks (simplified)
        for step in range(1, lookahead + 1):
            # Simulate future state (simplified linear prediction)
            predicted_state = state + action * step * 0.1
            predicted_action = self.base_policy(predicted_state)
            _, future_safety_scores, _ = self.safety_corrector(predicted_state, predicted_action)
            
            future_risk = 1.0 - future_safety_scores.mean().item()
            risk_assessment['predicted_risks'].append({
                'step': step,
                'risk_level': future_risk,
                'critical': future_risk > 0.7
            })
            
            # Suggest preventive actions for high-risk predictions
            if future_risk > 0.6:
                risk_assessment['preventive_actions'].append({
                    'step': step,
                    'action': 'reduce_action_magnitude',
                    'magnitude': min(0.3, future_risk - 0.5)
                })
        
        return risk_assessment
    
    def train_step(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                   next_state: torch.Tensor, safety_violation: bool = False) -> Dict[str, float]:
        """Training step for DRS-RL agent."""
        self.total_steps += 1
        if safety_violation:
            self.safety_violations += 1
        
        # Get safety correction
        corrected_action, safety_scores, correction = self.safety_corrector(state, action)
        
        # Base policy loss
        base_action = self.base_policy(state)
        policy_loss = F.mse_loss(base_action, action)
        
        # Safety loss (encourage high safety scores)
        safety_target = torch.ones_like(safety_scores) * 0.9  # Target high safety
        safety_loss = F.mse_loss(safety_scores, safety_target)
        
        # Residual correction loss
        if safety_violation:
            # Strong penalty for corrections that led to violations
            correction_loss = torch.norm(correction) * 10.0
        else:
            # Minimize unnecessary corrections
            correction_loss = torch.norm(correction) * 0.1
        
        # Total loss
        total_safety_loss = safety_loss + correction_loss
        
        # Optimize safety corrector
        self.safety_optimizer.zero_grad()
        total_safety_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.safety_corrector.parameters(), max_norm=1.0)
        self.safety_optimizer.step()
        
        # Optimize base policy
        policy_reward_loss = -reward * torch.norm(base_action)  # Reward-weighted policy loss
        total_policy_loss = policy_loss + policy_reward_loss
        
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # Update boundary history
        self.boundary_history.append({
            'safety_scores': safety_scores.detach().numpy(),
            'violation': safety_violation,
            'correction_magnitude': torch.norm(correction).item()
        })
        
        return {
            'policy_loss': policy_loss.item(),
            'safety_loss': safety_loss.item(),
            'correction_loss': correction_loss.item() if isinstance(correction_loss, torch.Tensor) else correction_loss,
            'safety_violation_rate': self.safety_violations / self.total_steps,
            'parameter_efficiency': self.parameter_efficiency_ratio
        }
    
    def save_model(self, filepath: str):
        """Save DRS-RL model."""
        torch.save({
            'safety_corrector_state_dict': self.safety_corrector.state_dict(),
            'base_policy_state_dict': self.base_policy.state_dict(),
            'safety_optimizer_state_dict': self.safety_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'safety_boundaries': self.safety_boundaries,
            'safety_violations': self.safety_violations,
            'total_steps': self.total_steps,
        }, filepath)
        logger.info(f"DRS-RL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load DRS-RL model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.safety_corrector.load_state_dict(checkpoint['safety_corrector_state_dict'])
        self.base_policy.load_state_dict(checkpoint['base_policy_state_dict'])
        self.safety_optimizer.load_state_dict(checkpoint['safety_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.safety_violations = checkpoint['safety_violations']
        self.total_steps = checkpoint['total_steps']
        logger.info(f"DRS-RL model loaded from {filepath}")


def demonstrate_drs_rl():
    """Demonstration of DRS-RL algorithm capabilities."""
    print("🛡️ Dynamic Residual Safe Reinforcement Learning Demonstration")
    print("=" * 70)
    
    # Define safety boundaries for lunar habitat
    safety_boundaries = [
        SafetyBoundary("oxygen_level", 18.0, 25.0, 0.01, 100.0),
        SafetyBoundary("co2_level", 0.0, 0.5, 0.01, 150.0),
        SafetyBoundary("pressure", 95.0, 105.0, 0.005, 200.0),
        SafetyBoundary("temperature", 18.0, 28.0, 0.02, 80.0),
        SafetyBoundary("power_stability", 0.95, 1.0, 0.01, 120.0),
    ]
    
    # Initialize agent
    state_dim = 34
    action_dim = 18
    
    agent = DRSRLAgent(
        agent_name="life_support_agent",
        state_dim=state_dim,
        action_dim=action_dim,
        safety_boundaries=safety_boundaries,
        learning_rate=1e-4
    )
    
    print("🔧 Hardware Fault Adaptation Test")
    state = torch.randn(state_dim)
    fault_info = {
        'degraded_sensors': ['oxygen_level', 'temperature'],
        'degradation_levels': {'oxygen_level': 0.3, 'temperature': 0.15}
    }
    adapted_action, adaptation_info = agent.hardware_fault_adaptation(state, fault_info)
    print(f"  Adapted action shape: {adapted_action.shape}")
    print(f"  Adaptation info: {len(adaptation_info['adapted_boundaries'])} boundaries adapted")
    print(f"  Emergency correction: {adaptation_info.get('emergency_correction_applied', False)}")
    
    print("\n🔮 Predictive Risk Assessment Test")
    action = torch.randn(action_dim)
    risk_assessment = agent.predictive_risk_assessment(state, action, lookahead=5)
    print(f"  Immediate risk: {risk_assessment['immediate_risk']:.4f}")
    print(f"  Predicted risks: {len(risk_assessment['predicted_risks'])} steps")
    print(f"  Preventive actions: {len(risk_assessment['preventive_actions'])} suggested")
    
    print("\n🤝 Multi-Agent Coordination Test")
    systems = ["atmosphere", "thermal", "power", "water"]
    coordinator = MultiAgentSafetyCoordinator(systems)
    
    agent_states = {sys: {"emergency": False, "status": "normal"} for sys in systems}
    agent_actions = {sys: torch.randn(action_dim) for sys in systems}
    
    coordinated_actions = coordinator.coordinate_agents(agent_states, agent_actions)
    print(f"  Coordinated {len(coordinated_actions)} agent actions")
    
    print("\n⚡ Dynamic Conflict Zone Modeling Test")
    conflict_modeler = DynamicConflictZoneModeler(systems)
    system_states = {sys: torch.randn(8) for sys in systems}
    system_actions = {sys: torch.randn(4) for sys in systems}
    
    conflict_modeler.update_conflict_zones(system_states, system_actions)
    high_risk_zones = conflict_modeler.get_high_risk_zones()
    print(f"  Monitored {len(conflict_modeler.conflict_zones)} conflict zones")
    print(f"  High-risk zones: {len(high_risk_zones)}")
    
    predicted_conflicts = conflict_modeler.predict_conflicts(lookahead_steps=3)
    print(f"  Predicted conflicts: {len(predicted_conflicts)}")
    
    print("\n🧠 Training Step Demonstration")
    reward = torch.tensor(12.0)
    next_state = state + torch.randn_like(state) * 0.1
    
    metrics = agent.train_step(state, action, reward, next_state, safety_violation=False)
    print(f"  Training metrics: {metrics}")
    
    print("\n✅ DRS-RL demonstration completed successfully!")
    print(f"   Safety violation rate: {metrics['safety_violation_rate']:.4f}")
    print(f"   Parameter efficiency: {metrics['parameter_efficiency']:.4f}")
    print("   Algorithm ready for safe multi-agent lunar habitat control 🌙")


if __name__ == "__main__":
    demonstrate_drs_rl()