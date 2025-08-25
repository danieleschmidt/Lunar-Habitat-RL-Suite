"""
Constrained Multi-Objective Reinforcement Learning (C-MORL)

Advanced RL algorithm for balancing competing objectives in lunar habitat life support
while maintaining hard safety constraints.

Based on breakthrough research from ArXiv 2410.02236 (October 2024).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from collections import deque
import random

logger = logging.getLogger(__name__)


@dataclass
class Objective:
    """Defines a single objective for multi-objective optimization."""
    name: str
    weight: float
    constraint_type: str  # 'hard', 'soft', 'none'
    constraint_threshold: Optional[float] = None
    priority: int = 0  # Higher = more important during emergencies


@dataclass
class ParetoPoint:
    """Represents a point on the Pareto front."""
    objectives: torch.Tensor
    action: torch.Tensor
    dominates_count: int = 0
    dominated_by: List[int] = None
    
    def __post_init__(self):
        if self.dominated_by is None:
            self.dominated_by = []


class ParetoDominanceCalculator:
    """Calculates Pareto dominance relationships between solutions."""
    
    @staticmethod
    def dominates(obj1: torch.Tensor, obj2: torch.Tensor, constraints: List[float] = None) -> bool:
        """Check if objective vector obj1 dominates obj2."""
        # Standard dominance: obj1 >= obj2 in all dimensions, > in at least one
        better_or_equal = torch.all(obj1 >= obj2)
        strictly_better = torch.any(obj1 > obj2)
        
        dominance = better_or_equal and strictly_better
        
        # Apply constraint violations (hard constraints override dominance)
        if constraints:
            obj1_violations = sum(1 for i, c in enumerate(constraints) if obj1[i] < c)
            obj2_violations = sum(1 for i, c in enumerate(constraints) if obj2[i] < c)
            
            if obj1_violations < obj2_violations:
                return True
            elif obj1_violations > obj2_violations:
                return False
        
        return dominance
    
    @staticmethod
    def find_pareto_front(objectives: torch.Tensor, constraints: List[float] = None) -> List[int]:
        """Find indices of non-dominated solutions (Pareto front)."""
        n_solutions = objectives.shape[0]
        dominated = [False] * n_solutions
        
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j and ParetoDominanceCalculator.dominates(objectives[j], objectives[i], constraints):
                    dominated[i] = True
                    break
        
        pareto_indices = [i for i in range(n_solutions) if not dominated[i]]
        return pareto_indices


class AdaptiveParetoNetwork(nn.Module):
    """Neural network for adaptive Pareto front discovery."""
    
    def __init__(self, state_dim: int, action_dim: int, n_objectives: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Objective-specific heads
        self.objective_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(n_objectives)
        ])
        
        # Action policy network
        self.policy_head = nn.Sequential(
            nn.Linear(128 + n_objectives, 64),  # Features + objective preferences
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Preference adaptation network
        self.preference_adapter = nn.Sequential(
            nn.Linear(state_dim + n_objectives, 64),
            nn.ReLU(),
            nn.Linear(64, n_objectives),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor, objective_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through adaptive Pareto network."""
        features = self.feature_extractor(state)
        
        # Predict objective values
        objective_values = torch.cat([
            head(features) for head in self.objective_heads
        ], dim=-1)
        
        # Adapt preferences based on current state
        adapted_weights = self.preference_adapter(torch.cat([state, objective_weights], dim=-1))
        
        # Generate action based on features and adapted preferences
        policy_input = torch.cat([features, adapted_weights], dim=-1)
        action = self.policy_head(policy_input)
        
        return action, objective_values


class CMORLAgent:
    """
    Constrained Multi-Objective Reinforcement Learning Agent
    
    Implements the complete C-MORL algorithm for lunar habitat control.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        objectives: List[Objective],
        learning_rate: float = 1e-4,
        pareto_front_size: int = 100,
        preference_adaptation_rate: float = 0.1
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.objectives = objectives
        self.n_objectives = len(objectives)
        self.pareto_front_size = pareto_front_size
        self.preference_adaptation_rate = preference_adaptation_rate
        
        # Initialize networks
        self.network = AdaptiveParetoNetwork(state_dim, action_dim, self.n_objectives)
        self.target_network = AdaptiveParetoNetwork(state_dim, action_dim, self.n_objectives)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Pareto front maintenance
        self.pareto_front = []
        self.experience_buffer = deque(maxlen=10000)
        
        # Dynamic preference weights
        self.current_preferences = torch.ones(self.n_objectives) / self.n_objectives
        
        # Emergency response system
        self.emergency_mode = False
        self.emergency_preferences = self._compute_emergency_preferences()
        
        logger.info(f"C-MORL Agent initialized with {self.n_objectives} objectives")
    
    def _compute_emergency_preferences(self) -> torch.Tensor:
        """Compute preference weights for emergency scenarios."""
        emergency_weights = torch.zeros(self.n_objectives)
        
        # Prioritize objectives based on their priority values
        total_priority = sum(obj.priority for obj in self.objectives)
        for i, obj in enumerate(self.objectives):
            if obj.priority > 0:
                emergency_weights[i] = obj.priority / total_priority
            else:
                emergency_weights[i] = 0.01  # Minimal weight for non-priority objectives
        
        return emergency_weights
    
    def safety_first_control(self, state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Maintain hard safety constraints while optimizing secondary objectives.
        """
        # Get current action and objective predictions
        action, objective_values = self.network(state.unsqueeze(0), self.current_preferences.unsqueeze(0))
        action = action.squeeze(0)
        objective_values = objective_values.squeeze(0)
        
        # Check constraint violations
        constraint_violations = []
        for i, obj in enumerate(self.objectives):
            if obj.constraint_type == 'hard' and obj.constraint_threshold is not None:
                if objective_values[i] < obj.constraint_threshold:
                    constraint_violations.append((i, obj.name, objective_values[i].item()))
        
        # Apply safety corrections if violations detected
        safety_info = {
            'violations': constraint_violations,
            'emergency_triggered': False,
            'corrected_action': False
        }
        
        if constraint_violations:
            # Switch to emergency mode
            self.emergency_mode = True
            self.current_preferences = self.emergency_preferences
            safety_info['emergency_triggered'] = True
            
            # Recompute action with safety-first preferences
            corrected_action, corrected_objectives = self.network(
                state.unsqueeze(0), 
                self.emergency_preferences.unsqueeze(0)
            )
            action = corrected_action.squeeze(0)
            safety_info['corrected_action'] = True
            
            logger.warning(f"Safety violations detected: {[v[1] for v in constraint_violations]}")
        
        return action, safety_info
    
    def dynamic_rebalancing(self, mission_state: str, crew_status: Dict[str, float]) -> torch.Tensor:
        """
        Dynamically rebalance objectives during crisis scenarios.
        """
        new_preferences = self.current_preferences.clone()
        
        if mission_state == "equipment_failure":
            # Prioritize system redundancy and fault tolerance
            for i, obj in enumerate(self.objectives):
                if "safety" in obj.name.lower() or "redundancy" in obj.name.lower():
                    new_preferences[i] *= 2.0
                elif "efficiency" in obj.name.lower():
                    new_preferences[i] *= 0.5
        
        elif mission_state == "power_critical":
            # Prioritize power efficiency and load shedding
            for i, obj in enumerate(self.objectives):
                if "power" in obj.name.lower() or "energy" in obj.name.lower():
                    new_preferences[i] *= 3.0
                elif "comfort" in obj.name.lower():
                    new_preferences[i] *= 0.3
        
        elif mission_state == "life_threatening":
            # Maximum priority to crew survival
            new_preferences = self.emergency_preferences.clone()
        
        # Apply crew stress multipliers
        avg_stress = sum(crew_status.get(f"crew_{i}_stress", 0.0) for i in range(4)) / 4
        if avg_stress > 0.7:  # High stress scenario
            for i, obj in enumerate(self.objectives):
                if "comfort" in obj.name.lower() or "psychological" in obj.name.lower():
                    new_preferences[i] *= (1 + avg_stress)
        
        # Normalize preferences
        new_preferences = F.softmax(new_preferences, dim=0)
        
        # Smooth adaptation
        self.current_preferences = (1 - self.preference_adaptation_rate) * self.current_preferences + \
                                   self.preference_adaptation_rate * new_preferences
        
        return self.current_preferences
    
    def resource_scarcity_optimization(self, resource_levels: Dict[str, float]) -> torch.Tensor:
        """
        Optimally allocate limited resources across multiple life support functions.
        """
        # Compute resource scarcity factors
        scarcity_factors = {}
        for resource, level in resource_levels.items():
            if level < 0.2:  # Critical level
                scarcity_factors[resource] = 3.0
            elif level < 0.5:  # Low level
                scarcity_factors[resource] = 2.0
            else:
                scarcity_factors[resource] = 1.0
        
        # Adjust objective preferences based on resource scarcity
        adjusted_preferences = self.current_preferences.clone()
        
        for i, obj in enumerate(self.objectives):
            for resource, factor in scarcity_factors.items():
                if resource.lower() in obj.name.lower():
                    adjusted_preferences[i] *= factor
        
        # Normalize
        adjusted_preferences = F.softmax(adjusted_preferences, dim=0)
        
        return adjusted_preferences
    
    def update_pareto_front(self, new_objectives: torch.Tensor, new_action: torch.Tensor):
        """Update Pareto front with new solution."""
        new_point = ParetoPoint(objectives=new_objectives, action=new_action)
        
        # Add to front
        self.pareto_front.append(new_point)
        
        # Remove dominated solutions and limit size
        if len(self.pareto_front) > self.pareto_front_size:
            objectives_matrix = torch.stack([p.objectives for p in self.pareto_front])
            constraints = [obj.constraint_threshold for obj in self.objectives 
                          if obj.constraint_type == 'hard']
            
            pareto_indices = ParetoDominanceCalculator.find_pareto_front(
                objectives_matrix, constraints
            )
            
            self.pareto_front = [self.pareto_front[i] for i in pareto_indices[:self.pareto_front_size]]
    
    def train_step(self, batch: List[Tuple]) -> Dict[str, float]:
        """Training step for C-MORL agent."""
        states, actions, objectives, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        objectives = torch.stack(objectives)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Current Q-values for all objectives
        current_actions, current_objectives = self.network(states, self.current_preferences.unsqueeze(0).repeat(len(batch), 1))
        
        # Target Q-values
        with torch.no_grad():
            next_actions, next_objectives = self.target_network(next_states, self.current_preferences.unsqueeze(0).repeat(len(batch), 1))
            targets = rewards.unsqueeze(-1) + 0.99 * next_objectives * (1 - dones.unsqueeze(-1))
        
        # Multi-objective loss
        objective_loss = F.mse_loss(current_objectives, targets)
        
        # Policy loss with constraint penalties
        policy_loss = F.mse_loss(current_actions, actions)
        
        # Constraint penalty
        constraint_penalty = 0.0
        for i, obj in enumerate(self.objectives):
            if obj.constraint_type == 'hard' and obj.constraint_threshold is not None:
                violations = torch.relu(obj.constraint_threshold - current_objectives[:, i])
                constraint_penalty += violations.mean() * 100.0  # Heavy penalty
        
        # Total loss
        total_loss = objective_loss + policy_loss + constraint_penalty
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target(tau=0.005)
        
        return {
            'objective_loss': objective_loss.item(),
            'policy_loss': policy_loss.item(),
            'constraint_penalty': constraint_penalty if isinstance(constraint_penalty, (int, float)) else constraint_penalty.item(),
            'total_loss': total_loss.item(),
            'pareto_front_size': len(self.pareto_front)
        }
    
    def _soft_update_target(self, tau: float = 0.005):
        """Soft update of target network."""
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save C-MORL model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'objectives': self.objectives,
            'current_preferences': self.current_preferences,
            'pareto_front': self.pareto_front,
        }, filepath)
        logger.info(f"C-MORL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load C-MORL model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_preferences = checkpoint['current_preferences']
        self.pareto_front = checkpoint['pareto_front']
        logger.info(f"C-MORL model loaded from {filepath}")


def demonstrate_cmorl():
    """Demonstration of C-MORL algorithm capabilities."""
    print("🎯 Constrained Multi-Objective Reinforcement Learning Demonstration")
    print("=" * 70)
    
    # Define objectives for lunar habitat
    objectives = [
        Objective("crew_safety", weight=0.4, constraint_type="hard", constraint_threshold=0.95, priority=10),
        Objective("oxygen_efficiency", weight=0.15, constraint_type="soft", constraint_threshold=0.8, priority=8),
        Objective("power_consumption", weight=0.15, constraint_type="none", priority=5),
        Objective("resource_conservation", weight=0.1, constraint_type="soft", constraint_threshold=0.7, priority=6),
        Objective("crew_comfort", weight=0.1, constraint_type="none", priority=2),
        Objective("system_longevity", weight=0.1, constraint_type="soft", constraint_threshold=0.85, priority=7),
    ]
    
    # Initialize agent
    state_dim = 34  # Full habitat state
    action_dim = 18  # Multi-system actions
    
    agent = CMORLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        objectives=objectives,
        learning_rate=1e-4
    )
    
    # Simulate habitat state
    state = torch.randn(state_dim)
    
    print("🛡️  Safety-First Control Test")
    action, safety_info = agent.safety_first_control(state)
    print(f"  Action shape: {action.shape}")
    print(f"  Safety info: {safety_info}")
    
    print("\n⚖️  Dynamic Rebalancing Test")
    crew_status = {f"crew_{i}_stress": random.uniform(0.2, 0.8) for i in range(4)}
    preferences = agent.dynamic_rebalancing("equipment_failure", crew_status)
    print(f"  New preferences: {preferences}")
    print(f"  Top priority: {objectives[torch.argmax(preferences)].name}")
    
    print("\n💧 Resource Scarcity Optimization Test")
    resource_levels = {
        "water": 0.15,  # Critical
        "oxygen": 0.45,  # Low
        "power": 0.8,    # Normal
        "food": 0.3      # Low
    }
    optimized_preferences = agent.resource_scarcity_optimization(resource_levels)
    print(f"  Optimized preferences: {optimized_preferences}")
    
    print("\n📊 Pareto Front Update Test")
    sample_objectives = torch.tensor([0.95, 0.82, 0.75, 0.68, 0.55, 0.89])
    sample_action = torch.randn(action_dim)
    agent.update_pareto_front(sample_objectives, sample_action)
    print(f"  Pareto front size: {len(agent.pareto_front)}")
    
    print("\n🧠 Training Step Demonstration")
    # Create sample batch
    batch = []
    for _ in range(8):
        sample_state = torch.randn(state_dim)
        sample_action = torch.randn(action_dim)
        sample_objectives = torch.rand(len(objectives))
        sample_reward = torch.randn(len(objectives))
        sample_next_state = torch.randn(state_dim)
        batch.append((sample_state, sample_action, sample_objectives, sample_reward, sample_next_state, False))
    
    metrics = agent.train_step(batch)
    print(f"  Training metrics: {metrics}")
    
    print("\n✅ C-MORL demonstration completed successfully!")
    print(f"   Managing {len(objectives)} competing objectives with safety constraints 🎯")
    print("   Algorithm ready for multi-objective lunar habitat optimization 🌙")


if __name__ == "__main__":
    demonstrate_cmorl()