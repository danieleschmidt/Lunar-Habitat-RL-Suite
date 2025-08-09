"""Lightweight baseline algorithms without heavy ML dependencies - Generation 1"""

import random
import math
from typing import Dict, List, Tuple, Optional, Any


class RandomAgent:
    """Random action baseline agent."""
    
    def __init__(self, action_dims: int = 22):
        """Initialize random agent.
        
        Args:
            action_dims: Number of action dimensions.
        """
        self.action_dims = action_dims
        self.name = "RandomAgent"
    
    def predict(self, observation: List[float]) -> Tuple[List[float], Optional[Any]]:
        """Predict action given observation.
        
        Args:
            observation: Current observation.
            
        Returns:
            Tuple of (action, state) where state is None for stateless policies.
        """
        action = [random.random() for _ in range(self.action_dims)]
        return action, None
    
    def train(self, *args, **kwargs):
        """Random agent doesn't need training."""
        pass


class HeuristicAgent:
    """Rule-based heuristic agent for habitat control."""
    
    def __init__(self, action_dims: int = 22):
        """Initialize heuristic agent.
        
        Args:
            action_dims: Number of action dimensions.
        """
        self.action_dims = action_dims
        self.name = "HeuristicAgent"
        
        # Target setpoints
        self.target_o2 = 21.3
        self.target_co2_max = 0.4
        self.target_temp = 22.5
        self.target_battery = 60.0
        
    def predict(self, observation: List[float]) -> Tuple[List[float], Optional[Any]]:
        """Predict action using heuristic rules.
        
        Args:
            observation: Current observation (state array).
            
        Returns:
            Tuple of (action, state) where state is None.
        """
        # Parse observation (assuming standard state format)
        # [atmosphere(7), power(6), thermal(8), water(5), crew(18), environment(4)]
        
        if len(observation) < 30:
            # Fallback to random if observation is too short
            return [random.random() for _ in range(self.action_dims)], None
            
        # Extract key state variables
        o2_pressure = observation[0]
        co2_pressure = observation[1]
        temperature = observation[5] if len(observation) > 5 else 22.5
        battery_charge = observation[8] if len(observation) > 8 else 50.0
        
        # Initialize action array
        action = [0.5] * self.action_dims  # Default middle values
        
        # Life support control (actions 0-5)
        # O2 generation
        if o2_pressure < self.target_o2 - 2:
            action[0] = 0.8  # High O2 generation
        elif o2_pressure < self.target_o2:
            action[0] = 0.6  # Medium O2 generation
        else:
            action[0] = 0.2  # Low O2 generation
            
        # CO2 scrubbing
        if co2_pressure > self.target_co2_max:
            action[1] = 0.9  # High CO2 scrubbing
        elif co2_pressure > self.target_co2_max * 0.7:
            action[1] = 0.6  # Medium CO2 scrubbing
        else:
            action[1] = 0.3  # Low CO2 scrubbing
            
        # Air circulation
        action[3] = 0.7  # Keep air circulation relatively high
        
        # Power management (actions 6-14)
        # Battery charge rate
        if battery_charge < 30:
            action[6] = 0.8  # High charging when low
        elif battery_charge < 60:
            action[6] = 0.5  # Medium charging
        else:
            action[6] = 0.2  # Low charging when high
            
        # Load shedding (actions 7-10) - avoid unless critical
        if battery_charge < 20:
            action[7:11] = [0.3, 0.7, 0.3, 0.7]  # Shed some non-critical loads
        else:
            action[7:11] = [0.0, 0.0, 0.0, 0.0]  # No load shedding
            
        # Fuel cell activation
        if battery_charge < 25:
            action[11] = 0.6  # Activate fuel cell when battery low
        else:
            action[11] = 0.1  # Keep fuel cell on standby
            
        # Thermal control (actions 15-22)
        # Heating zones
        if temperature < self.target_temp - 2:
            action[15:19] = [0.8, 0.8, 0.8, 0.8]  # High heating
        elif temperature < self.target_temp:
            action[15:19] = [0.6, 0.6, 0.6, 0.6]  # Medium heating
        elif temperature > self.target_temp + 2:
            action[15:19] = [0.2, 0.2, 0.2, 0.2]  # Low heating
        else:
            action[15:19] = [0.4, 0.4, 0.4, 0.4]  # Maintain
            
        # Radiator flow
        if temperature > self.target_temp + 1:
            action[19:21] = [0.8, 0.8]  # High radiator flow for cooling
        else:
            action[19:21] = [0.5, 0.5]  # Medium radiator flow
            
        # Ensure all actions are in [0, 1] range
        action = [max(0, min(1, a)) for a in action]
        
        # Pad or trim to correct size
        while len(action) < self.action_dims:
            action.append(0.5)
        action = action[:self.action_dims]
        
        return action, None
    
    def train(self, *args, **kwargs):
        """Heuristic agent doesn't need training."""
        pass


class PIDControllerAgent:
    """PID controller-based agent for precise control."""
    
    def __init__(self, action_dims: int = 22):
        """Initialize PID controller agent.
        
        Args:
            action_dims: Number of action dimensions.
        """
        self.action_dims = action_dims
        self.name = "PIDControllerAgent"
        
        # PID parameters for different systems
        self.o2_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.co2_pid = PIDController(kp=2.0, ki=0.2, kd=0.1)
        self.temp_pid = PIDController(kp=0.5, ki=0.05, kd=0.02)
        self.battery_pid = PIDController(kp=0.8, ki=0.1, kd=0.03)
        
        # Setpoints
        self.o2_setpoint = 21.3
        self.co2_setpoint = 0.2  # Target low CO2
        self.temp_setpoint = 22.5
        self.battery_setpoint = 70.0
        
    def predict(self, observation: List[float]) -> Tuple[List[float], Optional[Any]]:
        """Predict action using PID controllers.
        
        Args:
            observation: Current observation.
            
        Returns:
            Tuple of (action, state).
        """
        if len(observation) < 30:
            return [0.5] * self.action_dims, None
            
        # Extract state variables
        o2_pressure = observation[0]
        co2_pressure = observation[1]
        temperature = observation[5] if len(observation) > 5 else 22.5
        battery_charge = observation[8] if len(observation) > 8 else 50.0
        
        # Compute PID outputs
        o2_control = self.o2_pid.update(self.o2_setpoint, o2_pressure)
        co2_control = self.co2_pid.update(self.co2_setpoint, co2_pressure)
        temp_control = self.temp_pid.update(self.temp_setpoint, temperature)
        battery_control = self.battery_pid.update(self.battery_setpoint, battery_charge)
        
        # Initialize action
        action = [0.5] * self.action_dims
        
        # Apply PID outputs to actions
        # O2 generation (action 0)
        action[0] = max(0, min(1, 0.5 + o2_control))
        
        # CO2 scrubbing (action 1) - reverse control (more scrubbing for higher CO2)
        action[1] = max(0, min(1, 0.5 - co2_control))
        
        # Battery charging (action 6)
        action[6] = max(0, min(1, 0.5 + battery_control))
        
        # Thermal control (actions 15-18)
        for i in range(4):
            action[15 + i] = max(0, min(1, 0.5 + temp_control))
            
        return action, None
    
    def train(self, *args, **kwargs):
        """PID controller doesn't need training."""
        pass


class PIDController:
    """Simple PID controller implementation."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        """Initialize PID controller.
        
        Args:
            kp: Proportional gain.
            ki: Integral gain.
            kd: Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.prev_error = 0.0
        self.integral = 0.0
        
    def update(self, setpoint: float, process_variable: float) -> float:
        """Update PID controller.
        
        Args:
            setpoint: Desired value.
            process_variable: Current measured value.
            
        Returns:
            Control output.
        """
        error = setpoint - process_variable
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = error - self.prev_error
        d_term = self.kd * derivative
        
        # Update for next iteration
        self.prev_error = error
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Clamp output to reasonable range
        return max(-2.0, min(2.0, output))
    
    def reset(self):
        """Reset PID controller state."""
        self.prev_error = 0.0
        self.integral = 0.0


class GreedyAgent:
    """Greedy agent that always takes the action that maximizes immediate reward."""
    
    def __init__(self, action_dims: int = 22, n_samples: int = 10):
        """Initialize greedy agent.
        
        Args:
            action_dims: Number of action dimensions.
            n_samples: Number of random actions to sample and evaluate.
        """
        self.action_dims = action_dims
        self.n_samples = n_samples
        self.name = "GreedyAgent"
        
    def predict(self, observation: List[float]) -> Tuple[List[float], Optional[Any]]:
        """Predict action by sampling and choosing the best.
        
        Args:
            observation: Current observation.
            
        Returns:
            Tuple of (action, state).
        """
        best_action = None
        best_score = float('-inf')
        
        for _ in range(self.n_samples):
            # Sample random action
            action = [random.random() for _ in range(self.action_dims)]
            
            # Score action based on heuristics
            score = self._score_action(observation, action)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action or [0.5] * self.action_dims, None
    
    def _score_action(self, observation: List[float], action: List[float]) -> float:
        """Score an action based on expected reward.
        
        Args:
            observation: Current observation.
            action: Action to score.
            
        Returns:
            Expected score/reward.
        """
        if len(observation) < 6:
            return 0.0
            
        score = 0.0
        
        # Extract current state
        o2_pressure = observation[0]
        co2_pressure = observation[1]
        temperature = observation[5]
        
        # O2 control score
        if action[0] > 0.5 and o2_pressure < 21.3:
            score += 1.0  # Good to increase O2 when low
        elif action[0] < 0.5 and o2_pressure > 21.3:
            score += 0.5  # Good to decrease O2 when high
            
        # CO2 control score
        if action[1] > 0.5 and co2_pressure > 0.4:
            score += 1.0  # Good to scrub CO2 when high
        elif action[1] < 0.5 and co2_pressure < 0.2:
            score += 0.5  # Good to reduce scrubbing when CO2 is low
            
        # Temperature control score
        if temperature < 22.5 and sum(action[15:19]) > 2.0:  # High heating when cold
            score += 0.5
        elif temperature > 22.5 and sum(action[15:19]) < 2.0:  # Low heating when warm
            score += 0.5
            
        # Add some randomness to break ties
        score += random.uniform(-0.1, 0.1)
        
        return score
    
    def train(self, *args, **kwargs):
        """Greedy agent doesn't need training."""
        pass


# Registry of available agents
BASELINE_AGENTS = {
    'random': RandomAgent,
    'heuristic': HeuristicAgent, 
    'pid': PIDControllerAgent,
    'greedy': GreedyAgent
}


def get_baseline_agent(agent_type: str, **kwargs) -> Any:
    """Create a baseline agent of the specified type.
    
    Args:
        agent_type: Type of agent ('random', 'heuristic', 'pid', 'greedy').
        **kwargs: Additional arguments for the agent constructor.
        
    Returns:
        Initialized agent instance.
    """
    if agent_type not in BASELINE_AGENTS:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(BASELINE_AGENTS.keys())}")
        
    agent_class = BASELINE_AGENTS[agent_type]
    return agent_class(**kwargs)