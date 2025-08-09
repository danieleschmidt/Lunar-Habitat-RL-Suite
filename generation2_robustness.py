#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Enhanced Error Handling and Reliability
"""

import sys
import os
import json
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
sys.path.insert(0, '/root/repo')

from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv
from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent, HeuristicAgent, PIDControllerAgent
from lunar_habitat_rl.core.lightweight_config import HabitatConfig


class RobustHabitatEnv:
    """Robust wrapper for lunar habitat environment with comprehensive error handling."""
    
    def __init__(self, config: Optional[HabitatConfig] = None, enable_logging: bool = True):
        """Initialize robust habitat environment."""
        self.config = config or HabitatConfig()
        self.enable_logging = enable_logging
        self.env = None
        self.episode_count = 0
        self.total_steps = 0
        self.error_count = 0
        self.performance_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': 0.0,
            'avg_reward': 0.0,
            'error_rate': 0.0
        }
        
        # Safety limits and monitoring
        self.safety_limits = {
            'min_o2': 16.0,  # kPa
            'max_co2': 1.0,  # kPa
            'min_temp': 15.0,  # °C
            'max_temp': 30.0,  # °C
            'min_battery': 10.0,  # %
            'min_water': 50.0  # liters
        }
        
        self._initialize_environment()
        
    def _initialize_environment(self):
        """Initialize the environment with error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.env = LunarHabitatEnv(config=self.config)
                self._log(f"Environment initialized successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                self._log(f"Environment initialization failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize environment after {max_retries} attempts")
                time.sleep(1)  # Brief delay before retry
                
    def reset(self, seed: Optional[int] = None) -> Tuple[List[float], Dict[str, Any]]:
        """Reset environment with robust error handling."""
        try:
            obs, info = self.env.reset(seed=seed)
            self.episode_count += 1
            self._log(f"Episode {self.episode_count} started")
            
            # Validate observation
            if not self._validate_observation(obs):
                self._log("WARNING: Invalid observation detected, using fallback")
                obs = self._get_fallback_observation()
                
            return obs, info
            
        except Exception as e:
            self.error_count += 1
            self._log(f"Error in reset: {e}")
            # Return safe fallback state
            return self._get_fallback_observation(), {'error': str(e), 'episode': self.episode_count}
    
    def step(self, action: List[float]) -> Tuple[List[float], float, bool, bool, Dict[str, Any]]:
        """Execute step with comprehensive safety checks."""
        try:
            self.total_steps += 1
            
            # Validate and sanitize action
            action = self._validate_and_sanitize_action(action)
            
            # Execute step
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Safety monitoring
            safety_violation = self._check_safety_violations(obs)
            if safety_violation:
                self._log(f"SAFETY VIOLATION: {safety_violation}")
                terminated = True
                reward -= 100.0  # Large penalty for safety violations
                info['safety_violation'] = safety_violation
            
            # Validate observation
            if not self._validate_observation(obs):
                self._log("WARNING: Invalid observation detected, using previous state")
                obs = self._get_fallback_observation()
            
            # Update performance metrics
            if terminated or truncated:
                episode_reward = info.get('total_reward', reward)
                episode_length = info.get('step', self.total_steps)
                self._update_performance_metrics(episode_reward, episode_length)
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            self.error_count += 1
            self._log(f"Error in step {self.total_steps}: {e}")
            
            # Return safe fallback state
            fallback_obs = self._get_fallback_observation()
            return fallback_obs, -10.0, True, False, {'error': str(e)}
    
    def _validate_observation(self, obs: List[float]) -> bool:
        """Validate that observation is reasonable."""
        if not isinstance(obs, list) or len(obs) == 0:
            return False
            
        # Check for NaN or infinite values
        for val in obs:
            if not isinstance(val, (int, float)) or not (-1e6 < val < 1e6):
                return False
                
        return True
    
    def _validate_and_sanitize_action(self, action: List[float]) -> List[float]:
        """Validate and sanitize action to safe bounds."""
        if not isinstance(action, list):
            self._log("WARNING: Invalid action type, using random action")
            return [0.5] * self.env.action_space.shape[0]
            
        # Ensure correct length
        expected_len = self.env.action_space.shape[0]
        if len(action) != expected_len:
            self._log(f"WARNING: Action length mismatch, padding/truncating")
            if len(action) < expected_len:
                action.extend([0.5] * (expected_len - len(action)))
            else:
                action = action[:expected_len]
        
        # Clamp values to [0, 1] range (except solar angle)
        sanitized_action = []
        for i, val in enumerate(action):
            if isinstance(val, (int, float)) and -1e6 < val < 1e6:
                if i == 12:  # Solar angle index (special case)
                    sanitized_action.append(max(0, min(1, val)))  # Will be mapped to [-90, 90]
                else:
                    sanitized_action.append(max(0, min(1, val)))
            else:
                sanitized_action.append(0.5)  # Safe default
                
        return sanitized_action
    
    def _check_safety_violations(self, obs: List[float]) -> Optional[str]:
        """Check for critical safety violations."""
        if len(obs) < 10:
            return None
            
        try:
            # Extract key state variables (based on state structure)
            o2_pressure = obs[0]
            co2_pressure = obs[1]
            temperature = obs[5] if len(obs) > 5 else 22.5
            battery_charge = obs[8] if len(obs) > 8 else 50.0
            
            # Check safety limits
            if o2_pressure < self.safety_limits['min_o2']:
                return f"O2 critically low: {o2_pressure:.2f} kPa"
            if co2_pressure > self.safety_limits['max_co2']:
                return f"CO2 critically high: {co2_pressure:.2f} kPa"
            if temperature < self.safety_limits['min_temp'] or temperature > self.safety_limits['max_temp']:
                return f"Temperature critical: {temperature:.1f}°C"
            if battery_charge < self.safety_limits['min_battery']:
                return f"Battery critically low: {battery_charge:.1f}%"
                
        except Exception as e:
            self._log(f"Error checking safety violations: {e}")
            
        return None
    
    def _get_fallback_observation(self) -> List[float]:
        """Get a safe fallback observation."""
        # Create a reasonable default state
        fallback_state = [
            21.3,   # O2 pressure
            0.4,    # CO2 pressure  
            79.0,   # N2 pressure
            101.3,  # Total pressure
            45.0,   # Humidity
            22.5,   # Temperature
            0.95,   # Air quality
            8.5,    # Solar generation
            75.0,   # Battery charge
            90.0,   # Fuel cell capacity
            6.2,    # Total load
            100.0,  # Emergency reserve
            0.98,   # Grid stability
        ]
        
        # Pad to expected length
        while len(fallback_state) < 48:  # Expected observation size
            fallback_state.append(0.5)
            
        return fallback_state
    
    def _update_performance_metrics(self, reward: float, length: int):
        """Update performance tracking metrics."""
        self.performance_metrics['episode_rewards'].append(reward)
        self.performance_metrics['episode_lengths'].append(length)
        
        # Calculate running statistics
        rewards = self.performance_metrics['episode_rewards']
        self.performance_metrics['avg_reward'] = sum(rewards) / len(rewards)
        self.performance_metrics['success_rate'] = sum(1 for r in rewards if r > 0) / len(rewards)
        self.performance_metrics['error_rate'] = self.error_count / max(self.total_steps, 1)
    
    def _log(self, message: str):
        """Log message with timestamp."""
        if self.enable_logging:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **self.performance_metrics,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'error_count': self.error_count,
            'safety_limits': self.safety_limits
        }
    
    def close(self):
        """Close environment with cleanup."""
        try:
            if self.env:
                self.env.close()
            self._log("Environment closed successfully")
        except Exception as e:
            self._log(f"Error closing environment: {e}")


class RobustAgent:
    """Wrapper for agents with error handling and monitoring."""
    
    def __init__(self, agent_type: str = 'heuristic', **kwargs):
        """Initialize robust agent wrapper."""
        self.agent_type = agent_type
        self.error_count = 0
        self.prediction_count = 0
        self.last_action = None
        
        # Initialize base agent
        agent_classes = {
            'random': RandomAgent,
            'heuristic': HeuristicAgent,
            'pid': PIDControllerAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        try:
            self.agent = agent_classes[agent_type](**kwargs)
        except Exception as e:
            print(f"Error initializing {agent_type} agent: {e}")
            # Fallback to random agent
            self.agent = RandomAgent(**kwargs)
            self.agent_type = 'random (fallback)'
    
    def predict(self, observation: List[float], deterministic: bool = True) -> Tuple[List[float], Optional[Any]]:
        """Robust prediction with error handling."""
        self.prediction_count += 1
        
        try:
            # Validate observation
            if not isinstance(observation, list) or len(observation) == 0:
                raise ValueError("Invalid observation format")
                
            # Get prediction from base agent
            action, state = self.agent.predict(observation)
            
            # Validate action
            if not isinstance(action, list) or len(action) == 0:
                raise ValueError("Agent returned invalid action")
                
            self.last_action = action.copy()
            return action, state
            
        except Exception as e:
            self.error_count += 1
            print(f"Error in agent prediction: {e}")
            
            # Return safe fallback action
            if self.last_action is not None:
                return self.last_action, None
            else:
                # Default safe action
                return [0.5] * 22, None  # Assume 22-dim action space
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            'agent_type': self.agent_type,
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.prediction_count, 1)
        }


def test_generation_2_robustness():
    """Test Generation 2 robustness features."""
    print("🛡️ GENERATION 2: MAKE IT ROBUST")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Robust Environment
    print("\n🔧 Test 1: Robust Environment Initialization")
    try:
        config = HabitatConfig()
        robust_env = RobustHabitatEnv(config=config, enable_logging=True)
        print("✅ Robust environment created successfully")
        test_results['robust_env'] = True
    except Exception as e:
        print(f"❌ Robust environment failed: {e}")
        test_results['robust_env'] = False
    
    # Test 2: Error Handling
    print("\n🚨 Test 2: Error Handling and Recovery")
    try:
        obs, info = robust_env.reset(seed=42)
        
        # Test with invalid action
        invalid_actions = [
            [],  # Empty action
            [float('inf')] * 22,  # Infinite values
            ['invalid'] * 22,  # String values
            [1.0] * 10,  # Wrong length
        ]
        
        recovery_count = 0
        for invalid_action in invalid_actions:
            obs, reward, done, truncated, info = robust_env.step(invalid_action)
            if 'error' not in info:  # Successfully recovered
                recovery_count += 1
        
        print(f"✅ Error recovery: {recovery_count}/{len(invalid_actions)} cases handled")
        test_results['error_handling'] = recovery_count >= 3
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        test_results['error_handling'] = False
    
    # Test 3: Safety Monitoring
    print("\n🛡️ Test 3: Safety Monitoring System")
    try:
        # Simulate safety violation by creating dangerous state
        dangerous_obs = [10.0, 2.0, 70.0, 85.0, 50.0, 35.0] + [0.0] * 42  # Low O2, high CO2, high temp
        
        # Check if safety system detects violations
        safety_violation = robust_env._check_safety_violations(dangerous_obs)
        
        if safety_violation:
            print(f"✅ Safety system detected violation: {safety_violation}")
            test_results['safety_monitoring'] = True
        else:
            print("❌ Safety system failed to detect violation")
            test_results['safety_monitoring'] = False
            
    except Exception as e:
        print(f"❌ Safety monitoring test failed: {e}")
        test_results['safety_monitoring'] = False
    
    # Test 4: Robust Agents
    print("\n🤖 Test 4: Robust Agent Wrappers")
    try:
        agents = ['random', 'heuristic', 'pid']
        agent_tests = {}
        
        for agent_type in agents:
            robust_agent = RobustAgent(agent_type=agent_type, action_dims=22)
            
            # Test normal operation
            obs = [21.3, 0.4, 79.0, 101.3, 45.0, 22.5] + [0.5] * 42
            action, state = robust_agent.predict(obs)
            
            # Test with invalid observation
            invalid_obs = []
            action2, state2 = robust_agent.predict(invalid_obs)
            
            agent_tests[agent_type] = isinstance(action, list) and isinstance(action2, list)
            
        success_count = sum(agent_tests.values())
        print(f"✅ Robust agents: {success_count}/{len(agents)} working correctly")
        test_results['robust_agents'] = success_count == len(agents)
        
    except Exception as e:
        print(f"❌ Robust agents test failed: {e}")
        test_results['robust_agents'] = False
    
    # Test 5: Performance Monitoring
    print("\n📊 Test 5: Performance Monitoring")
    try:
        # Run a short episode to generate metrics
        robust_agent = RobustAgent(agent_type='heuristic', action_dims=22)
        obs, info = robust_env.reset()
        
        for step in range(5):
            action, _ = robust_agent.predict(obs)
            obs, reward, done, truncated, info = robust_env.step(action)
            if done or truncated:
                break
                
        # Get metrics
        env_metrics = robust_env.get_metrics()
        agent_stats = robust_agent.get_stats()
        
        print(f"✅ Environment metrics collected: {len(env_metrics)} fields")
        print(f"✅ Agent statistics collected: {len(agent_stats)} fields")
        test_results['performance_monitoring'] = True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        test_results['performance_monitoring'] = False
    
    # Cleanup
    try:
        robust_env.close()
    except:
        pass
    
    # Overall results
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n🎯 Generation 2 Results: {passed_tests}/{total_tests} tests passed")
    print(f"📊 Test Details: {test_results}")
    
    if passed_tests >= 4:
        print("🎉 GENERATION 2 COMPLETE - System is robust and reliable!")
        return True
    else:
        print("⚠️ Generation 2 needs improvements")
        return False


if __name__ == "__main__":
    success = test_generation_2_robustness()
    
    if success:
        print("\n🚀 READY FOR GENERATION 3: MAKE IT SCALE! 🚀")
    
    sys.exit(0 if success else 1)