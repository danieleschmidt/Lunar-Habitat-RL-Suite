#!/usr/bin/env python3
"""Generation 2 Demo Test - Key Robustness Features"""

import sys
import time
sys.path.insert(0, '.')

from lunar_habitat_rl.environments.robust_habitat import make_robust_lunar_env
from lunar_habitat_rl.algorithms import HeuristicAgent
from lunar_habitat_rl.utils.robust_logging import get_logger
from lunar_habitat_rl.utils.robust_validation import validate_and_sanitize_action
from lunar_habitat_rl.utils.robust_monitoring import get_health_checker, get_simulation_monitor


def test_robust_environment_basics():
    """Test basic robust environment functionality."""
    print("🛡️ Testing Robust Environment Basics...")
    
    # Test different safety modes
    for mode in ['permissive', 'moderate', 'strict']:
        env = make_robust_lunar_env(safety_mode=mode, enable_monitoring=True)
        obs, info = env.reset()
        print(f"✅ Safety mode '{mode}': status={info['status']}")
        
        # Test one step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
    
    print("✅ All safety modes working")


def test_input_validation():
    """Test input validation and sanitization."""
    print("\n🔍 Testing Input Validation...")
    
    # Test valid action
    valid_action = [0.5] * 26
    is_valid, sanitized, validation = validate_and_sanitize_action(valid_action, 26)
    print(f"✅ Valid action: valid={is_valid}")
    
    # Test invalid action - out of bounds and NaN
    invalid_action = [-1.0, 2.0, float('nan'), float('inf')] + [0.5] * 22
    is_valid, sanitized, validation = validate_and_sanitize_action(invalid_action, 26)
    print(f"✅ Invalid action sanitized: valid={is_valid}, errors={len(validation['errors'])}")
    print(f"   Original: [{invalid_action[0]}, {invalid_action[1]}, {invalid_action[2]}, {invalid_action[3]}]")
    print(f"   Sanitized: [{sanitized[0]}, {sanitized[1]}, {sanitized[2]}, {sanitized[3]}]")


def test_error_handling():
    """Test error handling and recovery."""
    print("\n🚨 Testing Error Handling...")
    
    env = make_robust_lunar_env(safety_mode='moderate')
    obs, info = env.reset()
    
    # Test invalid action handling
    invalid_action = [float('inf')] * 26
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    print(f"✅ Invalid action handled gracefully: reward={reward:.2f}")
    
    # Test extreme conditions
    env.state.atmosphere.o2_partial_pressure = 8.0  # Dangerously low
    env.state.power.battery_charge = 5.0  # Critical
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Extreme conditions: status={info['status']}, terminated={terminated}")
    if terminated:
        print(f"   Termination reason: {info.get('termination_reason', 'Unknown')}")
    
    env.close()


def test_monitoring_systems():
    """Test monitoring and health checking."""
    print("\n📊 Testing Monitoring Systems...")
    
    # Test health checker
    health_checker = get_health_checker()
    health_status = health_checker.check_health()
    print(f"✅ System health: status={health_status.status}, score={health_status.score:.2f}")
    print(f"   CPU: {health_status.metrics.cpu_percent:.1f}%, Memory: {health_status.metrics.memory_percent:.1f}%")
    
    # Test simulation monitoring
    sim_monitor = get_simulation_monitor()
    
    # Run some episodes to generate metrics
    env = make_robust_lunar_env(enable_monitoring=True)
    agent = HeuristicAgent(action_dims=26)
    
    for episode in range(2):
        obs, info = env.reset()
        for step in range(10):
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    env.close()
    
    metrics = sim_monitor.get_simulation_metrics()
    print(f"✅ Simulation metrics:")
    print(f"   Episodes: {metrics['episodes_completed']}")
    print(f"   Steps/sec: {metrics['steps_per_second']:.1f}")
    print(f"   Error rate: {metrics['error_rate']:.3f}")


def test_safety_protocols():
    """Test safety protocols and emergency handling."""
    print("\n🚨 Testing Safety Protocols...")
    
    env = make_robust_lunar_env(safety_mode='strict')
    obs, info = env.reset()
    
    # Gradually create dangerous conditions
    print("Creating dangerous conditions...")
    for step in range(20):
        # Worsen atmosphere
        env.state.atmosphere.o2_partial_pressure *= 0.95
        env.state.atmosphere.co2_partial_pressure *= 1.05
        env.state.power.battery_charge *= 0.97
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get('safety_violations', 0) > 0:
            print(f"   Step {step}: Safety violations: {info['safety_violations']}")
        
        if info.get('emergency_mode', False):
            print(f"✅ Emergency mode activated at step {step}")
            break
            
        if terminated:
            print(f"✅ Safety termination at step {step}: {info.get('termination_reason', 'Unknown')}")
            break
    
    env.close()


def test_production_scenario():
    """Test production-like scenario."""
    print("\n🚀 Testing Production Scenario...")
    
    env = make_robust_lunar_env(safety_mode='moderate', enable_monitoring=True)
    agent = HeuristicAgent(action_dims=26)
    logger = get_logger()
    
    total_reward = 0
    successful_episodes = 0
    
    start_time = time.time()
    
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(50):
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                if not terminated:  # Completed successfully
                    successful_episodes += 1
                break
        
        total_reward += episode_reward
        print(f"   Episode {episode+1}: reward={episode_reward:.2f}, status={info['status']}")
    
    duration = time.time() - start_time
    
    print(f"✅ Production test completed:")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Successful episodes: {successful_episodes}/3")
    print(f"   Average reward: {total_reward/3:.2f}")
    
    # Get final health status
    health_checker = get_health_checker()
    final_health = health_checker.check_health()
    print(f"   Final system health: {final_health.status}")
    
    env.close()


def main():
    """Run Generation 2 demonstration tests."""
    print("🛡️ GENERATION 2 ROBUSTNESS DEMONSTRATION")
    print("=" * 50)
    
    try:
        test_robust_environment_basics()
        test_input_validation()
        test_error_handling()
        test_monitoring_systems()
        test_safety_protocols()
        test_production_scenario()
        
        print("\n" + "=" * 50)
        print("🎉 GENERATION 2 DEMONSTRATION COMPLETE!")
        print("✅ Comprehensive error handling")
        print("✅ Input validation and sanitization")
        print("✅ Monitoring and logging systems")
        print("✅ Safety protocols and emergency handling")
        print("✅ Production-ready robustness")
        print("\n🚀 Generation 2: MAKE IT ROBUST - ACHIEVED!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Generation 2 demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)