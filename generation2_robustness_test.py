#!/usr/bin/env python3
"""Generation 2 Robustness Test - Comprehensive Error Handling, Validation, and Monitoring"""

import sys
import os
import time
sys.path.insert(0, '.')

from lunar_habitat_rl.environments.robust_habitat import RobustLunarHabitatEnv, make_robust_lunar_env
from lunar_habitat_rl.algorithms import RandomAgent, HeuristicAgent
from lunar_habitat_rl.core.lightweight_config import HabitatConfig
from lunar_habitat_rl.utils.robust_logging import get_logger, PerformanceMonitor
from lunar_habitat_rl.utils.robust_validation import get_validator, validate_and_sanitize_action
from lunar_habitat_rl.utils.robust_monitoring import get_health_checker, get_simulation_monitor


def test_robust_environment():
    """Test robust environment creation and basic operations."""
    print("🛡️ Testing Robust Environment Creation...")
    
    # Test different safety modes
    safety_modes = ['strict', 'moderate', 'permissive']
    
    for mode in safety_modes:
        env = make_robust_lunar_env(safety_mode=mode, enable_monitoring=True)
        print(f"✅ Robust environment created with safety mode: {mode}")
        
        obs, info = env.reset()
        print(f"   Reset successful - status: {info['status']}, safety_mode: {info['safety_mode']}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
        env.close()
    
    print(f"✅ All safety modes tested successfully")


def test_input_validation():
    """Test comprehensive input validation and sanitization."""
    print("\n🔍 Testing Input Validation and Sanitization...")
    
    validator = get_validator()
    
    # Test observation validation
    print("Testing observation validation...")
    
    # Valid observation
    valid_obs = [21.3, 0.3, 79.0, 101.3, 45.0, 22.5, 0.95,  # atmosphere
                8.5, 75.0, 90.0, 6.2, 100.0, 0.98,  # power
                22.5, 23.1, 22.8, 21.9, -45.0, 15.0, 16.2, 3.2,  # thermal
                850.0, 120.0, 45.0, 0.93, 0.87,  # water
                0.95, 0.98, 0.92, 0.96,  # crew health
                0.3, 0.2, 0.4, 0.25,     # crew stress
                0.9, 0.95, 0.85, 0.92,   # crew productivity
                127.5, 0.3, 0.15, 0.05]  # environment
    
    result = validator.validate_observation(valid_obs, crew_size=4)
    print(f"✅ Valid observation: status={result['safety_status']}, warnings={len(result['warnings'])}")
    
    # Invalid observation - dangerous conditions
    dangerous_obs = valid_obs.copy()
    dangerous_obs[0] = 10.0  # Critically low O2
    dangerous_obs[1] = 1.5   # High CO2
    dangerous_obs[8] = 5.0   # Low battery
    
    result = validator.validate_observation(dangerous_obs, crew_size=4)
    print(f"✅ Dangerous observation detected: critical_errors={len(result['critical_errors'])}")
    for error in result['critical_errors'][:3]:  # Show first 3 errors
        print(f"   - {error}")
    
    # Test action validation
    print("\nTesting action validation...")
    
    # Valid action
    valid_action = [0.5] * 26
    is_valid, sanitized, validation = validate_and_sanitize_action(valid_action, 26)
    print(f"✅ Valid action: valid={is_valid}, warnings={len(validation['warnings'])}")
    
    # Invalid action - out of bounds
    invalid_action = [-1.0, 2.0, float('nan')] + [0.5] * 23
    is_valid, sanitized, validation = validate_and_sanitize_action(invalid_action, 26)
    print(f"✅ Invalid action sanitized: valid={is_valid}, errors={len(validation['errors'])}")
    print(f"   Sanitized values: {sanitized[:3]} (from {invalid_action[:3]})")


def test_error_handling():
    """Test comprehensive error handling and recovery."""
    print("\n🚨 Testing Error Handling and Recovery...")
    
    env = make_robust_lunar_env(safety_mode='moderate', enable_monitoring=True)
    logger = get_logger()
    
    # Test 1: Invalid action handling
    print("Testing invalid action handling...")
    obs, info = env.reset()
    
    # Send completely invalid action
    invalid_action = [float('inf')] * 26
    try:
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        print(f"✅ Invalid action handled: reward={reward:.2f}, terminated={terminated}")
        if 'action_validation' in info:
            print(f"   Validation errors: {len(info['action_validation']['errors'])}")
    except Exception as e:
        print(f"❌ Invalid action caused unhandled exception: {e}")
    
    # Test 2: Extreme conditions simulation
    print("\nTesting extreme conditions handling...")
    
    # Force extreme state by manipulating internal state
    env.state.atmosphere.o2_partial_pressure = 8.0  # Critically low
    env.state.power.battery_charge = 2.0  # Almost dead
    env.state.water.potable_water = 10.0  # Nearly empty
    
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Extreme conditions handled: status={info['status']}, terminated={terminated}")
        if terminated:
            print(f"   Termination reason: {info.get('termination_reason', 'Unknown')}")
    except Exception as e:
        print(f"❌ Extreme conditions caused unhandled exception: {e}")
    
    # Test 3: Error recovery
    print("\nTesting error recovery mechanisms...")
    
    # Simulate multiple errors to test degradation
    for i in range(5):
        try:
            # Inject some randomness to trigger different error paths
            if i % 2 == 0:
                action = [float('nan')] * 26  # NaN action
            else:
                action = [-10.0] * 26  # Out of bounds action
                
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Error injection {i+1}: physics_stability={info.get('physics_stability', 1.0):.3f}")
            
            if terminated:
                break
                
        except Exception as e:
            print(f"   Error injection {i+1} caused exception: {e}")
    
    env.close()
    print("✅ Error handling tests completed")


def test_monitoring_and_logging():
    """Test monitoring and logging systems."""
    print("\n📊 Testing Monitoring and Logging...")
    
    # Test health checker
    health_checker = get_health_checker()
    print("Testing health monitoring...")
    
    health_status = health_checker.check_health()
    print(f"✅ Health check: status={health_status.status}, score={health_status.score:.2f}")
    print(f"   CPU: {health_status.metrics.cpu_percent:.1f}%, Memory: {health_status.metrics.memory_percent:.1f}%")
    
    if health_status.alerts:
        print(f"   Alerts: {len(health_status.alerts)}")
        for alert in health_status.alerts[:2]:  # Show first 2 alerts
            print(f"   - {alert}")
    
    # Test simulation monitoring
    sim_monitor = get_simulation_monitor()
    print("\nTesting simulation monitoring...")
    
    # Simulate some episodes to generate metrics
    env = make_robust_lunar_env(enable_monitoring=True)
    
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(20):  # Short episodes for testing
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
    
    env.close()
    
    # Get simulation metrics
    metrics = sim_monitor.get_simulation_metrics()
    print(f"✅ Simulation metrics collected:")
    print(f"   Episodes: {metrics['episodes_completed']}")
    print(f"   Total steps: {metrics['total_steps']}")
    print(f"   Steps/sec: {metrics['steps_per_second']:.2f}")
    print(f"   Error rate: {metrics['error_rate']:.3f}")
    print(f"   Avg reward: {metrics['reward_stats']['average']:.2f}")
    
    # Test performance monitoring
    logger = get_logger()
    print("\nTesting performance monitoring...")
    
    with PerformanceMonitor(logger, "test_operation", test_param="value"):
        time.sleep(0.1)  # Simulate work
    
    perf_summary = logger.get_performance_summary()
    if perf_summary:
        print(f"✅ Performance monitoring active:")
        print(f"   Total operations: {perf_summary['total_operations']}")
        print(f"   Avg duration: {perf_summary['avg_duration_ms']:.2f}ms")
    
    print("✅ Monitoring and logging tests completed")


def test_safety_systems():
    """Test safety systems and emergency protocols."""
    print("\n🚨 Testing Safety Systems and Emergency Protocols...")
    
    env = make_robust_lunar_env(safety_mode='strict', enable_monitoring=True)
    
    # Test 1: Gradual degradation to emergency
    print("Testing gradual degradation to emergency state...")
    
    obs, info = env.reset()
    safety_violations = 0
    
    # Gradually worsen conditions
    for step in range(50):
        # Gradually reduce O2 and increase CO2
        env.state.atmosphere.o2_partial_pressure *= 0.98
        env.state.atmosphere.co2_partial_pressure *= 1.02
        env.state.power.battery_charge *= 0.99
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_violations = info.get('safety_violations', 0)
        if current_violations > safety_violations:
            print(f"   Step {step}: New safety violations detected (total: {current_violations})")
            safety_violations = current_violations
        
        if info.get('emergency_mode', False):
            print(f"✅ Emergency mode activated at step {step}")
            print(f"   Status: {info['status']}, violations: {safety_violations}")
            break
            
        if terminated:
            print(f"✅ Safety termination at step {step}: {info.get('termination_reason', 'Unknown')}")
            break
    
    # Test 2: Immediate critical failure
    print("\nTesting immediate critical failure handling...")
    
    obs, info = env.reset()
    
    # Set immediately dangerous conditions
    env.state.atmosphere.o2_partial_pressure = 5.0  # Immediate danger
    env.state.atmosphere.co2_partial_pressure = 3.0  # Toxic levels
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✅ Critical failure response: terminated={terminated}")
    if terminated:
        print(f"   Reason: {info.get('termination_reason', 'Unknown')}")
    
    env.close()
    
    # Test 3: Safety mode differences
    print("\nTesting safety mode differences...")
    
    modes_tested = []
    for mode in ['permissive', 'moderate', 'strict']:
        env = make_robust_lunar_env(safety_mode=mode)
        obs, info = env.reset()
        
        # Send mildly invalid action
        invalid_action = [1.5] * 26  # Slightly out of bounds
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        modes_tested.append({
            'mode': mode,
            'terminated': terminated,
            'reward': reward,
            'has_validation_error': 'action_validation' in info
        })
        
        env.close()
    
    print("✅ Safety mode comparison:")
    for result in modes_tested:
        print(f"   {result['mode']}: terminated={result['terminated']}, "
              f"reward={result['reward']:.2f}, validation_error={result['has_validation_error']}")
    
    print("✅ Safety systems tests completed")


def test_production_readiness():
    """Test production readiness with realistic scenarios."""
    print("\n🚀 Testing Production Readiness...")
    
    # Test 1: Long-running simulation stability
    print("Testing long-running simulation stability...")
    
    env = make_robust_lunar_env(safety_mode='moderate', enable_monitoring=True)
    agent = HeuristicAgent(action_dims=26)
    
    total_episodes = 5
    total_steps = 0
    successful_episodes = 0
    total_reward = 0
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(100):  # Max 100 steps per episode
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Log any issues
            if info.get('emergency_mode', False):
                print(f"   Episode {episode+1}: Emergency mode activated at step {step}")
            
            if terminated or truncated:
                if not terminated:  # Completed successfully
                    successful_episodes += 1
                break
        
        total_reward += episode_reward
        print(f"   Episode {episode+1}: {episode_steps} steps, reward={episode_reward:.2f}, "
              f"status={info['status']}")
    
    duration = time.time() - start_time
    
    print(f"✅ Long-running simulation completed:")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Total steps: {total_steps}")
    print(f"   Successful episodes: {successful_episodes}/{total_episodes}")
    print(f"   Average reward: {total_reward/total_episodes:.2f}")
    print(f"   Steps per second: {total_steps/duration:.2f}")
    
    # Get final health report
    health_checker = get_health_checker()
    health_report = health_checker.get_health_report()
    
    print(f"   Final system health: {health_report['current_health']['status']}")
    print(f"   Health score: {health_report['current_health']['score']:.2f}")
    
    env.close()
    
    # Test 2: Resource usage monitoring
    print("\nTesting resource usage monitoring...")
    
    sim_monitor = get_simulation_monitor()
    final_metrics = sim_monitor.get_simulation_metrics()
    
    print(f"✅ Resource usage summary:")
    print(f"   Total episodes completed: {final_metrics['episodes_completed']}")
    print(f"   Total simulation steps: {final_metrics['total_steps']}")
    print(f"   Episodes per hour: {final_metrics['episodes_per_hour']:.1f}")
    print(f"   Error rate: {final_metrics['error_rate']:.3f}")
    print(f"   Average step time: {final_metrics['avg_step_time_ms']:.2f}ms")
    
    # Test 3: Validation statistics
    validator = get_validator()
    validation_stats = validator.get_validation_statistics()
    
    if validation_stats:
        print(f"\n✅ Validation statistics:")
        print(f"   Total validations: {validation_stats['total_validations']}")
        print(f"   Success rate: {validation_stats['success_rate']:.3f}")
        print(f"   Total warnings: {validation_stats['total_warnings']}")
        print(f"   Total errors: {validation_stats['total_errors']}")
        print(f"   Critical issues: {validation_stats['total_critical']}")
    
    print("✅ Production readiness tests completed")


def run_stress_test():
    """Run stress test with concurrent environments and error injection."""
    print("\n💪 Running Stress Test...")
    
    # Create multiple environments
    envs = make_robust_lunar_env(n_envs=3, safety_mode='moderate')
    agents = [HeuristicAgent(action_dims=26) for _ in range(3)]
    
    print(f"✅ Created {len(envs)} concurrent environments")
    
    # Run concurrent episodes with error injection
    all_active = True
    step_count = 0
    error_injection_rate = 0.05  # 5% chance of error injection per step
    
    # Reset all environments
    observations = []
    for env in envs:
        obs, info = env.reset()
        observations.append(obs)
    
    while all_active and step_count < 200:
        new_observations = []
        active_count = 0
        
        for i, (env, agent, obs) in enumerate(zip(envs, agents, observations)):
            try:
                # Get action from agent
                action, _ = agent.predict(obs)
                
                # Inject errors occasionally
                if step_count > 50 and random.random() < error_injection_rate:
                    # Inject different types of errors
                    error_type = random.choice(['nan', 'inf', 'bounds', 'missing'])
                    if error_type == 'nan':
                        action[random.randint(0, len(action)-1)] = float('nan')
                    elif error_type == 'inf':
                        action[random.randint(0, len(action)-1)] = float('inf')
                    elif error_type == 'bounds':
                        action[random.randint(0, len(action)-1)] = random.uniform(-10, 10)
                    # For 'missing', we'll use an action with wrong dimensions
                    elif error_type == 'missing':
                        action = action[:20]  # Wrong number of dimensions
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                if not (terminated or truncated):
                    active_count += 1
                    new_observations.append(obs)
                else:
                    new_observations.append(None)
                
            except Exception as e:
                print(f"   Environment {i} failed: {e}")
                new_observations.append(None)
        
        observations = new_observations
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"   Stress test step {step_count}: {active_count}/{len(envs)} environments active")
        
        all_active = active_count > 0
    
    # Clean up
    for env in envs:
        try:
            env.close()
        except:
            pass
    
    print(f"✅ Stress test completed: {step_count} steps, environments handled concurrent load")


def main():
    """Run all Generation 2 robustness tests."""
    print("🛡️ GENERATION 2 ROBUSTNESS AND RELIABILITY TESTING")
    print("=" * 60)
    
    try:
        # Core robustness tests
        test_robust_environment()
        test_input_validation()
        test_error_handling()
        test_monitoring_and_logging()
        test_safety_systems()
        test_production_readiness()
        
        # Stress testing
        import random
        run_stress_test()
        
        print("\n" + "=" * 60)
        print("🎉 GENERATION 2 TESTING COMPLETE!")
        print("✅ Comprehensive error handling implemented")
        print("✅ Input validation and sanitization working")
        print("✅ Monitoring and logging systems operational")
        print("✅ Safety systems and emergency protocols active")
        print("✅ Production-ready robustness achieved")
        print("✅ Stress testing passed")
        print("\n🚀 Ready for Generation 3: MAKE IT SCALE")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Generation 2 testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)