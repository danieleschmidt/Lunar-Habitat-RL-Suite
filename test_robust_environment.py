#!/usr/bin/env python3
"""Test robust environment with error handling and validation."""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

def test_environment_validation():
    """Test environment validation and error handling."""
    print("🧪 Testing Environment Validation...")
    
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        from lunar_habitat_rl.utils.exceptions import EnvironmentError, ValidationError, SafetyError
        
        # Test valid environment creation
        env = LunarHabitatEnv(
            crew_size=4,
            difficulty="nominal",
            physics_enabled=False
        )
        print("✅ Valid environment created successfully")
        
        # Test invalid crew size
        try:
            LunarHabitatEnv(crew_size=15, physics_enabled=False)
            print("❌ Should have failed with invalid crew size")
            return False
        except (EnvironmentError, ValidationError):
            print("✅ Correctly rejected invalid crew size")
        
        # Test invalid difficulty
        try:
            LunarHabitatEnv(crew_size=4, difficulty="impossible", physics_enabled=False)
            print("❌ Should have failed with invalid difficulty")
            return False
        except EnvironmentError:
            print("✅ Correctly rejected invalid difficulty")
        
        # Test invalid reward config
        try:
            LunarHabitatEnv(crew_size=4, reward_config="invalid_config", physics_enabled=False)
            print("❌ Should have failed with invalid reward config")
            return False
        except EnvironmentError:
            print("✅ Correctly rejected invalid reward config")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_validation():
    """Test action validation."""
    print("\n🎯 Testing Action Validation...")
    
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        from lunar_habitat_rl.utils.exceptions import ValidationError, SafetyError
        
        env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
        obs, info = env.reset(seed=42)
        
        # Test valid action
        valid_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(valid_action)
        print("✅ Valid action processed successfully")
        
        # Test invalid action shape
        try:
            invalid_action = np.array([0.5, 0.5])  # Wrong shape
            env.step(invalid_action)
            print("❌ Should have failed with wrong action shape")
            return False
        except Exception:
            print("✅ Correctly rejected wrong action shape")
        
        # Test action with NaN values
        try:
            nan_action = valid_action.copy()
            nan_action[0] = np.nan
            env.step(nan_action)
            print("❌ Should have failed with NaN action")
            return False
        except Exception:
            print("✅ Correctly rejected NaN action values")
        
        # Test action with infinite values  
        try:
            inf_action = valid_action.copy()
            inf_action[1] = np.inf
            env.step(inf_action)
            print("❌ Should have failed with infinite action")
            return False
        except Exception:
            print("✅ Correctly rejected infinite action values")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Action validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_monitoring():
    """Test state monitoring and safety checks."""
    print("\n🔍 Testing State Monitoring...")
    
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        from lunar_habitat_rl.algorithms.baselines import HeuristicAgent
        
        env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
        agent = HeuristicAgent(env.action_space)
        
        obs, info = env.reset(seed=42)
        
        # Run a few steps and monitor state
        for i in range(10):
            action = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # Check state is finite
            assert np.all(np.isfinite(obs)), f"Non-finite observation at step {i}"
            
            # Check critical parameters are reasonable
            if len(obs) >= 7:
                o2_pressure = obs[0]
                co2_pressure = obs[1] 
                temperature = obs[5]
                
                assert 0.0 <= o2_pressure <= 50.0, f"O2 pressure out of range: {o2_pressure}"
                assert 0.0 <= co2_pressure <= 10.0, f"CO2 pressure out of range: {co2_pressure}"
                assert -50.0 <= temperature <= 50.0, f"Temperature out of range: {temperature}"
            
            if done or truncated:
                break
        
        print(f"✅ State monitoring passed for {i+1} steps")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ State monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_recovery():
    """Test error recovery mechanisms.""" 
    print("\n🚑 Testing Error Recovery...")
    
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        
        env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
        
        # Test reset after error
        obs, info = env.reset(seed=42)
        
        # Simulate some steps
        for i in range(5):
            action = env.action_space.sample()
            try:
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    # Test environment can be reset after termination
                    obs, info = env.reset()
                    print("✅ Environment reset successfully after termination")
                    break
            except Exception as e:
                print(f"Step {i} failed with error: {e}")
                # Try to reset environment
                obs, info = env.reset()
                print("✅ Environment recovered after error")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🛡️  Lunar Habitat RL - Robust Environment Tests")
    print("=" * 55)
    
    tests = [
        test_environment_validation,
        test_action_validation,
        test_state_monitoring,
        test_error_recovery
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 55)
    print("📊 Robust Environment Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All robust environment tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Environment robustness needs improvement.")
        return 1

if __name__ == "__main__":
    sys.exit(main())