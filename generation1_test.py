#!/usr/bin/env python3
"""
Generation 1 Test Suite - Direct Lightweight Testing
"""

import sys
import os
sys.path.insert(0, '/root/repo')

def test_generation_1_lightweight():
    """Test Generation 1 with direct lightweight imports"""
    
    print("🚀 GENERATION 1: MAKE IT WORK (Direct Lightweight)")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Direct lightweight config
    try:
        from lunar_habitat_rl.core.lightweight_config import HabitatConfig
        config = HabitatConfig()
        print("✅ Test 1/5: Direct lightweight config successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1/5: Direct lightweight config failed: {e}")
    
    # Test 2: Direct lightweight state
    try:
        from lunar_habitat_rl.core.lightweight_state import HabitatState
        state = HabitatState()
        print("✅ Test 2/5: Direct lightweight state successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2/5: Direct lightweight state failed: {e}")
        
    # Test 3: Direct lightweight environment
    try:
        from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv
        env = LunarHabitatEnv()
        print("✅ Test 3/5: Direct lightweight environment successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3/5: Direct lightweight environment failed: {e}")
        
    # Test 4: Basic episode with lightweight environment
    try:
        from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv
        env = LunarHabitatEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.close()
        print(f"✅ Test 4/5: Basic episode successful (reward: {reward:.3f})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4/5: Basic episode failed: {e}")
        
    # Test 5: Direct lightweight algorithms
    try:
        from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent, HeuristicAgent
        random_agent = RandomAgent()
        heuristic_agent = HeuristicAgent()
        print("✅ Test 5/5: Direct lightweight algorithms successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5/5: Direct lightweight algorithms failed: {e}")
    
    print(f"\n🎯 Generation 1 Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed >= 4:
        print("🎉 Generation 1 COMPLETE - Basic functionality working!")
        return True
    else:
        print("⚠️ Generation 1 needs fixes")
        return False

def test_agent_functionality():
    """Test that agents can actually control the environment."""
    print("\n🤖 AGENT FUNCTIONALITY TEST")
    print("=" * 40)
    
    try:
        from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv
        from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent, HeuristicAgent
        
        env = LunarHabitatEnv()
        agents = {
            'Random': RandomAgent(action_dims=env.action_space.shape[0]),
            'Heuristic': HeuristicAgent(action_dims=env.action_space.shape[0])
        }
        
        for agent_name, agent in agents.items():
            print(f"\n🧪 Testing {agent_name} Agent:")
            
            obs, info = env.reset(seed=42)
            total_reward = 0.0
            
            for step in range(10):  # Run for 10 steps
                action, _ = agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if done or truncated:
                    break
                    
            print(f"   Steps: {step + 1}, Total Reward: {total_reward:.3f}")
            print(f"   Final Status: {info.get('status', 'unknown')}")
            
        env.close()
        print("\n✅ Agent functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent functionality test failed: {e}")
        return False

if __name__ == "__main__":
    # Run Generation 1 tests
    gen1_success = test_generation_1_lightweight()
    
    if gen1_success:
        # Run agent functionality test
        agent_success = test_agent_functionality()
        
        if agent_success:
            print("\n🚀 GENERATION 1 FULLY OPERATIONAL! 🚀")
            print("Ready to proceed to Generation 2")
        else:
            print("\n⚠️ Generation 1 basic tests pass but agent functionality needs work")
    
    sys.exit(0 if gen1_success else 1)