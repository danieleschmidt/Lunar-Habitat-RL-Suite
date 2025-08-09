#!/usr/bin/env python3
"""Test environment creation and basic functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_environment():
    """Test basic environment functionality."""
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        print("✅ Imported LunarHabitatEnv")
        
        # Create environment with physics disabled for testing
        env = LunarHabitatEnv(
            crew_size=4,
            difficulty="nominal",
            scenario="nominal_operations", 
            physics_enabled=False
        )
        print("✅ Created environment successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.3f}, done={terminated or truncated}")
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✅ Environment test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agents():
    """Test baseline agents.""" 
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        from lunar_habitat_rl.algorithms.baselines import RandomAgent, HeuristicAgent
        
        env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
        print("✅ Created test environment for agents")
        
        # Test random agent
        random_agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        
        for i in range(3):
            action = random_agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        print("✅ Random agent test passed")
        
        # Test heuristic agent
        heuristic_agent = HeuristicAgent(env.action_space)
        obs, _ = env.reset(seed=42)
        
        for i in range(3):
            action = heuristic_agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
                
        print("✅ Heuristic agent test passed")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🌙 Lunar Habitat RL - Environment Tests")
    print("=" * 45)
    
    success = True
    
    print("\n🏠 Testing Environment...")
    if not test_environment():
        success = False
        
    print("\n🤖 Testing Agents...")
    if not test_agents():
        success = False
    
    print("\n" + "=" * 45)
    if success:
        print("🎉 All environment tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())