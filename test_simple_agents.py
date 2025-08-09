#!/usr/bin/env python3
"""Test just the simple agents without torch dependencies."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_simple_agents():
    """Test only the torch-free agents."""
    try:
        from lunar_habitat_rl import LunarHabitatEnv
        
        # Direct import of simple agents
        from lunar_habitat_rl.algorithms.baselines import RandomAgent, HeuristicAgent
        
        print("✅ Imported simple agents successfully")
        
        env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
        print("✅ Created test environment")
        
        # Test random agent
        random_agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        
        total_reward = 0.0
        for i in range(5):
            action = random_agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            print(f"   Random agent step {i+1}: reward={reward:.3f}")
            if done or truncated:
                break
        
        print(f"✅ Random agent test passed, total reward: {total_reward:.3f}")
        
        # Test heuristic agent
        heuristic_agent = HeuristicAgent(env.action_space)
        obs, _ = env.reset(seed=42)
        
        total_reward = 0.0
        for i in range(5):
            action = heuristic_agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            print(f"   Heuristic agent step {i+1}: reward={reward:.3f}")
            if done or truncated:
                break
                
        print(f"✅ Heuristic agent test passed, total reward: {total_reward:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Simple agents test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🌙 Lunar Habitat RL - Simple Agents Test")
    print("=" * 45)
    
    if test_simple_agents():
        print("\n🎉 Simple agents test passed!")
        return 0
    else:
        print("\n⚠️  Simple agents test failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())