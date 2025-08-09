#!/usr/bin/env python3
"""Test basic functionality of the lunar habitat RL environment."""

import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from lunar_habitat_rl import LunarHabitatEnv, make_lunar_env
    from lunar_habitat_rl.algorithms.baselines import RandomAgent, HeuristicAgent
    print("✅ Successfully imported lunar_habitat_rl modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def test_single_environment():
    """Test basic single environment functionality."""
    print("\n🧪 Testing Single Environment...")
    
    try:
        # Create environment
        env = LunarHabitatEnv(
            crew_size=4,
            difficulty="nominal", 
            scenario="nominal_operations",
            physics_enabled=False  # Disable physics for faster testing
        )
        
        print(f"   Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"   Reset successful, observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step successful")
        print(f"   Reward: {reward:.3f}")
        print(f"   Terminated: {terminated}, Truncated: {truncated}")
        print(f"   Mission time: {info.get('mission_time', 0):.3f} sols")
        
        # Test multiple steps
        for i in range(10):
            if terminated or truncated:
                obs, info = env.reset()
                terminated = truncated = False
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Multi-step test successful")
        
        env.close()
        print("✅ Single environment test passed")
        return True
        
    except Exception as e:
        print(f"❌ Single environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agents():
    """Test baseline agents."""
    print("\n🤖 Testing Baseline Agents...")
    
    try:
        env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
        
        # Test Random Agent
        random_agent = RandomAgent(env.action_space, seed=42)
        obs, _ = env.reset(seed=42)
        
        for i in range(5):
            action = random_agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print("   Random agent test passed")
        
        # Test Heuristic Agent 
        heuristic_agent = HeuristicAgent(env.action_space)
        obs, _ = env.reset(seed=42)
        
        for i in range(5):
            action = heuristic_agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
        print("   Heuristic agent test passed")
        
        env.close()
        print("✅ Agent tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vectorized_environment():
    """Test vectorized environment."""
    print("\n🔀 Testing Vectorized Environment...")
    
    try:
        # Create vectorized environment
        vec_env = make_lunar_env(
            n_envs=2,
            crew_size=2, 
            parallel=False,  # Use sequential for testing
            seed=42
        )
        
        print(f"   Vectorized environment created with {vec_env.num_envs} environments")
        
        # Test reset
        observations = vec_env.reset()
        if isinstance(observations, tuple):
            observations = observations[0]  # Handle new gymnasium format
            
        print(f"   Reset successful, observations shape: {observations.shape}")
        
        # Test step
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        actions = np.array(actions)
        
        results = vec_env.step(actions)
        if len(results) == 5:  # New format
            observations, rewards, dones, truncateds, infos = results
        else:  # Old format
            observations, rewards, dones, infos = results
            truncateds = dones
        
        print(f"   Step successful")
        print(f"   Rewards: {rewards}")
        print(f"   Dones: {dones}")
        
        vec_env.close()
        print("✅ Vectorized environment test passed")
        return True
        
    except Exception as e:
        print(f"❌ Vectorized environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_simulation():
    """Test complete episode simulation."""
    print("\n🎬 Testing Episode Simulation...")
    
    try:
        env = LunarHabitatEnv(
            crew_size=2,
            physics_enabled=False,
            difficulty="easy"
        )
        
        agent = HeuristicAgent(env.action_space)
        
        obs, info = env.reset(seed=42)
        total_reward = 0.0
        steps = 0
        max_steps = 100  # Limit for testing
        
        while steps < max_steps:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"   Episode completed in {steps} steps")
        print(f"   Total reward: {total_reward:.3f}")
        print(f"   Survival time: {info.get('survival_time', 0):.3f} sols")
        print(f"   Final crew health: {np.mean(env.habitat_state.crew.health):.3f}")
        
        env.close()
        print("✅ Episode simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Episode simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🌙 Lunar Habitat RL - Basic Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_single_environment,
        test_agents,
        test_vectorized_environment,
        test_episode_simulation
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Basic functionality is working.")
        return 0
    else:
        print("⚠️  Some tests failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())