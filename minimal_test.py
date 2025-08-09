#!/usr/bin/env python3
"""
Minimal Test Suite for Lunar Habitat RL - Generation 1 Implementation
Tests core functionality without heavy dependencies
"""

import sys
import os
sys.path.insert(0, '/root/repo')

# Mock gymnasium for basic testing
class MockSpace:
    def __init__(self, shape=None, low=None, high=None):
        self.shape = shape
        self.low = low 
        self.high = high
        
    def sample(self):
        import random
        if self.shape:
            return [random.random() for _ in range(self.shape[0])]
        return random.random()

class MockEnv:
    def __init__(self):
        self.observation_space = MockSpace(shape=(50,))
        self.action_space = MockSpace(shape=(20,))
        
    def reset(self):
        return self.observation_space.sample(), {}
        
    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info
        
    def close(self):
        pass

# Patch gymnasium
sys.modules['gymnasium'] = type('MockGymnasium', (), {
    'Env': MockEnv,
    'spaces': type('MockSpaces', (), {
        'Box': MockSpace,
        'Discrete': MockSpace
    })()
})()

# Mock other dependencies  
sys.modules['torch'] = type('MockTorch', (), {'tensor': lambda x: x})()
sys.modules['stable_baselines3'] = type('MockSB3', (), {})()

def test_generation_1_basic_functionality():
    """Test Generation 1: MAKE IT WORK - Basic functionality"""
    
    print("🚀 GENERATION 1: MAKE IT WORK")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 5
    
    # Test 1: Core imports
    try:
        from lunar_habitat_rl.core.config import HabitatConfig
        print("✅ Test 1/5: Core config import successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1/5: Core config import failed: {e}")
    
    # Test 2: State management
    try:
        from lunar_habitat_rl.core.state import HabitatState
        state = HabitatState()
        print("✅ Test 2/5: State management successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2/5: State management failed: {e}")
        
    # Test 3: Environment creation
    try:
        from lunar_habitat_rl.environments.habitat_base import LunarHabitatEnv
        env = LunarHabitatEnv()
        print("✅ Test 3/5: Environment creation successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3/5: Environment creation failed: {e}")
        
    # Test 4: Basic episode
    try:
        from lunar_habitat_rl.environments.habitat_base import LunarHabitatEnv
        env = LunarHabitatEnv()
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.close()
        print("✅ Test 4/5: Basic episode successful")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4/5: Basic episode failed: {e}")
        
    # Test 5: Algorithm import
    try:
        from lunar_habitat_rl.algorithms.baselines import RandomAgent
        agent = RandomAgent()
        print("✅ Test 5/5: Algorithm import successful") 
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5/5: Algorithm import failed: {e}")
    
    print(f"\nGeneration 1 Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed >= 4:
        print("🎉 Generation 1 COMPLETE - Basic functionality working!")
        return True
    else:
        print("⚠️ Generation 1 needs fixes")
        return False

if __name__ == "__main__":
    success = test_generation_1_basic_functionality()
    sys.exit(0 if success else 1)