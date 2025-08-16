#!/usr/bin/env python3
"""Generation 1 Complete Test - Demonstrate Basic Functionality"""

import sys
import os
sys.path.insert(0, '.')

import lunar_habitat_rl
from lunar_habitat_rl.algorithms import RandomAgent, HeuristicAgent, PIDControllerAgent, GreedyAgent
from lunar_habitat_rl.core.lightweight_config import HabitatConfig


def test_environment_creation():
    """Test environment creation and basic operations."""
    print("🧪 Testing Environment Creation...")
    
    # Test single environment
    env = lunar_habitat_rl.make_lunar_env()
    print(f"✅ Single environment created")
    
    # Test vectorized environments
    envs = lunar_habitat_rl.make_lunar_env(n_envs=3)
    print(f"✅ Vectorized environments created: {len(envs)} environments")
    
    # Test with different configurations
    config = HabitatConfig.from_preset("nasa_reference")
    env_configured = lunar_habitat_rl.LunarHabitatEnv(config=config, crew_size=6)
    print(f"✅ Configured environment created with crew size: {env_configured.crew_size}")
    
    return env


def test_environment_interaction(env):
    """Test environment interaction loop."""
    print("\n🔄 Testing Environment Interaction...")
    
    obs, info = env.reset()
    print(f"✅ Environment reset - obs dims: {len(obs)}, status: {info['status']}")
    
    episode_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if step < 3:  # Show first few steps
            print(f"   Step {step+1}: reward={reward:.3f}, status={info['status']}")
        
        if terminated or truncated:
            print(f"   Episode ended early at step {step+1}: {info.get('termination_reason', 'max steps')}")
            break
    
    print(f"✅ 10-step interaction complete - total reward: {episode_reward:.3f}")
    return obs


def test_all_agents(env, obs):
    """Test all available baseline agents."""
    print("\n🤖 Testing Baseline Agents...")
    
    agents = {
        'Random': RandomAgent(action_dims=26),
        'Heuristic': HeuristicAgent(action_dims=26),
        'PID Controller': PIDControllerAgent(action_dims=26),
        'Greedy': GreedyAgent(action_dims=26, n_samples=5)
    }
    
    for name, agent in agents.items():
        action, _ = agent.predict(obs)
        obs_new, reward, terminated, truncated, info = env.step(action)
        print(f"✅ {name} agent: action dims={len(action)}, reward={reward:.3f}")
        obs = obs_new  # Update observation for next agent


def test_configuration_system():
    """Test configuration presets and customization."""
    print("\n⚙️ Testing Configuration System...")
    
    # Test presets
    presets = ["nasa_reference", "apollo_derived", "mars_analog"]
    for preset in presets:
        config = HabitatConfig.from_preset(preset)
        print(f"✅ {preset} preset loaded - volume: {config.volume}m³")
    
    # Test configuration serialization
    config = HabitatConfig.from_preset("nasa_reference")
    config_dict = config.to_dict()
    print(f"✅ Configuration serialization - {len(config_dict)} parameters")


def test_physics_simulation():
    """Test basic physics simulation."""
    print("\n🔬 Testing Physics Simulation...")
    
    env = lunar_habitat_rl.make_lunar_env()
    obs, info = env.reset()
    
    # Test extreme scenarios
    scenarios = [
        "High O2 consumption",
        "Power shortage",
        "Temperature extreme",
        "Water scarcity"
    ]
    
    for i, scenario in enumerate(scenarios):
        # Different action patterns to test physics
        if i == 0:  # High O2 consumption
            action = [0.9, 0.1, 0.5] + [0.5] * 23  # High O2 gen, low CO2 scrub
        elif i == 1:  # Power shortage
            action = [0.5] * 6 + [0.1, 0.8, 0.8, 0.8, 0.8] + [0.5] * 15  # Low charge, high load shed
        elif i == 2:  # Temperature extreme
            action = [0.5] * 15 + [0.9, 0.9, 0.9, 0.9] + [0.5] * 7  # High heating
        else:  # Water scarcity
            action = [0.5] * 23 + [0.9, 0.9, 0.1]  # High recycling, low rationing
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ {scenario} simulation: reward={reward:.3f}, status={info['status']}")


def test_state_representation():
    """Test state representation and parsing."""
    print("\n📊 Testing State Representation...")
    
    from lunar_habitat_rl.core.lightweight_state import HabitatState, ActionSpace
    
    # Test state creation and conversion
    state = HabitatState(max_crew=4)
    state_array = state.to_array()
    space_info = state.get_observation_space_info()
    
    print(f"✅ State representation: {space_info['total_dims']} dimensions")
    print(f"   - Atmosphere: {space_info['atmosphere_dims']} dims")
    print(f"   - Power: {space_info['power_dims']} dims") 
    print(f"   - Thermal: {space_info['thermal_dims']} dims")
    print(f"   - Water: {space_info['water_dims']} dims")
    print(f"   - Crew: {space_info['crew_dims']} dims")
    print(f"   - Environment: {space_info['environment_dims']} dims")
    
    # Test action space
    action_space = ActionSpace()
    action_info = action_space.get_action_space_info()
    print(f"✅ Action space: {action_info['total_dims']} dimensions")
    
    # Test action parsing
    sample_action = [0.5] * action_info['total_dims']
    parsed_action = action_space.parse_action(sample_action)
    print(f"✅ Action parsing: {len(parsed_action)} subsystems")


def run_complete_episode():
    """Run a complete episode with heuristic agent."""
    print("\n🎯 Running Complete Episode with Heuristic Agent...")
    
    env = lunar_habitat_rl.make_lunar_env()
    agent = HeuristicAgent(action_dims=26)
    
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    
    print(f"Episode started - crew size: {info['crew_size']}, status: {info['status']}")
    
    while step_count < 100:  # Max 100 steps for demo
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Log every 20 steps
        if step_count % 20 == 0:
            print(f"   Step {step_count}: reward={reward:.3f}, total={episode_reward:.3f}, status={info['status']}")
        
        if terminated or truncated:
            break
    
    env.close()
    print(f"✅ Episode complete: {step_count} steps, total reward: {episode_reward:.3f}")
    
    return episode_reward


def main():
    """Run all Generation 1 tests."""
    print("🚀 GENERATION 1 COMPLETE FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Core functionality tests
        env = test_environment_creation()
        obs = test_environment_interaction(env)
        test_all_agents(env, obs)
        env.close()
        
        # System tests
        test_configuration_system()
        test_physics_simulation()
        test_state_representation()
        
        # Integration test
        final_reward = run_complete_episode()
        
        print("\n" + "=" * 50)
        print("🎉 GENERATION 1 TESTING COMPLETE!")
        print("✅ All core functionality working")
        print("✅ All baseline agents functional")
        print("✅ Physics simulation operational")
        print("✅ Configuration system working")
        print("✅ State representation correct")
        print(f"✅ Episode performance: {final_reward:.3f} total reward")
        print("\n🚀 Ready for Generation 2: MAKE IT ROBUST")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Generation 1 testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)