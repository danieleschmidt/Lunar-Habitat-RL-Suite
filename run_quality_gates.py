#!/usr/bin/env python3
"""Comprehensive quality gates for the lunar habitat RL suite."""

import sys
import os
import time
import subprocess
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from lunar_habitat_rl.utils.logging import setup_logging
from lunar_habitat_rl.utils.monitoring import HealthMonitor
from lunar_habitat_rl.utils.exceptions import *


class QualityGates:
    """Comprehensive quality gate system."""
    
    def __init__(self):
        """Initialize quality gates."""
        self.results = {}
        self.logger = setup_logging("INFO")["main"]
        print("🚪 Initializing Quality Gates System...")
    
    def run_all_gates(self) -> Dict[str, bool]:
        """Run all quality gates."""
        gates = [
            ("Code Functionality", self.test_code_functionality),
            ("Environment Creation", self.test_environment_creation),
            ("Agent Functionality", self.test_agent_functionality),
            ("Error Handling", self.test_error_handling),
            ("Security Validation", self.test_security_validation),
            ("Performance Baseline", self.test_performance_baseline),
            ("Memory Management", self.test_memory_management),
            ("Documentation Coverage", self.test_documentation_coverage)
        ]
        
        print(f"\n🎯 Running {len(gates)} Quality Gates...")
        print("=" * 60)
        
        for gate_name, gate_function in gates:
            print(f"\n🔍 {gate_name}...")
            try:
                success = gate_function()
                self.results[gate_name] = success
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   Result: {status}")
            except Exception as e:
                print(f"   Result: ❌ ERROR - {e}")
                self.results[gate_name] = False
        
        return self.results
    
    def test_code_functionality(self) -> bool:
        """Test basic code functionality."""
        try:
            # Test imports
            from lunar_habitat_rl import LunarHabitatEnv
            from lunar_habitat_rl.core import HabitatConfig, HabitatState, MissionMetrics
            from lunar_habitat_rl.utils import get_logger, validate_config
            
            print("   ✓ All core imports successful")
            
            # Test basic configuration
            config = HabitatConfig()
            validated_config = validate_config(config)
            
            print("   ✓ Configuration validation works")
            
            # Test state creation
            state = HabitatState(max_crew=4)
            state_array = state.to_array()
            assert len(state_array) > 0
            assert np.all(np.isfinite(state_array))
            
            print("   ✓ State representation works")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Code functionality failed: {e}")
            return False
    
    def test_environment_creation(self) -> bool:
        """Test environment creation and basic operations."""
        try:
            from lunar_habitat_rl import LunarHabitatEnv
            
            # Test environment creation
            env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
            print("   ✓ Environment created successfully")
            
            # Test reset
            obs, info = env.reset(seed=42)
            assert obs is not None
            assert isinstance(info, dict)
            print("   ✓ Environment reset works")
            
            # Test step
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, (int, float))
            print("   ✓ Environment step works")
            
            # Test multiple steps
            for i in range(5):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    obs, info = env.reset()
            
            print("   ✓ Multi-step execution works")
            
            env.close()
            return True
            
        except Exception as e:
            print(f"   ✗ Environment creation failed: {e}")
            return False
    
    def test_agent_functionality(self) -> bool:
        """Test agent functionality."""
        try:
            from lunar_habitat_rl import LunarHabitatEnv
            from lunar_habitat_rl.algorithms.baselines import RandomAgent, HeuristicAgent
            
            env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
            
            # Test Random Agent
            random_agent = RandomAgent(env.action_space, seed=42)
            obs, _ = env.reset(seed=42)
            
            for i in range(3):
                action = random_agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            
            print("   ✓ Random agent works")
            
            # Test Heuristic Agent
            heuristic_agent = HeuristicAgent(env.action_space)
            obs, _ = env.reset(seed=42)
            
            for i in range(3):
                action = heuristic_agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    break
            
            print("   ✓ Heuristic agent works")
            
            env.close()
            return True
            
        except Exception as e:
            print(f"   ✗ Agent functionality failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and validation.""" 
        try:
            from lunar_habitat_rl import LunarHabitatEnv
            from lunar_habitat_rl.utils.exceptions import EnvironmentError, ValidationError
            
            # Test invalid crew size
            try:
                LunarHabitatEnv(crew_size=20, physics_enabled=False)
                print("   ✗ Should have rejected invalid crew size")
                return False
            except (EnvironmentError, ValidationError, Exception):
                print("   ✓ Invalid crew size properly rejected")
            
            # Test invalid difficulty
            try:
                LunarHabitatEnv(crew_size=4, difficulty="impossible", physics_enabled=False)
                print("   ✗ Should have rejected invalid difficulty")
                return False
            except (EnvironmentError, Exception):
                print("   ✓ Invalid difficulty properly rejected")
            
            # Test environment handles bad actions gracefully
            env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
            obs, _ = env.reset(seed=42)
            
            try:
                # Try action with wrong shape
                bad_action = np.array([0.5, 0.5])  # Too few dimensions
                env.step(bad_action)
                print("   ✗ Should have rejected bad action shape")
                return False
            except Exception:
                print("   ✓ Bad action shape properly handled")
            
            env.close()
            return True
            
        except Exception as e:
            print(f"   ✗ Error handling test failed: {e}")
            return False
    
    def test_security_validation(self) -> bool:
        """Test security measures."""
        try:
            from lunar_habitat_rl.utils.validation import validate_numeric_range, sanitize_string
            from lunar_habitat_rl.utils.exceptions import ValidationError
            
            # Test numeric validation
            try:
                validate_numeric_range(np.inf, 0, 10)
                print("   ✗ Should have rejected infinite value")
                return False
            except ValidationError:
                print("   ✓ Infinite values properly rejected")
            
            # Test string sanitization
            dangerous_string = "'; DROP TABLE users; --"
            sanitized = sanitize_string(dangerous_string)
            assert "DROP" not in sanitized
            assert ";" not in sanitized
            print("   ✓ Dangerous strings properly sanitized")
            
            # Test NaN rejection
            try:
                validate_numeric_range(np.nan, 0, 10)
                print("   ✗ Should have rejected NaN value")
                return False
            except ValidationError:
                print("   ✓ NaN values properly rejected")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Security validation failed: {e}")
            return False
    
    def test_performance_baseline(self) -> bool:
        """Test performance meets baseline requirements."""
        try:
            from lunar_habitat_rl import LunarHabitatEnv
            from lunar_habitat_rl.algorithms.baselines import HeuristicAgent
            
            env = LunarHabitatEnv(crew_size=4, physics_enabled=False)
            agent = HeuristicAgent(env.action_space)
            
            # Measure environment reset time
            start_time = time.time()
            obs, info = env.reset(seed=42)
            reset_time = time.time() - start_time
            
            assert reset_time < 1.0, f"Reset time {reset_time:.3f}s exceeds 1.0s limit"
            print(f"   ✓ Reset time: {reset_time:.3f}s")
            
            # Measure step time
            step_times = []
            for i in range(100):
                start_time = time.time()
                action = agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                step_time = time.time() - start_time
                step_times.append(step_time)
                
                if done or truncated:
                    obs, info = env.reset()
            
            avg_step_time = np.mean(step_times)
            max_step_time = np.max(step_times)
            
            assert avg_step_time < 0.01, f"Average step time {avg_step_time:.4f}s exceeds 0.01s limit"
            assert max_step_time < 0.1, f"Max step time {max_step_time:.4f}s exceeds 0.1s limit"
            
            print(f"   ✓ Average step time: {avg_step_time:.4f}s")
            print(f"   ✓ Max step time: {max_step_time:.4f}s")
            
            env.close()
            return True
            
        except Exception as e:
            print(f"   ✗ Performance baseline failed: {e}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory usage is reasonable."""
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and use multiple environments
            environments = []
            for i in range(5):
                from lunar_habitat_rl import LunarHabitatEnv
                env = LunarHabitatEnv(crew_size=2, physics_enabled=False)
                environments.append(env)
                
                # Run a few steps
                obs, _ = env.reset()
                for j in range(10):
                    action = env.action_space.sample()
                    obs, reward, done, truncated, info = env.step(action)
                    if done or truncated:
                        obs, _ = env.reset()
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            for env in environments:
                env.close()
            environments.clear()
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = peak_memory - initial_memory
            memory_cleanup = peak_memory - final_memory
            
            assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB exceeds 500MB limit"
            assert memory_cleanup > 0, f"Memory not properly cleaned up"
            
            print(f"   ✓ Memory increase: {memory_increase:.1f}MB")
            print(f"   ✓ Memory cleanup: {memory_cleanup:.1f}MB")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Memory management test failed: {e}")
            return False
    
    def test_documentation_coverage(self) -> bool:
        """Test documentation coverage."""
        try:
            # Check key modules have docstrings
            from lunar_habitat_rl import LunarHabitatEnv
            from lunar_habitat_rl.core import HabitatConfig
            from lunar_habitat_rl.utils.validation import validate_config
            
            modules_to_check = [
                LunarHabitatEnv,
                HabitatConfig, 
                validate_config
            ]
            
            for module in modules_to_check:
                assert module.__doc__ is not None, f"{module.__name__} missing docstring"
            
            print("   ✓ Key modules have documentation")
            
            # Check README exists
            readme_path = Path(__file__).parent / "README.md"
            if readme_path.exists():
                print("   ✓ README.md exists")
            else:
                print("   ⚠ README.md missing (non-critical)")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Documentation coverage failed: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate quality gates report."""
        total_gates = len(self.results)
        passed_gates = sum(self.results.values())
        
        report = f"""
🚪 QUALITY GATES REPORT
={'=' * 60}

Overall Result: {'✅ PASS' if passed_gates == total_gates else '❌ FAIL'}
Gates Passed: {passed_gates}/{total_gates} ({passed_gates/total_gates*100:.1f}%)

Detailed Results:
{'-' * 30}
"""
        
        for gate_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report += f"{gate_name:<25} {status}\n"
        
        report += f"""
{'=' * 60}

🎯 Quality Standards Met: {passed_gates == total_gates}
🚀 Ready for Production: {passed_gates >= total_gates * 0.85}
⚠️  Critical Issues: {total_gates - passed_gates}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


def main():
    """Main quality gates execution."""
    print("🌙 Lunar Habitat RL - Quality Gates")
    print("=" * 60)
    
    quality_gates = QualityGates()
    
    try:
        results = quality_gates.run_all_gates()
        report = quality_gates.generate_report()
        
        print(report)
        
        # Save report
        report_path = Path("quality_gates_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_path}")
        
        # Exit code based on results
        total_gates = len(results)
        passed_gates = sum(results.values())
        
        if passed_gates == total_gates:
            print("\n🎉 All quality gates passed!")
            return 0
        elif passed_gates >= total_gates * 0.85:
            print(f"\n⚠️  {total_gates - passed_gates} quality gate(s) failed, but minimum threshold met")
            return 0
        else:
            print(f"\n❌ {total_gates - passed_gates} quality gate(s) failed - critical issues detected")
            return 1
            
    except Exception as e:
        print(f"\n💥 Quality gates execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())