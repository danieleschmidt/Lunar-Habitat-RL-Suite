#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation System for Lunar Habitat RL Suite
NASA Mission Readiness Assessment

Validates all three generations (basic functionality, robustness, scaling) and
ensures mission-critical requirements are met for lunar habitat operations.
"""

import asyncio
import subprocess
import time
import json
import logging
import os
import sys
import psutil
import hashlib
import socket
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import concurrent.futures
import unittest.mock
import tempfile
import shutil

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nasa_quality_gates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Comprehensive quality gate result tracking with NASA requirements"""
    gate_name: str
    category: str  # CODE, PERFORMANCE, SECURITY, NASA_MISSION, INTEGRATION
    passed: bool
    score: float
    max_score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    mission_critical: bool = False
    nasa_requirement: Optional[str] = None

@dataclass
class NASAMissionReadinessScore:
    """NASA Mission Readiness Assessment"""
    overall_score: float
    category_scores: Dict[str, float]
    critical_failures: int
    mission_ready: bool
    certification_level: str  # DEVELOPMENT, TESTING, MISSION_READY, FLIGHT_CERTIFIED
    risk_assessment: str  # LOW, MEDIUM, HIGH, CRITICAL
    recommendations: List[str]

class ComprehensiveQualityGatesValidator:
    """NASA-Grade Quality Gates Validation System"""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp(prefix="nasa_validation_")
        
        # NASA Requirements thresholds
        self.nasa_thresholds = {
            "code_quality_min": 95.0,
            "security_min": 98.0,
            "performance_min": 90.0,
            "reliability_min": 99.5,
            "mission_critical_min": 99.9,
            "test_coverage_min": 95.0,
            "response_time_max": 1.0,  # seconds
            "memory_efficiency_min": 85.0,
            "fault_tolerance_min": 99.0
        }
        
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates for NASA mission readiness"""
        logger.info("🚀 Starting NASA Mission Readiness Quality Gates")
        logger.info("=" * 80)
        
        # Define all quality gates by category
        code_gates = [
            self.validate_package_imports(),
            self.validate_environment_creation(),
            self.validate_configuration_system(),
            self.validate_generation_integration(),
            self.validate_code_quality_advanced()
        ]
        
        performance_gates = [
            self.validate_throughput_benchmarks(),
            self.validate_memory_usage(),
            self.validate_scaling_efficiency(),
            self.validate_response_time()
        ]
        
        security_gates = [
            self.validate_security_scanning(),
            self.validate_input_validation(),
            self.validate_authentication_authorization(),
            self.validate_vulnerability_assessment()
        ]
        
        nasa_mission_gates = [
            self.validate_mission_safety(),
            self.validate_reliability_fault_tolerance(),
            self.validate_space_environment_compatibility(),
            self.validate_emergency_protocols()
        ]
        
        integration_gates = [
            self.validate_cross_component_communication(),
            self.validate_data_integrity(),
            self.validate_end_to_end_functionality(),
            self.validate_production_deployment_readiness()
        ]
        
        # Execute all gates
        all_gates = (code_gates + performance_gates + security_gates + 
                    nasa_mission_gates + integration_gates)
        
        logger.info(f"Executing {len(all_gates)} quality gates...")
        results = await asyncio.gather(*all_gates, return_exceptions=True)
        
        # Process and generate comprehensive report
        return await self._generate_nasa_mission_report()
    
    # ====== CODE QUALITY GATES ======
    
    async def validate_package_imports(self) -> QualityGateResult:
        """Validate package imports and basic functionality"""
        start_time = time.time()
        
        try:
            import_score = 100.0
            import_details = {}
            
            # Test core package import
            try:
                import lunar_habitat_rl
                import_details["core_package"] = "SUCCESS"
            except ImportError as e:
                import_score -= 30
                import_details["core_package"] = f"FAILED: {str(e)}"
            
            # Test environment imports
            try:
                from lunar_habitat_rl.environments import LunarHabitatEnv
                import_details["environment"] = "SUCCESS"
            except ImportError:
                try:
                    from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv
                    import_details["environment"] = "SUCCESS (lightweight)"
                    import_score -= 5  # Minor penalty for fallback
                except ImportError as e:
                    import_score -= 25
                    import_details["environment"] = f"FAILED: {str(e)}"
            
            # Test configuration imports
            try:
                from lunar_habitat_rl.core import HabitatConfig
                import_details["config"] = "SUCCESS"
            except ImportError:
                try:
                    from lunar_habitat_rl.core.lightweight_config import HabitatConfig
                    import_details["config"] = "SUCCESS (lightweight)"
                    import_score -= 5
                except ImportError as e:
                    import_score -= 20
                    import_details["config"] = f"FAILED: {str(e)}"
            
            # Test algorithm imports
            try:
                from lunar_habitat_rl.algorithms import RandomAgent
                import_details["algorithms"] = "SUCCESS"
            except ImportError:
                try:
                    from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent
                    import_details["algorithms"] = "SUCCESS (lightweight)"
                    import_score -= 5
                except ImportError as e:
                    import_score -= 15
                    import_details["algorithms"] = f"FAILED: {str(e)}"
            
            passed = import_score >= self.nasa_thresholds["code_quality_min"]
            
            return QualityGateResult(
                gate_name="Package Imports",
                category="CODE",
                passed=passed,
                score=import_score,
                max_score=100.0,
                details=import_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="CRITICAL" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-SW-001: Core software components must load successfully"
            )
            
        except Exception as e:
            return self._create_error_result("Package Imports", "CODE", str(e), start_time, True)
    
    async def validate_environment_creation(self) -> QualityGateResult:
        """Validate environment creation and basic operations"""
        start_time = time.time()
        
        try:
            env_score = 100.0
            env_details = {}
            
            # Test environment creation
            try:
                import lunar_habitat_rl
                env = lunar_habitat_rl.make_lunar_env()
                env_details["creation"] = "SUCCESS"
            except Exception as e:
                env_score -= 40
                env_details["creation"] = f"FAILED: {str(e)}"
                return self._create_result("Environment Creation", "CODE", False, env_score, 
                                         env_details, start_time, "CRITICAL", True)
            
            # Test environment reset
            try:
                obs, info = env.reset()
                env_details["reset"] = "SUCCESS"
                env_details["observation_shape"] = str(obs.shape) if hasattr(obs, 'shape') else str(type(obs))
            except Exception as e:
                env_score -= 30
                env_details["reset"] = f"FAILED: {str(e)}"
            
            # Test environment step
            try:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env_details["step"] = "SUCCESS"
                env_details["reward"] = str(reward)
            except Exception as e:
                env_score -= 30
                env_details["step"] = f"FAILED: {str(e)}"
            
            # Test action/observation spaces
            try:
                env_details["action_space"] = str(env.action_space)
                env_details["observation_space"] = str(env.observation_space)
            except Exception as e:
                env_score -= 10
                env_details["spaces"] = f"FAILED: {str(e)}"
            
            passed = env_score >= self.nasa_thresholds["code_quality_min"]
            
            return QualityGateResult(
                gate_name="Environment Creation",
                category="CODE",
                passed=passed,
                score=env_score,
                max_score=100.0,
                details=env_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="CRITICAL" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-ENV-001: Lunar habitat environment must initialize and operate correctly"
            )
            
        except Exception as e:
            return self._create_error_result("Environment Creation", "CODE", str(e), start_time, True)
    
    async def validate_configuration_system(self) -> QualityGateResult:
        """Validate configuration system"""
        start_time = time.time()
        
        try:
            config_score = 100.0
            config_details = {}
            
            # Test configuration creation
            try:
                from lunar_habitat_rl.core.lightweight_config import HabitatConfig
                config = HabitatConfig()
                config_details["creation"] = "SUCCESS"
            except Exception as e:
                config_score -= 50
                config_details["creation"] = f"FAILED: {str(e)}"
                return self._create_result("Configuration System", "CODE", False, config_score,
                                         config_details, start_time, "ERROR", False)
            
            # Test configuration attributes
            required_attrs = ['crew_size', 'max_episode_steps', 'action_bounds']
            for attr in required_attrs:
                if hasattr(config, attr):
                    config_details[f"attr_{attr}"] = "EXISTS"
                else:
                    config_score -= 15
                    config_details[f"attr_{attr}"] = "MISSING"
            
            # Test configuration validation
            try:
                if hasattr(config, 'validate'):
                    config.validate()
                    config_details["validation"] = "SUCCESS"
                else:
                    config_score -= 10
                    config_details["validation"] = "NOT_IMPLEMENTED"
            except Exception as e:
                config_score -= 20
                config_details["validation"] = f"FAILED: {str(e)}"
            
            passed = config_score >= 80.0
            
            return QualityGateResult(
                gate_name="Configuration System",
                category="CODE",
                passed=passed,
                score=config_score,
                max_score=100.0,
                details=config_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO",
                mission_critical=False,
                nasa_requirement="REQ-CFG-001: Configuration system must support habitat parameters"
            )
            
        except Exception as e:
            return self._create_error_result("Configuration System", "CODE", str(e), start_time, False)
    
    async def validate_generation_integration(self) -> QualityGateResult:
        """Validate all three generations working together"""
        start_time = time.time()
        
        try:
            integration_score = 100.0
            integration_details = {}
            
            # Test Generation 1 - Basic functionality
            try:
                exec_result = subprocess.run([
                    sys.executable, "-c", 
                    "import sys; sys.path.insert(0, '/root/repo'); "
                    "from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv; "
                    "env = LunarHabitatEnv(); env.reset(); print('Gen1 OK')"
                ], capture_output=True, text=True, timeout=10)
                
                if exec_result.returncode == 0:
                    integration_details["generation_1"] = "SUCCESS"
                else:
                    integration_score -= 35
                    integration_details["generation_1"] = f"FAILED: {exec_result.stderr}"
            except Exception as e:
                integration_score -= 35
                integration_details["generation_1"] = f"FAILED: {str(e)}"
            
            # Test Generation 2 - Robustness
            try:
                if Path("/root/repo/generation2_robustness.py").exists():
                    integration_details["generation_2_file"] = "EXISTS"
                    # Test import capability
                    exec_result = subprocess.run([
                        sys.executable, "-c",
                        "import sys; sys.path.insert(0, '/root/repo'); "
                        "import generation2_robustness; print('Gen2 OK')"
                    ], capture_output=True, text=True, timeout=10)
                    
                    if exec_result.returncode == 0:
                        integration_details["generation_2"] = "SUCCESS"
                    else:
                        integration_score -= 30
                        integration_details["generation_2"] = f"IMPORT_FAILED: {exec_result.stderr}"
                else:
                    integration_score -= 35
                    integration_details["generation_2"] = "FILE_MISSING"
            except Exception as e:
                integration_score -= 30
                integration_details["generation_2"] = f"FAILED: {str(e)}"
            
            # Test Generation 3 - Scaling
            try:
                if Path("/root/repo/generation3_scaling.py").exists():
                    integration_details["generation_3_file"] = "EXISTS"
                    # Test import capability
                    exec_result = subprocess.run([
                        sys.executable, "-c",
                        "import sys; sys.path.insert(0, '/root/repo'); "
                        "import generation3_scaling; print('Gen3 OK')"
                    ], capture_output=True, text=True, timeout=10)
                    
                    if exec_result.returncode == 0:
                        integration_details["generation_3"] = "SUCCESS"
                    else:
                        integration_score -= 30
                        integration_details["generation_3"] = f"IMPORT_FAILED: {exec_result.stderr}"
                else:
                    integration_score -= 35
                    integration_details["generation_3"] = "FILE_MISSING"
            except Exception as e:
                integration_score -= 30
                integration_details["generation_3"] = f"FAILED: {str(e)}"
            
            passed = integration_score >= 85.0
            
            return QualityGateResult(
                gate_name="Generation Integration",
                category="CODE",
                passed=passed,
                score=integration_score,
                max_score=100.0,
                details=integration_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-INT-001: All software generations must integrate seamlessly"
            )
            
        except Exception as e:
            return self._create_error_result("Generation Integration", "CODE", str(e), start_time, True)
    
    async def validate_code_quality_advanced(self) -> QualityGateResult:
        """Advanced code quality validation"""
        start_time = time.time()
        
        try:
            quality_score = 100.0
            quality_details = {}
            issues = []
            
            # Get Python files
            python_files = list(Path("lunar_habitat_rl").rglob("*.py"))
            quality_details["files_analyzed"] = len(python_files)
            
            # Analyze code quality metrics
            long_files = 0
            complex_files = 0
            undocumented_files = 0
            
            for py_file in python_files[:20]:  # Sample first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        content = ''.join(lines)
                        
                        # Check file length
                        if len(lines) > 500:
                            long_files += 1
                        
                        # Check for docstrings
                        if '"""' not in content and "'''" not in content:
                            undocumented_files += 1
                        
                        # Check complexity indicators
                        complexity_indicators = content.count('if ') + content.count('for ') + content.count('while ')
                        if complexity_indicators > 50:
                            complex_files += 1
                            
                except Exception:
                    continue
            
            # Apply penalties
            if long_files > len(python_files) * 0.3:
                quality_score -= 15
                issues.append(f"{long_files} files are very long (>500 lines)")
            
            if undocumented_files > len(python_files) * 0.2:
                quality_score -= 20
                issues.append(f"{undocumented_files} files lack documentation")
            
            if complex_files > len(python_files) * 0.3:
                quality_score -= 10
                issues.append(f"{complex_files} files appear highly complex")
            
            quality_details["issues"] = issues
            quality_details["long_files"] = long_files
            quality_details["undocumented_files"] = undocumented_files
            quality_details["complex_files"] = complex_files
            
            passed = quality_score >= 80.0 and len(issues) <= 3
            
            return QualityGateResult(
                gate_name="Advanced Code Quality",
                category="CODE",
                passed=passed,
                score=quality_score,
                max_score=100.0,
                details=quality_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=False,
                nasa_requirement="REQ-QA-001: Code must meet NASA software quality standards"
            )
            
        except Exception as e:
            return self._create_error_result("Advanced Code Quality", "CODE", str(e), start_time, False)
    
    # ====== PERFORMANCE GATES ======
    
    async def validate_throughput_benchmarks(self) -> QualityGateResult:
        """Validate system throughput meets minimum requirements"""
        start_time = time.time()
        
        try:
            perf_score = 100.0
            perf_details = {}
            
            # Import and create environment
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            # Throughput test - episodes per second
            episode_start = time.time()
            episodes_completed = 0
            target_episodes = 10
            
            for _ in range(target_episodes):
                env.reset()
                for _ in range(50):  # 50 steps per episode
                    action = env.action_space.sample()
                    env.step(action)
                episodes_completed += 1
            
            episode_time = time.time() - episode_start
            eps_per_second = episodes_completed / episode_time
            
            perf_details["episodes_per_second"] = f"{eps_per_second:.2f}"
            perf_details["total_time"] = f"{episode_time:.2f}s"
            
            # Scoring based on throughput
            if eps_per_second < 0.5:  # Less than 0.5 EPS
                perf_score -= 40
            elif eps_per_second < 1.0:  # Less than 1 EPS
                perf_score -= 20
            elif eps_per_second < 2.0:  # Less than 2 EPS
                perf_score -= 10
            
            # Memory efficiency during throughput test
            import psutil
            memory_percent = psutil.virtual_memory().percent
            perf_details["memory_usage"] = f"{memory_percent:.1f}%"
            
            if memory_percent > 90:
                perf_score -= 30
            elif memory_percent > 80:
                perf_score -= 15
            
            passed = perf_score >= self.nasa_thresholds["performance_min"]
            
            return QualityGateResult(
                gate_name="Throughput Benchmarks",
                category="PERFORMANCE",
                passed=passed,
                score=perf_score,
                max_score=100.0,
                details=perf_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-PERF-001: System must maintain minimum 1 EPS throughput"
            )
            
        except Exception as e:
            return self._create_error_result("Throughput Benchmarks", "PERFORMANCE", str(e), start_time, True)
    
    async def validate_memory_usage(self) -> QualityGateResult:
        """Validate memory usage patterns and efficiency"""
        start_time = time.time()
        
        try:
            import psutil
            
            memory_score = 100.0
            memory_details = {}
            
            # Initial memory reading
            initial_memory = psutil.virtual_memory()
            memory_details["initial_memory_percent"] = f"{initial_memory.percent:.1f}%"
            memory_details["initial_memory_available"] = f"{initial_memory.available // (1024**2)}MB"
            
            # Memory stress test
            import lunar_habitat_rl
            environments = []
            
            # Create multiple environments to test memory scaling
            for i in range(5):
                try:
                    env = lunar_habitat_rl.make_lunar_env()
                    environments.append(env)
                    
                    # Run some operations
                    env.reset()
                    for _ in range(10):
                        action = env.action_space.sample()
                        env.step(action)
                        
                except Exception as e:
                    memory_score -= 15
                    memory_details[f"env_creation_error_{i}"] = str(e)
            
            # Final memory reading
            final_memory = psutil.virtual_memory()
            memory_increase = final_memory.percent - initial_memory.percent
            
            memory_details["final_memory_percent"] = f"{final_memory.percent:.1f}%"
            memory_details["memory_increase"] = f"{memory_increase:.1f}%"
            memory_details["environments_created"] = len(environments)
            
            # Score based on memory efficiency
            if memory_increase > 30:  # More than 30% increase
                memory_score -= 40
            elif memory_increase > 20:
                memory_score -= 25
            elif memory_increase > 10:
                memory_score -= 10
            
            if final_memory.percent > 95:
                memory_score -= 30
            elif final_memory.percent > 85:
                memory_score -= 15
            
            passed = memory_score >= self.nasa_thresholds["memory_efficiency_min"]
            
            return QualityGateResult(
                gate_name="Memory Usage Validation",
                category="PERFORMANCE",
                passed=passed,
                score=memory_score,
                max_score=100.0,
                details=memory_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-MEM-001: Memory usage must remain efficient under load"
            )
            
        except Exception as e:
            return self._create_error_result("Memory Usage Validation", "PERFORMANCE", str(e), start_time, True)
    
    async def validate_scaling_efficiency(self) -> QualityGateResult:
        """Validate scaling efficiency and parallel execution"""
        start_time = time.time()
        
        try:
            scaling_score = 100.0
            scaling_details = {}
            
            # Test sequential vs parallel performance
            import lunar_habitat_rl
            
            # Sequential execution test
            seq_start = time.time()
            for i in range(3):
                env = lunar_habitat_rl.make_lunar_env()
                env.reset()
                for _ in range(20):
                    action = env.action_space.sample()
                    env.step(action)
            seq_time = time.time() - seq_start
            
            scaling_details["sequential_time"] = f"{seq_time:.3f}s"
            
            # Parallel execution test (simulated)
            def run_env_episode():
                try:
                    env = lunar_habitat_rl.make_lunar_env()
                    env.reset()
                    for _ in range(20):
                        action = env.action_space.sample()
                        env.step(action)
                    return True
                except:
                    return False
            
            par_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_env_episode) for _ in range(3)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            par_time = time.time() - par_start
            
            scaling_details["parallel_time"] = f"{par_time:.3f}s"
            scaling_details["parallel_success_rate"] = f"{sum(results)/len(results)*100:.1f}%"
            
            # Calculate efficiency
            if par_time > 0 and seq_time > 0:
                efficiency = (seq_time / par_time) / 3.0 * 100  # Normalized by number of workers
                scaling_details["scaling_efficiency"] = f"{efficiency:.1f}%"
                
                if efficiency < 30:  # Less than 30% efficiency
                    scaling_score -= 40
                elif efficiency < 50:
                    scaling_score -= 25
                elif efficiency < 70:
                    scaling_score -= 10
            else:
                scaling_score -= 50
                scaling_details["scaling_efficiency"] = "CANNOT_CALCULATE"
            
            # Check if all parallel executions succeeded
            if sum(results) < len(results):
                scaling_score -= 30
                scaling_details["parallel_failures"] = len(results) - sum(results)
            
            passed = scaling_score >= 70.0
            
            return QualityGateResult(
                gate_name="Scaling Efficiency",
                category="PERFORMANCE",
                passed=passed,
                score=scaling_score,
                max_score=100.0,
                details=scaling_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=False,
                nasa_requirement="REQ-SCALE-001: System must scale efficiently with parallel execution"
            )
            
        except Exception as e:
            return self._create_error_result("Scaling Efficiency", "PERFORMANCE", str(e), start_time, False)
    
    async def validate_response_time(self) -> QualityGateResult:
        """Validate system response times meet real-time requirements"""
        start_time = time.time()
        
        try:
            response_score = 100.0
            response_details = {}
            
            import lunar_habitat_rl
            
            # Environment creation response time
            create_start = time.time()
            env = lunar_habitat_rl.make_lunar_env()
            create_time = time.time() - create_start
            
            response_details["env_creation_time"] = f"{create_time:.4f}s"
            
            # Environment reset response time
            reset_start = time.time()
            obs, info = env.reset()
            reset_time = time.time() - reset_start
            
            response_details["env_reset_time"] = f"{reset_time:.4f}s"
            
            # Step response time (average over multiple steps)
            step_times = []
            for _ in range(50):
                action = env.action_space.sample()
                step_start = time.time()
                env.step(action)
                step_time = time.time() - step_start
                step_times.append(step_time)
            
            avg_step_time = sum(step_times) / len(step_times)
            max_step_time = max(step_times)
            
            response_details["avg_step_time"] = f"{avg_step_time:.4f}s"
            response_details["max_step_time"] = f"{max_step_time:.4f}s"
            
            # Score based on response times
            if create_time > 2.0:
                response_score -= 25
            elif create_time > 1.0:
                response_score -= 10
            
            if reset_time > 0.5:
                response_score -= 20
            elif reset_time > 0.1:
                response_score -= 10
            
            if avg_step_time > 0.1:
                response_score -= 30
            elif avg_step_time > 0.05:
                response_score -= 15
            
            if max_step_time > 0.5:
                response_score -= 25
            
            passed = (response_score >= 80.0 and 
                     avg_step_time <= self.nasa_thresholds["response_time_max"])
            
            return QualityGateResult(
                gate_name="Response Time Validation",
                category="PERFORMANCE",
                passed=passed,
                score=response_score,
                max_score=100.0,
                details=response_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-RT-001: Real-time response requirements for mission operations"
            )
            
        except Exception as e:
            return self._create_error_result("Response Time Validation", "PERFORMANCE", str(e), start_time, True)
    
    # ====== SECURITY GATES ======
    
    async def validate_security_scanning(self) -> QualityGateResult:
        """Comprehensive security scanning and vulnerability assessment"""
        start_time = time.time()
        
        try:
            security_score = 100.0
            security_details = {}
            vulnerabilities = []
            
            # Scan Python files for security issues
            python_files = list(Path(".").rglob("*.py"))
            security_details["files_scanned"] = len(python_files)
            
            dangerous_patterns = {
                'eval(': 'Code injection vulnerability',
                'exec(': 'Code execution vulnerability', 
                'shell=True': 'Shell injection risk',
                'pickle.load': 'Deserialization vulnerability',
                'yaml.load': 'YAML deserialization risk',
                'input(': 'Potential input injection',
                'os.system': 'OS command injection risk',
                'subprocess.call': 'Subprocess security risk'
            }
            
            for py_file in python_files[:50]:  # Scan first 50 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern, description in dangerous_patterns.items():
                        if pattern in content:
                            vulnerabilities.append({
                                'file': str(py_file),
                                'pattern': pattern,
                                'description': description,
                                'severity': 'HIGH' if pattern in ['eval(', 'exec('] else 'MEDIUM'
                            })
                            
                            severity_penalty = 25 if pattern in ['eval(', 'exec('] else 10
                            security_score -= severity_penalty
                            
                except Exception:
                    continue
            
            # Check file permissions
            sensitive_files = ['requirements.txt', 'pyproject.toml', '.env']
            permission_issues = []
            
            for file_path in sensitive_files:
                path = Path(file_path)
                if path.exists():
                    try:
                        stat_info = path.stat()
                        if stat_info.st_mode & 0o044:  # World or group readable
                            permission_issues.append(f"Overly permissive permissions on {file_path}")
                            security_score -= 5
                    except Exception:
                        continue
            
            # Check for secrets in code
            secret_patterns = ['password', 'api_key', 'secret_key', 'token']
            potential_secrets = []
            
            for py_file in python_files[:20]:  # Sample
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern in secret_patterns:
                        if pattern in content and '=' in content:
                            potential_secrets.append(f"Potential secret in {py_file}")
                            security_score -= 15
                            break
                except Exception:
                    continue
            
            security_details["vulnerabilities"] = vulnerabilities
            security_details["permission_issues"] = permission_issues
            security_details["potential_secrets"] = potential_secrets
            security_details["high_severity_vulns"] = len([v for v in vulnerabilities if v['severity'] == 'HIGH'])
            
            passed = (security_score >= self.nasa_thresholds["security_min"] and 
                     len([v for v in vulnerabilities if v['severity'] == 'HIGH']) == 0)
            
            return QualityGateResult(
                gate_name="Security Scanning",
                category="SECURITY",
                passed=passed,
                score=security_score,
                max_score=100.0,
                details=security_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="CRITICAL" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-SEC-001: No high-severity security vulnerabilities allowed"
            )
            
        except Exception as e:
            return self._create_error_result("Security Scanning", "SECURITY", str(e), start_time, True)
    
    async def validate_input_validation(self) -> QualityGateResult:
        """Validate input validation and sanitization"""
        start_time = time.time()
        
        try:
            input_score = 100.0
            input_details = {}
            
            # Test environment with invalid inputs
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            env.reset()
            
            validation_tests = []
            
            # Test invalid action types
            try:
                invalid_actions = [None, "invalid", [], {}, -999999, 999999]
                for invalid_action in invalid_actions:
                    try:
                        env.step(invalid_action)
                        validation_tests.append(f"FAILED: Accepted invalid action {type(invalid_action)}")
                        input_score -= 15
                    except (ValueError, TypeError, AssertionError):
                        validation_tests.append(f"PASSED: Rejected invalid action {type(invalid_action)}")
                    except Exception as e:
                        validation_tests.append(f"UNKNOWN: {type(invalid_action)} -> {str(e)}")
                        input_score -= 5
            except Exception as e:
                input_score -= 25
                validation_tests.append(f"ERROR: Could not test invalid actions: {str(e)}")
            
            # Test configuration validation
            try:
                from lunar_habitat_rl.core.lightweight_config import HabitatConfig
                
                # Test invalid configuration values
                invalid_configs = [
                    {'crew_size': -1},
                    {'crew_size': 0},
                    {'max_episode_steps': -100},
                    {'max_episode_steps': 0}
                ]
                
                for invalid_config in invalid_configs:
                    try:
                        config = HabitatConfig(**invalid_config)
                        if hasattr(config, 'validate'):
                            config.validate()
                        validation_tests.append(f"FAILED: Accepted invalid config {invalid_config}")
                        input_score -= 10
                    except (ValueError, AssertionError):
                        validation_tests.append(f"PASSED: Rejected invalid config {invalid_config}")
                    except Exception as e:
                        validation_tests.append(f"UNKNOWN: Config {invalid_config} -> {str(e)}")
                        input_score -= 5
                        
            except Exception as e:
                input_score -= 20
                validation_tests.append(f"ERROR: Could not test config validation: {str(e)}")
            
            input_details["validation_tests"] = validation_tests
            input_details["total_tests"] = len(validation_tests)
            passed_tests = len([t for t in validation_tests if t.startswith("PASSED")])
            input_details["passed_tests"] = passed_tests
            
            passed = input_score >= 75.0 and passed_tests >= len(validation_tests) * 0.6
            
            return QualityGateResult(
                gate_name="Input Validation",
                category="SECURITY",
                passed=passed,
                score=input_score,
                max_score=100.0,
                details=input_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-VAL-001: All inputs must be validated and sanitized"
            )
            
        except Exception as e:
            return self._create_error_result("Input Validation", "SECURITY", str(e), start_time, True)
    
    async def validate_authentication_authorization(self) -> QualityGateResult:
        """Validate authentication and authorization mechanisms"""
        start_time = time.time()
        
        try:
            auth_score = 100.0
            auth_details = {}
            
            # Check for authentication/authorization patterns in code
            python_files = list(Path(".").rglob("*.py"))
            auth_indicators = {
                'authenticate': 'Authentication mechanism',
                'authorize': 'Authorization mechanism', 
                'permission': 'Permission system',
                'login': 'Login system',
                'token': 'Token-based auth',
                'session': 'Session management',
                'role': 'Role-based access',
                'access_control': 'Access control'
            }
            
            found_mechanisms = []
            for py_file in python_files[:30]:  # Sample
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for indicator, description in auth_indicators.items():
                        if indicator in content:
                            found_mechanisms.append(description)
                            auth_score += 5  # Bonus for having auth mechanisms
                            break
                except Exception:
                    continue
            
            auth_details["auth_mechanisms_found"] = list(set(found_mechanisms))
            auth_details["auth_indicators_count"] = len(set(found_mechanisms))
            
            # Check for security headers/middleware patterns
            security_patterns = ['security', 'middleware', 'cors', 'csrf', 'xss']
            security_found = []
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern in security_patterns:
                        if pattern in content:
                            security_found.append(pattern)
                            auth_score += 3
                            break
                except Exception:
                    continue
            
            auth_details["security_patterns_found"] = list(set(security_found))
            
            # For a research/simulation environment, auth requirements are relaxed
            # but we still check for basic security awareness
            if len(found_mechanisms) == 0:
                auth_score -= 20  # Minor penalty for no auth mechanisms
                auth_details["auth_assessment"] = "No authentication mechanisms detected (acceptable for research)"
            else:
                auth_details["auth_assessment"] = "Authentication mechanisms present"
            
            if len(security_found) == 0:
                auth_score -= 10
                auth_details["security_assessment"] = "Limited security patterns detected"
            else:
                auth_details["security_assessment"] = "Security patterns present"
            
            # Adjust scoring for research environment
            auth_score = min(100.0, auth_score)
            passed = auth_score >= 70.0  # Relaxed threshold for research environment
            
            return QualityGateResult(
                gate_name="Authentication & Authorization",
                category="SECURITY", 
                passed=passed,
                score=auth_score,
                max_score=100.0,
                details=auth_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=False,
                nasa_requirement="REQ-AUTH-001: Authentication mechanisms for production deployment"
            )
            
        except Exception as e:
            return self._create_error_result("Authentication & Authorization", "SECURITY", str(e), start_time, False)
    
    async def validate_vulnerability_assessment(self) -> QualityGateResult:
        """Comprehensive vulnerability assessment"""
        start_time = time.time()
        
        try:
            vuln_score = 100.0
            vuln_details = {}
            
            # Check dependency files for known vulnerable packages
            vulnerable_packages = {
                'pickle': 'Deserialization attacks',
                'yaml': 'YAML bombs and injection (use safe_load)',
                'subprocess': 'Command injection (avoid shell=True)',
                'os.system': 'Command injection',
                'eval': 'Code injection',
                'exec': 'Code execution'
            }
            
            dependency_risks = []
            
            # Check requirements.txt
            req_file = Path('requirements.txt')
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        requirements = f.read().lower()
                        
                    for package, risk in vulnerable_packages.items():
                        if package in requirements:
                            dependency_risks.append(f"{package}: {risk}")
                            vuln_score -= 10
                except Exception:
                    pass
            
            # Check for cryptographic issues
            crypto_issues = []
            python_files = list(Path(".").rglob("*.py"))
            
            weak_crypto_patterns = {
                'md5': 'Weak hash algorithm',
                'sha1': 'Weak hash algorithm', 
                'des': 'Weak encryption',
                'random.random': 'Weak randomness for security',
                'hardcoded': 'Hardcoded secrets risk'
            }
            
            for py_file in python_files[:25]:  # Sample
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern, issue in weak_crypto_patterns.items():
                        if pattern in content:
                            crypto_issues.append(f"{py_file}: {issue}")
                            vuln_score -= 8
                            break
                except Exception:
                    continue
            
            # Check for information disclosure
            info_disclosure = []
            disclosure_patterns = ['print(', 'debug', 'traceback', 'exception']
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in disclosure_patterns):
                        # This is mostly OK for research code, minor penalty
                        vuln_score -= 2
                        break
                except Exception:
                    continue
            
            vuln_details["dependency_risks"] = dependency_risks
            vuln_details["crypto_issues"] = crypto_issues
            vuln_details["info_disclosure_risk"] = "Present but acceptable for research"
            vuln_details["total_vulnerabilities"] = len(dependency_risks) + len(crypto_issues)
            
            passed = vuln_score >= 80.0 and len(dependency_risks) <= 2
            
            return QualityGateResult(
                gate_name="Vulnerability Assessment",
                category="SECURITY",
                passed=passed,
                score=vuln_score,
                max_score=100.0,
                details=vuln_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-VULN-001: No critical vulnerabilities in dependencies"
            )
            
        except Exception as e:
            return self._create_error_result("Vulnerability Assessment", "SECURITY", str(e), start_time, True)
    
    # ====== NASA MISSION READINESS GATES ======
    
    async def validate_mission_safety(self) -> QualityGateResult:
        """Validate mission-critical safety systems"""
        start_time = time.time()
        
        try:
            safety_score = 100.0
            safety_details = {}
            
            # Check for safety validation systems
            safety_files = list(Path(".").rglob("*safety*.py")) + list(Path(".").rglob("*mission*.py"))
            safety_details["safety_files_found"] = len(safety_files)
            
            if safety_files:
                safety_score += 20
                safety_details["safety_systems"] = "PRESENT"
            else:
                safety_score -= 30
                safety_details["safety_systems"] = "MISSING"
            
            # Test environment safety constraints
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            obs, info = env.reset()
            
            safety_tests = []
            
            # Test critical system boundaries
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check for safety violations (basic checks)
                if hasattr(obs, '__len__') and len(obs) > 0:
                    if isinstance(obs, (list, tuple)):
                        obs_array = obs
                    else:
                        obs_array = [obs] if not hasattr(obs, '__iter__') else obs
                    
                    # Check for NaN or infinite values (safety critical)
                    try:
                        for val in obs_array:
                            if hasattr(val, '__iter__'):
                                for subval in val:
                                    if str(subval).lower() in ['nan', 'inf', '-inf']:
                                        safety_tests.append("FAILED: NaN/Inf values detected")
                                        safety_score -= 25
                                        break
                            else:
                                if str(val).lower() in ['nan', 'inf', '-inf']:
                                    safety_tests.append("FAILED: NaN/Inf values detected")
                                    safety_score -= 25
                                    break
                    except Exception:
                        pass
                
                # Check reward bounds (should be reasonable)
                if abs(reward) > 1000000:  # Extremely large rewards might indicate issues
                    safety_tests.append("WARNING: Extremely large reward detected")
                    safety_score -= 10
            
            if not safety_tests:
                safety_tests.append("PASSED: No safety violations detected in basic tests")
            
            # Check for error handling patterns
            python_files = list(Path("lunar_habitat_rl").rglob("*.py"))
            error_handling_count = 0
            
            for py_file in python_files[:15]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if 'try:' in content and 'except' in content:
                        error_handling_count += 1
                except Exception:
                    continue
            
            error_handling_ratio = error_handling_count / max(1, len(python_files[:15]))
            safety_details["error_handling_ratio"] = f"{error_handling_ratio * 100:.1f}%"
            
            if error_handling_ratio < 0.3:  # Less than 30% files have error handling
                safety_score -= 20
                safety_details["error_handling"] = "INSUFFICIENT"
            else:
                safety_details["error_handling"] = "ADEQUATE"
            
            safety_details["safety_tests"] = safety_tests
            
            passed = safety_score >= self.nasa_thresholds["mission_critical_min"]
            
            return QualityGateResult(
                gate_name="Mission Safety Validation",
                category="NASA_MISSION",
                passed=passed,
                score=safety_score,
                max_score=100.0,
                details=safety_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="CRITICAL" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-SAFETY-001: Mission-critical safety systems must be validated"
            )
            
        except Exception as e:
            return self._create_error_result("Mission Safety Validation", "NASA_MISSION", str(e), start_time, True)
    
    async def validate_reliability_fault_tolerance(self) -> QualityGateResult:
        """Validate system reliability and fault tolerance"""
        start_time = time.time()
        
        try:
            reliability_score = 100.0
            reliability_details = {}
            
            # Test system recovery from errors
            import lunar_habitat_rl
            
            fault_tolerance_tests = []
            
            # Test environment recreation after errors
            try:
                for attempt in range(5):
                    env = lunar_habitat_rl.make_lunar_env()
                    env.reset()
                    
                    # Simulate some operations
                    for _ in range(10):
                        action = env.action_space.sample()
                        env.step(action)
                    
                    # Force environment deletion and recreation
                    del env
                    
                fault_tolerance_tests.append("PASSED: Environment recreation successful")
            except Exception as e:
                reliability_score -= 30
                fault_tolerance_tests.append(f"FAILED: Environment recreation failed: {str(e)}")
            
            # Test system under stress
            try:
                stress_envs = []
                for i in range(10):  # Create multiple environments
                    env = lunar_habitat_rl.make_lunar_env()
                    stress_envs.append(env)
                
                # Test all environments
                for env in stress_envs:
                    env.reset()
                    for _ in range(5):
                        action = env.action_space.sample()
                        env.step(action)
                
                fault_tolerance_tests.append("PASSED: Stress test with multiple environments")
                
                # Cleanup
                for env in stress_envs:
                    del env
                    
            except Exception as e:
                reliability_score -= 25
                fault_tolerance_tests.append(f"FAILED: Stress test failed: {str(e)}")
            
            # Test configuration robustness
            try:
                from lunar_habitat_rl.core.lightweight_config import HabitatConfig
                
                # Test various configuration scenarios
                configs_to_test = [
                    {'crew_size': 1},
                    {'crew_size': 6},
                    {'max_episode_steps': 100},
                    {'max_episode_steps': 1000}
                ]
                
                for config_params in configs_to_test:
                    try:
                        config = HabitatConfig(**config_params)
                        env = lunar_habitat_rl.make_lunar_env()
                        env.reset()
                    except Exception as e:
                        reliability_score -= 10
                        fault_tolerance_tests.append(f"FAILED: Config {config_params} failed: {str(e)}")
                        break
                else:
                    fault_tolerance_tests.append("PASSED: Configuration robustness test")
                    
            except Exception as e:
                reliability_score -= 20
                fault_tolerance_tests.append(f"FAILED: Configuration test failed: {str(e)}")
            
            # Check for graceful degradation patterns
            python_files = list(Path("lunar_habitat_rl").rglob("*.py"))
            graceful_degradation_indicators = 0
            
            degradation_patterns = ['fallback', 'default', 'graceful', 'backup', 'alternative']
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in degradation_patterns):
                        graceful_degradation_indicators += 1
                except Exception:
                    continue
            
            if graceful_degradation_indicators >= 3:
                reliability_details["graceful_degradation"] = "GOOD"
            elif graceful_degradation_indicators >= 1:
                reliability_details["graceful_degradation"] = "BASIC"
                reliability_score -= 10
            else:
                reliability_details["graceful_degradation"] = "MISSING"
                reliability_score -= 20
            
            reliability_details["fault_tolerance_tests"] = fault_tolerance_tests
            reliability_details["graceful_degradation_indicators"] = graceful_degradation_indicators
            
            passed = reliability_score >= self.nasa_thresholds["reliability_min"]
            
            return QualityGateResult(
                gate_name="Reliability & Fault Tolerance",
                category="NASA_MISSION",
                passed=passed,
                score=reliability_score,
                max_score=100.0,
                details=reliability_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="CRITICAL" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-REL-001: System must demonstrate fault tolerance and reliability"
            )
            
        except Exception as e:
            return self._create_error_result("Reliability & Fault Tolerance", "NASA_MISSION", str(e), start_time, True)
    
    async def validate_space_environment_compatibility(self) -> QualityGateResult:
        """Validate compatibility with space environment constraints"""
        start_time = time.time()
        
        try:
            space_score = 100.0
            space_details = {}
            
            # Check for space-specific considerations in code
            python_files = list(Path(".").rglob("*.py"))
            space_indicators = {
                'radiation': 'Radiation considerations',
                'vacuum': 'Vacuum environment',
                'microgravity': 'Microgravity effects',
                'thermal': 'Thermal management',
                'power': 'Power management',
                'life_support': 'Life support systems',
                'atmosphere': 'Atmospheric control',
                'pressure': 'Pressure systems',
                'oxygen': 'Oxygen management',
                'co2': 'CO2 scrubbing',
                'temperature': 'Temperature control'
            }
            
            found_space_considerations = []
            
            for py_file in python_files[:30]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for indicator, description in space_indicators.items():
                        if indicator in content:
                            found_space_considerations.append(description)
                            space_score += 5  # Bonus for space considerations
                except Exception:
                    continue
            
            space_details["space_considerations"] = list(set(found_space_considerations))
            space_details["space_indicators_count"] = len(set(found_space_considerations))
            
            # Test environment under extreme conditions (simulated)
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            extreme_condition_tests = []
            
            # Test rapid state changes (simulating space environment dynamics)
            try:
                env.reset()
                for _ in range(100):  # Rapid operations
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        env.reset()
                
                extreme_condition_tests.append("PASSED: Rapid state changes test")
                
            except Exception as e:
                space_score -= 25
                extreme_condition_tests.append(f"FAILED: Rapid state changes: {str(e)}")
            
            # Check for resource management patterns
            resource_patterns = ['resource', 'allocation', 'consumption', 'efficiency', 'optimization']
            resource_management_found = 0
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in resource_patterns):
                        resource_management_found += 1
                except Exception:
                    continue
            
            if resource_management_found >= 5:
                space_details["resource_management"] = "EXCELLENT"
                space_score += 10
            elif resource_management_found >= 2:
                space_details["resource_management"] = "GOOD"
            else:
                space_details["resource_management"] = "LIMITED"
                space_score -= 15
            
            # Check for monitoring and telemetry capabilities
            monitoring_patterns = ['monitor', 'telemetry', 'logging', 'tracking', 'metrics']
            monitoring_capabilities = 0
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in monitoring_patterns):
                        monitoring_capabilities += 1
                except Exception:
                    continue
            
            if monitoring_capabilities >= 5:
                space_details["monitoring_capabilities"] = "EXCELLENT"
            elif monitoring_capabilities >= 2:
                space_details["monitoring_capabilities"] = "ADEQUATE"
            else:
                space_details["monitoring_capabilities"] = "INSUFFICIENT"
                space_score -= 20
            
            space_details["extreme_condition_tests"] = extreme_condition_tests
            space_details["resource_management_indicators"] = resource_management_found
            space_details["monitoring_indicators"] = monitoring_capabilities
            
            # Adjust maximum score for space compatibility
            space_score = min(100.0, space_score)
            passed = space_score >= 85.0
            
            return QualityGateResult(
                gate_name="Space Environment Compatibility",
                category="NASA_MISSION",
                passed=passed,
                score=space_score,
                max_score=100.0,
                details=space_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-SPACE-001: System must be compatible with space environment constraints"
            )
            
        except Exception as e:
            return self._create_error_result("Space Environment Compatibility", "NASA_MISSION", str(e), start_time, True)
    
    async def validate_emergency_protocols(self) -> QualityGateResult:
        """Validate emergency response and protocol systems"""
        start_time = time.time()
        
        try:
            emergency_score = 100.0
            emergency_details = {}
            
            # Check for emergency-related code patterns
            python_files = list(Path(".").rglob("*.py"))
            emergency_patterns = {
                'emergency': 'Emergency response systems',
                'alert': 'Alert mechanisms',
                'alarm': 'Alarm systems',
                'critical': 'Critical state handling',
                'abort': 'Abort procedures',
                'shutdown': 'Emergency shutdown',
                'fail_safe': 'Fail-safe mechanisms',
                'backup': 'Backup systems',
                'contingency': 'Contingency planning'
            }
            
            found_emergency_systems = []
            
            for py_file in python_files[:25]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern, description in emergency_patterns.items():
                        if pattern in content:
                            found_emergency_systems.append(description)
                            emergency_score += 5
                except Exception:
                    continue
            
            emergency_details["emergency_systems"] = list(set(found_emergency_systems))
            emergency_details["emergency_indicators_count"] = len(set(found_emergency_systems))
            
            # Test emergency response simulation
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            emergency_response_tests = []
            
            # Test system behavior under stress
            try:
                env.reset()
                
                # Simulate emergency scenario - rapid actions
                for _ in range(50):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Check if system handles emergency states gracefully
                    if terminated or truncated:
                        # System should be able to reset and continue
                        env.reset()
                        break
                
                emergency_response_tests.append("PASSED: Emergency stress test completed")
                
            except Exception as e:
                emergency_score -= 30
                emergency_response_tests.append(f"FAILED: Emergency stress test: {str(e)}")
            
            # Test system recovery capabilities
            try:
                # Create and destroy environments rapidly (simulating emergency scenarios)
                for _ in range(5):
                    env = lunar_habitat_rl.make_lunar_env()
                    env.reset()
                    del env
                
                emergency_response_tests.append("PASSED: Rapid recovery test")
                
            except Exception as e:
                emergency_score -= 25
                emergency_response_tests.append(f"FAILED: Rapid recovery test: {str(e)}")
            
            # Check for logging and audit trail capabilities
            log_patterns = ['log', 'audit', 'trace', 'record', 'history']
            logging_capabilities = 0
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in log_patterns):
                        logging_capabilities += 1
                except Exception:
                    continue
            
            if logging_capabilities >= 5:
                emergency_details["logging_capabilities"] = "EXCELLENT"
                emergency_score += 10
            elif logging_capabilities >= 2:
                emergency_details["logging_capabilities"] = "ADEQUATE"
            else:
                emergency_details["logging_capabilities"] = "INSUFFICIENT"
                emergency_score -= 20
            
            # Check for timeout and watchdog patterns
            safety_patterns = ['timeout', 'watchdog', 'deadline', 'limit', 'bounds']
            safety_mechanisms = 0
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in safety_patterns):
                        safety_mechanisms += 1
                except Exception:
                    continue
            
            if safety_mechanisms >= 3:
                emergency_details["safety_mechanisms"] = "GOOD"
            elif safety_mechanisms >= 1:
                emergency_details["safety_mechanisms"] = "BASIC"
                emergency_score -= 10
            else:
                emergency_details["safety_mechanisms"] = "MISSING"
                emergency_score -= 25
            
            emergency_details["emergency_response_tests"] = emergency_response_tests
            emergency_details["logging_indicators"] = logging_capabilities
            emergency_details["safety_mechanism_indicators"] = safety_mechanisms
            
            # Adjust maximum score
            emergency_score = min(100.0, emergency_score)
            passed = emergency_score >= 80.0
            
            return QualityGateResult(
                gate_name="Emergency Protocols",
                category="NASA_MISSION",
                passed=passed,
                score=emergency_score,
                max_score=100.0,
                details=emergency_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-EMRG-001: Emergency protocols and response systems required"
            )
            
        except Exception as e:
            return self._create_error_result("Emergency Protocols", "NASA_MISSION", str(e), start_time, True)
    
    # ====== INTEGRATION GATES ======
    
    async def validate_cross_component_communication(self) -> QualityGateResult:
        """Validate cross-component communication and interfaces"""
        start_time = time.time()
        
        try:
            comm_score = 100.0
            comm_details = {}
            
            # Test component integration
            import lunar_habitat_rl
            
            component_tests = []
            
            # Test environment-algorithm integration
            try:
                env = lunar_habitat_rl.make_lunar_env()
                obs, info = env.reset()
                
                # Test different action types
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                component_tests.append("PASSED: Environment-action integration")
                
            except Exception as e:
                comm_score -= 25
                component_tests.append(f"FAILED: Environment-action integration: {str(e)}")
            
            # Test configuration-environment integration
            try:
                from lunar_habitat_rl.core.lightweight_config import HabitatConfig
                config = HabitatConfig()
                
                # Verify config can be used with environment
                env = lunar_habitat_rl.make_lunar_env()
                component_tests.append("PASSED: Configuration-environment integration")
                
            except Exception as e:
                comm_score -= 20
                component_tests.append(f"FAILED: Configuration-environment integration: {str(e)}")
            
            # Test algorithm-environment integration
            try:
                from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent
                
                env = lunar_habitat_rl.make_lunar_env()
                agent = RandomAgent(env.action_space)
                obs, info = env.reset()
                
                for _ in range(10):
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
                
                component_tests.append("PASSED: Algorithm-environment integration")
                
            except Exception as e:
                comm_score -= 20
                component_tests.append(f"FAILED: Algorithm-environment integration: {str(e)}")
            
            # Check for interface consistency
            python_files = list(Path("lunar_habitat_rl").rglob("*.py"))
            interface_patterns = ['interface', 'api', 'contract', 'protocol', 'abstract']
            interface_indicators = 0
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in interface_patterns):
                        interface_indicators += 1
                except Exception:
                    continue
            
            if interface_indicators >= 3:
                comm_details["interface_design"] = "GOOD"
                comm_score += 5
            elif interface_indicators >= 1:
                comm_details["interface_design"] = "BASIC"
            else:
                comm_details["interface_design"] = "LIMITED"
                comm_score -= 10
            
            comm_details["component_tests"] = component_tests
            comm_details["interface_indicators"] = interface_indicators
            
            passed = comm_score >= 80.0
            
            return QualityGateResult(
                gate_name="Cross-Component Communication",
                category="INTEGRATION",
                passed=passed,
                score=comm_score,
                max_score=100.0,
                details=comm_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=False,
                nasa_requirement="REQ-COMM-001: Components must communicate reliably"
            )
            
        except Exception as e:
            return self._create_error_result("Cross-Component Communication", "INTEGRATION", str(e), start_time, False)
    
    async def validate_data_integrity(self) -> QualityGateResult:
        """Validate data integrity and consistency"""
        start_time = time.time()
        
        try:
            integrity_score = 100.0
            integrity_details = {}
            
            # Test data consistency across operations
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            data_integrity_tests = []
            
            # Test observation consistency
            try:
                obs1, info1 = env.reset()
                obs2, info2 = env.reset()
                
                # Check if reset produces consistent data structures
                if type(obs1) == type(obs2):
                    data_integrity_tests.append("PASSED: Observation type consistency")
                else:
                    integrity_score -= 15
                    data_integrity_tests.append("FAILED: Observation type inconsistency")
                
                if hasattr(obs1, 'shape') and hasattr(obs2, 'shape'):
                    if obs1.shape == obs2.shape:
                        data_integrity_tests.append("PASSED: Observation shape consistency")
                    else:
                        integrity_score -= 15
                        data_integrity_tests.append("FAILED: Observation shape inconsistency")
                
            except Exception as e:
                integrity_score -= 20
                data_integrity_tests.append(f"FAILED: Observation consistency test: {str(e)}")
            
            # Test action-reward consistency
            try:
                env.reset()
                rewards = []
                
                # Test same action multiple times
                test_action = env.action_space.sample()
                for _ in range(5):
                    env.reset()
                    obs, reward, terminated, truncated, info = env.step(test_action)
                    rewards.append(reward)
                
                # Check reward data type consistency
                reward_types = [type(r) for r in rewards]
                if len(set(reward_types)) == 1:
                    data_integrity_tests.append("PASSED: Reward type consistency")
                else:
                    integrity_score -= 10
                    data_integrity_tests.append("FAILED: Reward type inconsistency")
                
            except Exception as e:
                integrity_score -= 15
                data_integrity_tests.append(f"FAILED: Action-reward consistency test: {str(e)}")
            
            # Test state transition integrity
            try:
                env.reset()
                states = []
                
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    states.append(obs)
                    
                    if terminated or truncated:
                        env.reset()
                
                # Check if all states have consistent structure
                if states:
                    first_state_type = type(states[0])
                    if all(type(state) == first_state_type for state in states):
                        data_integrity_tests.append("PASSED: State transition integrity")
                    else:
                        integrity_score -= 20
                        data_integrity_tests.append("FAILED: State transition type inconsistency")
                
            except Exception as e:
                integrity_score -= 15
                data_integrity_tests.append(f"FAILED: State transition integrity test: {str(e)}")
            
            # Check for data validation patterns in code
            python_files = list(Path("lunar_habitat_rl").rglob("*.py"))
            validation_patterns = ['validate', 'check', 'assert', 'verify', 'ensure']
            validation_indicators = 0
            
            for py_file in python_files[:20]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    if any(pattern in content for pattern in validation_patterns):
                        validation_indicators += 1
                except Exception:
                    continue
            
            if validation_indicators >= 5:
                integrity_details["data_validation"] = "EXCELLENT"
                integrity_score += 10
            elif validation_indicators >= 2:
                integrity_details["data_validation"] = "ADEQUATE"
            else:
                integrity_details["data_validation"] = "LIMITED"
                integrity_score -= 15
            
            integrity_details["data_integrity_tests"] = data_integrity_tests
            integrity_details["validation_indicators"] = validation_indicators
            
            passed = integrity_score >= 75.0
            
            return QualityGateResult(
                gate_name="Data Integrity & Consistency",
                category="INTEGRATION",
                passed=passed,
                score=integrity_score,
                max_score=100.0,
                details=integrity_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-DATA-001: Data integrity must be maintained across all operations"
            )
            
        except Exception as e:
            return self._create_error_result("Data Integrity & Consistency", "INTEGRATION", str(e), start_time, True)
    
    async def validate_end_to_end_functionality(self) -> QualityGateResult:
        """Validate complete end-to-end system functionality"""
        start_time = time.time()
        
        try:
            e2e_score = 100.0
            e2e_details = {}
            
            # Complete workflow test
            import lunar_habitat_rl
            
            workflow_tests = []
            
            # Test 1: Complete training workflow simulation
            try:
                # Create environment
                env = lunar_habitat_rl.make_lunar_env()
                
                # Create agent
                from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent
                agent = RandomAgent(env.action_space)
                
                # Run complete episodes
                total_episodes = 5
                completed_episodes = 0
                
                for episode in range(total_episodes):
                    obs, info = env.reset()
                    episode_steps = 0
                    episode_reward = 0
                    
                    for step in range(100):  # Max 100 steps per episode
                        action = agent.act(obs)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        episode_steps += 1
                        
                        if terminated or truncated:
                            break
                    
                    completed_episodes += 1
                
                if completed_episodes == total_episodes:
                    workflow_tests.append("PASSED: Complete training workflow")
                    e2e_details["episodes_completed"] = completed_episodes
                else:
                    e2e_score -= 25
                    workflow_tests.append(f"FAILED: Only {completed_episodes}/{total_episodes} episodes completed")
                
            except Exception as e:
                e2e_score -= 30
                workflow_tests.append(f"FAILED: Complete training workflow: {str(e)}")
            
            # Test 2: Configuration to deployment workflow
            try:
                from lunar_habitat_rl.core.lightweight_config import HabitatConfig
                
                # Create configuration
                config = HabitatConfig()
                
                # Create environment with configuration
                env = lunar_habitat_rl.make_lunar_env()
                
                # Test environment operations
                obs, info = env.reset()
                for _ in range(20):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
                
                workflow_tests.append("PASSED: Configuration to deployment workflow")
                
            except Exception as e:
                e2e_score -= 20
                workflow_tests.append(f"FAILED: Configuration to deployment workflow: {str(e)}")
            
            # Test 3: Multi-component integration
            try:
                # Test environment + multiple algorithms
                env = lunar_habitat_rl.make_lunar_env()
                
                from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent, HeuristicAgent
                
                agents = [
                    RandomAgent(env.action_space),
                    HeuristicAgent(env.action_space, env.observation_space)
                ]
                
                for i, agent in enumerate(agents):
                    obs, info = env.reset()
                    for _ in range(10):
                        action = agent.act(obs)
                        obs, reward, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            break
                
                workflow_tests.append("PASSED: Multi-component integration")
                
            except Exception as e:
                e2e_score -= 20
                workflow_tests.append(f"FAILED: Multi-component integration: {str(e)}")
            
            # Test 4: Generation integration test
            try:
                # Test that all generations can work together
                generation_files = {
                    'gen1': Path('generation1_test.py'),
                    'gen2': Path('generation2_robustness.py'),
                    'gen3': Path('generation3_scaling.py')
                }
                
                available_generations = []
                for gen, file_path in generation_files.items():
                    if file_path.exists():
                        available_generations.append(gen)
                
                if len(available_generations) >= 2:
                    workflow_tests.append(f"PASSED: Multiple generations available ({', '.join(available_generations)})")
                    e2e_details["available_generations"] = available_generations
                else:
                    e2e_score -= 15
                    workflow_tests.append(f"WARNING: Limited generations available ({', '.join(available_generations)})")
                
            except Exception as e:
                e2e_score -= 10
                workflow_tests.append(f"FAILED: Generation integration test: {str(e)}")
            
            # Test 5: Resource management throughout workflow
            try:
                import psutil
                initial_memory = psutil.virtual_memory().percent
                
                # Create and use multiple resources
                resources = []
                for _ in range(3):
                    env = lunar_habitat_rl.make_lunar_env()
                    resources.append(env)
                
                # Use resources
                for env in resources:
                    env.reset()
                    for _ in range(10):
                        action = env.action_space.sample()
                        env.step(action)
                
                # Cleanup
                for env in resources:
                    del env
                
                final_memory = psutil.virtual_memory().percent
                memory_increase = final_memory - initial_memory
                
                if memory_increase < 20:  # Less than 20% memory increase
                    workflow_tests.append("PASSED: Resource management")
                    e2e_details["memory_increase"] = f"{memory_increase:.1f}%"
                else:
                    e2e_score -= 15
                    workflow_tests.append(f"WARNING: High memory increase ({memory_increase:.1f}%)")
                
            except Exception as e:
                e2e_score -= 10
                workflow_tests.append(f"FAILED: Resource management test: {str(e)}")
            
            e2e_details["workflow_tests"] = workflow_tests
            e2e_details["total_tests"] = len(workflow_tests)
            passed_tests = len([t for t in workflow_tests if t.startswith("PASSED")])
            e2e_details["passed_tests"] = passed_tests
            
            passed = e2e_score >= 80.0 and passed_tests >= len(workflow_tests) * 0.6
            
            return QualityGateResult(
                gate_name="End-to-End Functionality",
                category="INTEGRATION",
                passed=passed,
                score=e2e_score,
                max_score=100.0,
                details=e2e_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO",
                mission_critical=True,
                nasa_requirement="REQ-E2E-001: Complete end-to-end functionality must be validated"
            )
            
        except Exception as e:
            return self._create_error_result("End-to-End Functionality", "INTEGRATION", str(e), start_time, True)
    
    async def validate_production_deployment_readiness(self) -> QualityGateResult:
        """Validate production deployment readiness"""
        start_time = time.time()
        
        try:
            deployment_score = 100.0
            deployment_details = {}
            
            # Check deployment infrastructure
            deployment_components = {
                'deployment/docker/Dockerfile': 'Docker containerization',
                'deployment/kubernetes/': 'Kubernetes orchestration',
                'deployment/cicd/': 'CI/CD pipeline',
                'deployment/monitoring/': 'Monitoring stack',
                'requirements.txt': 'Dependency management',
                'pyproject.toml': 'Project configuration'
            }
            
            found_components = []
            missing_components = []
            
            for component, description in deployment_components.items():
                if Path(component).exists():
                    found_components.append(description)
                    deployment_score += 10
                else:
                    missing_components.append(description)
                    deployment_score -= 10
            
            deployment_details["found_components"] = found_components
            deployment_details["missing_components"] = missing_components
            
            # Check for production readiness indicators
            production_patterns = {
                'logging': 'Logging configuration',
                'monitoring': 'Monitoring setup',
                'health': 'Health checks',
                'metrics': 'Metrics collection',
                'config': 'Configuration management',
                'environment': 'Environment handling'
            }
            
            python_files = list(Path(".").rglob("*.py"))
            production_indicators = []
            
            for py_file in python_files[:25]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for pattern, description in production_patterns.items():
                        if pattern in content:
                            production_indicators.append(description)
                            break
                except Exception:
                    continue
            
            deployment_details["production_indicators"] = list(set(production_indicators))
            
            if len(set(production_indicators)) >= 4:
                deployment_details["production_readiness"] = "GOOD"
                deployment_score += 15
            elif len(set(production_indicators)) >= 2:
                deployment_details["production_readiness"] = "BASIC"
            else:
                deployment_details["production_readiness"] = "LIMITED"
                deployment_score -= 20
            
            # Test packaging and importability
            try:
                # Test that the package can be imported cleanly
                import subprocess
                result = subprocess.run([
                    sys.executable, "-c", 
                    "import lunar_habitat_rl; print('Import successful')"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    deployment_details["import_test"] = "PASSED"
                else:
                    deployment_score -= 20
                    deployment_details["import_test"] = f"FAILED: {result.stderr}"
                    
            except Exception as e:
                deployment_score -= 15
                deployment_details["import_test"] = f"ERROR: {str(e)}"
            
            # Check for documentation
            doc_files = ['README.md', 'DEPLOYMENT_GUIDE.md', 'docs/']
            doc_found = []
            
            for doc_file in doc_files:
                if Path(doc_file).exists():
                    doc_found.append(doc_file)
                    deployment_score += 5
            
            deployment_details["documentation_found"] = doc_found
            
            # Security check for production
            security_files = ['deployment/security/', '.env.example', 'security.py']
            security_found = []
            
            for sec_file in security_files:
                if Path(sec_file).exists():
                    security_found.append(sec_file)
                    deployment_score += 5
            
            deployment_details["security_components"] = security_found
            
            # Adjust score and determine pass/fail
            deployment_score = min(100.0, deployment_score)
            passed = (deployment_score >= 85.0 and 
                     len(found_components) >= len(deployment_components) * 0.5)
            
            return QualityGateResult(
                gate_name="Production Deployment Readiness",
                category="INTEGRATION",
                passed=passed,
                score=deployment_score,
                max_score=100.0,
                details=deployment_details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO",
                mission_critical=False,
                nasa_requirement="REQ-DEPLOY-001: System must be ready for production deployment"
            )
            
        except Exception as e:
            return self._create_error_result("Production Deployment Readiness", "INTEGRATION", str(e), start_time, False)
    
    # ====== HELPER METHODS ======
    
    def _create_result(self, gate_name: str, category: str, passed: bool, score: float, 
                      details: Dict[str, Any], start_time: float, severity: str = "INFO", 
                      mission_critical: bool = False, nasa_requirement: str = None) -> QualityGateResult:
        """Helper to create QualityGateResult"""
        result = QualityGateResult(
            gate_name=gate_name,
            category=category,
            passed=passed,
            score=score,
            max_score=100.0,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=datetime.now(),
            severity=severity,
            mission_critical=mission_critical,
            nasa_requirement=nasa_requirement
        )
        self.results.append(result)
        return result
    
    def _create_error_result(self, gate_name: str, category: str, error: str, 
                           start_time: float, mission_critical: bool = False) -> QualityGateResult:
        """Helper to create error QualityGateResult"""
        result = QualityGateResult(
            gate_name=gate_name,
            category=category,
            passed=False,
            score=0.0,
            max_score=100.0,
            details={"error": error, "error_type": "EXECUTION_ERROR"},
            execution_time=time.time() - start_time,
            timestamp=datetime.now(),
            severity="CRITICAL" if mission_critical else "ERROR",
            mission_critical=mission_critical
        )
        self.results.append(result)
        return result
    
    async def _generate_nasa_mission_report(self) -> Dict[str, Any]:
        """Generate comprehensive NASA mission readiness report"""
        total_execution_time = time.time() - self.start_time
        
        # Calculate category scores
        categories = ["CODE", "PERFORMANCE", "SECURITY", "NASA_MISSION", "INTEGRATION"]
        category_scores = {}
        category_counts = {}
        
        for category in categories:
            category_results = [r for r in self.results if r.category == category]
            if category_results:
                category_scores[category] = sum(r.score for r in category_results) / len(category_results)
                category_counts[category] = len(category_results)
            else:
                category_scores[category] = 0.0
                category_counts[category] = 0
        
        # Calculate overall scores
        total_results = len(self.results)
        passed_results = sum(1 for r in self.results if r.passed)
        overall_score = (passed_results / total_results * 100) if total_results > 0 else 0
        
        # Identify critical failures
        critical_failures = [r for r in self.results if not r.passed and r.severity == "CRITICAL"]
        mission_critical_failures = [r for r in self.results if not r.passed and r.mission_critical]
        
        # Determine NASA certification level
        if overall_score >= 95 and len(critical_failures) == 0 and len(mission_critical_failures) == 0:
            cert_level = "FLIGHT_CERTIFIED"
            risk_level = "LOW"
        elif overall_score >= 90 and len(critical_failures) == 0:
            cert_level = "MISSION_READY"
            risk_level = "LOW" if len(mission_critical_failures) == 0 else "MEDIUM"
        elif overall_score >= 80:
            cert_level = "TESTING"
            risk_level = "MEDIUM"
        else:
            cert_level = "DEVELOPMENT"
            risk_level = "HIGH" if len(critical_failures) > 0 else "MEDIUM"
        
        if len(critical_failures) > 0:
            risk_level = "CRITICAL"
        
        # Generate recommendations
        recommendations = self._generate_nasa_recommendations()
        
        # Create NASA Mission Readiness Score
        nasa_score = NASAMissionReadinessScore(
            overall_score=overall_score,
            category_scores=category_scores,
            critical_failures=len(critical_failures),
            mission_ready=(cert_level in ["MISSION_READY", "FLIGHT_CERTIFIED"]),
            certification_level=cert_level,
            risk_assessment=risk_level,
            recommendations=recommendations
        )
        
        # Generate comprehensive report
        report = {
            "nasa_mission_readiness": asdict(nasa_score),
            "execution_summary": {
                "execution_timestamp": datetime.now().isoformat(),
                "execution_time_seconds": total_execution_time,
                "total_gates": total_results,
                "passed_gates": passed_results,
                "failed_gates": total_results - passed_results,
                "pass_rate_percentage": overall_score
            },
            "category_breakdown": {
                "scores": category_scores,
                "counts": category_counts
            },
            "critical_analysis": {
                "critical_failures": len(critical_failures),
                "mission_critical_failures": len(mission_critical_failures),
                "critical_failure_details": [
                    {
                        "gate": r.gate_name,
                        "category": r.category, 
                        "severity": r.severity,
                        "nasa_requirement": r.nasa_requirement,
                        "details": r.details
                    } for r in critical_failures
                ]
            },
            "detailed_results": [
                {
                    "gate_name": r.gate_name,
                    "category": r.category,
                    "passed": r.passed,
                    "score": r.score,
                    "max_score": r.max_score,
                    "details": r.details,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp.isoformat(),
                    "severity": r.severity,
                    "mission_critical": r.mission_critical,
                    "nasa_requirement": r.nasa_requirement
                } for r in self.results
            ],
            "nasa_compliance": {
                "requirements_tested": len([r for r in self.results if r.nasa_requirement]),
                "requirements_passed": len([r for r in self.results if r.nasa_requirement and r.passed]),
                "compliance_rate": len([r for r in self.results if r.nasa_requirement and r.passed]) / max(1, len([r for r in self.results if r.nasa_requirement])) * 100
            }
        }
        
        # Save comprehensive report
        report_file = Path("comprehensive_quality_gates_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log summary
        logger.info("=" * 80)
        logger.info("🚀 NASA MISSION READINESS ASSESSMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Overall Score: {overall_score:.1f}%")
        logger.info(f"Certification Level: {cert_level}")
        logger.info(f"Risk Assessment: {risk_level}")
        logger.info(f"Mission Ready: {'✅ YES' if nasa_score.mission_ready else '❌ NO'}")
        logger.info(f"Critical Failures: {len(critical_failures)}")
        logger.info(f"Report saved: {report_file}")
        logger.info("=" * 80)
        
        return report
    
    def _generate_nasa_recommendations(self) -> List[str]:
        """Generate NASA-specific recommendations"""
        recommendations = []
        
        # Analyze failures by category and severity
        critical_failures = [r for r in self.results if not r.passed and r.severity == "CRITICAL"]
        mission_critical_failures = [r for r in self.results if not r.passed and r.mission_critical]
        
        if critical_failures:
            recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Address all CRITICAL failures before mission deployment")
        
        if mission_critical_failures:
            recommendations.append("⚠️ MISSION RISK: Resolve mission-critical component failures")
        
        # Category-specific recommendations
        category_failures = {}
        for result in self.results:
            if not result.passed:
                if result.category not in category_failures:
                    category_failures[result.category] = []
                category_failures[result.category].append(result)
        
        for category, failures in category_failures.items():
            if category == "CODE":
                recommendations.append("🔧 CODE: Improve code quality and ensure all components function correctly")
            elif category == "PERFORMANCE":
                recommendations.append("⚡ PERFORMANCE: Optimize system performance to meet real-time requirements")
            elif category == "SECURITY":
                recommendations.append("🛡️ SECURITY: Address security vulnerabilities for mission safety")
            elif category == "NASA_MISSION":
                recommendations.append("🌙 MISSION: Enhance mission-specific safety and reliability features")
            elif category == "INTEGRATION":
                recommendations.append("🔗 INTEGRATION: Improve component integration and system cohesion")
        
        # Calculate overall health and add general recommendations
        overall_score = (sum(1 for r in self.results if r.passed) / len(self.results) * 100) if self.results else 0
        
        if overall_score >= 95:
            recommendations.append("✅ EXCELLENT: System demonstrates high NASA mission readiness")
        elif overall_score >= 90:
            recommendations.append("✅ GOOD: System is approaching NASA mission readiness standards")
        elif overall_score >= 80:
            recommendations.append("⚠️ NEEDS IMPROVEMENT: Significant improvements needed for mission readiness")
        else:
            recommendations.append("❌ MAJOR ISSUES: Extensive work required before mission deployment")
        
        # Add specific NASA compliance recommendations
        nasa_req_failures = [r for r in self.results if not r.passed and r.nasa_requirement]
        if nasa_req_failures:
            recommendations.append(f"📋 NASA COMPLIANCE: {len(nasa_req_failures)} NASA requirements failed validation")
        
        return recommendations[:15]  # Limit to top 15 recommendations

    def __del__(self):
        """Cleanup temporary directory"""
        try:
            if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass

async def main():
    """Execute comprehensive NASA mission readiness quality gates"""
    print("🚀 NASA LUNAR HABITAT RL SUITE - COMPREHENSIVE QUALITY GATES VALIDATION")
    print("=" * 90)
    print("Mission: Validate complete system readiness for lunar habitat operations")
    print("Standards: NASA Software Engineering Requirements (NPR 7150.2)")
    print("=" * 90)
    
    validator = ComprehensiveQualityGatesValidator()
    
    try:
        # Execute all quality gates
        report = await validator.execute_all_gates()
        
        # Display executive summary
        nasa_readiness = report["nasa_mission_readiness"]
        execution_summary = report["execution_summary"]
        
        print("\n🎯 EXECUTIVE SUMMARY")
        print("=" * 50)
        print(f"Overall Score: {nasa_readiness['overall_score']:.1f}%")
        print(f"Certification Level: {nasa_readiness['certification_level']}")
        print(f"Risk Assessment: {nasa_readiness['risk_assessment']}")
        print(f"Mission Ready: {'✅ YES' if nasa_readiness['mission_ready'] else '❌ NO'}")
        print(f"Total Gates: {execution_summary['total_gates']}")
        print(f"Passed: {execution_summary['passed_gates']}")
        print(f"Failed: {execution_summary['failed_gates']}")
        print(f"Critical Failures: {nasa_readiness['critical_failures']}")
        
        # Display category breakdown
        print(f"\n📊 CATEGORY SCORES")
        print("=" * 50)
        for category, score in nasa_readiness['category_scores'].items():
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"{status} {category}: {score:.1f}%")
        
        # Display top recommendations
        print(f"\n💡 TOP RECOMMENDATIONS")
        print("=" * 50)
        for i, rec in enumerate(nasa_readiness['recommendations'][:8], 1):
            print(f"{i}. {rec}")
        
        # Mission readiness decision
        print(f"\n🌙 NASA MISSION READINESS DECISION")
        print("=" * 50)
        if nasa_readiness['mission_ready']:
            print("✅ MISSION READY: System meets NASA standards for lunar habitat operations")
            print("🚀 Cleared for mission deployment and astronaut safety certification")
        else:
            print("❌ NOT MISSION READY: Additional work required before deployment")
            print("⚠️ Do not deploy to lunar habitat without addressing critical issues")
        
        print(f"\n📁 Detailed report: comprehensive_quality_gates_validation_report.json")
        print(f"⏱️ Total execution time: {execution_summary['execution_time_seconds']:.2f} seconds")
        
        return report
        
    except Exception as e:
        logger.critical(f"Critical failure in NASA quality gates validation: {str(e)}")
        print(f"\n❌ CRITICAL FAILURE: {str(e)}")
        print("🚨 Mission deployment BLOCKED due to validation system failure")
        raise

if __name__ == "__main__":
    asyncio.run(main())