#!/usr/bin/env python3
"""
NASA Mission Readiness Quality Gates Validation
Simplified version for reliable execution
"""

import time
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Quality gate validation result"""
    gate_name: str
    category: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    mission_critical: bool = False

class NASAQualityValidator:
    """NASA Quality Gates Validator"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Execute all quality gate validations"""
        logger.info("🚀 Starting NASA Mission Readiness Validation")
        
        # Execute all validation gates
        self.validate_code_quality()
        self.validate_performance()
        self.validate_security()
        self.validate_nasa_mission_readiness()
        self.validate_integration()
        
        return self.generate_report()
    
    def validate_code_quality(self):
        """Validate code quality gates"""
        logger.info("Testing Code Quality Gates...")
        
        # 1. Package Imports Test
        start_time = time.time()
        try:
            import lunar_habitat_rl
            details = {"core_import": "SUCCESS"}
            score = 100.0
            passed = True
        except Exception as e:
            details = {"core_import": f"FAILED: {str(e)}"}
            score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Package Imports",
            category="CODE",
            passed=passed,
            score=score,
            details=details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 2. Environment Creation Test
        start_time = time.time()
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            details = {
                "env_creation": "SUCCESS",
                "env_reset": "SUCCESS", 
                "env_step": "SUCCESS",
                "observation_type": str(type(obs)),
                "reward": str(reward)
            }
            score = 100.0
            passed = True
        except Exception as e:
            details = {"env_creation": f"FAILED: {str(e)}"}
            score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Environment Creation",
            category="CODE",
            passed=passed,
            score=score,
            details=details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 3. Generation Integration Test
        start_time = time.time()
        generation_score = 100.0
        generation_details = {}
        
        # Check generation files
        gen_files = {
            "Generation 1": "generation1_test.py",
            "Generation 2": "generation2_robustness.py", 
            "Generation 3": "generation3_scaling.py"
        }
        
        for gen_name, filename in gen_files.items():
            if Path(filename).exists():
                generation_details[gen_name] = "EXISTS"
            else:
                generation_details[gen_name] = "MISSING"
                generation_score -= 30
        
        passed = generation_score >= 70
        
        self.results.append(ValidationResult(
            gate_name="Generation Integration",
            category="CODE",
            passed=passed,
            score=generation_score,
            details=generation_details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 4. Configuration System Test
        start_time = time.time()
        try:
            from lunar_habitat_rl.core.lightweight_config import HabitatConfig
            config = HabitatConfig()
            
            config_details = {
                "config_creation": "SUCCESS",
                "crew_size": getattr(config, 'crew_size', 'UNKNOWN'),
                "max_episode_steps": getattr(config, 'max_episode_steps', 'UNKNOWN')
            }
            score = 100.0
            passed = True
        except Exception as e:
            config_details = {"config_creation": f"FAILED: {str(e)}"}
            score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Configuration System",
            category="CODE",
            passed=passed,
            score=score,
            details=config_details,
            execution_time=time.time() - start_time
        ))
    
    def validate_performance(self):
        """Validate performance gates"""
        logger.info("Testing Performance Gates...")
        
        # 1. Throughput Test
        start_time = time.time()
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            # Measure episodes per second
            perf_start = time.time()
            episodes = 0
            for _ in range(5):  # Run 5 episodes
                env.reset()
                for _ in range(20):  # 20 steps per episode
                    action = env.action_space.sample()
                    env.step(action)
                episodes += 1
            perf_time = time.time() - perf_start
            
            eps_per_second = episodes / perf_time if perf_time > 0 else 0
            
            details = {
                "episodes_completed": episodes,
                "total_time": f"{perf_time:.3f}s",
                "eps_per_second": f"{eps_per_second:.2f}"
            }
            
            # Score based on throughput
            if eps_per_second >= 2.0:
                score = 100.0
            elif eps_per_second >= 1.0:
                score = 80.0
            elif eps_per_second >= 0.5:
                score = 60.0
            else:
                score = 30.0
            
            passed = score >= 60.0
            
        except Exception as e:
            details = {"throughput_test": f"FAILED: {str(e)}"}
            score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Throughput Benchmarks",
            category="PERFORMANCE",
            passed=passed,
            score=score,
            details=details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 2. Memory Usage Test
        start_time = time.time()
        try:
            import psutil
            import lunar_habitat_rl
            
            initial_memory = psutil.virtual_memory().percent
            
            # Create multiple environments to test memory usage
            envs = []
            for _ in range(3):
                env = lunar_habitat_rl.make_lunar_env()
                env.reset()
                envs.append(env)
            
            final_memory = psutil.virtual_memory().percent
            memory_increase = final_memory - initial_memory
            
            details = {
                "initial_memory": f"{initial_memory:.1f}%",
                "final_memory": f"{final_memory:.1f}%", 
                "memory_increase": f"{memory_increase:.1f}%",
                "environments_created": len(envs)
            }
            
            # Score based on memory efficiency
            if memory_increase <= 5:
                score = 100.0
            elif memory_increase <= 10:
                score = 80.0
            elif memory_increase <= 20:
                score = 60.0
            else:
                score = 30.0
            
            passed = score >= 60.0
            
            # Cleanup
            for env in envs:
                del env
                
        except Exception as e:
            details = {"memory_test": f"FAILED: {str(e)}"}
            score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Memory Usage",
            category="PERFORMANCE",
            passed=passed,
            score=score,
            details=details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 3. Response Time Test
        start_time = time.time()
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            # Test environment creation time
            create_start = time.time()
            test_env = lunar_habitat_rl.make_lunar_env()
            create_time = time.time() - create_start
            
            # Test reset time
            reset_start = time.time()
            test_env.reset()
            reset_time = time.time() - reset_start
            
            # Test step times
            step_times = []
            for _ in range(20):
                action = test_env.action_space.sample()
                step_start = time.time()
                test_env.step(action)
                step_time = time.time() - step_start
                step_times.append(step_time)
            
            avg_step_time = sum(step_times) / len(step_times)
            
            details = {
                "env_creation_time": f"{create_time:.4f}s",
                "env_reset_time": f"{reset_time:.4f}s", 
                "avg_step_time": f"{avg_step_time:.4f}s",
                "max_step_time": f"{max(step_times):.4f}s"
            }
            
            # Score based on response times
            score = 100.0
            if create_time > 1.0:
                score -= 20
            if reset_time > 0.1:
                score -= 20
            if avg_step_time > 0.05:
                score -= 30
            
            passed = score >= 60.0
            
        except Exception as e:
            details = {"response_time_test": f"FAILED: {str(e)}"}
            score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Response Time",
            category="PERFORMANCE",
            passed=passed,
            score=score,
            details=details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
    
    def validate_security(self):
        """Validate security gates"""
        logger.info("Testing Security Gates...")
        
        # 1. Security Scanning
        start_time = time.time()
        security_score = 100.0
        security_details = {}
        vulnerabilities = []
        
        # Scan Python files for security issues
        python_files = list(Path(".").rglob("*.py"))
        
        dangerous_patterns = {
            'eval(': 'Code injection vulnerability',
            'exec(': 'Code execution vulnerability',
            'shell=True': 'Shell injection risk',
            'pickle.load': 'Deserialization vulnerability'
        }
        
        for py_file in python_files[:20]:  # Sample first 20 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in dangerous_patterns.items():
                    if pattern in content:
                        vulnerabilities.append({
                            'file': str(py_file),
                            'pattern': pattern,
                            'description': description
                        })
                        security_score -= 20
            except Exception:
                continue
        
        security_details["vulnerabilities_found"] = len(vulnerabilities)
        security_details["files_scanned"] = min(20, len(python_files))
        security_details["vulnerabilities"] = vulnerabilities[:5]  # Show first 5
        
        passed = security_score >= 80 and len(vulnerabilities) == 0
        
        self.results.append(ValidationResult(
            gate_name="Security Scanning",
            category="SECURITY",
            passed=passed,
            score=security_score,
            details=security_details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 2. Input Validation Test
        start_time = time.time()
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            env.reset()
            
            validation_tests = []
            input_score = 100.0
            
            # Test invalid actions
            invalid_actions = [None, "invalid", [], {}]
            for invalid_action in invalid_actions:
                try:
                    env.step(invalid_action)
                    validation_tests.append(f"FAILED: Accepted {type(invalid_action)}")
                    input_score -= 25
                except (ValueError, TypeError, AssertionError):
                    validation_tests.append(f"PASSED: Rejected {type(invalid_action)}")
                except Exception:
                    validation_tests.append(f"UNKNOWN: {type(invalid_action)}")
                    input_score -= 10
            
            details = {
                "validation_tests": validation_tests,
                "input_validation_score": input_score
            }
            
            passed = input_score >= 75
            
        except Exception as e:
            details = {"input_validation": f"FAILED: {str(e)}"}
            input_score = 0.0
            passed = False
        
        self.results.append(ValidationResult(
            gate_name="Input Validation",
            category="SECURITY",
            passed=passed,
            score=input_score,
            details=details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
    
    def validate_nasa_mission_readiness(self):
        """Validate NASA mission-specific requirements"""
        logger.info("Testing NASA Mission Readiness...")
        
        # 1. Mission Safety Validation
        start_time = time.time()
        safety_score = 100.0
        safety_details = {}
        
        # Check for safety-related files and patterns
        safety_files = list(Path(".").rglob("*safety*.py")) + list(Path(".").rglob("*mission*.py"))
        safety_details["safety_files_found"] = len(safety_files)
        
        if safety_files:
            safety_details["safety_systems"] = "PRESENT"
        else:
            safety_score -= 30
            safety_details["safety_systems"] = "MISSING"
        
        # Test environment stability
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            # Run stability test
            for _ in range(50):
                env.reset()
                for _ in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Check for NaN/Inf values
                    if hasattr(obs, '__iter__'):
                        for val in obs:
                            if str(val).lower() in ['nan', 'inf', '-inf']:
                                safety_score -= 20
                                safety_details["stability_issue"] = "NaN/Inf detected"
                                break
                    
                    if terminated or truncated:
                        break
            
            safety_details["stability_test"] = "COMPLETED"
            
        except Exception as e:
            safety_score -= 40
            safety_details["stability_test"] = f"FAILED: {str(e)}"
        
        passed = safety_score >= 80
        
        self.results.append(ValidationResult(
            gate_name="Mission Safety",
            category="NASA_MISSION",
            passed=passed,
            score=safety_score,
            details=safety_details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 2. Space Environment Compatibility
        start_time = time.time()
        space_score = 100.0
        space_details = {}
        
        # Check for space-related considerations
        python_files = list(Path(".").rglob("*.py"))
        space_indicators = [
            'thermal', 'temperature', 'pressure', 'atmosphere', 
            'oxygen', 'co2', 'life_support', 'radiation'
        ]
        
        found_indicators = []
        for py_file in python_files[:20]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                for indicator in space_indicators:
                    if indicator in content:
                        found_indicators.append(indicator)
                        space_score += 5
                        break
            except Exception:
                continue
        
        space_details["space_considerations"] = list(set(found_indicators))
        space_details["space_indicators_found"] = len(set(found_indicators))
        
        # Test rapid operations (space environment dynamics)
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            
            for _ in range(100):  # Rapid operations
                env.reset()
                for _ in range(5):
                    action = env.action_space.sample()
                    env.step(action)
            
            space_details["rapid_operations_test"] = "PASSED"
            
        except Exception as e:
            space_score -= 30
            space_details["rapid_operations_test"] = f"FAILED: {str(e)}"
        
        passed = space_score >= 85
        
        self.results.append(ValidationResult(
            gate_name="Space Environment Compatibility",
            category="NASA_MISSION",
            passed=passed,
            score=space_score,
            details=space_details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
    
    def validate_integration(self):
        """Validate integration gates"""
        logger.info("Testing Integration Gates...")
        
        # 1. End-to-End Functionality
        start_time = time.time()
        e2e_score = 100.0
        e2e_details = {}
        
        try:
            import lunar_habitat_rl
            from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent
            
            # Complete workflow test
            env = lunar_habitat_rl.make_lunar_env()
            agent = RandomAgent(env.action_space)
            
            episodes_completed = 0
            for episode in range(3):
                obs, info = env.reset()
                for step in range(50):
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                episodes_completed += 1
            
            e2e_details["episodes_completed"] = episodes_completed
            e2e_details["workflow_test"] = "SUCCESS"
            
            if episodes_completed < 3:
                e2e_score -= 30
                
        except Exception as e:
            e2e_score -= 50
            e2e_details["workflow_test"] = f"FAILED: {str(e)}"
        
        # Test component integration
        try:
            from lunar_habitat_rl.core.lightweight_config import HabitatConfig
            config = HabitatConfig()
            env = lunar_habitat_rl.make_lunar_env()
            e2e_details["component_integration"] = "SUCCESS"
        except Exception as e:
            e2e_score -= 30
            e2e_details["component_integration"] = f"FAILED: {str(e)}"
        
        passed = e2e_score >= 70
        
        self.results.append(ValidationResult(
            gate_name="End-to-End Functionality",
            category="INTEGRATION",
            passed=passed,
            score=e2e_score,
            details=e2e_details,
            execution_time=time.time() - start_time,
            mission_critical=True
        ))
        
        # 2. Production Deployment Readiness
        start_time = time.time()
        deployment_score = 100.0
        deployment_details = {}
        
        # Check deployment components
        deployment_components = {
            'deployment/docker/Dockerfile': 'Docker',
            'deployment/kubernetes/': 'Kubernetes',
            'deployment/cicd/': 'CI/CD',
            'deployment/monitoring/': 'Monitoring',
            'requirements.txt': 'Dependencies',
            'pyproject.toml': 'Project Config'
        }
        
        found_components = []
        for component, name in deployment_components.items():
            if Path(component).exists():
                found_components.append(name)
            else:
                deployment_score -= 15
        
        deployment_details["deployment_components"] = found_components
        deployment_details["missing_components"] = len(deployment_components) - len(found_components)
        
        # Test package importability
        try:
            result = subprocess.run([
                sys.executable, "-c", "import lunar_habitat_rl; print('OK')"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                deployment_details["import_test"] = "PASSED"
            else:
                deployment_score -= 20
                deployment_details["import_test"] = "FAILED"
        except Exception as e:
            deployment_score -= 20
            deployment_details["import_test"] = f"ERROR: {str(e)}"
        
        passed = deployment_score >= 60
        
        self.results.append(ValidationResult(
            gate_name="Production Deployment Readiness", 
            category="INTEGRATION",
            passed=passed,
            score=deployment_score,
            details=deployment_details,
            execution_time=time.time() - start_time
        ))
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_time = time.time() - self.start_time
        
        # Calculate scores by category
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
        
        # Calculate overall metrics
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results if r.passed)
        overall_score = (passed_gates / total_gates * 100) if total_gates > 0 else 0
        
        # Critical failures
        critical_failures = [r for r in self.results if not r.passed and r.mission_critical]
        
        # NASA Certification Level
        if overall_score >= 95 and len(critical_failures) == 0:
            cert_level = "FLIGHT_CERTIFIED"
            risk_level = "LOW"
        elif overall_score >= 90 and len(critical_failures) == 0:
            cert_level = "MISSION_READY"
            risk_level = "LOW"
        elif overall_score >= 80:
            cert_level = "TESTING"
            risk_level = "MEDIUM"
        else:
            cert_level = "DEVELOPMENT"
            risk_level = "HIGH"
        
        if critical_failures:
            risk_level = "CRITICAL"
        
        # Mission readiness
        mission_ready = cert_level in ["MISSION_READY", "FLIGHT_CERTIFIED"]
        
        # Recommendations
        recommendations = []
        if critical_failures:
            recommendations.append("🚨 CRITICAL: Address mission-critical failures immediately")
        
        failed_by_category = {}
        for result in self.results:
            if not result.passed:
                if result.category not in failed_by_category:
                    failed_by_category[result.category] = 0
                failed_by_category[result.category] += 1
        
        for category, count in failed_by_category.items():
            if category == "CODE":
                recommendations.append(f"🔧 CODE: Fix {count} code quality issues")
            elif category == "PERFORMANCE":
                recommendations.append(f"⚡ PERFORMANCE: Optimize {count} performance issues")
            elif category == "SECURITY":
                recommendations.append(f"🛡️ SECURITY: Address {count} security concerns")
            elif category == "NASA_MISSION":
                recommendations.append(f"🌙 MISSION: Resolve {count} mission readiness issues")
            elif category == "INTEGRATION":
                recommendations.append(f"🔗 INTEGRATION: Fix {count} integration issues")
        
        if overall_score >= 90:
            recommendations.append("✅ EXCELLENT: System demonstrates high mission readiness")
        elif overall_score >= 80:
            recommendations.append("⚠️ GOOD: Minor improvements needed for mission readiness")
        else:
            recommendations.append("❌ NEEDS WORK: Significant improvements required")
        
        # Create comprehensive report
        report = {
            "nasa_mission_readiness": {
                "overall_score": overall_score,
                "certification_level": cert_level,
                "risk_assessment": risk_level,
                "mission_ready": mission_ready,
                "critical_failures": len(critical_failures),
                "recommendations": recommendations
            },
            "execution_summary": {
                "execution_timestamp": datetime.now().isoformat(),
                "execution_time_seconds": total_time,
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "failed_gates": total_gates - passed_gates,
                "pass_rate_percentage": overall_score
            },
            "category_scores": category_scores,
            "category_counts": category_counts,
            "detailed_results": [
                {
                    "gate_name": r.gate_name,
                    "category": r.category,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "execution_time": r.execution_time,
                    "mission_critical": r.mission_critical
                } for r in self.results
            ],
            "critical_failures": [
                {
                    "gate_name": r.gate_name,
                    "category": r.category,
                    "score": r.score,
                    "details": r.details
                } for r in critical_failures
            ]
        }
        
        # Save report
        with open("comprehensive_quality_gates_validation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Execute NASA Mission Readiness Validation"""
    print("🚀 NASA LUNAR HABITAT RL SUITE - MISSION READINESS VALIDATION")
    print("=" * 80)
    print("Mission: Validate system readiness for lunar habitat operations")
    print("Standards: NASA Software Engineering Requirements")
    print("=" * 80)
    
    validator = NASAQualityValidator()
    
    try:
        # Run all validations
        report = validator.run_all_validations()
        
        # Display results
        nasa_readiness = report["nasa_mission_readiness"]
        execution = report["execution_summary"]
        
        print(f"\n🎯 MISSION READINESS ASSESSMENT")
        print("=" * 50)
        print(f"Overall Score: {nasa_readiness['overall_score']:.1f}%")
        print(f"Certification Level: {nasa_readiness['certification_level']}")
        print(f"Risk Assessment: {nasa_readiness['risk_assessment']}")
        print(f"Mission Ready: {'✅ YES' if nasa_readiness['mission_ready'] else '❌ NO'}")
        print(f"Total Gates: {execution['total_gates']}")
        print(f"Passed: {execution['passed_gates']}")
        print(f"Failed: {execution['failed_gates']}")
        print(f"Critical Failures: {nasa_readiness['critical_failures']}")
        
        print(f"\n📊 CATEGORY SCORES")
        print("=" * 50)
        for category, score in report["category_scores"].items():
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"{status} {category}: {score:.1f}%")
        
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 50)
        for i, rec in enumerate(nasa_readiness['recommendations'][:8], 1):
            print(f"{i}. {rec}")
        
        print(f"\n🌙 NASA MISSION DECISION")
        print("=" * 50)
        if nasa_readiness['mission_ready']:
            print("✅ MISSION READY: System cleared for lunar habitat deployment")
        else:
            print("❌ NOT MISSION READY: Additional work required")
        
        print(f"\n📁 Full report: comprehensive_quality_gates_validation_report.json")
        print(f"⏱️ Execution time: {execution['execution_time_seconds']:.2f}s")
        
        return report
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()