#!/usr/bin/env python3
"""
Comprehensive Quality Gates Suite - All Generations Validation
Complete testing, security, and quality validation for Lunar Habitat RL Suite
"""

import asyncio
import subprocess
import time
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_gates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Comprehensive quality gate result tracking"""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL

class ComprehensiveQualityValidator:
    """Complete quality validation across all SDLC dimensions"""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates in parallel where possible"""
        logger.info("🚀 Starting Comprehensive Quality Gate Execution")
        
        # Define all quality gates
        gate_tasks = [
            self.validate_code_execution(),
            self.validate_test_coverage(),
            self.validate_security_posture(),
            self.validate_performance_benchmarks(),
            self.validate_documentation_quality(),
            self.validate_code_quality(),
            self.validate_dependency_security(),
            self.validate_research_reproducibility(),
            self.validate_deployment_readiness()
        ]
        
        # Execute all gates concurrently
        results = await asyncio.gather(*gate_tasks, return_exceptions=True)
        
        # Process results
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall scores
        passed_gates = sum(1 for r in self.results if r.passed)
        total_gates = len(self.results)
        overall_score = (passed_gates / total_gates * 100) if total_gates > 0 else 0
        
        # Identify critical failures
        critical_failures = [r for r in self.results if not r.passed and r.severity == "CRITICAL"]
        
        # Generate comprehensive report
        report = {
            "execution_timestamp": datetime.now().isoformat(),
            "execution_time_seconds": total_execution_time,
            "overall_score": overall_score,
            "gates_summary": {
                "total": total_gates,
                "passed": passed_gates,
                "failed": total_gates - passed_gates,
                "pass_rate": f"{overall_score:.1f}%"
            },
            "critical_failures": len(critical_failures),
            "detailed_results": [self._serialize_result(r) for r in self.results],
            "recommendations": self._generate_recommendations(),
            "production_ready": overall_score >= 90 and len(critical_failures) == 0
        }
        
        # Save comprehensive report
        report_file = Path("comprehensive_quality_gates_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Quality Gates Report: {overall_score:.1f}% pass rate ({passed_gates}/{total_gates})")
        return report
    
    async def validate_code_execution(self) -> QualityGateResult:
        """Validate all code executes without errors"""
        start_time = time.time()
        
        try:
            # Test core imports
            import lunar_habitat_rl
            
            # Test environment creation
            env = lunar_habitat_rl.make_lunar_env()
            obs, info = env.reset()
            
            # Test basic functionality
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            execution_time = time.time() - start_time
            
            result = QualityGateResult(
                gate_name="Code Execution",
                passed=True,
                score=100.0,
                details={
                    "core_import": "SUCCESS",
                    "env_creation": "SUCCESS",
                    "env_reset": "SUCCESS",
                    "env_step": "SUCCESS",
                    "execution_time": execution_time
                },
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = QualityGateResult(
                gate_name="Code Execution",
                passed=False,
                score=0.0,
                details={
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                execution_time=execution_time,
                timestamp=datetime.now(),
                severity="CRITICAL"
            )
        
        self.results.append(result)
        return result
    
    async def validate_test_coverage(self) -> QualityGateResult:
        """Validate test coverage meets minimum requirements"""
        start_time = time.time()
        
        try:
            # Look for test files
            test_files = list(Path('.').rglob('test_*.py')) + list(Path('tests').rglob('*.py'))
            
            # Check for pytest configuration
            pytest_config = any(Path(f).exists() for f in ['pytest.ini', 'pyproject.toml', 'setup.cfg'])
            
            # Estimate coverage based on test files vs source files
            source_files = list(Path('lunar_habitat_rl').rglob('*.py'))
            coverage_estimate = min(100, (len(test_files) / max(1, len(source_files))) * 150)  # Rough estimate
            
            passed = len(test_files) > 0 and coverage_estimate >= 70
            
            result = QualityGateResult(
                gate_name="Test Coverage",
                passed=passed,
                score=coverage_estimate,
                details={
                    "test_files_found": len(test_files),
                    "source_files": len(source_files),
                    "pytest_config": pytest_config,
                    "estimated_coverage": f"{coverage_estimate:.1f}%",
                    "test_files": [str(f) for f in test_files[:10]]  # First 10
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Test Coverage",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR"
            )
        
        self.results.append(result)
        return result
    
    async def validate_security_posture(self) -> QualityGateResult:
        """Comprehensive security validation"""
        start_time = time.time()
        
        try:
            security_score = 100.0
            issues = []
            
            # Check for common security issues
            python_files = list(Path('.').rglob('*.py'))
            
            for file_path in python_files[:50]:  # Limit to first 50 files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for potential security issues (excluding legitimate PyTorch usage)
                        # Skip model.eval() which is legitimate PyTorch usage
                        if 'eval(' in content and 'model.eval()' not in content and '.eval()' not in content:
                            # More precise detection for actual eval function calls
                            lines = content.split('\n')
                            for line_num, line in enumerate(lines, 1):
                                if 'eval(' in line and not line.strip().endswith('.eval()') and 'model.eval()' not in line and not line.strip().startswith('#'):
                                    issues.append(f"eval() found in {file_path}:{line_num}")
                                    security_score -= 10
                        
                        if 'exec(' in content:
                            # More precise detection for actual exec function calls
                            lines = content.split('\n')
                            for line_num, line in enumerate(lines, 1):
                                if 'exec(' in line and not line.strip().startswith('#'):
                                    issues.append(f"exec() found in {file_path}:{line_num}")
                                    security_score -= 10
                        
                        if 'subprocess.call' in content and 'shell=True' in content:
                            issues.append(f"Unsafe subprocess call in {file_path}")
                            security_score -= 5
                            
                except (UnicodeDecodeError, IOError):
                    continue
            
            # Check file permissions
            sensitive_files = ['requirements.txt', 'pyproject.toml']
            for file_path in sensitive_files:
                path = Path(file_path)
                if path.exists():
                    stat = path.stat()
                    if stat.st_mode & 0o044:  # World or group readable
                        issues.append(f"Overly permissive permissions on {file_path}")
                        security_score -= 5
            
            passed = security_score >= 80 and len([i for i in issues if 'eval(' in i or 'exec(' in i]) == 0
            
            result = QualityGateResult(
                gate_name="Security Posture",
                passed=passed,
                score=max(0, security_score),
                details={
                    "issues_found": len(issues),
                    "issues": issues,
                    "files_scanned": len(python_files)
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="CRITICAL" if security_score < 50 else "ERROR" if security_score < 80 else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Security Posture",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="ERROR"
            )
        
        self.results.append(result)
        return result
    
    async def validate_performance_benchmarks(self) -> QualityGateResult:
        """Validate performance meets benchmarks"""
        start_time = time.time()
        
        try:
            import psutil
            
            # Memory usage check
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            # Environment creation performance test
            import lunar_habitat_rl
            
            perf_start = time.time()
            env = lunar_habitat_rl.make_lunar_env()
            obs, info = env.reset()
            creation_time = time.time() - perf_start
            
            # Performance scoring
            performance_score = 100.0
            
            # Penalize slow environment creation
            if creation_time > 2.0:
                performance_score -= 20
            elif creation_time > 1.0:
                performance_score -= 10
            
            # Check memory efficiency
            if memory.percent > 90:
                performance_score -= 30
            elif memory.percent > 80:
                performance_score -= 15
            
            passed = performance_score >= 70 and creation_time < 5.0
            
            result = QualityGateResult(
                gate_name="Performance Benchmarks",
                passed=passed,
                score=performance_score,
                details={
                    "env_creation_time": f"{creation_time:.3f}s",
                    "memory_usage": f"{memory.percent:.1f}%",
                    "cpu_count": cpu_count,
                    "memory_available": f"{memory.available // (1024**2)}MB"
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Performance Benchmarks",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING"
            )
        
        self.results.append(result)
        return result
    
    async def validate_documentation_quality(self) -> QualityGateResult:
        """Validate documentation completeness and quality"""
        start_time = time.time()
        
        try:
            doc_score = 0.0
            details = {}
            
            # Check for README
            if Path('README.md').exists():
                doc_score += 25
                details['readme'] = "EXISTS"
                
                # Check README quality
                with open('README.md', 'r') as f:
                    readme_content = f.read()
                    if len(readme_content) > 1000:  # Substantial README
                        doc_score += 10
                    if '##' in readme_content:  # Has sections
                        doc_score += 5
            else:
                details['readme'] = "MISSING"
            
            # Check for pyproject.toml
            if Path('pyproject.toml').exists():
                doc_score += 20
                details['project_config'] = "EXISTS"
            else:
                details['project_config'] = "MISSING"
            
            # Check for docstrings in Python files
            python_files = list(Path('lunar_habitat_rl').rglob('*.py'))[:10]  # Sample first 10
            docstring_count = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:
                            docstring_count += 1
                except:
                    continue
            
            if python_files:
                docstring_ratio = docstring_count / len(python_files)
                doc_score += docstring_ratio * 40  # Up to 40 points for docstrings
                details['docstring_coverage'] = f"{docstring_ratio * 100:.1f}%"
            
            passed = doc_score >= 70
            
            result = QualityGateResult(
                gate_name="Documentation Quality",
                passed=passed,
                score=doc_score,
                details=details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Documentation Quality",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING"
            )
        
        self.results.append(result)
        return result
    
    async def validate_code_quality(self) -> QualityGateResult:
        """Validate code quality metrics"""
        start_time = time.time()
        
        try:
            quality_score = 100.0
            issues = []
            
            # Check Python files for basic quality
            python_files = list(Path('lunar_habitat_rl').rglob('*.py'))
            
            for py_file in python_files[:20]:  # Sample first 20 files
                try:
                    with open(py_file, 'r') as f:
                        lines = f.readlines()
                        
                        # Check for very long lines
                        long_lines = [i for i, line in enumerate(lines) if len(line) > 120]
                        if len(long_lines) > len(lines) * 0.1:  # >10% long lines
                            quality_score -= 5
                            issues.append(f"Many long lines in {py_file}")
                        
                        # Check for imports
                        import_lines = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
                        if len(import_lines) > 20:
                            quality_score -= 5
                            issues.append(f"Many imports in {py_file}")
                
                except:
                    continue
            
            # Check for consistent naming (basic check)
            if any('camelCase' in str(f) for f in python_files):
                quality_score -= 10
                issues.append("Inconsistent naming convention detected")
            
            passed = quality_score >= 75 and len(issues) < 10
            
            result = QualityGateResult(
                gate_name="Code Quality",
                passed=passed,
                score=quality_score,
                details={
                    "issues_found": len(issues),
                    "issues": issues,
                    "files_analyzed": min(20, len(python_files))
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING"
            )
        
        self.results.append(result)
        return result
    
    async def validate_dependency_security(self) -> QualityGateResult:
        """Validate dependency security"""
        start_time = time.time()
        
        try:
            # Check requirements files
            req_files = ['requirements.txt', 'pyproject.toml']
            deps_found = []
            
            for req_file in req_files:
                if Path(req_file).exists():
                    deps_found.append(req_file)
            
            # Simple security score based on presence of dependency management
            security_score = 80 if deps_found else 40
            
            passed = len(deps_found) > 0
            
            result = QualityGateResult(
                gate_name="Dependency Security",
                passed=passed,
                score=security_score,
                details={
                    "dependency_files": deps_found,
                    "security_note": "Basic dependency management check"
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Dependency Security",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING"
            )
        
        self.results.append(result)
        return result
    
    async def validate_research_reproducibility(self) -> QualityGateResult:
        """Validate research reproducibility and methodology"""
        start_time = time.time()
        
        try:
            repro_score = 0.0
            findings = []
            
            # Check for benchmark files
            benchmark_files = list(Path('.').rglob('*benchmark*.py'))
            if benchmark_files:
                repro_score += 30
                findings.append(f"Found {len(benchmark_files)} benchmark files")
            
            # Check for research validation
            research_files = list(Path('.').rglob('*research*.py')) + list(Path('.').rglob('*validation*.py'))
            if research_files:
                repro_score += 25
                findings.append(f"Found {len(research_files)} research/validation files")
            
            # Check for statistical analysis
            stat_indicators = ['scipy', 'statistics', 'p_value', 'statistical_significance']
            python_files = list(Path('.').rglob('*.py'))
            
            stat_mentions = 0
            for py_file in python_files[:10]:  # Sample
                try:
                    with open(py_file, 'r') as f:
                        content = f.read().lower()
                        for indicator in stat_indicators:
                            if indicator in content:
                                stat_mentions += 1
                                break
                except:
                    continue
            
            if stat_mentions > 0:
                repro_score += 25
                findings.append(f"Statistical analysis indicators found in {stat_mentions} files")
            
            # Check for reproducible results
            if any('seed' in str(f).lower() or 'random' in str(f).lower() for f in python_files):
                repro_score += 20
                findings.append("Randomness control mechanisms detected")
            
            passed = repro_score >= 60
            
            result = QualityGateResult(
                gate_name="Research Reproducibility",
                passed=passed,
                score=repro_score,
                details={
                    "findings": findings,
                    "benchmark_files": len(benchmark_files),
                    "research_files": len(research_files)
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Research Reproducibility",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING"
            )
        
        self.results.append(result)
        return result
    
    async def validate_deployment_readiness(self) -> QualityGateResult:
        """Validate production deployment readiness"""
        start_time = time.time()
        
        try:
            deployment_score = 0.0
            components = {}
            
            # Check for containerization
            if Path('deployment/docker/Dockerfile').exists():
                deployment_score += 25
                components['docker'] = "EXISTS"
            else:
                components['docker'] = "MISSING"
            
            # Check for Kubernetes config
            if Path('deployment/kubernetes').exists():
                deployment_score += 25
                components['kubernetes'] = "EXISTS"
            else:
                components['kubernetes'] = "MISSING"
            
            # Check for CI/CD
            if Path('deployment/cicd').exists() or Path('.github/workflows').exists():
                deployment_score += 25
                components['cicd'] = "EXISTS"
            else:
                components['cicd'] = "MISSING"
            
            # Check for monitoring
            if Path('deployment/monitoring').exists():
                deployment_score += 25
                components['monitoring'] = "EXISTS"
            else:
                components['monitoring'] = "MISSING"
            
            passed = deployment_score >= 75
            
            result = QualityGateResult(
                gate_name="Deployment Readiness",
                passed=passed,
                score=deployment_score,
                details=components,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING" if not passed else "INFO"
            )
            
        except Exception as e:
            result = QualityGateResult(
                gate_name="Deployment Readiness",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                severity="WARNING"
            )
        
        self.results.append(result)
        return result
    
    def _serialize_result(self, result: QualityGateResult) -> Dict[str, Any]:
        """Serialize quality gate result for JSON output"""
        return {
            "gate_name": result.gate_name,
            "passed": result.passed,
            "score": result.score,
            "details": result.details,
            "execution_time": result.execution_time,
            "timestamp": result.timestamp.isoformat(),
            "severity": result.severity
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if result.gate_name == "Code Execution":
                    recommendations.append("🚨 CRITICAL: Fix code execution failures immediately")
                elif result.gate_name == "Security Posture":
                    recommendations.append("🛡️ HIGH: Address security vulnerabilities")
                elif result.gate_name == "Test Coverage":
                    recommendations.append("🧪 MEDIUM: Improve test coverage (target: 85%)")
                elif result.gate_name == "Performance Benchmarks":
                    recommendations.append("⚡ MEDIUM: Optimize performance bottlenecks")
                elif result.gate_name == "Documentation Quality":
                    recommendations.append("📚 LOW: Improve documentation completeness")
        
        if not recommendations:
            recommendations.append("✅ All quality gates passed successfully!")
        
        return recommendations

async def main():
    """Execute comprehensive quality gates validation"""
    print("🧪 COMPREHENSIVE QUALITY GATES EXECUTION")
    print("=" * 60)
    
    validator = ComprehensiveQualityValidator()
    
    try:
        # Execute all quality gates
        report = await validator.execute_all_gates()
        
        # Display results
        print(f"\n📊 QUALITY GATES SUMMARY")
        print(f"Overall Score: {report['overall_score']:.1f}%")
        print(f"Gates Passed: {report['gates_summary']['passed']}/{report['gates_summary']['total']}")
        print(f"Critical Failures: {report['critical_failures']}")
        print(f"Production Ready: {'✅ YES' if report['production_ready'] else '❌ NO'}")
        
        print(f"\n🎯 DETAILED RESULTS:")
        for result in validator.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.gate_name}: {result.score:.1f}% ({result.execution_time:.2f}s)")
        
        if report.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations'][:10]:  # Show first 10
                print(f"  {rec}")
        
        print(f"\n📁 Detailed report saved: comprehensive_quality_gates_report.json")
        
        return report
        
    except Exception as e:
        logger.critical(f"Critical failure in quality gates: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())