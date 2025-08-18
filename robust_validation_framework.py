#!/usr/bin/env python3
"""
Robust Validation Framework - Generation 2 Enhancement
Comprehensive error handling, validation, and security for Lunar Habitat RL Suite
"""

import logging
import traceback
import hashlib
import time
import json
import os
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import warnings

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Structured validation result with comprehensive details"""
    passed: bool
    metric: str
    value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=datetime.now)
    details: str = ""
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    recommendation: str = ""

@dataclass
class SecurityAuditResult:
    """Security audit findings with actionable recommendations"""
    vulnerability_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_components: List[str]
    remediation: str
    cve_references: List[str] = field(default_factory=list)

class RobustErrorHandler:
    """Advanced error handling with recovery mechanisms"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_history: List[Dict[str, Any]] = []
    
    def with_retry(self, operation_name: str = "Unknown Operation"):
        """Decorator for automatic retry with exponential backoff"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    
                    except Exception as e:
                        last_exception = e
                        error_record = {
                            "operation": operation_name,
                            "attempt": attempt + 1,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "traceback": traceback.format_exc()
                        }
                        self.error_history.append(error_record)
                        
                        if attempt < self.max_retries:
                            wait_time = (self.backoff_factor ** attempt)
                            logger.warning(
                                f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                                f"Retrying in {wait_time:.1f}s. Error: {str(e)}"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(
                                f"{operation_name} failed after {self.max_retries + 1} attempts. "
                                f"Final error: {str(e)}"
                            )
                
                # If all retries failed, raise the last exception
                raise last_exception
            
            return wrapper
        return decorator
    
    @asynccontextmanager
    async def error_context(self, operation_name: str, critical: bool = False):
        """Context manager for comprehensive error handling"""
        start_time = time.time()
        try:
            logger.info(f"Starting {operation_name}")
            yield
            execution_time = time.time() - start_time
            logger.info(f"Completed {operation_name} successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_record = {
                "operation": operation_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "traceback": traceback.format_exc(),
                "critical": critical
            }
            self.error_history.append(error_record)
            
            if critical:
                logger.critical(f"CRITICAL ERROR in {operation_name}: {str(e)}")
            else:
                logger.error(f"Error in {operation_name}: {str(e)}")
            
            raise

class ComprehensiveValidator:
    """Comprehensive validation framework with extensive checks"""
    
    def __init__(self):
        self.error_handler = RobustErrorHandler()
        self.validation_history: List[ValidationResult] = []
        self.security_auditor = SecurityAuditor()
    
    @RobustErrorHandler().with_retry("Environment Validation")
    async def validate_environment_integrity(self) -> List[ValidationResult]:
        """Validate environment setup and dependencies"""
        results = []
        
        # Python version check
        import sys
        python_version = sys.version_info
        min_version = (3, 9)
        results.append(ValidationResult(
            passed=python_version >= min_version,
            metric="python_version",
            value=f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            threshold=f">={min_version[0]}.{min_version[1]}",
            severity="CRITICAL" if python_version < min_version else "INFO",
            details=f"Python version: {python_version}",
            recommendation="Update Python to version 3.9 or higher" if python_version < min_version else "Python version acceptable"
        ))
        
        # Critical imports validation
        critical_imports = [
            'numpy', 'gymnasium', 'pathlib', 'json', 'asyncio'
        ]
        
        for module_name in critical_imports:
            try:
                __import__(module_name)
                results.append(ValidationResult(
                    passed=True,
                    metric=f"import_{module_name}",
                    value="Available",
                    threshold="Required",
                    details=f"Successfully imported {module_name}",
                    severity="INFO"
                ))
            except ImportError as e:
                results.append(ValidationResult(
                    passed=False,
                    metric=f"import_{module_name}",
                    value="Missing",
                    threshold="Required",
                    details=f"Failed to import {module_name}: {str(e)}",
                    severity="CRITICAL",
                    recommendation=f"Install {module_name} package"
                ))
        
        # File system validation
        required_files = ['README.md', 'pyproject.toml', 'lunar_habitat_rl/__init__.py']
        for file_path in required_files:
            path = Path(file_path)
            results.append(ValidationResult(
                passed=path.exists(),
                metric=f"file_exists_{file_path}",
                value="Exists" if path.exists() else "Missing",
                threshold="Required",
                details=f"File: {file_path}",
                severity="ERROR" if not path.exists() else "INFO",
                recommendation=f"Create missing file: {file_path}" if not path.exists() else "File exists"
            ))
        
        self.validation_history.extend(results)
        return results
    
    @RobustErrorHandler().with_retry("Performance Validation")
    async def validate_performance_benchmarks(self) -> List[ValidationResult]:
        """Validate performance meets acceptable thresholds"""
        results = []
        
        # Memory usage check
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_threshold = 90.0
            
            results.append(ValidationResult(
                passed=memory_usage_percent < memory_threshold,
                metric="memory_usage",
                value=f"{memory_usage_percent:.1f}%",
                threshold=f"<{memory_threshold}%",
                details=f"System memory usage: {memory_usage_percent:.1f}%",
                severity="WARNING" if memory_usage_percent > 80 else "INFO",
                recommendation="Free up system memory" if memory_usage_percent > memory_threshold else "Memory usage acceptable"
            ))
        except ImportError:
            results.append(ValidationResult(
                passed=False,
                metric="memory_monitoring",
                value="Unavailable",
                threshold="Required",
                details="psutil not available for memory monitoring",
                severity="WARNING",
                recommendation="Install psutil for system monitoring"
            ))
        
        # Disk space validation
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100
        min_free_percent = 10.0
        
        results.append(ValidationResult(
            passed=free_percent > min_free_percent,
            metric="disk_space",
            value=f"{free_percent:.1f}% free",
            threshold=f">{min_free_percent}% free",
            details=f"Free disk space: {free / (1024**3):.1f} GB ({free_percent:.1f}%)",
            severity="CRITICAL" if free_percent < 5 else "WARNING" if free_percent < min_free_percent else "INFO",
            recommendation="Free up disk space" if free_percent <= min_free_percent else "Disk space sufficient"
        ))
        
        # Environment creation performance test
        start_time = time.time()
        try:
            import lunar_habitat_rl
            env = lunar_habitat_rl.make_lunar_env()
            obs, info = env.reset()
            env_creation_time = time.time() - start_time
            max_creation_time = 5.0  # seconds
            
            results.append(ValidationResult(
                passed=env_creation_time < max_creation_time,
                metric="env_creation_time",
                value=f"{env_creation_time:.3f}s",
                threshold=f"<{max_creation_time}s",
                details=f"Environment creation took {env_creation_time:.3f} seconds",
                severity="WARNING" if env_creation_time > max_creation_time else "INFO",
                recommendation="Optimize environment initialization" if env_creation_time > max_creation_time else "Environment creation time acceptable"
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                passed=False,
                metric="env_creation",
                value="Failed",
                threshold="Success",
                details=f"Environment creation failed: {str(e)}",
                severity="CRITICAL",
                recommendation="Fix environment creation issues"
            ))
        
        self.validation_history.extend(results)
        return results
    
    @RobustErrorHandler().with_retry("Security Validation")
    async def validate_security_posture(self) -> List[ValidationResult]:
        """Comprehensive security validation"""
        results = []
        security_audit = await self.security_auditor.conduct_security_audit()
        
        # Convert security audit to validation results
        critical_vulns = [v for v in security_audit if v.severity == "CRITICAL"]
        high_vulns = [v for v in security_audit if v.severity == "HIGH"]
        
        results.append(ValidationResult(
            passed=len(critical_vulns) == 0,
            metric="critical_vulnerabilities",
            value=len(critical_vulns),
            threshold=0,
            details=f"Found {len(critical_vulns)} critical vulnerabilities",
            severity="CRITICAL" if critical_vulns else "INFO",
            recommendation="Fix critical vulnerabilities immediately" if critical_vulns else "No critical vulnerabilities found"
        ))
        
        results.append(ValidationResult(
            passed=len(high_vulns) == 0,
            metric="high_vulnerabilities",
            value=len(high_vulns),
            threshold=0,
            details=f"Found {len(high_vulns)} high-severity vulnerabilities",
            severity="ERROR" if high_vulns else "INFO",
            recommendation="Address high-severity vulnerabilities" if high_vulns else "No high-severity vulnerabilities found"
        ))
        
        # File permissions check
        sensitive_files = ['pyproject.toml', 'requirements.txt']
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                stat = path.stat()
                # Check if file is readable by others (octal 004)
                readable_by_others = bool(stat.st_mode & 0o004)
                
                results.append(ValidationResult(
                    passed=not readable_by_others,
                    metric=f"file_permissions_{file_path}",
                    value=f"Mode: {oct(stat.st_mode)[-3:]}",
                    threshold="Not world-readable",
                    details=f"File permissions for {file_path}",
                    severity="WARNING" if readable_by_others else "INFO",
                    recommendation=f"Secure file permissions for {file_path}" if readable_by_others else "File permissions acceptable"
                ))
        
        self.validation_history.extend(results)
        return results
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        async with self.error_handler.error_context("Report Generation"):
            # Run all validations
            env_results = await self.validate_environment_integrity()
            perf_results = await self.validate_performance_benchmarks()
            sec_results = await self.validate_security_posture()
            
            all_results = env_results + perf_results + sec_results
            
            # Categorize results
            passed = [r for r in all_results if r.passed]
            failed = [r for r in all_results if not r.passed]
            critical_issues = [r for r in all_results if r.severity == "CRITICAL"]
            
            # Calculate scores
            total_checks = len(all_results)
            passed_checks = len(passed)
            pass_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            
            report = {
                "validation_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_checks": total_checks,
                    "passed": passed_checks,
                    "failed": len(failed),
                    "pass_rate": f"{pass_rate:.1f}%",
                    "critical_issues": len(critical_issues)
                },
                "categories": {
                    "environment": {
                        "total": len(env_results),
                        "passed": len([r for r in env_results if r.passed]),
                        "results": [self._serialize_result(r) for r in env_results]
                    },
                    "performance": {
                        "total": len(perf_results),
                        "passed": len([r for r in perf_results if r.passed]),
                        "results": [self._serialize_result(r) for r in perf_results]
                    },
                    "security": {
                        "total": len(sec_results),
                        "passed": len([r for r in sec_results if r.passed]),
                        "results": [self._serialize_result(r) for r in sec_results]
                    }
                },
                "critical_issues": [self._serialize_result(r) for r in critical_issues],
                "recommendations": self._generate_recommendations(all_results),
                "overall_status": "PASSED" if pass_rate >= 90 and not critical_issues else "FAILED"
            }
            
            # Save report
            report_file = Path("robust_validation_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Generated comprehensive validation report: {report_file}")
            return report
    
    def _serialize_result(self, result: ValidationResult) -> Dict[str, Any]:
        """Serialize validation result for JSON output"""
        return {
            "metric": result.metric,
            "passed": result.passed,
            "value": str(result.value),
            "threshold": str(result.threshold),
            "severity": result.severity,
            "details": result.details,
            "recommendation": result.recommendation,
            "timestamp": result.timestamp.isoformat()
        }
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations from validation results"""
        recommendations = []
        failed_results = [r for r in results if not r.passed and r.recommendation]
        
        # Priority order: CRITICAL, ERROR, WARNING
        severity_order = ["CRITICAL", "ERROR", "WARNING"]
        for severity in severity_order:
            severity_results = [r for r in failed_results if r.severity == severity]
            if severity_results:
                recommendations.append(f"=== {severity} Priority ===")
                for result in severity_results:
                    recommendations.append(f"• {result.recommendation}")
        
        if not recommendations:
            recommendations.append("All validations passed successfully!")
        
        return recommendations

class SecurityAuditor:
    """Advanced security auditing capabilities"""
    
    def __init__(self):
        self.audit_timestamp = datetime.now()
    
    async def conduct_security_audit(self) -> List[SecurityAuditResult]:
        """Conduct comprehensive security audit"""
        audit_results = []
        
        # Check for hardcoded secrets
        audit_results.extend(await self._scan_for_secrets())
        
        # Check file permissions
        audit_results.extend(await self._audit_file_permissions())
        
        # Check for unsafe imports
        audit_results.extend(await self._audit_unsafe_imports())
        
        # Check configuration security
        audit_results.extend(await self._audit_configurations())
        
        return audit_results
    
    async def _scan_for_secrets(self) -> List[SecurityAuditResult]:
        """Scan for potential hardcoded secrets"""
        results = []
        
        # Common secret patterns
        secret_patterns = [
            (r'password\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded password detected'),
            (r'api[_-]?key\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded API key detected'),
            (r'secret[_-]?key\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded secret key detected'),
            (r'token\s*=\s*[\'"][^\'"]+[\'"]', 'Hardcoded token detected')
        ]
        
        # Search in Python files
        import re
        python_files = list(Path('.').rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        results.append(SecurityAuditResult(
                            vulnerability_type="Hardcoded Secrets",
                            severity="HIGH",
                            description=f"{description} in {file_path}",
                            affected_components=[str(file_path)],
                            remediation="Remove hardcoded secrets and use environment variables or secure key management"
                        ))
            
            except (UnicodeDecodeError, IOError):
                # Skip files that can't be read as text
                continue
        
        return results
    
    async def _audit_file_permissions(self) -> List[SecurityAuditResult]:
        """Audit file permissions for security issues"""
        results = []
        
        # Check for world-writable files
        sensitive_files = list(Path('.').rglob('*.py')) + list(Path('.').rglob('*.json')) + list(Path('.').rglob('*.yml'))
        
        for file_path in sensitive_files:
            try:
                stat = file_path.stat()
                # Check if file is writable by others (octal 002)
                world_writable = bool(stat.st_mode & 0o002)
                
                if world_writable:
                    results.append(SecurityAuditResult(
                        vulnerability_type="Insecure File Permissions",
                        severity="MEDIUM",
                        description=f"File {file_path} is world-writable",
                        affected_components=[str(file_path)],
                        remediation=f"Remove world-write permissions: chmod o-w {file_path}"
                    ))
            
            except OSError:
                continue
        
        return results
    
    async def _audit_unsafe_imports(self) -> List[SecurityAuditResult]:
        """Check for potentially unsafe imports"""
        results = []
        
        unsafe_imports = [
            ('eval', 'Use of eval() can execute arbitrary code'),
            ('exec', 'Use of exec() can execute arbitrary code'),
            ('subprocess.call', 'Direct subprocess calls can be dangerous'),
            ('os.system', 'os.system() can execute arbitrary commands'),
            ('pickle.load', 'pickle.load() can execute arbitrary code')
        ]
        
        python_files = list(Path('.').rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for unsafe_func, description in unsafe_imports:
                    if unsafe_func in content:
                        results.append(SecurityAuditResult(
                            vulnerability_type="Unsafe Code Patterns",
                            severity="MEDIUM",
                            description=f"{description} found in {file_path}",
                            affected_components=[str(file_path)],
                            remediation=f"Review and secure usage of {unsafe_func}"
                        ))
            
            except (UnicodeDecodeError, IOError):
                continue
        
        return results
    
    async def _audit_configurations(self) -> List[SecurityAuditResult]:
        """Audit configuration files for security issues"""
        results = []
        
        # Check for debug mode in configurations
        config_files = list(Path('.').rglob('*.json')) + list(Path('.').rglob('*.yml')) + list(Path('.').rglob('*.yaml'))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                if 'debug=true' in content or '"debug": true' in content:
                    results.append(SecurityAuditResult(
                        vulnerability_type="Insecure Configuration",
                        severity="LOW",
                        description=f"Debug mode enabled in {config_file}",
                        affected_components=[str(config_file)],
                        remediation="Disable debug mode in production configurations"
                    ))
            
            except (UnicodeDecodeError, IOError):
                continue
        
        return results

# Health monitoring system
class HealthMonitor:
    """Continuous health monitoring for the system"""
    
    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self.health_history: List[Dict[str, Any]] = []
        self.validator = ComprehensiveValidator()
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        logger.info("Starting health monitoring system")
        
        while True:
            try:
                health_check = await self._perform_health_check()
                self.health_history.append(health_check)
                
                # Keep only last 24 hours of history
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.health_history = [
                    h for h in self.health_history 
                    if datetime.fromisoformat(h['timestamp']) > cutoff_time
                ]
                
                # Log health status
                if health_check['status'] == 'HEALTHY':
                    logger.info(f"Health check passed: {health_check['score']:.1f}% healthy")
                else:
                    logger.warning(f"Health check issues detected: {health_check['score']:.1f}% healthy")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(self.check_interval)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            # Quick validation checks
            env_results = await self.validator.validate_environment_integrity()
            
            # Calculate health score
            total_checks = len(env_results)
            passed_checks = len([r for r in env_results if r.passed])
            health_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            
            # Determine status
            if health_score >= 90:
                status = "HEALTHY"
            elif health_score >= 70:
                status = "DEGRADED"
            else:
                status = "UNHEALTHY"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "score": health_score,
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks
            }
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "ERROR",
                "score": 0,
                "error": str(e)
            }

async def main():
    """Main execution for Generation 2 robust validation"""
    print("🛡️ GENERATION 2: ROBUST VALIDATION FRAMEWORK")
    print("=" * 60)
    
    validator = ComprehensiveValidator()
    
    try:
        # Generate comprehensive validation report
        report = await validator.generate_comprehensive_report()
        
        print(f"📊 VALIDATION SUMMARY")
        print(f"Total Checks: {report['summary']['total_checks']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Pass Rate: {report['summary']['pass_rate']}")
        print(f"Critical Issues: {report['summary']['critical_issues']}")
        print(f"Overall Status: {report['overall_status']}")
        print()
        
        if report['recommendations']:
            print("🔧 RECOMMENDATIONS:")
            for rec in report['recommendations'][:10]:  # Show first 10
                print(f"  {rec}")
        
        print(f"\n📁 Detailed report saved: robust_validation_report.json")
        
        # Demonstrate error handling capabilities
        error_handler = RobustErrorHandler()
        print(f"📈 Error History: {len(error_handler.error_history)} recorded errors")
        
        return report
        
    except Exception as e:
        logger.critical(f"Critical failure in robust validation: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())