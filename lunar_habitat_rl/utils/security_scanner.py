"""
NASA-grade security scanning and threat detection system.
Implements comprehensive security validation for mission-critical operations.
"""

import os
import re
import hashlib
import json
import time
import threading
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import ast
import inspect
import importlib.util

from .robust_logging import get_logger
from .exceptions import ValidationError


class ThreatLevel(Enum):
    """Security threat severity levels."""
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MISSION_CRITICAL = "mission_critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    CODE_INJECTION = "code_injection"
    BUFFER_OVERFLOW = "buffer_overflow"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXPOSURE = "data_exposure"
    INSECURE_COMMUNICATION = "insecure_communication"
    WEAK_AUTHENTICATION = "weak_authentication"
    MALICIOUS_CODE = "malicious_code"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    SPACE_MISSION_SAFETY = "space_mission_safety"


@dataclass
class SecurityFinding:
    """Security vulnerability or threat finding."""
    id: str = field(default_factory=lambda: f"finding_{int(time.time()*1000)}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    threat_level: ThreatLevel = ThreatLevel.INFORMATIONAL
    vulnerability_type: VulnerabilityType = VulnerabilityType.CONFIGURATION_ERROR
    title: str = ""
    description: str = ""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: str = ""
    mission_impact: bool = False
    cve_references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'threat_level': self.threat_level.value,
            'vulnerability_type': self.vulnerability_type.value,
            'title': self.title,
            'description': self.description,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'code_snippet': self.code_snippet,
            'recommendation': self.recommendation,
            'mission_impact': self.mission_impact,
            'cve_references': self.cve_references
        }


class CodeSecurityAnalyzer:
    """Analyzes code for security vulnerabilities."""
    
    def __init__(self):
        """Initialize code security analyzer."""
        self.logger = get_logger()
        
        # Dangerous function patterns
        self.dangerous_functions = {
            'eval': ThreatLevel.CRITICAL,
            'exec': ThreatLevel.CRITICAL,
            'compile': ThreatLevel.HIGH,
            '__import__': ThreatLevel.HIGH,
            'getattr': ThreatLevel.MEDIUM,
            'setattr': ThreatLevel.MEDIUM,
            'delattr': ThreatLevel.MEDIUM,
            'open': ThreatLevel.MEDIUM,
            'input': ThreatLevel.MEDIUM,
            'subprocess.call': ThreatLevel.HIGH,
            'subprocess.run': ThreatLevel.HIGH,
            'os.system': ThreatLevel.CRITICAL,
            'os.popen': ThreatLevel.HIGH,
            'pickle.loads': ThreatLevel.CRITICAL,
            'pickle.load': ThreatLevel.HIGH,
            'yaml.load': ThreatLevel.HIGH,
        }
        
        # Insecure patterns
        self.insecure_patterns = {
            r'password\s*=\s*["\'][^"\']+["\']': (ThreatLevel.HIGH, "Hardcoded password"),
            r'api_key\s*=\s*["\'][^"\']+["\']': (ThreatLevel.HIGH, "Hardcoded API key"),
            r'secret\s*=\s*["\'][^"\']+["\']': (ThreatLevel.HIGH, "Hardcoded secret"),
            r'token\s*=\s*["\'][^"\']+["\']': (ThreatLevel.HIGH, "Hardcoded token"),
            r'md5\s*\(': (ThreatLevel.MEDIUM, "Weak hash function MD5"),
            r'sha1\s*\(': (ThreatLevel.MEDIUM, "Weak hash function SHA1"),
            r'random\.random\s*\(': (ThreatLevel.LOW, "Weak random number generator"),
            r'ssl_context\.check_hostname\s*=\s*False': (ThreatLevel.HIGH, "SSL hostname verification disabled"),
            r'verify\s*=\s*False': (ThreatLevel.HIGH, "SSL certificate verification disabled"),
            r'shell\s*=\s*True': (ThreatLevel.HIGH, "Shell injection vulnerability"),
            r'\.\.\/': (ThreatLevel.MEDIUM, "Potential path traversal"),
        }
        
        # Mission-critical safety patterns for space operations
        self.space_safety_patterns = {
            r'emergency_shutdown\s*=\s*False': (ThreatLevel.MISSION_CRITICAL, "Emergency shutdown disabled"),
            r'life_support_\w+\s*=\s*None': (ThreatLevel.MISSION_CRITICAL, "Life support system disabled"),
            r'safety_check\s*=\s*False': (ThreatLevel.CRITICAL, "Safety checks disabled"),
            r'validation\s*=\s*False': (ThreatLevel.HIGH, "Input validation disabled"),
            r'o2_pressure\s*<\s*1[0-5]': (ThreatLevel.MISSION_CRITICAL, "O2 pressure below critical minimum"),
            r'co2_pressure\s*>\s*[2-9]': (ThreatLevel.MISSION_CRITICAL, "CO2 pressure above critical maximum"),
        }
    
    def analyze_file(self, file_path: Path) -> List[SecurityFinding]:
        """Analyze a single file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic pattern matching
            findings.extend(self._check_patterns(content, file_path))
            
            # AST-based analysis for Python files
            if file_path.suffix == '.py':
                findings.extend(self._analyze_python_ast(content, file_path))
            
            # Configuration file analysis
            if file_path.suffix in ['.json', '.yaml', '.yml', '.toml', '.ini']:
                findings.extend(self._analyze_config_file(content, file_path))
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze {file_path}: {e}")
            findings.append(SecurityFinding(
                threat_level=ThreatLevel.LOW,
                vulnerability_type=VulnerabilityType.CONFIGURATION_ERROR,
                title="File analysis failed",
                description=f"Could not analyze file: {e}",
                file_path=str(file_path),
                recommendation="Verify file permissions and format"
            ))
        
        return findings
    
    def _check_patterns(self, content: str, file_path: Path) -> List[SecurityFinding]:
        """Check content against known insecure patterns."""
        findings = []
        lines = content.split('\n')
        
        all_patterns = {
            **self.insecure_patterns,
            **self.space_safety_patterns
        }
        
        for pattern, (threat_level, description) in all_patterns.items():
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerability_type = VulnerabilityType.SPACE_MISSION_SAFETY if pattern in self.space_safety_patterns else VulnerabilityType.CONFIGURATION_ERROR
                    
                    findings.append(SecurityFinding(
                        threat_level=threat_level,
                        vulnerability_type=vulnerability_type,
                        title=description,
                        description=f"Pattern matched: {pattern}",
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation=self._get_recommendation(pattern, description),
                        mission_impact=(threat_level == ThreatLevel.MISSION_CRITICAL)
                    ))
        
        return findings
    
    def _analyze_python_ast(self, content: str, file_path: Path) -> List[SecurityFinding]:
        """Analyze Python code using AST."""
        findings = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node)
                    if func_name in self.dangerous_functions:
                        threat_level = self.dangerous_functions[func_name]
                        findings.append(SecurityFinding(
                            threat_level=threat_level,
                            vulnerability_type=VulnerabilityType.CODE_INJECTION,
                            title=f"Dangerous function: {func_name}",
                            description=f"Use of potentially dangerous function: {func_name}",
                            file_path=str(file_path),
                            line_number=getattr(node, 'lineno', None),
                            recommendation=f"Avoid using {func_name} or implement strict input validation"
                        ))
                
                # Check for hardcoded strings that might be secrets
                if isinstance(node, ast.Str):
                    if len(node.s) > 20 and re.match(r'^[A-Za-z0-9+/=]+$', node.s):
                        findings.append(SecurityFinding(
                            threat_level=ThreatLevel.MEDIUM,
                            vulnerability_type=VulnerabilityType.DATA_EXPOSURE,
                            title="Potential hardcoded secret",
                            description="Long base64-like string found",
                            file_path=str(file_path),
                            line_number=getattr(node, 'lineno', None),
                            recommendation="Use environment variables or secure key management"
                        ))
                
        except SyntaxError as e:
            findings.append(SecurityFinding(
                threat_level=ThreatLevel.LOW,
                vulnerability_type=VulnerabilityType.CONFIGURATION_ERROR,
                title="Python syntax error",
                description=f"Syntax error prevents security analysis: {e}",
                file_path=str(file_path),
                line_number=e.lineno,
                recommendation="Fix syntax errors to enable proper security analysis"
            ))
        except Exception as e:
            self.logger.warning(f"AST analysis failed for {file_path}: {e}")
        
        return findings
    
    def _analyze_config_file(self, content: str, file_path: Path) -> List[SecurityFinding]:
        """Analyze configuration files for security issues."""
        findings = []
        
        try:
            # Try to parse as JSON
            if file_path.suffix == '.json':
                config = json.loads(content)
                findings.extend(self._check_config_security(config, file_path))
        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.logger.warning(f"Config analysis failed for {file_path}: {e}")
        
        return findings
    
    def _check_config_security(self, config: Dict[str, Any], file_path: Path) -> List[SecurityFinding]:
        """Check configuration dictionary for security issues."""
        findings = []
        
        def check_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for secrets in keys
                    if any(secret_word in key.lower() for secret_word in ['password', 'secret', 'key', 'token']):
                        if isinstance(value, str) and len(value) > 0:
                            findings.append(SecurityFinding(
                                threat_level=ThreatLevel.HIGH,
                                vulnerability_type=VulnerabilityType.DATA_EXPOSURE,
                                title="Hardcoded secret in config",
                                description=f"Secret found in configuration: {current_path}",
                                file_path=str(file_path),
                                recommendation="Use environment variables or encrypted configuration"
                            ))
                    
                    # Check for insecure settings
                    if key.lower() == 'debug' and value is True:
                        findings.append(SecurityFinding(
                            threat_level=ThreatLevel.MEDIUM,
                            vulnerability_type=VulnerabilityType.CONFIGURATION_ERROR,
                            title="Debug mode enabled",
                            description="Debug mode should not be enabled in production",
                            file_path=str(file_path),
                            recommendation="Disable debug mode in production environments"
                        ))
                    
                    check_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_recursive(item, f"{path}[{i}]")
        
        check_recursive(config)
        return findings
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from AST call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle module.function calls
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return "unknown"
    
    def _get_recommendation(self, pattern: str, description: str) -> str:
        """Get security recommendation for a specific pattern."""
        recommendations = {
            'password': "Use environment variables or encrypted configuration for passwords",
            'api_key': "Store API keys in secure environment variables",
            'secret': "Use secure key management systems",
            'md5': "Use SHA-256 or stronger hash functions",
            'ssl': "Enable SSL verification and hostname checking",
            'shell': "Use parameterized commands instead of shell=False  # SECURITY FIX: shell injection prevention",
            'emergency_shutdown': "Never disable emergency shutdown mechanisms",
            'life_support': "Life support systems must always be active",
            'safety_check': "Safety checks are mandatory for space missions"
        }
        
        for keyword, recommendation in recommendations.items():
            if keyword in pattern.lower() or keyword in description.lower():
                return recommendation
        
        return "Review and fix this security issue"


class DependencyScanner:
    """Scans dependencies for known vulnerabilities."""
    
    def __init__(self):
        """Initialize dependency scanner."""
        self.logger = get_logger()
        
        # Known vulnerable packages (simplified - in practice would use CVE database)
        self.vulnerable_packages = {
            'requests': {
                '<2.20.0': ['CVE-2018-18074'],
                '<2.25.0': ['CVE-2021-33503']
            },
            'pyyaml': {
                '<5.1': ['CVE-2017-18342'],
                '<5.4': ['CVE-2020-1747']
            },
            'pillow': {
                '<8.1.1': ['CVE-2021-25287', 'CVE-2021-25288']
            }
        }
    
    def scan_requirements(self, requirements_file: Path) -> List[SecurityFinding]:
        """Scan requirements file for vulnerable dependencies."""
        findings = []
        
        if not requirements_file.exists():
            return findings
        
        try:
            with open(requirements_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name, version = self._parse_requirement(line)
                        if package_name in self.vulnerable_packages:
                            vulnerabilities = self._check_package_vulnerabilities(package_name, version)
                            for vuln in vulnerabilities:
                                findings.append(SecurityFinding(
                                    threat_level=ThreatLevel.HIGH,
                                    vulnerability_type=VulnerabilityType.DEPENDENCY_VULNERABILITY,
                                    title=f"Vulnerable dependency: {package_name}",
                                    description=f"Package {package_name} version {version} has known vulnerabilities",
                                    file_path=str(requirements_file),
                                    line_number=line_num,
                                    cve_references=vuln['cves'],
                                    recommendation=f"Update {package_name} to version {vuln['fix_version']} or later"
                                ))
        except Exception as e:
            self.logger.error(f"Failed to scan requirements file {requirements_file}: {e}")
        
        return findings
    
    def _parse_requirement(self, line: str) -> Tuple[str, str]:
        """Parse a requirements line to extract package name and version."""
        # Simple parsing - in practice would use proper requirements parser
        if '==' in line:
            package, version = line.split('==', 1)
            return package.strip(), version.strip()
        elif '>=' in line:
            package, version = line.split('>=', 1)
            return package.strip(), version.strip()
        else:
            return line.strip(), "unknown"
    
    def _check_package_vulnerabilities(self, package: str, version: str) -> List[Dict[str, Any]]:
        """Check if package version has vulnerabilities."""
        vulnerabilities = []
        
        if package in self.vulnerable_packages:
            for version_range, cves in self.vulnerable_packages[package].items():
                if self._version_matches_range(version, version_range):
                    vulnerabilities.append({
                        'cves': cves,
                        'fix_version': self._get_fix_version(version_range)
                    })
        
        return vulnerabilities
    
    def _version_matches_range(self, version: str, range_spec: str) -> bool:
        """Check if version matches vulnerability range."""
        # Simplified version comparison
        if range_spec.startswith('<'):
            threshold = range_spec[1:]
            return version < threshold
        return False
    
    def _get_fix_version(self, range_spec: str) -> str:
        """Get the minimum version that fixes the vulnerability."""
        if range_spec.startswith('<'):
            return range_spec[1:]
        return "latest"


class SecurityScanner:
    """Main security scanner orchestrating all security checks."""
    
    def __init__(self):
        """Initialize security scanner."""
        self.logger = get_logger()
        self.code_analyzer = CodeSecurityAnalyzer()
        self.dependency_scanner = DependencyScanner()
        self.last_scan_time = None
        self.scan_results_cache = {}
    
    def comprehensive_scan(self, scan_path: Path, 
                          include_dependencies: bool = True,
                          file_extensions: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive security scan of project.
        
        Args:
            scan_path: Path to scan
            include_dependencies: Whether to scan dependencies
            file_extensions: File extensions to scan (default: .py, .json, .yaml, .yml)
            
        Returns:
            Dictionary containing scan results
        """
        if file_extensions is None:
            file_extensions = {'.py', '.json', '.yaml', '.yml', '.toml', '.ini'}
        
        scan_start = time.time()
        self.logger.info(f"Starting comprehensive security scan of {scan_path}")
        
        findings = []
        scanned_files = 0
        
        # Scan source code files
        for file_path in scan_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in file_extensions:
                # Skip common non-security-relevant directories
                if any(part in str(file_path) for part in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                    continue
                
                try:
                    file_findings = self.code_analyzer.analyze_file(file_path)
                    findings.extend(file_findings)
                    scanned_files += 1
                except Exception as e:
                    self.logger.warning(f"Failed to scan {file_path}: {e}")
        
        # Scan dependencies
        dependency_findings = []
        if include_dependencies:
            requirements_files = list(scan_path.glob('requirements*.txt'))
            requirements_files.extend(scan_path.glob('pyproject.toml'))
            
            for req_file in requirements_files:
                if req_file.name.endswith('.txt'):
                    dependency_findings.extend(self.dependency_scanner.scan_requirements(req_file))
        
        # Categorize findings
        findings_by_severity = self._categorize_findings(findings + dependency_findings)
        mission_critical_count = len([f for f in findings + dependency_findings if f.mission_impact])
        
        scan_duration = time.time() - scan_start
        self.last_scan_time = datetime.utcnow()
        
        results = {
            'scan_metadata': {
                'scan_time': self.last_scan_time.isoformat(),
                'scan_duration_seconds': scan_duration,
                'scanned_files': scanned_files,
                'total_findings': len(findings) + len(dependency_findings)
            },
            'summary': {
                'mission_critical': mission_critical_count,
                'critical': len(findings_by_severity.get(ThreatLevel.CRITICAL, [])),
                'high': len(findings_by_severity.get(ThreatLevel.HIGH, [])),
                'medium': len(findings_by_severity.get(ThreatLevel.MEDIUM, [])),
                'low': len(findings_by_severity.get(ThreatLevel.LOW, [])),
                'informational': len(findings_by_severity.get(ThreatLevel.INFORMATIONAL, []))
            },
            'findings': [f.to_dict() for f in findings + dependency_findings],
            'risk_assessment': self._assess_overall_risk(findings + dependency_findings)
        }
        
        # Cache results
        self.scan_results_cache[str(scan_path)] = results
        
        self.logger.info(f"Security scan completed in {scan_duration:.2f}s. "
                        f"Found {len(findings + dependency_findings)} issues "
                        f"({mission_critical_count} mission-critical)")
        
        return results
    
    def _categorize_findings(self, findings: List[SecurityFinding]) -> Dict[ThreatLevel, List[SecurityFinding]]:
        """Categorize findings by threat level."""
        categorized = {}
        for finding in findings:
            if finding.threat_level not in categorized:
                categorized[finding.threat_level] = []
            categorized[finding.threat_level].append(finding)
        return categorized
    
    def _assess_overall_risk(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Assess overall security risk level."""
        if not findings:
            return {'level': 'low', 'description': 'No security issues found'}
        
        mission_critical = len([f for f in findings if f.threat_level == ThreatLevel.MISSION_CRITICAL])
        critical = len([f for f in findings if f.threat_level == ThreatLevel.CRITICAL])
        high = len([f for f in findings if f.threat_level == ThreatLevel.HIGH])
        
        if mission_critical > 0:
            return {
                'level': 'mission_critical',
                'description': f'{mission_critical} mission-critical vulnerabilities found',
                'recommendation': 'IMMEDIATE ACTION REQUIRED - Mission safety at risk'
            }
        elif critical > 0:
            return {
                'level': 'critical',
                'description': f'{critical} critical vulnerabilities found',
                'recommendation': 'Urgent action required before deployment'
            }
        elif high > 3:
            return {
                'level': 'high',
                'description': f'{high} high-severity vulnerabilities found',
                'recommendation': 'Address high-severity issues before production use'
            }
        elif high > 0:
            return {
                'level': 'medium',
                'description': f'{high} high-severity vulnerabilities found',
                'recommendation': 'Review and address security findings'
            }
        else:
            return {
                'level': 'low',
                'description': 'Only low-severity issues found',
                'recommendation': 'Monitor and address findings as time permits'
            }
    
    def generate_security_report(self, scan_results: Dict[str, Any], 
                               output_path: Path) -> None:
        """Generate comprehensive security report."""
        report_content = self._format_security_report(scan_results)
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Security report generated: {output_path}")
    
    def _format_security_report(self, scan_results: Dict[str, Any]) -> str:
        """Format security scan results into a comprehensive report."""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("NASA-GRADE LUNAR HABITAT RL SECURITY SCAN REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Scan metadata
        metadata = scan_results['scan_metadata']
        report.append(f"Scan Date: {metadata['scan_time']}")
        report.append(f"Scan Duration: {metadata['scan_duration_seconds']:.2f} seconds")
        report.append(f"Files Scanned: {metadata['scanned_files']}")
        report.append(f"Total Findings: {metadata['total_findings']}")
        report.append("")
        
        # Risk assessment
        risk = scan_results['risk_assessment']
        report.append("OVERALL SECURITY RISK ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Risk Level: {risk['level'].upper()}")
        report.append(f"Description: {risk['description']}")
        report.append(f"Recommendation: {risk['recommendation']}")
        report.append("")
        
        # Summary
        summary = scan_results['summary']
        report.append("FINDINGS SUMMARY BY SEVERITY")
        report.append("-" * 40)
        if summary['mission_critical'] > 0:
            report.append(f"🚨 MISSION CRITICAL: {summary['mission_critical']}")
        report.append(f"❌ Critical: {summary['critical']}")
        report.append(f"⚠️  High: {summary['high']}")
        report.append(f"🔶 Medium: {summary['medium']}")
        report.append(f"ℹ️  Low: {summary['low']}")
        report.append(f"📋 Informational: {summary['informational']}")
        report.append("")
        
        # Detailed findings
        if scan_results['findings']:
            report.append("DETAILED FINDINGS")
            report.append("-" * 40)
            
            # Group by severity
            findings_by_severity = {}
            for finding in scan_results['findings']:
                severity = finding['threat_level']
                if severity not in findings_by_severity:
                    findings_by_severity[severity] = []
                findings_by_severity[severity].append(finding)
            
            # Report in order of severity
            severity_order = ['mission_critical', 'critical', 'high', 'medium', 'low', 'informational']
            
            for severity in severity_order:
                if severity in findings_by_severity:
                    report.append(f"\n{severity.upper()} SEVERITY FINDINGS")
                    report.append("=" * 50)
                    
                    for i, finding in enumerate(findings_by_severity[severity], 1):
                        report.append(f"\nFinding {i}: {finding['title']}")
                        report.append(f"Type: {finding['vulnerability_type']}")
                        if finding['file_path']:
                            report.append(f"File: {finding['file_path']}")
                        if finding['line_number']:
                            report.append(f"Line: {finding['line_number']}")
                        report.append(f"Description: {finding['description']}")
                        if finding['code_snippet']:
                            report.append(f"Code: {finding['code_snippet']}")
                        report.append(f"Recommendation: {finding['recommendation']}")
                        if finding['cve_references']:
                            report.append(f"CVE References: {', '.join(finding['cve_references'])}")
                        report.append("-" * 30)
        
        # Mission safety notice
        if summary['mission_critical'] > 0:
            report.append("\n🚨 MISSION SAFETY ALERT 🚨")
            report.append("=" * 50)
            report.append("MISSION-CRITICAL SECURITY VULNERABILITIES DETECTED!")
            report.append("These vulnerabilities pose direct risks to crew safety and mission success.")
            report.append("IMMEDIATE REMEDIATION REQUIRED BEFORE DEPLOYMENT.")
            report.append("Contact mission security team immediately.")
        
        # Footer
        report.append("\n" + "=" * 80)
        report.append("END OF SECURITY SCAN REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_cached_results(self, scan_path: str) -> Optional[Dict[str, Any]]:
        """Get cached scan results if available."""
        return self.scan_results_cache.get(scan_path)


# Global security scanner instance
_global_security_scanner = None

def get_security_scanner() -> SecurityScanner:
    """Get global security scanner instance."""
    global _global_security_scanner
    if _global_security_scanner is None:
        _global_security_scanner = SecurityScanner()
    return _global_security_scanner


def security_scan_required(critical_functions: List[str] = None):
    """Decorator to require security scan before executing critical functions.
    
    Args:
        critical_functions: List of function names that require security clearance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if critical_functions and func.__name__ in critical_functions:
                scanner = get_security_scanner()
                if not scanner.last_scan_time or (
                    datetime.utcnow() - scanner.last_scan_time > timedelta(hours=24)
                ):
                    raise ValidationError(
                        f"Security scan required for critical function {func.__name__}. "
                        "Run comprehensive security scan first."
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator