#!/usr/bin/env python3
"""
Security scanning and vulnerability assessment for Lunar Habitat RL Suite.

This script performs comprehensive security checks including:
- Static code analysis for security vulnerabilities
- Dependency vulnerability scanning
- Configuration security validation
- Access control and permission verification
"""

import os
import sys
import json
import subprocess
import importlib
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SecurityFinding:
    """Represents a security finding or vulnerability."""
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"
    category: str  # "injection", "auth", "crypto", "config", etc.
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cve_id: Optional[str] = None


@dataclass
class SecurityReport:
    """Security assessment report."""
    scan_timestamp: str
    project_path: str
    total_files_scanned: int
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def add_finding(self, finding: SecurityFinding):
        """Add a security finding to the report."""
        self.findings.append(finding)
        
        # Update summary
        severity = finding.severity
        self.summary[severity] = self.summary.get(severity, 0) + 1
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of findings by severity."""
        return {
            "CRITICAL": self.summary.get("CRITICAL", 0),
            "HIGH": self.summary.get("HIGH", 0),
            "MEDIUM": self.summary.get("MEDIUM", 0),
            "LOW": self.summary.get("LOW", 0),
            "INFO": self.summary.get("INFO", 0)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "scan_timestamp": self.scan_timestamp,
            "project_path": self.project_path,
            "total_files_scanned": self.total_files_scanned,
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "code_snippet": f.code_snippet,
                    "recommendation": f.recommendation,
                    "cve_id": f.cve_id
                }
                for f in self.findings
            ],
            "summary": self.get_severity_counts()
        }


class StaticCodeAnalyzer:
    """Static code analysis for security vulnerabilities."""
    
    def __init__(self):
        self.dangerous_functions = {
            # Command injection risks
            'os.system': ('HIGH', 'Command injection risk'),
            'subprocess.call': ('MEDIUM', 'Potential command injection'),
            'subprocess.run': ('MEDIUM', 'Potential command injection'),
            'subprocess.Popen': ('MEDIUM', 'Potential command injection'),
            'eval': ('CRITICAL', 'Code injection risk'),
            'exec': ('CRITICAL', 'Code injection risk'),
            'compile': ('HIGH', 'Dynamic code compilation risk'),
            
            # File operations
            'open': ('LOW', 'File access - verify path validation'),
            'pickle.load': ('HIGH', 'Deserialization risk'),
            'pickle.loads': ('HIGH', 'Deserialization risk'),
            'yaml.load': ('HIGH', 'Unsafe YAML loading'),
            
            # Network operations
            'urllib.request.urlopen': ('MEDIUM', 'Network request - validate URLs'),
            'requests.get': ('LOW', 'Network request - validate URLs'),
            'socket.socket': ('MEDIUM', 'Raw socket usage'),
        }
        
        self.insecure_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'HIGH', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'HIGH', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'HIGH', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'MEDIUM', 'Hardcoded token'),
            (r'md5\s*\(', 'MEDIUM', 'Weak hashing algorithm (MD5)'),
            (r'sha1\s*\(', 'MEDIUM', 'Weak hashing algorithm (SHA1)'),
            (r'random\.random\(\)', 'LOW', 'Cryptographically weak RNG'),
            (r'ssl_verify\s*=\s*False', 'HIGH', 'SSL verification disabled'),
            (r'verify\s*=\s*False', 'HIGH', 'Certificate verification disabled'),
        ]
    
    def analyze_file(self, file_path: Path) -> List[SecurityFinding]:
        """Analyze a single Python file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for function calls
            try:
                tree = ast.parse(content)
                findings.extend(self._analyze_ast(tree, file_path, content))
            except SyntaxError:
                # Skip files with syntax errors
                pass
            
            # Pattern-based analysis
            findings.extend(self._analyze_patterns(content, file_path))
            
        except Exception as e:
            # Add finding for files that couldn't be analyzed
            findings.append(SecurityFinding(
                severity="INFO",
                category="analysis",
                title="File analysis failed",
                description=f"Could not analyze file: {e}",
                file_path=str(file_path)
            ))
        
        return findings
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, content: str) -> List[SecurityFinding]:
        """Analyze AST for dangerous function calls."""
        findings = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node)
                
                if func_name in self.dangerous_functions:
                    severity, description = self.dangerous_functions[func_name]
                    
                    code_snippet = None
                    if hasattr(node, 'lineno') and node.lineno <= len(lines):
                        code_snippet = lines[node.lineno - 1].strip()
                    
                    findings.append(SecurityFinding(
                        severity=severity,
                        category="dangerous_function",
                        title=f"Dangerous function usage: {func_name}",
                        description=description,
                        file_path=str(file_path),
                        line_number=getattr(node, 'lineno', None),
                        code_snippet=code_snippet,
                        recommendation=f"Review usage of {func_name} and ensure input validation"
                    ))
        
        return findings
    
    def _get_function_name(self, node: ast.Call) -> str:
        """Extract function name from AST call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            elif isinstance(node.func.value, ast.Attribute):
                # Handle nested attributes like os.path.join
                base = self._get_attribute_chain(node.func.value)
                return f"{base}.{node.func.attr}"
        return ""
    
    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get full attribute chain for nested attributes."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_chain(node.value)}.{node.attr}"
        return node.attr
    
    def _analyze_patterns(self, content: str, file_path: Path) -> List[SecurityFinding]:
        """Analyze content using regex patterns."""
        findings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, severity, description in self.insecure_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(SecurityFinding(
                        severity=severity,
                        category="pattern_match",
                        title=f"Security pattern detected: {description}",
                        description=description,
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation="Review and remediate security issue"
                    ))
        
        return findings


class DependencyScanner:
    """Scans dependencies for known vulnerabilities."""
    
    def __init__(self):
        self.known_vulnerabilities = {
            # Example known vulnerable packages (would be loaded from CVE database)
            'pickle': {
                'cve': 'CVE-2019-16781',
                'severity': 'HIGH',
                'description': 'Arbitrary code execution via pickle deserialization'
            },
            'pyyaml': {
                'versions': ['<5.1'],
                'cve': 'CVE-2017-18342',
                'severity': 'HIGH',
                'description': 'Arbitrary code execution via yaml.load'
            }
        }
    
    def scan_requirements(self, requirements_path: Path) -> List[SecurityFinding]:
        """Scan requirements.txt for vulnerable dependencies."""
        findings = []
        
        if not requirements_path.exists():
            return findings
        
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.read()
            
            # Simple parsing - in production would use proper dependency parser
            for line in requirements.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    
                    if package_name.lower() in self.known_vulnerabilities:
                        vuln = self.known_vulnerabilities[package_name.lower()]
                        findings.append(SecurityFinding(
                            severity=vuln['severity'],
                            category="dependency",
                            title=f"Vulnerable dependency: {package_name}",
                            description=vuln['description'],
                            file_path=str(requirements_path),
                            cve_id=vuln.get('cve'),
                            recommendation=f"Update {package_name} to latest secure version"
                        ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity="INFO",
                category="dependency",
                title="Dependency scan failed",
                description=f"Could not scan dependencies: {e}",
                file_path=str(requirements_path)
            ))
        
        return findings
    
    def scan_imports(self, python_files: List[Path]) -> List[SecurityFinding]:
        """Scan Python imports for dangerous packages."""
        findings = []
        dangerous_imports = {
            'pickle': 'HIGH - Deserialization risk',
            'eval': 'CRITICAL - Code execution risk',
            'subprocess': 'MEDIUM - Command execution risk',
            'os': 'LOW - System access'
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            module_name = alias.name
                            if node.__class__ == ast.ImportFrom and node.module:
                                module_name = node.module
                            
                            base_module = module_name.split('.')[0]
                            if base_module in dangerous_imports:
                                severity_desc = dangerous_imports[base_module]
                                severity = severity_desc.split(' - ')[0]
                                
                                findings.append(SecurityFinding(
                                    severity=severity,
                                    category="import",
                                    title=f"Potentially dangerous import: {module_name}",
                                    description=severity_desc,
                                    file_path=str(file_path),
                                    line_number=getattr(node, 'lineno', None),
                                    recommendation=f"Review usage of {module_name} for security implications"
                                ))
            
            except Exception:
                # Skip files that can't be parsed
                continue
        
        return findings


class ConfigurationScanner:
    """Scans configuration files for security issues."""
    
    def scan_config_files(self, project_path: Path) -> List[SecurityFinding]:
        """Scan configuration files for security issues."""
        findings = []
        
        # Common config file patterns
        config_patterns = [
            "*.yaml", "*.yml", "*.json", "*.cfg", "*.ini",
            ".env*", "config.*", "settings.*"
        ]
        
        config_files = []
        for pattern in config_patterns:
            config_files.extend(project_path.rglob(pattern))
        
        for config_file in config_files:
            findings.extend(self._scan_config_file(config_file))
        
        return findings
    
    def _scan_config_file(self, config_file: Path) -> List[SecurityFinding]:
        """Scan individual configuration file."""
        findings = []
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded secrets
            secret_patterns = [
                (r'password\s*[:=]\s*["\']?[^"\'\s]+["\']?', 'HIGH', 'Hardcoded password in config'),
                (r'api_key\s*[:=]\s*["\']?[^"\'\s]+["\']?', 'HIGH', 'Hardcoded API key in config'),
                (r'secret\s*[:=]\s*["\']?[^"\'\s]+["\']?', 'HIGH', 'Hardcoded secret in config'),
                (r'token\s*[:=]\s*["\']?[^"\'\s]+["\']?', 'MEDIUM', 'Hardcoded token in config'),
                (r'debug\s*[:=]\s*true', 'MEDIUM', 'Debug mode enabled'),
                (r'ssl_verify\s*[:=]\s*false', 'HIGH', 'SSL verification disabled'),
            ]
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern, severity, description in secret_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append(SecurityFinding(
                            severity=severity,
                            category="config",
                            title=f"Configuration security issue: {description}",
                            description=description,
                            file_path=str(config_file),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation="Use environment variables or secure secret management"
                        ))
        
        except Exception as e:
            findings.append(SecurityFinding(
                severity="INFO",
                category="config",
                title="Config scan failed",
                description=f"Could not scan config file: {e}",
                file_path=str(config_file)
            ))
        
        return findings


class PermissionScanner:
    """Scans file permissions and access controls."""
    
    def scan_permissions(self, project_path: Path) -> List[SecurityFinding]:
        """Scan file permissions for security issues."""
        findings = []
        
        # Check for world-writable files
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                try:
                    # Check file permissions (Unix-like systems)
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check if world-writable (others can write)
                    if mode & 0o002:
                        findings.append(SecurityFinding(
                            severity="MEDIUM",
                            category="permissions",
                            title="World-writable file",
                            description="File is writable by all users",
                            file_path=str(file_path),
                            recommendation="Restrict file permissions to necessary users only"
                        ))
                    
                    # Check for executable config files
                    if mode & 0o111 and any(ext in file_path.suffix for ext in ['.json', '.yaml', '.yml', '.cfg', '.ini']):
                        findings.append(SecurityFinding(
                            severity="LOW",
                            category="permissions", 
                            title="Executable configuration file",
                            description="Configuration file has execute permissions",
                            file_path=str(file_path),
                            recommendation="Remove execute permissions from configuration files"
                        ))
                
                except OSError:
                    # Permission denied or other OS error
                    continue
        
        return findings


class SecurityScanner:
    """Main security scanner orchestrating all security checks."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.code_analyzer = StaticCodeAnalyzer()
        self.dependency_scanner = DependencyScanner()
        self.config_scanner = ConfigurationScanner()
        self.permission_scanner = PermissionScanner()
    
    def scan(self) -> SecurityReport:
        """Perform comprehensive security scan."""
        report = SecurityReport(
            scan_timestamp=datetime.now().isoformat(),
            project_path=str(self.project_path),
            total_files_scanned=0
        )
        
        print(f"🔍 Starting security scan of {self.project_path}")
        
        # Find Python files
        python_files = list(self.project_path.rglob('*.py'))
        report.total_files_scanned = len(python_files)
        
        print(f"📁 Found {len(python_files)} Python files to scan")
        
        # Static code analysis
        print("🔬 Performing static code analysis...")
        for py_file in python_files:
            findings = self.code_analyzer.analyze_file(py_file)
            for finding in findings:
                report.add_finding(finding)
        
        # Dependency scanning
        print("📦 Scanning dependencies...")
        requirements_files = [
            self.project_path / 'requirements.txt',
            self.project_path / 'pyproject.toml',
        ]
        
        for req_file in requirements_files:
            if req_file.exists():
                findings = self.dependency_scanner.scan_requirements(req_file)
                for finding in findings:
                    report.add_finding(finding)
        
        # Import analysis
        findings = self.dependency_scanner.scan_imports(python_files)
        for finding in findings:
            report.add_finding(finding)
        
        # Configuration scanning
        print("⚙️  Scanning configuration files...")
        findings = self.config_scanner.scan_config_files(self.project_path)
        for finding in findings:
            report.add_finding(finding)
        
        # Permission scanning
        print("🔐 Scanning file permissions...")
        findings = self.permission_scanner.scan_permissions(self.project_path)
        for finding in findings:
            report.add_finding(finding)
        
        return report
    
    def generate_report(self, report: SecurityReport, output_file: Optional[str] = None) -> str:
        """Generate human-readable security report."""
        
        # Console output
        print("\n" + "="*80)
        print("🛡️  SECURITY SCAN RESULTS")
        print("="*80)
        
        severity_counts = report.get_severity_counts()
        
        print(f"\n📊 SUMMARY:")
        print(f"   • Total files scanned: {report.total_files_scanned}")
        print(f"   • Total findings: {len(report.findings)}")
        print(f"   • CRITICAL: {severity_counts['CRITICAL']}")
        print(f"   • HIGH: {severity_counts['HIGH']}")
        print(f"   • MEDIUM: {severity_counts['MEDIUM']}")
        print(f"   • LOW: {severity_counts['LOW']}")
        print(f"   • INFO: {severity_counts['INFO']}")
        
        # Risk level assessment
        risk_level = "LOW"
        if severity_counts['CRITICAL'] > 0:
            risk_level = "CRITICAL"
        elif severity_counts['HIGH'] > 0:
            risk_level = "HIGH"
        elif severity_counts['MEDIUM'] > 0:
            risk_level = "MEDIUM"
        
        print(f"\n🎯 OVERALL RISK LEVEL: {risk_level}")
        
        # Detailed findings
        if report.findings:
            print(f"\n🔍 DETAILED FINDINGS:")
            print("-" * 80)
            
            # Group by severity
            findings_by_severity = {}
            for finding in report.findings:
                severity = finding.severity
                if severity not in findings_by_severity:
                    findings_by_severity[severity] = []
                findings_by_severity[severity].append(finding)
            
            # Display in order of severity
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
                if severity in findings_by_severity:
                    print(f"\n🚨 {severity} SEVERITY ({len(findings_by_severity[severity])} findings):")
                    
                    for i, finding in enumerate(findings_by_severity[severity][:10], 1):  # Limit to top 10
                        print(f"\n  {i}. {finding.title}")
                        print(f"     Category: {finding.category}")
                        print(f"     Description: {finding.description}")
                        if finding.file_path:
                            location = finding.file_path
                            if finding.line_number:
                                location += f":{finding.line_number}"
                            print(f"     Location: {location}")
                        if finding.code_snippet:
                            print(f"     Code: {finding.code_snippet}")
                        if finding.recommendation:
                            print(f"     Recommendation: {finding.recommendation}")
                        if finding.cve_id:
                            print(f"     CVE: {finding.cve_id}")
                    
                    if len(findings_by_severity[severity]) > 10:
                        remaining = len(findings_by_severity[severity]) - 10
                        print(f"\n     ... and {remaining} more {severity.lower()} findings")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if severity_counts['CRITICAL'] > 0:
            print("   • Address CRITICAL findings immediately - these pose severe security risks")
        if severity_counts['HIGH'] > 0:
            print("   • Address HIGH findings as priority - these pose significant security risks")
        if severity_counts['MEDIUM'] > 0:
            print("   • Review MEDIUM findings and implement fixes where appropriate")
        print("   • Implement security code review process")
        print("   • Consider using automated security scanning in CI/CD pipeline")
        print("   • Regular dependency updates and vulnerability monitoring")
        
        print("\n" + "="*80)
        
        # Save to file if requested
        if output_file:
            print(f"💾 Saving detailed report to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        
        return json.dumps(report.to_dict(), indent=2, default=str)


def main():
    """Main entry point for security scanner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security scanner for Lunar Habitat RL Suite")
    parser.add_argument("project_path", nargs='?', default=".", help="Path to project directory")
    parser.add_argument("--output", "-o", help="Output file for detailed JSON report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress console output")
    
    args = parser.parse_args()
    
    try:
        scanner = SecurityScanner(args.project_path)
        report = scanner.scan()
        
        if not args.quiet:
            scanner.generate_report(report, args.output)
        elif args.output:
            with open(args.output, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Exit with appropriate code
        severity_counts = report.get_severity_counts()
        if severity_counts['CRITICAL'] > 0:
            sys.exit(3)  # Critical issues
        elif severity_counts['HIGH'] > 0:
            sys.exit(2)  # High severity issues
        elif severity_counts['MEDIUM'] > 0:
            sys.exit(1)  # Medium severity issues
        else:
            sys.exit(0)  # No significant issues
    
    except Exception as e:
        print(f"❌ Error during security scan: {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()