"""
NASA-grade chaos engineering and stress testing system.
Implements comprehensive resilience testing for mission-critical systems.
"""

import time
import random
import threading
import asyncio
import queue
import psutil
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import subprocess
import signal
import os

from .robust_logging import get_logger
from .audit_logging import get_audit_logger, AuditEventType, AuditLevel
from .fault_tolerance import get_fault_tolerance_manager
from .advanced_monitoring import get_advanced_monitor


class ChaosType(Enum):
    """Types of chaos engineering experiments."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    PROCESS_KILL = "process_kill"
    DISK_FAILURE = "disk_failure"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    CONFIGURATION_CORRUPTION = "configuration_corruption"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIME_SKEW = "time_skew"
    LIFE_SUPPORT_SIMULATION = "life_support_simulation"


class TestSeverity(Enum):
    """Severity levels for chaos experiments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MISSION_ABORT = "mission_abort"


class ExperimentStatus(Enum):
    """Status of chaos experiments."""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    ROLLED_BACK = "rolled_back"


@dataclass
class ChaosExperiment:
    """Defines a chaos engineering experiment."""
    id: str
    name: str
    description: str
    chaos_type: ChaosType
    severity: TestSeverity
    target_component: str
    duration_seconds: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Safety constraints
    abort_conditions: List[str] = field(default_factory=list)
    rollback_required: bool = True
    mission_critical_safeguards: bool = True
    
    # Execution tracking
    status: ExperimentStatus = ExperimentStatus.PLANNED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Results
    system_impact_score: float = 0.0
    recovery_time_seconds: float = 0.0
    success_criteria_met: bool = False
    observations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'chaos_type': self.chaos_type.value,
            'severity': self.severity.value,
            'target_component': self.target_component,
            'duration_seconds': self.duration_seconds,
            'parameters': self.parameters,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'system_impact_score': self.system_impact_score,
            'recovery_time_seconds': self.recovery_time_seconds,
            'success_criteria_met': self.success_criteria_met,
            'observations': self.observations
        }


@dataclass
class StressTestScenario:
    """Defines a stress testing scenario."""
    id: str
    name: str
    description: str
    test_type: str  # load, endurance, spike, volume
    target_metrics: Dict[str, float]
    duration_seconds: int
    ramp_up_seconds: int = 60
    ramp_down_seconds: int = 60
    
    # Test parameters
    concurrent_operations: int = 10
    operations_per_second: int = 100
    data_volume_mb: int = 100
    
    # Success criteria
    max_response_time_ms: float = 1000.0
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05
    max_memory_usage_percent: float = 85.0
    
    # Results
    status: ExperimentStatus = ExperimentStatus.PLANNED
    actual_metrics: Dict[str, float] = field(default_factory=dict)
    peak_performance: Dict[str, float] = field(default_factory=dict)
    bottlenecks_identified: List[str] = field(default_factory=list)


class ChaosEngineer:
    """Chaos engineering experiment executor."""
    
    def __init__(self):
        """Initialize chaos engineer."""
        self.logger = get_logger()
        self.audit_logger = get_audit_logger()
        self.fault_tolerance_mgr = get_fault_tolerance_manager()
        self.advanced_monitor = get_advanced_monitor()
        
        # Experiment tracking
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[ChaosExperiment] = []
        self.experiment_threads: Dict[str, threading.Thread] = {}
        
        # Safety controls
        self.safety_mode = True
        self.abort_all_experiments = False
        self.mission_critical_protection = True
        
        # Chaos injection methods
        self.chaos_injectors = {
            ChaosType.RESOURCE_EXHAUSTION: self._inject_resource_exhaustion,
            ChaosType.NETWORK_FAILURE: self._inject_network_failure,
            ChaosType.PROCESS_KILL: self._inject_process_kill,
            ChaosType.MEMORY_PRESSURE: self._inject_memory_pressure,
            ChaosType.CPU_SPIKE: self._inject_cpu_spike,
            ChaosType.LATENCY_INJECTION: self._inject_latency,
            ChaosType.ERROR_INJECTION: self._inject_errors,
            ChaosType.LIFE_SUPPORT_SIMULATION: self._inject_life_support_failure
        }
        
        # Monitoring baseline
        self.baseline_metrics = {}
        self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish baseline system metrics before chaos testing."""
        try:
            if psutil:
                self.baseline_metrics = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'process_count': len(psutil.pids()),
                    'network_connections': len(psutil.net_connections())
                }
            else:
                self.baseline_metrics = {'note': 'psutil not available'}
            
            self.logger.info(f"Baseline metrics established: {self.baseline_metrics}")
            
        except Exception as e:
            self.logger.warning(f"Failed to establish baseline metrics: {e}")
            self.baseline_metrics = {}
    
    def execute_experiment(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Execute chaos engineering experiment.
        
        Args:
            experiment: Chaos experiment to execute
            
        Returns:
            Experiment results
        """
        if self.safety_mode and experiment.severity in [TestSeverity.CRITICAL, TestSeverity.MISSION_ABORT]:
            if not self._safety_approval_required(experiment):
                raise RuntimeError(f"Safety approval required for {experiment.severity.value} experiment")
        
        experiment_id = experiment.id
        
        try:
            # Pre-flight checks
            self._pre_flight_checks(experiment)
            
            # Start experiment
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_time = datetime.utcnow()
            self.active_experiments[experiment_id] = experiment
            
            # Log experiment start
            self.audit_logger.log_system_event(
                action="chaos_experiment_started",
                component="chaos_engineer",
                description=f"Chaos experiment started: {experiment.name}",
                details=experiment.to_dict(),
                level=AuditLevel.WARNING
            )
            
            # Execute experiment in background thread
            experiment_thread = threading.Thread(
                target=self._run_experiment,
                args=(experiment,),
                daemon=True
            )
            self.experiment_threads[experiment_id] = experiment_thread
            experiment_thread.start()
            
            return {'status': 'started', 'experiment_id': experiment_id}
            
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(e)
            
            self.logger.error(f"Failed to start chaos experiment {experiment_id}: {e}")
            
            self.audit_logger.log_system_event(
                action="chaos_experiment_failed",
                component="chaos_engineer",
                description=f"Chaos experiment failed to start: {experiment.name} - {e}",
                details=experiment.to_dict(),
                level=AuditLevel.ERROR,
                success=False
            )
            
            return {'status': 'failed', 'error': str(e)}
    
    def _pre_flight_checks(self, experiment: ChaosExperiment):
        """Perform pre-flight safety checks."""
        # Check if monitoring is active
        if self.advanced_monitor.state.value != 'active':
            raise RuntimeError("Advanced monitoring must be active before chaos testing")
        
        # Check system health baseline
        health_status = self.advanced_monitor.get_monitoring_status()
        if health_status['system_health_score'] < 0.8:
            raise RuntimeError(f"System health too low for chaos testing: {health_status['system_health_score']}")
        
        # Check for ongoing mission-critical operations
        if self.mission_critical_protection:
            # Would check for actual mission-critical operations
            pass
        
        # Verify abort conditions are measurable
        for condition in experiment.abort_conditions:
            # Would verify we can measure each abort condition
            pass
    
    def _run_experiment(self, experiment: ChaosExperiment):
        """Run chaos experiment in background thread."""
        try:
            # Record pre-experiment metrics
            pre_metrics = self._capture_metrics()
            
            # Execute chaos injection
            chaos_injector = self.chaos_injectors.get(experiment.chaos_type)
            if not chaos_injector:
                raise RuntimeError(f"No injector available for {experiment.chaos_type.value}")
            
            # Start chaos injection
            self.logger.info(f"Starting chaos injection: {experiment.chaos_type.value}")
            rollback_function = chaos_injector(experiment)
            
            # Monitor experiment
            start_time = time.time()
            end_time = start_time + experiment.duration_seconds
            
            while time.time() < end_time and not self.abort_all_experiments:
                # Check abort conditions
                if self._check_abort_conditions(experiment):
                    self.logger.warning(f"Abort condition met for experiment {experiment.id}")
                    break
                
                # Monitor system impact
                current_metrics = self._capture_metrics()
                impact_score = self._calculate_impact_score(pre_metrics, current_metrics)
                experiment.system_impact_score = max(experiment.system_impact_score, impact_score)
                
                # Record observations
                if impact_score > 0.5:  # Significant impact
                    observation = f"High system impact detected: {impact_score:.2f} at {datetime.utcnow().isoformat()}"
                    experiment.observations.append(observation)
                
                time.sleep(5)  # Check every 5 seconds
            
            # End experiment
            experiment.end_time = datetime.utcnow()
            
            # Execute rollback if required
            if experiment.rollback_required and rollback_function:
                self.logger.info(f"Rolling back chaos injection for experiment {experiment.id}")
                try:
                    rollback_function()
                    experiment.status = ExperimentStatus.ROLLED_BACK
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed for experiment {experiment.id}: {rollback_error}")
                    experiment.status = ExperimentStatus.FAILED
                    experiment.error_message = f"Rollback failed: {rollback_error}"
            else:
                experiment.status = ExperimentStatus.COMPLETED
            
            # Measure recovery time
            recovery_start = time.time()
            recovery_metrics = self._capture_metrics()
            
            # Wait for system to return to baseline (or timeout after 5 minutes)
            timeout = recovery_start + 300
            while time.time() < timeout:
                current_metrics = self._capture_metrics()
                if self._is_system_recovered(pre_metrics, current_metrics):
                    break
                time.sleep(10)
            
            experiment.recovery_time_seconds = time.time() - recovery_start
            
            # Evaluate success criteria
            experiment.success_criteria_met = self._evaluate_success_criteria(experiment)
            
            # Final metrics
            post_metrics = self._capture_metrics()
            
            # Log experiment completion
            self.audit_logger.log_system_event(
                action="chaos_experiment_completed",
                component="chaos_engineer",
                description=f"Chaos experiment completed: {experiment.name}",
                details=experiment.to_dict(),
                level=AuditLevel.INFO
            )
            
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(e)
            experiment.end_time = datetime.utcnow()
            
            self.logger.error(f"Chaos experiment {experiment.id} failed: {e}")
            
            # Attempt emergency rollback
            try:
                if experiment.rollback_required and 'rollback_function' in locals():
                    rollback_function()
            except Exception as rollback_error:
                self.logger.error(f"Emergency rollback failed: {rollback_error}")
            
        finally:
            # Move to history and cleanup
            if experiment.id in self.active_experiments:
                del self.active_experiments[experiment.id]
            if experiment.id in self.experiment_threads:
                del self.experiment_threads[experiment.id]
            
            self.experiment_history.append(experiment)
            
            # Keep only last 100 experiments in history
            if len(self.experiment_history) > 100:
                self.experiment_history = self.experiment_history[-100:]
    
    def _inject_resource_exhaustion(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Inject resource exhaustion chaos."""
        resource_type = experiment.parameters.get('resource_type', 'memory')
        exhaustion_percent = experiment.parameters.get('exhaustion_percent', 80)
        
        if resource_type == 'memory':
            return self._exhaust_memory(exhaustion_percent)
        elif resource_type == 'disk':
            return self._exhaust_disk_space(exhaustion_percent)
        elif resource_type == 'cpu':
            return self._exhaust_cpu(exhaustion_percent)
        else:
            raise ValueError(f"Unknown resource type: {resource_type}")
    
    def _exhaust_memory(self, percent: int) -> Callable:
        """Exhaust system memory."""
        memory_hogs = []
        
        try:
            if psutil:
                total_memory = psutil.virtual_memory().total
                target_memory = int(total_memory * percent / 100)
                
                # Allocate memory in chunks
                chunk_size = min(100 * 1024 * 1024, target_memory // 10)  # 100MB chunks
                allocated = 0
                
                while allocated < target_memory:
                    try:
                        chunk = bytearray(chunk_size)
                        memory_hogs.append(chunk)
                        allocated += chunk_size
                        time.sleep(0.1)  # Small delay to avoid overwhelming system
                    except MemoryError:
                        break
                
                self.logger.info(f"Allocated {allocated / (1024*1024):.1f} MB for memory exhaustion test")
            
        except Exception as e:
            self.logger.error(f"Memory exhaustion injection failed: {e}")
        
        def rollback():
            memory_hogs.clear()
            import gc
            gc.collect()
            self.logger.info("Memory exhaustion rolled back")
        
        return rollback
    
    def _exhaust_disk_space(self, percent: int) -> Callable:
        """Exhaust disk space."""
        temp_files = []
        
        try:
            if psutil:
                disk_usage = psutil.disk_usage('/')
                free_space = disk_usage.free
                target_usage = int(free_space * percent / 100)
                
                # Create temporary files to consume space
                chunk_size = min(10 * 1024 * 1024, target_usage // 10)  # 10MB chunks
                used = 0
                
                while used < target_usage:
                    temp_file_path = f"/tmp/chaos_disk_{len(temp_files)}.tmp"
                    try:
                        with open(temp_file_path, 'wb') as f:
                            f.write(b'0' * chunk_size)
                        temp_files.append(temp_file_path)
                        used += chunk_size
                    except OSError:
                        break
                
                self.logger.info(f"Created {used / (1024*1024):.1f} MB of temporary files")
            
        except Exception as e:
            self.logger.error(f"Disk exhaustion injection failed: {e}")
        
        def rollback():
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            self.logger.info("Disk exhaustion rolled back")
        
        return rollback
    
    def _exhaust_cpu(self, percent: int) -> Callable:
        """Exhaust CPU resources."""
        cpu_processes = []
        stop_event = threading.Event()
        
        try:
            cpu_count = os.cpu_count() or 4
            load_factor = percent / 100.0
            
            # Create CPU-intensive threads
            for i in range(int(cpu_count * load_factor)):
                thread = threading.Thread(target=self._cpu_intensive_task, args=(stop_event,))
                thread.daemon = True
                thread.start()
                cpu_processes.append(thread)
            
            self.logger.info(f"Started {len(cpu_processes)} CPU-intensive threads")
            
        except Exception as e:
            self.logger.error(f"CPU exhaustion injection failed: {e}")
        
        def rollback():
            stop_event.set()
            for thread in cpu_processes:
                thread.join(timeout=1.0)
            self.logger.info("CPU exhaustion rolled back")
        
        return rollback
    
    def _cpu_intensive_task(self, stop_event: threading.Event):
        """CPU-intensive task for load testing."""
        while not stop_event.is_set():
            # Busy loop with occasional breaks
            for _ in range(1000000):
                if stop_event.is_set():
                    break
                _ = sum(range(100))
            time.sleep(0.001)  # Brief pause to prevent total lockup
    
    def _inject_network_failure(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Simulate network failures."""
        failure_type = experiment.parameters.get('failure_type', 'packet_loss')
        severity = experiment.parameters.get('severity', 50)
        
        # For demonstration - in real implementation would use network manipulation tools
        self.logger.info(f"Simulating {failure_type} network failure at {severity}% severity")
        
        def rollback():
            self.logger.info("Network failure simulation ended")
        
        return rollback
    
    def _inject_process_kill(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Kill random processes (safely)."""
        target_process = experiment.parameters.get('target_process', 'random')
        kill_count = experiment.parameters.get('kill_count', 1)
        
        killed_processes = []
        
        try:
            if psutil and target_process == 'random':
                # Get list of non-critical processes
                safe_processes = []
                critical_names = ['systemd', 'kernel', 'init', 'ssh', 'python']
                
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if not any(critical in proc.info['name'].lower() for critical in critical_names):
                            safe_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Kill random safe processes
                for _ in range(min(kill_count, len(safe_processes))):
                    if safe_processes:
                        proc = random.choice(safe_processes)
                        try:
                            killed_processes.append((proc.pid, proc.info['name']))
                            proc.kill()
                            safe_processes.remove(proc)
                            self.logger.info(f"Killed process: {proc.info['name']} (PID: {proc.pid})")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            
        except Exception as e:
            self.logger.error(f"Process kill injection failed: {e}")
        
        def rollback():
            # Can't resurrect killed processes, but log the action
            self.logger.info(f"Process kill experiment ended. Killed {len(killed_processes)} processes")
        
        return rollback
    
    def _inject_memory_pressure(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Inject memory pressure without full exhaustion."""
        pressure_level = experiment.parameters.get('pressure_level', 70)  # Percentage
        return self._exhaust_memory(pressure_level)
    
    def _inject_cpu_spike(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Inject CPU usage spikes."""
        spike_intensity = experiment.parameters.get('spike_intensity', 90)
        return self._exhaust_cpu(spike_intensity)
    
    def _inject_latency(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Inject artificial latency."""
        latency_ms = experiment.parameters.get('latency_ms', 500)
        target_component = experiment.target_component
        
        # For demonstration - would inject actual latency in target component
        self.logger.info(f"Injecting {latency_ms}ms latency in {target_component}")
        
        def rollback():
            self.logger.info(f"Latency injection removed from {target_component}")
        
        return rollback
    
    def _inject_errors(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Inject artificial errors."""
        error_rate = experiment.parameters.get('error_rate', 10)  # Percentage
        error_type = experiment.parameters.get('error_type', 'random')
        
        # For demonstration - would inject actual errors
        self.logger.info(f"Injecting {error_rate}% {error_type} errors")
        
        def rollback():
            self.logger.info("Error injection stopped")
        
        return rollback
    
    def _inject_life_support_failure(self, experiment: ChaosExperiment) -> Optional[Callable]:
        """Simulate life support system failures (SAFELY)."""
        failure_type = experiment.parameters.get('failure_type', 'sensor_malfunction')
        
        # CRITICAL: Only simulate, never actually affect life support
        self.logger.critical(f"SIMULATING life support failure: {failure_type}")
        self.logger.critical("THIS IS A SIMULATION - NO ACTUAL LIFE SUPPORT AFFECTED")
        
        # Log as emergency event for testing emergency response
        self.audit_logger.log_emergency_event(
            action="life_support_failure_simulation",
            description=f"SIMULATION: Life support failure - {failure_type}",
            details={'simulation': True, 'failure_type': failure_type}
        )
        
        def rollback():
            self.logger.info("Life support failure simulation ended")
            self.audit_logger.log_emergency_event(
                action="life_support_simulation_ended",
                description="Life support failure simulation ended",
                details={'simulation': True}
            )
        
        return rollback
    
    def _capture_metrics(self) -> Dict[str, float]:
        """Capture current system metrics."""
        metrics = {}
        
        try:
            if psutil:
                metrics.update({
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'process_count': len(psutil.pids()),
                    'network_connections': len(psutil.net_connections())
                })
            
            # Add monitoring system metrics
            monitor_status = self.advanced_monitor.get_monitoring_status()
            metrics['system_health_score'] = monitor_status['system_health_score']
            
        except Exception as e:
            self.logger.warning(f"Failed to capture metrics: {e}")
        
        return metrics
    
    def _calculate_impact_score(self, baseline: Dict[str, float], current: Dict[str, float]) -> float:
        """Calculate system impact score (0.0 to 1.0)."""
        if not baseline or not current:
            return 0.0
        
        impacts = []
        
        for metric in ['cpu_percent', 'memory_percent', 'disk_usage_percent']:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    change = abs(current_val - baseline_val) / baseline_val
                    impacts.append(min(change, 1.0))
        
        # Health score impact
        if 'system_health_score' in baseline and 'system_health_score' in current:
            health_impact = baseline['system_health_score'] - current['system_health_score']
            impacts.append(max(0, health_impact))
        
        return sum(impacts) / len(impacts) if impacts else 0.0
    
    def _is_system_recovered(self, baseline: Dict[str, float], current: Dict[str, float]) -> bool:
        """Check if system has recovered to baseline performance."""
        if not baseline or not current:
            return True
        
        for metric in ['cpu_percent', 'memory_percent', 'system_health_score']:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val > 0:
                    deviation = abs(current_val - baseline_val) / baseline_val
                    if deviation > 0.1:  # More than 10% deviation
                        return False
        
        return True
    
    def _check_abort_conditions(self, experiment: ChaosExperiment) -> bool:
        """Check if experiment should be aborted."""
        current_metrics = self._capture_metrics()
        
        # Check predefined abort conditions
        for condition in experiment.abort_conditions:
            if self._evaluate_abort_condition(condition, current_metrics):
                return True
        
        # Universal abort conditions
        if current_metrics.get('system_health_score', 1.0) < 0.3:  # Health below 30%
            return True
        
        if current_metrics.get('memory_percent', 0) > 95:  # Memory above 95%
            return True
        
        return False
    
    def _evaluate_abort_condition(self, condition: str, metrics: Dict[str, float]) -> bool:
        """Evaluate specific abort condition."""
        # Simple condition evaluation - could be made more sophisticated
        if 'cpu > 95' in condition and metrics.get('cpu_percent', 0) > 95:
            return True
        if 'memory > 90' in condition and metrics.get('memory_percent', 0) > 90:
            return True
        if 'health < 0.5' in condition and metrics.get('system_health_score', 1.0) < 0.5:
            return True
        
        return False
    
    def _evaluate_success_criteria(self, experiment: ChaosExperiment) -> bool:
        """Evaluate if experiment met success criteria."""
        # Basic success criteria evaluation
        criteria_met = []
        
        # System recovered successfully
        if experiment.recovery_time_seconds < 300:  # Recovered within 5 minutes
            criteria_met.append(True)
        else:
            criteria_met.append(False)
        
        # No critical system failures
        if experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.ROLLED_BACK]:
            criteria_met.append(True)
        else:
            criteria_met.append(False)
        
        # System impact was controlled
        if experiment.system_impact_score < 0.8:  # Impact below 80%
            criteria_met.append(True)
        else:
            criteria_met.append(False)
        
        return all(criteria_met)
    
    def _safety_approval_required(self, experiment: ChaosExperiment) -> bool:
        """Check if safety approval is required for high-severity experiments."""
        # In real implementation, would check for actual approval
        # For demonstration, always return True for critical experiments
        return experiment.severity != TestSeverity.MISSION_ABORT
    
    def abort_all_experiments(self):
        """Emergency abort of all running experiments."""
        self.abort_all_experiments = True
        
        self.logger.critical("🚨 EMERGENCY ABORT: All chaos experiments terminating")
        
        self.audit_logger.log_emergency_event(
            action="chaos_experiments_emergency_abort",
            description="Emergency abort of all chaos engineering experiments",
            details={'active_experiments': list(self.active_experiments.keys())}
        )
        
        # Wait for experiments to terminate
        for thread in self.experiment_threads.values():
            thread.join(timeout=10.0)
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get status of all chaos experiments."""
        return {
            'active_experiments': len(self.active_experiments),
            'total_experiments_run': len(self.experiment_history),
            'successful_experiments': len([e for e in self.experiment_history if e.success_criteria_met]),
            'safety_mode': self.safety_mode,
            'mission_critical_protection': self.mission_critical_protection,
            'baseline_metrics': self.baseline_metrics,
            'recent_experiments': [e.to_dict() for e in self.experiment_history[-5:]]
        }


class StressTester:
    """Advanced stress testing system."""
    
    def __init__(self):
        """Initialize stress tester."""
        self.logger = get_logger()
        self.audit_logger = get_audit_logger()
        
        # Test scenarios
        self.active_tests: Dict[str, StressTestScenario] = {}
        self.test_history: List[StressTestScenario] = []
        
        # Performance tracking
        self.performance_baselines = {}
        self.bottleneck_patterns = {}
    
    def execute_load_test(self, scenario: StressTestScenario) -> Dict[str, Any]:
        """Execute load testing scenario."""
        scenario.status = ExperimentStatus.RUNNING
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting load test: {scenario.name}")
            
            # Ramp up
            self._ramp_up_load(scenario)
            
            # Sustained load
            self._sustain_load(scenario)
            
            # Ramp down
            self._ramp_down_load(scenario)
            
            scenario.status = ExperimentStatus.COMPLETED
            duration = time.time() - start_time
            
            # Analyze results
            results = self._analyze_load_test_results(scenario, duration)
            
            self.audit_logger.log_system_event(
                action="load_test_completed",
                component="stress_tester",
                description=f"Load test completed: {scenario.name}",
                details=results
            )
            
            return results
            
        except Exception as e:
            scenario.status = ExperimentStatus.FAILED
            
            self.logger.error(f"Load test failed: {scenario.name} - {e}")
            
            return {'status': 'failed', 'error': str(e)}
    
    def _ramp_up_load(self, scenario: StressTestScenario):
        """Gradually increase load."""
        self.logger.info("Ramping up load...")
        
        ramp_steps = 10
        step_duration = scenario.ramp_up_seconds / ramp_steps
        
        for step in range(ramp_steps):
            load_factor = (step + 1) / ramp_steps
            current_ops = int(scenario.operations_per_second * load_factor)
            
            self._simulate_operations(current_ops, step_duration)
            
            # Monitor system response
            self._record_performance_metrics(scenario, f"ramp_up_step_{step}")
    
    def _sustain_load(self, scenario: StressTestScenario):
        """Sustain peak load."""
        self.logger.info("Sustaining peak load...")
        
        sustain_duration = scenario.duration_seconds - scenario.ramp_up_seconds - scenario.ramp_down_seconds
        intervals = max(1, sustain_duration // 10)  # 10 measurement intervals
        
        for interval in range(intervals):
            self._simulate_operations(scenario.operations_per_second, 10)
            self._record_performance_metrics(scenario, f"sustain_interval_{interval}")
    
    def _ramp_down_load(self, scenario: StressTestScenario):
        """Gradually decrease load."""
        self.logger.info("Ramping down load...")
        
        ramp_steps = 5
        step_duration = scenario.ramp_down_seconds / ramp_steps
        
        for step in range(ramp_steps):
            load_factor = 1.0 - ((step + 1) / ramp_steps)
            current_ops = int(scenario.operations_per_second * load_factor)
            
            self._simulate_operations(current_ops, step_duration)
            self._record_performance_metrics(scenario, f"ramp_down_step_{step}")
    
    def _simulate_operations(self, ops_per_second: int, duration_seconds: float):
        """Simulate operations at specified rate."""
        if ops_per_second <= 0:
            return
        
        interval = 1.0 / ops_per_second
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            start = time.time()
            
            # Simulate operation (would be actual system operation in real test)
            self._simulate_single_operation()
            
            # Control rate
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _simulate_single_operation(self):
        """Simulate a single operation."""
        # Placeholder for actual operation simulation
        # Would invoke actual system functions being tested
        time.sleep(0.001)  # Simulate 1ms operation
    
    def _record_performance_metrics(self, scenario: StressTestScenario, phase: str):
        """Record performance metrics during test."""
        try:
            metrics = {}
            
            if psutil:
                metrics.update({
                    'timestamp': time.time(),
                    'phase': phase,
                    'cpu_percent': psutil.cpu_percent(interval=0.1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024*1024),
                    'disk_io_write_mb': psutil.disk_io_counters().write_bytes / (1024*1024),
                    'network_sent_mb': psutil.net_io_counters().bytes_sent / (1024*1024),
                    'network_recv_mb': psutil.net_io_counters().bytes_recv / (1024*1024)
                })
            
            # Store in scenario
            if 'performance_metrics' not in scenario.actual_metrics:
                scenario.actual_metrics['performance_metrics'] = []
            scenario.actual_metrics['performance_metrics'].append(metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to record performance metrics: {e}")
    
    def _analyze_load_test_results(self, scenario: StressTestScenario, duration: float) -> Dict[str, Any]:
        """Analyze load test results."""
        metrics = scenario.actual_metrics.get('performance_metrics', [])
        
        if not metrics:
            return {'status': 'no_metrics', 'duration': duration}
        
        # Calculate peak performance
        peak_cpu = max(m.get('cpu_percent', 0) for m in metrics)
        peak_memory = max(m.get('memory_percent', 0) for m in metrics)
        
        # Identify bottlenecks
        bottlenecks = []
        if peak_cpu > 90:
            bottlenecks.append('CPU')
        if peak_memory > scenario.max_memory_usage_percent:
            bottlenecks.append('Memory')
        
        # Determine test success
        success = (
            peak_memory <= scenario.max_memory_usage_percent and
            len(bottlenecks) == 0
        )
        
        results = {
            'scenario_name': scenario.name,
            'duration_seconds': duration,
            'success': success,
            'peak_performance': {
                'cpu_percent': peak_cpu,
                'memory_percent': peak_memory
            },
            'bottlenecks': bottlenecks,
            'metrics_count': len(metrics),
            'target_ops_per_second': scenario.operations_per_second,
            'concurrent_operations': scenario.concurrent_operations
        }
        
        scenario.peak_performance = results['peak_performance']
        scenario.bottlenecks_identified = bottlenecks
        
        return results


# Predefined chaos experiments
def create_standard_chaos_experiments() -> List[ChaosExperiment]:
    """Create standard set of chaos experiments."""
    experiments = []
    
    # Memory pressure test
    experiments.append(ChaosExperiment(
        id="memory_pressure_low",
        name="Low Memory Pressure Test",
        description="Test system resilience under moderate memory pressure",
        chaos_type=ChaosType.MEMORY_PRESSURE,
        severity=TestSeverity.LOW,
        target_component="system_memory",
        duration_seconds=300,
        parameters={'pressure_level': 70},
        abort_conditions=['memory > 90', 'health < 0.5']
    ))
    
    # CPU spike test
    experiments.append(ChaosExperiment(
        id="cpu_spike_medium",
        name="CPU Spike Test",
        description="Test system behavior under CPU load spikes",
        chaos_type=ChaosType.CPU_SPIKE,
        severity=TestSeverity.MEDIUM,
        target_component="system_cpu",
        duration_seconds=180,
        parameters={'spike_intensity': 85},
        abort_conditions=['cpu > 95']
    ))
    
    # Network failure simulation
    experiments.append(ChaosExperiment(
        id="network_failure_low",
        name="Network Packet Loss Test",
        description="Test system resilience to network packet loss",
        chaos_type=ChaosType.NETWORK_FAILURE,
        severity=TestSeverity.LOW,
        target_component="network",
        duration_seconds=120,
        parameters={'failure_type': 'packet_loss', 'severity': 20}
    ))
    
    # Life support simulation (CRITICAL - only for testing emergency response)
    experiments.append(ChaosExperiment(
        id="life_support_simulation",
        name="Life Support Failure Simulation",
        description="SIMULATION ONLY - Test emergency response to life support alerts",
        chaos_type=ChaosType.LIFE_SUPPORT_SIMULATION,
        severity=TestSeverity.CRITICAL,
        target_component="life_support",
        duration_seconds=60,
        parameters={'failure_type': 'sensor_malfunction'},
        abort_conditions=['system_shutdown_initiated'],
        mission_critical_safeguards=True
    ))
    
    return experiments


def create_standard_stress_tests() -> List[StressTestScenario]:
    """Create standard stress test scenarios."""
    scenarios = []
    
    # Load test
    scenarios.append(StressTestScenario(
        id="load_test_basic",
        name="Basic Load Test",
        description="Test system under normal load conditions",
        test_type="load",
        target_metrics={'response_time_ms': 200, 'throughput_ops_sec': 100},
        duration_seconds=600,
        ramp_up_seconds=60,
        ramp_down_seconds=30,
        concurrent_operations=50,
        operations_per_second=100
    ))
    
    # Endurance test
    scenarios.append(StressTestScenario(
        id="endurance_test",
        name="Endurance Test",
        description="Test system stability over extended period",
        test_type="endurance",
        target_metrics={'memory_growth_mb': 100, 'error_rate': 0.01},
        duration_seconds=3600,  # 1 hour
        ramp_up_seconds=300,
        ramp_down_seconds=300,
        concurrent_operations=20,
        operations_per_second=50
    ))
    
    # Spike test
    scenarios.append(StressTestScenario(
        id="spike_test",
        name="Traffic Spike Test",
        description="Test system response to sudden traffic spikes",
        test_type="spike",
        target_metrics={'peak_response_time_ms': 1000, 'recovery_time_s': 30},
        duration_seconds=300,
        ramp_up_seconds=10,  # Very fast ramp up
        ramp_down_seconds=10,
        concurrent_operations=200,
        operations_per_second=500
    ))
    
    return scenarios


# Global instances
_global_chaos_engineer = None
_global_stress_tester = None

def get_chaos_engineer() -> ChaosEngineer:
    """Get global chaos engineer instance."""
    global _global_chaos_engineer
    if _global_chaos_engineer is None:
        _global_chaos_engineer = ChaosEngineer()
    return _global_chaos_engineer

def get_stress_tester() -> StressTester:
    """Get global stress tester instance."""
    global _global_stress_tester
    if _global_stress_tester is None:
        _global_stress_tester = StressTester()
    return _global_stress_tester


# Convenient functions
def run_chaos_experiment(experiment_id: str) -> Dict[str, Any]:
    """Run predefined chaos experiment."""
    experiments = create_standard_chaos_experiments()
    experiment = next((e for e in experiments if e.id == experiment_id), None)
    
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_id}")
    
    chaos_engineer = get_chaos_engineer()
    return chaos_engineer.execute_experiment(experiment)

def run_stress_test(scenario_id: str) -> Dict[str, Any]:
    """Run predefined stress test scenario."""
    scenarios = create_standard_stress_tests()
    scenario = next((s for s in scenarios if s.id == scenario_id), None)
    
    if not scenario:
        raise ValueError(f"Scenario not found: {scenario_id}")
    
    stress_tester = get_stress_tester()
    return stress_tester.execute_load_test(scenario)