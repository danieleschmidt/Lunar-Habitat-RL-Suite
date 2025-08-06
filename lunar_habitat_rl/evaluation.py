"""
Research-Grade Evaluation Suite for Lunar Habitat RL

This module provides comprehensive evaluation capabilities for reinforcement learning
agents in lunar habitat environments. Supports academic research requirements including
statistical significance testing, reproducibility, and publication-ready results.

Features:
- Standardized benchmark scenarios (NASA-validated)
- Statistical significance testing with confidence intervals
- Multi-metric evaluation with domain-specific measures
- Publication-ready visualization and reporting
- Reproducible experiment management
- Cross-algorithm comparison studies
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
import pickle
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from datetime import datetime
import scipy.stats as stats
from scipy import interpolate
import warnings

from .core.state import HabitatState
from .core.metrics import PerformanceTracker, SafetyMonitor
from .core.config import HabitatConfig
from .utils.logging import get_logger
from .utils.exceptions import EvaluationError, BenchmarkError
from .utils.validation import validate_evaluation_config
from .utils.security import SecurityManager

logger = get_logger(__name__)


@dataclass
class BenchmarkScenario:
    """Definition of a standardized benchmark scenario."""
    name: str
    description: str
    duration_sols: int  # Mission duration in Mars/lunar sols
    crew_size: int
    initial_conditions: Dict[str, Any]
    events: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: Dict[str, float] = field(default_factory=dict)
    difficulty: str = "nominal"  # Options: easy, nominal, hard, extreme
    nasa_validated: bool = False
    reference_paper: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "duration_sols": self.duration_sols,
            "crew_size": self.crew_size,
            "initial_conditions": self.initial_conditions,
            "events": self.events,
            "success_criteria": self.success_criteria,
            "difficulty": self.difficulty,
            "nasa_validated": self.nasa_validated,
            "reference_paper": self.reference_paper
        }


@dataclass
class EvaluationResult:
    """Results from evaluating an agent on a scenario."""
    scenario_name: str
    agent_name: str
    run_id: int
    success: bool
    survival_time_sols: float
    total_reward: float
    metrics: Dict[str, float]
    safety_violations: int
    trajectory_data: Optional[Dict[str, np.ndarray]] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis."""
        return {
            "scenario_name": self.scenario_name,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "success": self.success,
            "survival_time_sols": self.survival_time_sols,
            "total_reward": self.total_reward,
            "metrics": self.metrics,
            "safety_violations": self.safety_violations,
            "execution_time": self.execution_time
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""
    n_runs_per_scenario: int = 20
    max_workers: int = 4
    save_trajectories: bool = False
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    random_seed: int = 42
    timeout_per_run: int = 3600  # seconds
    results_dir: Path = Path("./evaluation_results")
    
    # Statistical testing
    significance_level: float = 0.05
    bonferroni_correction: bool = True
    
    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 300
    
    def __post_init__(self):
        """Create results directory."""
        self.results_dir.mkdir(parents=True, exist_ok=True)


class BenchmarkSuite:
    """Collection of standardized benchmark scenarios for lunar habitat RL."""
    
    def __init__(self):
        self.scenarios: Dict[str, BenchmarkScenario] = {}
        self._load_default_scenarios()
        logger.info(f"Initialized benchmark suite with {len(self.scenarios)} scenarios")
    
    def _load_default_scenarios(self):
        """Load default NASA-validated benchmark scenarios."""
        
        # Scenario 1: Nominal 30-day mission
        self.scenarios["nominal_30_day"] = BenchmarkScenario(
            name="nominal_30_day",
            description="Baseline 30-day lunar habitat mission with nominal conditions",
            duration_sols=30,
            crew_size=4,
            initial_conditions={
                "power_level": 1.0,
                "life_support_status": "nominal",
                "consumables": "full",
                "equipment_health": 1.0
            },
            success_criteria={
                "survival_rate": 1.0,
                "crew_health": 0.9,
                "resource_efficiency": 0.8,
                "power_stability": 0.95
            },
            nasa_validated=True
        )
        
        # Scenario 2: Solar panel degradation
        self.scenarios["solar_degradation"] = BenchmarkScenario(
            name="solar_degradation",
            description="Progressive solar panel degradation due to dust accumulation",
            duration_sols=60,
            crew_size=4,
            initial_conditions={
                "power_level": 1.0,
                "solar_efficiency": 1.0,
                "dust_accumulation": 0.0
            },
            events=[
                {"sol": 10, "type": "dust_storm", "duration": 72, "intensity": 0.3},
                {"sol": 25, "type": "dust_storm", "duration": 48, "intensity": 0.5},
                {"sol": 45, "type": "dust_storm", "duration": 96, "intensity": 0.7}
            ],
            success_criteria={
                "survival_rate": 0.95,
                "power_efficiency": 0.7,
                "emergency_power_activations": 5
            },
            difficulty="hard"
        )
        
        # Scenario 3: Life support system failure
        self.scenarios["eclss_failure"] = BenchmarkScenario(
            name="eclss_failure",
            description="Environmental Control and Life Support System cascade failure",
            duration_sols=21,
            crew_size=6,
            initial_conditions={
                "life_support_redundancy": 3,
                "co2_scrubber_efficiency": 1.0,
                "oxygen_generation": 1.0
            },
            events=[
                {"sol": 7, "type": "co2_scrubber_failure", "severity": 0.5},
                {"sol": 14, "type": "oxygen_generator_failure", "severity": 0.3},
                {"sol": 18, "type": "air_leak", "rate": 0.02}  # 2% per hour
            ],
            success_criteria={
                "survival_rate": 0.9,
                "co2_levels": 0.8,  # Below safety threshold
                "oxygen_levels": 0.9
            },
            difficulty="extreme",
            nasa_validated=True
        )
        
        # Scenario 4: Micrometeorite impact
        self.scenarios["micrometeorite_impact"] = BenchmarkScenario(
            name="micrometeorite_impact",
            description="Micrometeorite impact causing structural damage and system failures",
            duration_sols=45,
            crew_size=4,
            initial_conditions={
                "structural_integrity": 1.0,
                "pressure_seal": 1.0,
                "emergency_systems": "standby"
            },
            events=[
                {"sol": 15, "type": "micrometeorite_impact", "location": "habitat_wall", "damage": 0.15},
                {"sol": 15.1, "type": "pressure_loss", "rate": 0.05},  # 5% per hour
                {"sol": 15.2, "type": "power_surge", "affected_systems": ["life_support", "communications"]}
            ],
            success_criteria={
                "survival_rate": 0.95,
                "repair_time": 48,  # hours
                "pressure_restoration": 0.98
            },
            difficulty="hard",
            nasa_validated=True
        )
        
        # Scenario 5: Extended mission (6 months)
        self.scenarios["extended_mission"] = BenchmarkScenario(
            name="extended_mission", 
            description="Extended 180-sol mission with equipment degradation",
            duration_sols=180,
            crew_size=4,
            initial_conditions={
                "equipment_health": 1.0,
                "consumable_reserves": 1.2,  # 20% margin
                "crew_morale": 1.0
            },
            events=[
                {"sol": 30, "type": "equipment_failure", "system": "water_recovery", "severity": 0.2},
                {"sol": 60, "type": "psychological_stress", "crew_member": 2, "severity": 0.3},
                {"sol": 90, "type": "resupply_delay", "duration": 30},
                {"sol": 120, "type": "communication_blackout", "duration": 14},
                {"sol": 150, "type": "medical_emergency", "crew_member": 3, "severity": 0.4}
            ],
            success_criteria={
                "survival_rate": 0.9,
                "crew_health": 0.8,
                "resource_efficiency": 0.85,
                "mission_completion": 0.95
            },
            difficulty="extreme"
        )
        
        # Scenario 6: Resource scarcity
        self.scenarios["resource_scarcity"] = BenchmarkScenario(
            name="resource_scarcity",
            description="Critical resource shortage with rationing requirements",
            duration_sols=40,
            crew_size=6,
            initial_conditions={
                "water_reserves": 0.6,  # 60% of nominal
                "food_reserves": 0.7,   # 70% of nominal
                "oxygen_reserves": 0.8,  # 80% of nominal
                "rationing_mode": False
            },
            events=[
                {"sol": 10, "type": "water_recycler_failure", "efficiency_loss": 0.4},
                {"sol": 20, "type": "supply_contamination", "resource": "food", "loss": 0.2}
            ],
            success_criteria={
                "survival_rate": 0.95,
                "resource_conservation": 0.9,
                "crew_health": 0.85,
                "rationing_efficiency": 0.8
            },
            difficulty="hard"
        )
    
    def get_scenario(self, name: str) -> BenchmarkScenario:
        """Get a specific benchmark scenario."""
        if name not in self.scenarios:
            raise BenchmarkError(f"Unknown scenario: {name}")
        return self.scenarios[name]
    
    def list_scenarios(self) -> List[str]:
        """List available benchmark scenarios."""
        return list(self.scenarios.keys())
    
    def get_scenarios_by_difficulty(self, difficulty: str) -> List[BenchmarkScenario]:
        """Get all scenarios of a specific difficulty level."""
        return [scenario for scenario in self.scenarios.values() 
                if scenario.difficulty == difficulty]
    
    def get_nasa_validated_scenarios(self) -> List[BenchmarkScenario]:
        """Get all NASA-validated scenarios."""
        return [scenario for scenario in self.scenarios.values() 
                if scenario.nasa_validated]
    
    def add_custom_scenario(self, scenario: BenchmarkScenario):
        """Add a custom benchmark scenario."""
        self.scenarios[scenario.name] = scenario
        logger.info(f"Added custom scenario: {scenario.name}")
    
    def export_scenarios(self, filepath: Path):
        """Export all scenarios to JSON file."""
        scenarios_data = {name: scenario.to_dict() 
                         for name, scenario in self.scenarios.items()}
        
        with open(filepath, 'w') as f:
            json.dump(scenarios_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self.scenarios)} scenarios to {filepath}")


class StatisticalAnalyzer:
    """Statistical analysis tools for evaluation results."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compute_confidence_interval(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Compute mean and confidence interval for data."""
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        
        # Use t-distribution for small samples
        if len(data) < 30:
            t_val = stats.t.ppf(1 - self.alpha/2, len(data) - 1)
            margin = t_val * sem
        else:
            z_val = stats.norm.ppf(1 - self.alpha/2)
            margin = z_val * sem
        
        return mean, mean - margin, mean + margin
    
    def compare_algorithms(
        self, 
        results1: List[float], 
        results2: List[float],
        test_type: str = "welch"
    ) -> Dict[str, float]:
        """Compare two algorithms using statistical tests."""
        
        # Normality tests
        _, p1_shapiro = stats.shapiro(results1)
        _, p2_shapiro = stats.shapiro(results2)
        normal1 = p1_shapiro > 0.05
        normal2 = p2_shapiro > 0.05
        
        if normal1 and normal2:
            # Use t-test for normal data
            if test_type == "welch":
                statistic, p_value = stats.ttest_ind(results1, results2, equal_var=False)
                test_name = "Welch's t-test"
            else:
                statistic, p_value = stats.ttest_ind(results1, results2, equal_var=True)
                test_name = "Student's t-test"
        else:
            # Use Mann-Whitney U test for non-normal data
            statistic, p_value = stats.mannwhitneyu(results1, results2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(results1) - 1) * np.var(results1, ddof=1) + 
                             (len(results2) - 1) * np.var(results2, ddof=1)) / 
                            (len(results1) + len(results2) - 2))
        cohens_d = (np.mean(results1) - np.mean(results2)) / pooled_std
        
        return {
            "test_name": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": cohens_d,
            "effect_size_interpretation": self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_comparison_correction(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons."""
        return [min(p * len(p_values), 1.0) for p in p_values]
    
    def compute_performance_profiles(
        self, 
        results_dict: Dict[str, List[float]],
        tau_values: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute performance profiles for algorithm comparison.
        
        Based on Dolan & Moré (2002) benchmarking methodology.
        """
        if tau_values is None:
            tau_values = np.logspace(0, 2, 100)  # [1, 100]
        
        algorithms = list(results_dict.keys())
        n_problems = len(next(iter(results_dict.values())))
        
        # Convert to performance matrix (problems x algorithms)
        performance_matrix = np.array([results_dict[alg] for alg in algorithms]).T
        
        # Compute performance ratios
        best_performance = np.max(performance_matrix, axis=1, keepdims=True)
        ratios = performance_matrix / best_performance
        
        # Compute performance profiles
        profiles = {}
        for i, algorithm in enumerate(algorithms):
            profile = np.zeros_like(tau_values)
            for j, tau in enumerate(tau_values):
                profile[j] = np.sum(ratios[:, i] >= (1.0 / tau)) / n_problems
            profiles[algorithm] = profile
        
        return profiles, tau_values


class EvaluationEngine:
    """
    Comprehensive evaluation engine for lunar habitat RL agents.
    
    Provides research-quality evaluation with statistical analysis and reporting.
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        validate_evaluation_config(self.config)
        
        self.benchmark_suite = BenchmarkSuite()
        self.statistical_analyzer = StatisticalAnalyzer(self.config.confidence_level)
        self.security_manager = SecurityManager()
        
        # Results storage
        self.results: List[EvaluationResult] = []
        self.comparison_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Setup random seeds for reproducibility
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        
        logger.info("Initialized EvaluationEngine")
        logger.info(f"Available scenarios: {self.benchmark_suite.list_scenarios()}")
    
    def evaluate_agent(
        self,
        agent: Any,
        scenario_name: str,
        n_runs: Optional[int] = None,
        save_trajectories: bool = None
    ) -> List[EvaluationResult]:
        """
        Evaluate an agent on a specific scenario.
        
        Args:
            agent: RL agent to evaluate
            scenario_name: Name of benchmark scenario
            n_runs: Number of evaluation runs (default from config)
            save_trajectories: Whether to save trajectory data
            
        Returns:
            List of evaluation results
        """
        n_runs = n_runs or self.config.n_runs_per_scenario
        save_trajectories = save_trajectories or self.config.save_trajectories
        
        scenario = self.benchmark_suite.get_scenario(scenario_name)
        agent_name = getattr(agent, "__class__", type(agent)).__name__
        
        logger.info(f"Evaluating {agent_name} on {scenario_name} for {n_runs} runs")
        
        # Create environment for this scenario
        env = self._create_scenario_environment(scenario)
        
        results = []
        
        # Parallel evaluation for efficiency
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for run_id in range(n_runs):
                    future = executor.submit(
                        self._evaluate_single_run,
                        agent, env, scenario, run_id, agent_name, save_trajectories
                    )
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result(timeout=self.config.timeout_per_run)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Evaluation run failed: {str(e)}")
                        continue
        else:
            # Sequential evaluation
            for run_id in range(n_runs):
                try:
                    result = self._evaluate_single_run(
                        agent, env, scenario, run_id, agent_name, save_trajectories
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation run {run_id} failed: {str(e)}")
                    continue
        
        # Store results
        self.results.extend(results)
        for result in results:
            for metric_name, metric_value in result.metrics.items():
                self.comparison_data[scenario_name][f"{agent_name}_{metric_name}"].append(metric_value)
        
        logger.info(f"Completed evaluation: {len(results)}/{n_runs} runs successful")
        
        return results
    
    def _create_scenario_environment(self, scenario: BenchmarkScenario) -> Any:
        """Create environment configured for specific scenario."""
        # Import here to avoid circular dependencies
        from .environments import make_lunar_env
        
        env_config = {
            "crew_size": scenario.crew_size,
            "scenario": scenario.name,
            "duration_sols": scenario.duration_sols,
            "initial_conditions": scenario.initial_conditions,
            "events": scenario.events
        }
        
        return make_lunar_env(**env_config)
    
    def _evaluate_single_run(
        self,
        agent: Any,
        env: Any,
        scenario: BenchmarkScenario,
        run_id: int,
        agent_name: str,
        save_trajectories: bool
    ) -> EvaluationResult:
        """Evaluate agent on single run of scenario."""
        start_time = time.time()
        
        # Reset environment with scenario-specific seed
        seed = self.config.random_seed + run_id
        obs, info = env.reset(seed=seed)
        
        # Initialize tracking
        total_reward = 0.0
        survival_time_sols = 0.0
        safety_violations = 0
        trajectory_data = defaultdict(list) if save_trajectories else None
        
        # Performance metrics
        metrics = {
            "power_efficiency": 0.0,
            "resource_utilization": 0.0,
            "crew_health_avg": 0.0,
            "system_stability": 0.0,
            "response_time_avg": 0.0
        }
        
        step_count = 0
        done = False
        
        try:
            while not done:
                # Agent action selection
                action_start = time.time()
                
                if hasattr(agent, 'predict'):
                    # BaseAgent interface
                    action = agent.predict(obs, deterministic=True)
                elif hasattr(agent, 'act'):
                    # PyTorch agent interface  
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = agent.act(obs_tensor).cpu().numpy().flatten()
                else:
                    raise EvaluationError(f"Agent {agent_name} has no predict() or act() method")
                
                action_time = time.time() - action_start
                
                # Environment step
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Update tracking
                total_reward += reward
                survival_time_sols = step_info.get('mission_time_sols', step_count / 1000.0)
                
                if step_info.get('safety_violation', False):
                    safety_violations += 1
                
                # Update metrics
                metrics["power_efficiency"] += step_info.get('power_efficiency', 0.0)
                metrics["resource_utilization"] += step_info.get('resource_utilization', 0.0)
                metrics["crew_health_avg"] += step_info.get('crew_health_avg', 1.0)
                metrics["system_stability"] += step_info.get('system_stability', 1.0)
                metrics["response_time_avg"] += action_time
                
                # Save trajectory data
                if save_trajectories:
                    trajectory_data["observations"].append(obs.copy())
                    trajectory_data["actions"].append(action.copy())
                    trajectory_data["rewards"].append(reward)
                    trajectory_data["info"].append(step_info.copy())
                
                obs = next_obs
                step_count += 1
                
                # Prevent infinite loops
                if step_count > 100000:  # Sanity check
                    logger.warning(f"Run {run_id} exceeded maximum steps, terminating")
                    break
        
        except Exception as e:
            logger.error(f"Error during evaluation run {run_id}: {str(e)}")
            done = True
            survival_time_sols = step_count / 1000.0  # Approximate
        
        finally:
            env.close()
        
        # Normalize metrics
        if step_count > 0:
            for key in metrics:
                if key != "response_time_avg":
                    metrics[key] /= step_count
                else:
                    metrics[key] /= step_count  # Average response time
        
        # Determine success based on scenario criteria
        success = self._evaluate_success(scenario, metrics, survival_time_sols, safety_violations)
        
        # Convert trajectory data to numpy arrays
        if save_trajectories and trajectory_data:
            trajectory_data = {
                key: np.array(values) for key, values in trajectory_data.items()
            }
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            scenario_name=scenario.name,
            agent_name=agent_name,
            run_id=run_id,
            success=success,
            survival_time_sols=survival_time_sols,
            total_reward=total_reward,
            metrics=metrics,
            safety_violations=safety_violations,
            trajectory_data=trajectory_data,
            execution_time=execution_time
        )
    
    def _evaluate_success(
        self,
        scenario: BenchmarkScenario,
        metrics: Dict[str, float],
        survival_time: float,
        safety_violations: int
    ) -> bool:
        """Evaluate if run was successful based on scenario criteria."""
        
        # Check survival time
        required_survival = scenario.duration_sols * scenario.success_criteria.get("survival_rate", 1.0)
        if survival_time < required_survival:
            return False
        
        # Check safety violations
        max_violations = scenario.success_criteria.get("max_safety_violations", 0)
        if safety_violations > max_violations:
            return False
        
        # Check other success criteria
        for criterion, threshold in scenario.success_criteria.items():
            if criterion in ["survival_rate", "max_safety_violations"]:
                continue  # Already checked
            
            if criterion in metrics:
                if metrics[criterion] < threshold:
                    return False
        
        return True
    
    def compare_algorithms(
        self,
        algorithms: List[Any],
        scenarios: List[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms across scenarios and metrics.
        
        Returns comprehensive comparison results with statistical analysis.
        """
        scenarios = scenarios or self.benchmark_suite.list_scenarios()
        metrics = metrics or ["total_reward", "power_efficiency", "crew_health_avg"]
        
        logger.info(f"Comparing {len(algorithms)} algorithms on {len(scenarios)} scenarios")
        
        comparison_results = {
            "algorithms": [getattr(alg, "__class__", type(alg)).__name__ for alg in algorithms],
            "scenarios": scenarios,
            "metrics": metrics,
            "statistical_tests": {},
            "performance_profiles": {},
            "rankings": {},
            "summary_table": None
        }
        
        # Evaluate all algorithms if not already done
        for algorithm in algorithms:
            alg_name = getattr(algorithm, "__class__", type(algorithm)).__name__
            
            for scenario in scenarios:
                # Check if results already exist
                existing_results = [r for r in self.results 
                                  if r.agent_name == alg_name and r.scenario_name == scenario]
                
                if len(existing_results) < self.config.n_runs_per_scenario:
                    logger.info(f"Evaluating {alg_name} on {scenario}")
                    self.evaluate_agent(algorithm, scenario)
        
        # Perform statistical comparisons
        for scenario in scenarios:
            comparison_results["statistical_tests"][scenario] = {}
            
            for metric in metrics:
                results_by_algorithm = {}
                
                for algorithm in algorithms:
                    alg_name = getattr(algorithm, "__class__", type(algorithm)).__name__
                    
                    # Get results for this algorithm, scenario, and metric
                    alg_results = [r for r in self.results 
                                 if r.agent_name == alg_name and r.scenario_name == scenario]
                    
                    if metric == "total_reward":
                        metric_values = [r.total_reward for r in alg_results]
                    elif metric in alg_results[0].metrics if alg_results else {}:
                        metric_values = [r.metrics[metric] for r in alg_results]
                    else:
                        continue
                    
                    results_by_algorithm[alg_name] = metric_values
                
                # Pairwise statistical tests
                test_results = {}
                algorithm_names = list(results_by_algorithm.keys())
                
                for i in range(len(algorithm_names)):
                    for j in range(i + 1, len(algorithm_names)):
                        alg1, alg2 = algorithm_names[i], algorithm_names[j]
                        
                        test_result = self.statistical_analyzer.compare_algorithms(
                            results_by_algorithm[alg1],
                            results_by_algorithm[alg2]
                        )
                        
                        test_results[f"{alg1}_vs_{alg2}"] = test_result
                
                comparison_results["statistical_tests"][scenario][metric] = test_results
        
        # Compute performance profiles
        for metric in metrics:
            metric_data = {}
            
            for algorithm in algorithms:
                alg_name = getattr(algorithm, "__class__", type(algorithm)).__name__
                metric_values = []
                
                for scenario in scenarios:
                    alg_results = [r for r in self.results 
                                 if r.agent_name == alg_name and r.scenario_name == scenario]
                    
                    if metric == "total_reward":
                        scenario_values = [r.total_reward for r in alg_results]
                    elif alg_results and metric in alg_results[0].metrics:
                        scenario_values = [r.metrics[metric] for r in alg_results]
                    else:
                        scenario_values = [0.0] * self.config.n_runs_per_scenario
                    
                    metric_values.append(np.mean(scenario_values))
                
                metric_data[alg_name] = metric_values
            
            profiles, tau_values = self.statistical_analyzer.compute_performance_profiles(metric_data)
            comparison_results["performance_profiles"][metric] = {
                "profiles": profiles,
                "tau_values": tau_values.tolist()
            }
        
        # Generate summary table
        comparison_results["summary_table"] = self._generate_summary_table(
            algorithms, scenarios, metrics
        )
        
        return comparison_results
    
    def _generate_summary_table(
        self,
        algorithms: List[Any],
        scenarios: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """Generate summary table with means and confidence intervals."""
        
        rows = []
        
        for algorithm in algorithms:
            alg_name = getattr(algorithm, "__class__", type(algorithm)).__name__
            
            for scenario in scenarios:
                alg_results = [r for r in self.results 
                             if r.agent_name == alg_name and r.scenario_name == scenario]
                
                row = {"Algorithm": alg_name, "Scenario": scenario}
                
                for metric in metrics:
                    if metric == "total_reward":
                        values = [r.total_reward for r in alg_results]
                    elif alg_results and metric in alg_results[0].metrics:
                        values = [r.metrics[metric] for r in alg_results]
                    else:
                        values = []
                    
                    if values:
                        mean, ci_lower, ci_upper = self.statistical_analyzer.compute_confidence_interval(
                            np.array(values)
                        )
                        row[f"{metric}_mean"] = mean
                        row[f"{metric}_ci_lower"] = ci_lower
                        row[f"{metric}_ci_upper"] = ci_upper
                    else:
                        row[f"{metric}_mean"] = np.nan
                        row[f"{metric}_ci_lower"] = np.nan
                        row[f"{metric}_ci_upper"] = np.nan
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_report(self, output_path: Path = None) -> Path:
        """Generate comprehensive evaluation report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.results_dir / f"evaluation_report_{timestamp}.html"
        
        # Create visualizations if requested
        if self.config.generate_plots:
            self._generate_visualizations()
        
        # Generate HTML report
        html_content = self._create_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated evaluation report: {output_path}")
        
        return output_path
    
    def _generate_visualizations(self):
        """Generate evaluation plots and figures."""
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Performance comparison plots
        self._plot_performance_comparison()
        
        # Statistical significance heatmaps
        self._plot_significance_heatmaps()
        
        # Performance profiles
        self._plot_performance_profiles()
    
    def _plot_performance_comparison(self):
        """Create performance comparison box plots."""
        # Group results by scenario and algorithm
        plot_data = []
        
        for result in self.results:
            plot_data.append({
                "Algorithm": result.agent_name,
                "Scenario": result.scenario_name,
                "Total Reward": result.total_reward,
                "Power Efficiency": result.metrics.get("power_efficiency", 0),
                "Crew Health": result.metrics.get("crew_health_avg", 0),
                "Safety Violations": result.safety_violations
            })
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            return
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Algorithm Performance Comparison", fontsize=16)
        
        # Total Reward
        sns.boxplot(data=df, x="Scenario", y="Total Reward", hue="Algorithm", ax=axes[0, 0])
        axes[0, 0].set_title("Total Reward by Scenario")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Power Efficiency
        sns.boxplot(data=df, x="Scenario", y="Power Efficiency", hue="Algorithm", ax=axes[0, 1])
        axes[0, 1].set_title("Power Efficiency by Scenario")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Crew Health
        sns.boxplot(data=df, x="Scenario", y="Crew Health", hue="Algorithm", ax=axes[1, 0])
        axes[1, 0].set_title("Crew Health by Scenario")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Safety Violations
        sns.boxplot(data=df, x="Scenario", y="Safety Violations", hue="Algorithm", ax=axes[1, 1])
        axes[1, 1].set_title("Safety Violations by Scenario")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_path = self.config.results_dir / f"performance_comparison.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_significance_heatmaps(self):
        """Create statistical significance heatmaps."""
        # This would create heatmaps showing p-values between algorithm pairs
        # Implementation would depend on having comparison results
        pass
    
    def _plot_performance_profiles(self):
        """Create performance profile plots."""
        # This would create performance profile plots as per Dolan & Moré (2002)
        # Implementation would use the performance profiles computed earlier
        pass
    
    def _create_html_report(self) -> str:
        """Create HTML report content."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lunar Habitat RL Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-value { font-weight: bold; }
                .success { color: #27ae60; }
                .failure { color: #e74c3c; }
            </style>
        </head>
        <body>
            <h1>Lunar Habitat RL Evaluation Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Total Evaluations:</strong> {total_evaluations}</p>
            
            <h2>Benchmark Scenarios</h2>
            {scenarios_table}
            
            <h2>Results Summary</h2>
            {results_summary}
            
            <h2>Statistical Analysis</h2>
            {statistical_analysis}
            
        </body>
        </html>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_evaluations=len(self.results),
            scenarios_table=self._format_scenarios_table(),
            results_summary=self._format_results_summary(),
            statistical_analysis=self._format_statistical_analysis()
        )
        
        return html
    
    def _format_scenarios_table(self) -> str:
        """Format benchmark scenarios table."""
        scenarios_html = "<table><tr><th>Scenario</th><th>Description</th><th>Duration</th><th>Difficulty</th></tr>"
        
        for scenario in self.benchmark_suite.scenarios.values():
            scenarios_html += f"""
            <tr>
                <td>{scenario.name}</td>
                <td>{scenario.description}</td>
                <td>{scenario.duration_sols} sols</td>
                <td>{scenario.difficulty}</td>
            </tr>
            """
        
        scenarios_html += "</table>"
        return scenarios_html
    
    def _format_results_summary(self) -> str:
        """Format results summary table."""
        if not self.results:
            return "<p>No evaluation results available.</p>"
        
        # Group results by algorithm and scenario
        summary_data = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            summary_data[result.agent_name][result.scenario_name].append(result)
        
        html = "<table><tr><th>Algorithm</th><th>Scenario</th><th>Success Rate</th><th>Avg Reward</th><th>Avg Survival Time</th></tr>"
        
        for agent_name, scenarios in summary_data.items():
            for scenario_name, results in scenarios.items():
                success_rate = sum(1 for r in results if r.success) / len(results)
                avg_reward = np.mean([r.total_reward for r in results])
                avg_survival = np.mean([r.survival_time_sols for r in results])
                
                success_class = "success" if success_rate >= 0.8 else "failure"
                
                html += f"""
                <tr>
                    <td>{agent_name}</td>
                    <td>{scenario_name}</td>
                    <td class="{success_class}">{success_rate:.2%}</td>
                    <td class="metric-value">{avg_reward:.2f}</td>
                    <td class="metric-value">{avg_survival:.1f} sols</td>
                </tr>
                """
        
        html += "</table>"
        return html
    
    def _format_statistical_analysis(self) -> str:
        """Format statistical analysis section."""
        return "<p>Statistical analysis results would be displayed here based on comparison data.</p>"
    
    def export_results(self, filepath: Path = None) -> Path:
        """Export all results to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.config.results_dir / f"evaluation_results_{timestamp}.json"
        
        # Convert results to serializable format
        export_data = {
            "config": self.config.__dict__,
            "scenarios": {name: scenario.to_dict() 
                         for name, scenario in self.benchmark_suite.scenarios.items()},
            "results": [result.to_dict() for result in self.results],
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Handle non-serializable types
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported evaluation results to {filepath}")
        
        return filepath


# Export main classes
__all__ = [
    "BenchmarkScenario",
    "EvaluationResult", 
    "EvaluationConfig",
    "BenchmarkSuite",
    "StatisticalAnalyzer",
    "EvaluationEngine"
]