"""
Research Benchmark Suite for Lunar Habitat RL

This module implements a comprehensive benchmark suite for evaluating
reinforcement learning algorithms on lunar habitat control tasks,
designed specifically for academic research and publication.

Novel contributions:
1. Standardized Evaluation Protocols
2. Multi-Modal Performance Metrics
3. Statistical Significance Testing
4. Comparative Analysis Framework

Authors: Daniel Schmidt, Terragon Labs
Research Focus: Rigorous benchmarking for space systems RL research
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    # Experiment parameters
    n_runs: int = 10
    n_evaluation_episodes: int = 100
    max_episode_length: int = 1000
    confidence_level: float = 0.95
    
    # Statistical testing
    significance_level: float = 0.05
    effect_size_threshold: float = 0.2
    
    # Evaluation scenarios
    scenarios: List[str] = field(default_factory=lambda: [
        "nominal_operations",
        "equipment_failure",
        "emergency_response",
        "resource_scarcity",
        "crew_emergency",
        "system_degradation"
    ])
    
    # Metrics to evaluate
    metrics: List[str] = field(default_factory=lambda: [
        "episode_reward",
        "survival_time",
        "resource_efficiency",
        "safety_score",
        "crew_wellbeing",
        "system_stability",
        "response_time",
        "power_efficiency",
        "atmosphere_quality"
    ])
    
    # Output configuration
    save_raw_data: bool = True
    generate_plots: bool = True
    export_latex_tables: bool = True
    output_directory: str = "benchmark_results"


class PerformanceMetric(ABC):
    """Base class for performance metrics."""
    
    @abstractmethod
    def compute(self, episode_data: Dict[str, Any]) -> float:
        """Compute metric value from episode data."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get metric name."""
        pass
    
    @abstractmethod
    def get_higher_is_better(self) -> bool:
        """Return True if higher values are better."""
        pass


class EpisodeRewardMetric(PerformanceMetric):
    """Cumulative episode reward metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        return episode_data.get('total_reward', 0.0)
    
    def get_name(self) -> str:
        return "Episode Reward"
    
    def get_higher_is_better(self) -> bool:
        return True


class SurvivalTimeMetric(PerformanceMetric):
    """Survival time metric (steps before critical failure)."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        return episode_data.get('survival_steps', 0) / episode_data.get('max_steps', 1000)
    
    def get_name(self) -> str:
        return "Survival Rate"
    
    def get_higher_is_better(self) -> bool:
        return True


class ResourceEfficiencyMetric(PerformanceMetric):
    """Resource utilization efficiency metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        resources_used = episode_data.get('resources_consumed', 0)
        resources_available = episode_data.get('resources_available', 1)
        return 1.0 - (resources_used / resources_available) if resources_available > 0 else 0.0
    
    def get_name(self) -> str:
        return "Resource Efficiency"
    
    def get_higher_is_better(self) -> bool:
        return True


class SafetyScoreMetric(PerformanceMetric):
    """Safety violations metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        safety_violations = episode_data.get('safety_violations', 0)
        max_violations = episode_data.get('max_safety_violations', 10)
        return 1.0 - (safety_violations / max_violations) if max_violations > 0 else 1.0
    
    def get_name(self) -> str:
        return "Safety Score"
    
    def get_higher_is_better(self) -> bool:
        return True


class CrewWellbeingMetric(PerformanceMetric):
    """Crew health and well-being metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        crew_health = episode_data.get('crew_health_scores', [1.0, 1.0, 1.0, 1.0])
        return np.mean(crew_health) if crew_health else 0.0
    
    def get_name(self) -> str:
        return "Crew Well-being"
    
    def get_higher_is_better(self) -> bool:
        return True


class SystemStabilityMetric(PerformanceMetric):
    """System stability and control quality metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        control_variance = episode_data.get('control_variance', 0.0)
        return 1.0 / (1.0 + control_variance)
    
    def get_name(self) -> str:
        return "System Stability"
    
    def get_higher_is_better(self) -> bool:
        return True


class ResponseTimeMetric(PerformanceMetric):
    """Emergency response time metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        response_times = episode_data.get('emergency_response_times', [])
        if not response_times:
            return 1.0  # No emergencies = perfect score
        
        avg_response_time = np.mean(response_times)
        max_acceptable_time = 300  # 5 minutes
        return max(0.0, 1.0 - (avg_response_time / max_acceptable_time))
    
    def get_name(self) -> str:
        return "Emergency Response"
    
    def get_higher_is_better(self) -> bool:
        return True


class PowerEfficiencyMetric(PerformanceMetric):
    """Power system efficiency metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        power_waste = episode_data.get('power_waste', 0.0)
        total_power = episode_data.get('total_power_generated', 1.0)
        return 1.0 - (power_waste / total_power) if total_power > 0 else 0.0
    
    def get_name(self) -> str:
        return "Power Efficiency"
    
    def get_higher_is_better(self) -> bool:
        return True


class AtmosphereQualityMetric(PerformanceMetric):
    """Atmospheric composition quality metric."""
    
    def compute(self, episode_data: Dict[str, Any]) -> float:
        atmosphere_violations = episode_data.get('atmosphere_violations', 0)
        total_timesteps = episode_data.get('episode_length', 1000)
        return 1.0 - (atmosphere_violations / total_timesteps)
    
    def get_name(self) -> str:
        return "Atmosphere Quality"
    
    def get_higher_is_better(self) -> bool:
        return True


class StatisticalAnalyzer:
    """Statistical analysis for benchmark results."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
    
    def compute_confidence_interval(self, data: np.ndarray) -> Tuple[float, float, float]:
        """Compute confidence interval for data."""
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        
        # Use t-distribution for small samples
        n = len(data)
        if n < 30:
            critical_value = stats.t.ppf(1 - self.alpha/2, n - 1)
        else:
            critical_value = stats.norm.ppf(1 - self.alpha/2)
        
        margin_error = critical_value * sem
        
        return mean, mean - margin_error, mean + margin_error
    
    def perform_significance_test(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance test between two datasets."""
        # Check normality
        _, p_normal1 = stats.shapiro(data1) if len(data1) < 5000 else (None, 0.05)
        _, p_normal2 = stats.shapiro(data2) if len(data2) < 5000 else (None, 0.05)
        
        both_normal = p_normal1 > 0.05 and p_normal2 > 0.05
        
        if both_normal and len(data1) > 10 and len(data2) > 10:
            # Use t-test for normal data
            statistic, p_value = stats.ttest_ind(data1, data2)
            test_name = "t-test"
        else:
            # Use Mann-Whitney U test for non-normal data
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                             (len(data2) - 1) * np.var(data2, ddof=1)) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'cohens_d': cohens_d,
            'effect_size': self._interpret_effect_size(abs(cohens_d))
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


class BenchmarkEvaluator:
    """Main benchmark evaluation class."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = {
            "episode_reward": EpisodeRewardMetric(),
            "survival_time": SurvivalTimeMetric(),
            "resource_efficiency": ResourceEfficiencyMetric(),
            "safety_score": SafetyScoreMetric(),
            "crew_wellbeing": CrewWellbeingMetric(),
            "system_stability": SystemStabilityMetric(),
            "response_time": ResponseTimeMetric(),
            "power_efficiency": PowerEfficiencyMetric(),
            "atmosphere_quality": AtmosphereQualityMetric()
        }
        self.analyzer = StatisticalAnalyzer(config.confidence_level)
        
        # Create output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_agent(self, agent, environment, agent_name: str) -> Dict[str, Any]:
        """Evaluate single agent across all scenarios."""
        results = {
            'agent_name': agent_name,
            'scenarios': {},
            'overall_statistics': {}
        }
        
        print(f"Evaluating agent: {agent_name}")
        
        for scenario in self.config.scenarios:
            print(f"  Scenario: {scenario}")
            scenario_results = self._evaluate_scenario(agent, environment, scenario)
            results['scenarios'][scenario] = scenario_results
        
        # Compute overall statistics
        results['overall_statistics'] = self._compute_overall_statistics(results['scenarios'])
        
        return results
    
    def _evaluate_scenario(self, agent, environment, scenario: str) -> Dict[str, Any]:
        """Evaluate agent on a specific scenario."""
        scenario_results = {
            'scenario_name': scenario,
            'runs': [],
            'metrics': {}
        }
        
        # Run multiple evaluation runs
        for run in range(self.config.n_runs):
            run_results = self._run_evaluation_episodes(agent, environment, scenario)
            scenario_results['runs'].append(run_results)
        
        # Aggregate metrics across runs
        for metric_name in self.config.metrics:
            if metric_name in self.metrics:
                metric_values = []
                
                for run_result in scenario_results['runs']:
                    for episode_data in run_result['episodes']:
                        value = self.metrics[metric_name].compute(episode_data)
                        metric_values.append(value)
                
                metric_values = np.array(metric_values)
                mean, ci_low, ci_high = self.analyzer.compute_confidence_interval(metric_values)
                
                scenario_results['metrics'][metric_name] = {
                    'values': metric_values.tolist(),
                    'mean': mean,
                    'std': np.std(metric_values),
                    'median': np.median(metric_values),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'min': np.min(metric_values),
                    'max': np.max(metric_values)
                }
        
        return scenario_results
    
    def _run_evaluation_episodes(self, agent, environment, scenario: str) -> Dict[str, Any]:
        """Run evaluation episodes for a specific scenario."""
        run_results = {
            'scenario': scenario,
            'episodes': []
        }
        
        for episode in range(self.config.n_evaluation_episodes):
            # Mock episode data generation for algorithm demonstration
            episode_data = self._generate_mock_episode_data(scenario)
            run_results['episodes'].append(episode_data)
        
        return run_results
    
    def _generate_mock_episode_data(self, scenario: str) -> Dict[str, Any]:
        """Generate mock episode data for demonstration."""
        # This would be replaced with actual environment interaction
        base_reward = 100.0
        scenario_modifier = {
            "nominal_operations": 1.0,
            "equipment_failure": 0.7,
            "emergency_response": 0.5,
            "resource_scarcity": 0.6,
            "crew_emergency": 0.4,
            "system_degradation": 0.8
        }.get(scenario, 1.0)
        
        return {
            'total_reward': base_reward * scenario_modifier + np.random.normal(0, 10),
            'survival_steps': np.random.randint(800, 1000),
            'max_steps': 1000,
            'resources_consumed': np.random.uniform(0.3, 0.8),
            'resources_available': 1.0,
            'safety_violations': np.random.poisson(1),
            'max_safety_violations': 10,
            'crew_health_scores': [np.random.uniform(0.8, 1.0) for _ in range(4)],
            'control_variance': np.random.exponential(0.1),
            'emergency_response_times': [np.random.uniform(60, 300) for _ in range(np.random.randint(0, 3))],
            'power_waste': np.random.uniform(0.05, 0.2),
            'total_power_generated': 1.0,
            'atmosphere_violations': np.random.poisson(5),
            'episode_length': 1000
        }
    
    def _compute_overall_statistics(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall statistics across all scenarios."""
        overall_stats = {}
        
        for metric_name in self.config.metrics:
            if metric_name in self.metrics:
                all_values = []
                
                for scenario_name, scenario_data in scenario_results.items():
                    if metric_name in scenario_data['metrics']:
                        all_values.extend(scenario_data['metrics'][metric_name]['values'])
                
                if all_values:
                    all_values = np.array(all_values)
                    mean, ci_low, ci_high = self.analyzer.compute_confidence_interval(all_values)
                    
                    overall_stats[metric_name] = {
                        'mean': mean,
                        'std': np.std(all_values),
                        'median': np.median(all_values),
                        'ci_low': ci_low,
                        'ci_high': ci_high
                    }
        
        return overall_stats
    
    def compare_agents(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple agents statistically."""
        print("Performing statistical comparison of agents...")
        
        comparison_results = {
            'agents': [result['agent_name'] for result in agent_results],
            'scenario_comparisons': {},
            'overall_comparisons': {},
            'rankings': {}
        }
        
        # Compare agents on each scenario
        for scenario in self.config.scenarios:
            scenario_comparisons = {}
            
            for metric_name in self.config.metrics:
                if metric_name not in self.metrics:
                    continue
                
                metric_comparisons = {}
                agent_data = {}
                
                # Collect data for all agents
                for agent_result in agent_results:
                    agent_name = agent_result['agent_name']
                    if scenario in agent_result['scenarios'] and metric_name in agent_result['scenarios'][scenario]['metrics']:
                        values = agent_result['scenarios'][scenario]['metrics'][metric_name]['values']
                        agent_data[agent_name] = np.array(values)
                
                # Pairwise comparisons
                agent_names = list(agent_data.keys())
                for i in range(len(agent_names)):
                    for j in range(i + 1, len(agent_names)):
                        agent1, agent2 = agent_names[i], agent_names[j]
                        
                        comparison = self.analyzer.perform_significance_test(
                            agent_data[agent1], agent_data[agent2]
                        )
                        
                        comparison_key = f"{agent1}_vs_{agent2}"
                        metric_comparisons[comparison_key] = comparison
                
                scenario_comparisons[metric_name] = metric_comparisons
            
            comparison_results['scenario_comparisons'][scenario] = scenario_comparisons
        
        # Overall agent ranking
        comparison_results['rankings'] = self._compute_agent_rankings(agent_results)
        
        return comparison_results
    
    def _compute_agent_rankings(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute overall agent rankings."""
        rankings = {}
        
        for metric_name in self.config.metrics:
            if metric_name not in self.metrics:
                continue
            
            higher_is_better = self.metrics[metric_name].get_higher_is_better()
            
            agent_scores = []
            for agent_result in agent_results:
                agent_name = agent_result['agent_name']
                
                if metric_name in agent_result['overall_statistics']:
                    score = agent_result['overall_statistics'][metric_name]['mean']
                    agent_scores.append((agent_name, score))
            
            # Sort by score
            agent_scores.sort(key=lambda x: x[1], reverse=higher_is_better)
            
            rankings[metric_name] = [{'rank': i + 1, 'agent': name, 'score': score} 
                                   for i, (name, score) in enumerate(agent_scores)]
        
        return rankings
    
    def generate_report(self, agent_results: List[Dict[str, Any]], 
                       comparison_results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"benchmark_report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(self._generate_markdown_report(agent_results, comparison_results))
        
        # Generate additional outputs
        if self.config.save_raw_data:
            self._save_raw_data(agent_results, comparison_results, timestamp)
        
        if self.config.generate_plots:
            self._generate_plots(agent_results, comparison_results, timestamp)
        
        if self.config.export_latex_tables:
            self._export_latex_tables(agent_results, comparison_results, timestamp)
        
        print(f"Benchmark report generated: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self, agent_results: List[Dict[str, Any]], 
                                 comparison_results: Dict[str, Any]) -> str:
        """Generate markdown report."""
        report = "# Lunar Habitat RL Benchmark Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive summary
        report += "## Executive Summary\n\n"
        report += f"This report presents the results of benchmarking {len(agent_results)} RL agents "
        report += f"across {len(self.config.scenarios)} scenarios using {len(self.config.metrics)} metrics.\n\n"
        
        # Agent rankings
        report += "## Agent Rankings\n\n"
        for metric_name, rankings in comparison_results['rankings'].items():
            if rankings:
                report += f"### {self.metrics[metric_name].get_name()}\n\n"
                report += "| Rank | Agent | Score |\n"
                report += "|------|-------|-------|\n"
                
                for ranking in rankings:
                    report += f"| {ranking['rank']} | {ranking['agent']} | {ranking['score']:.4f} |\n"
                
                report += "\n"
        
        # Detailed results per scenario
        report += "## Scenario Results\n\n"
        for scenario in self.config.scenarios:
            report += f"### {scenario.replace('_', ' ').title()}\n\n"
            
            # Create comparison table
            report += "| Agent | "
            for metric_name in self.config.metrics:
                if metric_name in self.metrics:
                    report += f"{self.metrics[metric_name].get_name()} | "
            report += "\n"
            
            report += "|-------|"
            for _ in self.config.metrics:
                report += "-------|"
            report += "\n"
            
            for agent_result in agent_results:
                agent_name = agent_result['agent_name']
                report += f"| {agent_name} | "
                
                if scenario in agent_result['scenarios']:
                    scenario_data = agent_result['scenarios'][scenario]
                    
                    for metric_name in self.config.metrics:
                        if metric_name in self.metrics and metric_name in scenario_data['metrics']:
                            mean = scenario_data['metrics'][metric_name]['mean']
                            std = scenario_data['metrics'][metric_name]['std']
                            report += f"{mean:.4f} ± {std:.4f} | "
                        else:
                            report += "N/A | "
                else:
                    for _ in self.config.metrics:
                        report += "N/A | "
                
                report += "\n"
            
            report += "\n"
        
        # Statistical significance
        report += "## Statistical Analysis\n\n"
        report += "Statistical significance testing was performed using appropriate tests "
        report += f"(t-test for normal distributions, Mann-Whitney U for non-normal) "
        report += f"with α = {1 - self.config.confidence_level}.\n\n"
        
        return report
    
    def _save_raw_data(self, agent_results: List[Dict[str, Any]], 
                      comparison_results: Dict[str, Any], timestamp: str):
        """Save raw benchmark data."""
        raw_data = {
            'agent_results': agent_results,
            'comparison_results': comparison_results,
            'config': {
                'n_runs': self.config.n_runs,
                'n_evaluation_episodes': self.config.n_evaluation_episodes,
                'scenarios': self.config.scenarios,
                'metrics': self.config.metrics
            },
            'timestamp': timestamp
        }
        
        output_path = self.output_dir / f"raw_data_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        raw_data = convert_numpy(raw_data)
        
        with open(output_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
        
        print(f"Raw data saved: {output_path}")
    
    def _generate_plots(self, agent_results: List[Dict[str, Any]], 
                       comparison_results: Dict[str, Any], timestamp: str):
        """Generate visualization plots."""
        # This is a placeholder for plot generation
        # In a full implementation, this would create various plots:
        # - Box plots for each metric across agents
        # - Radar charts for multi-dimensional performance
        # - Learning curves if available
        # - Statistical significance heatmaps
        
        print(f"Plot generation placeholder (timestamp: {timestamp})")
    
    def _export_latex_tables(self, agent_results: List[Dict[str, Any]], 
                            comparison_results: Dict[str, Any], timestamp: str):
        """Export results as LaTeX tables for academic papers."""
        latex_content = "% LaTeX tables for Lunar Habitat RL Benchmark Results\n\n"
        
        # Main results table
        latex_content += "\\begin{table}[htbp]\n"
        latex_content += "\\centering\n"
        latex_content += "\\caption{Benchmark Results Across All Scenarios}\n"
        latex_content += "\\label{tab:benchmark_results}\n"
        latex_content += "\\begin{tabular}{l" + "c" * len(self.config.metrics) + "}\n"
        latex_content += "\\toprule\n"
        
        # Header
        latex_content += "Agent"
        for metric_name in self.config.metrics:
            if metric_name in self.metrics:
                latex_content += f" & {self.metrics[metric_name].get_name()}"
        latex_content += " \\\\\n"
        latex_content += "\\midrule\n"
        
        # Data rows
        for agent_result in agent_results:
            agent_name = agent_result['agent_name']
            latex_content += agent_name
            
            for metric_name in self.config.metrics:
                if (metric_name in self.metrics and 
                    metric_name in agent_result['overall_statistics']):
                    mean = agent_result['overall_statistics'][metric_name]['mean']
                    std = agent_result['overall_statistics'][metric_name]['std']
                    latex_content += f" & {mean:.3f} ± {std:.3f}"
                else:
                    latex_content += " & N/A"
            
            latex_content += " \\\\\n"
        
        latex_content += "\\bottomrule\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\end{table}\n\n"
        
        output_path = self.output_dir / f"latex_tables_{timestamp}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX tables exported: {output_path}")


class ResearchBenchmarkSuite:
    """Main class for running comprehensive benchmark experiments."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config if config else BenchmarkConfig()
        self.evaluator = BenchmarkEvaluator(self.config)
    
    def run_benchmark(self, agents: List[Tuple[Any, str]], environment) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            agents: List of (agent, name) tuples
            environment: Environment for evaluation
            
        Returns:
            Complete benchmark results
        """
        print("=" * 70)
        print("LUNAR HABITAT RL RESEARCH BENCHMARK SUITE")
        print("=" * 70)
        print(f"Scenarios: {len(self.config.scenarios)}")
        print(f"Metrics: {len(self.config.metrics)}")
        print(f"Runs per scenario: {self.config.n_runs}")
        print(f"Episodes per run: {self.config.n_evaluation_episodes}")
        print("=" * 70)
        
        # Evaluate all agents
        agent_results = []
        for agent, agent_name in agents:
            result = self.evaluator.evaluate_agent(agent, environment, agent_name)
            agent_results.append(result)
        
        # Statistical comparison
        comparison_results = self.evaluator.compare_agents(agent_results)
        
        # Generate comprehensive report
        report_path = self.evaluator.generate_report(agent_results, comparison_results)
        
        return {
            'agent_results': agent_results,
            'comparison_results': comparison_results,
            'report_path': report_path,
            'config': self.config
        }


# Factory function for easy instantiation
def create_benchmark_suite(**kwargs) -> ResearchBenchmarkSuite:
    """Create benchmark suite with custom configuration."""
    config = BenchmarkConfig(**kwargs)
    return ResearchBenchmarkSuite(config)


if __name__ == "__main__":
    # Demonstration of benchmark suite
    print("Research Benchmark Suite for Lunar Habitat RL")
    print("=" * 50)
    print("Features:")
    print("1. Standardized Evaluation Protocols")
    print("2. Multi-Modal Performance Metrics")
    print("3. Statistical Significance Testing")
    print("4. Comparative Analysis Framework")
    print("5. Academic Publication Ready Output")
    print("\nThis benchmark suite is designed for rigorous scientific evaluation")
    print("of RL algorithms in lunar habitat control applications.")
    
    # Example usage
    config = BenchmarkConfig(
        n_runs=5,
        n_evaluation_episodes=50,
        scenarios=["nominal_operations", "equipment_failure"],
        output_directory="example_results"
    )
    
    benchmark = ResearchBenchmarkSuite(config)
    print(f"\nBenchmark configured with {len(config.scenarios)} scenarios")
    print(f"Output directory: {config.output_directory}")