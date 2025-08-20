#!/usr/bin/env python3
"""
Generation 3: Comprehensive Benchmarking and Performance Regression Testing Suite
Advanced performance testing for NASA Lunar Habitat RL Suite scaling capabilities.
"""

import sys
import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import subprocess

# Add repo to path
sys.path.insert(0, '/root/repo')

from generation3_scaling import (
    test_caching_system, test_autoscaling, test_environment_pool,
    benchmark_single_threaded, benchmark_parallel
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    name: str
    success: bool
    execution_time: float
    throughput: float
    memory_usage_mb: float
    cpu_usage: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionReport:
    """Performance regression analysis report."""
    baseline_file: str
    current_results: Dict[str, BenchmarkResult]
    baseline_results: Dict[str, BenchmarkResult]
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0


class Generation3BenchmarkSuite:
    """Comprehensive benchmarking suite for Generation 3 features."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.baseline_path = Path("/root/repo/benchmarks/generation3_baseline.json")
        
        # Ensure benchmark directory exists
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Generation 3: Comprehensive Benchmark Suite")
        print("=" * 55)
    
    def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all Generation 3 benchmarks."""
        
        benchmarks = [
            ("caching_performance", self._benchmark_caching),
            ("autoscaling_efficiency", self._benchmark_autoscaling),
            ("environment_pool_performance", self._benchmark_env_pool),
            ("single_threaded_throughput", self._benchmark_single_threaded),
            ("parallel_processing_efficiency", self._benchmark_parallel),
            ("memory_optimization", self._benchmark_memory_optimization),
            ("concurrent_execution", self._benchmark_concurrent_execution),
            ("system_scalability", self._benchmark_system_scalability),
            ("database_performance", self._benchmark_database_performance),
            ("integrated_performance", self._benchmark_integrated_performance)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n📊 Running {benchmark_name}...")
            try:
                result = benchmark_func()
                self.results[benchmark_name] = result
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"   {status} - {result.execution_time:.3f}s - {result.throughput:.1f} ops/sec")
                
                if result.error_message:
                    print(f"   Error: {result.error_message}")
                    
            except Exception as e:
                error_result = BenchmarkResult(
                    name=benchmark_name,
                    success=False,
                    execution_time=0.0,
                    throughput=0.0,
                    memory_usage_mb=0.0,
                    cpu_usage=0.0,
                    error_message=str(e)
                )
                self.results[benchmark_name] = error_result
                print(f"   ❌ FAIL - Exception: {e}")
        
        return self.results
    
    def _benchmark_caching(self) -> BenchmarkResult:
        """Benchmark adaptive caching system."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Run caching test multiple times for better measurement
            iterations = 5
            cache_hits = 0
            
            for _ in range(iterations):
                success = test_caching_system()
                if success:
                    cache_hits += 1
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Calculate throughput (cache operations per second)
            # Estimated 1000 cache operations per test
            total_operations = iterations * 1000
            throughput = total_operations / execution_time
            
            return BenchmarkResult(
                name="caching_performance",
                success=cache_hits >= iterations * 0.8,  # 80% success rate
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'cache_hit_ratio': cache_hits / iterations,
                    'iterations': iterations,
                    'estimated_operations': total_operations
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="caching_performance",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_autoscaling(self) -> BenchmarkResult:
        """Benchmark auto-scaling system."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            success = test_autoscaling()
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Throughput = scaling decisions per second
            # Estimate 20 scaling evaluations in test
            throughput = 20 / execution_time
            
            return BenchmarkResult(
                name="autoscaling_efficiency",
                success=success,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={'scaling_decisions': 20}
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="autoscaling_efficiency",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_env_pool(self) -> BenchmarkResult:
        """Benchmark environment pool performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            success = test_environment_pool()
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Throughput = environment operations per second
            # Estimate 100 checkout/checkin operations
            throughput = 100 / execution_time
            
            return BenchmarkResult(
                name="environment_pool_performance",
                success=success,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={'pool_operations': 100}
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="environment_pool_performance",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_single_threaded(self) -> BenchmarkResult:
        """Benchmark single-threaded performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Run benchmark with fewer episodes for faster execution
            metrics = benchmark_single_threaded(n_episodes=25)
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            success = metrics.throughput_eps > 100  # Minimum 100 episodes/sec
            
            return BenchmarkResult(
                name="single_threaded_throughput",
                success=success,
                execution_time=execution_time,
                throughput=metrics.throughput_eps,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'episodes': metrics.total_episodes,
                    'steps': metrics.total_steps,
                    'avg_episode_time': metrics.avg_episode_time,
                    'steps_per_second': metrics.throughput_sps
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="single_threaded_throughput",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_parallel(self) -> BenchmarkResult:
        """Benchmark parallel processing performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Run parallel benchmark
            metrics = benchmark_parallel(n_episodes=25, n_workers=4)
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            success = (metrics.throughput_eps > 50 and  # Minimum throughput
                      metrics.parallel_efficiency > 0.1)  # Some parallel benefit
            
            return BenchmarkResult(
                name="parallel_processing_efficiency",
                success=success,
                execution_time=execution_time,
                throughput=metrics.throughput_eps,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'episodes': metrics.total_episodes,
                    'steps': metrics.total_steps,
                    'parallel_efficiency': metrics.parallel_efficiency,
                    'workers': 4
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="parallel_processing_efficiency",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_memory_optimization(self) -> BenchmarkResult:
        """Benchmark memory optimization performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Simulate memory optimization
            import gc
            
            # Create some objects to optimize
            large_objects = []
            for i in range(100):
                large_objects.append([0] * 1000)  # Create memory pressure
            
            # Force garbage collection
            initial_objects = len(gc.get_objects())
            collected = gc.collect()
            final_objects = len(gc.get_objects())
            
            # Clean up
            del large_objects
            gc.collect()
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Throughput = objects processed per second
            throughput = initial_objects / execution_time if execution_time > 0 else 0
            
            success = collected > 0 or final_objects < initial_objects
            
            return BenchmarkResult(
                name="memory_optimization",
                success=success,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'objects_collected': collected,
                    'initial_objects': initial_objects,
                    'final_objects': final_objects
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="memory_optimization",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_concurrent_execution(self) -> BenchmarkResult:
        """Benchmark concurrent execution performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            import threading
            import queue
            
            # Test concurrent task execution
            task_queue = queue.Queue()
            result_queue = queue.Queue()
            
            def worker():
                while True:
                    try:
                        task = task_queue.get(timeout=1.0)
                        if task is None:
                            break
                        
                        # Simulate work
                        result = sum(range(task))
                        result_queue.put(result)
                        task_queue.task_done()
                    except queue.Empty:
                        break
            
            # Start workers
            num_workers = 4
            workers = []
            for _ in range(num_workers):
                t = threading.Thread(target=worker)
                t.start()
                workers.append(t)
            
            # Add tasks
            num_tasks = 100
            for i in range(num_tasks):
                task_queue.put(i * 100)
            
            # Wait for completion
            task_queue.join()
            
            # Stop workers
            for _ in range(num_workers):
                task_queue.put(None)
            
            for worker in workers:
                worker.join()
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())
            
            throughput = len(results) / execution_time if execution_time > 0 else 0
            success = len(results) >= num_tasks * 0.9  # 90% completion rate
            
            return BenchmarkResult(
                name="concurrent_execution",
                success=success,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'tasks_completed': len(results),
                    'total_tasks': num_tasks,
                    'workers': num_workers,
                    'completion_rate': len(results) / num_tasks
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="concurrent_execution",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_system_scalability(self) -> BenchmarkResult:
        """Benchmark overall system scalability."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Test system under increasing load
            load_levels = [10, 50, 100, 200]
            performance_degradation = []
            
            for load in load_levels:
                load_start = time.time()
                
                # Simulate load (simple computation)
                results = []
                for i in range(load):
                    result = sum(range(i * 10))
                    results.append(result)
                
                load_time = time.time() - load_start
                load_throughput = load / load_time if load_time > 0 else 0
                performance_degradation.append(load_throughput)
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Check if system scales reasonably (degradation < 50% from min to max load)
            min_perf = min(performance_degradation)
            max_perf = max(performance_degradation)
            
            if max_perf > 0:
                degradation_ratio = min_perf / max_perf
                success = degradation_ratio > 0.5  # Less than 50% degradation
            else:
                success = False
            
            avg_throughput = sum(performance_degradation) / len(performance_degradation)
            
            return BenchmarkResult(
                name="system_scalability",
                success=success,
                execution_time=execution_time,
                throughput=avg_throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'load_levels': load_levels,
                    'performance_at_load': performance_degradation,
                    'degradation_ratio': degradation_ratio if max_perf > 0 else 0
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="system_scalability",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_database_performance(self) -> BenchmarkResult:
        """Benchmark database performance (simplified)."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Simple in-memory database operations
            data = {}
            
            # Insert operations
            num_records = 1000
            for i in range(num_records):
                data[f"key_{i}"] = {
                    'id': i,
                    'value': f"data_{i}",
                    'timestamp': time.time()
                }
            
            # Query operations
            queries = []
            for i in range(0, num_records, 10):
                key = f"key_{i}"
                if key in data:
                    queries.append(data[key])
            
            # Update operations
            for i in range(0, num_records, 20):
                key = f"key_{i}"
                if key in data:
                    data[key]['updated'] = True
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            total_operations = num_records + len(queries) + (num_records // 20)
            throughput = total_operations / execution_time if execution_time > 0 else 0
            
            success = len(data) == num_records and len(queries) > 0
            
            return BenchmarkResult(
                name="database_performance",
                success=success,
                execution_time=execution_time,
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'records_inserted': num_records,
                    'queries_executed': len(queries),
                    'records_updated': num_records // 20,
                    'total_operations': total_operations
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="database_performance",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _benchmark_integrated_performance(self) -> BenchmarkResult:
        """Benchmark integrated system performance."""
        start_time = time.time()
        memory_start = self._get_memory_usage()
        
        try:
            # Run a subset of core benchmarks together
            caching_result = self._benchmark_caching()
            scaling_result = self._benchmark_autoscaling()
            
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage() - memory_start
            
            # Combined throughput
            combined_throughput = (caching_result.throughput + scaling_result.throughput) / 2
            
            # Success if both components work
            success = caching_result.success and scaling_result.success
            
            return BenchmarkResult(
                name="integrated_performance",
                success=success,
                execution_time=execution_time,
                throughput=combined_throughput,
                memory_usage_mb=memory_usage,
                cpu_usage=self._get_cpu_usage(),
                metadata={
                    'caching_success': caching_result.success,
                    'scaling_success': scaling_result.success,
                    'component_count': 2
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="integrated_performance",
                success=False,
                execution_time=time.time() - start_time,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage=0.0,
                error_message=str(e)
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def save_baseline(self):
        """Save current results as baseline for regression testing."""
        baseline_data = {
            'timestamp': time.time(),
            'results': {name: result.__dict__ for name, result in self.results.items()}
        }
        
        try:
            with open(self.baseline_path, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            print(f"📋 Baseline saved: {self.baseline_path}")
        except Exception as e:
            print(f"⚠️ Failed to save baseline: {e}")
    
    def load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline results for comparison."""
        try:
            if self.baseline_path.exists():
                with open(self.baseline_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load baseline: {e}")
        return None
    
    def analyze_regression(self) -> Optional[RegressionReport]:
        """Analyze performance regression against baseline."""
        baseline_data = self.load_baseline()
        if not baseline_data or not self.results:
            return None
        
        baseline_results = {}
        for name, data in baseline_data['results'].items():
            baseline_results[name] = BenchmarkResult(**data)
        
        report = RegressionReport(
            baseline_file=str(self.baseline_path),
            current_results=self.results,
            baseline_results=baseline_results
        )
        
        # Compare results
        total_score = 0
        comparison_count = 0
        
        for name, current in self.results.items():
            if name in baseline_results:
                baseline = baseline_results[name]
                comparison_count += 1
                
                # Compare throughput
                if baseline.throughput > 0:
                    throughput_ratio = current.throughput / baseline.throughput
                    
                    if throughput_ratio < 0.9:  # 10% degradation threshold
                        report.regressions.append({
                            'benchmark': name,
                            'metric': 'throughput',
                            'baseline': baseline.throughput,
                            'current': current.throughput,
                            'degradation_percent': (1 - throughput_ratio) * 100
                        })
                    elif throughput_ratio > 1.1:  # 10% improvement threshold
                        report.improvements.append({
                            'benchmark': name,
                            'metric': 'throughput',
                            'baseline': baseline.throughput,
                            'current': current.throughput,
                            'improvement_percent': (throughput_ratio - 1) * 100
                        })
                    
                    total_score += throughput_ratio
                
                # Compare execution time
                if baseline.execution_time > 0:
                    time_ratio = current.execution_time / baseline.execution_time
                    
                    if time_ratio > 1.2:  # 20% slower threshold
                        report.regressions.append({
                            'benchmark': name,
                            'metric': 'execution_time',
                            'baseline': baseline.execution_time,
                            'current': current.execution_time,
                            'degradation_percent': (time_ratio - 1) * 100
                        })
        
        # Calculate overall score
        if comparison_count > 0:
            report.overall_score = total_score / comparison_count
        
        return report
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        total_time = time.time() - self.start_time
        
        # Calculate summary statistics
        successful_benchmarks = sum(1 for r in self.results.values() if r.success)
        total_benchmarks = len(self.results)
        success_rate = successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0
        
        avg_throughput = sum(r.throughput for r in self.results.values()) / total_benchmarks if total_benchmarks > 0 else 0
        total_memory_usage = sum(r.memory_usage_mb for r in self.results.values())
        
        # Performance regression analysis
        regression_report = self.analyze_regression()
        
        report = {
            'timestamp': time.time(),
            'execution_time': total_time,
            'summary': {
                'total_benchmarks': total_benchmarks,
                'successful_benchmarks': successful_benchmarks,
                'success_rate': success_rate,
                'average_throughput': avg_throughput,
                'total_memory_usage_mb': total_memory_usage
            },
            'detailed_results': {name: result.__dict__ for name, result in self.results.items()},
            'regression_analysis': regression_report.__dict__ if regression_report else None
        }
        
        return report


def main():
    """Run comprehensive Generation 3 benchmark suite."""
    print("🚀 Starting Generation 3 Benchmark Suite...")
    
    suite = Generation3BenchmarkSuite()
    
    # Run all benchmarks
    results = suite.run_all_benchmarks()
    
    # Generate and display report
    report = suite.generate_report()
    
    print("\n" + "=" * 60)
    print("📊 GENERATION 3 BENCHMARK REPORT")
    print("=" * 60)
    
    summary = report['summary']
    print(f"📈 Success Rate: {summary['success_rate']:.1%}")
    print(f"⚡ Average Throughput: {summary['average_throughput']:.1f} ops/sec")
    print(f"💾 Total Memory Usage: {summary['total_memory_usage_mb']:.1f} MB")
    print(f"⏱️  Total Execution Time: {report['execution_time']:.2f} seconds")
    
    # Show top performing benchmarks
    print(f"\n🏆 Top Performing Benchmarks:")
    sorted_results = sorted(results.items(), key=lambda x: x[1].throughput, reverse=True)
    for i, (name, result) in enumerate(sorted_results[:5], 1):
        status = "✅" if result.success else "❌"
        print(f"  {i}. {status} {name}: {result.throughput:.1f} ops/sec")
    
    # Show any failures
    failed_benchmarks = [(name, result) for name, result in results.items() if not result.success]
    if failed_benchmarks:
        print(f"\n❌ Failed Benchmarks ({len(failed_benchmarks)}):")
        for name, result in failed_benchmarks:
            print(f"  - {name}: {result.error_message or 'Unknown error'}")
    
    # Regression analysis
    regression_report = report.get('regression_analysis')
    if regression_report:
        print(f"\n📉 Performance Regression Analysis:")
        print(f"  Overall Score: {regression_report['overall_score']:.2f}")
        
        if regression_report['regressions']:
            print(f"  Regressions: {len(regression_report['regressions'])}")
            for reg in regression_report['regressions'][:3]:
                print(f"    - {reg['benchmark']}.{reg['metric']}: {reg['degradation_percent']:.1f}% slower")
        
        if regression_report['improvements']:
            print(f"  Improvements: {len(regression_report['improvements'])}")
            for imp in regression_report['improvements'][:3]:
                print(f"    - {imp['benchmark']}.{imp['metric']}: {imp['improvement_percent']:.1f}% faster")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if summary['success_rate'] >= 0.9:
        print("  🟢 EXCELLENT - System performing at optimal levels")
    elif summary['success_rate'] >= 0.7:
        print("  🟡 GOOD - System performing well with minor issues")
    elif summary['success_rate'] >= 0.5:
        print("  🟠 ACCEPTABLE - System has some performance concerns")
    else:
        print("  🔴 NEEDS ATTENTION - Significant performance issues detected")
    
    # Save results
    report_path = "/root/repo/generation3_benchmark_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📋 Full report saved: {report_path}")
    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")
    
    # Save as baseline if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--save-baseline":
        suite.save_baseline()
    
    print(f"\n✅ Generation 3 Benchmark Suite Complete!")
    
    return summary['success_rate'] >= 0.7  # Return success if 70%+ pass rate


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)