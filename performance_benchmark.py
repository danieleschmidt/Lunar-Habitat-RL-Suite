#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Lunar Habitat RL.

This script provides comprehensive performance benchmarking including:
- Training throughput measurement
- Inference latency profiling
- Memory usage analysis
- GPU utilization monitoring
- Scalability testing
- System resource profiling
"""

import os
import sys
import time
import psutil
import json
import numpy as np
import torch
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import tracemalloc
import gc


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    throughput_steps_per_sec: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    avg_gpu_utilization: Optional[float] = None
    training_time_sec: Optional[float] = None
    convergence_steps: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "throughput_steps_per_sec": self.throughput_steps_per_sec,
            "inference_latency_ms": self.inference_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "training_time_sec": self.training_time_sec,
            "convergence_steps": self.convergence_steps
        }


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    benchmark_name: str
    timestamp: str
    system_info: Dict[str, Any]
    metrics: PerformanceMetrics
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "metrics": self.metrics.to_dict(),
            "configuration": self.configuration
        }


class SystemProfiler:
    """System resource profiling utilities."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "platform": sys.platform,
        }
        
        # PyTorch info
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_memory_gb"] = [torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                                   for i in range(torch.cuda.device_count())]
        
        return info
    
    def start_monitoring(self, interval: float = 0.1):
        """Start system resource monitoring."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        
        def monitor():
            while self.monitoring:
                # CPU and memory
                self.cpu_samples.append(psutil.cpu_percent())
                self.memory_samples.append(psutil.virtual_memory().percent)
                
                # GPU utilization (if available)
                if torch.cuda.is_available():
                    try:
                        # Simple GPU utilization estimate based on memory usage
                        gpu_memory_used = []
                        for i in range(torch.cuda.device_count()):
                            mem_used = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                            gpu_memory_used.append(mem_used)
                        self.gpu_samples.append(np.mean(gpu_memory_used) if gpu_memory_used else 0)
                    except:
                        self.gpu_samples.append(0)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return average metrics."""
        self.monitoring = False
        
        return {
            "avg_cpu_percent": np.mean(self.cpu_samples) if self.cpu_samples else 0,
            "max_cpu_percent": np.max(self.cpu_samples) if self.cpu_samples else 0,
            "avg_memory_percent": np.mean(self.memory_samples) if self.memory_samples else 0,
            "max_memory_percent": np.max(self.memory_samples) if self.memory_samples else 0,
            "avg_gpu_utilization": np.mean(self.gpu_samples) if self.gpu_samples else 0,
            "max_gpu_utilization": np.max(self.gpu_samples) if self.gpu_samples else 0,
        }
    
    @contextmanager
    def profile_memory(self):
        """Context manager for memory profiling."""
        tracemalloc.start()
        
        # Force garbage collection before starting
        gc.collect()
        
        start_memory = tracemalloc.get_traced_memory()[0]
        
        try:
            yield
        finally:
            end_memory = tracemalloc.get_traced_memory()[0]
            peak_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            self.memory_usage = {
                "start_memory_mb": start_memory / 1024 / 1024,
                "end_memory_mb": end_memory / 1024 / 1024,
                "peak_memory_mb": peak_memory / 1024 / 1024,
                "memory_increase_mb": (end_memory - start_memory) / 1024 / 1024
            }


class MockEnvironment:
    """Mock environment for performance testing."""
    
    def __init__(self, obs_dim: int = 20, action_dim: int = 4):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_count = 0
        
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.obs_dim).astype(np.float32), {}
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = np.random.randn()
        terminated = self.step_count >= 1000  # Episode length
        truncated = False
        info = {'step': self.step_count}
        return obs, reward, terminated, truncated, info


class MockAgent:
    """Mock RL agent for performance testing."""
    
    def __init__(self, obs_dim: int = 20, action_dim: int = 4, hidden_size: int = 256):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_dim),
            torch.nn.Tanh()
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.network = self.network.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def predict(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.network(obs_tensor)
        return action.cpu().numpy().flatten()
    
    def train_step(self, batch_obs, batch_actions, batch_rewards):
        """Simulate training step."""
        batch_obs = torch.FloatTensor(batch_obs).to(self.device)
        batch_actions = torch.FloatTensor(batch_actions).to(self.device)
        
        # Forward pass
        pred_actions = self.network(batch_obs)
        
        # Dummy loss
        loss = torch.nn.functional.mse_loss(pred_actions, batch_actions)
        
        # Backward pass
        optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class InferenceLatencyBenchmark:
    """Benchmark inference latency performance."""
    
    def __init__(self):
        self.profiler = SystemProfiler()
    
    def benchmark(self, agent, n_samples: int = 1000, batch_sizes: List[int] = None) -> BenchmarkResult:
        """Benchmark inference latency."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 64]
        
        print(f"🚀 Running inference latency benchmark with {n_samples} samples...")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  📊 Testing batch size {batch_size}...")
            
            # Prepare batch data
            obs_batch = np.random.randn(batch_size, 20).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = agent.predict(obs_batch[0])
            
            # Benchmark
            latencies = []
            self.profiler.start_monitoring()
            
            for _ in range(n_samples // batch_size):
                start_time = time.perf_counter()
                
                if batch_size == 1:
                    _ = agent.predict(obs_batch[0])
                else:
                    # Simulate batch processing
                    for i in range(batch_size):
                        _ = agent.predict(obs_batch[i])
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            monitoring_stats = self.profiler.stop_monitoring()
            
            results[f"batch_size_{batch_size}"] = {
                "mean_latency_ms": np.mean(latencies),
                "std_latency_ms": np.std(latencies),
                "min_latency_ms": np.min(latencies),
                "max_latency_ms": np.max(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                **monitoring_stats
            }
        
        metrics = PerformanceMetrics(
            inference_latency_ms=results["batch_size_1"]["mean_latency_ms"],
            avg_cpu_percent=results["batch_size_1"]["avg_cpu_percent"],
            avg_gpu_utilization=results["batch_size_1"]["avg_gpu_utilization"]
        )
        
        return BenchmarkResult(
            benchmark_name="inference_latency",
            timestamp=datetime.now().isoformat(),
            system_info=self.profiler.get_system_info(),
            metrics=metrics,
            configuration={
                "n_samples": n_samples,
                "batch_sizes": batch_sizes,
                "detailed_results": results
            }
        )


class TrainingThroughputBenchmark:
    """Benchmark training throughput performance."""
    
    def __init__(self):
        self.profiler = SystemProfiler()
    
    def benchmark(self, agent, env, n_steps: int = 10000, batch_size: int = 256) -> BenchmarkResult:
        """Benchmark training throughput."""
        print(f"🏃 Running training throughput benchmark for {n_steps} steps...")
        
        # Generate training data
        print("  🔄 Generating training data...")
        obs_data = []
        action_data = []
        reward_data = []
        
        obs, _ = env.reset()
        for _ in range(n_steps):
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            obs_data.append(obs)
            action_data.append(action)
            reward_data.append(reward)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        obs_data = np.array(obs_data)
        action_data = np.array(action_data)
        reward_data = np.array(reward_data)
        
        # Benchmark training
        print("  🎯 Benchmarking training throughput...")
        
        with self.profiler.profile_memory():
            self.profiler.start_monitoring()
            
            start_time = time.perf_counter()
            
            n_batches = n_steps // batch_size
            losses = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_obs = obs_data[start_idx:end_idx]
                batch_actions = action_data[start_idx:end_idx]
                batch_rewards = reward_data[start_idx:end_idx]
                
                loss = agent.train_step(batch_obs, batch_actions, batch_rewards)
                losses.append(loss)
            
            end_time = time.perf_counter()
            
            monitoring_stats = self.profiler.stop_monitoring()
        
        training_time = end_time - start_time
        throughput = n_steps / training_time
        
        print(f"    ⚡ Achieved {throughput:.1f} steps/sec")
        
        metrics = PerformanceMetrics(
            throughput_steps_per_sec=throughput,
            training_time_sec=training_time,
            peak_memory_mb=self.profiler.memory_usage["peak_memory_mb"],
            avg_cpu_percent=monitoring_stats["avg_cpu_percent"],
            avg_gpu_utilization=monitoring_stats["avg_gpu_utilization"]
        )
        
        return BenchmarkResult(
            benchmark_name="training_throughput",
            timestamp=datetime.now().isoformat(),
            system_info=self.profiler.get_system_info(),
            metrics=metrics,
            configuration={
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_batches": n_batches,
                "final_loss": losses[-1] if losses else None,
                "memory_usage": self.profiler.memory_usage,
                "monitoring_stats": monitoring_stats
            }
        )


class ScalabilityBenchmark:
    """Benchmark scalability with different configurations."""
    
    def __init__(self):
        self.profiler = SystemProfiler()
    
    def benchmark(self, 
                 obs_dims: List[int] = None,
                 action_dims: List[int] = None,
                 hidden_sizes: List[int] = None,
                 n_samples: int = 1000) -> BenchmarkResult:
        """Benchmark scalability across different model sizes."""
        
        if obs_dims is None:
            obs_dims = [10, 20, 50, 100]
        if action_dims is None:
            action_dims = [2, 4, 8, 16]
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 256, 512]
        
        print(f"📈 Running scalability benchmark...")
        
        results = {}
        
        # Test observation dimension scaling
        print("  🔍 Testing observation dimension scaling...")
        obs_scaling_results = {}
        for obs_dim in obs_dims:
            agent = MockAgent(obs_dim=obs_dim, action_dim=4, hidden_size=256)
            env = MockEnvironment(obs_dim=obs_dim, action_dim=4)
            
            # Quick throughput test
            start_time = time.perf_counter()
            for _ in range(100):
                obs, _ = env.reset()
                _ = agent.predict(obs)
            end_time = time.perf_counter()
            
            throughput = 100 / (end_time - start_time)
            obs_scaling_results[obs_dim] = {
                "throughput_fps": throughput,
                "model_parameters": sum(p.numel() for p in agent.network.parameters())
            }
        
        results["obs_dimension_scaling"] = obs_scaling_results
        
        # Test action dimension scaling
        print("  🎮 Testing action dimension scaling...")
        action_scaling_results = {}
        for action_dim in action_dims:
            agent = MockAgent(obs_dim=20, action_dim=action_dim, hidden_size=256)
            
            start_time = time.perf_counter()
            for _ in range(100):
                obs = np.random.randn(20).astype(np.float32)
                _ = agent.predict(obs)
            end_time = time.perf_counter()
            
            throughput = 100 / (end_time - start_time)
            action_scaling_results[action_dim] = {
                "throughput_fps": throughput,
                "model_parameters": sum(p.numel() for p in agent.network.parameters())
            }
        
        results["action_dimension_scaling"] = action_scaling_results
        
        # Test hidden size scaling
        print("  🧠 Testing hidden size scaling...")
        hidden_scaling_results = {}
        for hidden_size in hidden_sizes:
            agent = MockAgent(obs_dim=20, action_dim=4, hidden_size=hidden_size)
            
            start_time = time.perf_counter()
            for _ in range(100):
                obs = np.random.randn(20).astype(np.float32)
                _ = agent.predict(obs)
            end_time = time.perf_counter()
            
            throughput = 100 / (end_time - start_time)
            hidden_scaling_results[hidden_size] = {
                "throughput_fps": throughput,
                "model_parameters": sum(p.numel() for p in agent.network.parameters())
            }
        
        results["hidden_size_scaling"] = hidden_scaling_results
        
        metrics = PerformanceMetrics()  # No single metric for scalability test
        
        return BenchmarkResult(
            benchmark_name="scalability",
            timestamp=datetime.now().isoformat(),
            system_info=self.profiler.get_system_info(),
            metrics=metrics,
            configuration={
                "obs_dims": obs_dims,
                "action_dims": action_dims,
                "hidden_sizes": hidden_sizes,
                "n_samples": n_samples,
                "scaling_results": results
            }
        )


class MemoryBenchmark:
    """Benchmark memory usage patterns."""
    
    def __init__(self):
        self.profiler = SystemProfiler()
    
    def benchmark(self, agent, env, episode_lengths: List[int] = None) -> BenchmarkResult:
        """Benchmark memory usage during training."""
        
        if episode_lengths is None:
            episode_lengths = [100, 500, 1000, 2000]
        
        print(f"💾 Running memory usage benchmark...")
        
        results = {}
        
        for episode_length in episode_lengths:
            print(f"  📊 Testing episode length {episode_length}...")
            
            with self.profiler.profile_memory():
                # Simulate episode
                obs, _ = env.reset()
                for _ in range(episode_length):
                    action = agent.predict(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            results[f"episode_length_{episode_length}"] = self.profiler.memory_usage.copy()
        
        # Find peak memory usage
        peak_memory = max(result["peak_memory_mb"] for result in results.values())
        
        metrics = PerformanceMetrics(
            peak_memory_mb=peak_memory
        )
        
        return BenchmarkResult(
            benchmark_name="memory_usage",
            timestamp=datetime.now().isoformat(),
            system_info=self.profiler.get_system_info(),
            metrics=metrics,
            configuration={
                "episode_lengths": episode_lengths,
                "memory_results": results
            }
        )


class PerformanceBenchmarkSuite:
    """Main benchmark suite orchestrator."""
    
    def __init__(self):
        self.results = []
    
    def run_all_benchmarks(self, 
                          quick_mode: bool = False,
                          output_file: Optional[str] = None) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        
        print("🎯 Starting Performance Benchmark Suite")
        print("=" * 60)
        
        # Create mock agent and environment
        agent = MockAgent(obs_dim=20, action_dim=4, hidden_size=256)
        env = MockEnvironment(obs_dim=20, action_dim=4)
        
        benchmarks = []
        
        # 1. Inference Latency Benchmark
        print("\n1️⃣ Inference Latency Benchmark")
        print("-" * 30)
        latency_bench = InferenceLatencyBenchmark()
        n_samples = 500 if quick_mode else 1000
        result = latency_bench.benchmark(agent, n_samples=n_samples)
        benchmarks.append(result)
        self.results.append(result)
        
        # 2. Training Throughput Benchmark
        print("\n2️⃣ Training Throughput Benchmark") 
        print("-" * 30)
        throughput_bench = TrainingThroughputBenchmark()
        n_steps = 5000 if quick_mode else 10000
        result = throughput_bench.benchmark(agent, env, n_steps=n_steps)
        benchmarks.append(result)
        self.results.append(result)
        
        # 3. Scalability Benchmark
        print("\n3️⃣ Scalability Benchmark")
        print("-" * 30)
        scalability_bench = ScalabilityBenchmark()
        result = scalability_bench.benchmark()
        benchmarks.append(result)
        self.results.append(result)
        
        # 4. Memory Usage Benchmark
        print("\n4️⃣ Memory Usage Benchmark")
        print("-" * 30)
        memory_bench = MemoryBenchmark()
        episode_lengths = [100, 500, 1000] if quick_mode else [100, 500, 1000, 2000]
        result = memory_bench.benchmark(agent, env, episode_lengths=episode_lengths)
        benchmarks.append(result)
        self.results.append(result)
        
        # Generate summary report
        self.generate_summary_report(benchmarks, output_file)
        
        return benchmarks
    
    def generate_summary_report(self, 
                              benchmarks: List[BenchmarkResult], 
                              output_file: Optional[str] = None):
        """Generate comprehensive benchmark report."""
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        # System information
        system_info = benchmarks[0].system_info
        print(f"\n🖥️  SYSTEM CONFIGURATION:")
        print(f"   • CPU: {system_info['cpu_count']} cores @ {system_info.get('cpu_freq_mhz', 'N/A')} MHz")
        print(f"   • Memory: {system_info['memory_total_gb']:.1f} GB")
        print(f"   • Python: {system_info['python_version'].split()[0]}")
        print(f"   • PyTorch: {system_info['torch_version']}")
        print(f"   • CUDA Available: {system_info['cuda_available']}")
        if system_info['cuda_available']:
            print(f"   • GPU Count: {system_info['gpu_count']}")
            for i, gpu_name in enumerate(system_info['gpu_names']):
                gpu_mem = system_info['gpu_memory_gb'][i]
                print(f"   • GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        
        print(f"\n📈 BENCHMARK RESULTS:")
        
        # Inference latency results
        latency_result = next(r for r in benchmarks if r.benchmark_name == "inference_latency")
        print(f"\n🚀 Inference Performance:")
        print(f"   • Mean Latency: {latency_result.metrics.inference_latency_ms:.2f} ms")
        batch_results = latency_result.configuration["detailed_results"]
        for batch_size in [1, 16, 64]:
            if f"batch_size_{batch_size}" in batch_results:
                result = batch_results[f"batch_size_{batch_size}"]
                print(f"   • Batch {batch_size} - Mean: {result['mean_latency_ms']:.2f} ms, "
                      f"P95: {result['p95_latency_ms']:.2f} ms")
        
        # Training throughput results
        throughput_result = next(r for r in benchmarks if r.benchmark_name == "training_throughput")
        print(f"\n🏃 Training Performance:")
        print(f"   • Throughput: {throughput_result.metrics.throughput_steps_per_sec:.1f} steps/sec")
        print(f"   • Training Time: {throughput_result.metrics.training_time_sec:.1f} seconds")
        print(f"   • Peak Memory: {throughput_result.metrics.peak_memory_mb:.1f} MB")
        print(f"   • Avg CPU Usage: {throughput_result.metrics.avg_cpu_percent:.1f}%")
        if throughput_result.metrics.avg_gpu_utilization:
            print(f"   • Avg GPU Usage: {throughput_result.metrics.avg_gpu_utilization:.1f}%")
        
        # Memory usage results
        memory_result = next(r for r in benchmarks if r.benchmark_name == "memory_usage")
        print(f"\n💾 Memory Performance:")
        print(f"   • Peak Memory Usage: {memory_result.metrics.peak_memory_mb:.1f} MB")
        memory_results = memory_result.configuration["memory_results"]
        for episode_length in sorted([int(k.split('_')[2]) for k in memory_results.keys()]):
            result = memory_results[f"episode_length_{episode_length}"]
            print(f"   • Episode {episode_length}: {result['peak_memory_mb']:.1f} MB peak")
        
        # Scalability results
        scalability_result = next(r for r in benchmarks if r.benchmark_name == "scalability")
        print(f"\n📈 Scalability Performance:")
        scaling_results = scalability_result.configuration["scaling_results"]
        
        if "obs_dimension_scaling" in scaling_results:
            print(f"   • Observation Dimension Scaling:")
            for obs_dim, result in scaling_results["obs_dimension_scaling"].items():
                print(f"     - {obs_dim}D: {result['throughput_fps']:.1f} FPS "
                      f"({result['model_parameters']} params)")
        
        if "hidden_size_scaling" in scaling_results:
            print(f"   • Hidden Size Scaling:")
            for hidden_size, result in scaling_results["hidden_size_scaling"].items():
                print(f"     - {hidden_size}: {result['throughput_fps']:.1f} FPS "
                      f"({result['model_parameters']} params)")
        
        # Performance recommendations
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
        
        # Based on inference latency
        if latency_result.metrics.inference_latency_ms > 10:
            print("   • Consider model optimization for real-time inference")
        
        # Based on memory usage
        if memory_result.metrics.peak_memory_mb > 1000:
            print("   • Consider memory optimization for large-scale training")
        
        # Based on GPU utilization
        if system_info['cuda_available'] and throughput_result.metrics.avg_gpu_utilization < 50:
            print("   • GPU utilization could be improved - consider larger batch sizes")
        
        # Based on CPU utilization
        if throughput_result.metrics.avg_cpu_percent < 50:
            print("   • CPU utilization could be improved - consider parallel processing")
        
        print("   • Monitor performance in production environments")
        print("   • Consider distributed training for scaling beyond single machine")
        print("   • Profile actual algorithms on real environments for accurate metrics")
        
        print("\n" + "=" * 60)
        
        # Save detailed results to file
        if output_file:
            detailed_results = {
                "timestamp": datetime.now().isoformat(),
                "system_info": system_info,
                "benchmark_results": [result.to_dict() for result in benchmarks]
            }
            
            print(f"💾 Saving detailed results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)


def main():
    """Main entry point for performance benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite for Lunar Habitat RL")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks (fewer samples)")
    parser.add_argument("--output", "-o", help="Output file for detailed JSON results")
    
    args = parser.parse_args()
    
    try:
        suite = PerformanceBenchmarkSuite()
        results = suite.run_all_benchmarks(
            quick_mode=args.quick,
            output_file=args.output
        )
        
        print(f"\n✅ Benchmark suite completed successfully!")
        print(f"📊 Total benchmarks run: {len(results)}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error during performance benchmarking: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())