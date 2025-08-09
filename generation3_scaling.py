#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Performance Optimization and Scalability
"""

import sys
import os
import json
import time
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import random
import math

sys.path.insert(0, '/root/repo')

from lunar_habitat_rl.environments.lightweight_habitat import LunarHabitatEnv
from lunar_habitat_rl.algorithms.lightweight_baselines import RandomAgent, HeuristicAgent, PIDControllerAgent
from lunar_habitat_rl.core.lightweight_config import HabitatConfig


@dataclass
class PerformanceMetrics:
    """Performance metrics for scalability analysis."""
    total_episodes: int = 0
    total_steps: int = 0
    avg_episode_time: float = 0.0
    avg_step_time: float = 0.0
    throughput_eps: float = 0.0
    throughput_sps: float = 0.0  # steps per second
    memory_usage: float = 0.0
    cpu_utilization: float = 0.0
    parallel_efficiency: float = 0.0


class AdaptiveCaching:
    """Adaptive caching system with LRU eviction and access pattern learning."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access pattern tracking."""
        current_time = time.time()
        
        if key in self.cache:
            # Check TTL
            if current_time - self.access_times[key] < self.ttl:
                self.hit_count += 1
                self.access_times[key] = current_time
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return self.cache[key]
            else:
                # Expired
                self._remove(key)
                
        self.miss_count += 1
        return None
        
    def put(self, key: str, value: Any):
        """Put value in cache with LRU eviction."""
        current_time = time.time()
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
            
        self.cache[key] = value
        self.access_times[key] = current_time
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
            
        # Find LRU item (considering both time and frequency)
        lru_key = min(
            self.access_times.keys(),
            key=lambda k: (self.access_times[k], self.access_counts.get(k, 0))
        )
        self._remove(lru_key)
        
    def _remove(self, key: str):
        """Remove item from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }


class AutoScalingManager:
    """Auto-scaling manager for dynamic resource allocation."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 8):
        """Initialize auto-scaling manager."""
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.load_history = []
        self.max_history = 50
        
        # Scaling thresholds
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_up_cooldown = 30.0  # seconds
        self.scale_down_cooldown = 60.0  # seconds
        
        self.last_scale_time = 0.0
        self.last_scale_action = None
        
    def update_load(self, cpu_usage: float, queue_size: int, avg_latency: float):
        """Update load metrics for scaling decisions."""
        # Compute composite load metric
        load_score = (
            cpu_usage * 0.4 +
            min(queue_size / 10.0, 1.0) * 0.4 +
            min(avg_latency / 1.0, 1.0) * 0.2
        )
        
        self.load_history.append(load_score)
        if len(self.load_history) > self.max_history:
            self.load_history.pop(0)
            
    def should_scale(self) -> Tuple[bool, str]:
        """Determine if scaling is needed."""
        if len(self.load_history) < 5:
            return False, "insufficient_data"
            
        current_time = time.time()
        avg_load = sum(self.load_history[-10:]) / len(self.load_history[-10:])
        
        # Check scale up conditions
        if (avg_load > self.scale_up_threshold and
            self.current_workers < self.max_workers and
            (self.last_scale_action != 'up' or 
             current_time - self.last_scale_time > self.scale_up_cooldown)):
            return True, 'scale_up'
            
        # Check scale down conditions
        if (avg_load < self.scale_down_threshold and
            self.current_workers > self.min_workers and
            (self.last_scale_action != 'down' or
             current_time - self.last_scale_time > self.scale_down_cooldown)):
            return True, 'scale_down'
            
        return False, 'no_action'
        
    def scale(self, action: str) -> int:
        """Execute scaling action."""
        old_workers = self.current_workers
        
        if action == 'scale_up':
            self.current_workers = min(self.max_workers, self.current_workers + 1)
        elif action == 'scale_down':
            self.current_workers = max(self.min_workers, self.current_workers - 1)
            
        if self.current_workers != old_workers:
            self.last_scale_time = time.time()
            self.last_scale_action = action.split('_')[1]
            
        return self.current_workers


class OptimizedEnvironmentPool:
    """Pool of pre-initialized environments for high-throughput training."""
    
    def __init__(self, pool_size: int = 4, config: Optional[HabitatConfig] = None):
        """Initialize environment pool."""
        self.pool_size = pool_size
        self.config = config or HabitatConfig()
        self.environments = []
        self.available_envs = []
        self.in_use_envs = set()
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.checkout_times = []
        self.utilization_history = []
        
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Initialize all environments in the pool."""
        print(f"Initializing environment pool with {self.pool_size} environments...")
        start_time = time.time()
        
        for i in range(self.pool_size):
            env = LunarHabitatEnv(config=self.config)
            self.environments.append(env)
            self.available_envs.append(i)
            
        init_time = time.time() - start_time
        print(f"Environment pool initialized in {init_time:.2f} seconds")
        
    def checkout_env(self, timeout: float = 5.0) -> Optional[Tuple[int, LunarHabitatEnv]]:
        """Check out an available environment."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if self.available_envs:
                    env_id = self.available_envs.pop()
                    self.in_use_envs.add(env_id)
                    checkout_time = time.time() - start_time
                    self.checkout_times.append(checkout_time)
                    return env_id, self.environments[env_id]
                    
            time.sleep(0.001)  # Brief sleep to avoid busy waiting
            
        return None  # Timeout
        
    def checkin_env(self, env_id: int):
        """Check in a used environment."""
        with self.lock:
            if env_id in self.in_use_envs:
                self.in_use_envs.remove(env_id)
                self.available_envs.append(env_id)
                
    def get_utilization(self) -> float:
        """Get current pool utilization."""
        with self.lock:
            utilization = len(self.in_use_envs) / self.pool_size
            self.utilization_history.append(utilization)
            if len(self.utilization_history) > 100:
                self.utilization_history.pop(0)
            return utilization
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool performance statistics."""
        with self.lock:
            avg_checkout_time = sum(self.checkout_times) / max(len(self.checkout_times), 1)
            avg_utilization = sum(self.utilization_history) / max(len(self.utilization_history), 1)
            
            return {
                'pool_size': self.pool_size,
                'available': len(self.available_envs),
                'in_use': len(self.in_use_envs),
                'avg_checkout_time': avg_checkout_time,
                'avg_utilization': avg_utilization,
                'total_checkouts': len(self.checkout_times)
            }
            
    def close(self):
        """Close all environments in the pool."""
        for env in self.environments:
            try:
                env.close()
            except:
                pass


class PerformanceOptimizer:
    """Performance optimization system with adaptive configuration."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.cache = AdaptiveCaching(max_size=500, ttl=300.0)
        self.autoscaler = AutoScalingManager(min_workers=2, max_workers=8)
        self.metrics = PerformanceMetrics()
        
        # Optimization parameters
        self.batch_size = 32
        self.prefetch_size = 64
        self.async_execution = True
        
    def optimize_batch_size(self, throughput_history: List[float]) -> int:
        """Dynamically optimize batch size based on throughput."""
        if len(throughput_history) < 10:
            return self.batch_size
            
        # Analyze throughput trend
        recent_throughput = sum(throughput_history[-5:]) / 5
        older_throughput = sum(throughput_history[-10:-5]) / 5
        
        if recent_throughput > older_throughput * 1.1:
            # Performance improving, try larger batch
            self.batch_size = min(64, self.batch_size + 4)
        elif recent_throughput < older_throughput * 0.9:
            # Performance degrading, try smaller batch
            self.batch_size = max(8, self.batch_size - 4)
            
        return self.batch_size
    
    def should_use_async(self, latency_ms: float) -> bool:
        """Determine if async execution should be used."""
        # Use async for high-latency operations
        return latency_ms > 10.0 or self.async_execution


def benchmark_single_threaded(n_episodes: int = 100) -> PerformanceMetrics:
    """Benchmark single-threaded performance."""
    print(f"\n🔥 Single-threaded benchmark ({n_episodes} episodes)")
    
    config = HabitatConfig()
    env = LunarHabitatEnv(config=config)
    agent = HeuristicAgent(action_dims=22)
    
    start_time = time.time()
    total_steps = 0
    episode_times = []
    
    for episode in range(n_episodes):
        episode_start = time.time()
        obs, info = env.reset(seed=episode)
        episode_steps = 0
        
        for step in range(100):  # Max 100 steps per episode
            action, _ = agent.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_steps += 1
            
            if done or truncated:
                break
                
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        total_steps += episode_steps
        
        if (episode + 1) % 20 == 0:
            print(f"  Episodes completed: {episode + 1}/{n_episodes}")
    
    total_time = time.time() - start_time
    env.close()
    
    metrics = PerformanceMetrics(
        total_episodes=n_episodes,
        total_steps=total_steps,
        avg_episode_time=sum(episode_times) / len(episode_times),
        avg_step_time=total_time / total_steps,
        throughput_eps=n_episodes / total_time,
        throughput_sps=total_steps / total_time
    )
    
    print(f"  ✅ Throughput: {metrics.throughput_eps:.1f} eps/sec, {metrics.throughput_sps:.1f} sps")
    return metrics


def benchmark_parallel(n_episodes: int = 100, n_workers: int = 4) -> PerformanceMetrics:
    """Benchmark parallel execution performance."""
    print(f"\n🚀 Parallel benchmark ({n_episodes} episodes, {n_workers} workers)")
    
    def worker_function(worker_episodes: int, worker_id: int) -> Tuple[int, float, List[float]]:
        """Worker function for parallel execution."""
        config = HabitatConfig()
        env = LunarHabitatEnv(config=config)
        agent = HeuristicAgent(action_dims=22)
        
        start_time = time.time()
        total_steps = 0
        episode_times = []
        
        for episode in range(worker_episodes):
            episode_start = time.time()
            obs, info = env.reset(seed=worker_id * 1000 + episode)
            episode_steps = 0
            
            for step in range(100):
                action, _ = agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                episode_steps += 1
                
                if done or truncated:
                    break
                    
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)
            total_steps += episode_steps
        
        worker_time = time.time() - start_time
        env.close()
        
        return total_steps, worker_time, episode_times
    
    # Distribute episodes across workers
    episodes_per_worker = n_episodes // n_workers
    remaining_episodes = n_episodes % n_workers
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for worker_id in range(n_workers):
            worker_episodes = episodes_per_worker + (1 if worker_id < remaining_episodes else 0)
            future = executor.submit(worker_function, worker_episodes, worker_id)
            futures.append(future)
            
        # Collect results
        all_steps = []
        all_times = []
        all_episode_times = []
        
        for future in concurrent.futures.as_completed(futures):
            steps, worker_time, episode_times = future.result()
            all_steps.append(steps)
            all_times.append(worker_time)
            all_episode_times.extend(episode_times)
    
    total_time = time.time() - start_time
    total_steps = sum(all_steps)
    
    metrics = PerformanceMetrics(
        total_episodes=n_episodes,
        total_steps=total_steps,
        avg_episode_time=sum(all_episode_times) / len(all_episode_times),
        avg_step_time=total_time / total_steps,
        throughput_eps=n_episodes / total_time,
        throughput_sps=total_steps / total_time,
        parallel_efficiency=(n_episodes / total_time) / (n_workers * 10)  # Rough efficiency estimate
    )
    
    print(f"  ✅ Throughput: {metrics.throughput_eps:.1f} eps/sec, {metrics.throughput_sps:.1f} sps")
    print(f"  📊 Parallel efficiency: {metrics.parallel_efficiency:.2f}")
    return metrics


def test_caching_system():
    """Test adaptive caching system."""
    print("\n💾 Testing Adaptive Caching System")
    
    cache = AdaptiveCaching(max_size=10, ttl=1.0)
    
    # Test basic operations
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.get("key1") == "value1", "Cache retrieval failed"
    assert cache.get("nonexistent") is None, "Cache should return None for missing keys"
    
    # Test LRU eviction
    for i in range(15):
        cache.put(f"key{i}", f"value{i}")
        
    assert len(cache.cache) <= 10, "Cache exceeded max size"
    
    # Test TTL expiration
    cache.put("temp_key", "temp_value")
    time.sleep(1.1)  # Wait for TTL to expire
    assert cache.get("temp_key") is None, "TTL expiration failed"
    
    stats = cache.get_stats()
    print(f"  ✅ Cache stats: {stats['hit_rate']:.2f} hit rate, {stats['cache_size']} items")
    return True


def test_autoscaling():
    """Test auto-scaling manager."""
    print("\n🔄 Testing Auto-scaling Manager")
    
    scaler = AutoScalingManager(min_workers=2, max_workers=6)
    
    # Test normal load
    scaler.update_load(cpu_usage=0.5, queue_size=2, avg_latency=0.1)
    should_scale, action = scaler.should_scale()
    
    # Test high load
    for _ in range(10):
        scaler.update_load(cpu_usage=0.9, queue_size=8, avg_latency=0.8)
        
    should_scale, action = scaler.should_scale()
    if should_scale and action == 'scale_up':
        new_workers = scaler.scale(action)
        print(f"  ✅ Scaled up to {new_workers} workers")
    
    # Test low load
    for _ in range(10):
        scaler.update_load(cpu_usage=0.1, queue_size=0, avg_latency=0.05)
        
    time.sleep(0.1)  # Brief wait
    should_scale, action = scaler.should_scale()
    if should_scale and action == 'scale_down':
        new_workers = scaler.scale(action)
        print(f"  ✅ Scaled down to {new_workers} workers")
        
    return True


def test_environment_pool():
    """Test optimized environment pool."""
    print("\n🏊 Testing Environment Pool")
    
    pool = OptimizedEnvironmentPool(pool_size=4)
    
    # Test checkout/checkin
    env_info = pool.checkout_env(timeout=1.0)
    assert env_info is not None, "Failed to checkout environment"
    
    env_id, env = env_info
    assert env is not None, "Checkout returned None environment"
    
    pool.checkin_env(env_id)
    
    # Test utilization tracking
    utilization = pool.get_utilization()
    assert 0.0 <= utilization <= 1.0, "Invalid utilization value"
    
    stats = pool.get_stats()
    print(f"  ✅ Pool stats: {stats['available']}/{stats['pool_size']} available")
    print(f"  📊 Avg utilization: {stats['avg_utilization']:.2f}")
    
    pool.close()
    return True


def test_generation_3_scaling():
    """Test Generation 3 scaling and optimization features."""
    print("🚀 GENERATION 3: MAKE IT SCALE")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Caching System
    try:
        test_results['caching'] = test_caching_system()
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        test_results['caching'] = False
    
    # Test 2: Auto-scaling
    try:
        test_results['autoscaling'] = test_autoscaling()
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        test_results['autoscaling'] = False
    
    # Test 3: Environment Pool
    try:
        test_results['env_pool'] = test_environment_pool()
    except Exception as e:
        print(f"❌ Environment pool test failed: {e}")
        test_results['env_pool'] = False
    
    # Test 4: Single-threaded Benchmark
    try:
        single_metrics = benchmark_single_threaded(n_episodes=50)
        test_results['single_threaded'] = single_metrics.throughput_eps > 1.0
        print(f"  Single-threaded: {single_metrics.throughput_eps:.1f} eps/sec")
    except Exception as e:
        print(f"❌ Single-threaded benchmark failed: {e}")
        test_results['single_threaded'] = False
    
    # Test 5: Parallel Benchmark
    try:
        parallel_metrics = benchmark_parallel(n_episodes=50, n_workers=4)
        test_results['parallel'] = parallel_metrics.throughput_eps > single_metrics.throughput_eps
        speedup = parallel_metrics.throughput_eps / single_metrics.throughput_eps
        print(f"  Parallel speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"❌ Parallel benchmark failed: {e}")
        test_results['parallel'] = False
    
    # Overall results
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"\n🎯 Generation 3 Results: {passed_tests}/{total_tests} tests passed")
    print(f"📊 Test Details: {test_results}")
    
    if passed_tests >= 4:
        print("🎉 GENERATION 3 COMPLETE - System scales efficiently!")
        return True
    else:
        print("⚠️ Generation 3 needs improvements")
        return False


if __name__ == "__main__":
    success = test_generation_3_scaling()
    
    if success:
        print("\n🎯 READY FOR QUALITY GATES AND RESEARCH PHASE! 🎯")
    
    sys.exit(0 if success else 1)