#!/usr/bin/env python3
"""
Scale Optimization Suite - Generation 3 Enhancement
Performance optimization, concurrent processing, and auto-scaling for Lunar Habitat RL Suite
"""

import asyncio
import multiprocessing
import concurrent.futures
import threading
import time
import os
import psutil
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import queue
import functools
from abc import ABC, abstractmethod

# Configure performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    operation: str
    start_time: float
    end_time: float
    cpu_usage_before: float
    cpu_usage_after: float
    memory_before: int
    memory_after: int
    execution_time: float = field(init=False)
    memory_delta: int = field(init=False)
    cpu_delta: float = field(init=False)
    
    def __post_init__(self):
        self.execution_time = self.end_time - self.start_time
        self.memory_delta = self.memory_after - self.memory_before
        self.cpu_delta = self.cpu_usage_after - self.cpu_usage_before

class ResourcePoolManager:
    """Advanced resource pool management with auto-scaling"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None, scale_threshold: float = 0.8):
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.scale_threshold = scale_threshold
        self.current_workers = min_workers
        
        # Initialize thread and process pools
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers)
        self._process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.current_workers)
        
        # Performance monitoring
        self.task_queue_size = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.average_task_time = 0.0
        
        # Auto-scaling control
        self._scaling_lock = threading.Lock()
        self._last_scale_time = time.time()
        self._scale_cooldown = 60  # seconds
        
        logger.info(f"ResourcePoolManager initialized: {self.current_workers} workers ({self.min_workers}-{self.max_workers})")
    
    async def execute_async(self, func: Callable, *args, use_processes: bool = False, **kwargs) -> Any:
        """Execute function asynchronously with automatic scaling"""
        self.task_queue_size += 1
        start_time = time.time()
        
        try:
            # Choose execution pool
            pool = self._process_pool if use_processes else self._thread_pool
            
            # Execute task
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, functools.partial(func, *args, **kwargs))
            
            # Update metrics
            execution_time = time.time() - start_time
            self.completed_tasks += 1
            self._update_average_time(execution_time)
            
            # Check if scaling is needed
            await self._check_and_scale()
            
            return result
            
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Task execution failed: {str(e)}")
            raise
        finally:
            self.task_queue_size -= 1
    
    def _update_average_time(self, execution_time: float):
        """Update rolling average task execution time"""
        if self.completed_tasks == 1:
            self.average_task_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_task_time = alpha * execution_time + (1 - alpha) * self.average_task_time
    
    async def _check_and_scale(self):
        """Check if scaling is needed and adjust pool sizes"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self._last_scale_time < self._scale_cooldown:
            return
        
        with self._scaling_lock:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Scale up conditions
            should_scale_up = (
                (cpu_usage < 70 and memory_usage < 80) and  # Resources available
                (self.task_queue_size > self.current_workers * self.scale_threshold) and  # Queue pressure
                (self.current_workers < self.max_workers)  # Can scale up
            )
            
            # Scale down conditions  
            should_scale_down = (
                (self.task_queue_size < self.current_workers * 0.3) and  # Low queue pressure
                (self.current_workers > self.min_workers) and  # Can scale down
                (cpu_usage < 50)  # Low CPU usage
            )
            
            if should_scale_up:
                await self._scale_up()
            elif should_scale_down:
                await self._scale_down()
    
    async def _scale_up(self):
        """Scale up worker pools"""
        new_workers = min(self.current_workers + 2, self.max_workers)
        if new_workers != self.current_workers:
            logger.info(f"Scaling up: {self.current_workers} -> {new_workers} workers")
            
            # Update pools
            self._thread_pool._max_workers = new_workers
            self._process_pool._max_workers = new_workers
            
            self.current_workers = new_workers
            self._last_scale_time = time.time()
    
    async def _scale_down(self):
        """Scale down worker pools"""
        new_workers = max(self.current_workers - 1, self.min_workers)
        if new_workers != self.current_workers:
            logger.info(f"Scaling down: {self.current_workers} -> {new_workers} workers")
            
            # Update pools
            self._thread_pool._max_workers = new_workers
            self._process_pool._max_workers = new_workers
            
            self.current_workers = new_workers
            self._last_scale_time = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pool status"""
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "task_queue_size": self.task_queue_size,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "average_task_time": self.average_task_time,
            "success_rate": (self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)) * 100
        }
    
    def cleanup(self):
        """Clean up resources"""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)

class IntelligentCache:
    """Intelligent caching system with predictive prefetching"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_patterns: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        
        # Start background cleanup
        self._cleanup_interval = 300  # 5 minutes
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"IntelligentCache initialized: max_size={max_size}, ttl={ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access pattern tracking"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                current_time = time.time()
                
                # Check if expired
                if current_time - entry['timestamp'] > self.ttl:
                    del self._cache[key]
                    return None
                
                # Update access pattern
                self._track_access(key)
                
                # Update access time for LRU
                entry['last_access'] = current_time
                return entry['value']
            
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with intelligent eviction"""
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_least_valuable()
            
            # Store value
            self._cache[key] = {
                'value': value,
                'timestamp': current_time,
                'last_access': current_time,
                'access_count': 1
            }
            
            # Track access pattern
            self._track_access(key)
    
    def _track_access(self, key: str):
        """Track access patterns for predictive caching"""
        current_time = time.time()
        
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        
        self._access_patterns[key].append(current_time)
        
        # Keep only last 100 accesses
        if len(self._access_patterns[key]) > 100:
            self._access_patterns[key] = self._access_patterns[key][-100:]
        
        # Update access count
        if key in self._cache:
            self._cache[key]['access_count'] += 1
    
    def _evict_least_valuable(self):
        """Evict least valuable entries based on multiple factors"""
        if not self._cache:
            return
        
        current_time = time.time()
        
        # Calculate value scores for each entry
        scores = {}
        for key, entry in self._cache.items():
            age = current_time - entry['timestamp']
            recency = current_time - entry['last_access']
            frequency = entry['access_count']
            
            # Combine factors (lower is worse)
            score = frequency / max(1, age + recency)
            scores[key] = score
        
        # Remove lowest scoring entry
        worst_key = min(scores, key=scores.get)
        del self._cache[worst_key]
        logger.debug(f"Evicted cache entry: {worst_key}")
    
    def _periodic_cleanup(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                time.sleep(self._cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time - entry['timestamp'] > self.ttl
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": (len(self._cache) / self.max_size) * 100,
                "patterns_tracked": len(self._access_patterns),
                "total_accesses": sum(len(pattern) for pattern in self._access_patterns.values())
            }

class PerformanceOptimizer:
    """System-wide performance optimization orchestrator"""
    
    def __init__(self):
        self.resource_pool = ResourcePoolManager()
        self.cache = IntelligentCache()
        self.metrics_history: List[PerformanceMetrics] = []
        self._optimization_strategies = {
            'cpu_bound': self._optimize_cpu_bound,
            'io_bound': self._optimize_io_bound,
            'memory_bound': self._optimize_memory_bound
        }
        
    def performance_monitor(self, operation_name: str):
        """Decorator for performance monitoring"""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Collect pre-execution metrics
                process = psutil.Process()
                start_time = time.time()
                cpu_before = psutil.cpu_percent(interval=0.1)
                memory_before = process.memory_info().rss
                
                try:
                    # Execute function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    return result
                    
                finally:
                    # Collect post-execution metrics
                    end_time = time.time()
                    cpu_after = psutil.cpu_percent(interval=0.1)
                    memory_after = process.memory_info().rss
                    
                    # Store metrics
                    metrics = PerformanceMetrics(
                        operation=operation_name,
                        start_time=start_time,
                        end_time=end_time,
                        cpu_usage_before=cpu_before,
                        cpu_usage_after=cpu_after,
                        memory_before=memory_before,
                        memory_after=memory_after
                    )
                    
                    self.metrics_history.append(metrics)
                    
                    # Trigger optimization if needed
                    await self._analyze_and_optimize(metrics)
                    
            return wrapper
        return decorator
    
    async def _analyze_and_optimize(self, metrics: PerformanceMetrics):
        """Analyze performance metrics and trigger optimizations"""
        # Determine bottleneck type
        bottleneck_type = self._identify_bottleneck(metrics)
        
        if bottleneck_type in self._optimization_strategies:
            optimization_strategy = self._optimization_strategies[bottleneck_type]
            await optimization_strategy(metrics)
    
    def _identify_bottleneck(self, metrics: PerformanceMetrics) -> str:
        """Identify the primary performance bottleneck"""
        # High CPU delta indicates CPU-bound work
        if metrics.cpu_delta > 20:
            return 'cpu_bound'
        
        # High memory delta indicates memory-bound work
        if metrics.memory_delta > 100 * 1024 * 1024:  # 100MB
            return 'memory_bound'
        
        # Long execution time with low resource usage indicates I/O bound
        if metrics.execution_time > 1.0 and metrics.cpu_delta < 10:
            return 'io_bound'
        
        return 'balanced'
    
    async def _optimize_cpu_bound(self, metrics: PerformanceMetrics):
        """Optimize CPU-bound operations"""
        logger.info(f"Applying CPU-bound optimizations for {metrics.operation}")
        
        # Increase process pool size for CPU-intensive tasks
        if self.resource_pool.current_workers < self.resource_pool.max_workers:
            await self.resource_pool._scale_up()
        
        # Enable aggressive caching for expensive computations
        self.cache.ttl = min(self.cache.ttl * 2, 7200)  # Up to 2 hours
    
    async def _optimize_io_bound(self, metrics: PerformanceMetrics):
        """Optimize I/O-bound operations"""
        logger.info(f"Applying I/O-bound optimizations for {metrics.operation}")
        
        # Increase thread pool for I/O operations
        if self.resource_pool.current_workers < self.resource_pool.max_workers:
            await self.resource_pool._scale_up()
        
        # Implement prefetching strategies
        await self._enable_prefetching()
    
    async def _optimize_memory_bound(self, metrics: PerformanceMetrics):
        """Optimize memory-bound operations"""
        logger.info(f"Applying memory-bound optimizations for {metrics.operation}")
        
        # Reduce cache size to free memory
        self.cache.max_size = max(100, self.cache.max_size // 2)
        
        # Trigger garbage collection
        import gc
        gc.collect()
    
    async def _enable_prefetching(self):
        """Enable intelligent prefetching based on access patterns"""
        # Analyze cache access patterns and prefetch likely next items
        for key, pattern in self.cache._access_patterns.items():
            if len(pattern) > 5:
                # Simple pattern detection: if regular intervals, prefetch
                intervals = [pattern[i] - pattern[i-1] for i in range(1, len(pattern))]
                avg_interval = sum(intervals) / len(intervals)
                
                if avg_interval > 0 and all(abs(interval - avg_interval) / avg_interval < 0.5 for interval in intervals[-5:]):
                    # Predictable pattern detected
                    logger.debug(f"Predictable access pattern for {key}: {avg_interval:.1f}s interval")
    
    async def optimize_environment_creation(self):
        """Optimize environment creation performance"""
        @self.performance_monitor("environment_creation_batch")
        async def create_environments_batch(n_envs: int = 10):
            """Create multiple environments in parallel"""
            import lunar_habitat_rl
            
            async def create_single_env():
                return lunar_habitat_rl.make_lunar_env()
            
            # Create environments concurrently
            tasks = [
                self.resource_pool.execute_async(create_single_env)
                for _ in range(n_envs)
            ]
            
            environments = await asyncio.gather(*tasks, return_exceptions=True)
            successful_envs = [env for env in environments if not isinstance(env, Exception)]
            
            return successful_envs
        
        # Test parallel environment creation
        logger.info("Testing parallel environment creation optimization")
        start_time = time.time()
        
        envs = await create_environments_batch(5)
        
        creation_time = time.time() - start_time
        logger.info(f"Created {len(envs)} environments in {creation_time:.2f}s ({creation_time/len(envs):.3f}s per env)")
        
        return envs
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {"status": "No performance data collected"}
        
        # Analyze metrics
        operations = {}
        for metric in self.metrics_history:
            op = metric.operation
            if op not in operations:
                operations[op] = {
                    'count': 0,
                    'total_time': 0,
                    'avg_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0,
                    'total_memory': 0,
                    'avg_memory': 0
                }
            
            ops_data = operations[op]
            ops_data['count'] += 1
            ops_data['total_time'] += metric.execution_time
            ops_data['min_time'] = min(ops_data['min_time'], metric.execution_time)
            ops_data['max_time'] = max(ops_data['max_time'], metric.execution_time)
            ops_data['total_memory'] += abs(metric.memory_delta)
        
        # Calculate averages
        for op_data in operations.values():
            op_data['avg_time'] = op_data['total_time'] / op_data['count']
            op_data['avg_memory'] = op_data['total_memory'] / op_data['count']
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "total_operations": len(self.metrics_history),
            "operations_summary": operations,
            "resource_pool_status": self.resource_pool.get_status(),
            "cache_statistics": self.cache.get_statistics(),
            "system_status": {
                "cpu_count": multiprocessing.cpu_count(),
                "memory_total": psutil.virtual_memory().total // (1024**2),  # MB
                "memory_available": psutil.virtual_memory().available // (1024**2),  # MB
                "memory_percent": psutil.virtual_memory().percent
            }
        }

class LoadBalancer:
    """Advanced load balancing for distributed processing"""
    
    def __init__(self, nodes: List[str] = None):
        self.nodes = nodes or ['localhost']
        self.node_stats: Dict[str, Dict[str, Any]] = {
            node: {
                'active_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'avg_response_time': 0.0,
                'last_health_check': 0,
                'healthy': True
            }
            for node in self.nodes
        }
        self._selection_strategy = 'least_connections'
        
    def select_node(self) -> str:
        """Select optimal node for task execution"""
        healthy_nodes = [node for node, stats in self.node_stats.items() if stats['healthy']]
        
        if not healthy_nodes:
            raise Exception("No healthy nodes available")
        
        if self._selection_strategy == 'least_connections':
            return min(healthy_nodes, key=lambda n: self.node_stats[n]['active_tasks'])
        elif self._selection_strategy == 'round_robin':
            # Simple round-robin (could be improved with state persistence)
            return healthy_nodes[0]
        else:
            return healthy_nodes[0]
    
    async def execute_on_node(self, node: str, task: Callable, *args, **kwargs):
        """Execute task on specific node with load tracking"""
        self.node_stats[node]['active_tasks'] += 1
        start_time = time.time()
        
        try:
            result = await task(*args, **kwargs)
            
            execution_time = time.time() - start_time
            stats = self.node_stats[node]
            
            # Update node statistics
            stats['completed_tasks'] += 1
            if stats['avg_response_time'] == 0:
                stats['avg_response_time'] = execution_time
            else:
                stats['avg_response_time'] = 0.9 * stats['avg_response_time'] + 0.1 * execution_time
            
            return result
            
        except Exception as e:
            self.node_stats[node]['failed_tasks'] += 1
            raise
        finally:
            self.node_stats[node]['active_tasks'] -= 1

async def main():
    """Main execution for Generation 3 scaling optimization"""
    print("⚡ GENERATION 3: SCALE OPTIMIZATION SUITE")
    print("=" * 60)
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer()
    
    try:
        # Test parallel environment creation optimization
        print("🚀 Testing Parallel Environment Creation...")
        envs = await optimizer.optimize_environment_creation()
        print(f"✅ Successfully created {len(envs)} environments in parallel")
        
        # Test caching performance
        print("\n💾 Testing Intelligent Caching...")
        for i in range(20):
            key = f"test_key_{i % 5}"  # Create some cache hits
            cached_value = optimizer.cache.get(key)
            if cached_value is None:
                optimizer.cache.set(key, f"value_{i}")
        
        cache_stats = optimizer.cache.get_statistics()
        print(f"✅ Cache utilization: {cache_stats['utilization']:.1f}%")
        print(f"✅ Total patterns tracked: {cache_stats['patterns_tracked']}")
        
        # Test resource pool scaling
        print("\n⚖️ Testing Resource Pool Auto-scaling...")
        pool_status = optimizer.resource_pool.get_status()
        print(f"✅ Current workers: {pool_status['current_workers']}")
        print(f"✅ Success rate: {pool_status['success_rate']:.1f}%")
        
        # Generate performance report
        print("\n📊 Generating Performance Report...")
        report = optimizer.generate_performance_report()
        
        # Save report
        report_file = Path("performance_optimization_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Performance report saved: {report_file}")
        print(f"📈 Total operations tracked: {report.get('total_operations', 0)}")
        
        # System status
        sys_status = report.get('system_status', {})
        print(f"🖥️ System: {sys_status.get('cpu_count', 'N/A')} CPUs, {sys_status.get('memory_available', 'N/A')}MB available memory")
        
        return report
        
    except Exception as e:
        logger.critical(f"Critical failure in scale optimization: {str(e)}")
        raise
    
    finally:
        # Cleanup resources
        optimizer.resource_pool.cleanup()

if __name__ == "__main__":
    asyncio.run(main())