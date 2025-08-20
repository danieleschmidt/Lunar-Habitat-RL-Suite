"""
Concurrent Processing and Parallel Execution System - Generation 3
High-performance concurrent execution for NASA space mission operations.
"""

import asyncio
import threading
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import time
import weakref
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Coroutine
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import psutil
import numpy as np
import torch

from ..utils.logging import get_logger

logger = get_logger("concurrent_execution")


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent execution."""
    
    # Thread pool configuration
    max_thread_workers: int = 16
    thread_pool_timeout: float = 30.0
    adaptive_thread_scaling: bool = True
    
    # Process pool configuration  
    max_process_workers: int = 8
    process_pool_timeout: float = 60.0
    use_process_pool: bool = False
    
    # Async configuration
    max_async_tasks: int = 100
    async_timeout: float = 30.0
    enable_async_scheduling: bool = True
    
    # Queue configuration
    max_queue_size: int = 1000
    queue_timeout: float = 5.0
    priority_queue: bool = True
    
    # Performance optimization
    enable_load_balancing: bool = True
    work_stealing: bool = True
    affinity_scheduling: bool = True
    
    # Resource monitoring
    cpu_utilization_threshold: float = 0.8
    memory_threshold_gb: float = 8.0
    auto_scale_interval: float = 10.0


class TaskPriority:
    """Task priority levels for scheduling."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Task:
    """Concurrent task representation."""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = TaskPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    exception: Optional[Exception] = None
    status: str = "pending"  # pending, running, completed, failed


class WorkerPool(ABC):
    """Abstract base class for worker pools."""
    
    @abstractmethod
    def submit(self, task: Task) -> concurrent.futures.Future:
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True):
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class AdaptiveThreadPool(WorkerPool):
    """Adaptive thread pool with dynamic scaling."""
    
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self.min_workers = max(1, config.max_thread_workers // 4)
        self.max_workers = config.max_thread_workers
        self.current_workers = self.min_workers
        
        self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        self.pending_tasks = queue.PriorityQueue(maxsize=config.max_queue_size)
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
        # Performance monitoring
        self.task_times = deque(maxlen=100)
        self.utilization_history = deque(maxlen=50)
        self.last_scale_time = time.time()
        
        # Worker management
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Start management thread
        self.management_thread = threading.Thread(target=self._manage_pool, daemon=True)
        self.management_thread.start()
        
        logger.info(f"Adaptive thread pool initialized with {self.current_workers} workers")
    
    def submit(self, task: Task) -> concurrent.futures.Future:
        """Submit task to thread pool."""
        with self.lock:
            if self.shutdown_event.is_set():
                raise RuntimeError("Thread pool is shutting down")
            
            # Submit to thread pool executor
            future = self.executor.submit(self._execute_task, task)
            self.active_tasks[task.id] = {
                'task': task,
                'future': future,
                'start_time': time.time()
            }
            
            return future
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        start_time = time.time()
        task.status = "running"
        
        try:
            if task.timeout:
                # Use signal for timeout (Unix only) or simple timeout
                result = task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            task.result = result
            task.status = "completed"
            
            execution_time = time.time() - start_time
            self.task_times.append(execution_time)
            
            return result
            
        except Exception as e:
            task.exception = e
            task.status = "failed"
            logger.error(f"Task {task.id} failed: {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count})")
                return self._execute_task(task)
            
            raise e
        finally:
            # Cleanup
            with self.lock:
                if task.id in self.active_tasks:
                    self.completed_tasks.append(self.active_tasks.pop(task.id))
    
    def _manage_pool(self):
        """Manage pool size based on load."""
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Calculate utilization
                with self.lock:
                    active_count = len(self.active_tasks)
                    utilization = active_count / self.current_workers
                    self.utilization_history.append(utilization)
                
                # Adaptive scaling
                if (self.config.adaptive_thread_scaling and 
                    current_time - self.last_scale_time > self.config.auto_scale_interval):
                    
                    self._adapt_pool_size(utilization)
                    self.last_scale_time = current_time
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Pool management error: {e}")
                time.sleep(5.0)
    
    def _adapt_pool_size(self, current_utilization: float):
        """Adapt pool size based on utilization."""
        if len(self.utilization_history) < 5:
            return
        
        avg_utilization = np.mean(list(self.utilization_history)[-10:])
        
        # Scale up if utilization is high
        if avg_utilization > 0.8 and self.current_workers < self.max_workers:
            new_size = min(self.max_workers, self.current_workers + 2)
            self._resize_pool(new_size)
            
        # Scale down if utilization is low
        elif avg_utilization < 0.3 and self.current_workers > self.min_workers:
            new_size = max(self.min_workers, self.current_workers - 1)
            self._resize_pool(new_size)
    
    def _resize_pool(self, new_size: int):
        """Resize the thread pool."""
        if new_size == self.current_workers:
            return
        
        with self.lock:
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_size)
            self.current_workers = new_size
            
            # Shutdown old executor
            old_executor.shutdown(wait=False)
            
            logger.info(f"Resized thread pool to {new_size} workers")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thread pool statistics."""
        with self.lock:
            avg_task_time = np.mean(self.task_times) if self.task_times else 0
            current_utilization = len(self.active_tasks) / self.current_workers
            
            return {
                'current_workers': self.current_workers,
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'avg_task_time': avg_task_time,
                'current_utilization': current_utilization,
                'avg_utilization': np.mean(self.utilization_history) if self.utilization_history else 0
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        self.shutdown_event.set()
        self.executor.shutdown(wait=wait)
        if self.management_thread.is_alive():
            self.management_thread.join(timeout=5.0)


class AsyncTaskScheduler:
    """Asynchronous task scheduler with priority queues."""
    
    def __init__(self, config: ConcurrencyConfig):
        self.config = config
        self.task_queue = asyncio.PriorityQueue(maxsize=config.max_queue_size)
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.semaphore = asyncio.Semaphore(config.max_async_tasks)
        
        # Scheduling state
        self.running = False
        self.scheduler_task = None
        
        # Performance tracking
        self.task_metrics = defaultdict(list)
        
        logger.info("Async task scheduler initialized")
    
    async def submit(self, task: Task) -> Any:
        """Submit async task."""
        await self.task_queue.put((task.priority, time.time(), task))
        
        # Start scheduler if not running
        if not self.running:
            await self.start_scheduler()
        
        # Wait for task completion (simplified - in practice would use futures)
        while task.status == "pending":
            await asyncio.sleep(0.01)
        
        if task.status == "completed":
            return task.result
        elif task.status == "failed":
            raise task.exception
    
    async def start_scheduler(self):
        """Start the async task scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._schedule_tasks())
        logger.info("Async scheduler started")
    
    async def _schedule_tasks(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Get next task
                try:
                    priority, timestamp, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute task with semaphore
                async with self.semaphore:
                    asyncio.create_task(self._execute_async_task(task))
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_async_task(self, task: Task):
        """Execute async task."""
        start_time = time.time()
        task.status = "running"
        
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, task.func, *task.args)
            
            task.result = result
            task.status = "completed"
            
            # Record metrics
            execution_time = time.time() - start_time
            self.task_metrics[task.func.__name__].append(execution_time)
            
        except Exception as e:
            task.exception = e
            task.status = "failed"
            logger.error(f"Async task {task.id} failed: {e}")
        
        finally:
            self.completed_tasks.append(task)
    
    async def stop_scheduler(self):
        """Stop the async scheduler."""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Async scheduler stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async scheduler statistics."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'queue_size': self.task_queue.qsize(),
            'max_concurrent_tasks': self.config.max_async_tasks,
            'running': self.running,
            'task_metrics': dict(self.task_metrics)
        }


class LoadBalancer:
    """Load balancer for distributing tasks across workers."""
    
    def __init__(self, worker_pools: List[WorkerPool], strategy: str = "round_robin"):
        self.worker_pools = worker_pools
        self.strategy = strategy
        self.current_index = 0
        self.load_metrics = defaultdict(list)
        self.lock = threading.Lock()
        
        # Load tracking
        self.pool_loads = [0.0] * len(worker_pools)
        
        logger.info(f"Load balancer initialized with {len(worker_pools)} pools")
    
    def submit_task(self, task: Task) -> concurrent.futures.Future:
        """Submit task with load balancing."""
        with self.lock:
            pool_index = self._select_pool(task)
            pool = self.worker_pools[pool_index]
            
            # Update load metrics
            self.pool_loads[pool_index] += 1
            
            try:
                future = pool.submit(task)
                
                # Callback to update load when task completes
                def update_load(f):
                    with self.lock:
                        self.pool_loads[pool_index] = max(0, self.pool_loads[pool_index] - 1)
                
                future.add_done_callback(update_load)
                
                return future
                
            except Exception as e:
                self.pool_loads[pool_index] = max(0, self.pool_loads[pool_index] - 1)
                raise e
    
    def _select_pool(self, task: Task) -> int:
        """Select worker pool based on load balancing strategy."""
        if self.strategy == "round_robin":
            index = self.current_index
            self.current_index = (self.current_index + 1) % len(self.worker_pools)
            return index
            
        elif self.strategy == "least_loaded":
            return int(np.argmin(self.pool_loads))
            
        elif self.strategy == "random":
            return np.random.randint(0, len(self.worker_pools))
            
        elif self.strategy == "priority_aware":
            # High priority tasks go to least loaded pool
            if task.priority <= TaskPriority.HIGH:
                return int(np.argmin(self.pool_loads))
            else:
                return self.current_index % len(self.worker_pools)
        
        else:
            return 0
    
    def get_load_distribution(self) -> List[float]:
        """Get current load distribution across pools."""
        with self.lock:
            return self.pool_loads.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'strategy': self.strategy,
            'num_pools': len(self.worker_pools),
            'load_distribution': self.get_load_distribution(),
            'total_load': sum(self.pool_loads)
        }


class ConcurrentExecutionManager:
    """Main manager for concurrent execution."""
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        self.config = config or ConcurrencyConfig()
        
        # Initialize worker pools
        self.thread_pool = AdaptiveThreadPool(self.config)
        self.process_pool = None
        if self.config.use_process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_process_workers)
        
        # Initialize async scheduler
        self.async_scheduler = AsyncTaskScheduler(self.config)
        
        # Initialize load balancer
        worker_pools = [self.thread_pool]
        if self.process_pool:
            worker_pools.append(self.process_pool)
        
        self.load_balancer = LoadBalancer(worker_pools, strategy="least_loaded")
        
        # Task tracking
        self.submitted_tasks = {}
        self.task_counter = 0
        
        # Performance monitoring
        self.execution_stats = defaultdict(list)
        
        logger.info("Concurrent execution manager initialized")
    
    def submit_sync_task(self, func: Callable, *args, priority: int = TaskPriority.MEDIUM,
                        timeout: Optional[float] = None, **kwargs) -> concurrent.futures.Future:
        """Submit synchronous task."""
        task_id = f"sync_task_{self.task_counter}"
        self.task_counter += 1
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        self.submitted_tasks[task_id] = task
        
        # Submit to load balancer
        future = self.load_balancer.submit_task(task)
        
        return future
    
    async def submit_async_task(self, func: Callable, *args, priority: int = TaskPriority.MEDIUM,
                               **kwargs) -> Any:
        """Submit asynchronous task."""
        task_id = f"async_task_{self.task_counter}"
        self.task_counter += 1
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        self.submitted_tasks[task_id] = task
        
        result = await self.async_scheduler.submit(task)
        return result
    
    def submit_batch_tasks(self, func: Callable, batch_data: List[Any],
                          priority: int = TaskPriority.MEDIUM,
                          use_processes: bool = False) -> List[concurrent.futures.Future]:
        """Submit batch of tasks."""
        futures = []
        
        for i, data in enumerate(batch_data):
            if use_processes and self.process_pool:
                future = self.process_pool.submit(func, data)
            else:
                future = self.submit_sync_task(func, data, priority=priority)
            futures.append(future)
        
        return futures
    
    def wait_for_tasks(self, futures: List[concurrent.futures.Future],
                      timeout: Optional[float] = None) -> List[Any]:
        """Wait for batch of tasks to complete."""
        results = []
        
        try:
            for future in as_completed(futures, timeout=timeout):
                result = future.result()
                results.append(result)
        except concurrent.futures.TimeoutError:
            logger.warning("Some tasks timed out")
        
        return results
    
    async def process_pipeline(self, stages: List[Callable], data: Any,
                              stage_priorities: Optional[List[int]] = None) -> Any:
        """Process data through a pipeline of async stages."""
        current_data = data
        
        for i, stage_func in enumerate(stages):
            priority = (stage_priorities[i] if stage_priorities 
                       else TaskPriority.MEDIUM)
            
            current_data = await self.submit_async_task(
                stage_func, current_data, priority=priority
            )
        
        return current_data
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        return {
            'thread_pool': self.thread_pool.get_stats(),
            'async_scheduler': self.async_scheduler.get_stats(),
            'load_balancer': self.load_balancer.get_stats(),
            'submitted_tasks': len(self.submitted_tasks),
            'config': {
                'max_thread_workers': self.config.max_thread_workers,
                'max_async_tasks': self.config.max_async_tasks,
                'use_process_pool': self.config.use_process_pool
            }
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Optimize configuration based on current performance."""
        stats = self.get_comprehensive_stats()
        suggestions = []
        
        # Thread pool optimization
        thread_stats = stats['thread_pool']
        if thread_stats['current_utilization'] > 0.9:
            suggestions.append("Consider increasing max_thread_workers")
        
        # Async optimization
        async_stats = stats['async_scheduler']
        if async_stats['queue_size'] > self.config.max_queue_size * 0.8:
            suggestions.append("Consider increasing max_async_tasks")
        
        # Load balancing optimization
        load_stats = stats['load_balancer']
        load_distribution = load_stats['load_distribution']
        if load_distribution and np.std(load_distribution) > np.mean(load_distribution):
            suggestions.append("Load imbalance detected - consider tuning load balancer")
        
        return {
            'current_stats': stats,
            'optimization_suggestions': suggestions
        }
    
    def cleanup(self):
        """Cleanup execution manager resources."""
        self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Stop async scheduler
        asyncio.create_task(self.async_scheduler.stop_scheduler())
        
        logger.info("Concurrent execution manager cleanup completed")


# Utility functions and decorators

def concurrent_task(priority: int = TaskPriority.MEDIUM, 
                   timeout: Optional[float] = None):
    """Decorator for marking functions as concurrent tasks."""
    def decorator(func: Callable) -> Callable:
        func._concurrent_task = True
        func._priority = priority
        func._timeout = timeout
        return func
    return decorator


def parallel_map(func: Callable, data: List[Any], 
                max_workers: int = None) -> List[Any]:
    """Parallel map function using thread pool."""
    max_workers = max_workers or min(len(data), mp.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, data))
    
    return results


async def async_parallel_map(func: Callable, data: List[Any],
                           max_concurrent: int = 10) -> List[Any]:
    """Async parallel map function."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_task(item):
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, item)
    
    tasks = [bounded_task(item) for item in data]
    results = await asyncio.gather(*tasks)
    
    return results


def demo_concurrent_execution():
    """Demonstrate concurrent execution capabilities."""
    print("🚀 Concurrent Execution Demo")
    print("=" * 40)
    
    config = ConcurrencyConfig(
        max_thread_workers=8,
        max_async_tasks=20,
        use_process_pool=False
    )
    
    manager = ConcurrentExecutionManager(config)
    
    # Demo sync tasks
    def cpu_intensive_task(n: int) -> int:
        """Simulate CPU-intensive task."""
        total = 0
        for i in range(n * 1000):
            total += i ** 2
        return total
    
    print("📊 Testing Synchronous Tasks")
    start_time = time.time()
    
    # Submit batch of tasks
    futures = manager.submit_batch_tasks(cpu_intensive_task, list(range(1, 21)))
    results = manager.wait_for_tasks(futures, timeout=30.0)
    
    sync_time = time.time() - start_time
    print(f"✅ Completed {len(results)} sync tasks in {sync_time:.2f} seconds")
    
    # Demo async tasks
    async def demo_async():
        async def io_intensive_task(duration: float) -> str:
            await asyncio.sleep(duration)
            return f"Task completed after {duration}s"
        
        print("\n🔄 Testing Asynchronous Tasks")
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = manager.submit_async_task(io_intensive_task, 0.1 * (i + 1))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        async_time = time.time() - start_time
        
        print(f"✅ Completed {len(results)} async tasks in {async_time:.2f} seconds")
        
        return results
    
    # Run async demo
    asyncio.run(demo_async())
    
    # Show stats
    stats = manager.get_comprehensive_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"  Thread pool workers: {stats['thread_pool']['current_workers']}")
    print(f"  Thread utilization: {stats['thread_pool']['current_utilization']:.2f}")
    print(f"  Completed tasks: {stats['thread_pool']['completed_tasks']}")
    
    # Optimization suggestions
    optimization = manager.optimize_configuration()
    if optimization['optimization_suggestions']:
        print(f"\n💡 Optimization Suggestions:")
        for suggestion in optimization['optimization_suggestions']:
            print(f"  - {suggestion}")
    
    manager.cleanup()
    print(f"\n✅ Concurrent execution demo completed!")


if __name__ == "__main__":
    demo_concurrent_execution()