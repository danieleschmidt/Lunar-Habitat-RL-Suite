
"""
Concurrent processing framework for scalable performance.
"""

import threading
import concurrent.futures
import asyncio
import queue
from typing import Any, Callable, List, Optional, Dict
import time

class ConcurrentProcessor:
    """Thread-safe concurrent processing manager."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.task_queue = queue.Queue()
        self.results_cache = {}
        self._lock = threading.Lock()
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task for concurrent execution."""
        return self.executor.submit(func, *args, **kwargs)
    
    def submit_batch(self, func: Callable, args_list: List[tuple]) -> List[concurrent.futures.Future]:
        """Submit batch of tasks."""
        futures = []
        for args in args_list:
            future = self.submit_task(func, *args)
            futures.append(future)
        return futures
    
    def map_concurrent(self, func: Callable, iterable, timeout: Optional[float] = None):
        """Map function over iterable concurrently."""
        return self.executor.map(func, iterable, timeout=timeout)
    
    def cached_execution(self, cache_key: str, func: Callable, *args, **kwargs):
        """Execute with caching for expensive operations."""
        with self._lock:
            if cache_key in self.results_cache:
                return self.results_cache[cache_key]
        
        result = func(*args, **kwargs)
        
        with self._lock:
            self.results_cache[cache_key] = result
        
        return result
    
    def shutdown(self):
        """Shutdown the processor."""
        self.executor.shutdown(wait=True)

class AsyncProcessor:
    """Asynchronous processing for I/O bound tasks."""
    
    def __init__(self):
        self.loop = None
        self.tasks = []
    
    async def process_async(self, coro_func, *args, **kwargs):
        """Process coroutine asynchronously."""
        return await coro_func(*args, **kwargs)
    
    async def batch_async(self, coro_funcs: List[Callable], args_list: List[tuple]):
        """Process batch of coroutines."""
        tasks = []
        for coro_func, args in zip(coro_funcs, args_list):
            task = asyncio.create_task(coro_func(*args))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

# Global processors
_thread_processor = ConcurrentProcessor()
_async_processor = AsyncProcessor()

def get_thread_processor() -> ConcurrentProcessor:
    """Get global thread processor."""
    return _thread_processor

def get_async_processor() -> AsyncProcessor:
    """Get global async processor."""
    return _async_processor
