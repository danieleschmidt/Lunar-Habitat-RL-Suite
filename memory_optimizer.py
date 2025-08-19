
"""
Memory optimization utilities for efficient resource usage.
"""

import gc
import sys
from typing import Any, Dict, List
import weakref

class MemoryPool:
    """Object pool for memory efficiency."""
    
    def __init__(self, factory, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: List[Any] = []
        self.in_use = set()
    
    def acquire(self):
        """Get object from pool."""
        if self.pool:
            obj = self.pool.pop()
        else:
            obj = self.factory()
        
        self.in_use.add(id(obj))
        return obj
    
    def release(self, obj):
        """Return object to pool."""
        obj_id = id(obj)
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
            if len(self.pool) < self.max_size:
                # Reset object state if needed
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

class LazyLoader:
    """Lazy loading for memory efficiency."""
    
    def __init__(self, loader_func):
        self.loader_func = loader_func
        self._loaded = None
        self._is_loaded = False
    
    def __call__(self):
        if not self._is_loaded:
            self._loaded = self.loader_func()
            self._is_loaded = True
        return self._loaded
    
    def clear(self):
        """Clear loaded data."""
        self._loaded = None
        self._is_loaded = False

def optimize_memory():
    """Force garbage collection and return memory stats."""
    gc.collect()
    return {
        "gc_collected": gc.collect(),
        "memory_usage": sys.getsizeof(sys.modules)
    }
