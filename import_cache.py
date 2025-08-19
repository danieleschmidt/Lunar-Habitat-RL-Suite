
"""
Import cache system for faster module loading.
"""

import sys
import time
from typing import Dict, Any

_import_cache: Dict[str, Any] = {}
_import_times: Dict[str, float] = {}

def cached_import(module_name: str):
    """Import with caching for performance."""
    if module_name in _import_cache:
        return _import_cache[module_name]
    
    start_time = time.time()
    module = __import__(module_name)
    import_time = time.time() - start_time
    
    _import_cache[module_name] = module
    _import_times[module_name] = import_time
    
    return module

def get_import_stats():
    """Get import performance statistics."""
    return _import_times.copy()

def clear_cache():
    """Clear import cache."""
    _import_cache.clear()
    _import_times.clear()
