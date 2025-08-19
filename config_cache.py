
"""
Configuration caching system for improved performance.
"""

import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

_config_cache: Dict[str, Any] = {}
_cache_timestamps: Dict[str, float] = {}

def load_cached_config(config_name: str, config_factory) -> Any:
    """Load configuration with caching."""
    cache_key = f"{config_name}_{config_factory.__name__}"
    
    # Check if cached and still valid (5 minute cache)
    if cache_key in _config_cache:
        cached_time = _cache_timestamps.get(cache_key, 0)
        if time.time() - cached_time < 300:  # 5 minutes
            return _config_cache[cache_key]
    
    # Create new config
    config = config_factory()
    _config_cache[cache_key] = config
    _cache_timestamps[cache_key] = time.time()
    
    return config

def clear_config_cache():
    """Clear configuration cache."""
    _config_cache.clear()
    _cache_timestamps.clear()
