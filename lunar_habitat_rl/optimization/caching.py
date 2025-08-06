"""Advanced caching system for performance optimization."""

import hashlib
import pickle
import time
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
import numpy as np
import json

from ..utils.logging import get_logger

logger = get_logger("caching")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """Thread-safe LRU cache with size limits and TTL."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds and time.time() - entry.created_at > self.ttl_seconds:
                self._remove_entry(key)
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            return entry.value
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if updating
            if key in self._cache:
                self._remove_entry(key)
            
            # Check if single item is too large
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Cache item too large: {size_bytes} bytes > {self.max_memory_bytes}")
                return False
            
            # Make space if needed
            while (len(self._cache) >= self.max_size or 
                   self._current_memory + size_bytes > self.max_memory_bytes):
                if not self._cache:
                    break
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._current_memory += size_bytes
            
            return True
    
    def _remove_entry(self, key: str):
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
    
    def _evict_lru(self):
        """Evict least recently used item.""" 
        if self._cache:
            key, _ = self._cache.popitem(last=False)  # Remove first (least recently used)
            self._current_memory -= self._cache.get(key, CacheEntry("", None, 0, 0)).size_bytes
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (str, bytes)):
                return len(obj.encode('utf-8')) if isinstance(obj, str) else len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            else:
                # Fallback to pickle size
                return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default size estimate
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': self._calculate_hit_rate(),
                'entries': len(self._cache)
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self._cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        return total_accesses / len(self._cache) if self._cache else 0.0


class PersistentCache:
    """Persistent cache using SQLite for long-term storage."""
    
    def __init__(self, db_path: str = "./cache/persistent_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._lock = threading.RLock()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    access_count INTEGER DEFAULT 0,
                    size_bytes INTEGER,
                    ttl_seconds REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from persistent cache."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT value, created_at, ttl_seconds FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    value_blob, created_at, ttl_seconds = row
                    
                    # Check TTL
                    if ttl_seconds and time.time() - created_at > ttl_seconds:
                        self.delete(key)
                        return None
                    
                    # Update access info
                    conn.execute(
                        "UPDATE cache_entries SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                        (time.time(), key)
                    )
                    
                    # Deserialize value
                    return pickle.loads(value_blob)
                    
            except Exception as e:
                logger.error(f"Error getting cache entry {key}: {e}")
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Put item in persistent cache."""
        with self._lock:
            try:
                # Serialize value
                value_blob = pickle.dumps(value)
                size_bytes = len(value_blob)
                current_time = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, created_at, accessed_at, access_count, size_bytes, ttl_seconds)
                        VALUES (?, ?, ?, ?, 0, ?, ?)
                        """,
                        (key, value_blob, current_time, current_time, size_bytes, ttl_seconds)
                    )
                    
                return True
                
            except Exception as e:
                logger.error(f"Error putting cache entry {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    return cursor.rowcount > 0
            except Exception as e:
                logger.error(f"Error deleting cache entry {key}: {e}")
                return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            try:
                current_time = time.time()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM cache_entries WHERE ttl_seconds IS NOT NULL AND ? - created_at > ttl_seconds",
                        (current_time,)
                    )
                    return cursor.rowcount
            except Exception as e:
                logger.error(f"Error cleaning up expired entries: {e}")
                return 0
    
    def vacuum(self):
        """Optimize database storage."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("VACUUM")
            except Exception as e:
                logger.error(f"Error vacuuming cache database: {e}")


class SimulationCache:
    """Specialized cache for physics simulation results."""
    
    def __init__(self, memory_cache_size: int = 500, persistent_cache: bool = True):
        self.memory_cache = LRUCache(max_size=memory_cache_size, max_memory_mb=200, ttl_seconds=3600)
        self.persistent_cache = PersistentCache() if persistent_cache else None
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_simulation_result(self, 
                            simulator_type: str,
                            initial_state: Dict[str, Any],
                            parameters: Dict[str, Any],
                            timestep: float,
                            duration: float) -> Optional[Dict[str, Any]]:
        """Get cached simulation result."""
        
        cache_key = self._generate_simulation_key(
            simulator_type, initial_state, parameters, timestep, duration
        )
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            self._cache_hits += 1
            return result
        
        # Try persistent cache
        if self.persistent_cache:
            result = self.persistent_cache.get(cache_key)
            if result is not None:
                # Promote to memory cache
                self.memory_cache.put(cache_key, result)
                self._cache_hits += 1
                return result
        
        self._cache_misses += 1
        return None
    
    def put_simulation_result(self,
                             simulator_type: str,
                             initial_state: Dict[str, Any],
                             parameters: Dict[str, Any],
                             timestep: float,
                             duration: float,
                             result: Dict[str, Any]) -> bool:
        """Cache simulation result."""
        
        cache_key = self._generate_simulation_key(
            simulator_type, initial_state, parameters, timestep, duration
        )
        
        # Store in memory cache
        success = self.memory_cache.put(cache_key, result)
        
        # Store in persistent cache
        if self.persistent_cache:
            self.persistent_cache.put(cache_key, result, ttl_seconds=24*3600)  # 24 hours
        
        return success
    
    def _generate_simulation_key(self,
                                simulator_type: str,
                                initial_state: Dict[str, Any],
                                parameters: Dict[str, Any],
                                timestep: float,
                                duration: float) -> str:
        """Generate deterministic cache key for simulation."""
        
        # Create hashable representation
        state_str = json.dumps(initial_state, sort_keys=True, default=str)
        params_str = json.dumps(parameters, sort_keys=True, default=str)
        
        key_data = f"{simulator_type}:{state_str}:{params_str}:{timestep}:{duration}"
        
        # Hash for fixed-length key
        return hashlib.sha256(key_data.encode('utf-8')).hexdigest()
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self._cache_hits + self._cache_misses
        return self._cache_hits / total_requests if total_requests > 0 else 0.0
    
    def clear_all(self):
        """Clear all caches."""
        self.memory_cache.clear()
        if self.persistent_cache:
            with sqlite3.connect(self.persistent_cache.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")


class StateCache:
    """Cache for environment states and transitions."""
    
    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self.episode_cache = LRUCache(max_size=max_episodes, max_memory_mb=50)
        self.transition_cache = LRUCache(max_size=max_episodes * 1000, max_memory_mb=100)
    
    def cache_episode(self, episode_id: str, states: List[np.ndarray], actions: List[np.ndarray], 
                     rewards: List[float], infos: List[Dict[str, Any]]):
        """Cache complete episode data."""
        
        episode_data = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'infos': infos,
            'episode_length': len(states)
        }
        
        self.episode_cache.put(episode_id, episode_data)
    
    def get_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached episode."""
        return self.episode_cache.get(episode_id)
    
    def cache_transition(self, state: np.ndarray, action: np.ndarray, 
                        next_state: np.ndarray, reward: float, done: bool):
        """Cache state transition."""
        
        # Generate key from state-action pair
        state_action_key = hashlib.sha256(
            np.concatenate([state, action]).tobytes()
        ).hexdigest()
        
        transition = {
            'next_state': next_state,
            'reward': reward,
            'done': done
        }
        
        self.transition_cache.put(state_action_key, transition)
    
    def get_transition(self, state: np.ndarray, action: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get cached transition result."""
        
        state_action_key = hashlib.sha256(
            np.concatenate([state, action]).tobytes()
        ).hexdigest()
        
        return self.transition_cache.get(state_action_key)


class ResultsCache:
    """Cache for training results and evaluations."""
    
    def __init__(self, cache_dir: str = "./cache/results"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = LRUCache(max_size=100, max_memory_mb=50)
    
    def cache_training_results(self, experiment_id: str, results: Dict[str, Any]):
        """Cache training experiment results."""
        
        # Save to file
        results_file = self.cache_dir / f"{experiment_id}_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error caching training results: {e}")
            return False
        
        # Cache in memory
        self.memory_cache.put(f"training:{experiment_id}", results)
        return True
    
    def get_training_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get cached training results."""
        
        # Try memory cache first
        memory_key = f"training:{experiment_id}"
        results = self.memory_cache.get(memory_key)
        if results:
            return results
        
        # Try file cache
        results_file = self.cache_dir / f"{experiment_id}_results.json"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    
                # Promote to memory cache
                self.memory_cache.put(memory_key, results)
                return results
            except Exception as e:
                logger.error(f"Error loading training results: {e}")
        
        return None
    
    def cache_evaluation_results(self, model_id: str, scenario: str, results: Dict[str, Any]):
        """Cache model evaluation results."""
        
        cache_key = f"eval:{model_id}:{scenario}"
        
        # Save to file
        eval_file = self.cache_dir / f"{model_id}_{scenario}_eval.json"
        
        try:
            with open(eval_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error caching evaluation results: {e}")
            return False
        
        # Cache in memory  
        self.memory_cache.put(cache_key, results)
        return True
    
    def get_evaluation_results(self, model_id: str, scenario: str) -> Optional[Dict[str, Any]]:
        """Get cached evaluation results."""
        
        cache_key = f"eval:{model_id}:{scenario}"
        
        # Try memory cache
        results = self.memory_cache.get(cache_key)
        if results:
            return results
        
        # Try file cache
        eval_file = self.cache_dir / f"{model_id}_{scenario}_eval.json"
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r') as f:
                    results = json.load(f)
                    
                # Promote to memory cache
                self.memory_cache.put(cache_key, results)
                return results
            except Exception as e:
                logger.error(f"Error loading evaluation results: {e}")
        
        return None


def cache_simulation_step(cache: SimulationCache):
    """Decorator to cache simulation step results."""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Extract caching parameters
            if hasattr(self, '__class__'):
                simulator_type = self.__class__.__name__
            else:
                simulator_type = "unknown"
            
            # Try to get from cache
            if len(args) >= 4:
                # Assume args are (dt, external_temp, internal_heat, heating_power, ...)
                cache_key_params = {
                    'args': args[:4],
                    'kwargs': kwargs
                }
                
                cached_result = cache.get_simulation_result(
                    simulator_type=simulator_type,
                    initial_state=getattr(self, '_get_current_state', lambda: {})(),
                    parameters=cache_key_params,
                    timestep=args[0] if args else 60.0,
                    duration=args[0] if args else 60.0
                )
                
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            
            # Cache the result
            if len(args) >= 4:
                cache.put_simulation_result(
                    simulator_type=simulator_type,
                    initial_state=getattr(self, '_get_current_state', lambda: {})(),
                    parameters=cache_key_params,
                    timestep=args[0] if args else 60.0,
                    duration=args[0] if args else 60.0,
                    result=result
                )
            
            return result
        return wrapper
    return decorator