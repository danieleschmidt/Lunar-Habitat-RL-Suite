"""
Advanced Memory Optimization and Garbage Collection System - Generation 3
Smart memory management for NASA space mission operations.
"""

import gc
import sys
import threading
import time
import psutil
import numpy as np
import weakref
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import json
import pickle
import mmap
import os
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

from ..utils.logging import get_logger

logger = get_logger("memory_optimization")


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    timestamp: float
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    swap_used_gb: float
    process_memory_gb: float
    gc_objects: int
    gc_collections: List[int]  # Collections for each generation
    largest_objects: List[str] = field(default_factory=list)


@dataclass
class GCConfiguration:
    """Garbage collection configuration."""
    enabled: bool = True
    generation_thresholds: Tuple[int, int, int] = (700, 10, 10)
    auto_tune: bool = True
    collection_interval: float = 30.0
    memory_pressure_threshold: float = 0.8
    aggressive_collection_threshold: float = 0.9


class ObjectTracker:
    """Track object creation and memory usage patterns."""
    
    def __init__(self, max_tracked_objects: int = 10000):
        self.max_tracked_objects = max_tracked_objects
        self.tracked_objects = weakref.WeakSet()
        self.object_stats = defaultdict(lambda: {'count': 0, 'total_size': 0})
        self.creation_times = {}
        self.lock = threading.RLock()
        
        # Memory growth tracking
        self.memory_snapshots = deque(maxlen=1000)
        self.leak_suspects = defaultdict(list)
    
    def track_object(self, obj: Any, obj_type: str = None):
        """Track an object for memory analysis."""
        with self.lock:
            if len(self.tracked_objects) >= self.max_tracked_objects:
                return  # Skip tracking if at capacity
            
            obj_id = id(obj)
            obj_type = obj_type or type(obj).__name__
            
            try:
                self.tracked_objects.add(obj)
                self.creation_times[obj_id] = time.time()
                
                # Estimate object size
                obj_size = self._estimate_object_size(obj)
                
                self.object_stats[obj_type]['count'] += 1
                self.object_stats[obj_type]['total_size'] += obj_size
                
            except Exception as e:
                logger.warning(f"Error tracking object: {e}")
    
    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_object_size(item) for item in obj[:100])  # Sample first 100
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_object_size(k) + self._estimate_object_size(v)
                    for k, v in list(obj.items())[:100]  # Sample first 100
                )
            elif hasattr(obj, '__len__'):
                return len(obj) * 8  # Rough estimate
            else:
                return sys.getsizeof(obj)
        except:
            return 64  # Default estimate
    
    def detect_memory_leaks(self) -> List[str]:
        """Detect potential memory leaks."""
        with self.lock:
            leaks = []
            current_time = time.time()
            
            # Look for object types with consistently growing counts
            for obj_type, stats in self.object_stats.items():
                if stats['count'] > 100:  # Only consider types with significant counts
                    # Check if objects of this type are old (potential leak)
                    old_objects = [
                        obj_id for obj_id, creation_time in self.creation_times.items()
                        if current_time - creation_time > 300 and  # 5 minutes old
                        any(type(obj).__name__ == obj_type for obj in self.tracked_objects)
                    ]
                    
                    if len(old_objects) > stats['count'] * 0.5:  # More than 50% are old
                        leak_info = (
                            f"Potential leak: {obj_type} has {stats['count']} objects, "
                            f"{len(old_objects)} are > 5 minutes old"
                        )
                        leaks.append(leak_info)
                        self.leak_suspects[obj_type].append({
                            'timestamp': current_time,
                            'count': stats['count'],
                            'old_count': len(old_objects)
                        })
            
            return leaks
    
    def get_memory_hotspots(self) -> List[Tuple[str, int, int]]:
        """Get object types consuming the most memory."""
        with self.lock:
            hotspots = []
            for obj_type, stats in self.object_stats.items():
                if stats['total_size'] > 1024 * 1024:  # > 1MB
                    hotspots.append((obj_type, stats['count'], stats['total_size']))
            
            return sorted(hotspots, key=lambda x: x[2], reverse=True)
    
    def cleanup_tracking(self):
        """Clean up tracking data."""
        with self.lock:
            # Remove tracking data for objects that no longer exist
            current_ids = {id(obj) for obj in self.tracked_objects}
            dead_ids = set(self.creation_times.keys()) - current_ids
            
            for dead_id in dead_ids:
                del self.creation_times[dead_id]
            
            # Reset stats periodically
            if len(self.object_stats) > 1000:
                self.object_stats.clear()


class SmartMemoryPool:
    """Smart memory pool with adaptive sizing and compression."""
    
    def __init__(self, initial_size_mb: int = 100, max_size_mb: int = 500):
        self.initial_size_mb = initial_size_mb
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        
        # Memory pools by size category
        self.pools = {
            'small': deque(),      # < 1KB
            'medium': deque(),     # 1KB - 1MB  
            'large': deque(),      # > 1MB
        }
        
        # Pool statistics
        self.allocation_stats = defaultdict(int)
        self.reuse_stats = defaultdict(int)
        self.fragmentation_stats = deque(maxlen=1000)
        
        # Memory mapping for large objects
        self.memory_maps = {}
        
        self.lock = threading.RLock()
        
        logger.info(f"Smart memory pool initialized: {initial_size_mb}MB initial, {max_size_mb}MB max")
    
    def allocate(self, size_bytes: int, object_type: str = "generic") -> Optional[memoryview]:
        """Allocate memory from pool."""
        with self.lock:
            pool_category = self._get_pool_category(size_bytes)
            
            # Try to reuse existing allocation
            if self.pools[pool_category]:
                existing_block = self.pools[pool_category].popleft()
                if len(existing_block) >= size_bytes:
                    self.reuse_stats[pool_category] += 1
                    return existing_block[:size_bytes]
                else:
                    # Return to pool if too small
                    self.pools[pool_category].append(existing_block)
            
            # Allocate new block
            if self.current_size_mb >= self.max_size_mb:
                self._garbage_collect_pool()
            
            try:
                if size_bytes > 10 * 1024 * 1024:  # > 10MB, use memory mapping
                    temp_file = self._create_temp_file(size_bytes)
                    memory_block = mmap.mmap(temp_file.fileno(), size_bytes)
                    self.memory_maps[id(memory_block)] = temp_file
                else:
                    memory_block = memoryview(bytearray(size_bytes))
                
                self.current_size_mb += size_bytes / (1024 * 1024)
                self.allocation_stats[pool_category] += 1
                
                return memory_block
                
            except MemoryError:
                logger.error(f"Failed to allocate {size_bytes} bytes")
                return None
    
    def deallocate(self, memory_block: memoryview):
        """Return memory to pool for reuse."""
        with self.lock:
            if not memory_block:
                return
            
            block_id = id(memory_block)
            
            # Handle memory-mapped blocks
            if block_id in self.memory_maps:
                memory_block.close()
                self.memory_maps[block_id].close()
                del self.memory_maps[block_id]
            else:
                # Return to appropriate pool
                pool_category = self._get_pool_category(len(memory_block))
                self.pools[pool_category].append(memory_block)
                
                # Limit pool size to prevent unlimited growth
                if len(self.pools[pool_category]) > 100:
                    self.pools[pool_category].popleft()
    
    def _get_pool_category(self, size_bytes: int) -> str:
        """Determine pool category for size."""
        if size_bytes < 1024:  # < 1KB
            return 'small'
        elif size_bytes < 1024 * 1024:  # < 1MB
            return 'medium'
        else:
            return 'large'
    
    def _create_temp_file(self, size_bytes: int):
        """Create temporary file for memory mapping."""
        import tempfile
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(b'\0' * size_bytes)
        temp_file.flush()
        return temp_file
    
    def _garbage_collect_pool(self):
        """Garbage collect pool to free memory."""
        with self.lock:
            initial_size = self.current_size_mb
            
            # Clear oldest allocations from each pool
            for pool_name, pool in self.pools.items():
                removed = min(len(pool) // 2, 10)  # Remove up to 50% or 10 items
                for _ in range(removed):
                    if pool:
                        old_block = pool.popleft()
                        self.current_size_mb -= len(old_block) / (1024 * 1024)
            
            freed_mb = initial_size - self.current_size_mb
            logger.info(f"Pool GC freed {freed_mb:.1f}MB")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'current_size_mb': self.current_size_mb,
                'max_size_mb': self.max_size_mb,
                'pool_sizes': {name: len(pool) for name, pool in self.pools.items()},
                'allocation_stats': dict(self.allocation_stats),
                'reuse_stats': dict(self.reuse_stats),
                'reuse_rate': (
                    sum(self.reuse_stats.values()) / 
                    max(sum(self.allocation_stats.values()), 1)
                ),
                'memory_mapped_blocks': len(self.memory_maps)
            }


class GarbageCollectionOptimizer:
    """Optimize Python garbage collection for better performance."""
    
    def __init__(self, config: GCConfiguration = None):
        self.config = config or GCConfiguration()
        self.gc_stats = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # Original GC settings for restoration
        self.original_thresholds = gc.get_threshold()
        self.original_enabled = gc.isenabled()
        
        # Performance metrics
        self.collection_times = defaultdict(list)
        self.objects_before_collection = defaultdict(list)
        self.objects_after_collection = defaultdict(list)
        
        self.lock = threading.RLock()
        
        # Apply initial configuration
        self._apply_gc_configuration()
        
        logger.info("GC optimizer initialized")
    
    def _apply_gc_configuration(self):
        """Apply garbage collection configuration."""
        if self.config.enabled:
            gc.enable()
            gc.set_threshold(*self.config.generation_thresholds)
        else:
            gc.disable()
        
        logger.info(f"GC configuration applied: {self.config}")
    
    def optimize_gc_thresholds(self):
        """Optimize GC thresholds based on performance data."""
        if not self.config.auto_tune:
            return
        
        with self.lock:
            if len(self.gc_stats) < 10:
                return  # Need more data
            
            # Analyze recent GC performance
            recent_stats = list(self.gc_stats)[-50:]
            
            # Calculate average collection times per generation
            avg_times = {}
            for generation in [0, 1, 2]:
                times = [s['collection_times'][generation] for s in recent_stats 
                        if len(s['collection_times']) > generation]
                avg_times[generation] = np.mean(times) if times else 0.0
            
            # Adjust thresholds based on collection frequency and performance
            current_thresholds = list(self.config.generation_thresholds)
            
            # If generation 0 collections are too frequent and fast, increase threshold
            if avg_times[0] < 0.001 and len([s for s in recent_stats if s['collections'][0] > 0]) > 30:
                current_thresholds[0] = min(current_thresholds[0] + 50, 1000)
            
            # If generation 1 collections are too slow, decrease threshold
            if avg_times[1] > 0.01:
                current_thresholds[1] = max(current_thresholds[1] - 1, 5)
            
            # Generation 2 is typically kept conservative
            if avg_times[2] > 0.1:  # Very slow
                current_thresholds[2] = max(current_thresholds[2] - 1, 5)
            
            # Apply new thresholds if they changed
            new_thresholds = tuple(current_thresholds)
            if new_thresholds != self.config.generation_thresholds:
                self.config.generation_thresholds = new_thresholds
                gc.set_threshold(*new_thresholds)
                
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'old_thresholds': self.original_thresholds,
                    'new_thresholds': new_thresholds,
                    'reason': 'auto_tune'
                })
                
                logger.info(f"GC thresholds optimized: {new_thresholds}")
    
    def collect_with_stats(self, generation: int = 2) -> Dict[str, Any]:
        """Perform garbage collection with detailed statistics."""
        start_time = time.perf_counter()
        
        # Collect statistics before
        objects_before = gc.get_count()
        stats_before = gc.get_stats() if hasattr(gc, 'get_stats') else []
        
        # Perform collection
        collected = gc.collect(generation)
        
        # Collect statistics after
        collection_time = time.perf_counter() - start_time
        objects_after = gc.get_count()
        stats_after = gc.get_stats() if hasattr(gc, 'get_stats') else []
        
        # Record statistics
        collection_stats = {
            'timestamp': time.time(),
            'generation': generation,
            'collection_time': collection_time,
            'objects_collected': collected,
            'objects_before': objects_before,
            'objects_after': objects_after,
            'collections': [stat['collections'] for stat in stats_after] if stats_after else [0, 0, 0],
            'collection_times': [collection_time, 0.0, 0.0]  # Simplified
        }
        
        with self.lock:
            self.gc_stats.append(collection_stats)
            self.collection_times[generation].append(collection_time)
            self.objects_before_collection[generation].append(sum(objects_before))
            self.objects_after_collection[generation].append(sum(objects_after))
        
        return collection_stats
    
    def adaptive_collection(self) -> Dict[str, Any]:
        """Perform adaptive garbage collection based on memory pressure."""
        memory_info = psutil.virtual_memory()
        memory_pressure = memory_info.percent / 100.0
        
        if memory_pressure > self.config.aggressive_collection_threshold:
            # Aggressive collection - all generations
            logger.info(f"Aggressive GC triggered (memory pressure: {memory_pressure:.2f})")
            stats = self.collect_with_stats(2)  # Full collection
        elif memory_pressure > self.config.memory_pressure_threshold:
            # Moderate collection - generations 0 and 1
            stats = self.collect_with_stats(1)
        else:
            # Light collection - generation 0 only
            stats = self.collect_with_stats(0)
        
        # Auto-tune thresholds if enabled
        self.optimize_gc_thresholds()
        
        return stats
    
    def get_gc_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive GC performance report."""
        with self.lock:
            if not self.gc_stats:
                return {}
            
            recent_stats = list(self.gc_stats)[-100:]
            
            # Calculate performance metrics
            total_collections = sum(s['objects_collected'] for s in recent_stats)
            total_time = sum(s['collection_time'] for s in recent_stats)
            avg_collection_time = total_time / len(recent_stats) if recent_stats else 0
            
            # Per-generation statistics
            gen_stats = {}
            for gen in [0, 1, 2]:
                gen_collections = [s for s in recent_stats if s['generation'] == gen]
                if gen_collections:
                    gen_stats[f'generation_{gen}'] = {
                        'collections': len(gen_collections),
                        'avg_time': np.mean([s['collection_time'] for s in gen_collections]),
                        'total_objects_collected': sum(s['objects_collected'] for s in gen_collections),
                        'avg_objects_collected': np.mean([s['objects_collected'] for s in gen_collections])
                    }
            
            return {
                'current_thresholds': gc.get_threshold(),
                'original_thresholds': self.original_thresholds,
                'gc_enabled': gc.isenabled(),
                'total_collections': len(recent_stats),
                'total_objects_collected': total_collections,
                'total_collection_time': total_time,
                'avg_collection_time': avg_collection_time,
                'collection_efficiency': total_collections / max(total_time, 0.001),  # objects per second
                'generation_stats': gen_stats,
                'optimization_count': len(self.optimization_history),
                'recent_optimizations': list(self.optimization_history)[-5:]
            }
    
    def restore_original_settings(self):
        """Restore original GC settings."""
        if self.original_enabled:
            gc.enable()
        else:
            gc.disable()
        
        gc.set_threshold(*self.original_thresholds)
        logger.info("Original GC settings restored")


class MemoryOptimizationManager:
    """Main memory optimization manager combining all optimization strategies."""
    
    def __init__(self):
        self.object_tracker = ObjectTracker()
        self.memory_pool = SmartMemoryPool()
        self.gc_optimizer = GarbageCollectionOptimizer()
        
        # Memory monitoring
        self.memory_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Optimization callbacks
        self.low_memory_callbacks = []
        self.high_memory_callbacks = []
        
        # Memory thresholds
        self.low_memory_threshold = 0.85  # 85% memory usage
        self.critical_memory_threshold = 0.95  # 95% memory usage
        
        # Start monitoring
        self.start_memory_monitoring()
        
        logger.info("Memory optimization manager initialized")
    
    def start_memory_monitoring(self, interval: float = 5.0):
        """Start memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    stats = self._collect_memory_stats()
                    if stats:
                        self.memory_history.append(stats)
                        self._check_memory_pressure(stats)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(30)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Memory monitoring stopped")
    
    def _collect_memory_stats(self) -> Optional[MemoryStats]:
        """Collect comprehensive memory statistics."""
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Garbage collection stats
            gc_count = sum(gc.get_count())
            gc_collections = [stat['collections'] for stat in gc.get_stats()] if hasattr(gc, 'get_stats') else [0, 0, 0]
            
            # Largest objects (simplified)
            largest_objects = []
            if TRACEMALLOC_AVAILABLE and tracemalloc.is_tracing():
                try:
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')[:5]
                    largest_objects = [str(stat) for stat in top_stats]
                except:
                    pass
            
            return MemoryStats(
                timestamp=time.time(),
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                used_memory_gb=memory.used / (1024**3),
                memory_percent=memory.percent,
                swap_used_gb=swap.used / (1024**3),
                process_memory_gb=process_memory.rss / (1024**3),
                gc_objects=gc_count,
                gc_collections=gc_collections,
                largest_objects=largest_objects
            )
            
        except Exception as e:
            logger.error(f"Error collecting memory stats: {e}")
            return None
    
    def _check_memory_pressure(self, stats: MemoryStats):
        """Check for memory pressure and take action."""
        memory_pressure = stats.memory_percent / 100.0
        
        if memory_pressure > self.critical_memory_threshold:
            logger.warning(f"Critical memory pressure: {memory_pressure:.2f}")
            self._handle_critical_memory()
        elif memory_pressure > self.low_memory_threshold:
            logger.info(f"Low memory detected: {memory_pressure:.2f}")
            self._handle_low_memory()
    
    def _handle_low_memory(self):
        """Handle low memory situation."""
        # Perform adaptive garbage collection
        gc_stats = self.gc_optimizer.adaptive_collection()
        
        # Clean up object tracking
        self.object_tracker.cleanup_tracking()
        
        # Trigger callbacks
        for callback in self.low_memory_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Low memory callback error: {e}")
    
    def _handle_critical_memory(self):
        """Handle critical memory situation."""
        # Aggressive garbage collection
        self.gc_optimizer.collect_with_stats(2)
        
        # Clean memory pool
        self.memory_pool._garbage_collect_pool()
        
        # Force PyTorch cache cleanup if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Trigger callbacks
        for callback in self.high_memory_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"High memory callback error: {e}")
    
    @contextmanager
    def optimized_allocation(self, size_mb: int):
        """Context manager for optimized memory allocation."""
        memory_block = self.memory_pool.allocate(size_mb * 1024 * 1024)
        try:
            yield memory_block
        finally:
            if memory_block:
                self.memory_pool.deallocate(memory_block)
    
    def track_object(self, obj: Any, obj_type: str = None):
        """Track object for memory analysis."""
        self.object_tracker.track_object(obj, obj_type)
    
    def add_memory_callback(self, callback: Callable[[], None], level: str = "low"):
        """Add callback for memory pressure events."""
        if level == "low":
            self.low_memory_callbacks.append(callback)
        elif level == "critical":
            self.high_memory_callbacks.append(callback)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        start_time = time.time()
        initial_stats = self._collect_memory_stats()
        
        # Optimize garbage collection
        gc_stats = self.gc_optimizer.adaptive_collection()
        
        # Clean up tracking
        self.object_tracker.cleanup_tracking()
        
        # Detect memory leaks
        leaks = self.object_tracker.detect_memory_leaks()
        
        # Get memory hotspots
        hotspots = self.object_tracker.get_memory_hotspots()
        
        # Collect final stats
        final_stats = self._collect_memory_stats()
        optimization_time = time.time() - start_time
        
        # Calculate improvement
        memory_freed_mb = 0
        if initial_stats and final_stats:
            memory_freed_mb = (initial_stats.used_memory_gb - final_stats.used_memory_gb) * 1024
        
        return {
            'optimization_time': optimization_time,
            'memory_freed_mb': memory_freed_mb,
            'gc_stats': gc_stats,
            'detected_leaks': leaks,
            'memory_hotspots': hotspots[:10],  # Top 10
            'initial_memory_gb': initial_stats.used_memory_gb if initial_stats else 0,
            'final_memory_gb': final_stats.used_memory_gb if final_stats else 0,
            'pool_stats': self.memory_pool.get_pool_stats()
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive memory optimization report."""
        current_stats = self._collect_memory_stats()
        
        return {
            'timestamp': time.time(),
            'current_memory': current_stats._asdict() if current_stats else {},
            'memory_pool': self.memory_pool.get_pool_stats(),
            'gc_performance': self.gc_optimizer.get_gc_performance_report(),
            'object_tracking': {
                'tracked_objects': len(self.object_tracker.tracked_objects),
                'object_types': len(self.object_tracker.object_stats),
                'memory_hotspots': self.object_tracker.get_memory_hotspots()[:5],
                'detected_leaks': self.object_tracker.detect_memory_leaks()
            },
            'memory_history_length': len(self.memory_history),
            'monitoring_active': self.monitoring_active,
            'optimization_recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if len(self.memory_history) < 10:
            return recommendations
        
        recent_stats = list(self.memory_history)[-10:]
        avg_memory_percent = np.mean([s.memory_percent for s in recent_stats])
        
        # High memory usage
        if avg_memory_percent > 80:
            recommendations.append("Consider increasing available memory or optimizing memory usage")
        
        # Memory leaks
        leaks = self.object_tracker.detect_memory_leaks()
        if leaks:
            recommendations.append(f"Investigate potential memory leaks: {len(leaks)} detected")
        
        # GC performance
        gc_report = self.gc_optimizer.get_gc_performance_report()
        if gc_report.get('avg_collection_time', 0) > 0.1:
            recommendations.append("Consider optimizing garbage collection thresholds")
        
        # Pool efficiency
        pool_stats = self.memory_pool.get_pool_stats()
        if pool_stats.get('reuse_rate', 0) < 0.5:
            recommendations.append("Memory pool reuse rate is low, consider adjusting pool sizes")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup memory optimization resources."""
        self.stop_memory_monitoring()
        self.gc_optimizer.restore_original_settings()
        
        logger.info("Memory optimization manager cleanup completed")


def demo_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    print("💾 Memory Optimization Demo")
    print("=" * 40)
    
    # Create memory optimizer
    optimizer = MemoryOptimizationManager()
    
    print("📊 Initial memory stats...")
    time.sleep(2)  # Let monitoring collect initial data
    
    # Create memory pressure
    print("🔄 Creating memory pressure...")
    large_objects = []
    
    for i in range(10):
        # Create large objects to simulate memory usage
        large_array = np.random.random((1000, 1000))  # ~8MB each
        optimizer.track_object(large_array, "demo_array")
        large_objects.append(large_array)
        
        # Use optimized allocation for some objects
        with optimizer.optimized_allocation(5) as memory_block:
            if memory_block:
                # Simulate work with allocated memory
                pass
    
    # Let monitoring detect the pressure
    time.sleep(3)
    
    print("🧹 Performing memory optimization...")
    
    # Perform optimization
    optimization_results = optimizer.optimize_memory()
    
    print(f"✅ Optimization Results:")
    print(f"  Memory freed: {optimization_results['memory_freed_mb']:.1f} MB")
    print(f"  Optimization time: {optimization_results['optimization_time']:.3f} seconds")
    print(f"  Objects collected: {optimization_results['gc_stats']['objects_collected']}")
    
    if optimization_results['detected_leaks']:
        print(f"  ⚠️  Detected {len(optimization_results['detected_leaks'])} potential leaks")
    
    if optimization_results['memory_hotspots']:
        print(f"  🔥 Top memory hotspot: {optimization_results['memory_hotspots'][0][0]}")
    
    # Get comprehensive report
    report = optimizer.get_comprehensive_report()
    
    print(f"\n📋 Memory Report:")
    current_memory = report['current_memory']
    if current_memory:
        print(f"  Current memory usage: {current_memory['memory_percent']:.1f}%")
        print(f"  Process memory: {current_memory['process_memory_gb']:.2f} GB")
        print(f"  GC objects: {current_memory['gc_objects']}")
    
    # Pool statistics
    pool_stats = report['memory_pool']
    print(f"  Memory pool reuse rate: {pool_stats['reuse_rate']:.2f}")
    print(f"  Pool size: {pool_stats['current_size_mb']:.1f} MB")
    
    # Recommendations
    recommendations = report['optimization_recommendations']
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Cleanup
    del large_objects  # Free the demo objects
    optimizer.cleanup()
    
    print(f"\n✅ Memory optimization demo completed!")


if __name__ == "__main__":
    demo_memory_optimization()