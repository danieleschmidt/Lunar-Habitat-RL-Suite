"""
Database Optimization and Efficient Data Pipelines - Generation 3
High-performance data management for NASA space missions.
"""

import asyncio
import sqlite3
import threading
import time
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import queue

from ..utils.logging import get_logger

logger = get_logger("database_optimization")


@dataclass
class QueryStats:
    """Database query performance statistics."""
    query_hash: str
    query_type: str
    execution_time: float
    rows_affected: int
    cache_hit: bool
    timestamp: float


class DatabaseConnectionPool:
    """High-performance database connection pool."""
    
    def __init__(self, db_path: str, min_connections: int = 2, max_connections: int = 10):
        self.db_path = db_path
        self.min_connections = min_connections
        self.max_connections = max_connections
        
        self.available_connections = queue.Queue()
        self.active_connections = set()
        self.connection_stats = defaultdict(int)
        self.lock = threading.RLock()
        
        # Initialize minimum connections
        self._initialize_pool()
        
        logger.info(f"Database pool initialized: {db_path} ({min_connections}-{max_connections} connections)")
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        for _ in range(self.min_connections):
            conn = self._create_connection()
            if conn:
                self.available_connections.put(conn)
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create optimized database connection."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            
            # Optimize SQLite settings
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            return conn
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool."""
        conn = None
        try:
            # Try to get available connection
            try:
                conn = self.available_connections.get(timeout=5.0)
            except queue.Empty:
                # Create new connection if under limit
                with self.lock:
                    if len(self.active_connections) < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # Wait for connection to become available
                        conn = self.available_connections.get(timeout=30.0)
            
            if not conn:
                raise RuntimeError("Could not obtain database connection")
            
            with self.lock:
                self.active_connections.add(conn)
                self.connection_stats['checkouts'] += 1
            
            yield conn
            
        finally:
            if conn:
                with self.lock:
                    self.active_connections.discard(conn)
                    self.connection_stats['returns'] += 1
                
                # Return to pool
                self.available_connections.put(conn)
    
    def close_all_connections(self):
        """Close all connections in pool."""
        # Close available connections
        while not self.available_connections.empty():
            try:
                conn = self.available_connections.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        # Close active connections
        with self.lock:
            for conn in list(self.active_connections):
                try:
                    conn.close()
                except:
                    pass
            self.active_connections.clear()


class QueryOptimizer:
    """Database query optimization and caching."""
    
    def __init__(self, cache_size: int = 1000):
        self.query_cache = {}
        self.cache_size = cache_size
        self.query_stats = deque(maxlen=10000)
        self.prepared_statements = {}
        self.lock = threading.RLock()
        
        # Query analysis
        self.slow_queries = deque(maxlen=100)
        self.query_patterns = defaultdict(list)
    
    def optimize_query(self, query: str, params: tuple = ()) -> Tuple[str, tuple]:
        """Optimize SQL query."""
        # Simple query optimizations
        optimized_query = query.strip()
        
        # Add LIMIT if not present in SELECT queries without it
        if (optimized_query.upper().startswith('SELECT') and 
            'LIMIT' not in optimized_query.upper() and 
            'COUNT(' not in optimized_query.upper()):
            optimized_query += ' LIMIT 10000'  # Prevent runaway queries
        
        # Use prepared statement if available
        query_hash = hash(optimized_query)
        if query_hash in self.prepared_statements:
            return self.prepared_statements[query_hash], params
        
        return optimized_query, params
    
    def cache_query_result(self, query: str, params: tuple, result: Any):
        """Cache query result."""
        if len(self.query_cache) >= self.cache_size:
            # Simple LRU eviction
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        cache_key = hash((query, params))
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'access_count': 0
        }
    
    def get_cached_result(self, query: str, params: tuple) -> Optional[Any]:
        """Get cached query result."""
        cache_key = hash((query, params))
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            
            # Check if cache is still valid (5 minutes)
            if time.time() - cache_entry['timestamp'] < 300:
                cache_entry['access_count'] += 1
                return cache_entry['result']
            else:
                del self.query_cache[cache_key]
        
        return None
    
    def record_query_stats(self, query: str, execution_time: float, 
                          rows_affected: int, cache_hit: bool):
        """Record query performance statistics."""
        query_type = query.strip().split()[0].upper()
        query_hash = str(hash(query))
        
        stats = QueryStats(
            query_hash=query_hash,
            query_type=query_type,
            execution_time=execution_time,
            rows_affected=rows_affected,
            cache_hit=cache_hit,
            timestamp=time.time()
        )
        
        with self.lock:
            self.query_stats.append(stats)
            
            # Track slow queries
            if execution_time > 1.0:  # > 1 second
                self.slow_queries.append({
                    'query': query[:200],  # First 200 chars
                    'execution_time': execution_time,
                    'timestamp': time.time()
                })
            
            # Track query patterns
            self.query_patterns[query_type].append(execution_time)
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get query performance analysis."""
        with self.lock:
            if not self.query_stats:
                return {}
            
            recent_stats = list(self.query_stats)[-1000:]  # Last 1000 queries
            
            # Calculate metrics
            total_queries = len(recent_stats)
            total_time = sum(s.execution_time for s in recent_stats)
            cache_hits = sum(1 for s in recent_stats if s.cache_hit)
            
            # Query type analysis
            query_types = defaultdict(list)
            for stats in recent_stats:
                query_types[stats.query_type].append(stats.execution_time)
            
            return {
                'total_queries': total_queries,
                'total_execution_time': total_time,
                'avg_query_time': total_time / max(total_queries, 1),
                'cache_hit_rate': cache_hits / max(total_queries, 1),
                'slow_queries_count': len([s for s in recent_stats if s.execution_time > 1.0]),
                'query_type_performance': {
                    qtype: {
                        'count': len(times),
                        'avg_time': np.mean(times),
                        'max_time': np.max(times)
                    } for qtype, times in query_types.items()
                },
                'recent_slow_queries': list(self.slow_queries)[-5:]
            }


class OptimizedDataPipeline:
    """High-performance data pipeline for training data."""
    
    def __init__(self, db_pool: DatabaseConnectionPool):
        self.db_pool = db_pool
        self.query_optimizer = QueryOptimizer()
        self.batch_size = 1000
        self.prefetch_size = 5000
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prefetch_queue = queue.Queue(maxsize=10)
        self.prefetch_active = False
        
        # Data transformation cache
        self.transform_cache = {}
        
        logger.info("Optimized data pipeline initialized")
    
    def execute_optimized_query(self, query: str, params: tuple = ()) -> Any:
        """Execute optimized database query."""
        start_time = time.time()
        
        # Check cache first
        cached_result = self.query_optimizer.get_cached_result(query, params)
        if cached_result is not None:
            self.query_optimizer.record_query_stats(
                query, 0.0, len(cached_result) if isinstance(cached_result, list) else 1, True
            )
            return cached_result
        
        # Optimize query
        optimized_query, optimized_params = self.query_optimizer.optimize_query(query, params)
        
        # Execute query
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(optimized_query, optimized_params)
            
            if optimized_query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
                conn.commit()
        
        execution_time = time.time() - start_time
        
        # Cache result if appropriate
        if (execution_time > 0.1 and  # Only cache slow queries
            optimized_query.strip().upper().startswith('SELECT') and
            len(str(result)) < 1024 * 1024):  # < 1MB results
            self.query_optimizer.cache_query_result(query, params, result)
        
        # Record stats
        self.query_optimizer.record_query_stats(
            query, execution_time, 
            len(result) if isinstance(result, list) else result, False
        )
        
        return result
    
    def batch_insert_data(self, table: str, data: List[Dict[str, Any]]) -> int:
        """Optimized batch data insertion."""
        if not data:
            return 0
        
        # Prepare batch insert
        columns = list(data[0].keys())
        placeholders = ','.join(['?' for _ in columns])
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
        
        # Convert data to tuples
        values = [tuple(row[col] for col in columns) for row in data]
        
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, values)
            conn.commit()
            return cursor.rowcount
    
    def stream_training_data(self, query: str, batch_size: int = None) -> Any:
        """Stream training data in batches."""
        batch_size = batch_size or self.batch_size
        
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                
                # Transform batch if needed
                transformed_batch = self._transform_batch(batch)
                yield transformed_batch
    
    def _transform_batch(self, batch: List[tuple]) -> np.ndarray:
        """Transform batch data for training."""
        # Simple transformation - convert to numpy array
        if not batch:
            return np.array([])
        
        try:
            return np.array(batch, dtype=np.float32)
        except:
            # Handle mixed types
            return np.array([list(row) for row in batch], dtype=object)
    
    def create_indexes(self, table: str, columns: List[str]):
        """Create database indexes for performance."""
        for column in columns:
            index_name = f"idx_{table}_{column}"
            query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({column})"
            self.execute_optimized_query(query)
            logger.info(f"Created index: {index_name}")
    
    def analyze_table_performance(self, table: str) -> Dict[str, Any]:
        """Analyze table performance and suggest optimizations."""
        # Get table info
        table_info = self.execute_optimized_query(f"PRAGMA table_info({table})")
        
        # Get table statistics
        count_query = f"SELECT COUNT(*) FROM {table}"
        row_count = self.execute_optimized_query(count_query)[0][0]
        
        # Check for indexes
        index_query = f"PRAGMA index_list({table})"
        indexes = self.execute_optimized_query(index_query)
        
        return {
            'table_name': table,
            'row_count': row_count,
            'column_count': len(table_info),
            'indexes': len(indexes),
            'columns': [col[1] for col in table_info],  # Column names
            'index_names': [idx[1] for idx in indexes]
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get data pipeline performance statistics."""
        return {
            'query_optimizer': self.query_optimizer.get_performance_analysis(),
            'connection_pool': self.db_pool.connection_stats,
            'batch_size': self.batch_size,
            'prefetch_size': self.prefetch_size,
            'cache_size': len(self.transform_cache)
        }


def demo_database_optimization():
    """Demonstrate database optimization capabilities."""
    print("🗄️ Database Optimization Demo")
    print("=" * 40)
    
    # Create test database
    db_path = "/tmp/test_optimization.db"
    
    # Initialize optimized database system
    db_pool = DatabaseConnectionPool(db_path, min_connections=2, max_connections=8)
    pipeline = OptimizedDataPipeline(db_pool)
    
    print("📊 Setting up test database...")
    
    # Create test table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY,
        episode_id INTEGER,
        state_data TEXT,
        action_data TEXT,
        reward REAL,
        timestamp REAL
    )
    """
    
    pipeline.execute_optimized_query(create_table_query)
    
    # Create indexes for performance
    pipeline.create_indexes('training_data', ['episode_id', 'timestamp'])
    
    print("📝 Inserting test data...")
    
    # Insert test data
    test_data = []
    for i in range(1000):
        test_data.append({
            'episode_id': i // 100,  # 10 episodes with 100 steps each
            'state_data': json.dumps([np.random.random() for _ in range(10)]),
            'action_data': json.dumps([np.random.random() for _ in range(5)]),
            'reward': np.random.random() * 100,
            'timestamp': time.time() + i
        })
    
    # Batch insert
    inserted_count = pipeline.batch_insert_data('training_data', test_data)
    print(f"✅ Inserted {inserted_count} records")
    
    print("🔍 Testing query optimization...")
    
    # Test various queries
    queries = [
        "SELECT COUNT(*) FROM training_data",
        "SELECT AVG(reward) FROM training_data WHERE episode_id = 5",
        "SELECT * FROM training_data WHERE reward > 50 ORDER BY timestamp DESC",
        "SELECT episode_id, AVG(reward) FROM training_data GROUP BY episode_id"
    ]
    
    for query in queries:
        start_time = time.time()
        result = pipeline.execute_optimized_query(query)
        query_time = time.time() - start_time
        
        result_size = len(result) if isinstance(result, list) else result
        print(f"  Query: {query[:50]}...")
        print(f"    Time: {query_time:.4f}s, Results: {result_size}")
    
    print("📈 Testing data streaming...")
    
    # Test streaming data
    stream_query = "SELECT * FROM training_data ORDER BY timestamp"
    batch_count = 0
    total_rows = 0
    
    for batch in pipeline.stream_training_data(stream_query, batch_size=250):
        batch_count += 1
        total_rows += len(batch)
        if batch_count >= 3:  # Limit demo output
            break
    
    print(f"✅ Streamed {total_rows} rows in {batch_count} batches")
    
    # Analyze performance
    print("📊 Performance Analysis:")
    stats = pipeline.get_pipeline_stats()
    
    query_stats = stats['query_optimizer']
    if query_stats:
        print(f"  Total queries: {query_stats['total_queries']}")
        print(f"  Average query time: {query_stats['avg_query_time']:.4f}s")
        print(f"  Cache hit rate: {query_stats['cache_hit_rate']:.2%}")
        
        if query_stats['slow_queries_count'] > 0:
            print(f"  Slow queries detected: {query_stats['slow_queries_count']}")
    
    connection_stats = stats['connection_pool']
    print(f"  Connection checkouts: {connection_stats['checkouts']}")
    print(f"  Connection returns: {connection_stats['returns']}")
    
    # Table analysis
    table_analysis = pipeline.analyze_table_performance('training_data')
    print(f"\n🔍 Table Analysis:")
    print(f"  Rows: {table_analysis['row_count']}")
    print(f"  Columns: {table_analysis['column_count']}")
    print(f"  Indexes: {table_analysis['indexes']}")
    
    # Cleanup
    db_pool.close_all_connections()
    
    # Remove test database
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print(f"\n✅ Database optimization demo completed!")


if __name__ == "__main__":
    demo_database_optimization()