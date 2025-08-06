"""Parallel processing and distributed computing utilities."""

import multiprocessing as mp
import threading
import concurrent.futures
import time
import queue
import pickle
from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from ..utils.logging import get_logger

logger = get_logger("parallel")


@dataclass
class TaskResult:
    """Result of a parallel task."""
    task_id: str
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: str = ""


@dataclass
class WorkerStats:
    """Statistics for a parallel worker."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_task_time: float = 0.0


class ParallelSimulator:
    """Parallel simulation execution with load balancing."""
    
    def __init__(self, max_workers: Optional[int] = None, backend: str = "process"):
        """
        Initialize parallel simulator.
        
        Args:
            max_workers: Maximum number of workers (default: CPU count)
            backend: Parallelization backend ('process', 'thread')
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.backend = backend
        self.executor = None
        self.worker_stats = {}
        self._task_counter = 0
        self._lock = threading.RLock()
        
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self):
        """Start the parallel executor."""
        if self.executor is not None:
            return
        
        if self.backend == "process":
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
        elif self.backend == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        logger.info(f"Started parallel simulator with {self.max_workers} {self.backend} workers")
    
    def stop(self):
        """Stop the parallel executor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        logger.info("Stopped parallel simulator")
    
    def simulate_batch(self, 
                      simulation_func: Callable,
                      parameter_sets: List[Dict[str, Any]],
                      timeout: Optional[float] = None) -> List[TaskResult]:
        """
        Run simulations in parallel with different parameter sets.
        
        Args:
            simulation_func: Function to run simulation
            parameter_sets: List of parameter dictionaries
            timeout: Timeout per task in seconds
            
        Returns:
            List of task results
        """
        if self.executor is None:
            raise RuntimeError("Parallel simulator not started")
        
        # Submit all tasks
        futures = {}
        results = []
        
        for i, params in enumerate(parameter_sets):
            with self._lock:
                task_id = f"sim_{self._task_counter}_{i}"
                self._task_counter += 1
            
            future = self.executor.submit(self._run_simulation_task, simulation_func, params, task_id)
            futures[future] = task_id
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            task_id = futures[future]
            
            try:
                result = future.result()
                self._update_worker_stats(result)
                results.append(result)
            except Exception as e:
                error_result = TaskResult(
                    task_id=task_id,
                    result=None,
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
        
        return results
    
    def _run_simulation_task(self, simulation_func: Callable, params: Dict[str, Any], task_id: str) -> TaskResult:
        """Run a single simulation task."""
        start_time = time.time()
        worker_id = f"{mp.current_process().pid}_{threading.get_ident()}"
        
        try:
            result = simulation_func(**params)
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                result=result,
                success=True,
                execution_time=execution_time,
                worker_id=worker_id
            )
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                result=None,
                success=False,
                error=str(e),
                execution_time=execution_time,
                worker_id=worker_id
            )
    
    def _update_worker_stats(self, result: TaskResult):
        """Update worker statistics."""
        with self._lock:
            worker_id = result.worker_id
            
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
            
            stats = self.worker_stats[worker_id]
            
            if result.success:
                stats.tasks_completed += 1
            else:
                stats.tasks_failed += 1
            
            stats.total_execution_time += result.execution_time
            stats.last_task_time = result.execution_time
            
            total_tasks = stats.tasks_completed + stats.tasks_failed
            if total_tasks > 0:
                stats.average_execution_time = stats.total_execution_time / total_tasks
    
    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """Get statistics for all workers."""
        with self._lock:
            return self.worker_stats.copy()
    
    def simulate_episodes_parallel(self,
                                  env_factory: Callable,
                                  agent_factory: Callable,
                                  num_episodes: int,
                                  episodes_per_worker: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run multiple episodes in parallel.
        
        Args:
            env_factory: Function that creates environment instances
            agent_factory: Function that creates agent instances  
            num_episodes: Total number of episodes
            episodes_per_worker: Episodes per worker (default: auto-balance)
            
        Returns:
            List of episode results
        """
        if episodes_per_worker is None:
            episodes_per_worker = max(1, num_episodes // self.max_workers)
        
        # Create parameter sets for workers
        parameter_sets = []
        episodes_assigned = 0
        
        while episodes_assigned < num_episodes:
            episodes_for_this_worker = min(episodes_per_worker, num_episodes - episodes_assigned)
            
            parameter_sets.append({
                'env_factory': env_factory,
                'agent_factory': agent_factory,
                'num_episodes': episodes_for_this_worker,
                'start_episode': episodes_assigned
            })
            
            episodes_assigned += episodes_for_this_worker
        
        # Run parallel simulation
        task_results = self.simulate_batch(self._run_episodes_batch, parameter_sets)
        
        # Flatten results
        all_episodes = []
        for task_result in task_results:
            if task_result.success:
                all_episodes.extend(task_result.result)
            else:
                logger.error(f"Episode batch failed: {task_result.error}")
        
        return all_episodes
    
    def _run_episodes_batch(self,
                           env_factory: Callable,
                           agent_factory: Callable,
                           num_episodes: int,
                           start_episode: int) -> List[Dict[str, Any]]:
        """Run a batch of episodes in a single worker."""
        episodes_data = []
        
        # Create environment and agent instances for this worker
        env = env_factory()
        agent = agent_factory()
        
        for episode_idx in range(num_episodes):
            episode_id = start_episode + episode_idx
            
            # Run episode
            obs, info = env.reset()
            episode_data = {
                'episode_id': episode_id,
                'steps': [],
                'total_reward': 0.0,
                'episode_length': 0,
                'success': False
            }
            
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated):
                action = agent.predict(obs)[0] if hasattr(agent, 'predict') else env.action_space.sample()
                next_obs, reward, done, truncated, step_info = env.step(action)
                
                episode_data['steps'].append({
                    'step': step,
                    'obs': obs.copy() if isinstance(obs, np.ndarray) else obs,
                    'action': action.copy() if isinstance(action, np.ndarray) else action,
                    'reward': reward,
                    'done': done,
                    'truncated': truncated
                })
                
                episode_data['total_reward'] += reward
                episode_data['episode_length'] += 1
                
                obs = next_obs
                step += 1
            
            episode_data['success'] = not done or step_info.get('success', False)
            episodes_data.append(episode_data)
        
        env.close()
        return episodes_data


class BatchProcessor:
    """Batch processing utility for efficient data handling."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = None
    
    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def process_batches(self,
                       data: List[Any],
                       process_func: Callable[[List[Any]], Any],
                       overlap_batches: bool = False) -> Iterator[Any]:
        """
        Process data in batches with optional parallelization.
        
        Args:
            data: Input data list
            process_func: Function to process each batch
            overlap_batches: Whether to overlap batch processing
            
        Yields:
            Processed batch results
        """
        if not data:
            return
        
        # Create batches
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batches.append(batch)
        
        if overlap_batches and self.executor:
            # Process batches in parallel with overlapping
            futures = []
            
            # Submit initial batches
            max_concurrent = min(self.max_workers, len(batches))
            for i in range(max_concurrent):
                future = self.executor.submit(process_func, batches[i])
                futures.append((future, i))
            
            batch_idx = max_concurrent
            
            # Process remaining batches as previous ones complete
            while futures:
                # Wait for next completion
                done_futures, _ = concurrent.futures.wait(
                    [f[0] for f in futures], 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Yield completed results
                remaining_futures = []
                for future, idx in futures:
                    if future in done_futures:
                        try:
                            result = future.result()
                            yield result
                        except Exception as e:
                            logger.error(f"Batch processing error: {e}")
                    else:
                        remaining_futures.append((future, idx))
                
                futures = remaining_futures
                
                # Submit new batches if available
                while batch_idx < len(batches) and len(futures) < self.max_workers:
                    future = self.executor.submit(process_func, batches[batch_idx])
                    futures.append((future, batch_idx))
                    batch_idx += 1
        
        else:
            # Sequential processing
            for batch in batches:
                try:
                    result = process_func(batch)
                    yield result
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
    
    def aggregate_results(self,
                         data: List[Any],
                         process_func: Callable[[List[Any]], Any],
                         aggregate_func: Callable[[List[Any]], Any]) -> Any:
        """
        Process data in batches and aggregate results.
        
        Args:
            data: Input data list
            process_func: Function to process each batch
            aggregate_func: Function to aggregate batch results
            
        Returns:
            Aggregated result
        """
        batch_results = list(self.process_batches(data, process_func))
        return aggregate_func(batch_results)


class DistributedTraining:
    """Distributed training coordination for RL algorithms."""
    
    def __init__(self, num_workers: int = 4, communication_backend: str = "shared_memory"):
        self.num_workers = num_workers
        self.communication_backend = communication_backend
        self.workers = []
        self.parameter_server = None
        self.gradient_queue = None
        self.parameter_queue = None
        self._shutdown_event = None
        
    def setup_distributed_training(self, 
                                  model_factory: Callable,
                                  env_factory: Callable,
                                  training_config: Dict[str, Any]):
        """
        Set up distributed training infrastructure.
        
        Args:
            model_factory: Function to create model instances
            env_factory: Function to create environment instances
            training_config: Training configuration parameters
        """
        if self.communication_backend == "shared_memory":
            self._setup_shared_memory_training(model_factory, env_factory, training_config)
        else:
            raise ValueError(f"Unsupported communication backend: {self.communication_backend}")
    
    def _setup_shared_memory_training(self, 
                                     model_factory: Callable,
                                     env_factory: Callable,
                                     training_config: Dict[str, Any]):
        """Set up shared memory distributed training."""
        
        # Create shared queues
        self.gradient_queue = mp.Queue(maxsize=100)
        self.parameter_queue = mp.Queue(maxsize=100)
        self._shutdown_event = mp.Event()
        
        # Start parameter server
        self.parameter_server = mp.Process(
            target=self._parameter_server_process,
            args=(model_factory, training_config, self.gradient_queue, 
                  self.parameter_queue, self._shutdown_event)
        )
        self.parameter_server.start()
        
        # Start worker processes
        for worker_id in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_process,
                args=(worker_id, model_factory, env_factory, training_config,
                      self.gradient_queue, self.parameter_queue, self._shutdown_event)
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started distributed training with {self.num_workers} workers")
    
    def _parameter_server_process(self,
                                 model_factory: Callable,
                                 training_config: Dict[str, Any],
                                 gradient_queue: mp.Queue,
                                 parameter_queue: mp.Queue,
                                 shutdown_event: mp.Event):
        """Parameter server process for coordinating distributed training."""
        
        # Create master model
        model = model_factory()
        optimizer = self._create_optimizer(model, training_config)
        
        gradient_buffer = []
        last_parameter_broadcast = time.time()
        parameter_broadcast_interval = training_config.get('parameter_broadcast_interval', 10.0)
        
        while not shutdown_event.is_set():
            try:
                # Collect gradients from workers
                try:
                    gradient_data = gradient_queue.get(timeout=1.0)
                    gradient_buffer.append(gradient_data)
                except queue.Empty:
                    continue
                
                # Apply gradients when buffer is full or timeout
                if (len(gradient_buffer) >= self.num_workers or
                    time.time() - last_parameter_broadcast > parameter_broadcast_interval):
                    
                    if gradient_buffer:
                        # Aggregate gradients
                        aggregated_gradients = self._aggregate_gradients(gradient_buffer)
                        
                        # Apply to model
                        self._apply_gradients(model, optimizer, aggregated_gradients)
                        
                        # Broadcast updated parameters
                        model_state = self._get_model_state(model)
                        for _ in range(self.num_workers):
                            try:
                                parameter_queue.put(model_state, timeout=1.0)
                            except queue.Full:
                                break
                        
                        gradient_buffer.clear()
                        last_parameter_broadcast = time.time()
                
            except Exception as e:
                logger.error(f"Parameter server error: {e}")
        
        logger.info("Parameter server shutting down")
    
    def _worker_process(self,
                       worker_id: int,
                       model_factory: Callable,
                       env_factory: Callable,
                       training_config: Dict[str, Any],
                       gradient_queue: mp.Queue,
                       parameter_queue: mp.Queue,
                       shutdown_event: mp.Event):
        """Worker process for distributed training."""
        
        # Create local model and environment
        model = model_factory()
        env = env_factory()
        
        episode_count = 0
        max_episodes = training_config.get('max_episodes', 1000)
        
        while not shutdown_event.is_set() and episode_count < max_episodes:
            try:
                # Check for parameter updates
                try:
                    updated_parameters = parameter_queue.get(timeout=0.1)
                    self._update_model_parameters(model, updated_parameters)
                except queue.Empty:
                    pass
                
                # Run episode and collect gradients
                gradients = self._run_training_episode(model, env, training_config)
                
                if gradients:
                    # Send gradients to parameter server
                    gradient_data = {
                        'worker_id': worker_id,
                        'episode': episode_count,
                        'gradients': gradients
                    }
                    
                    try:
                        gradient_queue.put(gradient_data, timeout=1.0)
                    except queue.Full:
                        logger.warning(f"Worker {worker_id}: gradient queue full")
                
                episode_count += 1
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        env.close()
        logger.info(f"Worker {worker_id} shutting down after {episode_count} episodes")
    
    def _create_optimizer(self, model: Any, training_config: Dict[str, Any]):
        """Create optimizer for parameter server."""
        # This would depend on the specific ML framework being used
        # Placeholder implementation
        return None
    
    def _aggregate_gradients(self, gradient_buffer: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate gradients from multiple workers."""
        # This would implement gradient aggregation (e.g., averaging)
        # Placeholder implementation
        return gradient_buffer[0]['gradients'] if gradient_buffer else {}
    
    def _apply_gradients(self, model: Any, optimizer: Any, gradients: Dict[str, Any]):
        """Apply aggregated gradients to model."""
        # This would apply gradients using the optimizer
        # Placeholder implementation
        pass
    
    def _get_model_state(self, model: Any) -> Dict[str, Any]:
        """Get model state for broadcasting to workers."""
        # This would extract model parameters
        # Placeholder implementation
        return {}
    
    def _update_model_parameters(self, model: Any, parameters: Dict[str, Any]):
        """Update local model with parameters from server."""
        # This would update model parameters
        # Placeholder implementation
        pass
    
    def _run_training_episode(self, model: Any, env: Any, training_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run training episode and compute gradients."""
        # This would run a training episode and compute gradients
        # Placeholder implementation
        return {}
    
    def shutdown(self):
        """Shutdown distributed training."""
        if self._shutdown_event:
            self._shutdown_event.set()
        
        # Wait for workers to complete
        for worker in self.workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                worker.terminate()
        
        # Wait for parameter server
        if self.parameter_server:
            self.parameter_server.join(timeout=10.0)
            if self.parameter_server.is_alive():
                self.parameter_server.terminate()
        
        logger.info("Distributed training shutdown complete")


def parallelize_function(func: Callable, max_workers: int = 4, backend: str = "thread"):
    """Decorator to parallelize function calls."""
    
    def decorator(*args_list, **kwargs):
        """Execute function with multiple argument sets in parallel."""
        
        if backend == "process":
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            executor_class = concurrent.futures.ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            
            if isinstance(args_list[0], (list, tuple)) and len(args_list) == 1:
                # Multiple argument sets provided
                for args in args_list[0]:
                    if isinstance(args, (list, tuple)):
                        future = executor.submit(func, *args, **kwargs)
                    else:
                        future = executor.submit(func, args, **kwargs)
                    futures.append(future)
            else:
                # Single argument set
                future = executor.submit(func, *args_list, **kwargs)
                futures.append(future)
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(e)
            
            return results if len(results) > 1 else results[0]
    
    return decorator