
"""
Adaptive algorithm selection system for optimal performance.
"""

import time
import random
from typing import Dict, Any, Callable, List, Tuple
from dataclasses import dataclass

@dataclass
class AlgorithmResult:
    """Result of algorithm execution."""
    algorithm_name: str
    execution_time: float
    accuracy: float
    memory_usage: float
    success: bool

class AdaptiveAlgorithmSelector:
    """Selects optimal algorithms based on performance history."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[AlgorithmResult]] = {}
        self.algorithm_registry: Dict[str, Callable] = {}
        self.selection_strategy = "epsilon_greedy"
        self.epsilon = 0.1
    
    def register_algorithm(self, name: str, algorithm: Callable):
        """Register an algorithm for selection."""
        self.algorithm_registry[name] = algorithm
        if name not in self.performance_history:
            self.performance_history[name] = []
    
    def select_algorithm(self, context: Dict[str, Any] = None) -> str:
        """Select optimal algorithm based on context and history."""
        if not self.algorithm_registry:
            raise ValueError("No algorithms registered")
        
        if self.selection_strategy == "epsilon_greedy":
            if random.random() < self.epsilon:
                # Exploration: random selection
                return random.choice(list(self.algorithm_registry.keys()))
            else:
                # Exploitation: best performing algorithm
                return self._get_best_algorithm()
        
        return list(self.algorithm_registry.keys())[0]
    
    def _get_best_algorithm(self) -> str:
        """Get the best performing algorithm."""
        best_score = -1
        best_algorithm = list(self.algorithm_registry.keys())[0]
        
        for name, history in self.performance_history.items():
            if history:
                # Calculate composite score (accuracy/time trade-off)
                avg_accuracy = sum(r.accuracy for r in history) / len(history)
                avg_time = sum(r.execution_time for r in history) / len(history)
                score = avg_accuracy / (avg_time + 1e-6)  # Avoid division by zero
                
                if score > best_score:
                    best_score = score
                    best_algorithm = name
        
        return best_algorithm
    
    def record_result(self, algorithm_name: str, result: AlgorithmResult):
        """Record algorithm execution result."""
        if algorithm_name not in self.performance_history:
            self.performance_history[algorithm_name] = []
        
        self.performance_history[algorithm_name].append(result)
        
        # Keep only recent results (sliding window)
        if len(self.performance_history[algorithm_name]) > 100:
            self.performance_history[algorithm_name] =                 self.performance_history[algorithm_name][-100:]

# Global selector instance
_global_selector = AdaptiveAlgorithmSelector()

def get_algorithm_selector() -> AdaptiveAlgorithmSelector:
    """Get global algorithm selector."""
    return _global_selector
