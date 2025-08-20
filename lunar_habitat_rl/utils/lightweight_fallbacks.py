"""Lightweight fallback mechanisms for when advanced features aren't available - Generation 1."""

import random
import math
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


class FallbackManager:
    """Manages fallbacks for missing dependencies and features."""
    
    def __init__(self):
        self.available_features = {}
        self.fallback_warnings = []
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which dependencies are available."""
        # Check for numpy
        try:
            import numpy
            self.available_features['numpy'] = True
        except ImportError:
            self.available_features['numpy'] = False
            self.fallback_warnings.append("NumPy not available - using Python lists")
        
        # Check for gymnasium
        try:
            import gymnasium
            self.available_features['gymnasium'] = True
        except ImportError:
            self.available_features['gymnasium'] = False
            self.fallback_warnings.append("Gymnasium not available - using mock spaces")
        
        # Check for matplotlib
        try:
            import matplotlib
            self.available_features['matplotlib'] = True
        except ImportError:
            self.available_features['matplotlib'] = False
            self.fallback_warnings.append("Matplotlib not available - visualization disabled")
        
        # Check for torch/tensorflow
        try:
            import torch
            self.available_features['torch'] = True
        except ImportError:
            try:
                import tensorflow
                self.available_features['tensorflow'] = True
            except ImportError:
                self.available_features['deep_learning'] = False
                self.fallback_warnings.append("No deep learning framework available")
        
        # Check for psutil
        try:
            import psutil
            self.available_features['psutil'] = True
        except ImportError:
            self.available_features['psutil'] = False
            self.fallback_warnings.append("psutil not available - basic monitoring only")
    
    def get_fallback_warnings(self) -> List[str]:
        """Get list of fallback warnings."""
        return self.fallback_warnings.copy()
    
    def is_available(self, feature: str) -> bool:
        """Check if a feature is available."""
        return self.available_features.get(feature, False)


# Global fallback manager instance
_fallback_manager = FallbackManager()


def get_fallback_manager() -> FallbackManager:
    """Get the global fallback manager."""
    return _fallback_manager


class NumpyFallback:
    """Fallback implementations for numpy functions using pure Python."""
    
    @staticmethod
    def array(data) -> List:
        """Create array from data."""
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]
    
    @staticmethod
    def zeros(shape) -> List:
        """Create array of zeros."""
        if isinstance(shape, int):
            return [0.0] * shape
        elif isinstance(shape, (list, tuple)) and len(shape) == 1:
            return [0.0] * shape[0]
        else:
            # Multi-dimensional not fully supported
            total_size = 1
            for dim in shape:
                total_size *= dim
            return [0.0] * total_size
    
    @staticmethod
    def ones(shape) -> List:
        """Create array of ones."""
        if isinstance(shape, int):
            return [1.0] * shape
        elif isinstance(shape, (list, tuple)) and len(shape) == 1:
            return [1.0] * shape[0]
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
            return [1.0] * total_size
    
    @staticmethod
    def random_normal(shape, mean=0.0, std=1.0) -> List:
        """Generate random normal values."""
        def normal_random():
            # Box-Muller transform for normal distribution
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return mean + std * z0
        
        if isinstance(shape, int):
            return [normal_random() for _ in range(shape)]
        elif isinstance(shape, (list, tuple)) and len(shape) == 1:
            return [normal_random() for _ in range(shape[0])]
        else:
            total_size = 1
            for dim in shape:
                total_size *= dim
            return [normal_random() for _ in range(total_size)]
    
    @staticmethod
    def clip(values, min_val, max_val) -> List:
        """Clip values to range."""
        if isinstance(values, (int, float)):
            return max(min_val, min(max_val, values))
        return [max(min_val, min(max_val, v)) for v in values]
    
    @staticmethod
    def mean(values) -> float:
        """Calculate mean."""
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def std(values) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0
        mean_val = NumpyFallback.mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    @staticmethod
    def dot(a, b) -> float:
        """Dot product."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a * b
        return sum(x * y for x, y in zip(a, b))


class GymnasiumFallback:
    """Fallback implementations for gymnasium spaces."""
    
    class Box:
        """Fallback for gymnasium.spaces.Box."""
        
        def __init__(self, low, high, shape=None, dtype=float):
            self.low = low
            self.high = high
            self.shape = shape if shape else (1,)
            self.dtype = dtype
        
        def sample(self):
            """Sample random value from space."""
            if isinstance(self.shape, int):
                size = self.shape
            else:
                size = self.shape[0] if len(self.shape) == 1 else self.shape[0]
            
            if isinstance(self.low, (int, float)) and isinstance(self.high, (int, float)):
                return [random.uniform(self.low, self.high) for _ in range(size)]
            else:
                # Assume low/high are lists
                return [random.uniform(l, h) for l, h in zip(self.low, self.high)]
        
        def contains(self, x):
            """Check if x is in space."""
            if not hasattr(x, '__len__'):
                return self.low <= x <= self.high
            
            if len(x) != (self.shape[0] if hasattr(self.shape, '__len__') else self.shape):
                return False
            
            if isinstance(self.low, (int, float)) and isinstance(self.high, (int, float)):
                return all(self.low <= xi <= self.high for xi in x)
            else:
                return all(l <= xi <= h for xi, l, h in zip(x, self.low, self.high))
    
    class Discrete:
        """Fallback for gymnasium.spaces.Discrete."""
        
        def __init__(self, n):
            self.n = n
            self.shape = ()
        
        def sample(self):
            """Sample random integer."""
            return random.randint(0, self.n - 1)
        
        def contains(self, x):
            """Check if x is valid discrete value."""
            return isinstance(x, int) and 0 <= x < self.n


class VisualizationFallback:
    """Fallback for visualization when matplotlib is not available."""
    
    def __init__(self):
        self.data_buffer = []
    
    def plot(self, x, y, label=None, **kwargs):
        """Store plot data instead of plotting."""
        self.data_buffer.append({
            'type': 'plot',
            'x': x,
            'y': y,
            'label': label,
            'timestamp': time.time()
        })
        print(f"[PLOT] {label or 'Data'}: {len(x)} points (min: {min(y):.2f}, max: {max(y):.2f})")
    
    def histogram(self, data, bins=10, label=None, **kwargs):
        """Create text-based histogram."""
        if not data:
            print("[HISTOGRAM] No data provided")
            return
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        
        if range_val == 0:
            print(f"[HISTOGRAM] {label or 'Data'}: All values equal to {min_val}")
            return
        
        # Create bins
        bin_width = range_val / bins
        hist_bins = [0] * bins
        
        for value in data:
            bin_idx = min(int((value - min_val) / bin_width), bins - 1)
            hist_bins[bin_idx] += 1
        
        # Print histogram
        max_count = max(hist_bins)
        scale = 50 / max_count if max_count > 0 else 1
        
        print(f"[HISTOGRAM] {label or 'Data'} ({len(data)} samples)")
        for i, count in enumerate(hist_bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bar = '#' * int(count * scale)
            print(f"  {bin_start:.2f}-{bin_end:.2f}: {bar} ({count})")
    
    def scatter(self, x, y, label=None, **kwargs):
        """Store scatter data."""
        self.data_buffer.append({
            'type': 'scatter',
            'x': x,
            'y': y,
            'label': label,
            'timestamp': time.time()
        })
        print(f"[SCATTER] {label or 'Data'}: {len(x)} points")
    
    def show(self):
        """Show message instead of displaying plot."""
        print("[PLOT] Visualization complete (matplotlib not available for display)")
    
    def save_data(self, filename):
        """Save plot data to file."""
        import json
        with open(filename, 'w') as f:
            json.dump(self.data_buffer, f, indent=2)
        print(f"[PLOT] Data saved to {filename}")


class SimpleStatistics:
    """Simple statistics calculations without external dependencies."""
    
    @staticmethod
    def describe(data) -> Dict[str, float]:
        """Descriptive statistics."""
        if not data:
            return {'error': 'No data provided'}
        
        sorted_data = sorted(data)
        n = len(data)
        
        stats = {
            'count': n,
            'mean': sum(data) / n,
            'min': sorted_data[0],
            'max': sorted_data[-1],
            'median': sorted_data[n//2] if n % 2 == 1 else (sorted_data[n//2-1] + sorted_data[n//2]) / 2
        }
        
        # Standard deviation
        mean_val = stats['mean']
        variance = sum((x - mean_val) ** 2 for x in data) / (n - 1) if n > 1 else 0
        stats['std'] = math.sqrt(variance)
        
        # Quartiles
        if n >= 4:
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            stats['q1'] = sorted_data[q1_idx]
            stats['q3'] = sorted_data[q3_idx]
            stats['iqr'] = stats['q3'] - stats['q1']
        
        return stats
    
    @staticmethod
    def correlation(x, y) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x)
        den_y = sum((yi - mean_y) ** 2 for yi in y)
        
        if den_x == 0 or den_y == 0:
            return 0.0
        
        return num / math.sqrt(den_x * den_y)
    
    @staticmethod
    def linear_regression(x, y) -> Tuple[float, float]:
        """Simple linear regression (slope, intercept)."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0, 0.0
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = sum((xi - mean_x) ** 2 for xi in x)
        
        if den == 0:
            return 0.0, mean_y
        
        slope = num / den
        intercept = mean_y - slope * mean_x
        
        return slope, intercept


class LightweightCSVWriter:
    """Simple CSV writer without pandas."""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.file = open(filename, 'w')
        self.header_written = False
    
    def write_header(self, columns: List[str]):
        """Write CSV header."""
        self.file.write(','.join(columns) + '\n')
        self.header_written = True
    
    def write_row(self, row: Dict[str, Any]):
        """Write a single row."""
        if not self.header_written:
            self.write_header(list(row.keys()))
        
        values = [str(row.get(col, '')) for col in row.keys()]
        self.file.write(','.join(values) + '\n')
    
    def write_rows(self, rows: List[Dict[str, Any]]):
        """Write multiple rows."""
        for row in rows:
            self.write_row(row)
    
    def close(self):
        """Close the file."""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ConfigurationManager:
    """Manage configuration with fallbacks."""
    
    def __init__(self):
        self.config = {}
        self.fallback_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'environment': {
                'type': 'lightweight',
                'crew_size': 4,
                'max_episode_steps': 1000,
                'reward_scale': 1.0
            },
            'training': {
                'algorithm': 'heuristic',
                'total_timesteps': 10000,
                'log_interval': 100,
                'save_interval': 1000
            },
            'monitoring': {
                'enable_performance_tracking': True,
                'enable_health_checks': True,
                'alert_thresholds': {
                    'memory_usage': 90,
                    'error_rate': 0.1
                }
            },
            'fallbacks': {
                'use_lightweight_mode': True,
                'skip_visualization': False,
                'minimal_logging': False
            }
        }
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file with fallbacks."""
        if config_path and Path(config_path).exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with default config
                self.config = self._merge_configs(self.fallback_config, file_config)
                print(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                print(f"Failed to load config from {config_path}: {e}")
                print("Using default configuration")
                self.config = self.fallback_config.copy()
        else:
            print("No config file provided or file not found. Using default configuration.")
            self.config = self.fallback_config.copy()
        
        return self.config
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Recursively merge configurations."""
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default=None):
        """Get configuration value with fallback."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


def create_fallback_environment():
    """Create environment with fallback mechanisms."""
    manager = get_fallback_manager()
    
    if manager.is_available('gymnasium'):
        print("Using full gymnasium environment")
        from ..environments.habitat_base import LunarHabitatEnv
        return LunarHabitatEnv()
    else:
        print("Using lightweight fallback environment")
        from ..environments.lightweight_habitat import LunarHabitatEnv
        return LunarHabitatEnv()


def create_fallback_agent(agent_type: str = 'heuristic', **kwargs):
    """Create agent with fallback mechanisms."""
    manager = get_fallback_manager()
    
    try:
        if manager.is_available('deep_learning') and agent_type in ['dqn', 'ppo', 'sac']:
            print(f"Using deep learning agent: {agent_type}")
            # Would import actual RL algorithms here
            from ..algorithms.lightweight_baselines import HeuristicAgent
            return HeuristicAgent(**kwargs)
        else:
            print(f"Using lightweight baseline agent: {agent_type}")
            from ..algorithms.lightweight_baselines import get_baseline_agent
            return get_baseline_agent(agent_type, **kwargs)
    
    except Exception as e:
        print(f"Failed to create {agent_type} agent: {e}")
        print("Falling back to random agent")
        from ..algorithms.lightweight_baselines import RandomAgent
        return RandomAgent(**kwargs)


def safe_import_with_fallback(module_name: str, fallback_class=None):
    """Safely import module with fallback."""
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        if fallback_class:
            print(f"Using fallback implementation")
            return fallback_class
        else:
            print("No fallback available")
            return None


# Convenience functions for common fallbacks
def safe_numpy():
    """Get numpy or fallback."""
    try:
        import numpy as np
        return np
    except ImportError:
        return NumpyFallback()


def safe_visualization():
    """Get matplotlib or fallback."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return VisualizationFallback()


def safe_spaces():
    """Get gymnasium spaces or fallback."""
    try:
        import gymnasium.spaces as spaces
        return spaces
    except ImportError:
        return GymnasiumFallback()


# Global configuration manager
_config_manager = ConfigurationManager()


def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager."""
    return _config_manager