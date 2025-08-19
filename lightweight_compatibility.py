
"""
Compatibility layer for lightweight dependencies.
Automatically imports available modules or lightweight replacements.
"""

import sys
import importlib
from pathlib import Path

# Add current directory to path for lightweight modules
sys.path.insert(0, str(Path(__file__).parent))

# Compatibility imports
try:
    import pydantic
except ImportError:
    try:
        import lightweight_pydantic as pydantic
        sys.modules['pydantic'] = pydantic
    except ImportError:
        pass

try:
    import pytest
except ImportError:
    try:
        import lightweight_pytest as pytest
        sys.modules['pytest'] = pytest
    except ImportError:
        pass

try:
    import gymnasium
except ImportError:
    try:
        import lightweight_gymnasium as gymnasium
        sys.modules['gymnasium'] = gymnasium
    except ImportError:
        pass

# Make numpy available if possible
try:
    import numpy as np
except ImportError:
    # Minimal numpy-like functionality
    class MinimalNumpy:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def random():
            import random
            class RandomModule:
                @staticmethod
                def uniform(low, high, size=None):
                    if size:
                        return [random.uniform(low, high) for _ in range(size)]
                    return random.uniform(low, high)
                
                @staticmethod
                def randint(low, high):
                    return random.randint(low, high-1)
                
                @staticmethod
                def normal(mean, std):
                    return random.gauss(mean, std)
            
            return RandomModule()
        
        float32 = float
        inf = float('inf')
    
    np = MinimalNumpy()
    sys.modules['numpy'] = np
