#!/usr/bin/env python3
"""Simple test without complex dependencies."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Test basic imports
def test_imports():
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NumPy: {e}")
        return False
    
    # Test basic state classes
    try:
        from lunar_habitat_rl.core.state import HabitatState, AtmosphereState
        print("✅ State classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import state classes: {e}")
        return False
    
    # Test configuration 
    try:
        from lunar_habitat_rl.core.config import HabitatConfig
        print("✅ Configuration classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config classes: {e}")
        return False
        
    # Test metrics
    try:
        from lunar_habitat_rl.core.metrics import MissionMetrics, PerformanceTracker
        print("✅ Metrics classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import metrics classes: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without complex dependencies."""
    
    try:
        from lunar_habitat_rl.core.state import HabitatState, AtmosphereState
        from lunar_habitat_rl.core.config import HabitatConfig
        from lunar_habitat_rl.core.metrics import MissionMetrics
        import numpy as np
        
        # Test configuration
        config = HabitatConfig()
        print(f"✅ Created habitat config: {config.name}")
        
        # Test state
        state = HabitatState(max_crew=4)
        state_array = state.to_array()
        print(f"✅ Created habitat state, array shape: {state_array.shape}")
        
        # Test atmosphere state
        atmo = AtmosphereState(
            o2_partial_pressure=21.3,
            co2_partial_pressure=0.4,
            n2_partial_pressure=79.0,
            total_pressure=101.3,
            humidity=45.0,
            temperature=22.5,
            air_quality_index=0.95
        )
        atmo_array = atmo.to_array()
        print(f"✅ Created atmosphere state, array shape: {atmo_array.shape}")
        
        # Test metrics
        metrics = MissionMetrics()
        score = metrics.compute_overall_score()
        print(f"✅ Created metrics, overall score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🌙 Lunar Habitat RL - Simple Tests")
    print("=" * 40)
    
    success = True
    
    print("\n📦 Testing imports...")
    if not test_imports():
        success = False
    
    print("\n🔧 Testing basic functionality...")  
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All simple tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())