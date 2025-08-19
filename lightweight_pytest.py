
"""Lightweight pytest replacement for basic testing."""

import sys
import traceback
from typing import Callable, List, Any


class TestResult:
    """Test execution result."""
    
    def __init__(self, name: str, passed: bool, error: str = ""):
        self.name = name
        self.passed = passed
        self.error = error


class LightweightPytest:
    """Minimal pytest-like test runner."""
    
    def __init__(self):
        self.tests: List[Callable] = []
        self.results: List[TestResult] = []
    
    def add_test(self, test_func: Callable):
        """Add a test function."""
        self.tests.append(test_func)
    
    def run_tests(self) -> bool:
        """Run all tests and return success status."""
        self.results = []
        all_passed = True
        
        for test_func in self.tests:
            try:
                test_func()
                result = TestResult(test_func.__name__, True)
                print(f"✅ {test_func.__name__} PASSED")
            except Exception as e:
                result = TestResult(test_func.__name__, False, str(e))
                print(f"❌ {test_func.__name__} FAILED: {e}")
                all_passed = False
            
            self.results.append(result)
        
        return all_passed


# Global test runner instance
_pytest = LightweightPytest()


def test(func: Callable) -> Callable:
    """Test decorator."""
    _pytest.add_test(func)
    return func


def main():
    """Main pytest entry point."""
    success = _pytest.run_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
