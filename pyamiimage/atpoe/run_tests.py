#!/usr/bin/env python3
"""
Test runner for ATPOE dashboard tests.

Run all tests or specific test modules.
"""

import sys
import unittest
from pathlib import Path

# Package should be installed with pip install -e .

def run_all_tests():
    """Run all ATPOE dashboard tests."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test(test_module):
    """Run a specific test module."""
    try:
        # Import the test module
        module_name = f"pyamiimage.atpoe.tests.{test_module}"
        test_module = __import__(module_name, fromlist=[''])
        
        # Run the tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"Error importing test module: {e}")
        return False

def main():
    """Main test runner function."""
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        print(f"Running tests for: {test_module}")
        success = run_specific_test(test_module)
    else:
        # Run all tests
        print("Running all ATPOE dashboard tests...")
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

