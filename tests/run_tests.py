#!/usr/bin/env python3
"""
Test runner script for NeuroTheque pipeline.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Find the test directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover and run all tests
    print(f"Discovering tests in {test_dir}")
    test_suite = unittest.defaultTestLoader.discover(test_dir)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

def run_unit_tests():
    """Run only unit tests."""
    # Find the unit test directory
    unit_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unit')
    
    # Discover and run unit tests
    print(f"Discovering unit tests in {unit_test_dir}")
    test_suite = unittest.defaultTestLoader.discover(unit_test_dir)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

def run_integration_tests():
    """Run only integration tests."""
    # Find the integration test directory
    integration_test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'integration')
    
    # Discover and run integration tests
    print(f"Discovering integration tests in {integration_test_dir}")
    test_suite = unittest.defaultTestLoader.discover(integration_test_dir)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for NeuroTheque pipeline')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    args = parser.parse_args()
    
    if args.unit:
        sys.exit(run_unit_tests())
    elif args.integration:
        sys.exit(run_integration_tests())
    else:
        sys.exit(run_all_tests()) 