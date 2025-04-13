# Neurotheque Tests

This directory contains the test suite for the Neurotheque pipeline. The tests are designed to ensure the functionality, reliability, and robustness of the EEG processing pipeline.

## Test Structure

The test suite is organized into two main categories:

- **Unit Tests** (`tests/unit/`): Test individual components (steps, utility functions) in isolation
- **Integration Tests** (`tests/integration/`): Test multiple components working together

### Unit Tests

Unit tests focus on testing individual steps and utility functions in isolation. Each test file generally corresponds to a specific module or class in the codebase:

- `test_autoreject_step.py` - Tests for the AutoRejectStep
- `test_filter_step.py` - Tests for the FilterStep
- `test_ica_step.py` - Tests for the ICA-related steps
- `test_epoching_step.py` - Tests for the EpochingStep
- `test_pipeline.py` - Tests for the Pipeline class
- `test_autoreject_utils.py` - Tests for the autoreject utility functions

### Integration Tests

Integration tests verify that multiple components of the system work together correctly:

- `test_pipeline_integration.py` - Tests for the Pipeline running with different configurations
- `test_workflow_integration.py` - Tests for complete workflows that combine multiple processing steps

## Running Tests

Several options are available for running tests:

### Using the Test Runner Script

The main test runner script (`run_tests.py`) is located in the project root directory and provides several options:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run only integration tests
python run_tests.py --integration

# Run tests with coverage reporting
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --html

# Run tests matching a specific pattern
python run_tests.py --specific=autoreject
```

### Running Individual Test Files

You can also run individual test files directly:

```bash
# Run a specific test file
python -m unittest tests/unit/test_filter_step.py

# Run a specific test case or method
python -m unittest tests.unit.test_filter_step.TestFilterStep.test_bandpass_filter
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Choose the Right Category**: Decide whether you're testing a single component (unit test) or multiple components working together (integration test).

2. **Name Conventions**:
   - Unit test files should be named `test_<module_name>.py`
   - Test classes should be named `Test<ClassName>`
   - Test methods should be named `test_<functionality>`

3. **Test Structure**:
   - Use `setUp` and `tearDown` methods for test-specific setup and cleanup
   - Use `setUpClass` and `tearDownClass` for class-level setup and cleanup
   - Include docstrings for tests explaining what's being tested

4. **Assertions**:
   - Be specific with assertions (use `assertEqual` rather than `assertTrue(a == b)`)
   - Test both positive and negative scenarios
   - For complex objects, test specific properties rather than simple equality

5. **Fixtures**:
   - Use temporary directories for test outputs (`tempfile.mkdtemp()`)
   - Clean up after tests to avoid file system pollution
   - Use the MNE sample dataset for EEG data in tests

6. **Mocking**:
   - Use the `unittest.mock` module to mock external dependencies
   - For steps that require user interaction, mock the input/output
   - For file operations, use in-memory or temporary files when possible

## Test Coverage

To ensure a robust test suite, aim for high test coverage. Key areas to cover include:

- **Step Execution**: Test each step with various configurations
- **Error Handling**: Test with invalid inputs and edge cases
- **Pipeline Flows**: Test different sequences of steps
- **File Operations**: Test loading/saving data in various formats
- **Parameter Variations**: Test with different parameter values

Run the coverage report to identify areas that need additional testing:

```bash
python run_tests.py --coverage
```

## Continuous Integration

In a continuous integration environment, you can run the tests using:

```bash
python run_tests.py
```

The script will return a non-zero exit code if any tests fail, making it suitable for CI pipelines. 