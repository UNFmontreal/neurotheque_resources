import unittest
import os
import tempfile
import shutil
import yaml
import sys
import mne
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the pipeline
from scr.pipeline import Pipeline
from scr.registry import STEP_REGISTRY
from scr.steps.base import BaseStep

# Create a simple mock step for testing
class MockStep(BaseStep):
    """A simple mock step for testing."""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.run_called = False
    
    def run(self, data):
        self.run_called = True
        # Return either transformed data or just the original data
        if self.params.get("transform", False):
            return "transformed"
        return data

class TestPipeline(unittest.TestCase):
    """Unit tests for the Pipeline class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and directories."""
        # Create temporary directory for test output
        cls.temp_dir = tempfile.mkdtemp()
        cls.raw_dir = os.path.join(cls.temp_dir, "raw")
        cls.processed_dir = os.path.join(cls.temp_dir, "processed")
        cls.reports_dir = os.path.join(cls.temp_dir, "reports")
        cls.derivatives_dir = os.path.join(cls.temp_dir, "derivatives")
        
        # Create directories
        for directory in [cls.raw_dir, cls.processed_dir, cls.reports_dir, cls.derivatives_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Create a simple raw file for testing
        cls.simple_raw = mne.io.RawArray(
            np.random.randn(10, 10000),  # 10 channels, 10000 samples
            mne.create_info(10, 1000.0, ch_types='eeg')  # 1000 Hz sampling rate
        )
        
        # Save test data to raw directory
        cls.test_file = os.path.join(cls.raw_dir, "sub-01_ses-001_task-test_run-01_raw.fif")
        cls.simple_raw.save(cls.test_file, overwrite=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up test variables."""
        # Register mock step
        self.original_registry = STEP_REGISTRY.copy()
        STEP_REGISTRY["MockStep"] = MockStep
        
        # Set up a basic config
        self.config = {
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports",
                "derivatives_dir": "derivatives"
            },
            "pipeline_mode": "restart",  # Always start from scratch for tests
            "default_subject": "01",
            "default_session": "001",
            "default_run": "01",
            "auto_save": False,  # Don't auto-save by default (simplifies tests)
            "pipeline": {
                "steps": [
                    {
                        "name": "MockStep",
                        "params": {}
                    }
                ]
            }
        }
        
        # Write the configuration to a file
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore original registry
        global STEP_REGISTRY
        STEP_REGISTRY = self.original_registry
    
    def test_init_with_config_file(self):
        """Test initialization with a config file."""
        pipeline = Pipeline(config_file=self.config_file)
        
        # Check that the config was loaded
        self.assertEqual(pipeline.config["default_subject"], "01")
        self.assertEqual(pipeline.config["pipeline"]["steps"][0]["name"], "MockStep")
    
    def test_init_with_config_dict(self):
        """Test initialization with a config dictionary."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Check that the config was loaded
        self.assertEqual(pipeline.config["default_subject"], "01")
        self.assertEqual(pipeline.config["pipeline"]["steps"][0]["name"], "MockStep")
    
    def test_load_config(self):
        """Test loading config from file."""
        pipeline = Pipeline(config_file=self.config_file)
        
        # Change the config and reload
        modified_config = self.config.copy()
        modified_config["default_subject"] = "02"
        
        with open(self.config_file, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Load the modified config
        reloaded_config = pipeline._load_config(self.config_file, None)
        
        self.assertEqual(reloaded_config["default_subject"], "02")
    
    def test_find_step_index(self):
        """Test finding a step index by name."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Create a sample steps list
        steps = [
            {"name": "StepA"},
            {"name": "StepB"},
            {"name": "StepC"}
        ]
        
        # Test finding existing steps
        self.assertEqual(pipeline._find_step_index(steps, "StepA"), 0)
        self.assertEqual(pipeline._find_step_index(steps, "StepB"), 1)
        self.assertEqual(pipeline._find_step_index(steps, "StepC"), 2)
        
        # Test finding non-existent step
        self.assertIsNone(pipeline._find_step_index(steps, "StepD"))
    
    def test_parse_sub_ses(self):
        """Test parsing subject and session from filenames."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Test standard BIDS filename
        sub_id, ses_id, task_id, run_id = pipeline._parse_sub_ses(
            "sub-01_ses-002_task-test_run-03_raw.edf"
        )
        self.assertEqual(sub_id, "01")
        self.assertEqual(ses_id, "002")
        self.assertEqual(task_id, "test")
        self.assertEqual(run_id, "03")
        
        # Test filename with missing components
        sub_id, ses_id, task_id, run_id = pipeline._parse_sub_ses(
            "sub-01_run-03_raw.edf"
        )
        self.assertEqual(sub_id, "01")
        self.assertEqual(ses_id, "001")  # Default from config
        self.assertEqual(task_id, "unknown")
        self.assertEqual(run_id, "03")
        
        # Test completely non-BIDS filename
        sub_id, ses_id, task_id, run_id = pipeline._parse_sub_ses(
            "random_file.edf"
        )
        self.assertEqual(sub_id, "01")  # Default from config
        self.assertEqual(ses_id, "001")  # Default from config
        self.assertEqual(task_id, "unknown")
        self.assertEqual(run_id, "01")  # Default from config
    
    def test_run_step(self):
        """Test running a single step."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Create a mock step using our MockStep class
        step_info = {
            "name": "MockStep",
            "params": {"transform": True}
        }
        
        # Set some initial data
        pipeline.data = "initial"
        
        # Run the step
        pipeline._run_step(step_info)
        
        # Check that the data was transformed
        self.assertEqual(pipeline.data, "transformed")
    
    def test_run_step_error(self):
        """Test error handling when running a step."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Create a step with an unregistered name
        step_info = {
            "name": "NonExistentStep",
            "params": {}
        }
        
        # Running should raise ValueError
        with self.assertRaises(ValueError):
            pipeline._run_step(step_info)
    
    @patch('glob.glob')
    def test_run_multi_subject(self, mock_glob):
        """Test running the pipeline in multi-subject mode."""
        # Configure for multi-subject mode
        multi_config = self.config.copy()
        multi_config["file_path_pattern"] = os.path.join(self.raw_dir, "sub-*_ses-*_*_raw.fif")
        
        # Set up mock for file search
        mock_glob.return_value = [self.test_file]
        
        # Create a pipeline
        pipeline = Pipeline(config_dict=multi_config)
        
        # Add a more interesting sequence of steps
        pipeline.config["pipeline"]["steps"] = [
            {
                "name": "MockStep",
                "params": {"step": 1}
            },
            {
                "name": "MockStep",
                "params": {"step": 2, "transform": True}
            },
            {
                "name": "MockStep",
                "params": {"step": 3}
            }
        ]
        
        # Run the pipeline
        with patch.object(pipeline, '_run_steps') as mock_run_steps:
            pipeline.run()
            
            # Check that _run_steps was called with the correct arguments
            mock_run_steps.assert_called_once()
            args, kwargs = mock_run_steps.call_args
            
            # Check that steps were passed
            self.assertEqual(len(args[0]), 3)  # 3 steps
            self.assertEqual(args[0][0]["name"], "MockStep")
            self.assertEqual(args[0][1]["name"], "MockStep")
            self.assertEqual(args[0][2]["name"], "MockStep")
            
            # Check that parameters like subject_id were passed
            self.assertEqual(kwargs["subject_id"], "01")
            self.assertEqual(kwargs["session_id"], "001")
            self.assertEqual(kwargs["task_id"], "test")
            self.assertEqual(kwargs["run_id"], "01")
            self.assertEqual(kwargs["file_path"], self.test_file)
    
    def test_run_steps(self):
        """Test running multiple steps."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Create a sequence of steps
        steps = [
            {
                "name": "MockStep",
                "params": {"step": 1}
            },
            {
                "name": "MockStep",
                "params": {"step": 2, "transform": True}
            },
            {
                "name": "MockStep",
                "params": {"step": 3}
            }
        ]
        
        # Test that steps are executed and can modify data
        pipeline.data = "original"
        pipeline._run_steps(steps)
        
        # The second step should have transformed the data
        self.assertEqual(pipeline.data, "transformed")
    
    def test_skip_steps(self):
        """Test skipping steps based on skip_index."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Create a sequence of steps
        steps = [
            {
                "name": "MockStep",
                "params": {"step": 1}
            },
            {
                "name": "MockStep",
                "params": {"step": 2, "transform": True}
            },
            {
                "name": "MockStep",
                "params": {"step": 3}
            }
        ]
        
        # Execute the pipeline with a skip_index of 1 (skip first and second steps)
        pipeline.data = "original"
        pipeline._run_steps(steps, skip_index=1)
        
        # Only the third step should be executed, and it doesn't transform
        # So data should still be "original"
        self.assertEqual(pipeline.data, "original")
        
        # For a more thorough test, we can use our spy to check
        # that only the appropriate step's run() was called
        from unittest.mock import MagicMock
        
        # Replace the real MockStep with a mock
        original_mock_step = STEP_REGISTRY["MockStep"]
        mock_step_instance = MagicMock()
        mock_step_class = MagicMock(return_value=mock_step_instance)
        # Make sure the run method returns the input
        mock_step_instance.run.return_value = "original"
        STEP_REGISTRY["MockStep"] = mock_step_class
        
        try:
            # Run with skip_index=1
            pipeline.data = "original"
            pipeline._run_steps(steps, skip_index=1)
            
            # Check that MockStep was instantiated 3 times (for all steps)
            self.assertEqual(mock_step_class.call_count, 3)
            
            # But run should only be called once (for the 3rd step)
            self.assertEqual(mock_step_instance.run.call_count, 1)
        finally:
            # Restore original MockStep
            STEP_REGISTRY["MockStep"] = original_mock_step
    
    def test_find_latest_checkpoint(self):
        """Test finding the latest checkpoint file."""
        pipeline = Pipeline(config_dict=self.config)
        
        # Create some mock checkpoint files with different timestamps
        checkpoints_dir = os.path.join(self.processed_dir, "sub-01", "ses-001")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Create files with different dates
        import time
        
        checkpoint_files = [
            os.path.join(checkpoints_dir, "sub-01_ses-001_checkpoint1.fif"),
            os.path.join(checkpoints_dir, "sub-01_ses-001_after_filter.fif"),
            os.path.join(checkpoints_dir, "sub-01_ses-001_after_autoreject.fif")
        ]
        
        # Create the files with different timestamps
        for i, file_path in enumerate(checkpoint_files):
            with open(file_path, 'w') as f:
                f.write("Test data")
            # Set access/modify time to be sequential
            # Each file is 1 day newer than the previous
            mtime = time.time() - (3 - i) * 86400
            os.utime(file_path, (mtime, mtime))
        
        # Find the latest checkpoint
        latest = pipeline._find_latest_checkpoint("01", "001")
        
        # The latest should be the last file (highest mtime)
        self.assertEqual(Path(latest).name, Path(checkpoint_files[2]).name)

if __name__ == '__main__':
    unittest.main() 