import unittest
import os
import tempfile
import shutil
import yaml
import sys
import mne
import numpy as np
from pathlib import Path
from mne.datasets import sample

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the pipeline
from scr.pipeline import Pipeline

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the Pipeline class."""
    
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
        
        # Load sample data from MNE
        data_path = sample.data_path()
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        cls.raw = mne.io.read_raw_fif(raw_fname, preload=True)
        
        # Crop to reduce computation time
        cls.raw.crop(tmin=0, tmax=10)
        
        # Save test data to raw directory
        cls.test_file = os.path.join(cls.raw_dir, "sub-01_ses-001_task-test_run-01_raw.fif")
        cls.raw.save(cls.test_file, overwrite=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up a test configuration."""
        # Create a basic pipeline configuration
        self.config = {
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports",
                "derivatives_dir": "derivatives"
            },
            "pipeline_mode": "restart",  # Always start from scratch
            "default_subject": "01",
            "default_session": "001",
            "default_run": "01",
            "auto_save": True,
            "file_path_pattern": os.path.join(self.raw_dir, "sub-01_ses-001_*_raw.fif"),
            "pipeline": {
                "steps": [
                    {
                        "name": "LoadData",
                        "params": {
                            "file_path_pattern": os.path.join(self.raw_dir, "sub-01_ses-001_*_raw.fif")
                        }
                    },
                    {
                        "name": "FilterStep",
                        "params": {
                            "l_freq": 1.0,
                            "h_freq": 40.0
                        }
                    },
                    {
                        "name": "AutoRejectStep",
                        "params": {
                            "ar_params": {"n_interpolate": [1, 4], "consensus": [0.2, 0.5]},
                            "mode": "fit",
                            "plot_results": False
                        }
                    },
                    {
                        "name": "SaveCheckpoint",
                        "params": {
                            "stage": "after_autoreject"
                        }
                    }
                ]
            }
        }
        
        # Write the configuration to a file
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def test_pipeline_execution(self):
        """Test that the pipeline can execute all steps successfully."""
        # Create and run the pipeline
        pipeline = Pipeline(config_file=self.config_file)
        
        try:
            # Run the pipeline and check if there are no exceptions
            pipeline.run()
            
            # Check if the checkpoint file was created
            checkpoint_dir = os.path.join(self.processed_dir, "sub-01", "ses-001")
            checkpoint_exists = any(
                Path(checkpoint_dir).glob("sub-01_ses-001_*_after_autoreject*.fif")
            )
            self.assertTrue(checkpoint_exists, "Checkpoint file was not created")
            
        except Exception as e:
            self.fail(f"Pipeline execution failed with error: {e}")
    
    def test_resume_from_checkpoint(self):
        """Test that the pipeline can resume from a checkpoint."""
        # First run the pipeline to create a checkpoint
        pipeline = Pipeline(config_file=self.config_file)
        pipeline.run()
        
        # Modify the config to resume from a checkpoint
        self.config["pipeline_mode"] = "resume"
        
        # Create a new pipeline with the updated config
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
        
        resume_pipeline = Pipeline(config_file=self.config_file)
        
        try:
            # Run the pipeline again - it should detect and use the checkpoint
            resume_pipeline.run()
            
            # This should succeed without errors
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Pipeline resumption failed with error: {e}")

if __name__ == '__main__':
    unittest.main() 