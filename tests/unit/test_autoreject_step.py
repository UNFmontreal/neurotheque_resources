import unittest
import os
import numpy as np
import mne
from mne.datasets import sample
from pathlib import Path
import sys
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the step to test
from scr.steps.autoreject import AutoRejectStep

class TestAutoRejectStep(unittest.TestCase):
    """Unit tests for the AutoRejectStep class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Create temporary directory for test output
        cls.temp_dir = tempfile.mkdtemp()
        
        # Load sample data from MNE
        data_path = sample.data_path()
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        cls.raw = mne.io.read_raw_fif(raw_fname, preload=True)
        
        # Crop to reduce computation time
        cls.raw.crop(tmin=0, tmax=10)
        
        # Create simple events for epoching
        cls.events = mne.make_fixed_length_events(cls.raw, id=1, start=0, duration=1.0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up test variables before each test."""
        # Create epochs for testing
        self.epochs = mne.Epochs(
            self.raw, 
            self.events, 
            tmin=0, 
            tmax=1.0, 
            baseline=None, 
            preload=True
        )
        
        # Basic parameters for the step
        self.basic_params = {
            "ar_params": {"n_interpolate": [1, 4, 8], "consensus": [0.2, 0.5, 0.7]},
            "subject_id": "01",
            "session_id": "001",
            "output_dir": self.temp_dir,
            "plot_results": False
        }
    
    def test_init(self):
        """Test initialization of the step."""
        step = AutoRejectStep(self.basic_params)
        self.assertEqual(step.params, self.basic_params)
    
    def test_fit_mode(self):
        """Test AutoRejectStep in 'fit' mode with raw data."""
        # Configure the step
        params = self.basic_params.copy()
        params["mode"] = "fit"
        
        # Create and run the step
        step = AutoRejectStep(params)
        raw_result = step.run(self.raw.copy())
        
        # Check that the result is a Raw object
        self.assertIsInstance(raw_result, mne.io.Raw)
        
        # Check that annotations were added
        self.assertGreaterEqual(len(raw_result.annotations), len(self.raw.annotations))
    
    def test_fit_transform_mode(self):
        """Test AutoRejectStep in 'fit_transform' mode with epochs."""
        # Configure the step
        params = self.basic_params.copy()
        params["mode"] = "fit_transform"
        
        # Create and run the step
        step = AutoRejectStep(params)
        epochs_result = step.run(self.epochs.copy())
        
        # In fit_transform mode with epochs input, we should get epochs back
        self.assertIsInstance(epochs_result, mne.Epochs)
        
        # The number of epochs might be reduced due to rejection
        self.assertLessEqual(len(epochs_result), len(self.epochs))
    
    def test_save_model(self):
        """Test saving the AutoReject model."""
        # Configure the step
        params = self.basic_params.copy()
        params["save_model"] = True
        params["model_filename"] = "test_autoreject_model.pkl"
        
        # Create and run the step
        step = AutoRejectStep(params)
        step.run(self.epochs.copy())
        
        # Check if model file exists
        model_path = Path(self.temp_dir) / "test_autoreject_model.pkl"
        self.assertTrue(model_path.exists())
    
    def test_store_reject_log(self):
        """Test storing reject log in info."""
        # Configure the step
        params = self.basic_params.copy()
        params["store_reject_log"] = True
        
        # Create and run the step
        step = AutoRejectStep(params)
        result = step.run(self.epochs.copy())
        
        # Check if reject log was stored
        self.assertIn("temp", result.info)
        self.assertIn("autoreject", result.info["temp"])

if __name__ == '__main__':
    unittest.main() 