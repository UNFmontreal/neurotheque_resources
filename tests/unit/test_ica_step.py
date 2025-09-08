import unittest
import os
import numpy as np
import mne
from mne.datasets import sample
from pathlib import Path
import tempfile
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the steps to test
from scr.steps.ica import ICAStep
from scr.steps.ica_extraction import ICAExtractionStep
from scr.steps.ica_labeling import ICALabelingStep

# Mock ProjectPaths class to avoid complex directory dependencies
class MockProjectPaths:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        
    def get_derivative_path(self, subject_id, session_id):
        """Mock implementation of get_derivative_path."""
        return self.base_dir / f'sub-{subject_id}' / f'ses-{session_id}'
    
    def get_ica_report_dir(self, subject_id, session_id):
        """Mock implementation of get_ica_report_dir."""
        return self.base_dir / f'sub-{subject_id}' / f'ses-{session_id}' / 'ica_reports'

class TestICAStep(unittest.TestCase):
    """Unit tests for the ICA-related steps."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Create temporary directory for test output
        cls.temp_dir = tempfile.mkdtemp()
        
        # Load sample data from MNE
        data_path = sample.data_path()
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        cls.raw = mne.io.read_raw_fif(raw_fname, preload=True)
        
        # Pick only EEG channels and rename a few to match typical EEG names
        cls.raw.pick_types(eeg=True, exclude=[])
        rename_dict = {
            'EEG 001': 'Fp1',
            'EEG 002': 'Fp2',
            'EEG 003': 'F7',
            'EEG 004': 'F8'
        }
        cls.raw.rename_channels(rename_dict)
        
        # Crop to reduce computation time
        cls.raw.crop(tmin=0, tmax=10)
        
        # Create epochs for testing
        events = mne.make_fixed_length_events(cls.raw, duration=1.0)
        cls.epochs = mne.Epochs(
            cls.raw,
            events,
            tmin=0,
            tmax=1.0,
            baseline=None,
            preload=True
        )
        
        # Create mock paths object
        cls.paths = MockProjectPaths(cls.temp_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up test variables before each test."""
        # Create test directories
        os.makedirs(os.path.join(self.temp_dir, 'sub-01', 'ses-001'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'sub-01', 'ses-001', 'ica_reports', 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'sub-01', 'ses-001', 'ica_reports', 'components'), exist_ok=True)
        
        # Basic parameters for ICA
        self.basic_params = {
            "n_components": 10,
            "method": "fastica",
            "max_iter": 200,
            "fit_params": {"extended": False},
            "subject_id": "01",
            "session_id": "001",
            "eog_ch_names": ["Fp1", "Fp2"],
            "plot_dir": os.path.join(self.temp_dir, "ica_plots"),
            "interactive": False,
            "plot_components": False,
            "plot_sources": False,
            "paths": self.paths
        }
    
    def test_ica_extraction_init(self):
        """Test initialization of ICAExtractionStep."""
        step = ICAExtractionStep(self.basic_params)
        self.assertEqual(step.params, self.basic_params)
    
    def test_ica_extraction_run(self):
        """Test the ICA extraction process."""
        step = ICAExtractionStep(self.basic_params)
        
        # Add annotations to test use_good_epochs_only
        annotation = mne.Annotations(
            onset=[5],
            duration=[1],
            description=['BAD_autoreject']
        )
        raw_with_annot = self.raw.copy()
        raw_with_annot.set_annotations(annotation)
        
        result = step.run(raw_with_annot)
        
        # Check that the ICA object was created
        self.assertIsInstance(result, dict)
        self.assertIn('ica', result)
        self.assertIn('raw', result)
        self.assertIsInstance(result['ica'], mne.preprocessing.ICA)
        
        # Check that the correct number of components was extracted
        self.assertEqual(result['ica'].n_components_, 10)
    
    def test_ica_labeling_eog(self):
        """Test ICA component labeling with EOG detection."""
        # First run extraction
        extraction_step = ICAExtractionStep(self.basic_params)
        extraction_result = extraction_step.run(self.raw.copy())
        
        # Then run labeling with EOG detection
        labeling_params = {
            "subject_id": "01",
            "session_id": "001",
            "eog_ch_names": ["Fp1", "Fp2"],
            "eog_threshold": 0.3,
            "ecg_threshold": 0.3,
            "interactive": False,
            "plot_components": False,
            "save_labeled_components": True,
            "output_dir": os.path.join(self.temp_dir, "ica_components"),
            "paths": self.paths
        }
        
        labeling_step = ICALabelingStep(labeling_params)
        
        # Create input dict
        labeled_result = labeling_step.run(extraction_result)
        
        # Verify that labeling worked
        self.assertIn('ica', labeled_result)
        self.assertIn('raw', labeled_result)
        
        # Check that exclude attribute was filled
        ica = labeled_result['ica']
        # We can't guarantee components will be excluded, but the attribute should exist
        self.assertTrue(hasattr(ica, 'exclude'))
    
    def test_combined_ica_step(self):
        """Test the combined ICAStep class."""
        params = self.basic_params.copy()
        params["output_dir"] = os.path.join(self.temp_dir, "ica_output")
        os.makedirs(params["output_dir"], exist_ok=True)
        
        step = ICAStep(params)
        cleaned_raw = step.run(self.raw.copy())
        
        # Check that the cleaned data is returned
        self.assertIsInstance(cleaned_raw, mne.io.Raw)
        
        # Check that ICA metadata was stored
        self.assertIn("temp", cleaned_raw.info)
        self.assertIn("ica", cleaned_raw.info["temp"])
        
        # Check that the cleaned data file was saved
        ica_dir = self.paths.get_derivative_path("01", "001") / 'sub-01_ses-001_desc-ica_cleaned.fif'
        self.assertTrue(ica_dir.exists())
    
    def test_handling_epochs_input(self):
        """Test that ICA steps can handle epoched data input."""
        # Use the extraction step with epochs
        extraction_params = self.basic_params.copy()
        extraction_step = ICAExtractionStep(extraction_params)
        extraction_result = extraction_step.run(self.epochs.copy())
        
        # Check that the extraction worked with epochs
        self.assertIsInstance(extraction_result, dict)
        self.assertIn('ica', extraction_result)
        self.assertIn('epochs', extraction_result)  # Should have epochs instead of raw
        
        # Now test the combined step with epochs
        ica_params = self.basic_params.copy()
        ica_step = ICAStep(ica_params)
        cleaned_epochs = ica_step.run(self.epochs.copy())
        
        # Check that the cleaned data is returned as epochs
        self.assertIsInstance(cleaned_epochs, mne.Epochs)
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        step = ICAStep(self.basic_params)
        
        # Test with None as input
        with self.assertRaises(ValueError):
            step.run(None)
        
        # Test with epochs with no data
        empty_epochs = mne.Epochs(
            self.raw, 
            np.array([[100, 0, 1]]),  # Events that are out of bounds
            tmin=0, 
            tmax=1.0, 
            baseline=None, 
            preload=True,
            reject_by_annotation=True
        )
        
        if len(empty_epochs) == 0:  # Only test if epochs are actually empty
            with self.assertRaises(Exception):  # Some kind of error should be raised
                step.run(empty_epochs)

if __name__ == '__main__':
    unittest.main() 