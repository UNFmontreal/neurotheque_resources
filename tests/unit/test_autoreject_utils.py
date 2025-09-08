import unittest
import os
import tempfile
import shutil
import pickle
import numpy as np
import mne
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the utils to test
from scr.utils.autoreject_utils import (find_autoreject_log, load_autoreject_log, 
                                      plot_autoreject_summary, plot_reject_log,
                                      scan_for_autoreject_logs)

# Mock RejectLog class for testing
class MockRejectLog:
    """Mock RejectLog class with the necessary attributes for testing."""
    
    def __init__(self, n_epochs=10, n_channels=5, bad_epochs=None, labels=None, ch_names=None):
        """Initialize a mock RejectLog with configurable properties."""
        self.n_epochs = n_epochs
        self.n_channels = n_channels
        
        # Set bad_epochs
        if bad_epochs is None:
            self.bad_epochs = np.zeros(n_epochs, dtype=bool)
            self.bad_epochs[::3] = True  # Mark every 3rd epoch as bad
        else:
            self.bad_epochs = np.array(bad_epochs, dtype=bool)
        
        # Set channel names
        if ch_names is None:
            self.ch_names = [f'EEG{i:03d}' for i in range(1, n_channels + 1)]
        else:
            self.ch_names = ch_names
        
        # Set labels matrix (0=good, 1=interpolated, 2=bad)
        if labels is None:
            self.labels = np.zeros((n_epochs, n_channels), dtype=int)
            # Mark some channels as interpolated or bad
            for i in range(n_epochs):
                if i % 3 == 0:  # Every 3rd epoch has bad channels
                    self.labels[i, :] = 2  # All channels marked bad in bad epochs
                elif i % 3 == 1:  # Every other epoch has some interpolated
                    self.labels[i, 1::2] = 1  # Interpolate every other channel
        else:
            self.labels = labels
    
    def plot(self, orientation='horizontal', show_names=False, show=True):
        """Mock plot method returning a matplotlib figure."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(self.labels)
        return fig

class TestAutorejectUtils(unittest.TestCase):
    """Unit tests for autoreject utility functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Create temporary directory for test output
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create subdirectories
        cls.processed_dir = os.path.join(cls.temp_dir, "processed")
        cls.derivatives_dir = os.path.join(cls.temp_dir, "derivatives")
        
        os.makedirs(cls.processed_dir, exist_ok=True)
        os.makedirs(cls.derivatives_dir, exist_ok=True)
        
        # Create subject directories
        cls.sub01_dir = os.path.join(cls.processed_dir, "sub-01", "ses-001")
        cls.sub02_dir = os.path.join(cls.processed_dir, "sub-02", "ses-001")
        cls.sub01_auto_dir = os.path.join(cls.processed_dir, "sub-01", "ses-001", "autoreject")
        
        os.makedirs(cls.sub01_dir, exist_ok=True)
        os.makedirs(cls.sub02_dir, exist_ok=True)
        os.makedirs(cls.sub01_auto_dir, exist_ok=True)
        
        # Create autoreject logs
        cls.create_mock_log_files()
    
    @classmethod
    def create_mock_log_files(cls):
        """Create mock autoreject log files for testing."""
        # Create a mock RejectLog
        reject_log1 = MockRejectLog(n_epochs=10, n_channels=5)
        reject_log2 = MockRejectLog(n_epochs=15, n_channels=8, 
                                  bad_epochs=[True, False, True, False, True, False, True, 
                                            False, True, False, True, False, True, False, True])
        
        # Save as pickle
        file1 = os.path.join(cls.sub01_auto_dir, "sub-01_ses-001_autoreject_log.pickle")
        with open(file1, 'wb') as f:
            pickle.dump(reject_log1, f)
        
        file2 = os.path.join(cls.sub01_dir, "sub-01_ses-001_task-test_autoreject_log.pickle")
        with open(file2, 'wb') as f:
            pickle.dump(reject_log2, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        plt.close('all')  # Close any open plot windows
    
    def test_find_autoreject_log(self):
        """Test finding autoreject log files."""
        # Test finding a file that exists
        found_file = find_autoreject_log(
            subject_id="01", 
            session_id="001", 
            base_dir=self.temp_dir
        )
        
        self.assertIsNotNone(found_file)
        self.assertTrue(os.path.exists(found_file))
        
        # Test finding a file with task included
        found_file = find_autoreject_log(
            subject_id="01", 
            session_id="001",
            task_id="test", 
            base_dir=self.temp_dir
        )
        
        self.assertIsNotNone(found_file)
        self.assertTrue(os.path.exists(found_file))
        self.assertIn("task-test", found_file)
        
        # Test finding a nonexistent file
        found_file = find_autoreject_log(
            subject_id="01", 
            session_id="002",  # Doesn't exist
            base_dir=self.temp_dir
        )
        
        self.assertIsNone(found_file)
        
        # Test with 'sub-' prefix in subject_id
        found_file = find_autoreject_log(
            subject_id="sub-01", 
            session_id="001", 
            base_dir=self.temp_dir
        )
        
        self.assertIsNotNone(found_file)
        self.assertTrue(os.path.exists(found_file))
    
    def test_load_autoreject_log(self):
        """Test loading autoreject log files."""
        # Test loading with filepath
        file_path = os.path.join(self.sub01_auto_dir, "sub-01_ses-001_autoreject_log.pickle")
        log = load_autoreject_log(filepath=file_path)
        
        self.assertIsNotNone(log)
        self.assertTrue(hasattr(log, 'bad_epochs'))
        self.assertTrue(hasattr(log, 'ch_names'))
        
        # Test loading with subject/session
        log = load_autoreject_log(
            subject_id="01", 
            session_id="001", 
            base_dir=self.temp_dir
        )
        
        self.assertIsNotNone(log)
        self.assertTrue(hasattr(log, 'bad_epochs'))
        
        # Test loading with task
        log = load_autoreject_log(
            subject_id="01", 
            session_id="001",
            task_id="test", 
            base_dir=self.temp_dir
        )
        
        self.assertIsNotNone(log)
        self.assertTrue(hasattr(log, 'bad_epochs'))
        
        # Test loading nonexistent file
        with self.assertRaises(ValueError):
            load_autoreject_log(
                subject_id="01", 
                session_id="002",  # Doesn't exist
                base_dir=self.temp_dir
            )
    
    def test_plot_autoreject_summary(self):
        """Test plotting autoreject summary."""
        # Load a reject log
        file_path = os.path.join(self.sub01_auto_dir, "sub-01_ses-001_autoreject_log.pickle")
        log = load_autoreject_log(filepath=file_path)
        
        # Test with default parameters
        fig = plot_autoreject_summary(log)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
        # Test with save_to parameter
        output_file = os.path.join(self.temp_dir, "test_summary.png")
        fig = plot_autoreject_summary(log, save_to=output_file)
        self.assertTrue(os.path.exists(output_file))
        plt.close(fig)
        
        # Test with None input
        fig = plot_autoreject_summary(None)
        self.assertIsNone(fig)
    
    def test_plot_reject_log(self):
        """Test plotting reject log."""
        # Load a reject log
        file_path = os.path.join(self.sub01_auto_dir, "sub-01_ses-001_autoreject_log.pickle")
        log = load_autoreject_log(filepath=file_path)
        
        # Test with default parameters
        fig = plot_reject_log(log)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
        # Test with different orientation
        fig = plot_reject_log(log, orientation='vertical')
        self.assertIsNotNone(fig)
        plt.close(fig)
        
        # Test with save_to parameter
        output_file = os.path.join(self.temp_dir, "test_log.png")
        fig = plot_reject_log(log, save_to=output_file)
        self.assertTrue(os.path.exists(output_file))
        plt.close(fig)
        
        # Test with None input
        fig = plot_reject_log(None)
        self.assertIsNone(fig)
        
        # Test with a mocked plot method that raises an exception
        with patch.object(log, 'plot', side_effect=Exception("Mock error")):
            fig = plot_reject_log(log)
            self.assertIsNotNone(fig)  # Should create a fallback figure
            plt.close(fig)
    
    def test_scan_for_autoreject_logs(self):
        """Test scanning for autoreject logs."""
        # Test scanning the test directory
        logs = scan_for_autoreject_logs(self.temp_dir)
        
        # Should find at least the two logs we created
        self.assertGreaterEqual(len(logs), 2)
        
        # Test with a specific subdirectory
        logs = scan_for_autoreject_logs(self.sub01_dir)
        
        # Should find only the logs in that directory
        self.assertGreaterEqual(len(logs), 1)
        
        # Test with a nonexistent directory
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        logs = scan_for_autoreject_logs(nonexistent_dir)
        
        # Should return an empty list
        self.assertEqual(len(logs), 0)

if __name__ == '__main__':
    unittest.main() 