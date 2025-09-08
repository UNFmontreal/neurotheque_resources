import unittest
import os
import numpy as np
import mne
from mne.datasets import sample
import tempfile
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the step to test
from scr.steps.filter import FilterStep

class TestFilterStep(unittest.TestCase):
    """Unit tests for the FilterStep class."""
    
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
        cls.raw.crop(tmin=0, tmax=5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test initialization with various parameters."""
        # Test with default parameters
        params = {}
        step = FilterStep(params)
        self.assertEqual(step.params, params)
        
        # Test with custom parameters
        params = {
            "l_freq": 2.0,
            "h_freq": 30.0,
            "notch_freqs": [50, 100]
        }
        step = FilterStep(params)
        self.assertEqual(step.params, params)
    
    def test_bandpass_filter(self):
        """Test basic bandpass filtering."""
        # Parameters for bandpass filter
        params = {
            "l_freq": 1.0,
            "h_freq": 40.0
        }
        
        # Create and run the step
        step = FilterStep(params)
        filtered_data = step.run(self.raw.copy())
        
        # Check that the object type is preserved
        self.assertIsInstance(filtered_data, mne.io.Raw)
        
        # Check that the data was actually modified (PSD should change)
        orig_psd = self.raw.compute_psd().get_data().mean(axis=0)
        filt_psd = filtered_data.compute_psd().get_data().mean(axis=0)
        
        # The PSD should be different after filtering
        self.assertFalse(np.allclose(orig_psd, filt_psd))
        
        # Low frequencies (below l_freq) should be attenuated
        freq_res = filtered_data.info['sfreq'] / filtered_data.compute_psd().n_fft
        low_freq_idx = int(params["l_freq"] / freq_res) - 1  # Index just below cutoff
        if low_freq_idx > 0:  # Only test if index is valid
            self.assertLess(filt_psd[low_freq_idx], orig_psd[low_freq_idx])
        
        # High frequencies (above h_freq) should be attenuated
        high_freq_idx = int(params["h_freq"] / freq_res) + 1  # Index just above cutoff
        if high_freq_idx < len(filt_psd):  # Only test if index is valid
            self.assertLess(filt_psd[high_freq_idx], orig_psd[high_freq_idx])
    
    def test_highpass_filter(self):
        """Test highpass filtering (low cutoff only)."""
        # Parameters for highpass filter
        params = {
            "l_freq": 1.0,
            "h_freq": None  # No high cutoff
        }
        
        # Create and run the step
        step = FilterStep(params)
        filtered_data = step.run(self.raw.copy())
        
        # Check that low frequencies are attenuated
        orig_psd = self.raw.compute_psd().get_data().mean(axis=0)
        filt_psd = filtered_data.compute_psd().get_data().mean(axis=0)
        
        freq_res = filtered_data.info['sfreq'] / filtered_data.compute_psd().n_fft
        low_freq_idx = int(params["l_freq"] / freq_res) - 1  # Index just below cutoff
        if low_freq_idx > 0:  # Only test if index is valid
            self.assertLess(filt_psd[low_freq_idx], orig_psd[low_freq_idx])
    
    def test_lowpass_filter(self):
        """Test lowpass filtering (high cutoff only)."""
        # Parameters for lowpass filter
        params = {
            "l_freq": None,  # No low cutoff
            "h_freq": 40.0
        }
        
        # Create and run the step
        step = FilterStep(params)
        filtered_data = step.run(self.raw.copy())
        
        # Check that high frequencies are attenuated
        orig_psd = self.raw.compute_psd().get_data().mean(axis=0)
        filt_psd = filtered_data.compute_psd().get_data().mean(axis=0)
        
        freq_res = filtered_data.info['sfreq'] / filtered_data.compute_psd().n_fft
        high_freq_idx = int(params["h_freq"] / freq_res) + 1  # Index just above cutoff
        if high_freq_idx < len(filt_psd):  # Only test if index is valid
            self.assertLess(filt_psd[high_freq_idx], orig_psd[high_freq_idx])
    
    def test_notch_filter(self):
        """Test notch filtering."""
        # Parameters with notch frequencies
        params = {
            "l_freq": 1.0,
            "h_freq": 40.0,
            "notch_freqs": [50]  # Target a specific frequency
        }
        
        # Create and run the step
        step = FilterStep(params)
        filtered_data = step.run(self.raw.copy())
        
        # Verify the notch filter by checking PSD
        orig_psd = self.raw.compute_psd().get_data().mean(axis=0)
        filt_psd = filtered_data.compute_psd().get_data().mean(axis=0)
        
        # The PSD at the notch frequency should be reduced
        freq_res = filtered_data.info['sfreq'] / filtered_data.compute_psd().n_fft
        notch_idx = int(params["notch_freqs"][0] / freq_res)
        
        # Get the index of the closest frequency to 50Hz
        freqs = filtered_data.compute_psd().freqs
        notch_idx = np.abs(freqs - params["notch_freqs"][0]).argmin()
        
        # The power at notch_idx should be lower in the filtered data
        # Allow for a small margin around the exact frequency
        window = slice(max(0, notch_idx - 1), min(len(filt_psd), notch_idx + 2))
        self.assertTrue(np.mean(filt_psd[window]) < np.mean(orig_psd[window]))
    
    def test_filter_with_epochs(self):
        """Test filtering with epoched data."""
        # Create epochs
        events = mne.make_fixed_length_events(self.raw, duration=1.0)
        epochs = mne.Epochs(
            self.raw, 
            events, 
            tmin=0, 
            tmax=1.0, 
            baseline=None, 
            preload=True
        )
        
        # Parameters for filtering
        params = {
            "l_freq": 1.0,
            "h_freq": 40.0
        }
        
        # Create and run the step
        step = FilterStep(params)
        filtered_epochs = step.run(epochs.copy())
        
        # Check that the object type is preserved
        self.assertIsInstance(filtered_epochs, mne.Epochs)
        
        # Check that the data was actually modified
        orig_psd = epochs.compute_psd().get_data().mean(axis=(0, 1))
        filt_psd = filtered_epochs.compute_psd().get_data().mean(axis=(0, 1))
        
        # The PSD should be different after filtering
        self.assertFalse(np.allclose(orig_psd, filt_psd))
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Parameters for filtering
        params = {
            "l_freq": 1.0,
            "h_freq": 40.0
        }
        
        # Create the step
        step = FilterStep(params)
        
        # Test with None as input
        with self.assertRaises(ValueError):
            step.run(None)

if __name__ == '__main__':
    unittest.main() 