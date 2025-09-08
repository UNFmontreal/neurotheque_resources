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
from scr.steps.epoching import EpochingStep

class TestEpochingStep(unittest.TestCase):
    """Unit tests for the EpochingStep class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Create temporary directory for test output
        cls.temp_dir = tempfile.mkdtemp()
        
        # Load sample data from MNE
        data_path = sample.data_path()
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        cls.raw = mne.io.read_raw_fif(raw_fname, preload=True)
        
        # Extract events from the sample data
        cls.events = mne.find_events(cls.raw, stim_channel='STI 014')
        
        # Create a simpler raw with known events for more controlled testing
        cls.simple_raw = mne.io.RawArray(
            np.random.randn(10, 10000),  # 10 channels, 10000 samples
            mne.create_info(10, 1000.0, ch_types='eeg')  # 1000 Hz sampling rate
        )
        
        # Create known events
        cls.simple_events = np.array([
            [1000, 0, 1],  # Event 1 at 1s
            [2000, 0, 2],  # Event 2 at 2s
            [3000, 0, 3],  # Event 3 at 3s
            [4000, 0, 1],  # Event 1 at 4s
            [5000, 0, 2],  # Event 2 at 5s
            [6000, 0, 3],  # Event 3 at 6s
        ])
        
        # Add a stim channel to the simple raw
        stim_data = np.zeros((1, cls.simple_raw.n_times))
        for event in cls.simple_events:
            stim_data[0, event[0]] = event[2]
        
        info = mne.create_info(['STI'], cls.simple_raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        cls.simple_raw.add_channels([stim_raw], force_update_info=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test initialization with various parameters."""
        # Test with minimal parameters
        params = {
            "tmin": -0.2,
            "tmax": 0.5,
            "event_id": {"auditory": 1, "visual": 3}
        }
        
        step = EpochingStep(params)
        self.assertEqual(step.params, params)
        
        # Test with comprehensive parameters
        params = {
            "tmin": -0.2,
            "tmax": 0.5,
            "event_id": {"auditory": 1, "visual": 3},
            "baseline": (None, 0),
            "reject": {"eeg": 100e-6},
            "flat": {"eeg": 1e-6},
            "reject_by_annotation": True,
            "decim": 2,
            "preload": True,
            "detrend": 1,
            "picks": "eeg"
        }
        
        step = EpochingStep(params)
        self.assertEqual(step.params, params)
    
    def test_basic_epoching(self):
        """Test basic epoching with default parameters."""
        # Set up parameters for the step
        params = {
            "tmin": -0.1,
            "tmax": 0.3,
            "event_id": {"1": 1, "2": 2, "3": 3},
            "baseline": (None, 0),
            "preload": True
        }
        
        # Create and run the step
        step = EpochingStep(params)
        result = step.run(self.simple_raw)
        
        # Check that the output is an Epochs object
        self.assertIsInstance(result, mne.Epochs)
        
        # Check that the epochs have the expected properties
        self.assertEqual(result.tmin, params["tmin"])
        self.assertEqual(result.tmax, params["tmax"])
        self.assertEqual(result.event_id, params["event_id"])
        
        # Check that we have the expected number of events
        self.assertEqual(len(result), len(self.simple_events))
        
        # Check that the data dimensions are correct
        self.assertEqual(result.get_data().shape[0], len(self.simple_events))  # Epochs
        self.assertEqual(result.get_data().shape[1], len(self.simple_raw.ch_names) - 1)  # Channels (excluding stim)
        expected_samples = int((params["tmax"] - params["tmin"]) * self.simple_raw.info["sfreq"]) + 1
        self.assertEqual(result.get_data().shape[2], expected_samples)  # Samples
    
    def test_event_selection(self):
        """Test selecting specific events."""
        # Set up parameters to select only event type 1
        params = {
            "tmin": -0.1,
            "tmax": 0.3,
            "event_id": {"target": 1},  # Only select event 1
            "baseline": (None, 0),
            "preload": True
        }
        
        # Create and run the step
        step = EpochingStep(params)
        result = step.run(self.simple_raw)
        
        # Check that we have the expected number of events (only type 1)
        event_1_count = np.sum(self.simple_events[:, 2] == 1)
        self.assertEqual(len(result), event_1_count)
        
        # Check that all epochs have the correct event type
        self.assertTrue(all(e == 1 for e in result.events[:, 2]))
    
    def test_baseline_correction(self):
        """Test baseline correction."""
        # Set up parameters with baseline correction
        params = {
            "tmin": -0.1,
            "tmax": 0.3,
            "event_id": {"1": 1},
            "baseline": (-0.1, 0),  # Apply baseline correction
            "preload": True
        }
        
        # Create and run the step
        step = EpochingStep(params)
        result = step.run(self.simple_raw)
        
        # Check that baseline correction was applied
        # In baseline period, mean should be close to 0 after correction
        baseline_data = result.get_data()[:, :, :result._raw_times.searchsorted(0)]
        self.assertTrue(np.allclose(baseline_data.mean(), 0, atol=1e-10))
    
    def test_rejection_parameters(self):
        """Test epoching with rejection parameters."""
        # Create a raw with large artifacts
        artifact_raw = self.simple_raw.copy()
        
        # Add a large artifact at 3.5s (should affect epoch with event at 3s)
        artifact_time_idx = int(3.5 * artifact_raw.info["sfreq"])
        artifact_raw._data[0, artifact_time_idx:artifact_time_idx+100] = 1000e-6  # Large artifact
        
        # Set up parameters with rejection
        params = {
            "tmin": -0.1,
            "tmax": 0.5,
            "event_id": {"1": 1, "2": 2, "3": 3},
            "baseline": (None, 0),
            "reject": {"eeg": 500e-6},  # Reject if EEG > 500µV
            "preload": True
        }
        
        # Create and run the step
        step = EpochingStep(params)
        result = step.run(artifact_raw)
        
        # Check that one epoch was rejected (the one with the artifact)
        self.assertLess(len(result), len(self.simple_events))
        
        # Find which event time was closest to the artifact
        artifact_event_time = 3.0  # As set above
        artifact_event_idx = np.where(np.isclose(self.simple_events[:, 0] / artifact_raw.info["sfreq"], 
                                                artifact_event_time))[0][0]
        
        # The corresponding event type should not be in the final events
        self.assertFalse(any(np.isclose(result.events[:, 0] / result.info["sfreq"], 
                                      self.simple_events[artifact_event_idx, 0] / artifact_raw.info["sfreq"])))
    
    def test_reject_by_annotation(self):
        """Test rejecting epochs based on annotations."""
        # Create a raw with annotations
        annotated_raw = self.simple_raw.copy()
        
        # Add a BAD annotation covering the event at 2s
        bad_annot = mne.Annotations(
            onset=[1.9],  # Start just before the event
            duration=[0.3],  # Cover the event
            description=['BAD']
        )
        annotated_raw.set_annotations(bad_annot)
        
        # Set up parameters with reject_by_annotation=True
        params = {
            "tmin": -0.1,
            "tmax": 0.3,
            "event_id": {"1": 1, "2": 2, "3": 3},
            "baseline": (None, 0),
            "reject_by_annotation": True,  # Reject epochs that overlap with annotations
            "preload": True
        }
        
        # Create and run the step
        step = EpochingStep(params)
        result = step.run(annotated_raw)
        
        # Check that we have one less epoch than expected
        self.assertEqual(len(result), len(self.simple_events) - 1)
        
        # Find which event time was covered by the annotation
        annotated_event_time = 2.0  # As set above
        annotated_event_idx = np.where(np.isclose(self.simple_events[:, 0] / annotated_raw.info["sfreq"], 
                                                annotated_event_time))[0][0]
        
        # The corresponding event type should not be in the final events
        self.assertFalse(any(np.isclose(result.events[:, 0] / result.info["sfreq"], 
                                      self.simple_events[annotated_event_idx, 0] / annotated_raw.info["sfreq"])))
    
    def test_error_handling(self):
        """Test error handling with various invalid inputs."""
        # Valid parameters
        params = {
            "tmin": -0.1,
            "tmax": 0.3,
            "event_id": {"1": 1},
            "baseline": (None, 0),
            "preload": True
        }
        
        # Create the step
        step = EpochingStep(params)
        
        # Test with None as input
        with self.assertRaises(ValueError):
            step.run(None)
        
        # Test with invalid event_id
        invalid_params = params.copy()
        invalid_params["event_id"] = {"nonexistent": 999}  # Event 999 doesn't exist
        
        invalid_step = EpochingStep(invalid_params)
        result = invalid_step.run(self.simple_raw)
        
        # The result should be an Epochs object with 0 epochs
        self.assertIsInstance(result, mne.Epochs)
        self.assertEqual(len(result), 0)
        
        # Test with illogical time parameters
        illogical_params = params.copy()
        illogical_params["tmin"] = 0.5
        illogical_params["tmax"] = 0.2  # tmax < tmin
        
        illogical_step = EpochingStep(illogical_params)
        with self.assertRaises(ValueError):
            illogical_step.run(self.simple_raw)

if __name__ == '__main__':
    unittest.main() 