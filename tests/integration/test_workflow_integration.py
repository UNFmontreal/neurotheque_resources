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

# Import components to test
from scr.pipeline import Pipeline
from scr.steps.load import LoadData
from scr.steps.filter import FilterStep
from scr.steps.autoreject import AutoRejectStep
from scr.steps.ica import ICAStep
from scr.steps.epoching import EpochingStep

class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for various workflows combining multiple processing steps."""
    
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
        
        # Create subject-specific directories
        cls.sub01_dir = os.path.join(cls.processed_dir, "sub-01", "ses-001")
        os.makedirs(cls.sub01_dir, exist_ok=True)
        
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
        
        # Save to the raw directory for testing
        cls.raw_file = os.path.join(cls.raw_dir, "sub-01_ses-001_task-test_run-01_raw.fif")
        cls.raw.save(cls.raw_file, overwrite=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up a test configuration."""
        # Create a path for a config file we'll use in tests
        self.config_file = os.path.join(self.temp_dir, "test_workflow.yaml")
    
    def test_load_filter_autoreject_workflow(self):
        """Test a workflow with LoadData -> FilterStep -> AutoRejectStep."""
        # Create a configuration for this workflow
        config = {
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
            "auto_save": False,
            "file_path_pattern": os.path.join(self.raw_dir, "sub-01_ses-001_*_raw.fif"),
            "pipeline": {
                "steps": [
                    {
                        "name": "LoadData",
                        "params": {
                            "input_file": self.raw_file
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
                    }
                ]
            }
        }
        
        # Write config to a file
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run the pipeline
        pipeline = Pipeline(config_file=self.config_file)
        pipeline.run()
        
        # Check that the pipeline ran successfully
        # In "fit" mode, we should have annotations for bad epochs
        self.assertIsNotNone(pipeline.data)
        self.assertIsInstance(pipeline.data, mne.io.Raw)
        self.assertGreater(len(pipeline.data.annotations), 0)
    
    def test_full_workflow_with_steps(self):
        """Test a full workflow manually by chaining steps together (without using Pipeline)."""
        # This tests the steps working together directly, without the Pipeline class
        
        # 1. Load data
        load_params = {"input_file": self.raw_file}
        load_step = LoadData(load_params)
        raw_data = load_step.run(None)  # Initial data is None
        
        # 2. Filter
        filter_params = {"l_freq": 1.0, "h_freq": 40.0}
        filter_step = FilterStep(filter_params)
        filtered_data = filter_step.run(raw_data)
        
        # 3. Run AutoReject
        ar_params = {
            "ar_params": {"n_interpolate": [1, 4], "consensus": [0.2, 0.5]},
            "mode": "fit",
            "plot_results": False
        }
        ar_step = AutoRejectStep(ar_params)
        ar_data = ar_step.run(filtered_data)
        
        # 4. Run ICA
        ica_params = {
            "n_components": 10,
            "method": "fastica",
            "max_iter": 200,
            "fit_params": {"extended": False},
            "subject_id": "01",
            "session_id": "001",
            "eog_ch_names": ["Fp1", "Fp2"],
            "interactive": False,
            "plot_components": False,
            "plot_sources": False,
            "output_dir": os.path.join(self.temp_dir, "ica_output"),
            "exclude": []  # No pre-exclusions
        }
        os.makedirs(ica_params["output_dir"], exist_ok=True)
        ica_step = ICAStep(ica_params)
        ica_data = ica_step.run(ar_data)
        
        # 5. Create epochs
        events = mne.find_events(ica_data, stim_channel='STI 014')
        if len(events) == 0:  # If no events found, create dummy events
            events = mne.make_fixed_length_events(ica_data, id=1, duration=1.0)
        
        epoch_params = {
            "tmin": -0.2,
            "tmax": 0.5,
            "event_id": {str(events[0, 2]): events[0, 2]},
            "baseline": (None, 0),
            "preload": True
        }
        epoch_step = EpochingStep(epoch_params)
        epochs = epoch_step.run(ica_data)
        
        # Verify each step of the process
        self.assertIsInstance(raw_data, mne.io.Raw)
        self.assertIsInstance(filtered_data, mne.io.Raw)
        self.assertIsInstance(ar_data, mne.io.Raw)
        self.assertIsInstance(ica_data, mne.io.Raw)
        self.assertIsInstance(epochs, mne.Epochs)
        
        # Check that data was actually processed
        self.assertGreater(len(ar_data.annotations), len(raw_data.annotations))
        self.assertIn("temp", ica_data.info)
        self.assertIn("ica", ica_data.info["temp"])
    
    def test_pipeline_workflow_integration(self):
        """Test a complete workflow using the Pipeline class."""
        # Create a configuration for a full workflow
        config = {
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports",
                "derivatives_dir": "derivatives"
            },
            "pipeline_mode": "restart",
            "default_subject": "01",
            "default_session": "001",
            "default_run": "01",
            "auto_save": False,
            "file_path_pattern": os.path.join(self.raw_dir, "sub-01_ses-001_*_raw.fif"),
            "pipeline": {
                "steps": [
                    {
                        "name": "LoadData",
                        "params": {
                            "input_file": self.raw_file
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
                            "plot_results": False,
                            "store_reject_log": True,
                            "save_model": False
                        }
                    },
                    {
                        "name": "ICAStep",
                        "params": {
                            "n_components": 10,
                            "method": "fastica",
                            "max_iter": 200,
                            "fit_params": {"extended": False},
                            "eog_ch_names": ["Fp1", "Fp2"],
                            "interactive": False,
                            "plot_components": False,
                            "plot_sources": False,
                            "exclude": []
                        }
                    },
                    {
                        "name": "EpochingStep",
                        "params": {
                            "tmin": -0.2,
                            "tmax": 0.5,
                            "baseline": (None, 0),
                            "preload": True,
                            "event_id": None  # Will be determined automatically
                        }
                    }
                ]
            }
        }
        
        # Write config to a file
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run the pipeline
        pipeline = Pipeline(config_file=self.config_file)
        try:
            pipeline.run()
            
            # Check the final result
            self.assertIsNotNone(pipeline.data)
            
            # The final result should be Epochs
            self.assertIsInstance(pipeline.data, mne.Epochs)
            
            # Check for temp/ica metadata (from ICA step)
            self.assertIn("temp", pipeline.data.info)
            self.assertIn("ica", pipeline.data.info["temp"])
            
        except Exception as e:
            self.fail(f"Pipeline execution failed with error: {e}")
    
    def test_checkpoint_resumption(self):
        """Test the ability to resume processing from a checkpoint."""
        # First, create and run a pipeline that saves a checkpoint
        checkpoint_config = {
            "directory": {
                "root": self.temp_dir,
                "raw_data_dir": "raw",
                "processed_dir": "processed",
                "reports_dir": "reports",
                "derivatives_dir": "derivatives"
            },
            "pipeline_mode": "restart",
            "default_subject": "01",
            "default_session": "001",
            "default_run": "01",
            "auto_save": True,  # Enable auto-save
            "file_path_pattern": os.path.join(self.raw_dir, "sub-01_ses-001_*_raw.fif"),
            "pipeline": {
                "steps": [
                    {
                        "name": "LoadData",
                        "params": {
                            "input_file": self.raw_file
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
                        "name": "SaveCheckpoint",
                        "params": {
                            "checkpoint_key": "after_filter"
                        }
                    },
                    {
                        "name": "AutoRejectStep",
                        "params": {
                            "ar_params": {"n_interpolate": [1, 4], "consensus": [0.2, 0.5]},
                            "mode": "fit",
                            "plot_results": False
                        }
                    }
                ]
            }
        }
        
        # Write config to a file
        checkpoint_file = os.path.join(self.temp_dir, "checkpoint_test.yaml")
        with open(checkpoint_file, 'w') as f:
            yaml.dump(checkpoint_config, f)
        
        # Run the pipeline to create the checkpoint
        pipeline1 = Pipeline(config_file=checkpoint_file)
        pipeline1.run()
        
        # Check that the checkpoint file was created
        checkpoint_path = os.path.join(self.sub01_dir, "sub-01_ses-001_after_filter.fif")
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Now modify the config to start from the AutoRejectStep
        resume_config = checkpoint_config.copy()
        resume_config["pipeline_mode"] = "resume"
        resume_config["start_from_step"] = "AutoRejectStep"
        
        resume_file = os.path.join(self.temp_dir, "resume_test.yaml")
        with open(resume_file, 'w') as f:
            yaml.dump(resume_config, f)
        
        # Run the pipeline again
        pipeline2 = Pipeline(config_file=resume_file)
        pipeline2.run()
        
        # Check the final result
        self.assertIsNotNone(pipeline2.data)
        self.assertIsInstance(pipeline2.data, mne.io.Raw)
        
        # Should have AutoReject annotations
        self.assertGreater(len(pipeline2.data.annotations), 0)

if __name__ == '__main__':
    unittest.main() 
