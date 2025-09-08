# Direct Usage of Pipeline Steps

The Neurotheque pipeline is designed with flexibility in mind. While the full pipeline orchestrates multiple steps for complete EEG processing, each step can also be used independently on MNE data objects.

This guide shows how to directly call individual steps on your loaded MNE data, making it easy to integrate specific processing steps into your custom workflows.

## Table of Contents

1. [Overview](#overview)
2. [Common Pattern](#common-pattern)
3. [Examples for Each Step](#examples-for-each-step)
   - [LoadData](#loaddata)
   - [PrepChannelsStep](#prepchannelsstep)
   - [FilterStep](#filterstep)
   - [AutoRejectStep](#autorejectstep)
   - [ICAStep](#icastep)
   - [EpochingStep](#epochingstep)
   - [SaveData](#savedata)
4. [Full Workflow Example](#full-workflow-example)

## Overview

Each step in the Neurotheque pipeline follows a consistent pattern:

1. Initialize the step with parameters
2. Call the `run()` method with your MNE data
3. Get back the processed data

This allows you to use any step directly with your MNE data objects, without needing to run the full pipeline.

## Quick Start (JSON)

Prefer JSON configs for full pipeline runs. Minimal example and command:

```json
{
  "auto_save": true,
  "default_subject": "01",
  "default_session": "001",
  "default_run": "01",
  "directory": {
    "root": ".",
    "raw_data_dir": "data/pilot_data/tasks",
    "processed_dir": "data/processed",
    "reports_dir": "reports",
    "derivatives_dir": "derivatives"
  },
  "file_path_pattern": "sub-01_ses-001_task-gng_image_run-01_raw.fif",
  "pipeline": {
    "steps": [
      {"name": "LoadData"},
      {"name": "FilterStep", "params": {"l_freq": 1.0, "h_freq": 40.0, "notch_freqs": [50, 60]}},
      {"name": "EpochingStep", "params": {"task_type": "gng", "trigger_ids": {"go": 1, "nogo": 2, "response": 3}, "returns_epochs": true}},
      {"name": "SaveCheckpoint", "params": {"checkpoint_key": "after_epoching"}}
    ]
  }
}
```

Run it:

```bash
python -m scr.pipeline --config configs/gonogo_minimal_pipeline.json
```

### Trigger Mapping Cheat‑Sheet

- task_type "5pt": use a single code for all events
  - trigger_ids: { "trigger_id": 8 }
- task_type "gng": use separate codes
  - trigger_ids: { "go": 1, "nogo": 2, "response": 3 }

## Common Pattern

The general pattern for using any step is:

```python
from scr.steps.your_step import YourStep

# Configure parameters for the step
params = {
    "param1": value1,
    "param2": value2,
    # ...
}

# Initialize the step
step = YourStep(params)

# Run the step on your MNE data
result = step.run(your_mne_data)
```

## Examples for Each Step

### LoadData

Load data from a file:

```python
from scr.steps.load import LoadData

# Configure parameters
load_params = {
    "input_file": "your_raw_file.fif",  # Path to your data file
}

# Initialize and run the step
load_step = LoadData(load_params)
raw_data = load_step.run(None)  # Input is None for LoadData
```

### PrepChannelsStep

Standardize channels and optionally re-reference (recommended place to set reference):

```python
from scr.steps.prepchannels import PrepChannelsStep

prep = PrepChannelsStep({
    "on_missing": "ignore",
    "reference": {"method": "average", "projection": False}
})
prepped = prep.run(raw_data)
```

### FilterStep

Apply frequency filters to your data:

```python
from scr.steps.filter import FilterStep

# Configure parameters
filter_params = {
    "l_freq": 1.0,        # High-pass filter (Hz)
    "h_freq": 40.0,       # Low-pass filter (Hz)
    "notch_freqs": [50]   # Optional notch filters (Hz)
}

# Initialize and run the step
filter_step = FilterStep(filter_params)
filtered_data = filter_step.run(raw_data)
```

### AutoRejectStep

Detect and handle artifacts using AutoReject:

```python
from scr.steps.autoreject import AutoRejectStep

# Configure parameters
ar_params = {
    "ar_params": {"n_interpolate": [1, 4], "consensus": [0.2, 0.5]},
    "mode": "fit",  # 'fit' marks bad epochs, 'fit_transform' cleans data
    "plot_results": True,
    "store_reject_log": True,
    "output_dir": "./output/autoreject"  # Optional - where to save outputs
}

# Create the output directory if needed
import os
os.makedirs(ar_params["output_dir"], exist_ok=True)

# Initialize and run the step
ar_step = AutoRejectStep(ar_params)
ar_data = ar_step.run(filtered_data)
```

### ICAStep

Apply ICA for artifact removal. You can use either the combined `ICAStep` or the separate extraction and labeling steps:

#### Option 1: Using separate extraction and labeling steps

```python
from scr.steps.ica_extraction import ICAExtractionStep
from scr.steps.ica_labeling import ICALabelingStep
from pathlib import Path

# MockPaths helper (only needed for steps that save outputs)
class MockPaths:
    def __init__(self, base_dir="./output"):
        self.base_dir = Path(base_dir)
    
    def get_derivative_path(self, subject_id, session_id):
        path = self.base_dir / f"sub-{subject_id}" / f"ses-{session_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_ica_report_dir(self, subject_id, session_id):
        path = self.base_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "ica_reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

# Create paths object
paths = MockPaths("./output")

# 1. ICA Extraction
extract_params = {
    "n_components": 10,
    "method": "fastica",
    "max_iter": 500,
    "subject_id": "01",  # Optional - for output organization
    "session_id": "001", # Optional - for output organization
    "paths": paths       # Only needed if saving outputs
}

extract_step = ICAExtractionStep(extract_params)
extract_result = extract_step.run(ar_data)

# 2. ICA Labeling
label_params = {
    "eog_ch_names": ["EEG001", "EEG002"],  # Frontal channels
    "eog_threshold": 0.3,
    "interactive": False,  # True if you want manual selection
    "plot_components": True,
    "subject_id": "01",
    "session_id": "001",
    "paths": paths
}

label_step = ICALabelingStep(label_params)
labeled_result = label_step.run(extract_result)
```

#### Option 2: Using the combined ICAStep

```python
from scr.steps.ica import ICAStep

# Configure parameters
ica_params = {
    "n_components": 10,
    "method": "fastica",
    "max_iter": 500,
    "fit_params": {"extended": True},
    "eog_ch_names": ["EEG001", "EEG002"],
    "interactive": False,
    "plot_components": True,
    "subject_id": "01",
    "session_id": "001",
    "paths": paths  # From previous example
}

# Initialize and run the step
ica_step = ICAStep(ica_params)
ica_data = ica_step.run(ar_data)
```

### EpochingStep

Create epochs from continuous data:

```python
from scr.steps.epoching import EpochingStep
import mne

# First, detect events in your data
events = mne.find_events(raw_data, stim_channel='STI')

# Configure parameters
epoch_params = {
    "tmin": -0.2,          # Start time (seconds relative to event)
    "tmax": 0.5,           # End time (seconds relative to event)
    "baseline": (None, 0), # Baseline correction period
    "event_id": {"1": 1},  # Event types to include
    "reject": {"eeg": 100e-6},  # Optional amplitude-based rejection
    "reject_by_annotation": True  # Use annotations to reject epochs
}

# Initialize and run the step
epoch_step = EpochingStep(epoch_params)
epochs = epoch_step.run(raw_data)
```

### SaveData

Save your processed data to a file:

```python
from scr.steps.save import SaveData

# Configure parameters
save_params = {"output_path": "./output/processed_data.fif", "overwrite": True}

# Initialize and run the step
save_step = SaveData(save_params)
save_step.run(epochs)  # Works with either Raw or Epochs
```

## Full Workflow Example

Here's a complete example showing how to chain multiple steps together:

```python
import mne
import numpy as np
import os
from pathlib import Path

# Import steps
from scr.steps.load import LoadData
from scr.steps.filter import FilterStep
from scr.steps.autoreject import AutoRejectStep
from scr.steps.ica import ICAStep
from scr.steps.epoching import EpochingStep
from scr.steps.save import SaveData

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Mock paths for steps that need it
class MockPaths:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_derivative_path(self, subject_id, session_id):
        path = self.base_dir / f"sub-{subject_id}" / f"ses-{session_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_ica_report_dir(self, subject_id, session_id):
        path = self.base_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "ica_reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

paths = MockPaths(output_dir)

# Step 1: Load Data
load_params = {"input_file": "sample_data.fif"}
load_step = LoadData(load_params)
raw_data = load_step.run(None)

# Step 2: Filter
filter_params = {"l_freq": 1.0, "h_freq": 40.0, "notch_freqs": [50]}
filter_step = FilterStep(filter_params)
filtered_data = filter_step.run(raw_data)

# Step 3: AutoReject
ar_params = {
    "ar_params": {"n_interpolate": [1, 4], "consensus": [0.2, 0.5]},
    "mode": "fit",
    "plot_results": True,
    "subject_id": "01",
    "session_id": "001"
}
ar_step = AutoRejectStep(ar_params)
ar_data = ar_step.run(filtered_data)

# Step 4: ICA
ica_params = {
    "n_components": 10,
    "method": "fastica",
    "eog_ch_names": ["EEG001", "EEG002"],
    "interactive": False,
    "plot_components": True,
    "subject_id": "01",
    "session_id": "001",
    "paths": paths
}
ica_step = ICAStep(ica_params)
ica_data = ica_step.run(ar_data)

# Step 5: Epoching
events = mne.find_events(ica_data)
epoch_params = {
    "tmin": -0.2, "tmax": 0.5,
    "baseline": (None, 0),
    "event_id": {str(events[0, 2]): events[0, 2]},
    "reject_by_annotation": True
}
epoch_step = EpochingStep(epoch_params)
epochs = epoch_step.run(ica_data)

# Step 6: Save
save_params = {"output_file": os.path.join(output_dir, "processed_data.fif")}
save_step = SaveData(save_params)
save_step.run(epochs)

print("Processing complete!")
```

For a complete working example, see the `example_direct_step_usage.py` script in the project root. 
