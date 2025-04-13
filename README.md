# Neurotheque Resources

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/UNFmontreal/neurotheque_resources.svg)](https://github.com/UNFmontreal/neurotheque_resources/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/UNFmontreal/neurotheque_resources.svg)](https://github.com/UNFmontreal/neurotheque_resources/network)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

<br />

<div align="center">
  <img src="docs/images/pipeline_animation_v2.gif" alt="Neurotheque Pipeline Steps" width="600" />
  <p><em> </em></p>
</div>

**Neurotheque Resources** is a comprehensive, modular toolkit for EEG data processing and analysis. Built on state‐of‐the‐art libraries such as [MNE-Python](https://mne.tools/stable/index.html), [autoreject](https://autoreject.github.io/) it enables:

- **Preprocessing & Channel Setup**: Automatic bad channel detection and re-referencing with standardized electrode montages.
- **Artifact Rejection**: Advanced methods like ICA and auto-rejection for high-quality EEG data.
- **Task & Trigger Parsing**: Flexible logic to merge triggers for tasks like Go/No-Go, finger tapping, or mental imagery.
- **Epoching & Analysis**: Efficient epoch slicing, ROI averaging, time–frequency transforms, and ERP analyses.
- **Report Generation**: Automated HTML reporting (via MNE's `Report`) for visual inspection and documentation of analysis results.

This repository is designed for both single-subject and multi-subject pipelines, ensuring reproducibility and scalability.


---
## Table of Contents

1. [Repository Structure](#repository-structure)  
2. [Installation & Quick Start](#installation--quick-start)  
3. [Usage Examples](#usage-examples)  
4. [Direct Step Usage](#direct-step-usage)  
5. [Testing](#testing)  
6. [Contributors & Acknowledgments](#contributors--acknowledgments)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Contact](#contact)

## Repository Structure

- **notebooks/**
  - **tutorials/** – Step-by-step Jupyter notebooks illustrating basic usage and pipeline demos.  
  - **preprocessing/** – Notebooks focusing on data cleaning (filtering, ICA, etc.).  
  - **analysis/** – Notebooks showing advanced analysis tasks (ERP, ROI-based metrics, time–frequency transforms).

- **src/**
  - **pipeline.py** – The main pipeline driver, reading YAML configs or Python dicts to run each pipeline step in sequence.
  - **steps/** – A suite of modular processing steps (e.g., `LoadData`, `FilterStep`, `AutoRejectStep`, `ICAStep`, `SplitTasksStep`, `TriggerParsingStep`) that can be combined in various orders.
  - **strategies/** – Specialized scripts/pipelines for different paradigms (e.g., finger tapping, mental imagery).
  - **utilities/** – Helper scripts like `combine_py_files.py` (for merging .py files) or `export_repo_structure.py` (for generating directory trees).

- **doc/** – Additional documentation, user guides, or methodology reports.

- **configs/** – YAML files detailing pipeline configurations (e.g., `pilot_preprocessing_strategy.yml`, `gonogo_analysis.yml`, etc.).

- **tests/** – Comprehensive test suite for all components of the pipeline.
  - **unit/** – Tests for individual steps and utility functions.
  - **integration/** – Tests for complete workflows and pipeline configurations.

---

## Getting Started

## Installation & Quick Start

1. **Clone the repository**:
    ```bash
    git clone https://github.com/UNFmontreal/neurotheque_resources.git
    cd neurotheque_resources
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
## Usage Examples

1. **Run a Pipeline with a YAML Configuration**
    ```bash
    python src/pipeline.py --config configs/pilot_preprocessing_strategy.yml
    ```
    This loads the steps specified in the config (e.g. `LoadData`, `FilterStep`, `AutoRejectStep`, etc.) and applies them to your EEG data.

2. **Use a Strategy Script**
    ```python
    # Example: Using the finger tapping strategy
    from src.strategies.finger_tapping_strategy import run_finger_tapping_pipeline

    input_path = "data/raw/sub-01_ses-001_eeg.edf"
    output_path = "data/processed/sub-01_ses-001_preprocessed.fif"

    run_finger_tapping_pipeline(input_path, output_path)
    ```
3. **Explore the Jupyter Notebooks**

## Direct Step Usage

Each processing step in the Neurotheque pipeline can be used independently on MNE data objects, allowing for flexible integration into your custom workflows.

### Basic Pattern

```python
from scr.steps.filter import FilterStep

# Configure parameters
filter_params = {
    "l_freq": 1.0,        # High-pass frequency (Hz)
    "h_freq": 40.0,       # Low-pass frequency (Hz)
    "notch_freqs": [50]   # Optional notch filter frequencies (Hz)
}

# Initialize the step
filter_step = FilterStep(filter_params)

# Run it on your MNE data
filtered_data = filter_step.run(your_mne_raw_data)
```

### Complete Example

Here's a simplified workflow example directly using the processing steps:

```python
import mne
from scr.steps.load import LoadData
from scr.steps.filter import FilterStep
from scr.steps.autoreject import AutoRejectStep
from scr.steps.epoching import EpochingStep

# Load data
load_step = LoadData({"input_file": "your_data.fif"})
raw_data = load_step.run(None)

# Apply filters
filter_step = FilterStep({"l_freq": 1.0, "h_freq": 40.0})
filtered_data = filter_step.run(raw_data)

# Apply AutoReject for artifact detection
ar_step = AutoRejectStep({"mode": "fit", "plot_results": True})
ar_data = ar_step.run(filtered_data)

# Create epochs
events = mne.find_events(ar_data)
epoch_step = EpochingStep({
    "tmin": -0.2, "tmax": 0.5, 
    "baseline": (None, 0),
    "event_id": {"1": 1}
})
epochs = epoch_step.run(ar_data)

print(f"Created {len(epochs)} clean epochs")
```

For detailed examples of direct step usage, see:
- [Direct Step Usage Guide](docs/direct_step_usage.md)
- [Example Script](example_direct_step_usage.py)

## Testing

The Neurotheque pipeline includes a comprehensive test suite to ensure reliability and robustness. The tests are organized into unit tests (for individual components) and integration tests (for workflows and complete pipelines).

### Installing Development Dependencies

To run tests, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Running Tests

The test runner script provides several options:

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
```

### Test Documentation

For detailed information about the test suite, including how to add new tests, see the [tests/README.md](tests/README.md) file.

---

