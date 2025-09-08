# Neurotheque Resources

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/UNFmontreal/neurotheque_resources.svg)](https://github.com/UNFmontreal/neurotheque_resources/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/UNFmontreal/neurotheque_resources.svg)](https://github.com/UNFmontreal/neurotheque_resources/network)
[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)

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

- **scr/**
  - `pipeline.py` – The main pipeline driver and CLI (supports JSON or YAML configs with schema validation).
  - `steps/` – Modular processing steps (e.g., `LoadData`, `FilterStep`, `AutoRejectStep`, `ICAStep`, `SplitTasksStep`, `TriggerParsingStep`).
  - `strategies/` – Strategy examples for different paradigms (e.g., finger tapping, mental imagery).
  - `utils/` – Helper utilities for referencing, MNE cleaning, spectra, etc.

- **docs/** – Additional documentation, user guides, or methodology reports.

- **configs/** – JSON/YAML pipeline configs (e.g., `gonogo_minimal_pipeline.json` / `.yml`). JSON is the primary format; YAML remains supported.

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
    pip install -r requirements.txt           # runtime deps
    # optional: dev tools
    pip install -r requirements-dev.txt       # adds pytest, black, flake8, etc.
    ```
3. **Supported Python**:
   - We test on Python 3.10–3.12 (recommended with MNE 1.5+).
## Usage Examples

1. **Run a Pipeline (JSON or YAML)**
    ```bash
    # JSON (preferred)
    python -m scr.pipeline --config configs/gonogo_minimal_pipeline.json

    # YAML (backward compatible)
    python -m scr.pipeline --config configs/pilot_preprocessing_strategy.yml
    ```
    Notes:
    - The CLI auto-detects JSON vs YAML by extension and validates against `scr/config_schema.json`.
    - Use `--no-validate` to skip schema checks if iterating quickly.
    - Use `--dry-run` to print the step plan (and resolved files in multi-file mode) without executing.

2. **Use a Strategy Script**
    ```python
    # Example: Using the finger tapping strategy
    from scr.strategies.finger_tapping_strategy import run_finger_tapping_pipeline

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
  (examples directory forthcoming)

---
## Quick Start for Students

- Minimal config: `configs/gonogo_minimal_pipeline.json` (DSI‑24 → Go/NoGo).
- Run: `python -m scr.pipeline --config configs/gonogo_minimal_pipeline.json`.
- Outputs: processed FIF and checkpoints under `data/processed/sub-<id>/ses-<id>/`, and reports under `reports/` as defined by `ProjectPaths` and your config `directory`.

Notes:
- The runner passes `subject_id/session_id/task_id/run_id` and a `paths` helper to each step. In multi‑file mode these are parsed from filenames; in single‑subject mode they default to the values in your config (`default_subject`, `default_session`, `default_run`).
- AutoSave (enabled by `auto_save: true`) writes checkpoints after each step using the convention `after_<step>`. SaveCheckpoint does the same using the explicit `checkpoint_key`. The resume logic recognizes both patterns.

### Optional Extras

Some examples (e.g., spectral modeling/FOOOF, mixed-effects) require additional packages that are not needed for core preprocessing:

- fooof (spectral parameterization)
- statsmodels (mixed-effects models)

Install on demand, for example: `pip install fooof statsmodels`.

## Testing

The Neurotheque pipeline includes a comprehensive test suite to ensure reliability and robustness. The tests are organized into unit tests (for individual components) and integration tests (for workflows and complete pipelines).

### Installing Development Dependencies

To run tests, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Running Tests

Use pytest directly or the bundled test runner:

```bash
# Run all tests (pytest)
pytest -q

# Or use the project test runner
python tests/run_tests.py               # all
python tests/run_tests.py --unit        # unit only
python tests/run_tests.py --integration # integration only
```

### Test Documentation

For detailed information about the test suite, including how to add new tests, see the [tests/README.md](tests/README.md) file.

---

