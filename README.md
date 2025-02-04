# Neurotheque Resources

**Neurotheque Resources** is a comprehensive repository for EEG data processing and analysis. It provides a modular and configurable pipeline for:
- Data loading and channel preparation
- Filtering and artifact rejection (including automated rejection and ICA)
- Task segmentation and trigger parsing
- Epoching and event merging (e.g. for Go/No-Go experiments)
- Advanced analyses such as time–frequency transformation and ERP/ROI analysis
- Generating HTML reports with MNE’s reporting tools

The code is built primarily on [MNE-Python](https://mne.tools/stable/index.html) and leverages several other libraries (e.g. [autoreject](https://autoreject.github.io/) to facilitate reproducible EEG research.

## Repository Structure

The repository is organized to facilitate reproducible research, modular pipeline execution, and collaborative development. The recommended folder structure is as follows:

- **notebooks/**  
  Contains Jupyter notebooks for:
  - **tutorials/** – Step-by-step examples to help users get started.
  - **preprocessing/** – Notebooks that demonstrate data cleaning, channel preparation, filtering, artifact rejection, and ICA.
  - **analysis/** – Notebooks that illustrate how to perform epoching, event merging, ROI analysis, time–frequency analysis, and report generation.

- **src/**  
  Contains all the Python source code for the EEG processing pipelines. Major components include:
  - **pipeline.py** – Main pipeline executor that loads a configuration (via YAML or a Python dictionary) and runs a series of processing steps.
  - **steps/** – A collection of modular pipeline steps (e.g., `LoadData`, `FilterStep`, `AutoRejectStep`, `ICAStep`, `ReferenceStep`, `SplitTasksStep`, `TriggerParsingStep`, etc.) that implement individual processing tasks.
  - **strategies/** – Example strategies for specific paradigms (e.g., finger tapping, mental imagery) that configure and run the pipeline.
  - **utilities** – Utility scripts such as `combine_py_files.py` (to merge source files) and `export_repo_structure.py` (to generate a repository structure overview).

- **doc/**  
  Contains documentation files (manuals, technical reports, or usage guides) to help users understand how to configure and extend the pipeline.

- **configs/**  
  (If present) Contains YAML configuration files that define various pipeline setups (e.g., global preprocessing, Go/No-Go analysis, multi-subject strategies).

## Getting Started

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/UNFmontreal/neurotheque_resources.git
   cd neurotheque_resources
