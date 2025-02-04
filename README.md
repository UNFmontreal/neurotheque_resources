# Neurotheque Resources

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/UNFmontreal/neurotheque_resources.svg)](https://github.com/UNFmontreal/neurotheque_resources/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/UNFmontreal/neurotheque_resources.svg)](https://github.com/UNFmontreal/neurotheque_resources/network)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

<br />

<div align="center">
  <img src="https://media.giphy.com/media/TrHjlqg31VfuhTtVKP/giphy.gif" alt="Neurotheque Animated Demo" width="600" />
  <p><em>Example animation </em></p>
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
4. [Contributors & Acknowledgments](#contributors--acknowledgments)  
5. [Contributing](#contributing)  
6. [License](#license)  
7. [Contact](#contact)

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

---

## Getting Started

## Installation & Quick Start

1. **Clone the repository**:
    ```bash
    git clone https://github.com/UNFmontreal/neurotheque_resources.git
    cd neurotheque_resources
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt

## Usage Examples

1. **Run a Pipeline with a YAML Configuration**

    ```bash
    python src/pipeline.py --config configs/pilot_preprocessing_strategy.yml
