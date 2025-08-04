# Neurotheque Resources

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

<div align="center">
  <img src="docs/images/pipeline_animation_v2.gif" alt="Neurotheque Pipeline Steps" width="600" />
</div>

**Neurotheque Resources** is a comprehensive, modular toolkit for EEG data processing and analysis. Built on state-of-the-art libraries such as [MNE-Python](https://mne.tools/stable/index.html) and [autoreject](https://autoreject.github.io/), it provides a flexible framework for building reproducible EEG processing pipelines.

## Core Features

- **Modular Pipeline Architecture**: Build custom pipelines by combining a series of processing steps.
- **Configuration-driven**: Define your pipelines using simple YAML files.
- **BIDS Compatibility**: Organized to work with BIDS-formatted data.
- **Artifact Rejection**: Includes advanced methods like ICA and autoreject.
- **Automated Reporting**: Generate HTML reports with key metrics and plots.
- **Command-Line Interface**: Run your pipelines easily from the terminal.

## Quick Start

Get up and running with Neurotheque in just a few steps.

### 1. Installation

First, clone the repository and install the package in editable mode. This will also install all the required dependencies.

```bash
git clone https://github.com/YannFeurprier/neurotheque_resources.git
cd neurotheque_resources
pip install -e .
```

### 2. Run a Demo Pipeline

You can run a demo pipeline using one of the provided configuration files. For example, to run the minimal Go/No-Go pipeline:

```bash
neurotheque run-pipeline configs/gonogo_minimal_pipeline.yml
```

This will process the sample data and generate the results in the `data/processed` and `reports` directories.

## Usage

### Command-Line Interface (CLI)

The primary way to use Neurotheque is through its command-line interface.

`neurotheque run-pipeline [CONFIG_FILE]`

- `CONFIG_FILE`: Path to the YAML configuration file for the pipeline.

### Creating a Custom Pipeline

To create your own pipeline, you can create a new YAML configuration file. The configuration file specifies the steps to be executed and their parameters.

Here is an example of a simple pipeline configuration:

```yaml
directory:
  root: "data"
  raw_data_dir: "raw"
  processed_dir: "processed"
  reports_dir: "reports"
  derivatives_dir: "derivatives"

file_path_pattern: "sub-01/ses-001/eeg/*_eeg.vhdr"

pipeline:
  steps:
    - name: LoadData
    - name: Filter
      params:
        l_freq: 1.0
        h_freq: 40.0
    - name: Epoching
      params:
        task_type: "gng"
        trigger_ids:
          go: 1
          nogo: 2
          response: 3
    - name: GenerateReport
```

You can find more examples in the `configs` directory.

## For Developers

### Setting up the Development Environment

To install the development dependencies, including `pytest` for testing, run:

```bash
pip install -e .[test]
```

### Running Tests

To run the test suite, use `pytest`:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.
