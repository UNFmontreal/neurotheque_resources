# Quick Start

This guide will walk you through the basic steps to get Neuroflow up and running.

## Installation

You can install Neuroflow directly from the Git repository:

```bash
pip install .
```

For development, install with the `dev` extras:

```bash
pip install -e .[dev]
```

## Creating a Configuration

Neuroflow uses YAML files for configuration. You can create one from scratch or use the interactive wizard:

```bash
neuroflow new-config
```

## Running a Pipeline

Once you have a `config.yml` file, you can run the pipeline with the `run` command:

```bash
neuroflow run config.yml
```
