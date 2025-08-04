# Concepts

This section covers some of the core concepts behind Neuroflow.

## Pipeline

The `Pipeline` is the main class that orchestrates the execution of a series of steps. It's responsible for loading the configuration, managing data, and running each step in the correct order.

## Steps

A `Step` is an individual processing unit in the pipeline. Each step takes data as input, performs a specific operation (e.g., filtering, epoching, ICA), and returns the transformed data as output.

## Configuration

Pipelines are configured using YAML files. The configuration file specifies the directories for data, the pipeline mode, and the sequence of steps to be executed.

## BIDS

Neuroflow follows the Brain Imaging Data Structure (BIDS) standard for organizing data. This makes your data and results easy to share and understand.
