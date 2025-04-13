# Go/No-Go Task Processing Pipeline Summary

## Overview

We have successfully created three preprocessing pipeline configurations for Go/No-Go task data using the Neurotheque Resources framework:

1. **Minimal Pipeline**: Basic preprocessing, filtering, and epoching
2. **Intermediate Pipeline**: Includes artifact rejection and ICA with automatic component removal
3. **Advanced Pipeline**: Includes ICA with manual component selection

## Implemented Custom Steps

To overcome limitations in the standard steps, we implemented several custom steps:

1. **CustomPrepChannelsStep**: Handles ECG channels and missing montage positions properly
2. **CustomEpochingStep**: Properly uses the specified stim_channel for finding events
3. **SimpleAutoRejectStep**: Uses basic thresholding for artifact rejection without requiring channel positions

## Pipeline Configuration Files

The pipelines are configured in YAML files:
- `configs/gonogo_minimal_pipeline.yml`
- `configs/gonogo_intermediate_pipeline.yml`
- `configs/gonogo_advanced_pipeline.yml`

## How to Use

To run the pipelines, use the script with the following command:

```bash
python gonogo_preprocessing_pipelines.py --config {minimal|intermediate|advanced} [--interactive]
```

- The `--config` option selects which pipeline to run
- The `--interactive` option enables interactive mode for ICA component selection (useful for advanced pipeline)
- You can also use `--save-only` to generate the configuration file without running the pipeline

## Generated Files

The pipeline outputs are saved in the `data/processed/sub-01/ses-001/` directory with the following naming convention:
- `sub-01_ses-001_task-gng_run-01_after_loaddata.fif`
- `sub-01_ses-001_task-gng_run-01_after_customprepchannels.fif`
- `sub-01_ses-001_task-gng_run-01_after_filter.fif`
- `sub-01_ses-001_task-gng_run-01_after_customepoching.fif`
- `sub-01_ses-001_task-gng_run-01_after_simpleautoreject.fif`
- `sub-01_ses-001_task-gng_run-01_after_autoreject.fif` (final output)

## Key Features

1. **Channel Handling**: Proper handling of different channel types (EEG, ECG, trigger)
2. **Trigger Detection**: Correctly identifies Go/No-Go task events from the Trigger channel
3. **Artifact Rejection**: Two-stage approach with initial artifact rejection, then ICA, followed by final artifact rejection
4. **Flexibility**: Choose between automatic or manual ICA component selection

## Next Steps

The processed data can now be used for further analysis, such as:
1. ERP analysis of Go vs. No-Go responses
2. Time-frequency analysis 
3. Statistical comparison between conditions
4. Machine learning on the cleaned EEG features 