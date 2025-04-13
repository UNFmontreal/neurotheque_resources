# Neurotheque Pipeline Testing Guide

This guide explains how to use the testing utilities for debugging the Neurotheque pipeline's direct step functionality.

## Quick Start

To run the test with the example dataset:

1. Windows: run `run_test.bat`
2. Linux/Mac: `python test_with_real_data.py`

## Available Test Files

### `test_with_real_data.py`

This script loads real EEG data and processes it through each step of the pipeline:

1. **LoadData**: Loads the EDF file with automatic stim channel detection
2. **PrepChannelsStep**: Handles channel renaming and exclusions
3. **ReferenceStep**: Applies average reference
4. **FilterStep**: Applies bandpass and notch filters
5. **ICAStep**: Performs ICA for artifact removal
6. **AutoRejectStep**: Detects and marks bad epochs
7. **EpochingStep**: Creates epochs from events in the data

The script includes robust error handling, so if one step fails, the test continues with the previous data.

## Debug Outputs

The test produces several debugging outputs:

1. **Console output**: Shows progress and basic information
2. **debug_log.txt**: Detailed log including errors and warnings
3. **test_output/plots/**: Visualization plots at each processing stage
4. **test_output/ica/**: ICA component plots
5. **test_output/autoreject/**: AutoReject model and results

## Troubleshooting Common Issues

### Data Loading Issues

If you encounter issues loading the data:

```
Error loading data: No stimulus channel found
```

Check that the EDF file is accessible and has the expected format. The script will try to load with `stim_channel=None` if auto-detection fails.

### Event Finding Issues

If the script can't find events in the data:

```
Error finding events: No events found
```

The script attempts to:
1. Use the default stim channel
2. Try channels with "TRIG" or "STI" in their name
3. Create fixed-length events if no events are found

### ICA Issues

If the ICA step fails:

```
Error in ICA step: ...
```

This could be due to:
- Insufficient or invalid data
- Convergence issues
- Memory constraints for large datasets

The test will automatically reduce the number of components if needed.

## Customizing the Test

You can modify `test_with_real_data.py` to:

1. Use a different data file by changing `data_file = "..."`
2. Adjust parameters for each step (e.g., filter frequencies, ICA components)
3. Add or remove processing steps
4. Change visualization options

## Example Pipeline Configuration

To create a config file based on successful test parameters:

```python
from test_with_real_data import generate_pipeline_config

# After running the test
generate_pipeline_config("my_config.yaml")
```

This function is not included by default but could be added based on your needs.

## Next Steps

After a successful test:

1. Review the plots in `test_output/plots/` to assess data quality
2. Check the log for warnings or performance issues
3. Examine the saved epochs file (`sub-01_ses-001_task-5pt_run-01_epo.fif`)
4. Refine parameters and rerun if needed 