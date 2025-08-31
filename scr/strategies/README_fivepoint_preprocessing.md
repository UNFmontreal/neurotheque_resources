# Five-Point Task Preprocessing Guide

This guide explains how to use the simplified preprocessing script for five-point EEG data.

## Quick Start

1. **Open the script**: `scr/strategies/fivepoint_preprocessing_simple.py`

2. **Edit the configuration section** at the top of the file:
   ```python
   # Subject information
   SUBJECT_ID = "01"      # Change to your subject ID
   SESSION_ID = "001"     # Change to your session ID
   TASK_ID = "5pt"        # Usually keep as "5pt"
   RUN_ID = "01"          # Change to your run ID
   
   # Optional: specify the exact file path
   INPUT_FILE = "path/to/your/data.edf"  # Or leave as None to use default
   ```

3. **Run the script**:
   ```bash
   python scr/strategies/fivepoint_preprocessing_simple.py
   ```

4. **Follow the interactive prompts**:
   - The script will show you ICA components with labels
   - Review the figure that pops up
   - Enter component numbers to exclude (or press Enter to accept suggestions)

## What the Script Does

1. **Loads your EEG data**
2. **Prepares channels** - sets average reference
3. **Filters data** - initial filtering (0.1-100 Hz bandpass + notch filter)
4. **Identifies bad segments** - uses AutoReject to find artifacts
5. **Runs ICA** - separates brain and artifact components
6. **Shows component labels** - uses ICLabel to identify artifacts
7. **Waits for your input** - you choose which components to remove
8. **Cleans the data** - removes selected artifact components
9. **Final filtering** - applies 0.1-40 Hz bandpass filter
10. **Saves cleaned data** - as continuous data for later analysis

## Output Files

All outputs are saved in: `data/processed/sub-XX/ses-XXX/`

- **Preprocessed data**: `sub-XX_ses-XXX_task-5pt_run-XX_preprocessed-raw.fif`
- **ICA solution**: `sub-XX_ses-XXX_task-5pt_run-XX_ica.fif`
- **Figures**: saved in `figures/` subdirectory
  - `01_raw_psd.png` - raw data frequency spectrum
  - `02_filtered_psd_100hz.png` - frequency spectrum after initial filtering (0.1-100 Hz)
  - `03a_psd_before_ica.png` - frequency spectrum before ICA cleaning (0.1-100 Hz)
  - `03b_psd_after_ica.png` - frequency spectrum after ICA cleaning (0.1-100 Hz)
  - `04_final_psd.png` - final frequency spectrum after 0.1-40 Hz filtering
  - `ica_components_with_labels.png` - all ICA components with labels
- **AutoReject plots**: saved in `autoreject/` subdirectory

## Understanding the ICA Component Selection

When the figure appears:
- **Green labels**: Brain components (keep these)
- **Red labels**: Artifact components (consider removing)
- **Orange labels**: Other/uncertain components

**⚠️ IMPORTANT: ICLabel may give incorrect predictions!**

If you see most components labeled as "brain" with high confidence, ICLabel is likely not working correctly. In this case, use manual inspection:

### Manual Inspection Guidelines:
- **Eye blinks**: Look for bilateral frontal patterns (usually ICA000, ICA001, or ICA002)
- **Eye movements**: Frontal or lateral patterns
- **Muscle artifacts**: High-frequency noise, often at edge electrodes
- **Heart artifacts**: Regular, rhythmic pattern, sometimes diagonal distribution
- **Bad channels**: Single electrode patterns

### Interactive Options:
- Type `plot 0` to see detailed plots for component 0 (time series, spectrum, etc.)
- Type `0,2,5` to exclude components 0, 2, and 5
- Press **Enter** to accept automatic suggestions
- Type `none` to keep all components

### Common Artifacts to Remove:
Based on topography alone, these patterns are typically artifacts:
- Frontal bilateral activity (eye blinks)
- Lateral frontal activity (horizontal eye movements)
- Edge electrode activity (muscle/movement)
- Single channel patterns (bad electrodes)

## Next Steps

After preprocessing, use the cleaned data for:
- Epoching around specific events
- Event-related potential (ERP) analysis
- Time-frequency analysis
- Statistical comparisons

The cleaned data is saved as continuous (not epoched) so you can apply any epoching strategy during analysis.

## Troubleshooting

**Error: "mne-icalabel is not installed"**
- Install it: `pip install mne-icalabel`

**Error: "File not found"**
- Check your SUBJECT_ID, SESSION_ID, etc. match your file naming
- Or specify the full path in INPUT_FILE

**Too many/few components removed**
- The script allows manual selection - choose fewer/more components
- You can always re-run with different choices

## Tips for Good Preprocessing

1. **Check the figures** - they show the effect of each step
2. **Be conservative** - when unsure, remove fewer components
3. **Document your choices** - note which components you removed
4. **Be consistent** - use similar criteria across subjects

## Why This Workflow?

The script uses a specific filtering strategy:
- **Initial 0.1-100 Hz filter** preserves all frequencies for ICA to work with
- **ICA can clean artifacts** across the full frequency spectrum
- **Final 0.1-40 Hz filter** removes high frequencies only after cleaning

This approach ensures ICA has access to all frequency information for better artifact identification and removal.

For questions or issues, consult the MNE-Python documentation or your lab's EEG expert.
