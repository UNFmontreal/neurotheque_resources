"""
Simplified Five-point task preprocessing script with ICLabel-based ICA component selection.

This script is designed to be easy to use for psychology students with minimal Python experience.
Simply edit the configuration section at the top of the file and run the script.

The script follows these preprocessing steps:
1. Load raw EEG data
2. Prepare channels (set average reference)
3. Filter data (bandpass 0.1-100 Hz, notch at 50/60 Hz)
4. Create fixed-length epochs for artifact detection
5. First AutoReject pass (identify bad epochs)
6. Run ICA on the data (fit on 1 Hz high-pass copy for better convergence)
7. Use ICLabel to identify artifact components
8. Show ICA components with labels and wait for manual selection
9. Apply ICA to remove selected artifacts from 0.1-100 Hz data
10. Apply final filtering (0.1-40 Hz)
11. Save the preprocessed continuous data

Note: Epoching and final artifact rejection are left for the analysis stage.

Usage:
    python scr/strategies/fivepoint_preprocessing_simple.py
"""

import os
import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mne

# Add the project root to Python path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import preprocessing steps from the pipeline
from scr.steps.load import LoadData
from scr.steps.prepchannels import PrepChannelsStep
from scr.steps.filter import FilterStep
from scr.steps.epoching import EpochingStep
from scr.steps.autoreject import AutoRejectStep
from scr.steps.project_paths import ProjectPaths

# ============================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES FOR YOUR DATA
# ============================================================================

# Subject information
SUBJECT_ID = "01"      # Subject ID (e.g., "01", "02", etc.)
SESSION_ID = "001"     # Session ID (e.g., "001", "002", etc.)
TASK_ID = "5pt"        # Task ID (usually "5pt" for five-point task)
RUN_ID = "01"          # Run ID (e.g., "01", "02", etc.)

# File paths
CONFIG_FILE = "configs/fivepoint_pipeline.yml"  # Pipeline configuration file
INPUT_FILE = None  # Set to None to use default path from config, or specify full path like:
                  # INPUT_FILE = "e:/Yann/neurotheque_resources/data/pilot_data/sub-01_ses-001_task-5pt_run-01_raw.edf"

# Filtering parameters
HIGH_PASS_FREQ = 0.1   # High-pass filter cutoff (Hz) - removes slow drifts
LOW_PASS_FREQ = 100.0  # Low-pass filter cutoff (Hz) - initially keep up to 100 Hz for ICA
FINAL_LOW_PASS = 40.0  # Final low-pass filter after ICA cleaning
NOTCH_FREQS = [50.0]   # Power line frequencies to remove (Hz) - use [60.0] for USA, [50.0] for Europe

# ICA parameters
N_ICA_COMPONENTS = 18  # Number of ICA components (usually ~number of good EEG channels)

# ICLabel thresholds for artifact rejection (0.0 to 1.0)
# Higher values = more conservative (fewer components rejected)
# Note: These are used for display only - final selection is manual
ICLABEL_THRESHOLDS = {
    'Eye': 0.7,         # Eye movements and blinks
    'Muscle': 0.6,      # Muscle artifacts
    'Heart': 0.6,       # Heartbeat artifacts
    'Line Noise': 0.6,  # Power line noise
    'Channel Noise': 0.6 # Bad channel noise
}

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = "INFO"

# ============================================================================
# END OF CONFIGURATION - DO NOT EDIT BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
# ============================================================================


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("fivepoint_preprocessing")
    mne.set_log_level(LOG_LEVEL)
    return logger


def load_config_and_paths():
    """Load configuration file and set up paths."""
    config_path = ROOT / CONFIG_FILE
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    paths = ProjectPaths(config)
    return config, paths


def get_input_file_path(config, paths):
    """Determine the input file path."""
    if INPUT_FILE:
        return Path(INPUT_FILE)
    
    # Build path from config pattern
    root_dir = Path(config["directory"]["root"]).resolve()
    raw_dir = root_dir / config["directory"]["raw_data_dir"]
    pattern = config.get("file_path_pattern", "")
    
    if "{" in pattern:
        # Format the pattern with our subject/session info
        filename = pattern.format(
            subject=SUBJECT_ID,
            session=SESSION_ID,
            task=TASK_ID,
            run=RUN_ID
        )
    else:
        # Default filename pattern
        filename = f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_raw.edf"
    
    return raw_dir / filename


def show_ica_components_and_get_selection(ica, epochs, ic_labels, thresholds, save_dir, logger):
    """
    Show ICA components with ICLabel classifications and get manual selection from user.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    epochs : mne.Epochs
        Epochs data for visualization
    ic_labels : dict
        ICLabel classification results
    thresholds : dict
        Dictionary with artifact types as keys and threshold values
    save_dir : Path
        Directory to save the figure
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    list
        Indices of components to exclude (manually selected by user)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract ICLabel results - handle different return formats
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]
    
    # Define the standard ICLabel classes
    iclabel_classes = ['brain', 'muscle', 'eye blink', 'heart', 'line_noise', 'channel_noise', 'other']
    
    # Get classes from ic_labels if available, otherwise use standard list
    if "classes" in ic_labels:
        classes = ic_labels["classes"]
    else:
        # Use the standard ICLabel class order
        classes = iclabel_classes
        logger.debug("Using standard ICLabel class order")
    
    # Create figure showing all components with labels
    logger.info("\nCreating ICA components figure with ICLabel predictions...")
    fig = ica.plot_components(title="ICA Components with ICLabel Predictions", show=False)
    
    # Get all axes (handling different matplotlib versions)
    if hasattr(fig, 'axes'):
        if hasattr(fig.axes, 'flatten'):
            axes_list = list(fig.axes.flatten())
        elif isinstance(fig.axes, list):
            if len(fig.axes) > 0 and hasattr(fig.axes[0], '__iter__'):
                axes_list = [ax for sublist in fig.axes for ax in sublist]
            else:
                axes_list = fig.axes
        else:
            axes_list = [fig.axes]
    else:
        axes_list = []
    
    # Add ICLabel predictions to each component
    suggested_exclude = []
    for i, ax in enumerate(axes_list[:len(labels)]):
        label = labels[i]
        
        # Get the probability for the predicted label
        if isinstance(label, str):
            # Label is a string, find its index
            try:
                label_idx = classes.index(label)
                prob = probs[i][label_idx]
            except (ValueError, IndexError):
                # If label not found, use max probability
                prob = probs[i].max()
                label_idx = probs[i].argmax()
                label = classes[label_idx] if label_idx < len(classes) else 'unknown'
        else:
            # Label might be an index
            label_idx = int(label)
            prob = probs[i][label_idx] if label_idx < len(probs[i]) else probs[i].max()
            label = classes[label_idx] if label_idx < len(classes) else 'unknown'
        
        # Check if this component exceeds threshold for any artifact type
        is_artifact = False
        for artifact_type, mne_label in [
            ('Eye', 'eye blink'),
            ('Heart', 'heart'),
            ('Line Noise', 'line_noise'),
            ('Channel Noise', 'channel_noise'),
            ('Muscle', 'muscle')
        ]:
            if label == mne_label and prob >= thresholds.get(artifact_type, 0.6):
                is_artifact = True
                suggested_exclude.append(i)
                # Highlight with red border
                ax.patch.set_edgecolor('red')
                ax.patch.set_linewidth(3)
                break
        
        # Add label text
        label_text = f"{label}\n{prob*100:.1f}%"
        color = 'red' if is_artifact else 'green' if label == 'brain' else 'orange'
        ax.text(0.5, 0.05, label_text,
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                color=color,
                fontweight='bold',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Save figure
    fig_path = save_dir / "ica_components_with_labels.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ICA components figure to: {fig_path}")
    
    # Show the figure
    plt.show()
    
    # Display ICLabel suggestions in terminal
    logger.info("\n" + "="*70)
    logger.info("ICLabel AUTOMATIC SUGGESTIONS (red borders in figure):")
    logger.info("="*70)
    
    # Check if ICLabel might be giving bad results
    brain_count = sum(1 for label in labels if label == 'brain')
    if brain_count > len(labels) * 0.8:  # More than 80% labeled as brain
        logger.warning("\n*** WARNING: ICLabel classified most components as 'brain' ***")
        logger.warning("This often indicates ICLabel is not working correctly.")
        logger.warning("Please use MANUAL inspection based on topography patterns!")
        logger.warning("Common issues: data filtering, channel montage, or ICLabel version.")
    
    for i in suggested_exclude:
        label = labels[i]
        # Get probability for display
        if isinstance(label, str):
            try:
                label_idx = classes.index(label)
                prob = probs[i][label_idx]
            except:
                prob = probs[i].max()
        else:
            label_idx = int(label)
            prob = probs[i][label_idx] if label_idx < len(probs[i]) else probs[i].max()
            label = classes[label_idx] if label_idx < len(classes) else 'unknown'
        logger.info(f"  IC{i}: {label} ({prob*100:.1f}%)")
    
    if not suggested_exclude:
        logger.info("  No components exceed the artifact thresholds.")
    
    logger.info("\n" + "="*70)
    logger.info("ALL COMPONENT CLASSIFICATIONS:")
    logger.info("="*70)
    for i, (label, prob_row) in enumerate(zip(labels, probs)):
        # Get probability for display
        if isinstance(label, str):
            try:
                label_idx = classes.index(label)
                prob = prob_row[label_idx]
            except:
                prob = prob_row.max()
        else:
            label_idx = int(label)
            prob = prob_row[label_idx] if label_idx < len(prob_row) else prob_row.max()
            label = classes[label_idx] if label_idx < len(classes) else 'unknown'
        logger.info(f"  IC{i}: {label} ({prob*100:.1f}%)")
    
    # Get manual selection from user
    logger.info("\n" + "="*70)
    logger.info("MANUAL COMPONENT SELECTION")
    logger.info("="*70)
    logger.info("IMPORTANT: ICLabel predictions may be incorrect!")
    logger.info("Common artifacts to look for:")
    logger.info("  - Eye blinks: Frontal, bilateral distribution (often ICA000, ICA001, ICA002)")
    logger.info("  - Eye movements: Frontal, lateral distribution")
    logger.info("  - Muscle: High frequency, edge electrodes")
    logger.info("  - Heart: Regular pattern, may show diagonal distribution")
    logger.info("\nPlease review the figure and enter the component numbers to exclude.")
    logger.info("Enter component numbers separated by commas (e.g., 0,2,5)")
    logger.info("Enter 'plot X' to see detailed plots for component X (e.g., 'plot 0')")
    logger.info("Press Enter with no input to accept the automatic suggestions.")
    logger.info("Enter 'none' to exclude no components.")
    
    # Based on typical patterns, suggest likely artifacts
    logger.info("\n" + "-"*50)
    logger.info("MANUAL INSPECTION SUGGESTIONS:")
    logger.info("Based on topography patterns, consider reviewing these components:")
    logger.info("  - ICA000: Frontal bilateral pattern (likely eye blinks)")
    logger.info("  - ICA002: Frontal pattern (likely eye-related)")
    logger.info("  - Check any components with activity concentrated at edges")
    logger.info("-"*50)
    
    while True:
        user_input = input("\nComponents to exclude (or 'plot X' to inspect): ").strip()
        
        if user_input.lower() == 'none':
            exclude_indices = []
            break
        elif user_input == '':
            exclude_indices = suggested_exclude
            logger.info(f"Using automatic suggestions: {exclude_indices}")
            break
        elif user_input.lower().startswith('plot '):
            # Handle plotting individual components
            try:
                comp_str = user_input[5:].strip()
                comp_idx = int(comp_str)
                if 0 <= comp_idx < ica.n_components_:
                    logger.info(f"Plotting properties for component {comp_idx}...")
                    # Plot component properties
                    fig_props = ica.plot_properties(epochs, picks=[comp_idx], 
                                                  psd_args={'fmax': 45}, 
                                                  show=True)
                    # Close the figure after viewing
                    for fig in fig_props:
                        plt.close(fig)
                else:
                    logger.warning(f"Component {comp_idx} is out of range (0-{ica.n_components_-1})")
            except (ValueError, IndexError) as e:
                logger.error(f"Invalid component number: {comp_str}")
            except Exception as e:
                logger.error(f"Error plotting component: {e}")
        else:
            try:
                # Parse user input for component numbers
                exclude_indices = []
                for comp in user_input.split(','):
                    comp = comp.strip()
                    if comp:
                        comp_idx = int(comp)
                        if 0 <= comp_idx < ica.n_components_:
                            exclude_indices.append(comp_idx)
                        else:
                            logger.warning(f"Component {comp_idx} is out of range (0-{ica.n_components_-1})")
                
                exclude_indices = sorted(set(exclude_indices))
                logger.info(f"Selected components to exclude: {exclude_indices}")
                break
                
            except ValueError:
                logger.error("Invalid input. Please enter numbers separated by commas.")
    
    return exclude_indices


def main():
    """Main preprocessing function."""
    import matplotlib.pyplot as plt
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting five-point task preprocessing...")
    
    # Load configuration
    config, paths = load_config_and_paths()
    
    # Get input file path
    input_file = get_input_file_path(config, paths)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    logger.info(f"Processing file: {input_file}")
    
    # Create output directories
    processed_dir = paths.processed_dir / f"sub-{SUBJECT_ID}" / f"ses-{SESSION_ID}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories for saving figures
    figures_dir = processed_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load raw data
    # ========================================================================
    logger.info("Step 1: Loading raw EEG data...")
    load_step = LoadData(params={
        "input_file": str(input_file),
        "stim_channel": "Trigger",
        "subject_id": SUBJECT_ID,
        "session_id": SESSION_ID,
        "paths": paths,
    })
    raw = load_step.run(None)
    logger.info(f"  Loaded {len(raw.ch_names)} channels, {raw.times[-1]:.1f} seconds of data")
    
    # Save raw data PSD instead of time series (Qt browser can't be saved)
    raw_psd = raw.compute_psd(fmax=80, picks='eeg')
    raw_psd_fig = raw_psd.plot(show=False)
    raw_psd_fig.suptitle('Raw Data - Power Spectral Density', y=0.98)
    raw_psd_fig.savefig(figures_dir / "01_raw_psd.png")
    plt.close(raw_psd_fig)
    
    # ========================================================================
    # STEP 2: Prepare channels (set average reference)
    # ========================================================================
    logger.info("Step 2: Preparing channels and setting average reference...")
    prep_step = PrepChannelsStep(params={
        "on_missing": "ignore",
        "reference": {
            "method": "average",
            "projection": False
        }
    })
    raw = prep_step.run(raw)
    logger.info(f"  Channels prepared. Using average reference.")
    
    # ========================================================================
    # STEP 3: Filter data (0.1-100 Hz for ICA processing)
    # ========================================================================
    logger.info(f"Step 3: Filtering data ({HIGH_PASS_FREQ}-{LOW_PASS_FREQ} Hz)...")
    filter_step = FilterStep(params={
        "l_freq": HIGH_PASS_FREQ,
        "h_freq": LOW_PASS_FREQ,
        "notch_freqs": NOTCH_FREQS,
    })
    raw = filter_step.run(raw)
    logger.info("  Filtering complete (0.1-100 Hz for ICA).")
    
    # Save filtered data PSD
    filtered_psd = raw.compute_psd(fmax=120, picks='eeg')
    psd_fig = filtered_psd.plot(show=False)
    psd_fig.suptitle('Power Spectral Density after Initial Filtering (0.1-100 Hz)', y=0.98)
    psd_fig.savefig(figures_dir / "02_filtered_psd_100hz.png")
    plt.close(psd_fig)
    
    # ========================================================================
    # STEP 4: Create fixed-length epochs for artifact detection
    # ========================================================================
    logger.info("Step 4: Creating fixed-length epochs for artifact detection...")
    epoch_step = EpochingStep(params={
        "task_type": "fixed",
        "stim_channel": "Trigger",
        "epoch_params": {
            "duration": 4.0,      # 4-second epochs
            "overlap": 0.0,       # No overlap
            "baseline": None,     # No baseline correction yet
            "preload": True,
            "reject_by_annotation": True
        }
    })
    epochs = epoch_step.run(raw)
    logger.info(f"  Created {len(epochs)} fixed-length epochs")
    
    # ========================================================================
    # STEP 5: First AutoReject pass (identify bad epochs)
    # ========================================================================
    logger.info("Step 5: Running first AutoReject pass to identify bad epochs...")
    ar_dir = processed_dir / "autoreject"
    ar_dir.mkdir(parents=True, exist_ok=True)
    
    ar_step = AutoRejectStep(params={
        "mode": "fit",  # Only identify bad epochs, don't clean yet
        "plot_results": True,
        "interactive": False,
        "plot_dir": str(ar_dir),
        "store_reject_log": True,
        "file_prefix": "ar_first_pass",
        "subject_id": SUBJECT_ID,
        "session_id": SESSION_ID,
        "run_id": RUN_ID,
    })
    epochs_with_ar = ar_step.run(epochs)
    
    # Check how many bad epochs were found
    if 'temp' in epochs_with_ar.info and 'autoreject_bad_epochs' in epochs_with_ar.info['temp']:
        bad_epochs = epochs_with_ar.info['temp']['autoreject_bad_epochs']
        bad_pct = (len(bad_epochs) / max(1, len(epochs))) * 100.0
        logger.info(f"  AutoReject identified {len(bad_epochs)} bad epochs ({bad_pct:.1f}%)")
        if bad_pct > 35.0:
            logger.warning("  High fraction of bad epochs (>35%). Inspect data quality, filtering, and channel montage.")
        
        # Save a summary plot of the AutoReject results if available
        try:
            # AutoReject plots are saved by the AutoRejectStep itself in the ar_dir
            logger.info(f"  AutoReject plots saved in: {ar_dir}")
        except Exception as e:
            logger.debug(f"  Could not generate additional AutoReject plots: {e}")
    
    # ========================================================================
    # STEP 6: ICA extraction on the 0.1-100 Hz filtered data
    # ========================================================================
    logger.info(f"Step 6: Running ICA with {N_ICA_COMPONENTS} components...")
    
    # Create a 1 Hz high-pass filtered copy for ICA fitting (better for ICA convergence)
    logger.info("  Creating 1 Hz high-pass filtered copy for ICA fitting...")
    raw_for_ica = raw.copy()
    raw_for_ica.filter(l_freq=1.0, h_freq=100.0, phase='zero-double', fir_design='firwin')
    logger.info("  Band-pass filtered copy created for ICA fitting (1-100 Hz).")
    
    # Create epochs from the filtered data, excluding bad epochs
    events = mne.make_fixed_length_events(raw_for_ica, duration=1)
    epochs_for_ica = mne.Epochs(raw_for_ica, events, tmin=0, tmax=1, 
                                baseline=None, preload=True, reject_by_annotation=True)
    
    # Verify the filtering
    logger.info(f"  Epochs for ICA: {len(epochs_for_ica)} epochs, high-pass filtered at 1 Hz")
    
    # Exclude bad epochs identified by AutoReject
    if 'temp' in epochs_with_ar.info and 'autoreject_bad_epochs' in epochs_with_ar.info['temp']:
        bad_indices = np.asarray(epochs_with_ar.info['temp']['autoreject_bad_epochs'], dtype=int)
        # Map 4s bad epochs to 1s epochs using onset times from events arrays
        onsets_4s = epochs.events[:, 0] / raw.info['sfreq']
        bad_onsets = onsets_4s[bad_indices]
        onsets_1s = epochs_for_ica.events[:, 0] / raw.info['sfreq']

        # Build mask: drop any 1s epoch whose onset falls within a bad 4s window
        drop_mask = np.zeros(len(onsets_1s), dtype=bool)
        for t_bad in bad_onsets:
            drop_mask |= (onsets_1s >= t_bad) & (onsets_1s < t_bad + 4.0)
        epochs_to_drop = np.where(drop_mask)[0].tolist()
        if epochs_to_drop:
            epochs_for_ica.drop(epochs_to_drop)
            pct = 100.0 * len(epochs_to_drop) / max(1, len(onsets_1s))
            logger.info(f"  Excluded {len(epochs_to_drop)}/{len(onsets_1s)} ( {pct:.1f}% ) of 1s epochs from ICA fitting based on AutoReject 4s segments")
            if pct > 35.0:
                logger.warning("  High exclusion rate (>35%). Consider revisiting filtering, sensors, or AutoReject parameters.")
    
    # Fit ICA
    from mne.preprocessing import ICA
    ica = ICA(n_components=N_ICA_COMPONENTS, 
              method='infomax', 
              max_iter=2000, 
              fit_params={"extended": True}, 
              random_state=0)
    
    logger.info("  Fitting ICA (this may take a few minutes)...")
    ica.fit(epochs_for_ica, decim=3)
    logger.info(f"  ICA fitting complete. Found {ica.n_components_} components.")
    
    # ========================================================================
    # STEP 7: ICLabel classification and manual component selection
    # ========================================================================
    logger.info("Step 7: Using ICLabel to identify artifact components...")
    
    # Import ICLabel
    try:
        from mne_icalabel import label_components
    except ImportError:
        logger.error("mne-icalabel is not installed. Please install it using: pip install mne-icalabel")
        raise
    
    # Run ICLabel classification on 1-100 Hz epochs
    ic_labels = label_components(epochs_for_ica, ica, method='iclabel')
    # Fallback: if ICLabel marks nearly everything as brain, retry with Raw copy
    try:
        labels_tmp = ic_labels.get("labels", [])
        brain_frac = sum(1 for lb in labels_tmp if lb == 'brain') / max(1, len(labels_tmp))
        if brain_frac > 0.85:
            logger.warning("ICLabel classified >85% as brain on epochs. Retrying classification on continuous Raw (1-100 Hz) …")
            ic_labels = label_components(inst=raw_for_ica, ica=ica, method='iclabel')
    except Exception as _:
        pass
    
    # Create ICA plots directory
    ica_dir = processed_dir / "ica_plots"
    ica_dir.mkdir(parents=True, exist_ok=True)
    
    # Get manual selection from user
    exclude_indices = show_ica_components_and_get_selection(
        ica, epochs_for_ica, ic_labels, ICLABEL_THRESHOLDS, ica_dir, logger
    )
    ica.exclude = exclude_indices
    
    # ========================================================================
    # STEP 8: Apply ICA to remove artifacts from 0.1-100 Hz data
    # ========================================================================
    logger.info("Step 8: Applying ICA to remove artifacts from 0.1-100 Hz filtered data...")
    raw_clean = raw.copy()  # This is the 0.1-100 Hz filtered data
    ica.apply(raw_clean)
    logger.info(f"  Removed {len(exclude_indices)} artifact components: {exclude_indices}")
    
    # Save comparison plot before final filtering
    psd_before = raw.compute_psd(fmax=120, picks='eeg')
    psd_after_ica = raw_clean.compute_psd(fmax=120, picks='eeg')
    
    fig_before = psd_before.plot(show=False)
    fig_before.suptitle('Before ICA Cleaning (0.1-100 Hz)', y=0.98)
    fig_before.savefig(figures_dir / "03a_psd_before_ica.png")
    plt.close(fig_before)
    
    fig_after = psd_after_ica.plot(show=False)
    fig_after.suptitle('After ICA Cleaning (0.1-100 Hz)', y=0.98)
    fig_after.savefig(figures_dir / "03b_psd_after_ica.png")
    plt.close(fig_after)
    
    # ========================================================================
    # STEP 9: Apply final filtering (0.1-40 Hz)
    # ========================================================================
    logger.info(f"Step 9: Applying final filtering ({HIGH_PASS_FREQ}-{FINAL_LOW_PASS} Hz)...")
    final_filter_step = FilterStep(params={
        "l_freq": HIGH_PASS_FREQ,
        "h_freq": FINAL_LOW_PASS,
        "notch_freqs": [],  # Notch already applied
    })
    raw_clean = final_filter_step.run(raw_clean)
    logger.info("  Final filtering complete.")
    
    # Save final PSD
    psd_final = raw_clean.compute_psd(fmax=80, picks='eeg')
    fig_final = psd_final.plot(show=False)
    fig_final.suptitle('Final Power Spectral Density (0.1-40 Hz)', y=0.98)
    fig_final.savefig(figures_dir / "04_final_psd.png")
    plt.close(fig_final)
    
    # ========================================================================
    # STEP 10: Save the preprocessed continuous data
    # ========================================================================
    logger.info("Step 10: Saving preprocessed continuous data...")
    
    # Save the cleaned continuous data
    output_file = processed_dir / f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_preprocessed-raw.fif"
    raw_clean.save(output_file, overwrite=True)
    logger.info(f"  Saved preprocessed continuous data to: {output_file}")
    
    # Save the ICA solution for reference
    ica_file = processed_dir / f"sub-{SUBJECT_ID}_ses-{SESSION_ID}_task-{TASK_ID}_run-{RUN_ID}_ica.fif"
    ica.save(ica_file, overwrite=True)
    logger.info(f"  Saved ICA solution to: {ica_file}")
    
    # Create a summary report
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Subject: {SUBJECT_ID}, Session: {SESSION_ID}, Task: {TASK_ID}, Run: {RUN_ID}")
    logger.info(f"Input file: {input_file.name}")
    logger.info(f"Output file: {output_file.name}")
    logger.info(f"Duration: {raw_clean.times[-1]:.1f} seconds")
    logger.info(f"Channels: {len(raw_clean.ch_names)} ({len([ch for ch in raw_clean.ch_names if ch != 'Trigger'])} EEG)")
    logger.info(f"Initial filtering: {HIGH_PASS_FREQ}-{LOW_PASS_FREQ} Hz")
    logger.info(f"Final filtering: {HIGH_PASS_FREQ}-{FINAL_LOW_PASS} Hz")
    logger.info(f"ICA components removed: {len(exclude_indices)} {exclude_indices}")
    logger.info(f"All outputs saved to: {processed_dir}")
    logger.info("\nNext steps:")
    logger.info("  - Use the cleaned continuous data for epoching in your analysis")
    logger.info("  - Apply task-specific epoching parameters")
    logger.info("  - Consider additional artifact rejection if needed")
    logger.info("="*60)


if __name__ == "__main__":
    main()
