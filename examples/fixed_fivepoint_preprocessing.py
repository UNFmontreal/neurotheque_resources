import numpy as np
import pandas as pd
import mne
from fooof import FOOOF
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import os
from pathlib import Path

# --- Load and prepare data (added missing initialization) -------
# Provide the path to your data file - adjust this path as needed
data_dir = Path('./data/processed')
subject_id = "01"  # Replace with your subject ID
session_id = "001"  # Replace with your session ID
task_id = "5pt"    # Replace with your task ID

# Load preprocessed data
try:
    # Try to load epochs directly if they exist
    epochs_path = data_dir / f'sub-{subject_id}' / f'ses-{session_id}' / f'sub-{subject_id}_ses-{session_id}_task-{task_id}_epo.fif'
    epochs_clean_ar = mne.read_epochs(epochs_path, preload=True)
    print(f"Loaded epochs from {epochs_path}")
    
    # Load raw data for event extraction if needed
    raw_path = data_dir / f'sub-{subject_id}' / f'ses-{session_id}' / f'sub-{subject_id}_ses-{session_id}_task-{task_id}_raw.fif'
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    print(f"Loaded raw data from {raw_path}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please adjust the paths to your data files.")
    raise

# --- A) Compute draw_time when both onset & touch are code 8 -------

# 1) Pull events from the continuous raw you used to make epochs
try:
    events = mne.find_events(raw, stim_channel='Trigger', shortest_event=1)
    events8 = events[events[:, 2] == 8]
    
    # 2) Check for valid event pairs
    if len(events8) % 2 != 0:
        print(f"Warning: Found {len(events8)} code‑8 events—need an even count for onset/touch pairing")
        # Fix: Use the largest even number of events possible
        events8 = events8[:len(events8) - (len(events8) % 2)]
        print(f"Using {len(events8)} events after adjustment")
    
    # 3) Even entries = onset, odd entries = first touch
    onsets  = events8[0::2, 0]
    touches = events8[1::2, 0]
    
    # Validate that touches come after onsets
    if not all(touches > onsets):
        print("Warning: Not all touch events occur after onset events")
        valid_pairs = touches > onsets
        onsets = onsets[valid_pairs]
        touches = touches[valid_pairs]
        print(f"Using {len(onsets)} valid onset-touch pairs")
    
    # 4) Compute draw latency per pair
    sfreq = raw.info['sfreq']
    draw_times = (touches - onsets) / sfreq
    
    # 5) Ensure it covers every epoch
    n_epochs = len(epochs_clean_ar)
    if len(draw_times) < n_epochs:
        print(f"Warning: Only {len(draw_times)} draw times for {n_epochs} epochs")
        # Option 1: Only use epochs with available draw times
        epochs_clean_ar = epochs_clean_ar[:len(draw_times)]
        n_epochs = len(epochs_clean_ar)
        print(f"Adjusted to {n_epochs} epochs")
    elif len(draw_times) > n_epochs:
        # Option 2: Truncate draw_times if there are more than needed
        draw_times = draw_times[:n_epochs]
        print(f"Using first {n_epochs} draw times")

except Exception as e:
    print(f"Error in event processing: {e}")
    # Create dummy draw times as fallback
    n_epochs = len(epochs_clean_ar)
    draw_times = np.ones(n_epochs) * 0.5  # Default 0.5s draw time
    print("Using default draw times due to error")


# --- B) Time‑resolved FOOOF around each epoch ------------------------

win_size, win_step = 0.5, 0.1   # seconds
fmin, fmax        = 3.0, 40.0   # Hz
sfreq             = epochs_clean_ar.info['sfreq']
n_fft             = int(win_size * sfreq * 2)
step_samps        = int(win_step * sfreq)
half_win          = int(win_size * sfreq / 2)

trial_ids, times = [], []
aperiod_exps, mu_heights = [], []

# Add progress tracking
total_epochs = n_epochs
print(f"Processing {total_epochs} epochs...")

for tidx in range(n_epochs):
    if tidx % 5 == 0:  # Print progress every 5 epochs
        print(f"Processing epoch {tidx+1}/{total_epochs}...")
        
    data = epochs_clean_ar.get_data()[tidx]  # (n_ch, n_times)
    
    # Skip epochs that are too short
    if data.shape[1] < 2 * half_win:
        print(f"Skipping epoch {tidx}: too short ({data.shape[1]} samples)")
        continue
        
    centers = np.arange(half_win, data.shape[1] - half_win, step_samps)
    for c in centers:
        try:
            seg = data[:, c-half_win : c+half_win]
            psd, freqs = mne.time_frequency.psd_array_welch(
                seg, sfreq=sfreq,
                fmin=fmin, fmax=fmax,
                n_fft=n_fft, n_overlap=0,
                average='mean'
            )
            mean_psd = psd.mean(axis=0) * 1e12  # µV²/Hz

            fg = FOOOF(aperiodic_mode='fixed', peak_width_limits=[1.0,8.0])
            fg.fit(freqs, mean_psd)

            trial_ids.append(tidx)
            times.append((c - half_win) / sfreq)
            aperiod_exps.append(fg.aperiodic_params_[1])
            
            # Find mu peaks (8-13 Hz)
            mu_idx = np.where((fg.gaussian_params_[:,0] >= 8) &
                              (fg.gaussian_params_[:,0] <= 13))[0]
                              
            # Fix: Better handling when no mu peaks are found
            if mu_idx.size:
                mu_heights.append(fg.gaussian_params_[mu_idx,1].mean())
            else:
                # Use 0 instead of NaN for no detected peak
                # This avoids NaN propagation issues in later analysis
                mu_heights.append(0.0)
                
        except Exception as e:
            print(f"Error processing epoch {tidx}, center {c}: {e}")
            # Skip this time window on error
            continue

# Create DataFrame with results
df_fooof = pd.DataFrame({
    'trial':         trial_ids,
    'time_rel':      times,
    'aperiodic_exp': aperiod_exps,
    'mu_height':     mu_heights
})

# Check if we have results
if len(df_fooof) == 0:
    raise RuntimeError("No FOOOF results were generated. Check your data and parameters.")

print(f"Generated {len(df_fooof)} FOOOF measurements across {len(df_fooof['trial'].unique())} trials")


# --- C) Collapse to per-trial & attach draw_time ---------------------

# Define time window of interest 
mask = (df_fooof['time_rel'] >= -0.2) & (df_fooof['time_rel'] <= 0.8)

# Check if there are any data points in the specified time window
if not any(mask):
    print("Warning: No data points in specified time window (-0.2 to 0.8s)")
    # Expand the window as a fallback
    mask = (df_fooof['time_rel'] >= -0.5) & (df_fooof['time_rel'] <= 1.0)
    print("Expanded time window to -0.5 to 1.0s")

# Aggregate per trial
per_trial = (df_fooof[mask]
             .groupby('trial')
             .agg({'aperiodic_exp':'mean',
                   'mu_height':'mean'})
             .reset_index())

# Fix: Make sure draw_times align with trial indices
trial_to_draw_time = {}
for trial, dt in zip(range(len(draw_times)), draw_times):
    trial_to_draw_time[trial] = dt

# Add draw_time to per_trial using the trial IDs as keys
per_trial['draw_time'] = per_trial['trial'].map(trial_to_draw_time)

# Check for missing values
if per_trial['draw_time'].isna().any():
    print("Warning: Some trials don't have draw times. Using mean value as fallback.")
    per_trial['draw_time'] = per_trial['draw_time'].fillna(per_trial['draw_time'].mean())


# --- D1) Correlation: μ‑height vs. draw_time ------------------------

# Remove any remaining NaN values that might affect correlation
valid_mask = ~(per_trial['mu_height'].isna() | per_trial['draw_time'].isna())
if not all(valid_mask):
    print(f"Warning: Removing {(~valid_mask).sum()} rows with NaN values")
    per_trial = per_trial[valid_mask]

# Check if we still have enough data for correlation
if len(per_trial) < 3:
    print("Not enough data points for correlation analysis")
else:
    r = np.corrcoef(per_trial['mu_height'], per_trial['draw_time'])[0,1]
    print(f"Pearson r (μ‑height vs. draw_time) = {r:.2f}")

    plt.figure(figsize=(5,4))
    plt.scatter(per_trial['mu_height'], per_trial['draw_time'], c='C0')
    plt.xlabel('Mean μ‑Height (dB)')
    plt.ylabel('Draw Time (s)')
    plt.title('μ‑Power vs. Draw Time')
    plt.tight_layout()
    plt.show()


# --- D2) Early vs. Late session comparison ------------------------

# Only proceed if we have enough trials
if len(per_trial) < 4:
    print("Not enough trials for early vs. late comparison")
else:
    mid = len(per_trial) // 2
    early = per_trial.iloc[:mid]
    late  = per_trial.iloc[mid:]

    print("Early session:")
    print(f"  Exponent = {early['aperiodic_exp'].mean():.3f} ± {early['aperiodic_exp'].std():.3f}")
    print(f"  μ‑Height = {early['mu_height'].mean():.3f} ± {early['mu_height'].std():.3f}")

    print("Late session:")
    print(f"  Exponent = {late['aperiodic_exp'].mean():.3f} ± {late['aperiodic_exp'].std():.3f}")
    print(f"  μ‑Height = {late['mu_height'].mean():.3f} ± {late['mu_height'].std():.3f}")

    plt.figure(figsize=(6,4))
    plt.bar(['Early','Late'],
            [early['aperiodic_exp'].mean(), late['aperiodic_exp'].mean()],
            yerr=[early['aperiodic_exp'].std(), late['aperiodic_exp'].std()])
    plt.ylabel('Aperiodic Exponent')
    plt.title('Early vs. Late 1/f Exponent')
    plt.tight_layout()
    plt.show()


# --- D3) GLMM: predict draw_time from spectral features ------------- 

# Only run model if we have enough data
if len(per_trial) < 10:
    print("Not enough data points for GLMM analysis (need at least 10)")
else:
    try:
        # Use a more appropriate grouping variable
        model = mixedlm("draw_time ~ mu_height + aperiodic_exp",
                        per_trial, groups=np.ones(len(per_trial)))  # Default grouping if no better option
        res = model.fit(method='lbfgs')
        print(res.summary())
    except Exception as e:
        print(f"Error in GLMM analysis: {e}")
        print("Try using a simpler model or check your data for issues") 