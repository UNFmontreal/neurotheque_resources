import os
import sys
import json
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Add the parent directory to path to import custom modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mario_preprocessing_bids')

import mne
from mne_bids import read_raw_bids, BIDSPath, make_dataset_description

# Prefer Qt browser; fall back to matplotlib if needed
try:
    mne.viz.set_browser_backend("qt")
except Exception:
    mne.viz.set_browser_backend("matplotlib")

# Input paths (update if needed)
mario_edf_path = r"D:\Yann\neurotheque_resources\data\mario\eeg\sub-01_ses-001_task-mario-eeg_run-01_raw.edf"
behav_tsv_path = r"D:\Yann\neurotheque_resources\data\mario\behav\sub-01\ses-001\sub-01_ses-001_20230203-102154_task-mario_run-01_events.tsv"

# Derive subject/session/run
filename = Path(mario_edf_path).stem
parts = filename.split('_')
sub_id = parts[0].replace('sub-', '')
ses_id = parts[1].replace('ses-', '')
run_id = parts[3].replace('run-', '')
task_label = 'mario'

print(f"BIDSifying: sub-{sub_id}, ses-{ses_id}, task-{task_label}, run-{run_id}")

# BIDS raw root (prefer derivatives/bids per repo examples)
bids_root = project_root / 'derivatives' / 'bids'
bids_root.mkdir(parents=True, exist_ok=True)

# Derivatives root for this preprocessing (BIDS derivatives layout)
pipeline_label = 'mario-preproc'
deriv_root = project_root / 'derivatives' / pipeline_label
deriv_root.mkdir(parents=True, exist_ok=True)

# Ensure derivatives dataset_description exists
make_dataset_description(
    path=str(deriv_root),
    name=pipeline_label,
    dataset_type='derivative',
    overwrite=False,
    generated_by=[{
        'Name': pipeline_label,
        'Version': '0.1',
        'Description': 'Mario preprocessing: bandpass filter, epoching, AutoReject.'
    }]
)


# Use the DSI→BIDS helper with Mario behavior alignment
from scr.utils.dsi24_bids import bidsify_edf

# Route Mario alignment summaries into derivatives (figures)
summary_dir = deriv_root / f'sub-{sub_id}' / f'ses-{ses_id}' / 'figures' / f'task-{task_label}' / f'run-{run_id}' / 'mario_alignment'
summary_dir.mkdir(parents=True, exist_ok=True)

logger.info("Running BIDSification with Mario behavior alignment...")
written = bidsify_edf(
    edf_path=mario_edf_path,
    bids_root=bids_root,
    subject=sub_id,
    session=ses_id,
    task=task_label,
    run=run_id,
    line_freq=60,
    overwrite=True,
    apply_montage=True,
    event_map_json=None,
    task_splits_json=None,
    mario_behav_tsv=behav_tsv_path,
    mario_align_params={
        'plot': True,
        'summary_dir': str(summary_dir),
        # Cosmetic: shift level-starts by 1 frame to reduce overlap in TSV
        'shift_level_starts_frames': 1,
        # Hard requirement for real frame pulses; set False only if none exist
        'require_frame_pulses': True,
        # Use EEG-estimated FPS to match measured frame pulses, not metadata
        'prefer_meta_fps': False,
        # Use photodiode level anchors for robust offset/drift
        # Adjust the bit below to the photodiode/level trigger bit on your stim channel
        'level_bit_mask': 4,
        # Let the aligner estimate offset + drift (no first-event forcing)
        'align_first_event': False,
        'force_zero_offset': False,
        'max_match_dist_s': 3.0,
        'disable_drift': False,
        'drift_threshold_ms_per_min': 0.5,
    },
)

bp = written[0]
print(f"Wrote BIDS to: {bp.directory}")

# Load BIDSified recording (BrainVision) with annotations/events
logger.info("Loading BIDSified raw...")

raw = read_raw_bids(bids_path=bp, verbose=False)

raw.load_data()

print(f"Loaded BIDS raw: {raw}")
print(f"Channels: {len(raw.ch_names)}, sfreq: {raw.info['sfreq']} Hz, duration: {raw.times[-1]:.1f}s")

# Re-apply standard montage to ensure channel positions (required by AutoReject)
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)
logger.info("Applied standard_1020 montage to Raw (ensures channel positions).")


# Optional: apply average reference (keeps annotations intact)
raw.set_eeg_reference(ref_channels='average')


# Events from annotations (Mario-aligned, per-key or aggregated)
events, event_id = mne.events_from_annotations(raw, verbose=False)
print(f"Found {len(events)} events; labels: {list(event_id.keys())}")

# Epoch around keypresses (aggregate per-key to a single label 'kp_all')
epochs = None
from scr.steps.filter import FilterStep

# Filter parameters
filter_params = {
    "l_freq": 1.0,  
    "h_freq": 100.0,  
    "method": "fir",
    "fir_design": "firwin",
    "phase": "zero",
    "filter_length": "auto",
    "l_trans_bandwidth": "auto",
    "h_trans_bandwidth": "auto",
    "picks": "eeg"  # Only filter EEG channels
}

# Run filtering
logger.info("Applying bandpass filter and saving derivative raw...")
filter_step = FilterStep(params=filter_params)
raw_filtered = filter_step.run(raw)

# Prepare BIDS-derivative paths
prefix = f"sub-{sub_id}_ses-{ses_id}_task-{task_label}_run-{run_id}"
eeg_deriv_dir = deriv_root / f'sub-{sub_id}' / f'ses-{ses_id}' / 'eeg'
eeg_deriv_dir.mkdir(parents=True, exist_ok=True)

# Save filtered Raw as derivative
bandpass_path = eeg_deriv_dir / f"{prefix}_desc-bandpass_raw.fif"
raw_filtered.save(str(bandpass_path), overwrite=True)
logger.info(f"Saved bandpass Raw: {bandpass_path}")

# Epoching around keypresses (preserve per-key labels)
# Select only keypress-related events
kp_event_id = {str(name): int(code) for name, code in event_id.items() if str(name).startswith('kp_') or str(name) == 'beh_keypress'}
kp_codes = np.array(list(kp_event_id.values()), dtype=int)
ev_kp = events[np.isin(events[:, 2], kp_codes)].copy()
# De-duplicate by sample within each code
if ev_kp.size:
    key = ev_kp[:, [0, 2]]
    _, uniq_idx = np.unique(key, axis=0, return_index=True)
    ev_kp = ev_kp[np.sort(uniq_idx)]
event_id_epoch = kp_event_id if kp_event_id else {'kp_all': 901}
tmin_kp, tmax_kp = -1.2, 0.6
logger.info(f"Epoching around keypresses: n={len(ev_kp)}")
epochs = mne.Epochs(
    raw_filtered, ev_kp, event_id=event_id_epoch,
    tmin=tmin_kp, tmax=tmax_kp,
    baseline=(None, 0), preload=True,
    reject_by_annotation=True, verbose=True,
    event_repeated='drop'
)

# Save epochs to derivatives
epochs_path = eeg_deriv_dir / f"{prefix}_desc-kp_epo.fif"
epochs.save(str(epochs_path), overwrite=True)
logger.info(f"Saved epochs: {epochs_path}")

from scr.steps.autoreject import AutoRejectStep
from scr.steps.ica_extraction import ICAExtractionStep
from scr.steps.ica_labeling import ICALabelingStep

# AutoReject outputs into derivatives
ar_base_dir = eeg_deriv_dir / 'autoreject'
ar_fig_dir = ar_base_dir / 'figures'
ar_model_dir = ar_base_dir / 'models'
for d in (ar_fig_dir, ar_model_dir):
    d.mkdir(parents=True, exist_ok=True)

# Toggle: try to reuse a previously saved AutoReject model
use_saved_ar_model = True

# First AutoReject parameters - only identify bad epochs
ar_params_first = {
    "ar_params": {
        "n_interpolate": [1, 4],
        "consensus": None,
        "thresh_method": "bayesian_optimization",
        "n_jobs": 1,
        "verbose": "tqdm"
    },
    "plot_results": True,
    "mode": "fit",  # Only identify bad epochs
    "file_prefix": "ar_first_pass",
    "plot_dir": str(ar_fig_dir),
    "store_reject_log": True,
    "save_model": True,
    "model_filename": f"{prefix}_ar_model.pkl",
    "output_dir": str(ar_model_dir),
    "load_model": use_saved_ar_model,
    "subject_id": sub_id,
    "session_id": ses_id,
    "run_id": run_id,
    "save_cleaned_data": False
}

# Run first AutoReject
logger.info("Running first AutoReject pass to identify bad epochs...")
autoreject_step_first = AutoRejectStep(params=ar_params_first)
epochs_with_ar = autoreject_step_first.run(epochs)

# Get bad epochs info
temp = epochs_with_ar.info['temp']
bad_epochs = temp.get('autoreject_bad_epochs', [])
print(f"\nIdentified {len(bad_epochs)} bad epochs out of {len(epochs)} ({(len(bad_epochs)/len(epochs)*100 if len(epochs) else 0):.1f}%)")
print(f"Bad epoch indices: {bad_epochs[:10]}..." if len(bad_epochs) > 10 else f"Bad epoch indices: {bad_epochs}")

# Persist AR summary as JSON alongside derivatives
ar_log = temp.get('autoreject', None)
arlog_path = ar_base_dir / f"{prefix}_desc-arlog.json"
# Make it JSON serializable
serializable = {
    "bad_epochs": list(map(int, ar_log.get('bad_epochs', []))) if isinstance(ar_log.get('bad_epochs', []), (list, tuple, np.ndarray)) else [],
    "ch_names": ar_log.get('ch_names', []),
    "n_interpolate": ar_log.get('n_interpolate', None),
    "consensus": ar_log.get('consensus', None),
    "n_epochs_total": int(ar_log.get('n_epochs_total', 0)),
    "n_epochs_bad": int(ar_log.get('n_epochs_bad', 0)),
}
with open(arlog_path, 'w', encoding='utf-8') as f:
    json.dump(serializable, f, indent=2)
logger.info(f"Saved AutoReject log: {arlog_path}")

# -----------------------------
# ICA extraction (EEG-only)
# -----------------------------
logger.info("Starting ICA extraction on epochs (EEG picks only)...")

# ICA directories under derivatives
ica_base_dir = eeg_deriv_dir / 'ica'
ica_extr_dir = ica_base_dir / 'extraction'
ica_label_dir = ica_base_dir / 'labeling'
for d in (ica_extr_dir, ica_label_dir):
    d.mkdir(parents=True, exist_ok=True)

ica_params = {
    "n_components": 0.99,
    "method": "infomax",
    "max_iter": 2000,
    "decim": 3,
    "picks": "eeg",  # enforce EEG-only
    "use_good_epochs_only": True,
    "interactive": False,
    "plot_components": True,
    "plot_sources": False,
    "plot_properties": True,
    "plot_psd": False,
    "plot_dir": str(ica_extr_dir),
    "subject_id": sub_id,
    "session_id": ses_id,
    "task_id": task_label,
    "run_id": run_id,
}
ica_step = ICAExtractionStep(params=ica_params)
epochs_with_ica = ica_step.run(epochs_with_ar)

# Save ICA decomposition to derivatives
ica = epochs_with_ica.info['temp']['ica']
ica_path = eeg_deriv_dir / f"{prefix}_desc-ica_decomposition.fif"
ica.save(str(ica_path), overwrite=True)
logger.info(f"Saved ICA decomposition: {ica_path}")


# -----------------------------
# ICA labeling (automatic) + reconstruction
# -----------------------------
logger.info("Running ICA labeling (automatic) and reconstructing cleaned epochs...")

ica_label_params = {
    "methods": ["iclabel", "correlation"],
    "thresholds": {  # keep robust defaults
        "iclabel": {
            "eye": 0.8, "heart": 0.8, "muscle": 0.8, "line_noise": 0.8, "channel_noise": 0.8, "other": 0.8
        },
        "correlation": {"eog": 0.8, "ecg": 0.8}
    },
    "interactive": False,
    "plot_labeled": True,
    "plot_before_after": True,
    "plot_dir": str(ica_label_dir),
    "reconstruct": True,
    "auto_exclude": True,
    "save_data": True,
    "output_file": str(eeg_deriv_dir / f"{prefix}_desc-ica_clean_epo.fif"),
    "subject_id": sub_id,
    "session_id": ses_id,
    "task_id": task_label,
    "run_id": run_id,
}
ica_label_step = ICALabelingStep(params=ica_label_params)
epochs_ica_clean = ica_label_step.run(epochs_with_ica)

# -----------------------------
# Final AutoReject: fit_transform and save cleaned
# -----------------------------
logger.info("Final AutoReject (fit_transform) on ICA-cleaned epochs...")

ar_params_final = {
    "ar_params": {
        "n_interpolate": [1, 4],
        "consensus": None,
        "thresh_method": "bayesian_optimization",
        "n_jobs": 1,
        "verbose": "tqdm"
    },
    "plot_results": True,
    "mode": "fit_transform",
    "file_prefix": f"task-{task_label}_desc-clean",  # yields sub-..._task-..._desc-clean_epo.fif
    "plot_dir": str(ar_fig_dir),
    "store_reject_log": True,
    "save_model": False,
    "output_dir": str(eeg_deriv_dir),
    "subject_id": sub_id,
    "session_id": ses_id,
    "run_id": run_id,
    "save_cleaned_data": True
}
autoreject_step_final = AutoRejectStep(params=ar_params_final)
epochs_final_clean = autoreject_step_final.run(epochs_ica_clean)

# Persist final AR log JSON
try:
    temp_final = epochs_final_clean.info.get('temp', {}) if hasattr(epochs_final_clean, 'info') else {}
    ar_log2 = temp_final.get('autoreject', None)
    arlog2_path = ar_base_dir / f"{prefix}_desc-final-arlog.json"
    if ar_log2 is not None:
        serializable2 = {
            "bad_epochs": list(map(int, ar_log2.get('bad_epochs', []))) if isinstance(ar_log2.get('bad_epochs', []), (list, tuple, np.ndarray)) else [],
            "ch_names": ar_log2.get('ch_names', []),
            "n_interpolate": ar_log2.get('n_interpolate', None),
            "consensus": ar_log2.get('consensus', None),
            "n_epochs_total": int(ar_log2.get('n_epochs_total', 0)),
            "n_epochs_bad": int(ar_log2.get('n_epochs_bad', 0)),
        }
        with open(arlog2_path, 'w', encoding='utf-8') as f:
            json.dump(serializable2, f, indent=2)
        logger.info(f"Saved final AutoReject log: {arlog2_path}")
except Exception as e:
    logger.warning(f"Failed to write final AutoReject log JSON: {e}")

# Optional quick summary of the final cleaned file path
final_clean_path = eeg_deriv_dir / f"sub-{sub_id}_ses-{ses_id}_run-{run_id}_task-{task_label}_desc-clean_epo.fif"
print(f"\nFinal cleaned epochs saved: {final_clean_path}")
