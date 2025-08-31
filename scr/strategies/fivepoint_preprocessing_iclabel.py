"""
Five-point task preprocessing script with ICLabel-based ICA component selection.


Outputs
- data/processed/sub-<sub>/ses-<ses>/sub-<sub>_ses-<ses>_desc-ica_cleaned.fif
- data/processed/sub-<sub>/ses-<ses>/sub-<sub>_ses-<ses>_task-<task>_run-<run>_preprocessed-epoched.fif (optional)

Usage (examples)
python -m scr.strategies.fivepoint_preprocessing_iclabel \
  --config configs/fivepoint_pipeline.yml --subject 01 --session 001 --run 01 --task 5pt \
  --input-file e:/Yann/neurotheque_resources/data/pilot_data/sub-01_ses-001_task-5pt_run-01_raw.edf

If --input-file is omitted, the script uses the config's directory and file_path_pattern.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import yaml

import mne
import numpy as np

# Repo-local imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scr.steps.load import LoadData
from scr.steps.prepchannels import PrepChannelsStep
from scr.steps.filter import FilterStep
from scr.steps.autoreject import AutoRejectStep
from scr.steps.project_paths import ProjectPaths


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess five-point EEG with ICLabel-based ICA")
    p.add_argument("--config", default=str(ROOT / "configs" / "fivepoint_pipeline.yml"), help="Path to YAML config")
    p.add_argument("--subject", default="01", help="Subject ID (e.g., 01)")
    p.add_argument("--session", default="001", help="Session ID (e.g., 001)")
    p.add_argument("--run", default="01", help="Run ID (e.g., 01)")
    p.add_argument("--task", default="5pt", help="Task label (default 5pt)")
    p.add_argument("--input-file", default=None, help="Explicit path to raw file (.edf/.fif)")
    p.add_argument("--hpf", type=float, default=0.1, help="High-pass cutoff (Hz)")
    p.add_argument("--lpf", type=float, default=40.0, help="Low-pass cutoff (Hz)")
    p.add_argument("--notch", nargs="*", type=float, default=[50.0, 60.0],
                   help="Notch freqs (Hz); prefer a single mains set, e.g., 60 120 or 50 100")
    p.add_argument("--no-epochs", action="store_true", help="Skip saving epoched file; only save ICA-cleaned raw")
    p.add_argument("--iclabel-thresholds", type=str, default="Eye=0.7,Muscle=0.6,Heart=0.6,Line Noise=0.6,Channel Noise=0.6", 
                   help="ICLabel thresholds as key=val pairs (e.g., 'Eye=0.7,Muscle=0.6,Heart=0.6,Line Noise=0.6,Channel Noise=0.6'). Also accepts 4 comma-separated values (legacy order: eye,heart,line_noise,channel_noise).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Log level")
    return p


def _resolve_input_path(cfg: dict, sub: str, ses: str, task: str, run: str, input_file: str | None) -> Path:
    if input_file:
        return Path(input_file)
    root = Path(cfg["directory"]["root"]).resolve()
    raw_dir = root / cfg["directory"]["raw_data_dir"]
    pattern = cfg.get("file_path_pattern")
    if pattern and ("{" in pattern):  # support formatting if provided
        fname = pattern.format(subject=sub, session=ses, task=task, run=run)
    else:
        fname = pattern or f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_raw.edf"
    return raw_dir / fname


def apply_iclabel_exclusion(ica, inst, thresholds):
    """
    Apply ICLabel classification and exclude components based on thresholds.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    inst : mne.io.Raw or mne.Epochs
        Data instance for ICLabel classification (ideally the same domain used for fitting)
    thresholds : dict
        Dict with keys among: 'Eye','Muscle','Heart','Line Noise','Channel Noise' mapping to prob thresholds in [0..1]
        
    Returns
    -------
    list
        Indices of components to exclude
    """
    # Import with compatibility across versions
    try:
        from mne_icalabel import label_components  # type: ignore
    except Exception:
        try:
            from mne_icalabel.label_components import label_components  # type: ignore
        except Exception:
            logging.error("mne-icalabel is not installed. Please install it using: pip install mne-icalabel")
            return []

    # Run ICLabel classification; normalize output to a dict
    res = label_components(inst=inst, ica=ica, method='iclabel')
    if isinstance(res, tuple):
        labels, y_pred_proba = res[0], res[1] if len(res) > 1 else None
        classes = list(res[2]) if len(res) > 2 else []
        ic_info = {"labels": labels, "y_pred_proba": y_pred_proba, "classes": classes}
    elif isinstance(res, dict):
        ic_info = res
        # Some versions use 'classes' or 'labels_set'
        if 'classes' not in ic_info and 'labels_set' in ic_info:
            ic_info['classes'] = list(ic_info['labels_set'])
    else:
        logging.error(f"Unexpected return type from label_components: {type(res)}")
        return []

    labels = list(ic_info.get("labels", []))
    proba = np.asarray(ic_info.get("y_pred_proba", []))
    classes = list(ic_info.get("classes", []))
    if proba.ndim == 1:
        proba = proba[None, :]

    # Index map for classes
    cls_idx = {c: i for i, c in enumerate(classes)}
    artifact_classes = ['Eye', 'Muscle', 'Heart', 'Line Noise', 'Channel Noise']

    exclude = []
    for ic in range(len(labels)):
        for art in artifact_classes:
            if art in cls_idx:
                thr = float(thresholds.get(art, 0.6))
                p = float(proba[ic, cls_idx[art]])
                if p >= thr:
                    exclude.append(ic)
                    break

    exclude = sorted(set(exclude))

    # Log summary
    logging.info(f"ICLabel exclude indices: {exclude}")
    for art in artifact_classes:
        if art in cls_idx:
            idxs = [ic for ic in exclude if float(proba[ic, cls_idx[art]]) >= float(thresholds.get(art, 0.6))]
            if idxs:
                logging.info(f"  {art}: {idxs}")

    return exclude


def _parse_iclabel_thresholds(arg_str: str) -> dict:
    """Parse thresholds from CLI arg supporting key=val pairs or legacy comma list.

    Returns a dict with keys matching ICLabel class names.
    """
    # Defaults
    out = {
        'Eye': 0.7,
        'Muscle': 0.6,
        'Heart': 0.6,
        'Line Noise': 0.6,
        'Channel Noise': 0.6,
    }

    try:
        if '=' in arg_str:
            # key=val pairs separated by commas
            parts = [p.strip() for p in arg_str.split(',') if p.strip()]
            for kv in parts:
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    k = k.strip()
                    v = float(v.strip())
                    # Normalize keys to ICLabel canonical names
                    key_map = {
                        'eye': 'Eye', 'Eye': 'Eye',
                        'muscle': 'Muscle', 'Muscle': 'Muscle',
                        'heart': 'Heart', 'Heart': 'Heart',
                        'line_noise': 'Line Noise', 'Line Noise': 'Line Noise', 'LineNoise': 'Line Noise',
                        'channel_noise': 'Channel Noise', 'Channel Noise': 'Channel Noise', 'ChannelNoise': 'Channel Noise',
                    }
                    out[key_map.get(k, k)] = v
        else:
            # Legacy: four comma-separated numeric values (eye, heart, line_noise, channel_noise)
            vals = [float(x) for x in arg_str.split(',') if x.strip()]
            if len(vals) >= 1:
                out['Eye'] = vals[0]
            if len(vals) >= 2:
                out['Heart'] = vals[1]
            if len(vals) >= 3:
                out['Line Noise'] = vals[2]
            if len(vals) >= 4:
                out['Channel Noise'] = vals[3]
    except Exception as e:
        logging.warning(f"Failed to parse --iclabel-thresholds '{arg_str}': {e}. Using defaults: {out}")
    return out


def main():
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("fivepoint_preprocessing_iclabel")
    mne.set_log_level(args.log_level)

    # Parse ICLabel thresholds (supports key=val pairs and legacy numeric list)
    iclabel_thresholds = _parse_iclabel_thresholds(args.iclabel_thresholds)
    log.info(f"ICLabel thresholds: {iclabel_thresholds}")

    # Load config and paths
    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = ProjectPaths(cfg)
    sub, ses, run, task = args.subject, args.session, args.run, args.task

    raw_path = _resolve_input_path(cfg, sub, ses, task, run, args.input_file)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    log.info(f"Input raw: {raw_path}")
    log.info(f"Outputs base: {paths.processed_dir / f'sub-{sub}' / f'ses-{ses}'}")

    # 1) Load raw
    load_step = LoadData(params={
        "input_file": str(raw_path),
        "stim_channel": "Trigger",
        "subject_id": sub,
        "session_id": ses,
        "paths": paths,
    })
    raw = load_step.run(None)

    # 2) Prepare channels (average reference)
    prep_step = PrepChannelsStep(params={
        "on_missing": "ignore",
        "reference": {"method": "average", "projection": False}
    })
    raw = prep_step.run(raw)

    # 3) Filter (slow-wave friendly)
    filt_step = FilterStep(params={
        "l_freq": args.hpf,
        "h_freq": args.lpf,
        "notch_freqs": args.notch,
    })
    raw = filt_step.run(raw)

    # 4) AutoReject (annotate only)
    ar_plot_dir = paths.get_autoreject_report_dir(f"{sub}", f"{ses}", task_id=f"{task}", run_id=f"{run}")
    ar_step = AutoRejectStep(params={
        "mode": "fit",  # annotate bad epochs; keep raw intact
        "plot_results": True,
        "interactive": False,
        "plot_dir": str(ar_plot_dir),
        "store_reject_log": True,
        "file_prefix": "ar_final_pass",
        "subject_id": sub,
        "session_id": ses,
        "run_id": run,
    })
    raw = ar_step.run(raw)

    # 5) ICA with ICLabel-based exclusion (fit once on 1 Hz HPF copy, apply to original)
    log.info("Preparing data for ICA (1 Hz high-pass copy for fitting)...")
    raw_for_ica = raw.copy()
    try:
        raw_for_ica.filter(l_freq=1.0, h_freq=None, phase='zero-double', fir_design='firwin')
    except Exception as e:
        log.warning(f"High-pass filtering at 1 Hz for ICA fit failed: {e}. Proceeding without extra HPF.")

    from mne.preprocessing import ICA

    ica = ICA(n_components=0.99, method='infomax', max_iter=2000, fit_params={"extended": True, "l_rate": 1e-3}, random_state=0)

    # Create 1s fixed-length epochs to respect BAD_autoreject annotations
    log.info("Creating fixed-length epochs for ICA fitting, excluding BAD annotations if present...")
    events = mne.make_fixed_length_events(raw_for_ica, duration=1)
    epochs_for_ica = mne.Epochs(raw_for_ica, events, tmin=0, tmax=1, baseline=None, preload=True, reject_by_annotation=True)

    log.info("Fitting ICA...")
    ica.fit(epochs_for_ica, decim=3)

    # ICLabel classification on the original-band data
    log.info("Applying ICLabel classification to select artifact components...")
    exclude_indices = apply_iclabel_exclusion(ica, epochs_for_ica, iclabel_thresholds)
    ica.exclude = exclude_indices
    log.info(f"Final ICA exclusion list: {ica.exclude}")

    # Apply to the original filtered raw
    raw_clean = raw.copy()
    try:
        ica.apply(raw_clean)
    except Exception as e:
        log.error(f"Failed to apply ICA to raw data: {e}")
        raw_clean = raw  # Fallback to uncleaned raw

    # Save cleaned raw
    try:
        ica_out = paths.get_derivative_path(sub, ses, task_id=task, run_id=run, stage='desc-ica_cleaned_eeg')
        ica_out.parent.mkdir(parents=True, exist_ok=True)
        raw_clean.save(str(ica_out), overwrite=True)
        log.info(f"Saved ICA-cleaned raw to {ica_out}")
    except Exception as e:
        log.warning(f"Failed to save ICA-cleaned raw: {e}")

    # 6) Optional: epoching disabled here (prefer robust epoching in analysis pipeline)
    skip_epochs = getattr(args, "no_epochs", False)
    if not skip_epochs:
        log.info("Skipping in-script epoching to avoid brittle event pairing. Perform epoching in analysis pipeline using explicit trigger codes from config.")

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
