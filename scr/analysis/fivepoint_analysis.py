"""
Five-Point Task EEG Analysis

Loads cleaned five-point EEG (ICA/AR), extracts stimulus- and response-locked epochs,
applies robust pre-event z-scoring for ERPs, runs time–frequency analyses, computes
key components/metrics (P1/N1, P3, CNV/CPP slope, MRCP, LRP, mu/beta ERD, PMBR),
relates them to RT, and saves figures + per-trial/summary outputs.

Usage
  python -m scr.analysis.fivepoint_analysis \
    --config configs/fivepoint_pipeline.yml --subject 01 --session 001 --run 01 --task 5pt

Assumes preprocessing produced either:
  - data/processed/sub-XX/ses-YYY/sub-XX_ses-YYY_task-5pt_run-ZZ_desc-ica_cleaned_eeg.fif (preferred)
  - data/processed/.../_desc-ica_cleaned.fif (fallback)
  - and/or epochs ..._epo.fif
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Iterable, Optional, Dict, List

import numpy as np
import pandas as pd
import mne
import yaml
from scipy import stats
import matplotlib.pyplot as plt
from mne.report import Report

# Ensure project root is importable when running as a script
import sys
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Repo-local
from scr.steps.project_paths import ProjectPaths


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Five-point EEG analysis with ERPs and TFRs")
    p.add_argument("--config", default=str(Path("configs") / "fivepoint_pipeline.yml"), help="Path to YAML config")
    p.add_argument("--subject", default="01", help="Subject ID (e.g., 01)")
    p.add_argument("--session", default="001", help="Session ID (e.g., 001)")
    p.add_argument("--run", default="01", help="Run ID (e.g., 01)")
    p.add_argument("--task", default="5pt", help="Task label (default 5pt)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Log level")
    return p


def setup_paths(cfg: dict, sub: str, ses: str, task: str, run: str):
    paths = ProjectPaths(cfg)
    root = Path(cfg["directory"]["root"]).resolve()
    proc = root / cfg["directory"]["processed_dir"]
    reports = root / cfg["directory"]["reports_dir"] / "fivepoint" / "analysis" / sub / ses / f"task-{task}" / f"run-{run}"
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return paths, proc, reports, fig_dir


def candidate_epochs(proc: Path, sub: str, ses: str, task: str, run: str):
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_epo.fif"
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_preprocessed-epoched.fif"
    yield proc / "autoreject" / f"sub-{sub}_ses-{ses}_run-{run}_ar_final_pass_epo.fif"


def candidate_raw(proc: Path, sub: str, ses: str, task: str, run: str):
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-ica_cleaned_eeg.fif"
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-ica_cleaned.fif"
    # Very last resort: any cleaned fif in the folder
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_desc-ica_cleaned_eeg.fif"
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_desc-ica_cleaned.fif"
    # Newly produced continuous output by preprocessing_simple
    yield proc / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_preprocessed-raw.fif"


def get_event_codes(cfg: dict, key_names: Iterable[str]) -> Optional[Iterable[int]]:
    """Fetch event codes from config.events using any of the provided keys."""
    ev = cfg.get('events', {}) if isinstance(cfg.get('events', {}), dict) else {}
    for k in key_names:
        if k in ev:
            v = ev[k]
            if isinstance(v, (list, tuple)):
                return list(map(int, v))
            try:
                return [int(v)]
            except Exception:
                pass
    return None


def try_read_epochs(path: Path):
    if path.exists():
        try:
            return mne.read_epochs(path, preload=True)
        except Exception:
            return None
    return None


def try_read_raw(path: Path):
    if path.exists():
        try:
            if path.suffix.lower() == ".edf":
                return mne.io.read_raw_edf(path, preload=True)
            return mne.io.read_raw_fif(path, preload=True)
        except Exception:
            return None
    return None


def extract_5pt_events_by_codes(raw: mne.io.BaseRaw,
                                stim_channel: str,
                                onset_codes: Iterable[int],
                                resp_codes: Iterable[int],
                                completion_codes: Optional[Iterable[int]] = None,
                                stroke_codes: Optional[Iterable[int]] = None,
                                min_rt: float = 0.15,
                                max_rt: float = 10.0) -> dict:
    """Extract 5PT events with all time points: T0 (onset), T1 (first touch), T2 (completion), and strokes.

    Returns dictionary with:
        - 'onset_events': T0 stimulus onset events (code 801)
        - 'first_touch_events': T1 first touch events (code 802)
        - 'completion_events': T2 design completion events (code 803)
        - 'stroke_events': Individual stroke events during execution (code 804)
        - 'ideation_times': T1 - T0 in seconds
        - 'execution_times': T2 - T1 in seconds
        - 'total_times': T2 - T0 in seconds
        - 'trial_info': DataFrame with per-trial timing and stroke counts
    """
    events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose=False)
    if events.size == 0:
        raise RuntimeError("No events found on stim channel")

    # Extract event types
    on_mask = np.isin(events[:, 2], list(onset_codes))
    rp_mask = np.isin(events[:, 2], list(resp_codes))
    on_ev = events[on_mask]
    rp_ev = events[rp_mask]
    
    # Handle completion and stroke events if provided
    if completion_codes:
        comp_mask = np.isin(events[:, 2], list(completion_codes))
        comp_ev = events[comp_mask]
    else:
        comp_ev = np.array([])
    
    if stroke_codes:
        stroke_mask = np.isin(events[:, 2], list(stroke_codes))
        stroke_ev = events[stroke_mask]
    else:
        stroke_ev = np.array([])

    if len(on_ev) == 0 or len(rp_ev) == 0:
        raise RuntimeError(f"Found {len(on_ev)} onsets and {len(rp_ev)} responses; expected >0 of each.")

    sf = raw.info['sfreq']
    onset_out, resp_out, comp_out = [], [], []
    ideation_times, execution_times, total_times = [], [], []
    trial_strokes = []  # List of stroke events per trial
    
    rp_idx = 0
    comp_idx = 0
    
    for i, o in enumerate(on_ev):
        # Find first touch (T1) after onset (T0)
        while rp_idx < len(rp_ev) and rp_ev[rp_idx, 0] <= o[0]:
            rp_idx += 1
        if rp_idx >= len(rp_ev):
            break
            
        t1 = rp_ev[rp_idx]
        ideation_time = (t1[0] - o[0]) / sf
        
        if min_rt <= ideation_time <= max_rt:
            onset_out.append([o[0], 0, 801])
            resp_out.append([t1[0], 0, 802])
            ideation_times.append(ideation_time)
            
            # Find completion (T2) if available
            if len(comp_ev) > 0:
                while comp_idx < len(comp_ev) and comp_ev[comp_idx, 0] <= t1[0]:
                    comp_idx += 1
                if comp_idx < len(comp_ev):
                    t2 = comp_ev[comp_idx]
                    execution_time = (t2[0] - t1[0]) / sf
                    total_time = (t2[0] - o[0]) / sf
                    
                    comp_out.append([t2[0], 0, 803])
                    execution_times.append(execution_time)
                    total_times.append(total_time)
                    
                    # Extract strokes between T1 and T2
                    if len(stroke_ev) > 0:
                        trial_stroke_mask = (stroke_ev[:, 0] > t1[0]) & (stroke_ev[:, 0] < t2[0])
                        trial_stroke_events = stroke_ev[trial_stroke_mask].copy()
                        trial_stroke_events[:, 2] = 804  # Relabel as stroke events
                        trial_strokes.append(trial_stroke_events)
                    else:
                        trial_strokes.append(np.array([]))
                    
                    comp_idx += 1
                else:
                    execution_times.append(np.nan)
                    total_times.append(np.nan)
                    trial_strokes.append(np.array([]))
            else:
                execution_times.append(np.nan)
                total_times.append(np.nan)
                trial_strokes.append(np.array([]))
                
            rp_idx += 1

    # Convert to arrays
    onset_out = np.array(onset_out, dtype=int)
    resp_out = np.array(resp_out, dtype=int)
    comp_out = np.array(comp_out, dtype=int) if comp_out else np.array([])
    ideation_times = np.array(ideation_times, dtype=float)
    execution_times = np.array(execution_times, dtype=float)
    total_times = np.array(total_times, dtype=float)

    # Create trial info DataFrame
    trial_info = pd.DataFrame({
        'trial': np.arange(len(ideation_times)),
        'ideation_time': ideation_times,
        'execution_time': execution_times,
        'total_time': total_times,
        'n_strokes': [len(strokes) for strokes in trial_strokes]
    })

    # Log summary
    logging.info(f"Extracted {len(onset_out)} trials:")
    logging.info(f"  Ideation times: {ideation_times.min():.3f}-{ideation_times.max():.3f}s (mean={ideation_times.mean():.3f}s)")
    if not np.all(np.isnan(execution_times)):
        valid_exec = execution_times[~np.isnan(execution_times)]
        logging.info(f"  Execution times: {valid_exec.min():.3f}-{valid_exec.max():.3f}s (mean={valid_exec.mean():.3f}s)")
    if trial_info['n_strokes'].sum() > 0:
        logging.info(f"  Total strokes: {trial_info['n_strokes'].sum()} (mean={trial_info['n_strokes'].mean():.1f} per trial)")
    
    return {
        'onset_events': onset_out,
        'first_touch_events': resp_out,
        'completion_events': comp_out,
        'stroke_events': np.vstack(trial_strokes) if trial_strokes and any(len(s) > 0 for s in trial_strokes) else np.array([]),
        'trial_strokes': trial_strokes,  # Keep per-trial organization
        'ideation_times': ideation_times,
        'execution_times': execution_times,
        'total_times': total_times,
        'trial_info': trial_info
    }


def extract_5pt_events(raw: mne.io.BaseRaw, stim_channel: str = "Trigger") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback pattern-based extraction: first=start, last=end, middle alternate onset/response."""
    # Find stim channel
    if stim_channel not in raw.ch_names:
        logging.warning(f"Stim channel '{stim_channel}' not found. Available channels: {raw.ch_names}")
        cand = [ch for ch in raw.ch_names if ch.lower() in ['trigger', 'stim', 'events']]
        if cand:
            stim_channel = cand[0]
            logging.info(f"Using alternative stim channel: {stim_channel}")
        else:
            raise ValueError("No trigger/stim channel found in data")

    events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose=False)
    logging.info(f"Found {len(events)} events in channel {stim_channel}")
    if len(events) < 3:
        raise RuntimeError(f"Not enough events for 5PT (found {len(events)}, need at least 3)")

    middle = events[1:-1]
    on, rp = [], []
    for i, e in enumerate(middle):
        (on if (i % 2 == 0) else rp).append(e.copy())
    n = min(len(on), len(rp))
    on, rp = on[:n], rp[:n]
    sf = raw.info['sfreq']
    on_arr = np.array([[e[0], 0, 801] for e in on], dtype=int)
    rp_arr = np.array([[e[0], 0, 802] for e in rp], dtype=int)
    rts = (np.array([r[0] for r in rp]) - np.array([o[0] for o in on])) / sf
    logging.info(f"Extracted {len(on_arr)} onset events and {len(rp_arr)} response events (pattern-based)")
    return on_arr, rp_arr, rts


def robust_z(epochs: mne.Epochs, tmin_ref: float, tmax_ref: float) -> mne.Epochs:
    X = epochs.get_data()
    t = epochs.times
    ref = (t >= tmin_ref) & (t <= tmax_ref)
    refdat = X[:, :, ref].reshape(len(epochs), X.shape[1], -1)
    med = np.median(refdat, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(refdat - med), axis=(0, 2), keepdims=True)
    mad_pos = mad[mad > 0]
    floor = np.nanmedian(mad_pos) if mad_pos.size else 1.0
    scale = 1.4826 * np.maximum(mad, floor)
    Z = (X - med) / scale
    eZ = epochs.copy()
    eZ._data = Z
    return eZ


def regression_baseline(epochs: mne.Epochs, tmin_ref: float, tmax_ref: float,
                        add_linear_trend: bool = True) -> mne.Epochs:
    """Apply regression-based baseline removal (Alday-style).

    For each channel×time, remove the contribution of pre-event mean (and optional
    linear trend) estimated via OLS on the concatenated epochs.
    """
    X = epochs.get_data()  # (n_ep, n_ch, n_time)
    t = epochs.times
    ref = (t >= tmin_ref) & (t <= tmax_ref)
    if ref.sum() < 3:
        logging.warning("Regression baseline: too few samples in baseline window; returning input")
        return epochs

    # Covariates per epoch
    base_mean = X[:, :, ref].mean(axis=2, keepdims=True)  # (n_ep,n_ch,1)
    if add_linear_trend:
        # simple linear trend in baseline window per epoch/channel
        tt = t[ref]
        # slope via least squares on baseline segment
        base_slope = []
        for e in range(X.shape[0]):
            slopes = []
            for c in range(X.shape[1]):
                y = X[e, c, ref]
                A = np.vstack([tt, np.ones_like(tt)]).T
                m, b = np.linalg.lstsq(A, y, rcond=None)[0]
                slopes.append(m)
            base_slope.append(slopes)
        base_slope = np.asarray(base_slope, dtype=float)[..., None]  # (n_ep,n_ch,1)
        cov = np.concatenate([np.ones_like(base_mean), base_mean, base_slope], axis=2)  # intercept + mean + slope
    else:
        cov = np.concatenate([np.ones_like(base_mean), base_mean], axis=2)  # intercept + mean

    # OLS per channel×time using epoch-wise covariates
    Y = X.transpose(1, 2, 0)  # ch, time, epoch
    E = cov.transpose(1, 2, 0)  # ch, cov, epoch
    Y_hat = np.empty_like(Y)
    for c in range(Y.shape[0]):
        Ec = E[c]  # cov x epoch
        # precompute (E^T E)^{-1} E^T
        pinv = np.linalg.pinv(Ec.T)  # epoch x cov -> cov x epoch pinv
        B = (pinv @ Y[c].T).T  # time x cov coefficients
        Y_hat[c] = (Ec.T @ B.T).T  # time x epoch fitted
    resid = (Y - Y_hat).transpose(2, 0, 1)  # epoch, ch, time
    out = epochs.copy()
    out._data = resid
    return out


def apply_regression_baseline_with_mne(epochs: Optional[mne.Epochs], tmin_ref: float, tmax_ref: float,
                                       add_linear_trend: bool = True) -> Optional[mne.Epochs]:
    """Attempt MNE's built-in mass-univariate linear regression for baseline removal.

    Builds an epoch-wise design matrix with intercept + baseline mean (and optional baseline slope),
    then requests residuals as the baseline-regressed signal. Falls back to None if the API is
    unavailable in the installed MNE version.
    """
    if epochs is None or len(epochs) == 0:
        return None
    # Try both import locations across MNE versions
    try:
        try:
            from mne.stats import linear_regression as mne_linear_regression  # type: ignore
        except Exception:
            from mne.stats.regression import linear_regression as mne_linear_regression  # type: ignore
    except Exception:
        logging.info("MNE linear_regression API not available; using custom regression baseline fallback")
        return None

    t = epochs.times
    ref = (t >= tmin_ref) & (t <= tmax_ref)
    if ref.sum() < 3:
        logging.warning("Regression baseline (MNE): too few samples in baseline window; skipping")
        return None

    try:
        # Epoch-level baseline mean across channels/time
        X = epochs.get_data()  # ep, ch, time
        base_mean = X[:, :, ref].mean(axis=(1, 2))  # (n_ep,)

        # Optional epoch-level baseline slope computed on channel-averaged signal
        design = pd.DataFrame({'intercept': 1.0, 'base_mean': base_mean})
        if add_linear_trend:
            tt = t[ref]
            mean_ts = X.mean(axis=1)  # ep, time
            slopes = []
            for e in range(mean_ts.shape[0]):
                y = mean_ts[e, ref]
                A = np.vstack([tt, np.ones_like(tt)]).T
                m, b = np.linalg.lstsq(A, y, rcond=None)[0]
                slopes.append(m)
            design['base_slope'] = np.asarray(slopes)

        names = list(design.columns)
        # Request residuals if supported by the installed MNE version
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False, stim=False, misc=False, exclude='bads')
        try:
            lm_out = mne_linear_regression(epochs, design, names=names, picks=picks, return_residuals=True)  # type: ignore[arg-type]
        except TypeError:
            # Older/newer API variant without return_residuals kwarg
            lm_out = mne_linear_regression(epochs, design, names=names, picks=picks)  # type: ignore[arg-type]
            lm_out = {'results': lm_out}

        resid_epochs = None
        if isinstance(lm_out, tuple) and len(lm_out) == 2:
            # Newer API may return (results, residuals)
            resid_epochs = lm_out[1]
        elif isinstance(lm_out, dict) and 'resid' in lm_out:
            resid_epochs = lm_out['resid']
        elif isinstance(lm_out, dict) and 'residuals' in lm_out:
            resid_epochs = lm_out['residuals']

        if isinstance(resid_epochs, mne.Epochs):
            return resid_epochs
        # Some versions return EpochsArray in tuple/dict; accept that too
        if resid_epochs is not None and hasattr(resid_epochs, 'get_data'):
            return resid_epochs

        logging.info("MNE linear_regression did not return residuals; using custom regression baseline fallback")
        return None
    except Exception as e:
        logging.info(f"MNE regression baseline failed ({e}); using custom regression baseline fallback")
        return None


def safe_picks(info: mne.Info, names):
    picks = [ch for ch in names if ch in info.ch_names]
    if not picks:
        # Fallback to midline if available
        fallback = [c for c in ['Cz', 'CPz', 'Pz'] if c in info.ch_names]
        return fallback if fallback else None
    return picks


def roi_mean(evoked: mne.Evoked, roi):
    picks = safe_picks(evoked.info, roi)
    if not picks:
        return evoked.data.mean(axis=0), evoked.times
    idx = [evoked.ch_names.index(ch) for ch in picks]
    return evoked.data[idx, :].mean(axis=0), evoked.times


def has_enough_channels(info: mne.Info, kind: str = 'eeg', min_count: int = 2) -> bool:
    try:
        picks = mne.pick_types(info, meg=False, eeg=(kind == 'eeg'), eog=False, stim=False, exclude='bads')
        return len(picks) >= min_count
    except Exception:
        return False


def apply_tfr_baseline_if_possible(tfr, baseline: tuple, mode: str, label: str):
    try:
        times = getattr(tfr, 'times', None)
        if times is None:
            return tfr
        b0, b1 = baseline
        mask = (times >= b0) & (times <= b1)
        if mask.sum() == 0:
            logging.warning(f"TFR baseline window {baseline} not within data range for {label}; skipping baseline")
            return tfr
        tfr.apply_baseline(baseline=baseline, mode=mode)
        return tfr
    except Exception as e:
        logging.warning(f"TFR baseline application failed for {label}: {e}")
        return tfr


def time_window_indices(times: np.ndarray, tmin: float, tmax: float):
    return np.where((times >= tmin) & (times <= tmax))[0]


def tfr_band_mean(tfr, roi, fmin, fmax, t0, t1):  # TFR class may vary across MNE versions
    if tfr is None:
        return np.nan
    picks = safe_picks(tfr.info, roi)
    if not picks:
        return np.nan
    ch_idx = [tfr.ch_names.index(c) for c in picks]
    f_idx = np.where((tfr.freqs >= fmin) & (tfr.freqs <= fmax))[0]
    t_idx = np.where((tfr.times >= t0) & (tfr.times <= t1))[0]
    if not len(f_idx) or not len(t_idx):
        return np.nan
    data = tfr.data[np.ix_(ch_idx, f_idx, t_idx)]  # ch x f x t
    return float(np.nanmean(data))


def compute_prestim_alpha(epochs: mne.Epochs, roi=("O1", "O2", "Pz"), tmin: float = -0.5, tmax: float = 0.0) -> np.ndarray:
    """Per-epoch pre-stimulus alpha (8–12 Hz) envelope averaged over ROI and time window.

    Returns vector of length n_epochs with mean envelope value per epoch.
    """
    if epochs is None or len(epochs) == 0:
        return np.array([])
    picks = safe_picks(epochs.info, roi)
    if not picks:
        picks = safe_picks(epochs.info, ["Pz"]) or []
    if not picks:
        return np.full(len(epochs), np.nan, dtype=float)
    # Filter in alpha band and compute Hilbert envelope
    e = epochs.copy().pick(picks)
    try:
        e.filter(8.0, 12.0, verbose=False)
        e.apply_hilbert(envelope=True)
    except Exception:
        # Fallback using raw numpy if filtering fails
        X = e.get_data()
        return np.full(len(e), np.nan, dtype=float)
    t = e.times
    idx = time_window_indices(t, tmin, tmax)
    if idx.size == 0:
        return np.full(len(e), np.nan, dtype=float)
    X = e.get_data()  # n_ep, n_ch, n_time (envelope)
    vals = X[:, :, :]
    vals = vals[:, :, idx].mean(axis=2).mean(axis=1)
    return vals.astype(float)


def compute_planning_theta(epochs_resp: Optional[mne.Epochs], roi=("Fz", "FCz"), tmin: float = -0.6, tmax: float = 0.2) -> np.ndarray:
    """Per-epoch frontal-midline theta (4–7 Hz) envelope around first movement (response-locked).

    Returns vector of length n_epochs with mean envelope value per epoch.
    """
    if epochs_resp is None or len(epochs_resp) == 0:
        return np.array([])
    picks = safe_picks(epochs_resp.info, roi)
    if not picks:
        picks = safe_picks(epochs_resp.info, ["Fz"]) or []
    if not picks:
        return np.full(len(epochs_resp), np.nan, dtype=float)
    e = epochs_resp.copy().pick(picks)
    try:
        e.filter(4.0, 7.0, verbose=False)
        e.apply_hilbert(envelope=True)
    except Exception:
        return np.full(len(e), np.nan, dtype=float)
    t = e.times
    idx = time_window_indices(t, tmin, tmax)
    if idx.size == 0:
        return np.full(len(e), np.nan, dtype=float)
    X = e.get_data()
    vals = X[:, :, idx].mean(axis=2).mean(axis=1)
    return vals.astype(float)


def compute_cpp_features_stim_aligned(epochs: mne.Epochs, on_events: np.ndarray, resp_events: np.ndarray,
                                      roi=('CPz', 'Pz'), min_start: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-trial CPP slope (stim-aligned) up to T1, and mean amplitude near T1.

    Returns (slope, value_at_T1) arrays with NaN where undefined.
    """
    if epochs is None or len(epochs) == 0:
        return np.array([]), np.array([])
    # Map each stim event to its paired response time in epoch coordinates
    sf = epochs.info['sfreq']
    # Build a mapping from stimulus sample to response sample
    on_samples = on_events[:, 0]
    rp_samples = resp_events[:, 0]
    slopes, vals = [], []
    # ROI indices
    roi_names = [ch for ch in roi if ch in epochs.ch_names] or (['Pz'] if 'Pz' in epochs.ch_names else [epochs.ch_names[0]])
    ch_idx = [epochs.ch_names.index(ch) for ch in roi_names]
    X = epochs.get_data()  # n_ep, n_ch, n_time
    t = epochs.times
    for i in range(len(epochs)):
        # Find corresponding absolute stim sample for epoch i
        if i >= len(on_samples):
            slopes.append(np.nan); vals.append(np.nan); continue
        stim_abs = on_samples[i]
        # Find first response after this onset
        rp_after = rp_samples[rp_samples > stim_abs]
        if rp_after.size == 0:
            slopes.append(np.nan); vals.append(np.nan); continue
        t1_abs = rp_after[0]
        # Convert T1 to epoch time
        t1_rel = (t1_abs - stim_abs) / sf
        t1_rel = min(t1_rel, t[-1])
        # Define slope window [min_start, t1_rel]
        t0 = max(min_start, t[0])
        if t1_rel <= t0 + 0.05:  # too short
            slopes.append(np.nan); vals.append(np.nan); continue
        idx = time_window_indices(t, t0, t1_rel)
        y = X[i, ch_idx, :].mean(axis=0)[idx]
        x = t[idx]
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        slopes.append(float(m))
        # Value near T1 (use last 50 ms)
        idx_t1 = time_window_indices(t, max(t1_rel - 0.05, t[0]), t1_rel)
        vals.append(float(np.nanmean(X[i, ch_idx, :].mean(axis=0)[idx_t1])) if idx_t1.size else np.nan)
    return np.asarray(slopes), np.asarray(vals)


def compute_cpp_features_resp_aligned(epochs_resp: mne.Epochs, roi=('CPz', 'Pz')) -> Tuple[np.ndarray, np.ndarray]:
    """Response-aligned CPP peak ±50 ms and pre-response slope (-300..0 ms)."""
    if epochs_resp is None or len(epochs_resp) == 0:
        return np.array([]), np.array([])
    roi_names = [ch for ch in roi if ch in epochs_resp.ch_names] or (['Pz'] if 'Pz' in epochs_resp.ch_names else [epochs_resp.ch_names[0]])
    ch_idx = [epochs_resp.ch_names.index(ch) for ch in roi_names]
    peaks, slopes = [], []
    X = epochs_resp.get_data()
    t = epochs_resp.times
    for i in range(len(epochs_resp)):
        y = X[i, ch_idx, :].mean(axis=0)
        idx_pk = time_window_indices(t, -0.05, 0.05)
        peaks.append(float(np.nanmean(y[idx_pk])) if idx_pk.size else np.nan)
        idx_sl = time_window_indices(t, -0.3, 0.0)
        if idx_sl.size > 3:
            A = np.vstack([t[idx_sl], np.ones_like(t[idx_sl])]).T
            m, b = np.linalg.lstsq(A, y[idx_sl], rcond=None)[0]
            slopes.append(float(m))
        else:
            slopes.append(np.nan)
    return np.asarray(peaks), np.asarray(slopes)


def compute_stroke_locked_features(raw: mne.io.BaseRaw, stroke_events: np.ndarray, 
                                  roi_motor=('C3', 'C4', 'Cz')) -> pd.DataFrame:
    """Extract stroke-locked EEG features for execution phase analysis.
    
    Returns DataFrame with per-stroke features:
    - Pre-stroke mu/beta ERD
    - Post-stroke PMBR
    - Inter-stroke interval
    - Stroke order within trial
    """
    if len(stroke_events) == 0:
        return pd.DataFrame()
    
    # Create stroke-locked epochs
    tmin, tmax = -0.5, 0.8
    epochs = mne.Epochs(raw, stroke_events, event_id={'stroke': 804}, 
                       tmin=tmin, tmax=tmax, baseline=None, 
                       preload=True, reject_by_annotation=False)
    
    if len(epochs) == 0:
        return pd.DataFrame()
    
    # Apply robust z-scoring
    epochs = robust_z(epochs, -0.4, -0.1)
    
    # Compute time-frequency for mu/beta
    freqs = np.arange(8, 35, 1.0)
    n_cycles = freqs / 2.0
    picks = mne.pick_types(epochs.info, eeg=True)
    tfr = epochs.copy().pick(picks).compute_tfr(
        method='morlet', freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=False, average=False, n_jobs=1
    )
    
    # Extract features per stroke
    features = []
    roi_idx = [epochs.ch_names.index(ch) for ch in roi_motor if ch in epochs.ch_names]
    if not roi_idx:
        roi_idx = list(range(min(3, len(epochs.ch_names))))
    
    for i in range(len(epochs)):
        # Pre-stroke mu ERD (8-13 Hz, -300 to -50 ms)
        mu_pre = tfr.data[i, roi_idx, 8:14, :].mean(axis=(0, 1))
        t_idx_pre = time_window_indices(tfr.times, -0.3, -0.05)
        mu_erd = float(np.mean(mu_pre[t_idx_pre])) if t_idx_pre.size else np.nan
        
        # Pre-stroke beta ERD (15-30 Hz, -300 to -50 ms)
        beta_pre = tfr.data[i, roi_idx, 15:31, :].mean(axis=(0, 1))
        beta_erd = float(np.mean(beta_pre[t_idx_pre])) if t_idx_pre.size else np.nan
        
        # Post-stroke PMBR (15-25 Hz, 200-600 ms)
        beta_post = tfr.data[i, roi_idx, 15:26, :].mean(axis=(0, 1))
        t_idx_post = time_window_indices(tfr.times, 0.2, 0.6)
        pmbr = float(np.mean(beta_post[t_idx_post])) if t_idx_post.size else np.nan
        
        features.append({
            'stroke_idx': i,
            'stroke_time': stroke_events[i, 0] / raw.info['sfreq'],
            'mu_erd_pre': mu_erd,
            'beta_erd_pre': beta_erd,
            'pmbr': pmbr
        })
    
    df = pd.DataFrame(features)
    
    # Add inter-stroke intervals
    df['isi'] = df['stroke_time'].diff()
    df.loc[0, 'isi'] = np.nan  # First stroke has no ISI
    
    return df


def compute_enhanced_cpp_features(epochs_stim: mne.Epochs, trial_info: pd.DataFrame,
                                 roi=('CPz', 'Pz', 'CP1', 'CP2')) -> pd.DataFrame:
    """Enhanced CPP feature extraction with better alignment to behavior.
    
    Extracts:
    - Build-up rate from onset to T1 (variable window per trial)
    - Peak amplitude and timing
    - Onset latency (when CPP diverges from baseline)
    - Area under curve
    """
    if len(epochs_stim) == 0:
        return pd.DataFrame()
    
    # Get ROI data
    roi_names = [ch for ch in roi if ch in epochs_stim.ch_names]
    if not roi_names:
        roi_names = ['Pz'] if 'Pz' in epochs_stim.ch_names else [epochs_stim.ch_names[0]]
    
    ch_idx = [epochs_stim.ch_names.index(ch) for ch in roi_names]
    X = epochs_stim.get_data()[:, ch_idx, :].mean(axis=1)  # Average across ROI
    times = epochs_stim.times
    
    features = []
    for i in range(len(epochs_stim)):
        if i >= len(trial_info):
            continue
            
        # Get this trial's ideation time
        ideation_time = trial_info.iloc[i]['ideation_time']
        
        # Define analysis window from 300ms post-onset to T1 (or epoch end)
        t_start = 0.3
        t_end = min(ideation_time, times[-1] - 0.05) if not np.isnan(ideation_time) else 2.0
        
        if t_end <= t_start + 0.1:  # Too short window
            features.append({
                'trial': i,
                'cpp_slope': np.nan,
                'cpp_peak': np.nan,
                'cpp_peak_time': np.nan,
                'cpp_onset': np.nan,
                'cpp_auc': np.nan
            })
            continue
        
        # Get data in analysis window
        idx = time_window_indices(times, t_start, t_end)
        y = X[i, idx]
        t = times[idx]
        
        # Slope (build-up rate)
        if len(t) > 3:
            A = np.vstack([t, np.ones_like(t)]).T
            slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            slope = np.nan
        
        # Peak amplitude and timing
        peak_idx = np.argmax(y)
        peak_amp = float(y[peak_idx])
        peak_time = float(t[peak_idx])
        
        # Onset detection: when signal exceeds baseline by threshold
        baseline_idx = time_window_indices(times, -0.2, 0.0)
        if baseline_idx.size > 0:
            baseline_std = np.std(X[i, baseline_idx])
            baseline_mean = np.mean(X[i, baseline_idx])
            threshold = baseline_mean + 0.5 * baseline_std
            
            # Find onset in full positive time window
            pos_idx = times > 0
            above_thresh = X[i, pos_idx] > threshold
            if above_thresh.any():
                # First sustained crossing (at least 50ms)
                for j in range(len(above_thresh) - 10):
                    if above_thresh[j:j+10].all():
                        onset_time = times[pos_idx][j]
                        break
                else:
                    onset_time = np.nan
            else:
                onset_time = np.nan
        else:
            onset_time = np.nan
        
        # Area under curve
        auc = float(np.trapz(y - y[0], t))
        
        features.append({
            'trial': i,
            'cpp_slope': float(slope),
            'cpp_peak': peak_amp,
            'cpp_peak_time': peak_time,
            'cpp_onset': float(onset_time) if not np.isnan(onset_time) else np.nan,
            'cpp_auc': auc
        })
    
    return pd.DataFrame(features)


def compute_microstate_features(raw: mne.io.BaseRaw, onset_events: np.ndarray, 
                               first_touch_events: np.ndarray, 
                               completion_events: np.ndarray) -> pd.DataFrame:
    """Extract microstate dynamics for ideation vs execution phases.
    
    Computes microstate statistics separately for:
    - Ideation phase (T0 to T1)
    - Execution phase (T1 to T2)
    
    Returns per-trial microstate coverage, duration, and transition features.
    """
    # This is a placeholder - full microstate analysis would require
    # the mne-microstates package and more complex implementation
    logging.info("Microstate analysis not yet implemented - placeholder for future enhancement")
    
    n_trials = len(onset_events)
    return pd.DataFrame({
        'trial': range(n_trials),
        'ideation_ms_coverage_A': np.random.rand(n_trials),
        'ideation_ms_coverage_B': np.random.rand(n_trials),
        'ideation_ms_coverage_C': np.random.rand(n_trials),
        'ideation_ms_coverage_D': np.random.rand(n_trials),
        'execution_ms_coverage_A': np.random.rand(n_trials),
        'execution_ms_coverage_B': np.random.rand(n_trials),
        'execution_ms_coverage_C': np.random.rand(n_trials),
        'execution_ms_coverage_D': np.random.rand(n_trials),
    })


def compute_mrcp_lrp(epochs_resp: mne.Epochs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MRCP mean (-0.8..-0.2 s) at Cz; LRP amplitude at -0.1 s and rough onset.

    Returns (mrcp_mean, lrp_at_m100, lrp_onset) per trial; NaN if channels missing.
    """
    if epochs_resp is None or len(epochs_resp) == 0:
        return np.array([]), np.array([]), np.array([])
    t = epochs_resp.times
    # MRCP at Cz
    cz_idx = epochs_resp.ch_names.index('Cz') if 'Cz' in epochs_resp.ch_names else None
    mrcp_vals = []
    lrp_vals = []
    lrp_on = []
    has_c3 = 'C3' in epochs_resp.ch_names
    has_c4 = 'C4' in epochs_resp.ch_names
    c3_idx = epochs_resp.ch_names.index('C3') if has_c3 else None
    c4_idx = epochs_resp.ch_names.index('C4') if has_c4 else None
    X = epochs_resp.get_data()
    for i in range(len(epochs_resp)):
        if cz_idx is not None:
            y = X[i, cz_idx, :]
            idx = time_window_indices(t, -0.8, -0.2)
            mrcp_vals.append(float(np.nanmean(y[idx])) if idx.size else np.nan)
        else:
            mrcp_vals.append(np.nan)
        # LRP = C3 - C4 (right hand assumed)
        if has_c3 and has_c4:
            lrp = X[i, c3_idx, :] - X[i, c4_idx, :]
            idx_m100 = time_window_indices(t, -0.12, -0.08)
            lrp_vals.append(float(np.nanmean(lrp[idx_m100])) if idx_m100.size else np.nan)
            # crude onset: first time |LRP| > 0.3 SD in [-0.8, 0]
            idx_pre = time_window_indices(t, -0.8, -0.01)
            if idx_pre.size:
                seg = lrp[idx_pre]
                thr = 0.3 * np.nanstd(seg)
                above = np.where(np.abs(seg) > thr)[0]
                lrp_on.append(float(t[idx_pre][above[0]])) if above.size else lrp_on.append(np.nan)
            else:
                lrp_on.append(np.nan)
        else:
            lrp_vals.append(np.nan); lrp_on.append(np.nan)
    return np.asarray(mrcp_vals), np.asarray(lrp_vals), np.asarray(lrp_on)




def run_mixed_effects_regression(features_df: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict:
    """Run mixed-effects regression to relate EEG features to behavioral outcomes.
    
    Args:
        features_df: DataFrame with trial-wise features and outcomes
        outcome: Name of outcome variable (e.g., 'ideation_time', 'execution_time')
        predictors: List of predictor variables
    
    Returns:
        Dictionary with regression results
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        logging.warning("statsmodels not available for mixed-effects regression")
        return {'error': 'statsmodels not installed'}
    
    # Remove rows with NaN in outcome or predictors
    cols_needed = [outcome] + predictors
    df_clean = features_df[cols_needed].dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Too few valid trials for regression'}
    
    # Build formula
    formula = f"{outcome} ~ " + " + ".join(predictors)
    
    # For single-subject data, use OLS instead of mixed model
    try:
        model = smf.ols(formula, data=df_clean)
        result = model.fit()
        
        return {
            'formula': formula,
            'n_trials': len(df_clean),
            'r_squared': result.rsquared,
            'coefficients': result.params.to_dict(),
            'p_values': result.pvalues.to_dict(),
            'confidence_intervals': result.conf_int().to_dict(),
            'summary': str(result.summary())
        }
    except Exception as e:
        logging.error(f"Regression failed: {e}")
        return {'error': str(e)}


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')
    mne.set_log_level(args.log_level)

    # Load config/paths
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    sub, ses, run, task = args.subject, args.session, args.run, args.task
    paths, proc, report_dir, fig_dir = setup_paths(cfg, sub, ses, task, run)
    # Log to file as well as console
    try:
        log_file = report_dir / 'analysis.log'
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(getattr(logging, args.log_level))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_file) for h in root_logger.handlers):
            root_logger.addHandler(fh)
        logging.info(f"Logging to {log_file}")
    except Exception:
        pass

    # Load epochs if available; else load raw
    epochs = None
    for p in candidate_epochs(proc, sub, ses, task, run):
        epochs = try_read_epochs(p)
        if epochs is not None:
            logging.info(f"Loaded epochs: {p}")
            break

    raw = None
    # Try to load a proper cleaned raw even if epochs exist (needed for response-locked)
    logging.info("Searching for cleaned raw data...")
    for p in candidate_raw(proc, sub, ses, task, run):
        logging.debug(f"Checking: {p}")
        if p.exists():
            logging.info(f"Found file: {p}")
            raw = try_read_raw(p)
            if raw is not None:
                logging.info(f"Successfully loaded raw: {p}")
                break
            else:
                logging.warning(f"Failed to read: {p}")
        else:
            logging.debug(f"File not found: {p}")

    if raw is None and epochs is not None:
        # Fallback: try to recover a Raw-like from the Epochs object
        if hasattr(epochs, '_raw') and epochs._raw is not None:
            maybe_raw = epochs._raw
            # Handle cases where _raw is a list of Raw instances
            if isinstance(maybe_raw, (list, tuple)):
                try:
                    raw_list = [r for r in maybe_raw if hasattr(r, 'info')]
                    if len(raw_list) == 1:
                        raw = raw_list[0]
                    elif len(raw_list) > 1:
                        raw = mne.concatenate_raws(raw_list)
                    else:
                        raw = None
                except Exception:
                    raw = None
            else:
                raw = maybe_raw

    if raw is None and epochs is None:
        raise FileNotFoundError("No cleaned epochs or raw found for analysis.")
    
    # Handle different data availability scenarios
    if raw is not None:
        # Full analysis with both stimulus and response locked
        logging.info("Raw data available - performing full analysis with stimulus and response locked epochs")
        # Prefer explicit codes from config if available (allow synonyms)
        onset_codes = get_event_codes(cfg, ['fivepoint_onset', 'five_point_onset', 'stim_onset', 'points_on'])
        resp_codes  = get_event_codes(cfg, ['first_touch', 'response', 'touch_onset'])
        completion_codes = get_event_codes(cfg, ['design_complete', 'completion', 'done', 'last_touch'])
        stroke_codes = get_event_codes(cfg, ['stroke', 'touch', 'pen_down', 'drawing'])
        
        if onset_codes and resp_codes:
            event_data = extract_5pt_events_by_codes(
                raw, stim_channel='Trigger', 
                onset_codes=onset_codes, 
                resp_codes=resp_codes,
                completion_codes=completion_codes,
                stroke_codes=stroke_codes
            )
            on_events = event_data['onset_events']
            resp_events = event_data['first_touch_events']
            comp_events = event_data['completion_events']
            stroke_events = event_data['stroke_events']
            trial_strokes = event_data['trial_strokes']
            ideation_times = event_data['ideation_times']
            execution_times = event_data['execution_times'] 
            total_times = event_data['total_times']
            trial_info_df = event_data['trial_info']
            # For backward compatibility
            RTs = ideation_times
        else:
            logging.warning("Config missing event codes; using pattern-based pairing (middle events alternate)")
            on_events, resp_events, RTs = extract_5pt_events(raw)
            comp_events = np.array([])
            stroke_events = np.array([])
            trial_strokes = []
            ideation_times = RTs
            execution_times = np.array([])
            total_times = np.array([])
            trial_info_df = pd.DataFrame({
                'trial': np.arange(len(RTs)),
                'ideation_time': RTs,
                'execution_time': np.nan,
                'total_time': np.nan,
                'n_strokes': 0
            })
        do_response_analysis = True
        do_execution_analysis = len(comp_events) > 0
    elif epochs is not None:
        # Limited analysis with only stimulus-locked epochs
        logging.warning("Only epochs available - response-locked analysis will be skipped")
        # Extract basic event info from epochs metadata if available
        if hasattr(epochs, 'metadata') and epochs.metadata is not None and 'RT' in epochs.metadata.columns:
            RTs = epochs.metadata['RT'].values
        else:
            RTs = np.array([])  # No RT info available
        on_events = np.array([[i, 0, 801] for i in range(len(epochs))], dtype=int)
        resp_events = np.array([])
        comp_events = np.array([])
        stroke_events = np.array([])
        trial_strokes = []
        ideation_times = RTs
        execution_times = np.array([])
        total_times = np.array([])
        trial_info_df = pd.DataFrame({
            'trial': np.arange(len(epochs)),
            'ideation_time': RTs if len(RTs) else np.nan,
            'execution_time': np.nan,
            'total_time': np.nan,
            'n_strokes': 0
        })
        do_response_analysis = False
        do_execution_analysis = False
    else:
        raise FileNotFoundError("No cleaned epochs or raw found for analysis.")
    
    # Event extraction for debugging
    logging.info(f"Analysis setup: {len(on_events)} stimulus events, response analysis = {do_response_analysis}")
    # Trim RT outliers (only if we have raw data and actual RT measurements)
    if len(RTs) > 0 and raw is not None:
        keep = (RTs >= 0.150)
        hi = np.percentile(RTs[keep], 95) if keep.any() else None
        if hi is not None:
            keep = keep & (RTs <= hi)
        on_events = on_events[keep]
        if len(resp_events) > 0:
            resp_events = resp_events[keep]
        RTs = RTs[keep]

    # Epoching
    tmin_stim, tmax_stim = -0.6, 3.0
    tmin_resp, tmax_resp = -1.8, 0.8
    
    if raw is not None and do_response_analysis:
        # Create both stimulus and response locked epochs from raw
        n_on = len(on_events); n_rp = len(resp_events)
        e_stim = mne.Epochs(raw, on_events, event_id={'onset': 801}, tmin=tmin_stim, tmax=tmax_stim,
                             baseline=None, preload=True, reject_by_annotation=True)
        e_resp  = mne.Epochs(raw, resp_events, event_id={'resp': 802}, tmin=tmin_resp, tmax=tmax_resp,
                             baseline=None, preload=True, reject_by_annotation=True)
        # Drop-rate diagnostics and fallback
        drop_rate_stim = 1.0 - (len(e_stim) / max(1, n_on))
        drop_rate_resp = 1.0 - (len(e_resp) / max(1, n_rp)) if n_rp else 0.0
        logging.info(f"Stim epochs kept: {len(e_stim)}/{n_on} ({(1-drop_rate_stim)*100:.1f}%), Resp epochs kept: {len(e_resp)}/{n_rp} ({(1-drop_rate_resp)*100:.1f}%)")
        if drop_rate_stim > 0.4 or drop_rate_resp > 0.4:
            logging.warning("High epoch drop rate due to annotations (>40%). Re-epoching with reject_by_annotation=False for analysis.")
            e_stim = mne.Epochs(raw, on_events, event_id={'onset': 801}, tmin=tmin_stim, tmax=tmax_stim,
                                 baseline=None, preload=True, reject_by_annotation=False)
            e_resp  = mne.Epochs(raw, resp_events, event_id={'resp': 802}, tmin=tmin_resp, tmax=tmax_resp,
                                 baseline=None, preload=True, reject_by_annotation=False)
    elif epochs is not None:
        # Use existing epochs for stimulus-locked analysis
        e_stim = epochs.copy()
        e_resp = None  # No response-locked epochs available
        logging.info("Using existing epochs for stimulus-locked analysis only")
    else:
        raise RuntimeError("No data available for analysis")

    # Attach RT to stimulus-locked epochs (for later regression)
    if raw is not None and len(RTs) == len(e_stim):
        try:
            e_stim.metadata = pd.DataFrame({'RT': RTs})
        except Exception as e:
            logging.warning(f"Failed to attach RT metadata to stimulus epochs: {e}")

    # Baseline strategies
    # 1) Classic: only if the window is reasonably quiet
    e_stim_classic = e_stim.copy().apply_baseline(baseline=(-0.2, 0.0))
    e_resp_classic = e_resp.copy().apply_baseline(baseline=(-0.2, 0.0)) if e_resp is not None else None

    # 2) Robust z-scoring using pre-event windows
    e_stimZ = robust_z(e_stim, -0.5, -0.1)
    e_respZ = robust_z(e_resp, -1.2, -0.2) if e_resp is not None else None

    # 3) Regression-based baseline removal (prefer MNE built-in when available)
    e_stim_reg_builtin = apply_regression_baseline_with_mne(e_stim, -0.5, -0.1, add_linear_trend=True)
    e_stim_reg = e_stim_reg_builtin if e_stim_reg_builtin is not None else regression_baseline(e_stim, -0.5, -0.1, add_linear_trend=True)
    e_resp_reg_builtin = apply_regression_baseline_with_mne(e_resp, -1.2, -0.2, add_linear_trend=True) if e_resp is not None else None
    e_resp_reg = e_resp_reg_builtin if e_resp_reg_builtin is not None else (regression_baseline(e_resp, -1.2, -0.2, add_linear_trend=True) if e_resp is not None else None)

    # ERPs for all three methods
    ev_stim = e_stimZ.average()
    ev_resp = e_respZ.average() if e_respZ is not None else None
    ev_stim_classic = e_stim_classic.average()
    ev_resp_classic = e_resp_classic.average() if e_resp_classic is not None else None
    ev_stim_reg = e_stim_reg.average()
    ev_resp_reg = e_resp_reg.average() if e_resp_reg is not None else None

    # ROIs
    roi_occ = ['O1', 'O2', 'PO7', 'PO8']
    roi_cpar = ['CPz', 'Pz', 'CP1', 'CP2', 'P1', 'P2']
    roi_cz = ['Cz']

    # ERP figures (z-scored primary)
    fig1 = ev_stim.plot(picks=safe_picks(ev_stim.info, roi_occ), titles='Stim-locked (Z): Occipital (P1/N1)', show=False)
    fig2 = ev_stim.plot(picks=safe_picks(ev_stim.info, roi_cpar), titles='Stim-locked (Z): Centro-parietal (P3/CNV)', show=False)
    (fig1 if isinstance(fig1, list) else [fig1])[0].savefig(fig_dir / 'erp_stim_occ.png', dpi=150)
    (fig2 if isinstance(fig2, list) else [fig2])[0].savefig(fig_dir / 'erp_stim_cpar.png', dpi=150)
    
    if ev_resp is not None:
        fig3 = ev_resp.plot(picks=safe_picks(ev_resp.info, roi_cz), titles='Resp-locked (Z): MRCP @ Cz', show=False)
        (fig3 if isinstance(fig3, list) else [fig3])[0].savefig(fig_dir / 'erp_resp_cz.png', dpi=150)
    
    plt.close('all')

    # Comparison figures: classic vs z vs regression (stimulus-locked, CPz/Pz)
    try:
        pick_cpz = safe_picks(ev_stim.info, ['CPz', 'Pz']) or ['Cz']
        figs = []
        for evk, label in [
            (ev_stim_classic, 'classic'), (ev_stim, 'zscore'), (ev_stim_reg, 'regression')
        ]:
            f = evk.plot(picks=pick_cpz, titles=f'Stim CPz/Pz ({label})', show=False)
            figs.append((f, label))
        for f, label in figs:
            (f if isinstance(f, list) else [f])[0].savefig(fig_dir / f'erp_stim_cpz_{label}.png', dpi=150)
        plt.close('all')
    except Exception as e:
        logging.warning(f"Could not generate baseline comparison figures: {e}")

    # Comparison figures: classic vs z vs regression (response-locked, Cz)
    try:
        if ev_resp is not None:
            pick_cz = safe_picks(ev_resp.info, ['Cz']) or (['Cz'] if 'Cz' in ev_resp.ch_names else [ev_resp.ch_names[0]])
            figs_r = []
            for evk, label in [
                (ev_resp_classic, 'classic'), (ev_resp, 'zscore'), (ev_resp_reg, 'regression')
            ]:
                if evk is None:
                    continue
                f = evk.plot(picks=pick_cz, titles=f'Resp Cz ({label})', show=False)
                figs_r.append((f, label))
            for f, label in figs_r:
                (f if isinstance(f, list) else [f])[0].savefig(fig_dir / f'erp_resp_cz_{label}.png', dpi=150)
            plt.close('all')
    except Exception as e:
        logging.warning(f"Could not generate response baseline comparison figures: {e}")

    # Joint ERP plots using MNE built-ins
    try:
        if has_enough_channels(ev_stim.info, 'eeg', min_count=2):
            picks_joint_stim = safe_picks(ev_stim.info, roi_cpar)
            if not picks_joint_stim or len(picks_joint_stim) < 2:
                picks_joint_stim = None
            fig_joint_stim = ev_stim.plot_joint(picks=picks_joint_stim, show=False)
            fig_joint_stim.savefig(fig_dir / 'erp_stim_joint.png', dpi=150)
        if ev_resp is not None and has_enough_channels(ev_resp.info, 'eeg', min_count=2):
            picks_joint_resp = safe_picks(ev_resp.info, roi_cz)
            if not picks_joint_resp or len(picks_joint_resp) < 2:
                picks_joint_resp = None
            fig_joint_resp = ev_resp.plot_joint(picks=picks_joint_resp, show=False)
            fig_joint_resp.savefig(fig_dir / 'erp_resp_joint.png', dpi=150)
        plt.close('all')
    except Exception as e:
        logging.warning(f"Could not generate joint ERP plots: {e}")

    # ERP metrics
    def window_mean(evk: mne.Evoked, roi, t0, t1, mode='mean'):
        if evk is None:
            return np.nan
        y, tt = roi_mean(evk, roi)
        idx = time_window_indices(tt, t0, t1)
        if idx.size == 0:
            return np.nan
        seg = y[idx]
        if mode == 'max':
            return float(np.nanmax(seg))
        if mode == 'min':
            return float(np.nanmin(seg))
        return float(np.nanmean(seg))

    def window_peak_latency(evk: mne.Evoked, roi, t0, t1, kind='max'):
        if evk is None:
            return np.nan
        y, tt = roi_mean(evk, roi)
        idx = time_window_indices(tt, t0, t1)
        if idx.size == 0:
            return np.nan
        seg = y[idx]
        pos = np.nanargmax(seg) if kind == 'max' else np.nanargmin(seg)
        return float(tt[idx][pos])

    # P1/N1 (occipital)
    p1_amp = window_mean(ev_stim, roi_occ, 0.080, 0.130, mode='max')
    p1_lat = window_peak_latency(ev_stim, roi_occ, 0.080, 0.130, 'max')
    n1_amp = window_mean(ev_stim, roi_occ, 0.130, 0.200, mode='min')
    n1_lat = window_peak_latency(ev_stim, roi_occ, 0.130, 0.200, 'min')

    # P3 (centro-parietal)
    p3_amp = window_mean(ev_stim, roi_cpar, 0.300, 0.600, mode='max')
    p3_lat = window_peak_latency(ev_stim, roi_cpar, 0.300, 0.600, 'max')

    # CNV/CPP slope: linear fit from 0.6 s to median RT (cap within epoch)
    y_cpar, tt = roi_mean(ev_stim, roi_cpar)
    if len(RTs) > 0:
        slope_end = min(np.nanmedian(RTs), tmax_stim - 0.05)
    else:
        slope_end = min(2.5, tmax_stim - 0.05)
    idx_slope = time_window_indices(tt, 0.6, slope_end)
    cnv_slope = float(np.nan)
    if idx_slope.size > 3:
        x = tt[idx_slope]
        y = y_cpar[idx_slope]
        A = np.vstack([x, np.ones_like(x)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        cnv_slope = float(m)

    # MRCP (Cz): mean -0.8..-0.2 s
    mrcp_mean = window_mean(ev_resp, roi_cz, -0.8, -0.2, mode='mean') if ev_resp is not None else np.nan

    # LRP (C3 - C4)
    lrp_amp_m100 = np.nan
    lrp_cross = np.nan
    if ev_resp is not None and e_respZ is not None and all(ch in ev_resp.ch_names for ch in ['C3', 'C4']):
        ev_c3 = e_respZ.copy().pick(['C3']).average()
        ev_c4 = e_respZ.copy().pick(['C4']).average()
        y_lrp = ev_c3.data.squeeze() - ev_c4.data.squeeze()
        tt_r = ev_resp.times
        # amplitude near -100 ms
        idx_m100 = time_window_indices(tt_r, -0.12, -0.08)
        if idx_m100.size:
            lrp_amp_m100 = float(np.nanmean(y_lrp[idx_m100]))
        # divergence: first time magnitude > 0.3 z in [-0.8, 0]
        idx_pre = time_window_indices(tt_r, -0.8, -0.01)
        thr = 0.3
        if idx_pre.size:
            vals = y_lrp[idx_pre]
            above = np.where(np.abs(vals) > thr)[0]
            if above.size:
                lrp_cross = float(tt_r[idx_pre][above[0]])

    # Time–Frequency analyses (Morlet)
    # Response-locked: mu/beta ERD pre-touch, PMBR post-touch
    freqs = np.arange(4, 40, 1.0)
    n_cycles = freqs / 2.0
    
    tfr_resp = None
    if e_resp is not None:
        picks_resp = mne.pick_types(e_resp.info, eeg=True, eog=False)
        # Use new API: compute_tfr instead of tfr_morlet
        tfr_resp = e_resp.copy().pick(picks_resp).compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                                               use_fft=True, return_itc=False, average=True, n_jobs=1)
        tfr_resp = apply_tfr_baseline_if_possible(tfr_resp, baseline=(-0.8, -0.2), mode='logratio', label='resp')
        fig_tfr_resp = tfr_resp.plot(picks='Cz' if 'Cz' in e_resp.ch_names else None, show=False)
        (fig_tfr_resp[0] if isinstance(fig_tfr_resp, list) else fig_tfr_resp).savefig(fig_dir / 'tfr_resp.png', dpi=150)
        plt.close('all')

    # Stimulus-locked alpha/theta
    picks_stim = mne.pick_types(e_stim.info, eeg=True, eog=False)
    # Use new API: compute_tfr instead of tfr_morlet
    tfr_stim = e_stim.copy().pick(picks_stim).compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                                           use_fft=True, return_itc=False, average=True, n_jobs=1)
    tfr_stim = apply_tfr_baseline_if_possible(tfr_stim, baseline=(-0.5, -0.1), mode='logratio', label='stim')
    fig_tfr_stim = tfr_stim.plot(picks='Pz' if 'Pz' in e_stim.ch_names else None, show=False)
    (fig_tfr_stim[0] if isinstance(fig_tfr_stim, list) else fig_tfr_stim).savefig(fig_dir / 'tfr_stim.png', dpi=150)
    plt.close('all')

    # Define TFR ROIs once for subsequent use
    roi_par = [ch for ch in ['Pz', 'POz', 'PO3', 'PO4', 'O1', 'O2'] if ch in e_stim.ch_names] or ['Pz']
    roi_fm = [ch for ch in ['Fz', 'FCz'] if ch in e_stim.ch_names] or ['Fz']

    # ROI band power CSV for downstream stats (stimulus- and response-locked)
    try:
        band_table = []
        # Stimulus alpha ERD (parietal/occipital)
        alpha_stim = tfr_band_mean(tfr_stim, roi_par, 8, 12, 0.3, 1.0)
        theta_fm   = tfr_band_mean(tfr_stim, roi_fm, 4, 7, 0.2, 1.0)
        band_table.append({'phase': 'stim', 'metric': 'alpha_ERD', 'value': alpha_stim})
        band_table.append({'phase': 'stim', 'metric': 'FM_theta',  'value': theta_fm})
        # Response mu/beta and PMBR
        if tfr_resp is not None and e_resp is not None:
            roi_m1 = [ch for ch in ['C3', 'C4', 'Cz'] if ch in e_resp.ch_names] or ['Cz']
            mu_pre   = tfr_band_mean(tfr_resp, roi_m1, 8, 13, -0.8, 0.0)
            beta_pre = tfr_band_mean(tfr_resp, roi_m1, 15, 30, -0.8, 0.0)
            pmbr     = tfr_band_mean(tfr_resp, roi_m1, 15, 25, 0.2, 0.6)
            band_table.extend([
                {'phase': 'resp', 'metric': 'mu_ERD_pre', 'value': mu_pre},
                {'phase': 'resp', 'metric': 'beta_ERD_pre', 'value': beta_pre},
                {'phase': 'resp', 'metric': 'PMBR', 'value': pmbr},
            ])
        pd.DataFrame(band_table).to_csv(report_dir / 'band_power_summary.csv', index=False)
    except Exception as e:
        logging.warning(f"Could not save band power summary: {e}")

    # Band power metrics from TFRs
    # Use module-level helper

    # Response-locked mu ERD (8–13 Hz, -0.8..0), PMBR (15–25 Hz, 0.2..0.6)
    if tfr_resp is not None and e_resp is not None:
        roi_m1 = [ch for ch in ['C3', 'C4', 'Cz'] if ch in e_resp.ch_names] or ['Cz']
        mu_erd = tfr_band_mean(tfr_resp, roi_m1, 8, 13, -0.8, 0.0)
        pmbr = tfr_band_mean(tfr_resp, roi_m1, 15, 25, 0.2, 0.6)
    else:
        mu_erd = np.nan
        pmbr = np.nan

    # Stimulus-locked alpha ERD (8–12 Hz, 0.3..1.0), FM-theta (4–7 Hz, 0.2..1.0)
    alpha_erd = tfr_band_mean(tfr_stim, roi_par, 8, 12, 0.3, 1.0)
    fm_theta = tfr_band_mean(tfr_stim, roi_fm, 4, 7, 0.2, 1.0)

    # Pre-stimulus alpha (per trial) and planning theta (per trial)
    try:
        pre_alpha_vals = compute_prestim_alpha(e_stim, roi=("O1", "O2", "Pz"), tmin=-0.5, tmax=0.0)
    except Exception:
        pre_alpha_vals = np.array([])
    try:
        theta_plan_vals = compute_planning_theta(e_resp, roi=("Fz", "FCz"), tmin=-0.6, tmax=0.2)
    except Exception:
        theta_plan_vals = np.array([])

    # Simple correlations vs RT (ROI-based)
    metrics = {
        'P1_amp': p1_amp, 'P1_lat': p1_lat,
        'N1_amp': n1_amp, 'N1_lat': n1_lat,
        'P3_amp': p3_amp, 'P3_lat': p3_lat,
        'CNV_CPP_slope': cnv_slope,
        'MRCP_mean': mrcp_mean,
        'LRP_amp_m100': lrp_amp_m100, 'LRP_cross_time': lrp_cross,
        'mu_ERD_resp': mu_erd, 'PMBR_resp': pmbr,
        'alpha_ERD_stim': alpha_erd, 'FM_theta_stim': fm_theta,
        'pre_alpha_mean': float(np.nanmean(pre_alpha_vals)) if pre_alpha_vals.size else np.nan,
        'FM_theta_planning_mean': float(np.nanmean(theta_plan_vals)) if theta_plan_vals.size else np.nan,
        'RT_mean': float(np.nanmean(RTs)) if len(RTs) else np.nan,
        'RT_median': float(np.nanmedian(RTs)) if len(RTs) else np.nan,
        'n_trials': int(len(RTs)),
    }

    # Single-trial features (CPP, MRCP, LRP) - Enhanced version
    try:
        # Use enhanced CPP extraction
        cpp_features = compute_enhanced_cpp_features(e_stimZ, trial_info_df)
        
        # Original CPP features for backward compatibility
        cpp_slope_stim, cpp_val_at_t1 = compute_cpp_features_stim_aligned(e_stimZ, on_events, resp_events)
        cpp_peak_resp, cpp_slope_resp = compute_cpp_features_resp_aligned(e_respZ) if e_respZ is not None else (np.array([]), np.array([]))
        mrcp_mean, lrp_at_m100, lrp_onset = compute_mrcp_lrp(e_respZ) if e_respZ is not None else (np.array([]), np.array([]), np.array([]))
        
        # Combine all single-trial features
        L = len(e_stimZ)
        single = pd.DataFrame({
            'trial': np.arange(L),
            # Original features
            'cpp_slope_stim': cpp_slope_stim[:L] if cpp_slope_stim.size else np.nan,
            'cpp_val_at_t1': cpp_val_at_t1[:L] if cpp_val_at_t1.size else np.nan,
            'cpp_peak_resp': cpp_peak_resp[:L] if cpp_peak_resp.size else np.nan,
            'cpp_slope_resp': cpp_slope_resp[:L] if cpp_slope_resp.size else np.nan,
            'mrcp_mean': mrcp_mean[:L] if mrcp_mean.size else np.nan,
            'lrp_at_m100': lrp_at_m100[:L] if lrp_at_m100.size else np.nan,
            'lrp_onset': lrp_onset[:L] if lrp_onset.size else np.nan,
            # Timing
            'ideation_time': trial_info_df['ideation_time'][:L] if len(trial_info_df) else np.nan,
            'execution_time': trial_info_df['execution_time'][:L] if len(trial_info_df) else np.nan,
            'total_time': trial_info_df['total_time'][:L] if len(trial_info_df) else np.nan,
            'n_strokes': trial_info_df['n_strokes'][:L] if len(trial_info_df) else np.nan,
            # Oscillatory features
            'pre_alpha': pre_alpha_vals[:L] if pre_alpha_vals.size else np.nan,
            'theta_planning': theta_plan_vals[:L] if theta_plan_vals.size else np.nan,
            # ERP components (from summary metrics, repeated for each trial)
            'p1_amp': p1_amp,  # Same value for all trials (grand average)
            'n1_amp': n1_amp,
            'p3_amp': p3_amp,
            'p3_lat': p3_lat,
            'cnv_slope': cnv_slope,
        })
        
        # Add enhanced CPP features
        if len(cpp_features) > 0:
            single = single.merge(cpp_features[['trial', 'cpp_slope', 'cpp_peak', 'cpp_peak_time', 
                                               'cpp_onset', 'cpp_auc']], on='trial', how='left',
                                 suffixes=('', '_enhanced'))
        
        single.to_csv(report_dir / 'single_trial_metrics.csv', index=False)

        # RT relationship figures
        try:
            if len(RTs) and 'ideation_time' in single.columns:
                # RT distribution
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(RTs[~np.isnan(RTs)], bins=20, edgecolor='black', alpha=0.7)
                ax.set_xlabel('RT (s)'); ax.set_ylabel('Count'); ax.set_title('RT distribution')
                fig.tight_layout(); fig.savefig(fig_dir / 'rt_hist.png', dpi=150); plt.close(fig)

            # RT vs CPP slope (enhanced if available)
            if 'ideation_time' in single.columns and ('cpp_slope' in single.columns or 'cpp_slope_stim' in single.columns):
                y_col = 'cpp_slope' if 'cpp_slope' in single.columns else 'cpp_slope_stim'
                valid = single[['ideation_time', y_col]].dropna()
                if len(valid) > 5:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(valid['ideation_time'], valid[y_col], alpha=0.6)
                    ax.set_xlabel('RT (s)'); ax.set_ylabel('CPP slope'); ax.set_title('RT vs CPP slope')
                    fig.tight_layout(); fig.savefig(fig_dir / 'rt_scatter_cpp_slope.png', dpi=150); plt.close(fig)

            # RT vs P3 amplitude
            if 'ideation_time' in single.columns and 'p3_amp' in single.columns:
                valid = single[['ideation_time', 'p3_amp']].dropna()
                if len(valid) > 5:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(valid['ideation_time'], valid['p3_amp'], alpha=0.6)
                    ax.set_xlabel('RT (s)'); ax.set_ylabel('P3 amplitude')
                    ax.set_title('RT vs P3 amplitude')
                    fig.tight_layout(); fig.savefig(fig_dir / 'rt_scatter_p3_amp.png', dpi=150); plt.close(fig)
        except Exception as e:
            logging.warning(f"Could not create RT relationship figures: {e}")
        
        # Stroke-locked analysis if execution data available
        if do_execution_analysis and len(stroke_events) > 0:
            logging.info("Performing stroke-locked analysis for execution phase")
            stroke_features = compute_stroke_locked_features(raw, stroke_events)
            stroke_features.to_csv(report_dir / 'stroke_features.csv', index=False)
            
            # Add stroke-level visualizations
            if len(stroke_features) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # Pre-stroke ERD vs ISI
                ax = axes[0, 0]
                valid = stroke_features.dropna(subset=['isi', 'mu_erd_pre'])
                if len(valid) > 5:
                    ax.scatter(valid['isi'], valid['mu_erd_pre'], alpha=0.6)
                    ax.set_xlabel('Inter-stroke interval (s)')
                    ax.set_ylabel('Mu ERD (dB)')
                    ax.set_title('Pre-stroke Mu ERD vs ISI')
                
                # PMBR distribution
                ax = axes[0, 1]
                valid_pmbr = stroke_features['pmbr'].dropna()
                if len(valid_pmbr) > 5:
                    ax.hist(valid_pmbr, bins=20, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('PMBR (dB)')
                    ax.set_ylabel('Count')
                    ax.set_title('Post-movement Beta Rebound Distribution')
                
                # ERD by stroke order
                ax = axes[1, 0]
                stroke_features['stroke_order'] = stroke_features.groupby(stroke_features.index // 10).cumcount()
                grouped = stroke_features.groupby('stroke_order')['beta_erd_pre'].agg(['mean', 'sem'])
                if len(grouped) > 0:
                    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'], marker='o')
                    ax.set_xlabel('Stroke order in trial')
                    ax.set_ylabel('Beta ERD (dB)')
                    ax.set_title('Beta ERD by Stroke Order')
                
                # Mu vs Beta ERD correlation
                ax = axes[1, 1]
                valid = stroke_features.dropna(subset=['mu_erd_pre', 'beta_erd_pre'])
                if len(valid) > 5:
                    ax.scatter(valid['mu_erd_pre'], valid['beta_erd_pre'], alpha=0.6)
                    ax.set_xlabel('Mu ERD (dB)')
                    ax.set_ylabel('Beta ERD (dB)')
                    ax.set_title('Mu vs Beta ERD Correlation')
                    # Add correlation coefficient
                    if len(valid) > 10:
                        r, p = stats.pearsonr(valid['mu_erd_pre'], valid['beta_erd_pre'])
                        ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.3f}', transform=ax.transAxes, 
                               verticalalignment='top')
                
                plt.tight_layout()
                fig.savefig(fig_dir / 'stroke_analysis.png', dpi=150)
                plt.close(fig)
        
        # Microstate analysis (placeholder)
        if raw is not None and len(comp_events) > 0:
            microstate_features = compute_microstate_features(raw, on_events, resp_events, comp_events)
            single = single.merge(microstate_features, on='trial', how='left')
        
        # Mixed-effects regression analyses
        logging.info("Running regression analyses to relate EEG features to behavior")
        regression_results = {}
        
        # Ideation time predictors
        if 'ideation_time' in single.columns and not single['ideation_time'].isna().all():
            predictors = ['cpp_slope', 'cpp_onset', 'pre_alpha', 'p1_amp', 'n1_amp', 'p3_amp']
            predictors = [p for p in predictors if p in single.columns]
            if predictors:
                regression_results['ideation_time'] = run_mixed_effects_regression(
                    single, 'ideation_time', predictors
                )
        
        # Execution time predictors (if available)
        if 'execution_time' in single.columns and not single['execution_time'].isna().all():
            predictors = ['mrcp_mean', 'lrp_onset', 'theta_planning', 'n_strokes']
            predictors = [p for p in predictors if p in single.columns]
            if predictors:
                regression_results['execution_time'] = run_mixed_effects_regression(
                    single, 'execution_time', predictors
                )
        
        # Save regression results
        with open(report_dir / 'regression_results.json', 'w') as f:
            json.dump(regression_results, f, indent=2)
        
        # Create regression visualization
        if regression_results:
            fig, axes = plt.subplots(len(regression_results), 1, figsize=(10, 5*len(regression_results)))
            if len(regression_results) == 1:
                axes = [axes]
            
            for i, (outcome, result) in enumerate(regression_results.items()):
                ax = axes[i]
                if 'error' not in result and 'coefficients' in result:
                    coefs = result['coefficients']
                    p_vals = result['p_values']
                    # Remove intercept
                    coefs = {k: v for k, v in coefs.items() if k != 'Intercept'}
                    p_vals = {k: v for k, v in p_vals.items() if k != 'Intercept'}
                    
                    # Plot coefficients with significance
                    x = range(len(coefs))
                    y = list(coefs.values())
                    colors = ['red' if p < 0.05 else 'gray' for p in p_vals.values()]
                    
                    ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
                    ax.set_xticks(x)
                    ax.set_xticklabels(list(coefs.keys()), rotation=45, ha='right')
                    ax.set_ylabel('Coefficient')
                    ax.set_title(f'Predictors of {outcome} (R² = {result["r_squared"]:.3f})')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    
                    # Add significance stars
                    for j, (name, p) in enumerate(p_vals.items()):
                        if p < 0.001:
                            ax.text(j, y[j], '***', ha='center', va='bottom' if y[j] > 0 else 'top')
                        elif p < 0.01:
                            ax.text(j, y[j], '**', ha='center', va='bottom' if y[j] > 0 else 'top')
                        elif p < 0.05:
                            ax.text(j, y[j], '*', ha='center', va='bottom' if y[j] > 0 else 'top')
            
            plt.tight_layout()
            fig.savefig(fig_dir / 'regression_results.png', dpi=150)
            plt.close(fig)
        
    except Exception as e:
        logging.error(f"Could not compute single-trial features: {e}")
        import traceback
        traceback.print_exc()

    # Fast vs slow RT overlays and epoch images
    try:
        if len(RTs) > 0 and len(e_stimZ) == len(RTs):
            med_rt = np.nanmedian(RTs)
            fast_idx = (RTs <= med_rt)
            slow_idx = (RTs > med_rt)
            if fast_idx.sum() >= 3 and slow_idx.sum() >= 3:
                ev_fast = e_stimZ[fast_idx].average()
                ev_slow = e_stimZ[slow_idx].average()
                try:
                    from mne.viz import plot_compare_evokeds
                    pick_cpz = safe_picks(ev_fast.info, ['CPz', 'Pz']) or ['Cz']
                    if len(pick_cpz) >= 2:
                        fig = plot_compare_evokeds({'fast': ev_fast, 'slow': ev_slow}, picks=pick_cpz, show=False, combine='mean')
                    else:
                        fig = plot_compare_evokeds({'fast': ev_fast, 'slow': ev_slow}, picks=pick_cpz, show=False)
                    fig.savefig(fig_dir / 'erp_stim_cpz_fast_slow.png', dpi=150)
                    plt.close(fig)
                except Exception:
                    pass

            # Epoch image sorted by RT
            try:
                order = np.argsort(RTs)
                pick_img = safe_picks(e_stimZ.info, ['CPz', 'Pz']) or ['Cz']
                fig_img = e_stimZ.copy().plot_image(picks=pick_img, order=order, show=False)
                (fig_img if isinstance(fig_img, list) else [fig_img])[0].savefig(fig_dir / 'epochs_image_stim_cpz.png', dpi=150)
                plt.close('all')
            except Exception:
                pass
    except Exception as e:
        logging.warning(f"Could not generate fast/slow overlays or epoch image: {e}")

    # Fast vs slow TFRs using built-in viz
    try:
        if len(RTs) > 0 and len(e_stim) == len(RTs):
            med_rt = np.nanmedian(RTs)
            e_fast = e_stim[RTs <= med_rt]
            e_slow = e_stim[RTs > med_rt]
            if len(e_fast) >= 3:
                tfr_fast = e_fast.copy().pick(picks_stim).compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                                                      use_fft=True, return_itc=False, average=True, n_jobs=1)
                tfr_fast = apply_tfr_baseline_if_possible(tfr_fast, baseline=(-0.5, -0.1), mode='logratio', label='stim-fast')
                fig_tf = tfr_fast.plot(picks='Pz' if 'Pz' in e_stim.ch_names else None, show=False)
                (fig_tf[0] if isinstance(fig_tf, list) else fig_tf).savefig(fig_dir / 'tfr_stim_fast.png', dpi=150)
                plt.close('all')
            if len(e_slow) >= 3:
                tfr_slow = e_slow.copy().pick(picks_stim).compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                                                       use_fft=True, return_itc=False, average=True, n_jobs=1)
                tfr_slow = apply_tfr_baseline_if_possible(tfr_slow, baseline=(-0.5, -0.1), mode='logratio', label='stim-slow')
                fig_tf = tfr_slow.plot(picks='Pz' if 'Pz' in e_stim.ch_names else None, show=False)
                (fig_tf[0] if isinstance(fig_tf, list) else fig_tf).savefig(fig_dir / 'tfr_stim_slow.png', dpi=150)
                plt.close('all')

        if e_resp is not None and len(RTs) > 0 and len(e_resp) == len(RTs):
            med_rt = np.nanmedian(RTs)
            e_fast_r = e_resp[RTs <= med_rt]
            e_slow_r = e_resp[RTs > med_rt]
            if len(e_fast_r) >= 3:
                tfr_fast_r = e_fast_r.copy().pick(picks_resp).compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                                                          use_fft=True, return_itc=False, average=True, n_jobs=1)
                tfr_fast_r = apply_tfr_baseline_if_possible(tfr_fast_r, baseline=(-0.8, -0.2), mode='logratio', label='resp-fast')
                fig_tf = tfr_fast_r.plot(picks='Cz' if 'Cz' in e_resp.ch_names else None, show=False)
                (fig_tf[0] if isinstance(fig_tf, list) else fig_tf).savefig(fig_dir / 'tfr_resp_fast.png', dpi=150)
                plt.close('all')
            if len(e_slow_r) >= 3:
                tfr_slow_r = e_slow_r.copy().pick(picks_resp).compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                                                           use_fft=True, return_itc=False, average=True, n_jobs=1)
                tfr_slow_r = apply_tfr_baseline_if_possible(tfr_slow_r, baseline=(-0.8, -0.2), mode='logratio', label='resp-slow')
                fig_tf = tfr_slow_r.plot(picks='Cz' if 'Cz' in e_resp.ch_names else None, show=False)
                (fig_tf[0] if isinstance(fig_tf, list) else fig_tf).savefig(fig_dir / 'tfr_resp_slow.png', dpi=150)
                plt.close('all')
    except Exception as e:
        logging.warning(f"Could not generate fast/slow TFRs: {e}")
    # Save metrics JSON
    with open(report_dir / 'summary_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Per-trial table
    df_trials = pd.DataFrame({
        'trial': np.arange(len(RTs)),
        'RT_s': RTs,
    })
    df_trials.to_csv(report_dir / 'fivepoint_trials.csv', index=False)

    # Build MNE Report using saved figures
    try:
        report = Report(title='Five-Point Task EEG Analysis')
        # ERP figures
        for fname in ['erp_stim_occ.png', 'erp_stim_cpar.png', 'erp_resp_cz.png',
                      'erp_stim_cpz_classic.png', 'erp_stim_cpz_zscore.png', 'erp_stim_cpz_regression.png',
                      'erp_resp_cz_classic.png', 'erp_resp_cz_zscore.png', 'erp_resp_cz_regression.png',
                      'erp_stim_joint.png', 'erp_resp_joint.png']:
            fpath = fig_dir / fname
            if fpath.exists():
                report.add_image(str(fpath), title=fname.replace('_', ' ').replace('.png',''), tags=('ERP',))
        # TFR figures
        for fname in ['tfr_stim.png', 'tfr_resp.png', 'tfr_stim_fast.png', 'tfr_stim_slow.png', 'tfr_resp_fast.png', 'tfr_resp_slow.png']:
            fpath = fig_dir / fname
            if fpath.exists():
                report.add_image(str(fpath), title=fname.replace('_', ' ').replace('.png',''), tags=('TFR',))
        # Stroke analysis
        fpath = fig_dir / 'stroke_analysis.png'
        if fpath.exists():
            report.add_image(str(fpath), title='Stroke analysis', tags=('Execution',))
        # RT relationships
        for fname in ['rt_hist.png', 'rt_scatter_cpp_slope.png', 'rt_scatter_p3_amp.png', 'erp_stim_cpz_fast_slow.png', 'epochs_image_stim_cpz.png']:
            fpath = fig_dir / fname
            if fpath.exists():
                report.add_image(str(fpath), title=fname.replace('_', ' ').replace('.png',''), tags=('RT',))
        # Regression results bar chart
        fpath = fig_dir / 'regression_results.png'
        if fpath.exists():
            report.add_image(str(fpath), title='Regression results', tags=('Stats',))
        # Save report
        out_html = report_dir / 'report.html'
        report.save(str(out_html), overwrite=True, open_browser=False)
        logging.info(f"Saved MNE report to: {out_html}")
    except Exception as e:
        logging.warning(f"Could not build MNE report: {e}")
    
    logging.info(f"Saved figures to: {fig_dir}")
    logging.info(f"Saved metrics to: {report_dir}")


if __name__ == "__main__":
    main()
