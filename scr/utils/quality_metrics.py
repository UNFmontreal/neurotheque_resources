"""
Utility helpers for computing EEG quality metrics at different pipeline stages.

The metrics here are intentionally lightweight so they can be called from multiple
processing steps without introducing heavy dependencies.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import mne
from mne.time_frequency import psd_array_welch


def _prepare_data_array(inst) -> Optional[np.ndarray]:
    """Return data as (n_series, n_times) in microvolts for metric computation."""
    try:
        picks = mne.pick_types(inst.info, eeg=True, eog=False, ecg=False, stim=False, misc=False, exclude="bads")
        if picks.size == 0:
            logging.warning("[quality_metrics] No EEG channels available for metrics.")
            return None

        if isinstance(inst, mne.io.BaseRaw):
            data = inst.get_data(picks=picks)
            series = data * 1e6  # convert to µV
        elif isinstance(inst, mne.BaseEpochs):
            data = inst.get_data()[:, picks, :]  # (n_epochs, n_channels, n_times)
            series = data.reshape(-1, data.shape[-1]) * 1e6
        else:
            # Generic object with get_data()
            data = inst.get_data()
            if data.ndim == 2:
                series = data[picks] * 1e6
            elif data.ndim == 3:
                series = data[:, picks, :].reshape(-1, data.shape[-1]) * 1e6
            else:
                logging.warning("[quality_metrics] Unsupported data dimension %s", data.shape)
                return None
        return series
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[quality_metrics] Failed to obtain data for metrics: %s", exc)
        return None


def compute_signal_quality_metrics(inst) -> Dict[str, float]:
    """
    Compute a compact set of quality metrics for the given MNE object.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.BaseEpochs
        Data container to analyse. Expected to contain EEG channels.

    Returns
    -------
    dict
        Dictionary with numeric metrics (values in float).
    """
    metrics: Dict[str, float] = {}
    series = _prepare_data_array(inst)
    if series is None:
        return metrics

    sfreq = float(inst.info.get("sfreq", 0.0))
    n_samples = series.shape[-1]

    # Peak-to-peak (95th percentile across segments)
    ptp_values = np.ptp(series, axis=-1)
    metrics["peak_to_peak_uV"] = float(np.percentile(ptp_values, 95))

    # RMS / standard deviation statistics
    std_values = np.std(series, axis=-1)
    metrics["median_std_uV"] = float(np.median(std_values))
    metrics["global_rms_uV"] = float(np.sqrt(np.mean(series ** 2)))

    # Power spectral density based metrics
    if sfreq > 0 and n_samples > 16:
        n_fft = min(4096, n_samples)
        try:
            psd, freqs = psd_array_welch(
                series,
                sfreq=sfreq,
                fmin=1.0,
                fmax=min(100.0, sfreq / 2.0),
                n_fft=n_fft,
                average="mean",
            )
            psd_mean = psd.mean(axis=0)

            # Line-noise ratio (60 Hz vs 45-55 Hz baseline)
            idx_60 = np.logical_and(freqs >= 58, freqs <= 62)
            idx_base = np.logical_and(freqs >= 45, freqs <= 55)
            if idx_60.any() and idx_base.any():
                metrics["line_noise_ratio"] = float(
                    psd_mean[idx_60].mean() / psd_mean[idx_base].mean()
                )

            # Alpha peak frequency
            idx_alpha = np.logical_and(freqs >= 8, freqs <= 12)
            if idx_alpha.any():
                alpha_freqs = freqs[idx_alpha]
                alpha_psd = psd_mean[idx_alpha]
                metrics["alpha_peak_hz"] = float(alpha_freqs[np.argmax(alpha_psd)])

            # Beta band relative power (13-30 Hz)
            idx_beta = np.logical_and(freqs >= 13, freqs <= 30)
            if idx_beta.any():
                beta_power = psd_mean[idx_beta].sum()
                total_power = psd_mean.sum()
                if total_power > 0:
                    metrics["beta_rel_power"] = float(beta_power / total_power)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("[quality_metrics] PSD computation failed: %s", exc)

    metrics["n_series"] = float(series.shape[0])
    metrics["n_samples"] = float(n_samples)
    metrics["sfreq"] = sfreq

    return metrics

