"""
JSON-driven event mapping for MNE/MNE-BIDS.

Converts a raw events array (n, 3) into:
  - mapped events (with debounced / onset-shifted samples)
  - event_id {label -> code} suitable for MNE/MNE-BIDS

Expected task JSON structure (per bidsfirst_config_schema.json):
{
  "event_map": { "GO": [1], "NOGO": [2], "START": [7] },
  "aliases?": { "Start": 7 },
  "onset_shift_sec": 0.0,
  "debounce_ms?": 5,
  "epoching": { "tmin": -0.2, "tmax": 0.8, "baseline": [-0.2, 0.0] }
}
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple
import numpy as np
import mne


def _normalize_event_map(event_map: Mapping[str, Iterable[int]] | None,
                         aliases: Mapping[str, int] | None) -> Dict[str, int]:
    """
    Accept lists, coerce to a compact mapping with unique integer codes.
    If a label lists multiple codes, we keep each code and return event_id [{label_i: code_i}].
    """
    if not event_map:
        return {}
    out: Dict[str, int] = {}
    for label, vals in event_map.items():
        for v in vals if isinstance(vals, (list, tuple)) else [vals]:
            if isinstance(v, str) and v.isdigit():
                code = int(v)
            elif isinstance(v, int):
                code = v
            elif aliases and isinstance(v, str) and v in aliases:
                code = int(aliases[v])
            else:
                raise ValueError(f"Unsupported trigger value {v!r} for label {label!r}")
            # allow duplicates across labels (user responsibility)
            out[f"{label}"] = code  # last wins for same label
    return out


def _collapse_with_refractory(events: np.ndarray, refractory_samples: int) -> np.ndarray:
    """
    Collapse runs of identical codes that occur within 'refractory_samples' samples.
    Keeps the first occurrence of each cluster.
    """
    if refractory_samples <= 0 or events.size == 0:
        return events
    keep = [0]
    last_idx = 0
    for i in range(1, events.shape[0]):
        same_code = events[i, 2] == events[last_idx, 2]
        close = (events[i, 0] - events[last_idx, 0]) <= refractory_samples
        if not (same_code and close):
            keep.append(i)
            last_idx = i
    return events[np.array(keep, dtype=int)]


def _apply_onset_shift(events: np.ndarray, sfreq: float, onset_shift_sec: float) -> np.ndarray:
    if events.size == 0 or onset_shift_sec == 0.0:
        return events
    shift = int(round(onset_shift_sec * sfreq))
    ev = events.copy()
    ev[:, 0] = np.maximum(ev[:, 0] + shift, 0)
    return ev


def map_events_from_config(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    task_config: Mapping[str, object],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Parameters
    ----------
    raw : BaseRaw
      Used to obtain sfreq for onset shift / debounce.
    events : (n, 3) int array
      Raw event detections (sample, prev, code)
    task_config : Dict
      Must contain "event_map"; may include "aliases", "onset_shift_sec", "debounce_ms"

    Returns
    -------
    mapped_events : (k, 3) int array
    event_id : Dict[str, int]
    """
    sfreq = float(raw.info["sfreq"])

    aliases = task_config.get("aliases") or {}
    event_map_raw = task_config.get("event_map") or {}
    event_id = _normalize_event_map(event_map_raw, aliases)

    if events.size == 0 or not event_id:
        return np.empty((0, 3), dtype=int), {}

    # Debounce identical codes within a short refractory window
    debounce_ms = float(task_config.get("debounce_ms", 5.0))
    refractory_samples = int(round(debounce_ms * 1e-3 * sfreq))
    events_sorted = events[np.argsort(events[:, 0])].astype(int)
    events_db = _collapse_with_refractory(events_sorted, refractory_samples)

    # Keep only codes referenced in event_id
    wanted_codes = set(event_id.values())
    mask = np.isin(events_db[:, 2], list(wanted_codes))
    ev_keep = events_db[mask]

    # Apply onset shift (if any)
    onset_shift_sec = float(task_config.get("onset_shift_sec", 0.0))
    ev_keep = _apply_onset_shift(ev_keep, sfreq, onset_shift_sec)

    # Final sort and unique
    if ev_keep.size:
        ev_keep = np.unique(ev_keep, axis=0)
        ev_keep = ev_keep[np.argsort(ev_keep[:, 0])]

    return ev_keep, dict(event_id)
