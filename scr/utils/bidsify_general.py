#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal config-driven EEG BIDS converter.

This streamlined version focuses on the current need: convert Mario (and similar)
datasets by discovering recordings from a root folder, extracting trigger-channel
 events with simple JSON-configurable rules, and writing BIDS outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne

try:
    from mne_bids import (
        BIDSPath,
        write_raw_bids,
        make_dataset_description,
        update_sidecar_json,
    )
    _HAS_MNE_BIDS = True
except Exception:  # pragma: no cover
    BIDSPath = None  # type: ignore[assignment]
    write_raw_bids = None  # type: ignore[assignment]
    make_dataset_description = None  # type: ignore[assignment]
    update_sidecar_json = None  # type: ignore[assignment]
    _HAS_MNE_BIDS = False

CONFIG_ENCODING = "utf-8-sig"


def _log(prefix: str, message: str) -> None:
    print(f"[{prefix}] {message}")


def log_info(message: str) -> None:
    _log("info", message)


def log_warn(message: str) -> None:
    _log("warn", message)


def log_plan(message: str) -> None:
    _log("plan", message)


# ---------------------------------------------------------------------------
# DSI-24 channel normalization (mirrors Mario implementation)
# ---------------------------------------------------------------------------

LEGACY_TO_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_RAW_STIM_NAMES = {
    "TRIGGER",
    "TRIG",
    "TRG",
    "STATUS",
    "STI 014",
    "DIGITAL",
    "DIGITALIO",
    "DIN",
    "EVENT",
    "EVENTS",
    "MARKER",
    "STIM",
}

_RAW_EOG_NAMES = {"HEOG", "VEOG", "EOG", "EOGL", "EOGR"}

STIM_NAMES = {name.upper() for name in _RAW_STIM_NAMES}
EOG_NAMES = {name.upper() for name in _RAW_EOG_NAMES}

DEFAULT_EEG_REFERENCE = "Pz (Common Mode Follower)"
DEFAULT_EEG_GROUND = "Fpz"

_VENDOR_PREFIX_RE = re.compile(r"^EEG\s+(?:X\d+:)?", flags=re.IGNORECASE)
_PZ_SUFFIX_RE = re.compile(r"-PZ$", flags=re.IGNORECASE)
_X_PREFIX_RE = re.compile(r"^EEG\s+X(\d+):", flags=re.IGNORECASE)


def rename_dsi_channels(raw: mne.io.BaseRaw) -> Dict[str, str]:
    """Replace legacy DSI electrode labels with their 10-20 equivalents."""
    mapping: Dict[str, str] = {ch: LEGACY_TO_1020[ch] for ch in raw.ch_names if ch in LEGACY_TO_1020}
    if mapping:
        raw.rename_channels(mapping)
    return mapping


def normalize_dsi_channel_names(raw: mne.io.BaseRaw) -> Dict[str, str]:
    """Strip vendor prefixes/suffixes so channel names line up with BIDS expectations."""
    original = list(raw.ch_names)
    taken = set(original)
    mapping: Dict[str, str] = {}

    for ch in original:
        new = ch.strip()
        x_match = _X_PREFIX_RE.match(new)
        x_idx = x_match.group(1) if x_match else None

        new = _VENDOR_PREFIX_RE.sub("", new)
        new = _PZ_SUFFIX_RE.sub("", new)

        if new == "" and x_idx is not None:
            new = f"X{x_idx}"

        if new != ch:
            candidate = new
            if candidate in taken:
                idx = 2
                while f"{candidate}_{idx}" in taken:
                    idx += 1
                candidate = f"{candidate}_{idx}"
            mapping[ch] = candidate
            taken.add(candidate)

    if mapping:
        raw.rename_channels(mapping)
    return mapping


def set_channel_types(raw: mne.io.BaseRaw, force: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Infer DSI-24 channel types (stim, eog, misc...) for BIDS metadata."""
    mapping: Dict[str, str] = {}
    for ch in raw.ch_names:
        cu = ch.upper()
        if cu in STIM_NAMES:
            mapping[ch] = "stim"
        elif cu in EOG_NAMES:
            mapping[ch] = "eog"
        elif "ECG" in cu:
            mapping[ch] = "ecg"
        elif cu in {"A1", "A2", "M1", "M2"}:
            mapping[ch] = "misc"
        elif cu == "CM" or cu.startswith("X"):
            mapping[ch] = "misc"

    if force:
        for k, v in force.items():
            if k in raw.ch_names:
                mapping[k] = v
    if mapping:
        raw.set_channel_types(mapping)
    return mapping


def _sanitize_channel_units(raw: mne.io.BaseRaw) -> None:
    """Ensure raw._orig_units contains ASCII BIDS-compliant unit labels."""
    orig_units = getattr(raw, "_orig_units", None)
    if orig_units is None:
        raw._orig_units = {}
        orig_units = raw._orig_units
    elif not isinstance(orig_units, dict):
        raw._orig_units = dict(orig_units)  # type: ignore[arg-type]
        orig_units = raw._orig_units

    channel_types = raw.get_channel_types()
    for ch, ch_type in zip(raw.ch_names, channel_types):
        ctype = ch_type.lower()
        unit = orig_units.get(ch)
        if ctype in {"stim", "misc"}:
            orig_units[ch] = "n/a"
            continue

        if unit is None:
            if ctype in {"eeg", "eog", "ecg", "emg"}:
                orig_units[ch] = "uV"
            else:
                orig_units[ch] = "n/a"
            continue

        unit_str = str(unit)
        if unit_str == "µV":
            unit_str = "uV"
        elif unit_str.lower() == "na":
            unit_str = "n/a"
        orig_units[ch] = unit_str

def find_stim_channel(raw: mne.io.BaseRaw) -> Optional[str]:
    """Best-effort search for a stimulus channel using heuristics and set channel types."""
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    for pref in ("Trigger", "STATUS"):
        if pref in raw.ch_names and types.get(pref) == "stim":
            return pref
    for ch, t in types.items():
        if t == "stim":
            return ch
    for name in STIM_NAMES:
        if name in raw.ch_names:
            return name
    return None


def apply_standard_montage(raw: mne.io.BaseRaw) -> bool:
    """Attach a 10-20 montage when possible; ignore failures silently."""
    try:
        from mne.channels import make_standard_montage

        montage = make_standard_montage("standard_1020")
        eeg_names = [ch for ch, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"]
        montage = montage.copy().pick_channels([ch for ch in eeg_names if ch in montage.ch_names])
        raw.set_montage(montage, match_case=False, on_missing="ignore")
        return True
    except Exception:
        return False


def prepare_dsi24_raw(
    raw: mne.io.BaseRaw,
    apply_montage: bool = True,
    force_types: Optional[Dict[str, str]] = None,
) -> mne.io.BaseRaw:
    """Run the standard normalization pipeline for DSI-24 EDF recordings."""
    normalize_dsi_channel_names(raw)
    rename_dsi_channels(raw)
    set_channel_types(raw, force=force_types)
    _sanitize_channel_units(raw)

    try:
        types = dict(zip(raw.ch_names, raw.get_channel_types()))
        stim_candidates = [ch for ch, t in types.items() if t == "stim"]
        if len(stim_candidates) > 1:
            primary = "Trigger" if "Trigger" in stim_candidates else (
                "STATUS" if "STATUS" in stim_candidates else stim_candidates[0]
            )
            for ch in stim_candidates:
                if ch != primary:
                    raw.set_channel_types({ch: "misc"})
    except Exception:
        pass

    if apply_montage:
        apply_standard_montage(raw)
    return raw

ALLOWED_SUFFIXES = {".edf", ".bdf"}


@dataclass
class RecordingSpec:
    """Normalized instructions for converting a single raw EEG file into BIDS."""
    raw_path: Path
    subject: str
    session: str
    task: str
    run: str
    stim_channel: Optional[str] = None
    line_freq: Optional[int] = None
    overwrite: bool = False
    trigger_rules: Dict[str, Any] = field(default_factory=dict)
    apply_montage: bool = False
    # Optional per-dataset channel type overrides
    force_channel_types: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level parameters that apply to the full conversion run."""
    bids_root: Path
    dataset_name: Optional[str]
    dataset_authors: List[str]
    recordings: List[RecordingSpec]
    overwrite_dataset_description: bool = True
    # Optional EEG hardware metadata to set in sidecar
    eeg_reference: Optional[str] = None
    eeg_ground: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _iter_raw_files(root: Path) -> Iterable[Path]:
    """Yield raw data files under ``root`` whose suffix is explicitly allowed."""
    for suffix in ALLOWED_SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def _parse_entities(stem: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract BIDS-style ``sub``, ``ses``, ``task`` and ``run`` entities from a filename stem."""
    subject = session = task = run = None
    for part in stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        key = key.lower()
        if key == "sub" and subject is None:
            subject = value
        elif key == "ses" and session is None:
            session = value
        elif key == "task" and task is None:
            task = value
        elif key == "run" and run is None:
            run = value
    return subject, session, task, run


def _as_list(value: Any) -> Optional[List[str]]:
    """Normalize scalar or iterable config values into a list of trimmed strings."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else None


def _sanitize_task_label(label: Optional[str]) -> Optional[str]:
    """Make sure task labels match BIDS expectations (lowercase, no hyphens)."""
    if label is None:
        return None
    task = label.lower()
    if task.endswith("-eeg"):
        task = task[:-4]
    return task.replace("-", "").strip() or None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _validate_config_dict(data: Dict[str, Any], source: Path) -> None:
    """Fail fast when required top-level keys are missing or malformed."""
    missing = [key for key in ("bids_root", "recordings") if key not in data]
    if missing:
        raise ValueError(f"Config {source} missing required keys: {', '.join(missing)}")
    if not isinstance(data["recordings"], list):
        raise ValueError("'recordings' must be a list of configuration blocks.")


def load_config(config_path: Path) -> PipelineConfig:
    """Parse a JSON configuration file into the normalized ``PipelineConfig`` model."""
    data = json.loads(config_path.read_text(encoding=CONFIG_ENCODING))
    _validate_config_dict(data, config_path)
    bids_root = Path(data["bids_root"]).expanduser()
    if not bids_root.is_absolute():
        bids_root = (config_path.parent / bids_root).resolve()

    recordings_cfg = data.get("recordings")
    if not isinstance(recordings_cfg, list) or not recordings_cfg:
        raise ValueError("Config must contain a non-empty 'recordings' list.")

    recordings: List[RecordingSpec] = []
    for idx, entry in enumerate(recordings_cfg):
        recordings.extend(_expand_entry(entry, config_path.parent, idx))

    if not recordings:
        raise ValueError("No recordings matched the configuration request.")

    return PipelineConfig(
        bids_root=bids_root,
        dataset_name=data.get("dataset_name"),
        dataset_authors=[str(a) for a in data.get("dataset_authors", [])],
        recordings=recordings,
        overwrite_dataset_description=bool(data.get("overwrite_dataset_description", True)),
        eeg_reference=(data.get("eeg_reference") or None),
        eeg_ground=(data.get("eeg_ground") or None),
    )


def _expand_entry(entry: Dict[str, Any], base_dir: Path, idx: int) -> List[RecordingSpec]:
    """Expand a single ``recordings`` block into one ``RecordingSpec`` per matching file."""
    if "root" not in entry:
        raise ValueError(f"Recording block #{idx} missing 'root'.")

    root = Path(entry["root"]).expanduser()
    if not root.is_absolute():
        root = (base_dir / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Recording block #{idx}: path does not exist -> {root}")

    subject_filter = _as_list(entry.get("subjects")) or _as_list(entry.get("subject"))
    session_filter = _as_list(entry.get("sessions")) or _as_list(entry.get("session"))
    run_filter = _as_list(entry.get("runs")) or _as_list(entry.get("run"))

    task_override = entry.get("task")
    task_override = _sanitize_task_label(task_override) if task_override else None

    line_freq = entry.get("line_freq")
    stim_channel = entry.get("stim_channel")
    overwrite = bool(entry.get("overwrite", False))
    apply_montage = bool(entry.get("apply_montage", False))
    trigger_rules = dict(entry.get("trigger_rules", {}))
    force_channel_types = dict(entry.get("force_channel_types", {}))

    specs: List[RecordingSpec] = []
    for raw_file in sorted(_iter_raw_files(root)):
        subject, session, raw_task, run = _parse_entities(raw_file.stem)
        if subject is None or session is None:
            continue
        if subject_filter and subject not in subject_filter:
            continue
        if session_filter and session not in session_filter:
            continue
        if run is None:
            run = "01"
        if run_filter and run not in run_filter:
            continue

        task = task_override or _sanitize_task_label(raw_task) or "task"
        specs.append(
            RecordingSpec(
                raw_path=raw_file.resolve(),
                subject=subject,
                session=session,
                task=task,
                run=run,
                stim_channel=stim_channel,
                line_freq=line_freq,
                overwrite=overwrite,
                trigger_rules=trigger_rules,
                apply_montage=apply_montage,
                force_channel_types=force_channel_types,
            )
        )

    if not specs:
        log_warn(f"No recordings matched configuration block #{idx} under {root}")
    return specs


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_raw_file(raw_path: Path) -> mne.io.BaseRaw:
    """Load a supported raw EEG file into MNE."""
    suffix = raw_path.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported raw format: {raw_path}")
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(raw_path), preload=True, verbose=False)
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(raw_path), preload=True, verbose=False)
    # Unreachable due to check above
    raise RuntimeError("Unsupported raw format")


def prepare_raw(
    raw: mne.io.BaseRaw,
    line_freq: Optional[int],
    apply_montage: bool,
    force_types: Optional[Dict[str, str]] = None,
) -> mne.io.BaseRaw:
    """Apply line frequency metadata and DSI-specific cleanups before conversion."""
    if line_freq is not None:
        raw.info["line_freq"] = int(line_freq)
    try:
        raw = prepare_dsi24_raw(raw, apply_montage=apply_montage, force_types=force_types)
    except Exception as exc:
        log_warn(f"prepare_dsi24_raw failed: {exc}")
        # Fallback: attempt basic normalization steps individually
        try:
            normalize_dsi_channel_names(raw)
            rename_dsi_channels(raw)
            set_channel_types(raw, force=force_types)
            _sanitize_channel_units(raw)
        except Exception as sub_exc:
            log_warn(f"DSI-24 fallback normalization failed: {sub_exc}")
    else:
        _sanitize_channel_units(raw)
    return raw


def detect_stim_channel(raw: mne.io.BaseRaw, preferred: Optional[str]) -> Optional[str]:
    """Find the best stimulus channel candidate, respecting explicit preferences."""
    if preferred:
        preferred_lower = preferred.lower()
        for ch in raw.ch_names:
            if ch.lower() == preferred_lower:
                return ch
    stim = find_stim_channel(raw)
    if stim is not None:
        return stim
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    for ch, kind in types.items():
        if kind == "stim":
            return ch
    return None


def _apply_trigger_rules(values: np.ndarray, rules: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Return possibly remapped trigger values and a boolean keep mask."""
    values = values.astype(int, copy=False)
    value_map = rules.get("value_map") or {}
    if value_map:
        mapped: List[int] = []
        for val in values:
            candidate = value_map.get(str(val), value_map.get(val, val))
            try:
                mapped.append(int(candidate))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"value_map entry for trigger {val!r} must be integer-like (got {candidate!r})."
                ) from exc
        values = np.asarray(mapped, dtype=int)

    keep = np.ones_like(values, dtype=bool)
    if bool(rules.get("drop_zero", False)):
        keep &= values != 0

    allowed = rules.get("keep_values") or rules.get("allowed_values")
    if allowed:
        try:
            allowed_set = {int(v) for v in allowed}
        except (TypeError, ValueError) as exc:
            raise ValueError("keep_values entries must be integer-like.") from exc
        keep &= np.array([int(val) in allowed_set for val in values], dtype=bool)

    return values, keep


def _trial_types_from_rules(values: np.ndarray, rules: Dict[str, Any]) -> List[str]:
    """Translate trigger values into trial_type strings using ``trial_type_map`` rules."""
    default_tt = rules.get("trial_type")
    tt_map = rules.get("trial_type_map", {})
    trial_types = []
    for val in values:
        key = str(int(val))
        trial_types.append(tt_map.get(key, default_tt or f"trigger_{int(val)}"))
    return trial_types


def _binary_edge_samples(
    raw: mne.io.BaseRaw,
    stim_name: str,
    threshold: float,
    direction: str = "rise",
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect transitions in a binary trigger channel based on a voltage threshold."""
    picks = mne.pick_channels(raw.ch_names, include=[stim_name])
    if len(picks) != 1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    data = raw.get_data(picks=picks)[0]
    binary = (data > threshold).astype(int)
    if binary.size <= 1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    edges = np.diff(binary)
    edge_indices = np.flatnonzero(edges != 0) + 1
    if edge_indices.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    edge_values = binary[edge_indices]
    # Filter by direction
    if direction == "rise":
        keep = edge_values == 1
    elif direction == "fall":
        keep = edge_values == 0
    else:  # both
        keep = np.ones_like(edge_values, dtype=bool)
    return edge_indices[keep].astype(int), edge_values[keep].astype(int)


def extract_trigger_events(raw: mne.io.BaseRaw, spec: RecordingSpec) -> pd.DataFrame:
    """Build an events table for downstream ``write_raw_bids`` consumption."""
    stim = detect_stim_channel(raw, spec.trigger_rules.get("stim_channel") or spec.stim_channel)
    columns = ["onset", "duration", "trial_type", "value", "sample", "stim_channel"]
    if stim is None:
        log_warn(f"No stimulus channel detected for {spec.raw_path.name}; events.tsv will be empty.")
        return pd.DataFrame(columns=columns)

    binary_edges = bool(spec.trigger_rules.get("binary_edges", False))
    threshold = float(spec.trigger_rules.get("binary_threshold", 0.5))
    edge_dir = str(spec.trigger_rules.get("edge_direction", "rise")).lower()

    samples: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None

    if binary_edges:
        try:
            samples, values = _binary_edge_samples(raw, stim, threshold, direction=edge_dir)
        except Exception as exc:
            log_warn(f"binary edge detection failed on {stim}: {exc}")
            samples = values = None

    if samples is None or values is None:
        try:
            events = mne.find_events(raw, stim_channel=stim, shortest_event=1, consecutive=True, verbose=False)
        except Exception as exc:
            log_warn(f"Failed to extract events from {stim}: {exc}")
            return pd.DataFrame(columns=columns)

        if events.size == 0:
            log_warn(f"No discrete events detected on {stim} for {spec.raw_path.name}.")
            return pd.DataFrame(columns=columns)

        samples = events[:, 0].astype(int)
        values = events[:, 2].astype(int)

    if samples.size == 0:
        return pd.DataFrame(columns=columns)

    sfreq = float(raw.info.get("sfreq", 1.0))
    onsets = samples.astype(float) / sfreq

    values, keep = _apply_trigger_rules(values, spec.trigger_rules)
    if not np.any(keep):
        return pd.DataFrame(columns=columns)

    onsets = onsets[keep]
    values = values[keep]
    samples = samples[keep]
    trial_types = _trial_types_from_rules(values, spec.trigger_rules)

    trigger_df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": np.zeros_like(onsets),
            "trial_type": trial_types,
            "value": values,
            "sample": samples,
            "stim_channel": stim,
        },
        columns=columns,
    )
    return trigger_df

def make_mne_events_and_metadata(
    events_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, int], pd.DataFrame, Dict[str, str]]:
    """Return (events_array, event_id_map, event_metadata_df, extra_cols_desc).

    - events_array: shape (n, 3) with columns [sample, 0, event_code]
    - event_id_map: {trial_type -> event_code}
    - event_metadata_df: extra columns to add to events.tsv (not onset/duration/trial_type)
    - extra_cols_desc: {column_name -> description string} required by mne-bids
    """
    if events_df.empty:
        return (
            np.empty((0, 3), dtype=int),
            {},
            pd.DataFrame(),
            {},
        )

    df = events_df.sort_values("sample", kind="stable").reset_index(drop=True)

    unique_tt = list(dict.fromkeys(df["trial_type"].astype(str).tolist()))

    # Build desired code per trial_type using raw TTL values if available, with mapping 0->1 and others unchanged
    desired_code_by_tt: Dict[str, int] = {}
    used_codes: set[int] = set()
    if "value" in df.columns:
        for tt, raw_val in zip(df["trial_type"].astype(str), df["value" ].astype(int)):
            if tt in desired_code_by_tt:
                continue
            code = int(raw_val)
            code = 1 if code == 0 else code
            # Ensure uniqueness across trial_types; if collision, bump to next free positive int
            if code in used_codes:
                next_code = max(1, code)
                while next_code in used_codes:
                    next_code += 1
                code = next_code
            desired_code_by_tt[tt] = code
            used_codes.add(code)

    # Fallback if no values column or something went wrong
    if not desired_code_by_tt:
        for i, tt in enumerate(unique_tt, start=1):
            desired_code_by_tt[tt] = i

    event_id = {tt: desired_code_by_tt[tt] for tt in unique_tt}

    samples = df["sample"].astype(int).to_numpy()
    zeros = np.zeros_like(samples, dtype=int)
    codes = np.array([event_id[str(tt)] for tt in df["trial_type"]], dtype=int)
    events = np.column_stack([samples, zeros, codes])

    # No extra metadata columns requested; mne-bids will still write trial_type
    event_metadata = pd.DataFrame(index=df.index)
    extra_desc: Dict[str, str] = {}

    return events, event_id, event_metadata, extra_desc


def maybe_update_eeg_sidecar(bids_path: 'BIDSPath', ref: Optional[str], ground: Optional[str]) -> None:
    """Optionally set EEGReference/EEGGround in *_eeg.json using mne-bids."""
    if not _HAS_MNE_BIDS or (ref is None and ground is None):
        return
    entries: Dict[str, Any] = {}
    if ref:
        entries["EEGReference"] = ref
    if ground:
        entries["EEGGround"] = ground
    try:
        update_sidecar_json(bids_path, entries)  # type: ignore[misc]
    except Exception as exc:
        log_warn(f"Failed to update EEG sidecar fields: {exc}")


def ensure_dataset_description(root: Path, name: Optional[str], authors: List[str], overwrite: bool) -> None:
    """Create or update the dataset_description.json required by BIDS."""
    if not _HAS_MNE_BIDS:
        log_warn("mne-bids not available; skipping dataset description update.")
        return
    root.mkdir(parents=True, exist_ok=True)
    try:
        make_dataset_description(
            path=str(root),
            name=name or "EEG dataset",
            authors=authors or None,
            dataset_type="raw",
            overwrite=overwrite,
        )
    except Exception:
        pass


def process_recording(spec: RecordingSpec, cfg: PipelineConfig, dry_run: bool = False) -> Optional['BIDSPath']:
    """Convert one recording to BIDS, returning the resulting ``BIDSPath``."""
    log_plan(
        f"{spec.raw_path} -> sub-{spec.subject}_ses-{spec.session}_task-{spec.task}_run-{spec.run}"
    )
    if dry_run:
        return None
    if not _HAS_MNE_BIDS:
        raise RuntimeError("mne-bids is required to run the conversion. Please install 'mne-bids'.")

    raw = load_raw_file(spec.raw_path)
    raw = prepare_raw(raw, spec.line_freq, spec.apply_montage, spec.force_channel_types)

    # Extract trigger events and prepare MNE events + metadata
    trigger_events = extract_trigger_events(raw, spec)
    events_array, event_id_map, event_metadata, extra_desc = make_mne_events_and_metadata(trigger_events)

    bids_path = BIDSPath(
        subject=spec.subject,
        session=spec.session,
        task=spec.task,
        run=spec.run,
        datatype="eeg",
        root=str(cfg.bids_root),
    )

    write_kwargs = dict(
        overwrite=spec.overwrite,
        allow_preload=True,
        format="BrainVision",
        verbose=False,
    )
    if events_array.size:
        write_kwargs["events"] = events_array
        if event_id_map:
            write_kwargs["event_id"] = event_id_map

    montage = None
    try:
        montage = raw.get_montage()
    except Exception:
        montage = None
    if montage is not None:
        try:
            raw.set_montage(None)
        except Exception:
            montage = None

    # Avoid mixing MNE annotations into events union when we supply our events
    try:
        raw.set_annotations(None)
    except Exception:
        pass
    write_raw_bids(
        raw,
        bids_path,
        event_metadata=None,
        extra_columns_descriptions=None,
        **write_kwargs,
    )

    if montage is not None:
        try:
            raw.set_montage(montage)
        except Exception:
            pass

    # Optional: EEGReference/EEGGround from config
    maybe_update_eeg_sidecar(bids_path, cfg.eeg_reference, cfg.eeg_ground)
    log_info(f"wrote {bids_path.fpath}")
    return bids_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI entry point definition for running the converter as a script."""
    parser = argparse.ArgumentParser(description="Minimal config-based EEG BIDS converter")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Plan actions without writing output")
    return parser


def run_from_config(config_path: Path, dry_run: bool = False) -> None:
    """Load configuration and process every requested recording."""
    cfg = load_config(config_path)
    if not _HAS_MNE_BIDS and not dry_run:
        raise RuntimeError("mne-bids is required for conversion. Install 'mne-bids'.")
    if not dry_run:
        ensure_dataset_description(
            cfg.bids_root,
            cfg.dataset_name,
            cfg.dataset_authors,
            overwrite=cfg.overwrite_dataset_description,
        )
        log_info(f"Dataset description ready at {cfg.bids_root}")
    else:
        log_info(f"[dry-run] would create/update dataset at {cfg.bids_root}")
    for spec in cfg.recordings:
        process_recording(spec, cfg, dry_run=dry_run)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_config(Path(args.config).resolve(), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
