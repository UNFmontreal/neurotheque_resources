#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG BIDS converter for DSI-24.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from collections import Counter, defaultdict, deque
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne


try:
    from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, update_sidecar_json
    _HAS_MNE_BIDS = True
except Exception:  # pragma: no cover - allow unit tests without mne-bids
    BIDSPath = object  # type: ignore
    write_raw_bids = None  # type: ignore
    make_dataset_description = None  # type: ignore
    update_sidecar_json = None  # type: ignore
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
# DSI-24 channel normalization
# ---------------------------------------------------------------------------

LEGACY_TO_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_RAW_STIM_NAMES = {"TRIGGER","TRIG","TRG","STIM"}

_RAW_EOG_NAMES = {"EOG"}
# (flag set above)

STIM_NAMES = {name.upper() for name in _RAW_STIM_NAMES}
EOG_NAMES = {name.upper() for name in _RAW_EOG_NAMES}

DEFAULT_EEG_REFERENCE = "Pz (Common Mode Follower)"
DEFAULT_EEG_GROUND = "Fpz"

DEFAULT_MANUFACTURER = "Wearable Sensing"

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
        elif cu == "EVENT":
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
    behavior: Optional['BehaviorConfig'] = None

def _spec_label(spec: RecordingSpec) -> str:
    return f"{spec.subject}-{spec.session}-{spec.run}"


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


@dataclass
class BehaviorConfig:
    enabled: bool
    source_path: Path
    source_label: str
    onset_column: str
    duration_column: str
    trial_type_column: str
    anchor_trial_types: Optional[List[str]]
    sync_method: str
    alignment_strategy: str
    write_sidecar: bool
    output_beh: bool


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

def _as_set(value: Any) -> Optional[set]:
    items = _as_list(value)
    return set(items) if items else None


def _sanitize_task_label(label: Optional[str]) -> Optional[str]:
    """Make sure task labels match BIDS expectations (lowercase, no hyphens)."""
    if label is None:
        return None
    task = label.lower()
    if task.endswith("-eeg"):
        task = task[:-4]
    return task.replace("-", "").strip() or None

def _resolve_path(base_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _coerce_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return None



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
    """Expand a single `recordings` block into one `RecordingSpec` per matching file."""
    if "root" not in entry:
        raise ValueError(f"Recording block #{idx} missing 'root'.")

    root = _resolve_path(base_dir, entry["root"])
    if not root.exists():
        raise FileNotFoundError(f"Recording block #{idx}: path does not exist -> {root}")

    subject_filter = _as_set(entry.get("subjects") or entry.get("subject"))
    session_filter = _as_set(entry.get("sessions") or entry.get("session"))
    run_filter = _as_set(entry.get("runs") or entry.get("run"))

    task_override = _sanitize_task_label(entry.get("task"))

    line_freq = entry.get("line_freq")
    stim_channel = entry.get("stim_channel")
    overwrite = bool(entry.get("overwrite", False))
    apply_montage = bool(entry.get("apply_montage", False))
    trigger_rules = dict(entry.get("trigger_rules", {}))
    if trigger_rules:
        trigger_rules.setdefault("__config_dir__", str(base_dir))
    force_channel_types = dict(entry.get("force_channel_types", {}))
    behavior_spec = _build_behavior_config(entry.get("behavior_events"), base_dir, idx)

    def _matches(value: Optional[str], allowed: Optional[set]) -> bool:
        if value is None:
            return False
        return not allowed or value in allowed

    specs: List[RecordingSpec] = []
    for raw_file in sorted(_iter_raw_files(root)):
        subject, session, raw_task, run = _parse_entities(raw_file.stem)
        if not _matches(subject, subject_filter):
            continue
        if not _matches(session, session_filter):
            continue
        run = run or "01"
        if not _matches(run, run_filter):
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
                behavior=behavior_spec,
            )
        )

    if not specs:
        log_warn(f"No recordings matched configuration block #{idx} under {root}")
    return specs




def _build_behavior_config(cfg: Any, base_dir: Path, idx: int) -> Optional[BehaviorConfig]:
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        log_warn(
            f"Recording block #{idx}: 'behavior_events' must be a mapping; got {type(cfg).__name__}."
        )
        return None
    if not bool(cfg.get("enabled", False)):
        return None

    path_value = cfg.get("path")
    if not path_value:
        log_warn(
            f"Recording block #{idx}: 'behavior_events.path' is required when enabled."
        )
        return None

    path = _resolve_path(base_dir, path_value)

    onset_col = str(cfg.get("onset_column", "onset"))
    duration_col = str(cfg.get("duration_column", "duration"))
    trial_type_col = str(cfg.get("trial_type_column", "trial_type"))

    anchors_raw = cfg.get("anchor_trial_types")
    anchors = _coerce_str_list(anchors_raw)
    if anchors_raw is not None and anchors is None:
        log_warn(
            f"Recording block #{idx}: 'behavior_events.anchor_trial_types' must be a string or list."
        )

    sync_method = str(cfg.get("sync_method", "next_frame_on")).lower()
    if sync_method not in {"next_frame_on", "nearest_frame_on"}:
        log_warn(
            f"Recording block #{idx}: unsupported behavior_events.sync_method={sync_method!r}; using 'next_frame_on'."
        )
        sync_method = "next_frame_on"

    strategy_aliases = {
        "mean_offset": "mean_offset",
        "mean": "mean_offset",
        "first_anchor": "first_anchor",
        "first": "first_anchor",
        "linear": "linear",
        "linear_drift": "linear",
        "piecewise": "piecewise",
        "piecewise_linear": "piecewise",
    }

    strategy_input = cfg.get("alignment_strategy")
    if strategy_input is None:
        strategy_input = cfg.get("drift_strategy")
    if isinstance(strategy_input, bytes):
        strategy_input = strategy_input.decode(errors="ignore")
    if strategy_input is not None and not isinstance(strategy_input, str):
        log_warn(
            f"Recording block #{idx}: 'alignment_strategy' should be a string; got {type(strategy_input).__name__}."
        )
        strategy_input = None

    strategy_key = strategy_input.lower() if isinstance(strategy_input, str) else "mean_offset"
    strategy = strategy_aliases.get(strategy_key, strategy_key)

    fit_linear_legacy = bool(cfg.get("fit_linear_drift"))
    valid_strategies = {"mean_offset", "first_anchor", "linear", "piecewise"}

    if strategy not in valid_strategies:
        if strategy_key not in {None, "mean_offset"}:
            log_warn(
                f"Recording block #{idx}: unknown alignment_strategy={strategy!r}; using 'mean_offset'."
            )
        strategy = "linear" if fit_linear_legacy else "mean_offset"
    elif fit_linear_legacy and strategy == "mean_offset":
        strategy = "linear"

    return BehaviorConfig(
        enabled=True,
        source_path=path,
        source_label=str(path_value),
        onset_column=onset_col,
        duration_column=duration_col,
        trial_type_column=trial_type_col,
        anchor_trial_types=anchors,
        sync_method=sync_method,
        alignment_strategy=strategy,
        write_sidecar=bool(cfg.get("write_sidecar", True)),
        output_beh=bool(cfg.get("output_beh", True)),
    )



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


def _use_pulse_width_rules(rules: Dict[str, Any]) -> bool:
    """Return True if the trigger rules request pulse-width handling."""
    return bool(rules.get("pulse_width_map"))


def detect_pulses_from_stim(
    stim_values: np.ndarray,
    sfreq: float,
    keep_values: Optional[Iterable[int]] = None,
    binary_edges: bool = True,
) -> List[Dict[str, Any]]:
    """Detect contiguous TTL pulses and capture onset/duration information."""
    if stim_values.ndim != 1:
        raise ValueError("stim_values must be a 1D array.")
    stim_int = stim_values.astype(int)
    if keep_values:
        keep_set = {int(v) for v in keep_values}
        active_mask = np.isin(stim_int, list(keep_set))
    else:
        active_mask = stim_int != 0

    if not np.any(active_mask):
        return []
    # Identify contiguous runs of active samples.
    diff = np.diff(active_mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)

    pulses: List[Dict[str, Any]] = []
    for start_idx, stop_idx in zip(starts, stops):
        if stop_idx <= start_idx:
            continue
        segment = stim_int[start_idx:stop_idx]
        ttl_vals = segment[segment != 0]
        ttl_code = 0
        if ttl_vals.size:
            ttl_code = int(Counter(ttl_vals).most_common(1)[0][0])
        pulses.append(
            {
                "sample": int(start_idx),
                "stop_sample": int(stop_idx),
                "onset": float(start_idx) / float(sfreq),
                "duration": float(stop_idx - start_idx) / float(sfreq),
                "ttl": ttl_code,
            }
        )
    return pulses


def _duration_in_range(duration: float, bounds: Dict[str, Any]) -> bool:
    return float(bounds.get("min_s", -np.inf)) <= duration <= float(bounds.get("max_s", np.inf))


def classify_by_width(
    pulses: List[Dict[str, Any]],
    pulse_width_map: Iterable[Dict[str, Any]],
    ignore_windows: Iterable[Dict[str, Any]],
    default_trial_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Assign trial_type/value based on pulse durations."""
    mapped_events: List[Dict[str, Any]] = []
    ignore_windows = list(ignore_windows or [])
    width_entries = list(pulse_width_map or [])

    for pulse in sorted(pulses, key=lambda item: item["sample"]):
        duration = float(pulse["duration"])
        # Drop explicit ignore windows first.
        ignored = False
        for window in ignore_windows:
            if _duration_in_range(duration, window):
                ignored = True
                break
        if ignored:
            continue

        matched_entry: Optional[Dict[str, Any]] = None
        for entry in width_entries:
            if _duration_in_range(duration, entry):
                matched_entry = entry
                break

        trial_type = default_trial_type
        value_code: Optional[int] = None
        label = None

        if matched_entry:
            trial_type = matched_entry.get("label") or trial_type
            label = matched_entry.get("label")
            if matched_entry.get("code") is not None:
                value_code = int(matched_entry["code"])

        event = {
            "sample": int(pulse["sample"]),
            "onset": float(pulse["onset"]),
            "duration": duration,
            "trial_type": trial_type,
            "value": value_code if value_code is not None else int(pulse.get("ttl", 0)),
            "label": label,
            "ttl": int(pulse.get("ttl", 0)),
        }
        mapped_events.append(event)

    return mapped_events


def _normalize_collapse_spec(spec: Any) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, str):
        return {"trial_type": spec}
    if isinstance(spec, dict):
        return spec
    raise TypeError(f"Unsupported collapse spec type: {type(spec)!r}")


# ---------------------------------------------------------------------------
# TextSemantic specialization: map short/long tones to START/fixation/cw/END
# with behavior-driven cw_cong/cw_incong (correct trials only)
# ---------------------------------------------------------------------------


def _group_by_gap(onsets: List[float], max_gap_s: float) -> List[List[int]]:
    """Group consecutive onset indices whenever the gap stays within ``max_gap_s``."""
    if not onsets:
        return []
    groups: List[List[int]] = [[0]]
    for idx in range(1, len(onsets)):
        gap = onsets[idx] - onsets[idx - 1]
        if gap <= max_gap_s:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _linear_fit_offset_scale(behavior_times: np.ndarray, eeg_times: np.ndarray) -> Tuple[float, float]:
    """Return offset and scale such that eeg ≈ offset + scale * behavior."""
    behavior_times = np.asarray(behavior_times, dtype=float)
    eeg_times = np.asarray(eeg_times, dtype=float)
    if behavior_times.size == 0 or eeg_times.size == 0 or behavior_times.size != eeg_times.size:
        return 0.0, 1.0
    if behavior_times.size == 1:
        return float(eeg_times[0] - behavior_times[0]), 1.0
    # Solve least squares for y = a + b * x
    X = np.vstack([np.ones_like(behavior_times), behavior_times]).T
    sol, *_ = np.linalg.lstsq(X, eeg_times, rcond=None)
    offset = float(sol[0])
    scale = float(sol[1])
    return offset, scale


def _normalize_bool_like(val: Any) -> Optional[bool]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    text = str(val).strip().lower()
    if text in {"yes", "y", "true", "t", "1"}:
        return True
    if text in {"no", "n", "false", "f", "0"}:
        return False
    return None


def _apply_sequence_rules(
    pulses: List[Dict[str, Any]],
    sequence_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Classify grouped pulses into higher-level events according to config."""
    if not pulses or not sequence_cfg:
        return []

    max_gap = float(sequence_cfg.get("max_gap_s", 0.1))
    groups_cfg = list(sequence_cfg.get("groups", []))
    if not groups_cfg:
        return []

    pulses_sorted = sorted(pulses, key=lambda item: item["onset"])
    onsets = [float(p["onset"]) for p in pulses_sorted]
    grouped_indices = _group_by_gap(onsets, max_gap)
    total_groups = len(grouped_indices)

    events: List[Dict[str, Any]] = []
    for group_idx, indices in enumerate(grouped_indices):
        size = len(indices)
        match_rule: Optional[Dict[str, Any]] = None
        for rule in groups_cfg:
            min_size = int(rule.get("min_size", rule.get("size", 1)))
            max_size = int(rule.get("max_size", rule.get("size", size)))
            if min_size <= size <= max_size:
                position = str(rule.get("position", "")).lower()
                if position:
                    if position == "first" and group_idx != 0:
                        continue
                    if position == "last" and group_idx != (total_groups - 1):
                        continue
                    if position == "middle" and group_idx in (0, total_groups - 1):
                        continue
                if "group_index" in rule and group_idx != int(rule.get("group_index", group_idx)):
                    continue
                match_rule = rule
                break
        if match_rule is None:
            continue

        labels_cfg = match_rule.get("labels")
        if labels_cfg:
            labels = list(labels_cfg)
        else:
            base_label = match_rule.get("label") or match_rule.get("trial_type")
            labels = [base_label] * size

        values_cfg = match_rule.get("values")
        if values_cfg is not None:
            values = [int(v) if v is not None else None for v in list(values_cfg)]
        else:
            base_value = match_rule.get("value")
            values = [int(base_value) if base_value is not None else None] * size

        emit = str(match_rule.get("emit", "all")).lower()
        duration_override = match_rule.get("duration")

        for pos, pulse_index in enumerate(indices):
            if emit == "first" and pos > 0:
                break
            pulse = pulses_sorted[pulse_index]
            label = labels[pos] if pos < len(labels) else labels[-1]
            if not label:
                continue
            value = values[pos] if pos < len(values) else values[-1]
            event = {
                "sample": int(pulse["sample"]),
                "onset": float(pulse["onset"]),
                "duration": float(duration_override) if duration_override is not None else float(pulse["duration"]),
                "trial_type": str(label),
                "label": str(label),
                "value": int(value) if value is not None else int(pulse.get("ttl", 0)),
                "ttl": int(pulse.get("ttl", 0)),
            }
            events.append(event)
    return events


def _apply_sequence_aggregate(events: List[Dict[str, Any]], aggregate_cfg: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rename or drop events according to aggregate rules."""
    if not aggregate_cfg:
        return events

    aggregated: List[Dict[str, Any]] = []
    for event in events:
        updated = dict(event)
        drop_event = False
        for rule in aggregate_cfg:
            source = rule.get("source") or rule.get("label") or rule.get("trial_type")
            if source and updated.get("trial_type") != source:
                continue
            if bool(rule.get("drop", False)):
                drop_event = True
                break
            rename = rule.get("rename_to")
            if rename:
                updated["trial_type"] = str(rename)
                updated["label"] = str(rename)
            if rule.get("value") is not None:
                updated["value"] = int(rule["value"])
            if rule.get("duration") is not None:
                updated["duration"] = float(rule["duration"])
        if not drop_event:
            aggregated.append(updated)
    return aggregated


def _matches_conditions(row: pd.Series, conditions: Optional[Dict[str, Any]]) -> bool:
    if not conditions:
        return True
    for column, allowed in conditions.items():
        values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        allowed_normalized = {str(item).strip().lower() for item in values}
        val = row.get(column)
        if pd.isna(val):
            return False
        if str(val).strip().lower() not in allowed_normalized:
            return False
    return True


def _resolve_behavior_events_path(
    spec: RecordingSpec,
    config: Dict[str, Any],
) -> Optional[Path]:
    template = config.get("path_template") or config.get("path")
    if template:
        formatted = template.format(
            subject=spec.subject,
            session=spec.session,
            run=spec.run,
            task=spec.task,
        )
        path = Path(formatted)
        if not path.is_absolute():
            config_dir = Path(spec.trigger_rules.get("__config_dir__", spec.raw_path.parent))
            path = (config_dir / path).resolve()
        return path
    return _resolve_behavior_path(spec)


def _derive_behavior_events(
    spec: RecordingSpec,
    config: Dict[str, Any],
    existing_events: List[Dict[str, Any]],
    sfreq: float,
) -> List[Dict[str, Any]]:
    """Generate events from behavior tables once aligned to EEG clock."""
    path = _resolve_behavior_events_path(spec, config)
    if path is None or not path.exists():
        log_warn(f"Behavior file for derived events not found: {path}")
        return []

    read_kwargs = {}
    if config.get("delimiter"):
        read_kwargs["sep"] = str(config["delimiter"])
    if config.get("encoding"):
        read_kwargs["encoding"] = str(config["encoding"])

    try:
        beh_df = pd.read_csv(path, **read_kwargs)
    except Exception as exc:
        log_warn(f"Failed to read behavior file {path}: {exc}")
        return []

    # Optional row filtering
    filters = config.get("filter") or {}
    for column, allowed in filters.items():
        values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        allowed_normalized = {str(item).strip().lower() for item in values}
        beh_df = beh_df[beh_df[column].astype(str).str.strip().str.lower().isin(allowed_normalized)]

    if beh_df.empty:
        return []

    # Gather anchor pairs for alignment
    anchors = config.get("anchors", [])
    if not anchors:
        log_warn(f"behavior_derived_events: no anchors provided for {_spec_label(spec)}")
        return []

    event_time_map: Dict[str, List[float]] = defaultdict(list)
    for event in existing_events:
        label = str(event.get("trial_type"))
        event_time_map[label].append(float(event.get("onset", 0.0)))

    beh_times: List[float] = []
    eeg_times: List[float] = []
    for anchor in anchors:
        label = str(anchor.get("event_label") or anchor.get("event"))
        column = anchor.get("column") or anchor.get("behavior_column")
        if not label or not column:
            continue
        event_idx = int(anchor.get("event_index", 0))
        event_list = event_time_map.get(label, [])
        if not event_list:
            log_warn(f"behavior_derived_events: missing EEG anchor '{label}' for {_spec_label(spec)}")
            return []
        eeg_time = event_list[min(event_idx, len(event_list) - 1)]

        series = pd.to_numeric(beh_df[column], errors="coerce")
        valid_series = series.dropna()
        if valid_series.empty:
            log_warn(f"behavior_derived_events: column '{column}' has no valid values for {_spec_label(spec)}")
            return []
        beh_index = int(anchor.get("behavior_index", 0))
        beh_index = min(beh_index, len(valid_series) - 1)
        beh_time = float(valid_series.iloc[beh_index])

        beh_times.append(beh_time)
        eeg_times.append(eeg_time)

    offset, scale = _linear_fit_offset_scale(np.asarray(beh_times), np.asarray(eeg_times))

    outputs = config.get("outputs", [])
    if not outputs:
        return []

    default_stim = str(config.get("stim_channel", "behavior"))

    derived_events: List[Dict[str, Any]] = []
    for _, row in beh_df.iterrows():
        for output in outputs:
            column = output.get("column")
            if not column or column not in row:
                continue

            if not _matches_conditions(row, output.get("condition")):
                continue

            time_val = pd.to_numeric([row[column]], errors="coerce").item()
            if pd.isna(time_val):
                continue

            relative_column = output.get("relative_to_column")
            if relative_column:
                base_val = pd.to_numeric([row.get(relative_column)], errors="coerce").item()
                if pd.isna(base_val):
                    continue
                time_val = base_val + time_val

            time_val += float(output.get("time_offset", 0.0))

            eeg_time = offset + scale * float(time_val)
            label = output.get("label")

            label_map = output.get("label_map")
            if label_map:
                key_column = output.get("condition_column")
                if not key_column or key_column not in row:
                    continue
                key_value = str(row[key_column]).strip().lower()
                label_lookup = {str(k).strip().lower(): v for k, v in label_map.items()}
                label = label_lookup.get(key_value)
            if not label:
                continue

            value = output.get("value")
            value_map = output.get("value_map")
            if value_map:
                key_column = output.get("condition_column")
                if key_column and key_column in row:
                    key_value = str(row[key_column]).strip().lower()
                    vm = {str(k).strip().lower(): v for k, v in value_map.items()}
                    if key_value in vm:
                        value = vm[key_value]

            duration = float(output.get("duration", 0.0))
            if output.get("duration_column"):
                duration_val = pd.to_numeric([row.get(output["duration_column"])], errors="coerce").item()
                if not pd.isna(duration_val):
                    duration = float(duration_val)

            event = {
                "sample": int(round(eeg_time * sfreq)),
                "onset": float(eeg_time),
                "duration": duration,
                "trial_type": str(label),
                "label": str(label),
                "value": int(value) if value is not None else 0,
                "stim_channel": str(output.get("stim_channel", default_stim)),
            }
            derived_events.append(event)

    return derived_events


def apply_collapse_rules(
    events: List[Dict[str, Any]],
    collapse_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Collapse placeholder trial_types (e.g., long_tone) to ordered labels."""
    if not collapse_cfg:
        return events

    events_sorted = sorted(events, key=lambda item: item["sample"])
    for placeholder, slots in collapse_cfg.items():
        indices = [idx for idx, ev in enumerate(events_sorted) if ev.get("label") == placeholder]
        if not indices:
            continue

        slot_specs = {slot: _normalize_collapse_spec(spec) for slot, spec in slots.items()}

        def _apply_slot(index: int, slot_name: str) -> None:
            spec = slot_specs.get(slot_name)
            if spec is None:
                return
            trial_type = spec.get("trial_type")
            if trial_type is not None:
                events_sorted[index]["trial_type"] = trial_type
            if spec.get("code") is not None:
                events_sorted[index]["value"] = int(spec["code"])

        if len(indices) == 1:
            _apply_slot(indices[0], "first")
        else:
            _apply_slot(indices[0], "first")
            _apply_slot(indices[-1], "last")
            for mid_index in indices[1:-1]:
                _apply_slot(mid_index, "middle")

    # Drop events marked as ignore or missing trial_type
    return [
        ev
        for ev in events_sorted
        if ev.get("trial_type") not in {None, "", "ignore"}
    ]


def _resolve_sequence_path(seq_cfg: Dict[str, Any], spec: RecordingSpec, config_dir: Path) -> Optional[Path]:
    template = seq_cfg.get("path_template") or seq_cfg.get("path")
    if not template:
        return None
    formatted = template.format(
        subject=spec.subject,
        session=spec.session,
        run=spec.run,
        task=spec.task,
    )
    sequence_path = Path(formatted)
    if not sequence_path.is_absolute():
        sequence_path = (config_dir / sequence_path).resolve()
    return sequence_path


def _load_behavior_sequence(
    seq_cfg: Dict[str, Any],
    spec: RecordingSpec,
) -> List[Dict[str, Any]]:
    config_dir = Path(spec.trigger_rules.get("__config_dir__", spec.raw_path.parent))
    sequence_path = _resolve_sequence_path(seq_cfg, spec, config_dir)
    if sequence_path is None:
        log_warn(f"behavior_sequence missing 'path_template' for {_spec_label(spec)}; skipping alignment.")
        return []
    if not sequence_path.exists():
        log_warn(f"behavior_sequence file not found -> {sequence_path}")
        return []

    column = str(seq_cfg.get("column", "trigger"))
    try:
        beh_df = pd.read_csv(sequence_path)
    except Exception as exc:
        log_warn(f"Failed to read behavior_sequence CSV {sequence_path}: {exc}")
        return []

    if column not in beh_df.columns:
        log_warn(f"behavior_sequence column '{column}' missing in {sequence_path}; skipping alignment.")
        return []

    mapping: Dict[str, Dict[str, Any]] = {
        str(key): value for key, value in (seq_cfg.get("mapping") or {}).items()
    }
    drop_empty = bool(seq_cfg.get("drop_empty", True))
    keep_unknown = bool(seq_cfg.get("keep_unknown", False))

    sequence: List[Dict[str, Any]] = []
    for raw_value in beh_df[column]:
        if isinstance(raw_value, float) and np.isnan(raw_value):
            if drop_empty:
                continue
            raw_value = ""
        value_str = str(raw_value).strip()
        if value_str == "" and drop_empty:
            continue

        if value_str in mapping:
            spec_map = dict(mapping[value_str])
            label = spec_map.get("label")
            if label is None:
                log_warn(f"behavior_sequence mapping for {value_str!r} missing 'label'; skipping.")
                continue
            spec_map.setdefault("trial_type", label)
            spec_map["behavior_label"] = value_str
            sequence.append(spec_map)
        elif keep_unknown:
            sequence.append(
                {
                    "behavior_label": value_str,
                    "label": None,
                    "trial_type": value_str,
                    "code": None,
                }
            )

    return sequence


def align_events_to_behavior(
    events: List[Dict[str, Any]],
    seq_cfg: Dict[str, Any],
    spec: RecordingSpec,
) -> List[Dict[str, Any]]:
    behavior_sequence = _load_behavior_sequence(seq_cfg, spec)
    if not behavior_sequence:
        return []

    queues: Dict[Optional[str], deque] = defaultdict(deque)
    for event in sorted(events, key=lambda ev: ev["sample"]):
        queues[event.get("label")].append(event)

    assigned: List[Dict[str, Any]] = []
    last_onset = -np.inf

    for idx, beh in enumerate(behavior_sequence):
        label = beh.get("label")
        if label is None:
            log_warn(
                f"behavior_sequence entry #{idx} ({beh.get('behavior_label')!r}) missing matched label; skipping."
            )
            continue
        queue = queues.get(label)
        if not queue:
            log_warn(
                f"No remaining events for behavior label {beh.get('behavior_label')!r} mapped to {label!r} "
                f"for {_spec_label(spec)}; stopping alignment."
            )
            break
        event = queue.popleft()
        trial_type = beh.get("trial_type") or event.get("trial_type") or label
        assigned_event = dict(event)
        assigned_event["trial_type"] = trial_type
        if beh.get("code") is not None:
            try:
                assigned_event["value"] = int(beh["code"])
            except (TypeError, ValueError):
                pass
        assigned.append(assigned_event)
        last_onset = assigned_event["onset"]

    if len(assigned) < len(behavior_sequence):
        log_warn(
            f"Only matched {len(assigned)} / {len(behavior_sequence)} behavior events for {_spec_label(spec)}."
        )

    return assigned

def _extract_trigger_samples(raw: mne.io.BaseRaw, stim: str, spec: RecordingSpec) -> Tuple[np.ndarray, np.ndarray]:
    rules = spec.trigger_rules
    binary_edges = bool(rules.get("binary_edges", False))
    threshold = float(rules.get("binary_threshold", 0.5))
    direction = str(rules.get("edge_direction", "rise")).lower()

    if binary_edges:
        try:
            samples, values = _binary_edge_samples(raw, stim, threshold, direction=direction)
        except Exception as exc:
            log_warn(f"binary edge detection failed on {stim}: {exc}")
        else:
            if samples.size:
                return samples, values

    try:
        events = mne.find_events(
            raw,
            stim_channel=stim,
            shortest_event=1,
            consecutive=True,
            verbose=False,
        )
    except Exception as exc:
        log_warn(f"Failed to extract events from {stim}: {exc}")
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    if events.size == 0:
        log_warn(f"No discrete events detected on {stim} for {spec.raw_path.name}.")
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    return events[:, 0].astype(int), events[:, 2].astype(int)


def extract_trigger_events(raw: mne.io.BaseRaw, spec: RecordingSpec) -> pd.DataFrame:
    """Build an events table for downstream ``write_raw_bids`` consumption."""
    columns = ["onset", "duration", "trial_type", "value", "sample", "stim_channel"]
    stim_hint = spec.trigger_rules.get("stim_channel") or spec.stim_channel
    stim = detect_stim_channel(raw, stim_hint)
    if stim is None:
        log_warn(f"No stimulus channel detected for {spec.raw_path.name}; events.tsv will be empty.")
        return pd.DataFrame(columns=columns)

    rules = spec.trigger_rules
    rename_maps = _prepare_label_maps(rules.get("label_renames"))
    exact_map, casefold_map, normalized_map = rename_maps
    sfreq = float(raw.info.get("sfreq", 1.0))

    events: List[Dict[str, Any]] = []
    pulses: List[Dict[str, Any]] = []
    need_pulses = bool(
        rules.get("sequence_rules")
        or _use_pulse_width_rules(rules)
        or rules.get("behavior_derived_events")
    )

    if need_pulses:
        try:
            stim_data = raw.get_data(picks=[stim])[0]
        except Exception as exc:
            log_warn(f"Failed to read stim channel {stim}: {exc}")
        else:
            pulses = detect_pulses_from_stim(
                stim_data,
                sfreq,
                keep_values=rules.get("keep_values"),
                binary_edges=bool(rules.get("binary_edges", True)),
            )

    if pulses and rules.get("sequence_rules"):
        events.extend(_apply_sequence_rules(pulses, rules["sequence_rules"]))

    if pulses and _use_pulse_width_rules(rules):
        events.extend(
            classify_by_width(
                pulses,
                pulse_width_map=rules.get("pulse_width_map", []),
                ignore_windows=rules.get("ignore_windows", []),
                default_trial_type=rules.get("default_trial_type"),
            )
        )

    behavior_seq_cfg = rules.get("behavior_sequence")
    if behavior_seq_cfg and events:
        aligned_events = align_events_to_behavior(events, behavior_seq_cfg, spec)
        if aligned_events:
            events = aligned_events

    if events:
        if rules.get("sequence_aggregate"):
            events = _apply_sequence_aggregate(events, rules.get("sequence_aggregate"))

        if rules.get("collapse"):
            events = apply_collapse_rules(events, rules.get("collapse"))

        derived_cfg = rules.get("behavior_derived_events")
        if derived_cfg:
            derived_events = _derive_behavior_events(spec, derived_cfg, list(events), sfreq)
            if derived_events:
                events.extend(derived_events)

        if any(rename_maps):
            events = [_rename_event_labels(ev, rename_maps) for ev in events]

        if events:
            df = pd.DataFrame(
                {
                    "onset": [ev.get("onset", 0.0) for ev in events],
                    "duration": [ev.get("duration", 0.0) for ev in events],
                    "trial_type": [ev.get("trial_type") for ev in events],
                    "value": [int(ev.get("value", 0)) for ev in events],
                    "sample": [int(ev.get("sample", int(round(ev.get("onset", 0.0) * sfreq)))) for ev in events],
                    "stim_channel": [str(ev.get("stim_channel", stim)) for ev in events],
                },
                columns=columns,
            )

            drop_types = rules.get("drop_trial_types") or []
            if drop_types:
                drop_set = {str(item) for item in drop_types}
                df = df[~df["trial_type"].astype(str).isin(drop_set)]

            if any(rename_maps):
                df["trial_type"] = df["trial_type"].apply(
                    lambda value: _apply_label_map(value, exact_map, casefold_map, normalized_map)
                )

            df = df.sort_values("sample", kind="stable").drop_duplicates(subset=["sample", "trial_type"], keep="first")
            return df.reset_index(drop=True)

    samples, values = _extract_trigger_samples(raw, stim, spec)
    if samples.size == 0:
        return pd.DataFrame(columns=columns)

    onsets = samples.astype(float) / sfreq

    values, keep = _apply_trigger_rules(values, rules)
    if not np.any(keep):
        return pd.DataFrame(columns=columns)

    onsets = onsets[keep]
    values = values[keep]
    samples = samples[keep]
    trial_types = _trial_types_from_rules(values, rules)

    df = pd.DataFrame(
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
    if any(rename_maps):
        df["trial_type"] = df["trial_type"].apply(
            lambda value: _apply_label_map(value, exact_map, casefold_map, normalized_map)
        )
    drop_types = rules.get("drop_trial_types") or []
    if drop_types:
        drop_set = {str(item) for item in drop_types}
        df = df[~df["trial_type"].astype(str).isin(drop_set)]
    return df


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
    if "duration" in df.columns:
        event_metadata["duration"] = df["duration"].astype(float)
    if "value" in df.columns:
        event_metadata["value"] = df["value"].astype(int)

    extra_desc: Dict[str, str] = {}
    if "value" in event_metadata.columns:
        extra_desc["value"] = "Integer code assigned by trigger rules."

    return events, event_id, event_metadata, extra_desc


def _write_events_metadata(
    bids_path: "BIDSPath",
    events_df: pd.DataFrame,
    extra_desc: Optional[Dict[str, str]] = None,
) -> None:
    """Write the *_events.json sidecar describing event columns."""
    if not _HAS_MNE_BIDS or events_df.empty:
        return

    try:
        events_json_path = bids_path.copy().update(suffix="events", extension=".json")
    except Exception as exc:
        log_warn(f"Failed to prepare events sidecar path: {exc}")
        return

    metadata: Dict[str, Any] = {
        "onset": {"Description": "Event onset in seconds from recording start."},
        "duration": {"Description": "Pulse width derived from Trigger TTL."},
        "trial_type": {
            "Description": "Event label assigned by configured trigger rules.",
            "Levels": {},
        },
    }
    extra_desc = extra_desc or {}
    if "value" in events_df.columns:
        metadata["value"] = {
            "Description": extra_desc.get(
                "value",
                "Integer code assigned by trigger rules.",
            )
        }

    for trial_type in events_df["trial_type"].astype(str).unique():
        metadata["trial_type"]["Levels"][trial_type] = f"Pulse classified as '{trial_type}'."

    try:
        events_json_path.fpath.parent.mkdir(parents=True, exist_ok=True)
        events_json_path.fpath.write_text(json.dumps(metadata, indent=2))
    except Exception as exc:
        log_warn(f"Failed to write events metadata: {exc}")


def _write_events_tsv(
    bids_path: "BIDSPath",
    events_df: pd.DataFrame,
) -> None:
    """Overwrite *_events.tsv with the classified trigger events."""
    if events_df.empty:
        return
    try:
        events_tsv_path = bids_path.copy().update(suffix="events", extension=".tsv")
    except Exception as exc:
        log_warn(f"Failed to prepare events.tsv path: {exc}")
        return

    ordered_cols = [
        "onset",
        "duration",
        "trial_type",
        "value",
        "sample",
        "stim_channel",
    ]
    df = events_df.copy()
    for col in ordered_cols:
        if col not in df.columns:
            if col in {"onset", "duration"}:
                df[col] = 0.0
            else:
                df[col] = ""

    df = df[ordered_cols]
    df = df.sort_values("onset", kind="stable")

    try:
        events_tsv_path.fpath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(events_tsv_path.fpath, sep="\t", index=False, float_format="%.6f")
    except Exception as exc:
        log_warn(f"Failed to write events.tsv: {exc}")


def maybe_update_eeg_sidecar(
    bids_path: "BIDSPath",
    ref: Optional[str],
    ground: Optional[str],
    manufacturer: Optional[str] = DEFAULT_MANUFACTURER,
) -> None:
    """Optionally set EEGReference/EEGGround/Manufacturer in *_eeg.json."""
    if not _HAS_MNE_BIDS:
        return
    entries: Dict[str, Any] = {}
    if manufacturer:
        entries["Manufacturer"] = manufacturer
    if ref:
        entries["EEGReference"] = ref
    if ground:
        entries["EEGGround"] = ground
    if not entries:
        return
    try:
        sidecar_path = bids_path.copy().update(suffix="eeg", extension=".json", datatype="eeg")
    except Exception as exc:
        log_warn(f"Failed to prepare EEG sidecar path: {exc}")
        return
    try:
        update_sidecar_json(sidecar_path, entries)  # type: ignore[misc]
    except Exception as exc:
        log_warn(f"Failed to update EEG sidecar fields: {exc}")




def _load_behavior_table(behavior: BehaviorConfig, spec: RecordingSpec) -> Optional[pd.DataFrame]:
    if not behavior.source_path.exists():
        log_warn(
            f"Behavioral file not found for {_spec_label(spec)}: {behavior.source_path}"
        )
        return None
    try:
        return pd.read_csv(behavior.source_path, sep="	")
    except Exception as exc:
        log_warn(f"Failed to read behavioral events TSV ({behavior.source_path}): {exc}")
        return None


def _get_frame_on_times(trigger_events: pd.DataFrame, spec: RecordingSpec) -> np.ndarray:
    frame_on = trigger_events[trigger_events["trial_type"] == "frame_on"]
    if frame_on.empty:
        log_warn(
            f"No 'frame_on' events available for {_spec_label(spec)}; cannot align behavioral file."
        )
        return np.empty(0, dtype=float)
    return frame_on["onset"].to_numpy()


def _select_anchor_types(trial_types: pd.Series, behavior: BehaviorConfig) -> List[str]:
    if behavior.anchor_trial_types:
        anchors = [tt for tt in behavior.anchor_trial_types if tt in trial_types.values]
    else:
        anchors = list(dict.fromkeys(trial_types))
    return anchors

def process_behavioral_events(spec: RecordingSpec, trigger_events: pd.DataFrame, bids_path: 'BIDSPath') -> None:
    behavior = spec.behavior
    if behavior is None or not behavior.enabled:
        _auto_sync_behavior_using_events(spec, trigger_events, bids_path)
        return

    beh_df = _load_behavior_table(behavior, spec)
    if beh_df is None:
        return

    required = {
        behavior.onset_column,
        behavior.duration_column,
        behavior.trial_type_column,
    }
    missing_cols = [col for col in required if col not in beh_df.columns]
    if missing_cols:
        log_warn(
            f"Behavioral TSV missing required columns {missing_cols} -> {behavior.source_path}; skipping alignment."
        )
        return

    frame_on_times = _get_frame_on_times(trigger_events, spec)
    if frame_on_times.size == 0:
        return

    beh_df = beh_df.copy()
    try:
        beh_df[behavior.onset_column] = beh_df[behavior.onset_column].astype(float)
    except Exception as exc:
        log_warn(
            f"Behavioral onset column '{behavior.onset_column}' could not be converted to float: {exc}; skipping."
        )
        return

    trial_types_series = beh_df[behavior.trial_type_column].astype(str)
    anchor_types = _select_anchor_types(trial_types_series, behavior)
    if not anchor_types:
        log_warn(
            f"No behavioral anchor trial types available for {_spec_label(spec)}; skipping alignment."
        )
        _maybe_write_behavior_copy(behavior, beh_df, beh_df, bids_path, None, corrected=False)
        return

    anchors_df = beh_df[trial_types_series.isin(anchor_types)].copy()
    if anchors_df.empty:
        log_warn(
            f"Behavioral anchors (trial_types={anchor_types}) not present for {_spec_label(spec)}."
        )
        _maybe_write_behavior_copy(behavior, beh_df, beh_df, bids_path, None, corrected=False)
        return

    align_records = _align_anchors_to_frames(
        anchors_df,
        behavior,
        frame_on_times,
    )
    if not align_records:
        log_warn(
            f"Failed to align behavioral anchors to frame_on events for {_spec_label(spec)}."
        )
        _maybe_write_behavior_copy(behavior, beh_df, beh_df, bids_path, None, corrected=False)
        return

    drift_info = _estimate_behavior_drift(align_records, behavior)
    corrected_df = beh_df.copy()
    corrected_df[behavior.onset_column] = _apply_behavior_correction(
        beh_df[behavior.onset_column].to_numpy(),
        drift_info,
    )

    _log_behavior_summary(spec, behavior, align_records, drift_info)
    _maybe_write_behavior_copy(behavior, beh_df, corrected_df, bids_path, drift_info, corrected=True)



def _align_anchors_to_frames(
    anchors_df: pd.DataFrame,
    behavior: BehaviorConfig,
    frame_on_times: np.ndarray,
) -> List[Dict[str, Any]]:
    frame_on_times = np.asarray(frame_on_times, dtype=float)
    if frame_on_times.size == 0:
        return []

    records: List[Dict[str, Any]] = []
    for idx, row in anchors_df.iterrows():
        onset = float(row[behavior.onset_column])
        trial_type = str(row[behavior.trial_type_column])
        matched_onset = _find_frame_on_onset(onset, frame_on_times, behavior.sync_method)
        if matched_onset is None:
            continue
        delta = matched_onset - onset
        records.append(
            {
                "index": idx,
                "trial_type": trial_type,
                "behavior_onset": onset,
                "frame_on_onset": float(matched_onset),
                "delta": float(delta),
            }
        )
    return records


def _find_frame_on_onset(
    onset: float,
    frame_on_times: np.ndarray,
    method: str,
) -> Optional[float]:
    idx = np.searchsorted(frame_on_times, onset, side="left")
    if method == "next_frame_on":
        if idx >= frame_on_times.size:
            return None
        return float(frame_on_times[idx])

    # nearest_frame_on
    candidates: List[Tuple[float, float]] = []
    if idx < frame_on_times.size:
        cand = float(frame_on_times[idx])
        candidates.append((abs(cand - onset), cand))
    if idx > 0:
        cand = float(frame_on_times[idx - 1])
        candidates.append((abs(cand - onset), cand))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _estimate_behavior_drift(
    align_records: List[Dict[str, Any]],
    behavior: BehaviorConfig,
) -> Dict[str, Any]:
    align_records_sorted = sorted(align_records, key=lambda rec: rec["behavior_onset"])
    deltas = np.array([rec["delta"] for rec in align_records_sorted], dtype=float)
    onset_vals = np.array([rec["behavior_onset"] for rec in align_records_sorted], dtype=float)

    per_label: Dict[str, Dict[str, Any]] = {}
    for rec in align_records_sorted:
        label = rec["trial_type"]
        stats = per_label.setdefault(label, {"count": 0, "deltas": []})
        stats["count"] += 1
        stats["deltas"].append(rec["delta"])

    for label, stats in per_label.items():
        deltas_arr = np.array(stats["deltas"], dtype=float)
        stats["mean"] = float(deltas_arr.mean())
        stats["std"] = float(deltas_arr.std(ddof=1)) if deltas_arr.size > 1 else 0.0
        stats["min"] = float(deltas_arr.min())
        stats["max"] = float(deltas_arr.max())
        del stats["deltas"]

    overall_stats = {
        "count": int(deltas.size),
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
        "min": float(deltas.min()),
        "max": float(deltas.max()),
    }

    drift_model = _compute_behavior_model(behavior.alignment_strategy, align_records_sorted)

    return {
        "overall_stats": overall_stats,
        "per_label": per_label,
        "model": drift_model,
        "records": align_records_sorted,
        "strategy": drift_model["type"],
    }


def _build_linear_model(onsets: np.ndarray, deltas: np.ndarray) -> Optional[Dict[str, Any]]:
    if onsets.size < 2:
        return None
    reference = float(onsets.mean())
    centered = onsets - reference
    try:
        slope, intercept = np.polyfit(centered, deltas, 1)
    except Exception as exc:
        log_warn(f"Linear drift fit failed ({exc}); reverting to mean offset.")
        return None
    return {
        "type": "linear",
        "offset": float(intercept),
        "slope": float(slope),
        "reference": reference,
    }


def _build_piecewise_model(align_records_sorted: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(align_records_sorted) < 2:
        return None

    segments: List[Dict[str, float]] = []
    for current, nxt in zip(align_records_sorted, align_records_sorted[1:]):
        b_start = float(current["behavior_onset"])
        b_end = float(nxt["behavior_onset"])
        f_start = float(current["frame_on_onset"])
        f_end = float(nxt["frame_on_onset"])
        slope = 1.0 if b_end == b_start else (f_end - f_start) / (b_end - b_start)
        intercept = f_start - slope * b_start
        segments.append(
            {
                "behavior_start": b_start,
                "behavior_end": b_end,
                "frame_start": f_start,
                "frame_end": f_end,
                "slope": slope,
                "intercept": intercept,
            }
        )

    if not segments:
        return None

    pre = {
        "slope": segments[0]["slope"],
        "intercept": segments[0]["intercept"],
        "behavior_end": segments[0]["behavior_start"],
    }
    post = {
        "slope": segments[-1]["slope"],
        "intercept": segments[-1]["intercept"],
        "behavior_start": segments[-1]["behavior_end"],
    }
    return {
        "type": "piecewise",
        "segments": segments,
        "pre": pre,
        "post": post,
    }

def _compute_behavior_model(strategy: str, align_records_sorted: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not align_records_sorted:
        return {"type": "mean_offset", "offset": 0.0}

    deltas = np.array([rec["delta"] for rec in align_records_sorted], dtype=float)
    onsets = np.array([rec["behavior_onset"] for rec in align_records_sorted], dtype=float)

    if strategy == "first_anchor":
        offset = float(align_records_sorted[0]["delta"])
        return {"type": "first_anchor", "offset": offset}

    if strategy == "linear":
        model = _build_linear_model(onsets, deltas)
        if model is not None:
            return model
        offset = float(deltas.mean())
        return {"type": "mean_offset", "offset": offset}

    if strategy == "piecewise":
        model = _build_piecewise_model(align_records_sorted)
        if model is not None:
            return model

    offset = float(deltas.mean())
    return {"type": "mean_offset", "offset": offset}



def _apply_behavior_correction(onsets: np.ndarray, drift_info: Dict[str, Any]) -> np.ndarray:
    model = drift_info["model"]
    strategy = model.get("type", "mean_offset")
    onsets = np.asarray(onsets, dtype=float)

    if strategy == "first_anchor" or strategy == "mean_offset":
        offset = float(model.get("offset", 0.0))
        return onsets + offset

    if strategy == "linear":
        offset = float(model.get("offset", 0.0))
        slope = float(model.get("slope", 0.0))
        reference = float(model.get("reference", 0.0))
        return onsets + offset + slope * (onsets - reference)

    if strategy == "piecewise":
        segments = model.get("segments", [])
        if not segments:
            return onsets
        behavior_breaks = [seg["behavior_end"] for seg in segments]
        behavior_starts = [seg["behavior_start"] for seg in segments]
        pre = model.get("pre", {})
        post = model.get("post", {})

        corrected = np.empty_like(onsets)
        for idx, t in enumerate(onsets):
            if t < behavior_starts[0]:
                slope = float(pre.get("slope", segments[0]["slope"]))
                intercept = float(pre.get("intercept", segments[0]["intercept"]))
            elif t >= behavior_breaks[-1]:
                slope = float(post.get("slope", segments[-1]["slope"]))
                intercept = float(post.get("intercept", segments[-1]["intercept"]))
            else:
                seg_idx = np.searchsorted(behavior_breaks, t, side="right")
                seg = segments[min(seg_idx, len(segments) - 1)]
                slope = seg["slope"]
                intercept = seg["intercept"]
            corrected[idx] = slope * t + intercept
        return corrected

    return onsets


def _maybe_write_behavior_copy(
    behavior: BehaviorConfig,
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    bids_path: 'BIDSPath',
    drift_info: Optional[Dict[str, Any]],
    corrected: bool,
) -> None:
    if not behavior.output_beh:
        return

    beh_bids_path = bids_path.copy().update(datatype="beh", suffix="beh", extension=".tsv")
    beh_dir = Path(beh_bids_path.directory)
    beh_dir.mkdir(parents=True, exist_ok=True)

    output_df = corrected_df if corrected else original_df
    output_df.to_csv(beh_bids_path.fpath, sep="\t", index=False, float_format="%.9f")

    if behavior.write_sidecar:
        sidecar_path = beh_bids_path.copy().update(extension=".json")
        payload = _build_behavior_sidecar(behavior, drift_info, bids_path, corrected)
        Path(sidecar_path.directory).mkdir(parents=True, exist_ok=True)
        Path(sidecar_path.fpath).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_behavior_sidecar(
    behavior: BehaviorConfig,
    drift_info: Optional[Dict[str, Any]],
    bids_path: 'BIDSPath',
    corrected: bool,
) -> Dict[str, Any]:
    relative_eeg_events = None
    root_path = Path(bids_path.root).resolve() if bids_path.root else None
    try:
        eeg_events_path = Path(bids_path.copy().update(suffix="events", extension=".tsv").fpath).resolve()
        if root_path:
            relative_eeg_events = str(eeg_events_path.relative_to(root_path))
        else:
            relative_eeg_events = str(eeg_events_path)
    except Exception:
        relative_eeg_events = str(bids_path.copy().update(suffix="events", extension=".tsv").fpath)

    payload: Dict[str, Any] = {
        "BehavioralSourceFile": behavior.source_label,
        "SynchronizationMethod": behavior.sync_method,
        "AlignmentStrategy": behavior.alignment_strategy,
        "CorrectionApplied": corrected,
        "EEGEventsFile": relative_eeg_events,
    }

    if drift_info:
        payload["AnchorStatistics"] = drift_info["per_label"]
        payload["OverallDeltaStatistics"] = drift_info["overall_stats"]
        payload["DriftModel"] = drift_info["model"]
        payload["AnchorsCount"] = len(drift_info["records"])
    else:
        payload["AnchorStatistics"] = {}
        payload["OverallDeltaStatistics"] = {}
        payload["DriftModel"] = {"type": "none"}
        payload["AnchorsCount"] = 0

    return payload


def _log_behavior_summary(
    spec: RecordingSpec,
    behavior: BehaviorConfig,
    align_records: List[Dict[str, Any]],
    drift_info: Dict[str, Any],
) -> None:
    overall = drift_info["overall_stats"]
    model = drift_info["model"]
    strategy = drift_info.get("strategy", model.get("type", "mean_offset"))
    if strategy == "linear":
        model_desc = (
            f"linear (offset={model.get('offset', 0.0):.4f}s"
            f" slope={model.get('slope', 0.0):.6f}s/s)"
        )
    elif strategy == "piecewise":
        segments = model.get("segments", [])
        model_desc = f"piecewise (segments={len(segments)})"
    elif strategy == "first_anchor":
        model_desc = f"first_anchor (offset={model.get('offset', 0.0):.4f}s)"
    else:
        model_desc = f"mean_offset (offset={model.get('offset', 0.0):.4f}s)"
    log_info(
        "[behavior] "
        f"sub-{spec.subject}_ses-{spec.session}_run-{spec.run}: anchors={overall['count']} mean_delta={overall['mean']:.4f}s std={overall['std']:.4f}s model={model_desc}"
    )


def _prepare_label_maps(mapping: Any) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    exact: Dict[str, str] = {}
    casefold_map: Dict[str, str] = {}
    normalized: Dict[str, str] = {}
    if isinstance(mapping, dict):
        for raw_key, raw_value in mapping.items():
            if raw_key is None or raw_value is None:
                continue
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or not value:
                continue
            exact[key] = value
            casefold_map[key.casefold()] = value
            norm_key = _normalize_event_label(key)
            if norm_key:
                normalized[norm_key] = value
    return exact, casefold_map, normalized


def _apply_label_map(
    value: Any,
    exact_map: Dict[str, str],
    casefold_map: Dict[str, str],
    normalized_map: Dict[str, str],
) -> Any:
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except Exception:
        pass
    text = str(value)
    if text in exact_map:
        return exact_map[text]
    lowered = text.casefold()
    if lowered in casefold_map:
        return casefold_map[lowered]
    normalized = _normalize_event_label(text)
    if normalized in normalized_map:
        return normalized_map[normalized]
    return value


def _rename_event_labels(
    event: Dict[str, Any],
    rename_maps: Tuple[Dict[str, str], Dict[str, str], Dict[str, str]],
) -> Dict[str, Any]:
    exact_map, casefold_map, normalized_map = rename_maps
    if not (exact_map or casefold_map or normalized_map):
        return event
    updated = dict(event)
    for key in ("label", "trial_type"):
        if key in updated:
            updated[key] = _apply_label_map(updated[key], exact_map, casefold_map, normalized_map)
    return updated


def _resolve_behavior_path(spec: RecordingSpec) -> Optional[Path]:
    raw_path = spec.raw_path
    parent = raw_path.parent
    base_stem = raw_path.stem
    candidate_stems = [
        base_stem,
        base_stem.replace("_raw", ""),
        base_stem.replace("_eeg", ""),
        f"sub-{spec.subject}_ses-{spec.session}_task-{spec.task}_run-{spec.run}",
    ]

    seen: set[str] = set()
    for stem in candidate_stems:
        stem = stem.strip()
        if not stem or stem in seen:
            continue
        seen.add(stem)
        for ext in (".csv", ".tsv"):
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate

    for ext in (".csv", ".tsv"):
        pattern = f"sub-{spec.subject}_ses-{spec.session}_task-{spec.task}_run-{spec.run}*{ext}"
        matches = sorted(parent.glob(pattern))
        if matches:
            return matches[0]

    return None


def _auto_sync_behavior_using_events(
    spec: RecordingSpec,
    trigger_events: pd.DataFrame,
    bids_path: "BIDSPath",
) -> None:
    if trigger_events.empty:
        return

    rename_maps = _prepare_label_maps(spec.trigger_rules.get("label_renames"))
    exact_map, casefold_map, normalized_map = rename_maps

    behavior_path = _resolve_behavior_path(spec)
    if behavior_path is None:
        return

    try:
        beh_df = pd.read_csv(behavior_path)
    except Exception as exc:
        log_warn(
            f"[behavior-auto] Failed to read behavioral CSV for {_spec_label(spec)}: {exc}"
        )
        return

    label_column = _find_behavior_label_column(beh_df)
    if label_column is None:
        log_warn(
            f"[behavior-auto] Could not identify an event label column in {behavior_path}; skipping."
        )
        return

    time_columns = _candidate_behavior_time_columns(beh_df)
    if not time_columns:
        log_warn(
            f"[behavior-auto] No timing columns detected in {behavior_path}; skipping synchronization."
        )
        return

    eeg_groups = _collect_eeg_event_groups(trigger_events, rename_maps)
    if not eeg_groups:
        log_warn(
            f"[behavior-auto] No classified EEG events available to anchor {_spec_label(spec)}."
        )
        return

    behavior_groups = _collect_behavior_event_groups(
        beh_df,
        label_column,
        time_columns,
        rename_maps,
    )
    if not behavior_groups:
        log_warn(
            f"[behavior-auto] Failed to extract behavioral anchors from column '{label_column}' in {behavior_path}."
        )
        return

    common_keys = [key for key in eeg_groups if key in behavior_groups]
    if not common_keys:
        log_warn(
            f"[behavior-auto] No overlapping event labels between EEG and behavior for {_spec_label(spec)}."
        )
        return

    matched_behavior: List[float] = []
    matched_eeg: List[float] = []
    anchor_records: List[Dict[str, Any]] = []
    matches_per_label: Dict[str, int] = {}

    for key in common_keys:
        eeg_group = eeg_groups[key]
        beh_group = behavior_groups[key]
        onsets = eeg_group["onsets"]
        records = beh_group["records"]
        count = min(len(onsets), len(records))
        if count == 0:
            continue
        matches_per_label[eeg_group["label"]] = count
        for idx in range(count):
            rec = records[idx]
            behavior_time = float(rec["time"])
            eeg_onset = float(onsets[idx])
            matched_behavior.append(behavior_time)
            matched_eeg.append(eeg_onset)
            anchor_records.append(
                {
                    "label": eeg_group["label"],
                    "behavior_index": int(rec["index"]),
                    "behavior_time": behavior_time,
                    "eeg_onset": eeg_onset,
                    "time_column": rec.get("column"),
                }
            )

    if not matched_behavior:
        log_warn(
            f"[behavior-auto] Unable to build anchor pairs for {_spec_label(spec)}; skipping."
        )
        return

    behavior_array = np.asarray(matched_behavior, dtype=float)
    eeg_array = np.asarray(matched_eeg, dtype=float)
    model = _fit_behavior_alignment_model(behavior_array, eeg_array)
    if model is None:
        log_warn(
            f"[behavior-auto] Failed to fit synchronization model for {_spec_label(spec)}."
        )
        return

    predictions = _apply_model_to_array(behavior_array, model)
    residuals = eeg_array - predictions
    for record, resid in zip(anchor_records, residuals):
        record["delta"] = float(resid)

    residual_stats_original = {
        "count": int(residuals.size),
        "mean": float(np.mean(residuals)) if residuals.size else 0.0,
        "std": float(np.std(residuals)) if residuals.size else 0.0,
        "max_abs": float(np.max(np.abs(residuals))) if residuals.size else 0.0,
    }

    corrected_df = _apply_behavior_time_transform(beh_df, time_columns, model)
    if label_column in corrected_df.columns:
        corrected_df[label_column] = corrected_df[label_column].apply(
            lambda value: _apply_label_map(value, exact_map, casefold_map, normalized_map)
        )

    if "eeg_onset" not in corrected_df.columns:
        corrected_df["eeg_onset"] = np.nan

    adjusted_residuals: List[float] = []
    for record in anchor_records:
        idx = record.get("behavior_index")
        eeg_onset = record.get("eeg_onset")
        column = record.get("time_column")
        if idx is None or eeg_onset is None:
            continue
        corrected_delta = float(record.get("delta", 0.0))
        if column and column in corrected_df.columns:
            corrected_df.at[idx, column] = eeg_onset
            corrected_delta = float(eeg_onset - corrected_df.at[idx, column])
        corrected_df.at[idx, "eeg_onset"] = eeg_onset
        record["corrected_delta"] = corrected_delta
        adjusted_residuals.append(corrected_delta)

    residual_stats_corrected = {
        "count": int(len(adjusted_residuals)),
        "mean": float(np.mean(adjusted_residuals)) if adjusted_residuals else 0.0,
        "std": float(np.std(adjusted_residuals)) if adjusted_residuals else 0.0,
        "max_abs": float(np.max(np.abs(adjusted_residuals))) if adjusted_residuals else 0.0,
    }

    time_columns_out = list(time_columns)
    if "eeg_onset" not in time_columns_out:
        time_columns_out.append("eeg_onset")

    summary = {
        "label_column": label_column,
        "time_columns": time_columns_out,
        "model": model,
        "anchors": anchor_records,
        "matches_per_label": matches_per_label,
        "residuals": {
            "original": residual_stats_original,
            "corrected": residual_stats_corrected,
        },
        "behavior_path": str(behavior_path),
    }

    _write_auto_behavior(behavior_path, corrected_df, summary, bids_path)

    log_info(
        "[behavior-auto] "
        f"sub-{spec.subject}_ses-{spec.session}_run-{spec.run}: "
        f"anchors={residual_stats_corrected['count']} "
        f"mean_delta={residual_stats_corrected['mean']:.4f}s "
        f"std={residual_stats_corrected['std']:.4f}s model={model['type']}"
    )


_BEHAVIOR_TIME_COLUMN_HINTS = (
    "started",
    "stopped",
    "time",
    "onset",
    "offset",
    "latency",
    "rt",
    "timeson",
    "timesoff",
)

_LABEL_COLUMN_CANDIDATES = ("trigger", "event", "trial_type", "event_type", "condition")
_LABEL_SANITIZER = re.compile(r"[^a-z0-9]+")


def _find_behavior_label_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in _LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        sample = series.dropna().astype(str)
        unique = sample.unique()
        if 0 < len(unique) <= 64:
            return column
    return None


def _candidate_behavior_time_columns(df: pd.DataFrame) -> List[str]:
    columns: List[str] = []
    for column in df.columns:
        lowered = column.lower()
        if not any(hint in lowered for hint in _BEHAVIOR_TIME_COLUMN_HINTS):
            continue
        series = df[column]
        if series.dropna().empty:
            continue
        if any(_coerce_single_time_value(value) is not None for value in series.dropna().head(10)):
            columns.append(column)
    return _prioritize_time_columns(columns)


def _prioritize_time_columns(columns: Iterable[str]) -> List[str]:
    priority_order = [
        "trial.started",
        "trial.onset",
        "trial.stopped",
        "trial.offset",
        "start_end_task_trigger.started",
        "start_end_task_trigger.stopped",
        "landscape_gate.started",
        "landscape_gate.stopped",
        "scoreboard.started",
        "scoreboard.stopped",
        "wait.started",
        "wait.stopped",
        "mouse.time",
        "timeson",
        "timesoff",
    ]

    def score(column: str) -> Tuple[int, str]:
        lowered = column.lower()
        for idx, key in enumerate(priority_order):
            if key in lowered:
                return idx, lowered
        return len(priority_order), lowered

    return sorted(columns, key=score)


def _collect_eeg_event_groups(
    events_df: pd.DataFrame,
    rename_maps: Tuple[Dict[str, str], Dict[str, str], Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    exact_map, casefold_map, normalized_map = rename_maps
    groups: Dict[str, Dict[str, Any]] = {}
    if events_df.empty or "trial_type" not in events_df.columns:
        return groups
    valid = events_df[events_df["trial_type"].notna()]
    for label, group in valid.groupby("trial_type"):
        canonical = _apply_label_map(label, exact_map, casefold_map, normalized_map)
        normalized = _normalize_event_label(canonical)
        if not normalized:
            continue
        groups[normalized] = {
            "label": str(canonical),
            "onsets": np.sort(group["onset"].astype(float).to_numpy()),
        }
    return groups


def _collect_behavior_event_groups(
    df: pd.DataFrame,
    label_column: str,
    time_columns: List[str],
    rename_maps: Tuple[Dict[str, str], Dict[str, str], Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    exact_map, casefold_map, normalized_map = rename_maps
    groups: Dict[str, Dict[str, Any]] = {}
    for index, row in df.iterrows():
        label_value = row.get(label_column)
        canonical = _apply_label_map(label_value, exact_map, casefold_map, normalized_map)
        normalized = _normalize_event_label(canonical)
        if not normalized:
            continue
        time_value, time_column = _extract_behavior_time_from_row(row, time_columns)
        if time_value is None:
            continue
        groups.setdefault(normalized, {"label": str(canonical), "records": []})
        groups[normalized]["records"].append(
            {"index": index, "time": float(time_value), "column": time_column}
        )

    for info in groups.values():
        info["records"].sort(key=lambda item: item["time"])

    return groups


def _normalize_event_label(label: Any) -> str:
    if label is None:
        return ""
    text = str(label).strip().lower()
    return _LABEL_SANITIZER.sub("", text)


def _extract_behavior_time_from_row(
    row: pd.Series,
    time_columns: Iterable[str],
) -> Tuple[Optional[float], Optional[str]]:
    for column in time_columns:
        if column not in row:
            continue
        value = row[column]
        time_value = _coerce_single_time_value(value)
        if time_value is not None:
            return time_value, column
    return None, None


def _coerce_single_time_value(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "nan":
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                return None
            return _coerce_single_time_value(parsed)
        try:
            return float(stripped)
        except ValueError:
            return None
    if isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            coerced = _coerce_single_time_value(item)
            if coerced is not None:
                return coerced
    return None


def _fit_behavior_alignment_model(behavior_times: np.ndarray, eeg_times: np.ndarray) -> Optional[Dict[str, Any]]:
    if behavior_times.size == 0 or eeg_times.size == 0:
        return None
    if behavior_times.size == 1:
        offset = float(eeg_times[0] - behavior_times[0])
        return {"type": "offset", "scale": 1.0, "offset": offset}
    try:
        A = np.column_stack([behavior_times, np.ones_like(behavior_times)])
        scale, offset = np.linalg.lstsq(A, eeg_times, rcond=None)[0]
    except Exception:
        return None
    return {"type": "linear", "scale": float(scale), "offset": float(offset)}


def _apply_model_to_array(values: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    scale = float(model.get("scale", 1.0))
    offset = float(model.get("offset", 0.0))
    return scale * values + offset


def _apply_model_to_scalar(value: Any, model: Dict[str, Any]) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    scale = float(model.get("scale", 1.0))
    offset = float(model.get("offset", 0.0))
    result = scale * numeric + offset
    if np.isnan(result):
        return float("nan")
    return result


def _transform_time_value(value: Any, model: Dict[str, Any]) -> Any:
    if value is None:
        return value
    if isinstance(value, (float, int, np.floating, np.integer)):
        if not np.isfinite(value):
            return value
        return _apply_model_to_scalar(float(value), model)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "nan":
            return np.nan
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                return value
            adjusted = _transform_time_sequence(parsed, model)
            return json.dumps(adjusted)
        try:
            return _apply_model_to_scalar(float(stripped), model)
        except ValueError:
            return value
    if isinstance(value, (list, tuple, np.ndarray)):
        adjusted = _transform_time_sequence(value, model)
        return json.dumps(adjusted)
    return value


def _transform_time_sequence(sequence: Any, model: Dict[str, Any]) -> List[Any]:
    if isinstance(sequence, np.ndarray):
        iterable = sequence.tolist()
    else:
        iterable = list(sequence)
    adjusted: List[Any] = []
    for item in iterable:
        adjusted.append(_apply_model_to_scalar(item, model))
    return adjusted


def _apply_behavior_time_transform(
    df: pd.DataFrame,
    time_columns: Iterable[str],
    model: Dict[str, Any],
) -> pd.DataFrame:
    corrected = df.copy()
    for column in time_columns:
        corrected[column] = corrected[column].apply(lambda value: _transform_time_value(value, model))
    return corrected


def _write_auto_behavior(
    source_path: Path,
    corrected_df: pd.DataFrame,
    summary: Dict[str, Any],
    bids_path: "BIDSPath",
) -> None:
    try:
        beh_bids_path = bids_path.copy().update(datatype="beh", suffix="beh", extension=".tsv")
    except Exception as exc:
        log_warn(f"[behavior-auto] Failed to prepare BIDS behavior path: {exc}")
        return

    output_dir = Path(beh_bids_path.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_df.to_csv(beh_bids_path.fpath, sep="\t", index=False, float_format="%.9f")

    try:
        sidecar_path = beh_bids_path.copy().update(extension=".json")
    except Exception as exc:
        log_warn(f"[behavior-auto] Failed to prepare behavior sidecar path: {exc}")
        return

    events_path = bids_path.copy().update(suffix="events", extension=".tsv")
    behavior_relative = str(source_path)
    dataset_root = Path(bids_path.root).resolve() if bids_path.root else None
    try:
        behavior_relative = (
            str(source_path.resolve().relative_to(dataset_root))
            if dataset_root is not None
            else str(source_path.resolve())
        )
    except Exception:
        behavior_relative = str(source_path)

    try:
        events_relative = (
            str(Path(events_path.fpath).resolve().relative_to(dataset_root))
            if dataset_root is not None
            else str(Path(events_path.fpath).resolve())
        )
    except Exception:
        events_relative = str(events_path.fpath)

    model = summary.get("model", {})
    model_payload = {
        "type": model.get("type", "offset"),
        "scale": float(model.get("scale", 1.0)),
        "offset": float(model.get("offset", 0.0)),
    }

    metadata = {
        "Description": "Behavioral log automatically synchronized to EEG trigger events.",
        "SourceFile": behavior_relative,
        "LabelColumn": summary.get("label_column"),
        "TimeColumnsCorrected": summary.get("time_columns", []),
        "SynchronizationModel": model_payload,
        "ResidualStatistics": summary.get("residuals", {}),
        "MatchesPerLabel": summary.get("matches_per_label", {}),
        "Anchors": summary.get("anchors", []),
        "EEGEventsFile": events_relative,
    }

    Path(sidecar_path.directory).mkdir(parents=True, exist_ok=True)
    Path(sidecar_path.fpath).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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


def _write_bids_dataset(
    raw: mne.io.BaseRaw,
    bids_path: "BIDSPath",
    write_kwargs: Dict[str, Any],
    events_df: Optional[pd.DataFrame] = None,
    events_desc: Optional[Dict[str, str]] = None,
) -> None:
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

    try:
        raw.set_annotations(None)
    except Exception:
        pass

    call_kwargs = {"raw": raw, "bids_path": bids_path, **write_kwargs}
    write_sig = inspect.signature(write_raw_bids)
    if "event_metadata" in write_sig.parameters:
        call_kwargs.setdefault("event_metadata", None)
    if "extra_columns_descriptions" in write_sig.parameters:
        call_kwargs.setdefault("extra_columns_descriptions", None)

    events_data = call_kwargs.get("events_data")
    if isinstance(events_data, pd.DataFrame) and events_data.empty:
        call_kwargs.pop("events_data", None)

    write_raw_bids(**call_kwargs)

    if events_df is not None and not events_df.empty:
        _write_events_tsv(bids_path, events_df)
        _write_events_metadata(bids_path, events_df, events_desc)

    if montage is not None:
        try:
            raw.set_montage(montage)
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

    trigger_events = extract_trigger_events(raw, spec)
    events_array, event_id_map, event_metadata_df, extra_desc = make_mne_events_and_metadata(trigger_events)

    bids_path = BIDSPath(
        subject=spec.subject,
        session=spec.session,
        task=spec.task,
        run=spec.run,
        datatype="eeg",
        root=str(cfg.bids_root),
    )

    write_kwargs: Dict[str, Any] = {
        "overwrite": spec.overwrite,
        "allow_preload": True,
        "format": "BrainVision",
        "verbose": False,
    }
    if events_array.size:
        write_kwargs["events"] = events_array
        if event_id_map:
            write_kwargs["event_id"] = event_id_map

    _write_bids_dataset(raw, bids_path, write_kwargs, trigger_events, extra_desc)
    maybe_update_eeg_sidecar(bids_path, cfg.eeg_reference, cfg.eeg_ground)
    process_behavioral_events(spec, trigger_events, bids_path)
    log_info(f"wrote {bids_path.fpath}")
    return bids_path



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI entry point definition for running the converter as a script."""
    parser = argparse.ArgumentParser(description="EEG BIDS converter")
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
